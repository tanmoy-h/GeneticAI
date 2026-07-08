"""
Stage 3 Week 9 — Entropy-Conditioned Dual-Mode Reasoning + HRPO Gate.

Architecture
------------
Three orthogonal mechanisms applied at each step:

  1. HRPO gate (always active, background DNA):
       input = (1-f)*embed + f*ThinkingResidualGate(embed, u_dna)
       Applied at input-embedding level before every transformer forward.

  2. LatentSp latent steps (low entropy → skip token, recycle h_{t-1}):
       When the model generates <start-latent>, the generation loop feeds
       h_{t-1} directly as the next input instead of a token embedding.
       Hidden-space-only computation — no distribution shift.

  3. DNA hidden injection (high entropy → inject u_dna into h_t before LM head):
       DNAHiddenInjector applies a gated residual on the final hidden state:
       h' = a_t * h + sqrt(1-a_t²) * proj(u_dna)
       Gate is LEARNED: activated at high-uncertainty positions.

Pipeline position
-----------------
  Stage 1    SFT  → train_dna_qwen.py  (week8tests/sh_stage1_sft_w8.sh)
  Stage 1.5  SFT  → train_latent_sft.py (week9tests/sh_stage1_5_latent_sft_w9.sh)
  Stage 2    HiRef → hiref_offline_multigpu.py (week8tests/sh_stage2_hiref_w8.sh)
  Stage 3    GRPO  → this file         (week9tests/sh_stage3_grpo_w9.sh)

Usage
-----
  accelerate launch adaptive_thinking_residual_w9.py \\
      --sft_checkpoint  <stage1.5_or_stage1.ckpt> \\
      --kegg_csv        genomorph/dataset/kegg_curriculum/global_stage1_anon_genes_mol_keep_chr.csv \\
      --output_dir      /scratch/tanmoyh_iitp/GenoMorph/checkpoints/week9tests/stage3_grpo_w9 \\
      --use_hrpo_gate   True

  # With Stage 2 manifold
  accelerate launch adaptive_thinking_residual_w9.py \\
      --stage2_dir      stage2_output_w8 ...

  # Resume
  RESUME=1 bash week9tests/sh_stage3_grpo_w9.sh
"""

import os
import pathlib
import types
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback
from trl import TrlParser

from genomorph.dna_modules import NucleotideDNAModule
from genomorph.models.dna_llm import DNALLMModel
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from genomorph.models.thinking_residual import (
    ThinkingResidualGate,
    ThinkingResidualLambda,
    make_thinking_residual_param_groups,
)
from genomorph.trainer import DNALLMGRPOConfig, DNALLMGRPOTrainer

from adaptive_latent_grpo import (
    GRPOModelConfig,
    GRPOScriptArguments,
    SaveWithPyTorchCallback,
    _load_sft_checkpoint,
    _prep_for_training,
    get_kegg_dataset,
    reward_funcs_registry,
)
from adaptive_latent_grpo import (
    ManifoldGateW6,
    _build_u_dna,
    make_ot_reward_func,
)
from adaptive_thinking_residual import (
    ThinkingBudgetProcessor,
    GateWarmupCallback,
    ThinkingResidualGRPOTrainer,
    SaveGateCallback,
    _make_tr_gate_loss,
)

register_evo2_tokenizer()

MAX_GATE_FACTOR = 0.5   # empirical safe cap (matches w8)


# ── Extended script args (LatentSp fields) ───────────────────────────────────────

@dataclass
class GRPOScriptArgumentsW9(GRPOScriptArguments):
    """GRPOScriptArguments + week9 LatentSp configuration."""

    latentSp_theta_low: float = field(
        default=1.0,
        metadata={"help": "Entropy threshold (nats) below which a step becomes latent "
                          "(h_{t-1} recycled as input). Typical confident prediction: 0–1 nat."},
    )
    latentSp_theta_high: float = field(
        default=2.5,
        metadata={"help": "Entropy threshold (nats) above which DNA injection is applied "
                          "to h_t before the LM head."},
    )
    latentSp_max_consec: int = field(
        default=3,
        metadata={"help": "Maximum consecutive latent steps before forcing a real token step."},
    )
    latent_lookahead_k: int = field(
        default=3,
        metadata={"help": "Number of tokens to generate tentatively at each Step N: boundary "
                          "before deciding latent vs normal. Mean entropy over these K tokens "
                          "is used as the step entropy estimate. K=1 → first-token only (cheapest). "
                          "K=3 → 2 extra forward passes per boundary, much better estimate."},
    )
    latentSp_warmup_steps: int = field(
        default=0,
        metadata={"help": "Steps to keep theta_low=0 before ramping (latent steps disabled). "
                          "0 = start ramping immediately."},
    )
    latentSp_ramp_steps: int = field(
        default=0,
        metadata={"help": "Steps over which theta_low ramps from 0 to latentSp_theta_low target. "
                          "0 = jump to target immediately after warmup."},
    )
    gate_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Path to thinking_gate.pt saved by Stage 1.5 with --use_gate. "
                          "When set together with gate_warmup_steps=0, the gate is active "
                          "from step 1 with pre-trained weights (no warmup discontinuity)."},
    )
    injector_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Path to dna_injector.pt saved by Stage 1.5 with --use_gate."},
    )

# Special token IDs (populated after tokenizer is available)
_LATENT_START_TOKEN = "<start-latent>"
_LATENT_END_TOKEN   = "<end-latent>"
_LATENT_PAD_TOKEN   = "<latent>"        # content placeholder, added by Stage 1.5


# ── DNA Hidden Injector ────────────────────────────────────────────────────────

class DNAHiddenInjector(nn.Module):
    """
    Learnable gated injection of u_dna into final transformer hidden states.

    Operates AFTER the last transformer layer, BEFORE the LM head.
    The gate learns to activate at high-uncertainty positions — functionally
    equivalent to entropy-conditioned injection but implemented as a learned
    residual so that GRPO gradients can train it end-to-end.

    Gate formula (mirrors ThinkingResidualGate):
        r_t = sigmoid(gate_h(h))                  [B, T, H]
        a_t = Lambda(r_t)  ∈ [r_min, r_max]       [B, T, H]
        h'  = a_t * h + sqrt(1 − a_t² + ε) * proj(u_dna_broadcast)

    Args:
        hidden_size: LLM hidden dimension.
        r_min: Minimum decay (default 0.7 → subtle injection even when open).
        r_max: Maximum decay (default 0.99 → near-identity at init).
    """

    def __init__(self, hidden_size: int, r_min: float = 0.7, r_max: float = 0.99):
        super().__init__()
        self.proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lambda_net = ThinkingResidualLambda(hidden_size, r_min, r_max)

        nn.init.normal_(self.proj.weight,   std=0.01)
        nn.init.normal_(self.gate_h.weight, std=0.01)

    def forward(
        self,
        h:     torch.Tensor,  # [B, T, H]
        u_dna: torch.Tensor,  # [B, H]
        eps:   float = 1e-8,
    ):
        """
        Returns:
            h_out: [B, T, H]  modified hidden states
            a_t:   [B, T, H]  decay coefficients (for logging / gate loss)
        """
        u_proj   = self.proj(u_dna).unsqueeze(1).expand_as(h)  # [B, T, H]
        r_t      = torch.sigmoid(self.gate_h(h))               # [B, T, H]
        a_t      = self.lambda_net(r_t)                        # [B, T, H]
        h_out    = (
            a_t * h
            + torch.sqrt(1.0 - a_t.pow(2) + eps) * u_proj.to(h)
        )
        return h_out, a_t


# ── LatentSp controller ──────────────────────────────────────────────────────────

class LatentSpController:
    """
    Manages latent-step decisions during autoregressive generation.

    Latent step = low-entropy position where h_{t-1} is recycled as input.
    DNA inject  = high-entropy position where u_dna is injected into h_t.

    Thresholds are in nats (natural-log entropy).  Typical Qwen3 token
    distributions:
      confident prediction  → entropy ≈ 0.0 – 1.0 nat
      uncertain prediction  → entropy ≈ 2.0 – 4.0 nats
    """

    def __init__(
        self,
        theta_low:          float = 1.0,   # below → latent step
        theta_high:         float = 2.5,   # above → DNA inject
        max_consecutive:    int   = 3,     # max consecutive latent steps
    ):
        self.theta_low       = theta_low
        self.theta_high      = theta_high
        self.max_consecutive = max_consecutive

    def is_latent_step(self, entropy: float, consecutive: int) -> bool:
        return entropy < self.theta_low and consecutive < self.max_consecutive

    def is_dna_inject(self, entropy: float) -> bool:
        return entropy > self.theta_high


class LatentSpWarmupCallback(TrainerCallback):
    """
    Gradually activates LatentSp latent steps over training by ramping theta_low:
      steps 0..warmup_steps          : theta_low = 0  (latent steps disabled)
      steps warmup_steps..+ramp_steps: theta_low 0 → target (linear)
      steps > warmup_steps+ramp_steps: theta_low = target

    Updates latentSp_ctrl.theta_low in-place each step.
    """

    def __init__(self, latentSp_ctrl: LatentSpController, target: float,
                 warmup_steps: int, ramp_steps: int):
        self._ctrl        = latentSp_ctrl
        self._target      = target
        self.warmup_steps = warmup_steps
        self.ramp_steps   = ramp_steps
        self._prev_phase  = None

    def _rank0_print(self, msg: str):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            print(msg, flush=True)

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        if step < self.warmup_steps:
            theta = 0.0
            phase = "warmup"
        elif step < self.warmup_steps + self.ramp_steps:
            frac  = (step - self.warmup_steps) / max(self.ramp_steps, 1)
            theta = frac * self._target
            phase = "ramp"
        else:
            theta = self._target
            phase = "active"

        self._ctrl.theta_low = theta

        if phase != self._prev_phase:
            if phase == "warmup":
                self._rank0_print(
                    f"\n[LatentSpWarmup] ── WARMUP  (step {step})\n"
                    f"[LatentSpWarmup]    theta_low=0 for {self.warmup_steps} steps "
                    f"— latent steps disabled.\n"
                )
            elif phase == "ramp":
                self._rank0_print(
                    f"\n[LatentSpWarmup] ── RAMP  (step {step})\n"
                    f"[LatentSpWarmup]    theta_low 0 → {self._target} over {self.ramp_steps} steps.\n"
                )
            elif phase == "active":
                self._rank0_print(
                    f"\n[LatentSpWarmup] ── ACTIVE  (step {step})\n"
                    f"[LatentSpWarmup]    theta_low={self._target} (target reached).\n"
                )
            self._prev_phase = phase

        if step % args.logging_steps == 0:
            self._rank0_print(
                f"[LatentSpWarmup] theta_low={theta:.4f}  target={self._target}  "
                f"phase={phase}  step={step}"
            )


# ── Custom autoregressive generation loop ────────────────────────────────────

def _make_dual_mode_generate_w9(
    thinking_gate:    ThinkingResidualGate,
    dna_injector:     DNAHiddenInjector,
    latentSp_ctrl:       LatentSpController,
    latent_start_id:  int,
    latent_end_id:    int,
    max_new_tokens:   int = 512,
    lookahead_k:      int = 3,
):
    """
    Replacement for model.generate_with_hrpo_gate.

    Custom token-by-token generation loop that implements:
      1. HRPO gate at input-embedding level (always active, capped at MAX_GATE_FACTOR)
      2. LatentSp latent steps: when low entropy, feed h_{t-1} instead of token embed
      3. DNA hidden injection: when high entropy, apply DNAHiddenInjector to h_t

    Stores per-step metadata in self._w9_gen_meta for the log-prob forward.
    """

    def _gen(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   Optional[torch.Tensor] = None,
        dna_tokenized=None,
        batch_idx_map=None,
        max_latent_steps: int   = 8,   # compat arg, unused
        min_latent_steps: int   = 0,   # compat arg, unused
        gate_threshold:   float = 0.5, # compat arg, unused
        **gen_kwargs,
    ) -> torch.Tensor:

        device = input_ids.device
        B      = input_ids.shape[0]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        u_dna = _build_u_dna(self, dna_tokenized, batch_idx_map, B, device)

        # ── Build prompt embeddings (DNA fusion at input level) ───────────────
        inputs_embeds, attn_mask = self.get_prompt_embeddings(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            dna_tokenized  = dna_tokenized,
            batch_idx_map  = batch_idx_map,
        )
        inputs_embeds = inputs_embeds.to(device)
        attn_mask     = attn_mask.to(device)

        # ── HRPO gate on prompt embeddings (1: input embed level) ────────────
        _min_f = getattr(self, "_gate_warmup_min_factor", 0.0)
        factor = max(min(getattr(self, "_gate_warmup_factor", 0.0), MAX_GATE_FACTOR), _min_f)
        if factor > 0 and u_dna is not None:
            B_e, T_e, H_e = inputs_embeds.shape
            residual = u_dna.detach().unsqueeze(1).expand(B_e, T_e, H_e).to(inputs_embeds)
            with torch.no_grad():
                gate_out, _ = thinking_gate(inputs_embeds, residual)
            modified_embeds = (1.0 - factor) * inputs_embeds + factor * gate_out
        else:
            modified_embeds = inputs_embeds

        # Cache for log-prob forward (mirrors w8)
        if u_dna is not None:
            self._v5_u_dna      = u_dna.detach()
            self._v5_ot_dist    = None
            self._tr_raw_embeds = inputs_embeds.detach()
            self._tr_u_dna      = u_dna.detach() * factor
            self._tr_ot_dist    = None
            self._gate_u_dna    = u_dna.detach()
        else:
            self._v5_u_dna = self._v5_ot_dist = None
            self._tr_raw_embeds = self._tr_u_dna = self._tr_ot_dist = None
            self._gate_u_dna    = None

        # ── Process prompt → KV-cache ─────────────────────────────────────────
        with torch.no_grad():
            prompt_out = self.text_model(
                inputs_embeds  = modified_embeds,
                attention_mask = attn_mask,
                use_cache      = True,
                output_hidden_states = True,
            )
        past_kv = prompt_out.past_key_values
        h_last  = prompt_out.hidden_states[-1][:, -1:, :]  # [B, 1, H]

        # ── Get logits from initial prompt hidden state ───────────────────────
        logits = self.text_model.lm_head(h_last)  # [B, 1, vocab]

        embed_layer   = self.text_model.get_input_embeddings()
        _eos_raw      = gen_kwargs.get(
            "eos_token_id",
            self.processor.tokenizer.eos_token_id
        )
        # Build a flat set of stop token IDs (handles scalar or list)
        _im_end_id    = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids: set  = set()
        if isinstance(_eos_raw, (list, tuple)):
            eos_ids.update(int(x) for x in _eos_raw)
        elif _eos_raw is not None:
            eos_ids.add(int(_eos_raw))
        if _im_end_id not in (None, self.processor.tokenizer.unk_token_id):
            eos_ids.add(int(_im_end_id))
        actual_max    = gen_kwargs.get("max_new_tokens", max_new_tokens)

        generated:   List[torch.Tensor] = []  # [B,1] token tensors
        gen_meta:    List[dict]         = []  # per-step metadata
        curr_mask    = attn_mask
        consecutive_latent = 0

        _tok = self.processor.tokenizer
        # Token IDs LatentSp must never intercept (always emit these normally)
        _structural_ids: set = set(eos_ids)
        for _s in ("</think>", "<|im_end|>"):
            _ids = _tok.encode(_s, add_special_tokens=False)
            _structural_ids.update(_ids)

        # Step-level LatentSp: fire at step-content boundaries only, matching the
        # Stage 1.5 training which replaced whole step contents not individual tokens.
        # Lookahead: at each boundary, tentatively generate _LOOKAHEAD_K tokens,
        # compute mean entropy over them, then decide latent or normal.
        # Latent  → restore saved state, discard lookahead tokens, insert latent block.
        # Normal  → commit lookahead tokens as-is, continue generating from there.
        import re as _re
        _step_re_c          = _re.compile(r'Step\s+(\d+)\s*:\Z')
        _step_state         = "normal"          # "normal" | "content_latent"
        _step_latent_count  = 0
        _current_step_num   = 0                 # step number of the active/last boundary
        _MAX_LATENT_PER_STEP = latentSp_ctrl.max_consecutive
        _at_step_boundary   = False             # True after emitting "Step N:" colon
        _LOOKAHEAD_K        = max(1, lookahead_k)

        # Stop on second </think>: first is legitimate end-of-reasoning,
        # second means a repetition loop — treat it as EOS.
        _think_close_ids = _tok.encode("</think>", add_special_tokens=False)
        _think_close_count = 0
        # After </think>, stop as soon as "Answer: [text]\n" is complete.
        # Without this, the model loops "Answer: X\nAnswer: X\n..." to 800 tokens
        # because </think> is masked and the model keeps emitting the answer line.
        _answer_ids     = _tok.encode("Answer:", add_special_tokens=False)
        _answer_started = False   # True once "Answer:" appears in tokens after </think>

        # Latent pad token (defined once, used in masking + emission + lookahead check)
        _latent_pad_id = _tok.convert_tokens_to_ids("<latent>")
        _latent_all_ids = {latent_start_id, latent_end_id, _latent_pad_id}

        for step in range(actual_max):
            # ── Entropy of current logit distribution ─────────────────────────
            probs = torch.softmax(logits[:, -1, :].float(), dim=-1)  # [B, V]
            token_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [B]
            mean_entropy  = token_entropy.mean().item()

            # ── Step-level LatentSp decision with K-token lookahead ──────────────
            argmax_tok    = int(logits[:, -1, :].argmax(dim=-1)[0].item())
            is_structural = argmax_tok in _structural_ids

            if _at_step_boundary and not is_structural:
                _at_step_boundary = False

                # Save state before lookahead so we can restore if going latent.
                # DynamicCache (transformers>=4.38) is mutated in-place by the
                # model forward pass, so a bare reference _saved_past_kv=past_kv
                # would be corrupted by the lookahead.  We deep-copy to snapshot
                # the exact key/value tensors at this boundary position.
                import copy as _kv_copy
                _saved_past_kv  = _kv_copy.deepcopy(past_kv)
                _saved_h_last   = h_last
                _saved_mask     = curr_mask
                _saved_logits   = logits

                # Separate KV / mask / logits for lookahead so past_kv is never
                # touched during the tentative forward passes.
                _la_kv       = _kv_copy.deepcopy(past_kv)
                _la_mask_cur = curr_mask
                _la_log_cur  = logits

                _la_entropies: List[float] = [mean_entropy]   # includes current position
                _la_tokens:    List[torch.Tensor] = []
                _la_masks:     List[torch.Tensor] = []
                _la_abort      = False                         # hit structural token early

                for _k in range(_LOOKAHEAD_K - 1):
                    # Sample (greedy) — entropy is distribution property, not sample
                    _la_tok = _la_log_cur[:, -1, :].argmax(dim=-1, keepdim=True)   # [B, 1]
                    _la_tid = _la_tok[0, 0].item()
                    _la_str = _tok.decode([_la_tid], skip_special_tokens=False)
                    if _la_tid in _structural_ids:
                        _la_abort = True
                        break
                    # If model greedily predicts a latent token, it is signalling
                    # strong confidence that this step should be compressed.
                    # Force entropy=0 → latent mode wins without running more lookahead.
                    if _la_tid in _latent_all_ids:
                        _la_entropies.append(0.0)
                        break
                    _la_tokens.append(_la_tok)

                    _la_inp = embed_layer(_la_tok).to(inputs_embeds)
                    if factor > 0 and u_dna is not None:
                        _r = u_dna.detach().unsqueeze(1).to(_la_inp)
                        with torch.no_grad():
                            _g, _ = thinking_gate(_la_inp, _r)
                        _la_inp = (1.0 - factor) * _la_inp + factor * _g

                    _ext = torch.ones(B, 1, dtype=_la_mask_cur.dtype, device=device)
                    _la_mask_cur = torch.cat([_la_mask_cur, _ext], dim=1)
                    _la_masks.append(_la_mask_cur)

                    with torch.no_grad():
                        _la_out = self.text_model(
                            inputs_embeds   = _la_inp,
                            attention_mask  = _la_mask_cur,
                            past_key_values = _la_kv,
                            use_cache       = True,
                            output_hidden_states = True,
                        )
                    _la_kv      = _la_out.past_key_values
                    _la_h_last  = _la_out.hidden_states[-1]
                    _la_log_cur = self.text_model.lm_head(_la_h_last)

                    _p  = torch.softmax(_la_log_cur[:, -1, :].float(), dim=-1)
                    _la_entropies.append(
                        -(_p * (_p + 1e-10).log()).sum(dim=-1).mean().item()
                    )

                _la_mean = sum(_la_entropies) / len(_la_entropies)

                if not _la_abort and latentSp_ctrl.is_latent_step(_la_mean, 0):
                    # Latent: restore h_last/mask/logits; past_kv already clean
                    # (LA used _la_kv so past_kv was never mutated).
                    h_last    = _saved_h_last
                    curr_mask = _saved_mask
                    logits    = _saved_logits
                    _step_state        = "content_latent"
                    _step_latent_count = 0
                else:
                    # Normal: restore h_last/mask/logits; past_kv already clean.
                    h_last    = _saved_h_last
                    curr_mask = _saved_mask
                    logits    = _saved_logits
                    _step_state = "normal"
                    # Fall through — sample first step-content token from boundary logits.

            elif _at_step_boundary:
                _at_step_boundary = False
                _step_state = "normal"

            # LatentSp fires only when entire step content is being replaced, not per-token
            if _step_state == "content_latent":
                if is_structural or _step_latent_count >= _MAX_LATENT_PER_STEP:
                    # Close any OPEN latent block before leaving latent mode.  If
                    # <start-latent> was emitted but the block is exiting early
                    # (is_structural fired before reaching _MAX_LATENT_PER_STEP —
                    # typically because the untrained post-<start-latent> argmax is
                    # <|im_end|>), <end-latent> was never emitted.  Force-emit it so
                    # every block is matched, mirroring SFT's [start][latent]*k[end].
                    # Without this the model produces an unclosed "<start-latent>
                    # <|im_end|>" and truncates before </think>/Answer.
                    if 1 <= _step_latent_count < _MAX_LATENT_PER_STEP:
                        _end_tok  = torch.full((B, 1), latent_end_id,
                                               dtype=torch.long, device=device)
                        _e_emb    = embed_layer(_end_tok).to(inputs_embeds)
                        if factor > 0 and u_dna is not None:
                            _er = u_dna.detach().unsqueeze(1).to(_e_emb)
                            with torch.no_grad():
                                _eg, _ = thinking_gate(_e_emb, _er)
                            _e_emb = (1.0 - factor) * _e_emb + factor * _eg
                        _e_ext    = torch.ones(B, 1, dtype=curr_mask.dtype, device=device)
                        curr_mask = torch.cat([curr_mask, _e_ext], dim=1)
                        with torch.no_grad():
                            _e_out = self.text_model(
                                inputs_embeds        = _e_emb,
                                attention_mask       = curr_mask,
                                past_key_values      = past_kv,
                                use_cache            = True,
                                output_hidden_states = True,
                            )
                        past_kv = _e_out.past_key_values
                        h_last  = _e_out.hidden_states[-1]
                        logits  = self.text_model.lm_head(h_last)
                        generated.append(_end_tok)
                        gen_meta.append({"step": step, "is_latent": True,
                                         "entropy": 0.0, "dna_injected": False})
                    _step_state        = "normal"
                    _step_latent_count = 0
                    is_latent          = False
                    consecutive_latent = 0
                    # Force "\nStep N+1:" through model forward passes on normal
                    # latent-block completion.  During Stage 1.5 SFT the step
                    # transition was always teacher-forced — the model was never
                    # trained to autoregressively predict it from the latent KV
                    # state alone.  We preserve the latent KV (no restore) and
                    # inject the header tokens explicitly so the model has the
                    # correct position context before the next boundary decision.
                    if not is_structural and 0 < _current_step_num < 15:
                        _next_step_num = _current_step_num + 1
                        _next_hdr      = f"\nStep {_next_step_num}:"
                        _force_ids     = _tok.encode(_next_hdr, add_special_tokens=False)
                        for _fid in _force_ids:
                            _f_tok = torch.full((B, 1), _fid, dtype=torch.long, device=device)
                            _f_emb = embed_layer(_f_tok).to(inputs_embeds)
                            if factor > 0 and u_dna is not None:
                                _fr = u_dna.detach().unsqueeze(1).to(_f_emb)
                                with torch.no_grad():
                                    _fg, _ = thinking_gate(_f_emb, _fr)
                                _f_emb = (1.0 - factor) * _f_emb + factor * _fg
                            _f_ext = torch.ones(B, 1, dtype=curr_mask.dtype, device=device)
                            curr_mask = torch.cat([curr_mask, _f_ext], dim=1)
                            with torch.no_grad():
                                _f_out = self.text_model(
                                    inputs_embeds    = _f_emb,
                                    attention_mask   = curr_mask,
                                    past_key_values  = past_kv,
                                    use_cache        = True,
                                    output_hidden_states = True,
                                )
                            past_kv = _f_out.past_key_values
                            h_last  = _f_out.hidden_states[-1]
                            generated.append(_f_tok)
                            gen_meta.append({"step": step, "is_latent": False,
                                             "entropy": 0.0, "dna_injected": False})
                        logits = self.text_model.lm_head(h_last)
                        _current_step_num = _next_step_num
                        _at_step_boundary = True
                        continue
                else:
                    is_latent          = True
                    _step_latent_count += 1
                    consecutive_latent += 1
            else:
                is_latent          = False
                consecutive_latent = 0

            if is_latent:
                # Emit the correct token for this position in the latent block,
                # matching Stage 1.5 inject_latent_markers pattern:
                #   [<start-latent>] [<latent>]*(N-2) [<end-latent>]
                # _step_latent_count was just incremented: 1=first, N=last.
                if _step_latent_count == 1:
                    _emit_id = latent_start_id
                elif _step_latent_count == _MAX_LATENT_PER_STEP:
                    _emit_id = latent_end_id
                else:
                    _emit_id = _latent_pad_id
                placeholder = torch.full(
                    (B, 1), _emit_id,
                    dtype=torch.long, device=device
                )
                next_input    = embed_layer(placeholder).to(inputs_embeds)  # [B, 1, H]
                generated.append(placeholder)
                meta_token_id = _emit_id
            else:
                # Normal: sample token from logits
                # Sampling params come via generation_config, not as bare kwargs
                _gc         = gen_kwargs.get("generation_config", None)
                def _gp(key, default):
                    v = gen_kwargs.get(key)
                    if v is not None:
                        return v
                    if _gc is not None:
                        v = getattr(_gc, key, None)
                        if v is not None:
                            return v
                    return default
                temperature = float(_gp("temperature", 1.0))
                top_k_v     = int(_gp("top_k", 0))
                top_p_v     = float(_gp("top_p", 1.0))
                rep_pen     = float(_gp("repetition_penalty", 1.0))

                if temperature > 0:
                    scaled = logits[:, -1, :].float() / temperature

                    # Mask latent special tokens — model must never self-generate
                    # these; latent steps are inserted only by the LatentSp criterion
                    for _lid in (latent_start_id, latent_end_id, _latent_pad_id):
                        if 0 <= _lid < scaled.size(-1):
                            scaled[:, _lid] = float("-inf")

                    # Suppress EOS/<|im_end|> until </think> is emitted — the model
                    # must produce reasoning + Answer before it can terminate.  The
                    # untrained post-<start-latent> distribution otherwise favours
                    # <|im_end|> mid-think and truncates the completion.  Masked
                    # BEFORE top-k/top-p so at least one non-EOS token always survives.
                    if _think_close_count == 0:
                        for _eid in eos_ids:
                            if 0 <= _eid < scaled.size(-1):
                                scaled[:, _eid] = float("-inf")

                    # Once the Answer line is started, block further "Answer:" tokens
                    # so the model is forced to end with EOS / <|im_end|> instead of looping
                    if _answer_started:
                        for _tid in _answer_ids:
                            if 0 <= _tid < scaled.size(-1):
                                scaled[:, _tid] = float("-inf")

                    # Repetition penalty
                    if rep_pen != 1.0 and generated:
                        prev = torch.cat(generated, dim=1)  # [B, T_so_far]
                        for b in range(B):
                            u = prev[b].unique()
                            scaled[b, u] = torch.where(
                                scaled[b, u] < 0,
                                scaled[b, u] * rep_pen,
                                scaled[b, u] / rep_pen,
                            )

                    # No-repeat-ngram blocking
                    no_rep_ng = int(_gp("no_repeat_ngram_size", 0))
                    if no_rep_ng > 0 and len(generated) >= no_rep_ng - 1:
                        prev_ng = torch.cat(generated, dim=1)  # [B, T_so_far]
                        ng_len = no_rep_ng - 1
                        for b in range(B):
                            T = prev_ng.shape[1]
                            if T >= ng_len:
                                prefix = prev_ng[b, -ng_len:].tolist()
                                for i in range(T - ng_len):
                                    if prev_ng[b, i:i + ng_len].tolist() == prefix:
                                        banned_tok = prev_ng[b, i + ng_len].item()
                                        scaled[b, banned_tok] = float("-inf")

                    # Top-k filtering (preserve EOS so generation can terminate —
                    # but only AFTER </think>; before it, EOS stays suppressed above)
                    if top_k_v > 0:
                        k = min(top_k_v, scaled.size(-1))
                        kth = torch.topk(scaled, k, dim=-1).values[:, -1, None]
                        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                        if _think_close_count > 0:
                            for _eid in eos_ids:
                                if _eid < scaled.size(-1):
                                    eos_masked = scaled[:, _eid].isinf()
                                    scaled[:, _eid] = torch.where(
                                        eos_masked, kth.squeeze(-1), scaled[:, _eid]
                                    )

                    # Top-p (nucleus) filtering
                    if top_p_v < 1.0:
                        s_logits, s_idx = torch.sort(scaled, descending=True, dim=-1)
                        cum_p = torch.cumsum(torch.softmax(s_logits, dim=-1), dim=-1)
                        remove = (cum_p - torch.softmax(s_logits, dim=-1)) > top_p_v
                        s_logits[remove] = float("-inf")
                        scaled.scatter_(-1, s_idx, s_logits)

                    next_token = torch.multinomial(
                        torch.softmax(scaled, dim=-1), num_samples=1
                    )  # [B, 1]
                else:
                    # Greedy: mask latent tokens and Answer: loop before argmax
                    _greedy_logits = logits[:, -1, :].clone()
                    for _lid in (latent_start_id, latent_end_id, _latent_pad_id):
                        if 0 <= _lid < _greedy_logits.size(-1):
                            _greedy_logits[:, _lid] = float("-inf")
                    if _answer_started:
                        for _tid in _answer_ids:
                            if 0 <= _tid < _greedy_logits.size(-1):
                                _greedy_logits[:, _tid] = float("-inf")
                    # Suppress EOS until </think> emitted (see sampling branch).
                    if _think_close_count == 0:
                        for _eid in eos_ids:
                            if 0 <= _eid < _greedy_logits.size(-1):
                                _greedy_logits[:, _eid] = float("-inf")
                    next_token = torch.argmax(_greedy_logits, dim=-1, keepdim=True)

                next_input        = embed_layer(next_token).to(inputs_embeds)  # [B, 1, H]
                generated.append(next_token)
                meta_token_id = next_token[0, 0].item()

                # Detect "Step N:" boundary: next token is the first content token
                if len(generated) >= 2:
                    _recent_ids  = [g[0, 0].item() for g in generated[-5:]]
                    _recent_text = _tok.decode(_recent_ids, skip_special_tokens=True)
                    _m = _step_re_c.search(_recent_text)
                    if _m:
                        _at_step_boundary = True
                        _step_state       = "normal"  # mode decided next iteration
                        _current_step_num = int(_m.group(1))

                # Check for EOS / <|im_end|>
                if next_token[0, 0].item() in eos_ids:
                    if next_token[0, 0].item() in set(_think_close_ids):
                        print("[w9] second </think> stopped generation", flush=True)
                    gen_meta.append({
                        "step": step, "is_latent": False,
                        "entropy": mean_entropy, "dna_injected": False,
                    })
                    break

                # Detect </think>: on first occurrence, promote it to EOS so the
                # second occurrence stops generation via the normal EOS path.
                _recent_gen = [g[0, 0].item() for g in generated[-len(_think_close_ids):]]
                if _recent_gen == list(_think_close_ids):
                    _think_close_count += 1
                    if _think_close_count == 1:
                        eos_ids.update(_think_close_ids)

                # After </think>, detect "Answer:" to set flag, then stop on newline.
                # Prevents the model looping "Answer: X\nAnswer: X\n..." to 800 tokens.
                if _think_close_count >= 1:
                    if not _answer_started:
                        _rec8 = _tok.decode(
                            [g[0, 0].item() for g in generated[-8:]], skip_special_tokens=False
                        )
                        if "Answer:" in _rec8:
                            _answer_started = True
                    else:
                        _cur_tok_str = _tok.decode([meta_token_id], skip_special_tokens=False)
                        if "\n" in _cur_tok_str:
                            break

            # ── HRPO gate on next input (2: always active) ────────────────────
            if factor > 0 and u_dna is not None:
                residual = u_dna.detach().unsqueeze(1).to(next_input)
                with torch.no_grad():
                    gate_out, _ = thinking_gate(next_input, residual)
                next_input = (1.0 - factor) * next_input + factor * gate_out

            # ── Extend attention mask ─────────────────────────────────────────
            extra     = torch.ones(B, 1, dtype=curr_mask.dtype, device=device)
            curr_mask = torch.cat([curr_mask, extra], dim=1)

            # ── One-step LM forward ───────────────────────────────────────────
            with torch.no_grad():
                step_out = self.text_model(
                    inputs_embeds  = next_input,
                    attention_mask = curr_mask,
                    past_key_values = past_kv,
                    use_cache      = True,
                    output_hidden_states = True,
                )
            past_kv = step_out.past_key_values
            h_last  = step_out.hidden_states[-1]  # [B, 1, H]

            # ── DNA hidden injection (3: high-entropy steps) ──────────────────
            dna_injected = False
            if (not is_latent
                    and u_dna is not None
                    and latentSp_ctrl.is_dna_inject(mean_entropy)):
                with torch.no_grad():
                    h_last, _ = dna_injector(h_last, u_dna.detach())
                dna_injected = True

            gen_meta.append({
                "step":         step,
                "is_latent":    is_latent,
                "entropy":      mean_entropy,
                "dna_injected": dna_injected,
            })

            logits = self.text_model.lm_head(h_last)  # logits for NEXT token

        # Cache metadata for log-prob forward and trainer metrics
        self._w9_gen_meta       = gen_meta
        self._gate_steps_taken  = len(gen_meta)
        self._gate_latent_steps = sum(1 for m in gen_meta if m.get("is_latent", False))

        if not generated:
            return torch.zeros(B, 0, dtype=torch.long, device=device)
        return torch.cat(generated, dim=1)

    return _gen


# ── Log-prob forward (w9): gate + DNA injection WITH gradients ────────────────

def _make_dual_mode_forward_w9(
    thinking_gate: ThinkingResidualGate,
    dna_injector:  DNAHiddenInjector,
):
    """
    Patched DNALLMModel.forward for GRPO log-prob computation.

    Applies:
      1. ThinkingResidualGate at embedding level (WITH grad, capped at MAX_GATE_FACTOR)
      2. DNAHiddenInjector hook on final hidden states (WITH grad)

    The injector is always applied (learned gate controls the effect strength).
    This is consistent with the generation path where the injector activates
    for high-entropy positions — the learned gate approximates this.
    """

    def _forward(
        self,
        input_ids      = None,
        attention_mask = None,
        dna_tokenized  = None,
        batch_idx_map  = None,
        labels         = None,
        **kwargs,
    ):
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required")

        batch_size = input_ids.shape[0]
        device     = input_ids.device

        # 1. Token embeddings
        embed_layer        = self.text_model.get_input_embeddings()
        text_inputs_embeds = embed_layer(input_ids)

        # 2. DNA cross-attention fusion (unchanged from DNALLMModel.forward)
        batch_dna_embeds = None
        if dna_tokenized is not None and batch_idx_map:
            batch_dna_embeds = self.process_dna_embeddings(
                dna_tokenized, batch_idx_map, batch_size,
                text_context = text_inputs_embeds,
                text_mask    = attention_mask,
            )
            mask            = input_ids == self.dna_token_id
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0).to(
                dtype=text_inputs_embeds.dtype, device=device
            )
            text_inputs_embeds[mask] = dna_embeds_flat

        # 3. HRPO gate WITH gradients — capped at MAX_GATE_FACTOR
        # u_dna: prefer cached value (set by generation loop); fall back to
        # mean-pooling batch_dna_embeds so the forward works standalone (Stage 1.5 SFT).
        _min_f       = getattr(self, "_gate_warmup_min_factor", 0.0)
        factor       = max(min(getattr(self, "_gate_warmup_factor", 0.0), MAX_GATE_FACTOR), _min_f)
        cached_u_dna = getattr(self, "_v5_u_dna", None)
        if cached_u_dna is None and batch_dna_embeds is not None:
            u_list = []
            for b in range(batch_size):
                parts = [batch_dna_embeds[s]
                         for s in (b, b + batch_size)
                         if s < len(batch_dna_embeds) and batch_dna_embeds[s].shape[0] > 0]
                u_list.append(
                    torch.cat(parts, 0).mean(0) if parts
                    else torch.zeros(self.text_hidden_size, dtype=text_inputs_embeds.dtype, device=device)
                )
            cached_u_dna = F.normalize(
                torch.stack(u_list).float(), dim=-1
            ).to(text_inputs_embeds.dtype)

        if factor > 0 and cached_u_dna is not None:
            B_gen = cached_u_dna.shape[0]
            if B_gen == 1:
                u_dna_fwd = cached_u_dna.expand(batch_size, -1).to(device)
            elif batch_size % B_gen == 0:
                G         = batch_size // B_gen
                u_dna_fwd = cached_u_dna.repeat_interleave(G, dim=0).to(device)
            else:
                u_dna_fwd = None

            if u_dna_fwd is not None:
                B, T, H  = text_inputs_embeds.shape
                residual = u_dna_fwd.unsqueeze(1).expand(B, T, H).to(text_inputs_embeds)
                gate_out, _ = thinking_gate(text_inputs_embeds, residual, None)
                text_inputs_embeds = (
                    (1.0 - factor) * text_inputs_embeds + factor * gate_out
                )
        else:
            u_dna_fwd = None

        # 4. Main LM forward — output_hidden_states to get final layer output
        lm_out = self.text_model(
            inputs_embeds  = text_inputs_embeds,
            attention_mask = attention_mask,
            labels         = None,       # compute loss manually after injection
            output_hidden_states = True,
            **kwargs,
        )

        # 5. DNAHiddenInjector on final hidden states WITH gradients
        if u_dna_fwd is not None:
            final_h = lm_out.hidden_states[-1]  # [B, T, H]
            final_h, _ = dna_injector(final_h, u_dna_fwd)
            # Re-apply LM head with injected hidden states
            logits = self.text_model.lm_head(final_h)
        else:
            logits = lm_out.logits

        # 6. Compute cross-entropy loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            # Return as a ModelOutput-compatible object
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(
                loss   = loss,
                logits = logits,
                past_key_values    = lm_out.past_key_values,
                hidden_states      = lm_out.hidden_states,
            )

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss   = lm_out.loss,
            logits = logits,
            past_key_values = lm_out.past_key_values,
            hidden_states   = lm_out.hidden_states,
        )

    return _forward


# ── Gate loss (w9): adds DNA injector entropy term ────────────────────────────

def _make_tr_gate_loss_w9(
    thinking_gate: ThinkingResidualGate,
    dna_injector:  DNAHiddenInjector,
):
    """
    Auxiliary gate loss = HRPO gate loss (entropy + OT) + injector collapse penalty.

    The injector penalty keeps a_t away from the extremes so it stays responsive.
    """
    base_gate_loss_fn = _make_tr_gate_loss(thinking_gate)

    def _gate_loss(self, gate_reg_weight: float = 0.01) -> torch.Tensor:
        base_loss = base_gate_loss_fn(self, gate_reg_weight)

        # Injector collapse penalty: keep gate open (a_t not stuck at r_max)
        raw = getattr(self, "_tr_raw_embeds", None)
        u   = getattr(self, "_tr_u_dna",     None)
        if raw is not None and u is not None:
            try:
                B_gen = u.shape[0]
                B_raw = raw.shape[0]
                if B_gen == 1:
                    u_exp = u.expand(B_raw, -1).to(raw)
                elif B_raw % B_gen == 0:
                    u_exp = u.repeat_interleave(B_raw // B_gen, dim=0).to(raw)
                else:
                    u_exp = None
                if u_exp is not None:
                    # Run injector on detached hidden states
                    h_sample = raw[:, -1:, :].detach()  # [B, 1, H] last prompt token
                    _, a_t   = dna_injector(h_sample, u_exp.detach())
                    # Entropy of a_t: encourages a_t ≈ 0.85 (mid-range, not collapsed)
                    inj_penalty = -gate_reg_weight * 0.5 * (
                        a_t * (1 - a_t) + 1e-8
                    ).log().mean()
                    return base_loss + inj_penalty
            except Exception:
                pass
        return base_loss

    return _gate_loss


# ── Save callback (gate + injector) ──────────────────────────────────────────

class SaveGateAndInjectorCallback(TrainerCallback):
    """Save thinking_gate.pt and dna_injector.pt at every checkpoint."""

    def __init__(
        self,
        thinking_gate: ThinkingResidualGate,
        dna_injector:  DNAHiddenInjector,
    ):
        self.thinking_gate = thinking_gate
        self.dna_injector  = dna_injector

    def on_save(self, args, state, control, **kwargs):
        folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(folder, exist_ok=True)
        torch.save(self.thinking_gate.state_dict(),
                   os.path.join(folder, "thinking_gate.pt"))
        torch.save(self.dna_injector.state_dict(),
                   os.path.join(folder, "dna_injector.pt"))
        print(f"[w9] thinking_gate + dna_injector → {folder}/")


# ── Tie-breaking: prefer latest checkpoint when metric is equal ───────────────

class PreferLatestOnTieCallback(TrainerCallback):
    """When eval metric equals the current best, update best_model_checkpoint to
    point at the latest checkpoint *before* _save_checkpoint runs rotation, so
    save_total_limit protects the most recent tied checkpoint instead of the
    older one. Fires inside on_evaluate, ahead of _determine_best_metric and
    _save_checkpoint — the folder need not exist yet (it's just a path string)."""

    def __init__(self, metric_name: str, greater_is_better: bool = True):
        self.metric_name       = metric_name
        self.greater_is_better = greater_is_better

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or state.best_metric is None:
            return
        key = f"eval_{self.metric_name}"
        current = metrics.get(key)
        if current is None:
            return
        if current == state.best_metric:
            latest = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            state.best_model_checkpoint = latest
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            except Exception:
                rank = 0
            if rank == 0:
                print(
                    f"[PreferLatest] Tie at {self.metric_name}={current:.4f} "
                    f"→ best_model_checkpoint updated to checkpoint-{state.global_step}"
                )


# ── Latent usage reward ───────────────────────────────────────────────────────

def make_latent_usage_reward_func(
    latentSp_ctrl:      LatentSpController,
    latent_start_id: int,
    latent_end_id:   int,
    target_ratio:    float = 0.02,   # ~4 latent starts / ~160 words (s=4 baseline)
    bandwidth:       float = 0.04,   # fixed ±bandwidth window; ratio in [0, 0.06] → positive
):
    """
    Reward function that encourages appropriate latent step usage.

    target_ratio=0.02: calibrated for s=4 Stage 1.5 checkpoint.
      Typical KEGG completion: ~150 reasoning words + 4 latent spans (each
      decoded as ~1 word) + ~5 answer words ≈ 160 total → 4/160 ≈ 0.025.

    bandwidth=0.04: fixed window (not scaled by target) so the reward shape
      is stable regardless of target value.
      ratio = target         → +0.5  (peak)
      ratio = target ± bw    →  0.0
      ratio = target ± 2*bw  → -0.5  (floor)

    Penalizes:
      - Too few latent steps (model not using LatentSp)
      - Too many latent steps (model hiding uncertainty behind latents)

    Returns per-sample reward in [-0.5, +0.5].
    """
    def _latent_reward(completions, **kwargs):
        # theta_low == 0 during LatentSp warmup — latent steps are disabled.
        # Scoring ratio=0 against target_ratio > 0 would give a spurious +0.25
        # reward and reinforce the no-latent behaviour before training even starts.
        if latentSp_ctrl.theta_low <= 0.0:
            return [0.0] * len(completions)

        import re as _re_lat

        def _is_valid(t: str) -> bool:
            # Only reward latent usage on a well-formed completion: exactly one
            # matched <think>…</think> and a non-empty Answer:.  A truncated block
            # ("<start-latent><|im_end|>") would otherwise earn latent reward and
            # reinforce the degenerate mode, driving a death spiral once correctness
            # collapses.  Return neutral (0.0) for malformed completions.
            if t.count("<think>") != 1 or t.count("</think>") != 1:
                return False
            _after = t.split("</think>", 1)[1]
            return _re_lat.search(r'[Aa]nswer:\s*\S', _after) is not None

        rewards = []
        for comp in completions:
            # comp arrives as [{"role": "assistant", "content": "..."}]
            text = None
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                text = comp[0].get("content", "")
                n_latent = text.count(_LATENT_START_TOKEN)
                n_total  = max(len(text.split()), 1)
            elif isinstance(comp, str):
                text = comp
                n_latent = comp.count(_LATENT_START_TOKEN)
                n_total  = max(len(comp.split()), 1)
            elif isinstance(comp, list):
                # list of token IDs — no text form available for the validity gate
                n_latent = sum(1 for t in comp if t == latent_start_id)
                n_total  = max(len(comp), 1)
            else:
                n_latent = 0
                n_total  = 1

            # Reward-hacking guard: latent density only counts on a valid completion.
            if text is not None and not _is_valid(text):
                rewards.append(0.0)
                continue

            ratio  = n_latent / n_total
            reward = 0.5 * (1.0 - abs(ratio - target_ratio) / bandwidth)
            reward = max(-0.5, min(0.5, reward))   # hard clamp
            rewards.append(float(reward))
        return rewards

    return _latent_reward


# ── Patch ─────────────────────────────────────────────────────────────────────

def patch_model_for_dual_mode_w9(
    model:           DNALLMModel,
    thinking_gate:   ThinkingResidualGate,
    dna_injector:    DNAHiddenInjector,
    latentSp_ctrl:      LatentSpController,
    latent_start_id: int,
    latent_end_id:   int,
    max_new_tokens:  int = 512,
    lookahead_k:     int = 3,
) -> None:
    """
    Patch three model methods with w9 dual-mode versions:
      generate_with_hrpo_gate — LatentSp + HRPO gate + DNA injection (no grad)
      forward                 — HRPO gate + DNA injection (WITH grad)
      compute_gate_loss       — entropy + OT + injector collapse penalty
    """
    model.generate_with_hrpo_gate = types.MethodType(
        _make_dual_mode_generate_w9(
            thinking_gate, dna_injector, latentSp_ctrl,
            latent_start_id, latent_end_id, max_new_tokens,
            lookahead_k = lookahead_k,
        ),
        model,
    )
    model.forward = types.MethodType(
        _make_dual_mode_forward_w9(thinking_gate, dna_injector),
        model,
    )
    model.compute_gate_loss = types.MethodType(
        _make_tr_gate_loss_w9(thinking_gate, dna_injector),
        model,
    )
    ng  = sum(p.numel() for p in thinking_gate.parameters())
    ni  = sum(p.numel() for p in dna_injector.parameters())
    print(
        f"[w9] Patched: generate_with_hrpo_gate | forward | compute_gate_loss\n"
        f"[w9] ThinkingResidualGate params = {ng:,}  MAX_GATE_FACTOR = {MAX_GATE_FACTOR}\n"
        f"[w9] DNAHiddenInjector params    = {ni:,}\n"
        f"[w9] LatentSp θ_low={latentSp_ctrl.theta_low}  θ_high={latentSp_ctrl.theta_high}  "
        f"max_consecutive={latentSp_ctrl.max_consecutive}\n"
        f"[w9] Mechanisms: HRPO gate (always) | LatentSp latent (low-H) | DNA inject (high-H)"
    )


# ── DNA embedding cache helper ────────────────────────────────────────────────

def _apply_dna_cache(model, cache_path: str) -> None:
    """Patch _evo2_embed with a cache lookup and offload Evo2 to CPU (~14 GB freed)."""
    import types as _types
    print(f"[w9] Loading DNA embedding cache: {cache_path}")
    _cache = torch.load(cache_path, map_location="cpu")
    print(f"[w9] DNA cache: {len(_cache)} sequences")

    # precompute_dna_embeddings.py keys sequences by their UNPADDED token IDs
    # (it uses attention_mask to trim trailing padding before calling .tobytes()).
    # At inference the DNA tokenizer pads all sequences in a batch item to the
    # same length, so the shorter sequence(s) carry trailing pad tokens that were
    # not present when the cache was built.  Strip them here before lookup.
    _pad_id = None
    if getattr(model, "dna_tokenizer", None) is not None:
        _pad_id = getattr(model.dna_tokenizer, "pad_token_id", None)

    _c = _cache
    _pid = _pad_id

    def _cached_embed(self, input_ids: torch.Tensor, layer_name: str) -> torch.Tensor:
        # Strip trailing pad tokens so the key matches precompute's unpadded keys
        ids = input_ids
        if _pid is not None and ids.shape[1] > 1:
            flat = ids[0]
            non_pad = (flat != _pid).nonzero(as_tuple=True)[0]
            if non_pad.numel() > 0:
                actual_len = int(non_pad[-1].item()) + 1
                if actual_len < ids.shape[1]:
                    ids = ids[:, :actual_len]
        key = ids.cpu().numpy().tobytes()
        emb = _c.get(key)
        if emb is None:
            raise KeyError(
                f"DNA sequence not in cache (shape={input_ids.shape}, "
                f"trimmed={ids.shape}). "
                "Re-run precompute_dna_embeddings.py with the same --max_length_dna."
            )
        _p = next(self.dna_projection.parameters())
        return emb.to(device=_p.device, dtype=_p.dtype)

    model._evo2_embed = _types.MethodType(_cached_embed, model)

    if model.dna_is_evo2 and getattr(model, "dna_model", None) is not None:
        model.dna_model.model.cpu()
        torch.cuda.empty_cache()
        print("[w9] Evo2 offloaded to CPU — ~14 GB VRAM freed")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(script_args, training_args, model_args):
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    manifold_gate = ManifoldGateW6(stage2_dir=getattr(script_args, "stage2_dir", None))

    model = DNALLMModel(
        text_model_name      = model_args.text_model_name,
        dna_model_name       = model_args.dna_model_name,
        cache_dir            = model_args.cache_dir,
        max_length_text      = model_args.max_length_text,
        max_length_dna       = model_args.max_length_dna,
        text_model_finetune  = True,
        dna_model_finetune   = model_args.dna_model_finetune,
        dna_is_evo2          = model_args.dna_is_evo2,
        dna_embedding_layer  = model_args.dna_embedding_layer,
        use_cross_attention  = model_args.use_cross_attention,
        use_hrpo_gate        = True,
        use_dna_gate         = False,
        device               = "cuda",
    ).to("cuda")
    model.text_model.config.use_cache = False

    hidden_size = model.text_hidden_size

    # ── ThinkingResidualGate (HRPO, input-embedding level) ────────────────────
    use_ot_dist   = manifold_gate.mode != "onthefly"
    thinking_gate = ThinkingResidualGate(
        hidden_size = hidden_size,
        use_ot_dist = use_ot_dist,
        r_min       = 0.5,
        r_max       = 0.99,
    ).to("cuda")

    # ── DNAHiddenInjector (output-hidden level) ───────────────────────────────
    dna_injector = DNAHiddenInjector(
        hidden_size = hidden_size,
        r_min       = 0.7,   # subtle: even fully open keeps 70% original signal
        r_max       = 0.99,
    ).to("cuda")

    # ── LatentSp controller ──────────────────────────────────────────────────────
    _latentSp_use_warmup = (script_args.latentSp_warmup_steps > 0
                         or script_args.latentSp_ramp_steps > 0)
    latentSp_ctrl = LatentSpController(
        theta_low       = 0.0 if _latentSp_use_warmup else script_args.latentSp_theta_low,
        theta_high      = script_args.latentSp_theta_high,
        max_consecutive = script_args.latentSp_max_consec,
    )
    if _latentSp_use_warmup:
        print(f"[w9] LatentSp warmup: theta_low starts at 0 → {script_args.latentSp_theta_low} "
              f"(warmup={script_args.latentSp_warmup_steps} ramp={script_args.latentSp_ramp_steps} steps)")

    print(
        f"[w9] ThinkingResidualGate: hidden={hidden_size}  "
        f"use_ot_dist={use_ot_dist}  "
        f"params={sum(p.numel() for p in thinking_gate.parameters()):,}"
    )
    print(
        f"[w9] DNAHiddenInjector:    hidden={hidden_size}  "
        f"params={sum(p.numel() for p in dna_injector.parameters()):,}"
    )

    # ── Load gate/injector from Stage 1.5 with-gate run (if provided) ────────
    gate_ckpt_path = getattr(script_args, "gate_ckpt", None)
    inj_ckpt_path  = getattr(script_args, "injector_ckpt", None)
    if gate_ckpt_path and os.path.exists(gate_ckpt_path):
        _missing, _unexpected = thinking_gate.load_state_dict(
            torch.load(gate_ckpt_path, map_location="cpu"), strict=False
        )
        if _missing:
            print(f"[w9] thinking_gate missing keys (fresh init): {_missing}")
        if _unexpected:
            print(f"[w9] thinking_gate unexpected keys (ignored): {_unexpected}")
        print(f"[w9] Loaded thinking_gate ← {gate_ckpt_path}")
    elif gate_ckpt_path:
        print(f"[w9] WARNING: gate_ckpt not found: {gate_ckpt_path} — fresh init")
    if inj_ckpt_path and os.path.exists(inj_ckpt_path):
        dna_injector.load_state_dict(
            torch.load(inj_ckpt_path, map_location="cpu"), strict=False
        )
        print(f"[w9] Loaded dna_injector  ← {inj_ckpt_path}")
    elif inj_ckpt_path:
        print(f"[w9] WARNING: injector_ckpt not found: {inj_ckpt_path} — fresh init")

    # ── Ensure latent tokens in vocab BEFORE loading checkpoint ─────────────
    # Stage 1.5 adds all 3 tokens (start/end/pad) expanding vocab 151672→151675.
    # The model must be resized to match before load_state_dict is called.
    tokenizer = model.processor.tokenizer
    _all_latent = [_LATENT_START_TOKEN, _LATENT_END_TOKEN, _LATENT_PAD_TOKEN]
    _missing = [t for t in _all_latent
                if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if _missing:
        tokenizer.add_special_tokens({"additional_special_tokens": _missing})
        model.text_model.resize_token_embeddings(len(tokenizer))
        print(f"[w9] Pre-load: added {_missing} → vocab size={len(tokenizer)}")
    else:
        print(f"[w9] Pre-load: latent tokens already in vocab (size={len(tokenizer)})")

    _load_sft_checkpoint(model, model_args, merge_lora=True)
    _prep_for_training(model, model_args)

    if hasattr(model, "gate_net") and model.gate_net is not None:
        model.gate_net.requires_grad_(False)
        print("[w9] Froze model.gate_net (unused)")

    # ── Resolve latent token IDs (tokens guaranteed present after pre-load step)
    latent_start_id = tokenizer.convert_tokens_to_ids(_LATENT_START_TOKEN)
    latent_end_id   = tokenizer.convert_tokens_to_ids(_LATENT_END_TOKEN)
    print(f"[w9] Latent tokens: {_LATENT_START_TOKEN}={latent_start_id}  "
          f"{_LATENT_END_TOKEN}={latent_end_id}")

    patch_model_for_dual_mode_w9(
        model, thinking_gate, dna_injector, latentSp_ctrl,
        latent_start_id, latent_end_id,
        max_new_tokens = script_args.max_think_tokens + 200,
        lookahead_k    = script_args.latent_lookahead_k,
    )
    model._grpo_force_full_trajectory = False
    # Gate factor initialisation.
    # When gate_ckpt is loaded the model was SFT-trained at MAX_GATE_FACTOR, so the
    # gate must never drop below MAX_GATE_FACTOR/2 during warmup/ramp — doing so puts
    # the model in OOD territory and causes DNA-nucleotide generation loops.
    # _gate_warmup_min_factor clamps the floor so the effective schedule becomes:
    #   steps 0..warmup_steps    : factor = floor  (half-gate, in-distribution)
    #   steps warmup..+ramp_steps: factor ramps floor → MAX_GATE_FACTOR
    #   steps > warmup+ramp      : factor = MAX_GATE_FACTOR
    _gate_ckpt_loaded = (
        script_args.gate_ckpt
        and os.path.isfile(str(script_args.gate_ckpt))
    )
    if script_args.gate_warmup_steps == 0 and script_args.gate_ramp_steps == 0:
        model._gate_warmup_factor = MAX_GATE_FACTOR
        model._gate_warmup_min_factor = MAX_GATE_FACTOR
        print(f"[w9] gate_warmup_steps=0, ramp=0 → gate active from step 0 (factor={MAX_GATE_FACTOR})")
    elif _gate_ckpt_loaded:
        model._gate_warmup_factor     = MAX_GATE_FACTOR
        model._gate_warmup_min_factor = MAX_GATE_FACTOR * 0.5
        print(f"[w9] gate ckpt loaded → floor={MAX_GATE_FACTOR*0.5:.2f}, "
              f"warmup={script_args.gate_warmup_steps} ramp={script_args.gate_ramp_steps} steps")
    else:
        model._gate_warmup_factor     = 0.0
        model._gate_warmup_min_factor = 0.0
    model._max_gate_factor = MAX_GATE_FACTOR  # for GateWarmupCallback logging
    model = model.to(training_args.device)

    # ── DNA embedding cache (optional) ────────────────────────────────────────
    if getattr(model_args, "dna_cache", None):
        _apply_dna_cache(model, model_args.dna_cache)

    # ── Parameter count ───────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[w9] Trainable (model):     {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")
    print(f"[w9] Trainable (gate):      "
          f"{sum(p.numel() for p in thinking_gate.parameters()):,}")
    print(f"[w9] Trainable (injector):  "
          f"{sum(p.numel() for p in dna_injector.parameters()):,}")

    data = get_kegg_dataset(
        kegg_csv              = script_args.kegg_csv,
        dataset_name          = getattr(script_args, "dataset_name", None),
        truncate_dna_per_side = model_args.truncate_dna_per_side,
    )

    _registry = {
        **reward_funcs_registry,
        "ot_distance":  make_ot_reward_func(model, manifold_gate),
        "latent_usage": make_latent_usage_reward_func(
            latentSp_ctrl, latent_start_id, latent_end_id,
        ),
    }
    reward_funcs = [_registry[f] for f in script_args.reward_funcs]
    print(f"[w9] Reward functions: {script_args.reward_funcs}")

    trainer = ThinkingResidualGRPOTrainer(
        thinking_gate    = thinking_gate,
        manifold_gate    = manifold_gate,
        manifold_weight  = getattr(script_args, "manifold_weight", 0.0),
        max_eval_samples = getattr(script_args, "max_eval_samples", None),
        model            = model,
        reward_funcs     = reward_funcs,
        args             = training_args,
        dna_module       = NucleotideDNAModule(),
        train_dataset    = data["train"],
        eval_dataset     = data["val"] if training_args.eval_strategy != "no" else None,
        peft_config      = None,
        callbacks        = [
            SaveWithPyTorchCallback(),
            SaveGateAndInjectorCallback(thinking_gate, dna_injector),
            GateWarmupCallback(
                model,
                warmup_steps = script_args.gate_warmup_steps,
                ramp_steps   = script_args.gate_ramp_steps,
            ),
            LatentSpWarmupCallback(
                latentSp_ctrl,
                target       = script_args.latentSp_theta_low,
                warmup_steps = script_args.latentSp_warmup_steps,
                ramp_steps   = script_args.latentSp_ramp_steps,
            ),
            PreferLatestOnTieCallback(
                metric_name       = training_args.metric_for_best_model,
                greater_is_better = training_args.greater_is_better,
            ),
        ],
        processing_class = model.processor,
    )
    training_args.save_safetensors = False

    # ── Resume ────────────────────────────────────────────────────────────────
    resume = training_args.resume_from_checkpoint
    if resume in ("True", "true"):
        checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        resume = str(max(checkpoints, key=os.path.getmtime)) if checkpoints else None
        print(f"[w9] Auto-resume: {resume}")

    if resume and isinstance(resume, str):
        gate_pt = os.path.join(resume, "thinking_gate.pt")
        inj_pt  = os.path.join(resume, "dna_injector.pt")
        if os.path.exists(gate_pt):
            thinking_gate.load_state_dict(torch.load(gate_pt, map_location="cpu"))
            print(f"[w9] Loaded thinking_gate ← {gate_pt}")
        else:
            print(f"[w9] No thinking_gate.pt in {resume} — fresh init")
        if os.path.exists(inj_pt):
            dna_injector.load_state_dict(torch.load(inj_pt, map_location="cpu"))
            print(f"[w9] Loaded dna_injector  ← {inj_pt}")
        else:
            print(f"[w9] No dna_injector.pt in {resume} — fresh init")

    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
    os.environ.setdefault("WANDB_PROJECT", "dna-grpo-week9")

    parser = TrlParser((GRPOScriptArgumentsW9, DNALLMGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.save_safetensors = False
    training_args.vllm_server_base_url = os.environ.get("VLLM_BASE_URL")

    main(script_args, training_args, model_args)
