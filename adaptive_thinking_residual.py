"""
Stage 3 — GRPO + ThinkingResidualGate.

Merged from:
  adaptive_thinking_residual_w6.py  (Week 6: gate bypassed during rollout generation)
  adaptive_thinking_residual_w8.py  (Week 8: gate applied in BOTH generation and log-prob forward)

Architecture (ThinkingResidualGate)
------------------------------------
    r_t = sigmoid(gate_r(embeds))
    i_t = sigmoid(gate_i(embeds))
    a_t = Lambda(r_t)                           # decay ∈ [r_min, r_max]
    out = a_t * embeds + √(1−a_t²+ε) · i_t · u_dna_broadcast

Applied at a capped factor:  effective = min(warmup_factor, MAX_GATE_FACTOR=0.5)

Week 6 behaviour (patch_model_for_thinking_residual):
  - Gate BYPASSED during rollout generation; applied only via auxiliary L_gate loss.

Week 8 behaviour (patch_model_for_thinking_residual_w8):
  - Gate applied at capped factor in BOTH generation (no grad) and log-prob
    forward (WITH grad) — no train-test distribution mismatch.
  - Gate state saved as thinking_gate.pt at every checkpoint.

Usage (week 8 — recommended)
------------------------------
  STAGE1_CKPT=<path> accelerate launch adaptive_thinking_residual.py \\
      --sft_checkpoint $STAGE1_CKPT \\
      --kegg_csv       genomorph/dataset/kegg_curriculum/global_stage1_anon_genes_mol_keep_chr.csv \\
      --output_dir     /scratch/tanmoyh_iitp/GenoMorph/checkpoints/stage3_grpo \\
      --use_hrpo_gate  True

  # Resume (gate reloaded automatically from thinking_gate.pt):
  RESUME=1 STAGE1_CKPT=<path> bash stage3_grpo.sh
"""

import os
import pathlib
import types
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import LogitsProcessor, LogitsProcessorList, TrainerCallback
from trl import TrlParser

from genomorph.dataset.kegg import format_kegg_for_dna_llm, load_kegg_from_anon_csv
from genomorph.dataset.utils import truncate_dna
from genomorph.dna_modules import NucleotideDNAModule
from genomorph.models.dna_llm import DNALLMModel, get_target_modules
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from genomorph.models.thinking_residual import (
    ThinkingResidualGate,
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
    ManifoldGateW6,
    _build_u_dna,
    make_ot_reward_func,
)

register_evo2_tokenizer()

MAX_GATE_FACTOR = 0.5   # empirical safe cap; uncapped collapsed at ~0.87


# ── Thinking budget: cap <think> block length ─────────────────────────────────

class ThinkingBudgetProcessor(LogitsProcessor):
    """
    Forces </think> after max_think_tokens inside a <think> block, then forces
    EOS after max_answer_tokens outside the think block.
    Each sequence in the batch is tracked independently.
    """

    def __init__(
        self,
        think_start_id:    int,
        think_end_id:      int,
        eos_token_id:      int,
        max_think_tokens:  int,
        max_answer_tokens: int = 150,
    ):
        self.think_start_id    = think_start_id
        self.think_end_id      = think_end_id
        self.eos_token_id      = eos_token_id
        self.max_think_tokens  = max_think_tokens
        self.max_answer_tokens = max_answer_tokens
        self._in_think:     Optional[List[bool]] = None
        self._think_count:  Optional[List[int]]  = None
        self._in_answer:    Optional[List[bool]] = None
        self._answer_count: Optional[List[int]]  = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores:    torch.FloatTensor,
    ) -> torch.FloatTensor:
        if input_ids.shape[1] == 0:
            return scores

        B = input_ids.shape[0]
        if self._in_think is None:
            self._in_think     = [False] * B
            self._think_count  = [0]     * B
            self._in_answer    = [False] * B
            self._answer_count = [0]     * B

        for i in range(B):
            last = input_ids[i, -1].item()
            if last == self.think_start_id:
                self._in_think[i]     = True
                self._think_count[i]  = 0
                self._in_answer[i]    = False
            elif last == self.think_end_id:
                self._in_think[i]     = False
                self._think_count[i]  = 0
                self._in_answer[i]    = True
                self._answer_count[i] = 0
            elif self._in_think[i]:
                self._think_count[i] += 1
            elif self._in_answer[i]:
                self._answer_count[i] += 1

            if self._in_think[i] and self._think_count[i] >= self.max_think_tokens:
                scores[i, :]                 = float("-inf")
                scores[i, self.think_end_id] = 0.0
            elif self._in_answer[i] and self._answer_count[i] >= self.max_answer_tokens:
                scores[i, :]                 = float("-inf")
                scores[i, self.eos_token_id] = 0.0

        return scores


# ── Week 6 generate: gate bypassed during rollout ─────────────────────────────

def _make_tr_generate(
    thinking_gate:    ThinkingResidualGate,
    manifold_gate:    ManifoldGateW6,
    max_think_tokens: int = 600,
):
    """
    Replacement for model.generate_with_hrpo_gate (Week 6 behaviour).
    Gate is NEVER applied during rollout generation; embeddings are cached so
    compute_gate_loss can apply the gate with gradients as an auxiliary loss.
    """

    def _gen(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   Optional[torch.Tensor] = None,
        dna_tokenized     = None,
        batch_idx_map     = None,
        max_latent_steps: int   = 8,
        min_latent_steps: int   = 0,
        gate_threshold:   float = 0.5,
        **gen_kwargs,
    ) -> torch.Tensor:

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        u_dna   = _build_u_dna(self, dna_tokenized, batch_idx_map,
                                input_ids.shape[0], device)
        ot_dist = None
        if u_dna is not None and manifold_gate.mode != "onthefly":
            ot_dist = manifold_gate.get_ot_dist(u_dna, u_dna)

        inputs_embeds, attention_mask = self.get_prompt_embeddings(
            input_ids=input_ids, attention_mask=attention_mask,
            dna_tokenized=dna_tokenized, batch_idx_map=batch_idx_map,
        )
        inputs_embeds  = inputs_embeds.to(device)
        attention_mask = attention_mask.to(device)

        warmup_factor = getattr(self, "_gate_warmup_factor", 1.0)

        if u_dna is not None:
            self._tr_raw_embeds = inputs_embeds.detach()
            self._tr_u_dna      = u_dna.detach() * warmup_factor
            self._tr_ot_dist    = ot_dist.detach() if ot_dist is not None else None
            self._gate_u_dna    = u_dna.detach()
        else:
            self._tr_raw_embeds = None
            self._tr_u_dna      = None
            self._tr_ot_dist    = None
            self._gate_u_dna    = None

        if ot_dist is not None:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank == 0:
                print(f"[TR] ot_dist mean={ot_dist.mean().item():.4f}  "
                      f"min={ot_dist.min().item():.4f}  max={ot_dist.max().item():.4f}  "
                      f"mode={manifold_gate.mode}")

        gen_kwargs.pop("disable_compile", None)
        with torch.no_grad():
            answer_ids = self.text_model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                **gen_kwargs,
            )
        return answer_ids

    return _gen


# ── Shared gate loss (used by both w6 and w8 patches) ────────────────────────

def _make_tr_gate_loss(thinking_gate: ThinkingResidualGate):
    """
    Auxiliary gate loss: entropy regularisation + OT alignment.
    Operates on cached embeddings/u_dna/ot_dist from the last generate call.
    """

    def _gate_loss(self, gate_reg_weight: float = 0.01) -> torch.Tensor:
        zero    = sum(p.sum() for p in thinking_gate.parameters()) * 0.0
        raw     = getattr(self, "_tr_raw_embeds", None)
        u_dna   = getattr(self, "_tr_u_dna",      None)
        ot_dist = getattr(self, "_tr_ot_dist",    None)
        if raw is None or u_dna is None:
            return zero

        B, T, H  = raw.shape
        residual = u_dna.unsqueeze(1).expand(B, T, H).to(raw)
        _, a_t   = thinking_gate(raw, residual, ot_dist)

        entropy_loss = -(a_t * (1.0 - a_t)).mean()

        if ot_dist is not None:
            buf = getattr(self, "_tr_ot_buffer", [])
            buf.append(float(ot_dist.mean().item()))
            if len(buf) > 200:
                buf = buf[-200:]
            self._tr_ot_buffer = buf

            buf_min = min(buf)
            buf_max = max(buf)
            if len(buf) >= 10 and (buf_max - buf_min) > 1e-4:
                blend_target = (
                    (ot_dist.mean() - buf_min) / (buf_max - buf_min + 1e-8)
                ).detach().clamp(0.0, 1.0)
                a_mean   = a_t.mean()
                desired_a = (
                    thinking_gate.lambda_net.r_max
                    - blend_target * (
                        thinking_gate.lambda_net.r_max - thinking_gate.lambda_net.r_min
                    )
                ).detach()
                w = (1.0 + 2.0 * blend_target).detach()
                ot_align_loss = w * (a_mean - desired_a).pow(2)
            else:
                ot_align_loss = torch.zeros(1, device=raw.device)
        else:
            ot_align_loss = torch.zeros(1, device=raw.device)

        return gate_reg_weight * (entropy_loss + ot_align_loss)

    return _gate_loss


# ── Week 6 patch ──────────────────────────────────────────────────────────────

def patch_model_for_thinking_residual(
    model:            DNALLMModel,
    thinking_gate:    ThinkingResidualGate,
    manifold_gate:    ManifoldGateW6,
    max_think_tokens: int = 600,
) -> None:
    """Week 6: patches generate_with_hrpo_gate and compute_gate_loss only."""
    model.generate_with_hrpo_gate = types.MethodType(
        _make_tr_generate(thinking_gate, manifold_gate, max_think_tokens), model
    )
    model.compute_gate_loss = types.MethodType(
        _make_tr_gate_loss(thinking_gate), model
    )
    if thinking_gate.use_ot_dist:
        print(f"[TR] Patched generate_with_hrpo_gate + compute_gate_loss | "
              f"ot_mode={manifold_gate.mode} | ot_scale_init={thinking_gate.ot_scale.item():.3f}")
    else:
        print("[TR] Patched | onthefly mode (no OT conditioning)")


# ── Gate warmup callback ──────────────────────────────────────────────────────

class GateWarmupCallback(TrainerCallback):
    """
    Ramps _gate_warmup_factor from 0 → 1 over training:
      steps 0..warmup_steps            : factor=0  (pure SFT embeddings)
      steps warmup_steps..+ramp_steps  : factor 0→1
      steps > warmup_steps+ramp_steps  : factor=1
    """

    def __init__(self, model, warmup_steps: int, ramp_steps: int):
        self._model       = model
        self.warmup_steps = warmup_steps
        self.ramp_steps   = ramp_steps
        self._prev_phase  = None

    def _rank0_print(self, msg: str):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            print(msg)

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        if step < self.warmup_steps:
            factor = 0.0
            phase  = "warmup"
        elif step < self.warmup_steps + self.ramp_steps:
            factor = (step - self.warmup_steps) / max(self.ramp_steps, 1)
            phase  = "ramp"
        else:
            factor = 1.0
            phase  = "gate"

        self._model._gate_warmup_factor = factor

        if phase != self._prev_phase:
            if phase == "warmup":
                self._rank0_print(
                    f"\n[GateWarmup] ── WARMUP START  (step {step}) ──────────────────────\n"
                    f"[GateWarmup]    Gate BYPASSED — pure SFT embeddings for {self.warmup_steps} steps.\n"
                )
            elif phase == "ramp":
                self._rank0_print(
                    f"\n[GateWarmup] ── RAMP START  (step {step}) ───────────────────────\n"
                    f"[GateWarmup]    Gate factor 0→1 over {self.ramp_steps} steps.\n"
                )
            elif phase == "gate":
                actual = min(factor, getattr(self._model, "_max_gate_factor", MAX_GATE_FACTOR))
                self._rank0_print(
                    f"\n[GateWarmup] ── FULL GATE  (step {step}) ───────────────────────\n"
                    f"[GateWarmup]    Actual embed blend = min(1.0, MAX_GATE_FACTOR={actual}).\n"
                )
            self._prev_phase = phase

        if step % args.logging_steps == 0:
            actual = min(factor, getattr(self._model, "_max_gate_factor", MAX_GATE_FACTOR))
            self._rank0_print(
                f"[GateWarmup] schedule_factor={factor:.3f}  actual_blend={actual:.3f}"
                f"  phase={phase}  step={step}"
            )


# ── Trainer with manifold loss ────────────────────────────────────────────────

class ThinkingResidualGRPOTrainer(DNALLMGRPOTrainer):
    """DNALLMGRPOTrainer with differential LRs for the gate and optional L_manifold."""

    def __init__(
        self,
        thinking_gate:    ThinkingResidualGate,
        manifold_gate:    ManifoldGateW6,
        manifold_weight:  float = 0.0,
        max_eval_samples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.thinking_gate    = thinking_gate
        self.manifold_gate    = manifold_gate
        self.manifold_weight  = manifold_weight
        self.max_eval_samples = max_eval_samples

    def create_optimizer(self):
        base_lr    = self.args.learning_rate
        param_groups = make_thinking_residual_param_groups(
            model                = self.model,
            thinking_gate        = self.thinking_gate,
            base_lr              = base_lr,
            lr_multiplier_gate   = 20.0,
            lr_multiplier_lambda = 20.0,
            weight_decay         = self.args.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(param_groups)
        print(f"[TR] Optimizer: {len(param_groups)} param groups | "
              f"base_lr={base_lr:.2e}  gate_lr={base_lr*20:.2e}")
        return self.optimizer

    def _compute_val_correctness(self, eval_dataset=None) -> dict:
        import time
        import torch.distributed as dist
        from functools import partial
        from collections import defaultdict
        from genomorph.dataset.kegg import qwen_dna_collate_fn

        empty = {"correctness": 0.0, "n_correct": 0, "n_total": 0,
                 "prec_macro": 0.0, "prec_weighted": 0.0,
                 "recall_macro": 0.0, "recall_weighted": 0.0,
                 "macro_f1": 0.0, "weighted_f1": 0.0, "mean_time_sec": 0.0}
        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            return empty

        max_samples = self.max_eval_samples
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        rank       = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        unwrapped  = self.accelerator.unwrap_model(self.model)
        processor  = self.processing_class
        device     = self.args.device

        collate_fn = partial(
            qwen_dna_collate_fn,
            processor               = processor,
            max_length_text         = unwrapped.max_length_text,
            max_length_dna          = unwrapped.max_length_dna,
            return_answer_in_batch  = True,
            truncate_for_generation = True,
        )

        text_model     = unwrapped.text_model
        gc_was_enabled = getattr(text_model, "is_gradient_checkpointing", False)
        if gc_was_enabled:
            text_model.gradient_checkpointing_disable()

        # Per-sample records: (gt, pred, is_correct, gen_time_sec)
        local_records = []
        for i in range(rank, len(dataset), world_size):
            sample = dataset[i]
            try:
                batch = collate_fn([sample])
            except Exception as exc:
                print(f"[CorrectnessEval] rank={rank} collation failed: {exc}")
                continue

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers        = batch.get("answer", [])
            dna_tokenized  = batch.get("dna_tokenized")
            batch_idx_map  = batch.get("batch_idx_map", [])

            with torch.no_grad():
                try:
                    _t0 = time.perf_counter()
                    generated_ids = unwrapped.generate_with_hrpo_gate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        dna_tokenized=dna_tokenized, batch_idx_map=batch_idx_map,
                        max_new_tokens=800, temperature=0.0,
                        repetition_penalty=getattr(self.args, "repetition_penalty", 1.2),
                    )
                    _gen_time = time.perf_counter() - _t0
                except Exception as exc:
                    print(f"[CorrectnessEval] rank={rank} generation failed: {exc}")
                    continue

            _per = _gen_time / max(len(answers), 1)   # per-sample share of this call
            for gen_ids, answer in zip(generated_ids, answers):
                text       = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
                extracted  = NucleotideDNAModule._extract_xml_answer(text)
                gt         = answer.lower().strip()
                pred       = extracted.lower().strip()
                is_correct = bool(pred) and gt in pred   # one-dir (matches test_06b)
                local_records.append((gt, pred, is_correct, _per))
                if rank == 0:
                    mark = "✓" if is_correct else "✗"
                    print(f"[CorrectnessEval] {mark} answer={repr(answer)} | "
                          f"extracted={repr(extracted[:80])}")

        if gc_was_enabled:
            text_model.gradient_checkpointing_enable()

        # Gather per-sample records across ranks so every rank sees the full set
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_records)
            all_records = [r for sub in gathered if sub for r in sub]
        else:
            all_records = local_records

        total = len(all_records)
        if total == 0:
            return empty

        correct   = sum(1 for (_, _, c, _) in all_records if c)
        acc       = correct / total
        mean_time = sum(t for (_, _, _, t) in all_records) / total

        # Precision / recall / F1 (macro + weighted) via sklearn — same set and
        # same one-directional rule as the standalone eval_stage1_51_checkpoints.py,
        # so the training monitor and the standalone test report identically.
        # A correct prediction counts as y_pred == gt; an incorrect one keeps its
        # raw pred (so it lands in the wrong class).
        prec_mac = rec_mac = macro_f1 = 0.0
        prec_w = rec_w = weighted_f1 = 0.0
        try:
            from sklearn.metrics import (f1_score as sk_f1,
                                         precision_score, recall_score)
            y_true = [gt for gt, _, _, _ in all_records]
            y_pred = [gt if c else pred for gt, pred, c, _ in all_records]
            labels = sorted(set(y_true))
            prec_mac    = float(precision_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0))
            rec_mac     = float(recall_score(   y_true, y_pred, labels=labels, average="macro",    zero_division=0))
            macro_f1    = float(sk_f1(          y_true, y_pred, labels=labels, average="macro",    zero_division=0))
            prec_w      = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
            rec_w       = float(recall_score(   y_true, y_pred, labels=labels, average="weighted", zero_division=0))
            weighted_f1 = float(sk_f1(          y_true, y_pred, labels=labels, average="weighted", zero_division=0))
        except Exception as exc:
            print(f"[CorrectnessEval] sklearn metrics failed: {exc}")

        return {"correctness": acc, "n_correct": correct, "n_total": total,
                "prec_macro": prec_mac, "prec_weighted": prec_w,
                "recall_macro": rec_mac, "recall_weighted": rec_w,
                "macro_f1": macro_f1, "weighted_f1": weighted_f1,
                "mean_time_sec": mean_time}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        res = self._compute_val_correctness(eval_dataset)
        p   = metric_key_prefix
        metrics = {
            f"{p}_correctness":     res["correctness"],   # metric_for_best_model
            f"{p}_prec_macro":      res["prec_macro"],
            f"{p}_prec_weighted":   res["prec_weighted"],
            f"{p}_recall_macro":    res["recall_macro"],
            f"{p}_recall_weighted": res["recall_weighted"],
            f"{p}_macro_f1":        res["macro_f1"],
            f"{p}_weighted_f1":     res["weighted_f1"],
            f"{p}_mean_time_sec":   res["mean_time_sec"],
        }
        self.log(metrics)
        if self.accelerator.process_index == 0:
            print(f"\n  Step            : {self.state.global_step}")
            print(f"  Accuracy        : {res['correctness']:.4f}  ({res['n_correct']}/{res['n_total']})")
            print(f"  Precision       : {res['prec_macro']:.4f}  (macro)   {res['prec_weighted']:.4f}  (weighted)")
            print(f"  Recall          : {res['recall_macro']:.4f}  (macro)   {res['recall_weighted']:.4f}  (weighted)")
            print(f"  F1              : {res['macro_f1']:.4f}  (macro)   {res['weighted_f1']:.4f}  (weighted)")
            print(f"  Mean time/sample: {res['mean_time_sec']:.2f}s\n")
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        if loss.item() == 0.0:
            return loss

        if (self.manifold_gate.mode == "onthefly"
                or not model.training
                or self.manifold_weight <= 0.0):
            return loss

        multimodal_inputs = inputs.get("multimodal_inputs", {})
        dna_tokenized     = multimodal_inputs.get("dna_tokenized")
        batch_idx_map     = multimodal_inputs.get("batch_idx_map", [])
        prompt_ids        = inputs["prompt_ids"]

        if dna_tokenized is None:
            return loss

        unwrapped  = self.accelerator.unwrap_model(model)
        batch_size = prompt_ids.shape[0]
        batch_dna_embeds = unwrapped.process_dna_embeddings(
            dna_tokenized, batch_idx_map, batch_size
        )
        _dtype  = next(unwrapped.dna_projection.parameters()).dtype
        _device = prompt_ids.device

        u_list = []
        for i in range(batch_size):
            parts = []
            for slot in (i, i + batch_size):
                if slot < len(batch_dna_embeds) and batch_dna_embeds[slot].shape[0] > 0:
                    parts.append(batch_dna_embeds[slot])
            u_list.append(
                torch.cat(parts, dim=0).mean(dim=0) if parts
                else torch.zeros(unwrapped.text_hidden_size, device=_device, dtype=_dtype)
            )
        u_norm = F.normalize(torch.stack(u_list).float(), dim=-1)

        l_manifold = self.manifold_gate.manifold_loss(u_norm)
        if l_manifold is None:
            return loss

        loss = loss + self.manifold_weight * l_manifold
        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("manifold_loss", []).append(
            self.accelerator.gather_for_metrics(l_manifold.detach()).mean().item()
        )
        return loss


# ── Week 8: save thinking_gate.pt at every checkpoint ────────────────────────

class SaveGateCallback(TrainerCallback):
    """Save thinking_gate.pt alongside every model checkpoint."""

    def __init__(self, thinking_gate: ThinkingResidualGate):
        self.thinking_gate = thinking_gate

    def on_save(self, args, state, control, **kwargs):
        folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "thinking_gate.pt")
        torch.save(self.thinking_gate.state_dict(), path)
        print(f"[SaveGate] thinking_gate → {path}")


# ── Week 8 generate: gate at capped factor ────────────────────────────────────

def _make_tr_generate_w8(
    thinking_gate:    ThinkingResidualGate,
    manifold_gate:    ManifoldGateW6,
    max_think_tokens: int = 600,
):
    """
    Replacement for model.generate_with_hrpo_gate (Week 8 behaviour).
    Gate applied at min(warmup_factor, MAX_GATE_FACTOR) — same cap as the
    log-prob forward — eliminating train-test distribution mismatch.
    """

    def _gen(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   Optional[torch.Tensor] = None,
        dna_tokenized     = None,
        batch_idx_map     = None,
        max_latent_steps: int   = 8,
        min_latent_steps: int   = 0,
        gate_threshold:   float = 0.5,
        **gen_kwargs,
    ) -> torch.Tensor:

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        u_dna   = _build_u_dna(self, dna_tokenized, batch_idx_map,
                                input_ids.shape[0], device)
        ot_dist = None
        if u_dna is not None and manifold_gate.mode != "onthefly":
            ot_dist = manifold_gate.get_ot_dist(u_dna, u_dna)

        inputs_embeds, attention_mask = self.get_prompt_embeddings(
            input_ids=input_ids, attention_mask=attention_mask,
            dna_tokenized=dna_tokenized, batch_idx_map=batch_idx_map,
        )
        inputs_embeds  = inputs_embeds.to(device)
        attention_mask = attention_mask.to(device)

        factor = min(getattr(self, "_gate_warmup_factor", 0.0), MAX_GATE_FACTOR)

        if factor > 0 and u_dna is not None:
            B, T, H  = inputs_embeds.shape
            residual = u_dna.detach().unsqueeze(1).expand(B, T, H).to(inputs_embeds)
            ot_for_gen = ot_dist.detach() if ot_dist is not None else None
            with torch.no_grad():
                gate_out, _ = thinking_gate(inputs_embeds, residual, ot_for_gen)
            modified_embeds = (1.0 - factor) * inputs_embeds + factor * gate_out
        else:
            modified_embeds = inputs_embeds

        if u_dna is not None:
            self._v5_u_dna      = u_dna.detach()
            self._v5_ot_dist    = ot_dist.detach() if ot_dist is not None else None
            self._tr_raw_embeds = inputs_embeds.detach()
            self._tr_u_dna      = u_dna.detach() * factor
            self._tr_ot_dist    = ot_dist.detach() if ot_dist is not None else None
            self._gate_u_dna    = u_dna.detach()
        else:
            self._v5_u_dna = self._v5_ot_dist = None
            self._tr_raw_embeds = self._tr_u_dna = self._tr_ot_dist = self._gate_u_dna = None

        if ot_dist is not None:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank == 0:
                print(f"[w8] ot_dist mean={ot_dist.mean().item():.4f}  "
                      f"factor={factor:.3f}  mode={manifold_gate.mode}")

        gen_kwargs.pop("disable_compile", None)
        with torch.no_grad():
            answer_ids = self.text_model.generate(
                inputs_embeds=modified_embeds, attention_mask=attention_mask,
                **gen_kwargs,
            )
        return answer_ids

    return _gen


# ── Week 8 log-prob forward: gate WITH gradients ──────────────────────────────

def _make_tr_forward_w8(thinking_gate: ThinkingResidualGate):
    """
    Patched DNALLMModel.forward (Week 8).
    Applies gate WITH gradients at the same capped factor as generation.
    GRPO policy gradient flows through gate parameters via this path.
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

        embed_layer        = self.text_model.get_input_embeddings()
        text_inputs_embeds = embed_layer(input_ids)

        if dna_tokenized is not None and batch_idx_map:
            batch_dna_embeds = self.process_dna_embeddings(
                dna_tokenized, batch_idx_map, batch_size,
                text_context=text_inputs_embeds, text_mask=attention_mask,
            )
            mask            = input_ids == self.dna_token_id
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0).to(
                dtype=text_inputs_embeds.dtype, device=device
            )
            text_inputs_embeds[mask] = dna_embeds_flat

        factor       = min(getattr(self, "_gate_warmup_factor", 0.0), MAX_GATE_FACTOR)
        cached_u_dna = getattr(self, "_v5_u_dna",  None)
        cached_ot    = getattr(self, "_v5_ot_dist", None)

        if factor > 0 and cached_u_dna is not None:
            B_gen = cached_u_dna.shape[0]
            if B_gen == 1:
                u_dna_fwd   = cached_u_dna.expand(batch_size, -1).to(device)
                ot_dist_fwd = (cached_ot.expand(batch_size).to(device)
                               if cached_ot is not None else None)
            elif batch_size % B_gen == 0:
                G           = batch_size // B_gen
                u_dna_fwd   = cached_u_dna.repeat_interleave(G, dim=0).to(device)
                ot_dist_fwd = (cached_ot.repeat_interleave(G, dim=0).to(device)
                               if cached_ot is not None else None)
            else:
                u_dna_fwd = ot_dist_fwd = None

            if u_dna_fwd is not None:
                B, T, H  = text_inputs_embeds.shape
                residual = u_dna_fwd.unsqueeze(1).expand(B, T, H).to(text_inputs_embeds)
                gate_out, _ = thinking_gate(text_inputs_embeds, residual, ot_dist_fwd)
                text_inputs_embeds = (1.0 - factor) * text_inputs_embeds + factor * gate_out

        return self.text_model(
            inputs_embeds=text_inputs_embeds, attention_mask=attention_mask,
            labels=labels, **kwargs,
        )

    return _forward


# ── Week 8 patch ──────────────────────────────────────────────────────────────

def patch_model_for_thinking_residual_w8(
    model:            DNALLMModel,
    thinking_gate:    ThinkingResidualGate,
    manifold_gate:    ManifoldGateW6,
    max_think_tokens: int = 600,
) -> None:
    """
    Week 8: patches generate_with_hrpo_gate, forward, and compute_gate_loss.
    All three use the same capped factor — no train-test distribution mismatch.
    """
    model.generate_with_hrpo_gate = types.MethodType(
        _make_tr_generate_w8(thinking_gate, manifold_gate, max_think_tokens), model
    )
    model.forward = types.MethodType(
        _make_tr_forward_w8(thinking_gate), model
    )
    model.compute_gate_loss = types.MethodType(
        _make_tr_gate_loss(thinking_gate), model
    )
    n = sum(p.numel() for p in thinking_gate.parameters())
    print(
        f"[w8] Patched: generate_with_hrpo_gate | forward | compute_gate_loss\n"
        f"[w8] Gate params={n:,}  MAX_GATE_FACTOR={MAX_GATE_FACTOR}\n"
        f"[w8] Gate in generation : YES (capped at {MAX_GATE_FACTOR})\n"
        f"[w8] Gate in log-prob   : YES (capped at {MAX_GATE_FACTOR})\n"
        f"[w8] Gate in gate_loss  : YES (auxiliary entropy + OT alignment)"
    )


# ── Main (Week 8) ─────────────────────────────────────────────────────────────

def main(script_args, training_args, model_args):
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    manifold_gate = ManifoldGateW6(stage2_dir=getattr(script_args, "stage2_dir", None))

    model = DNALLMModel(
        text_model_name     = model_args.text_model_name,
        dna_model_name      = model_args.dna_model_name,
        cache_dir           = model_args.cache_dir,
        max_length_text     = model_args.max_length_text,
        max_length_dna      = model_args.max_length_dna,
        text_model_finetune = True,
        dna_model_finetune  = model_args.dna_model_finetune,
        dna_is_evo2         = model_args.dna_is_evo2,
        dna_embedding_layer = model_args.dna_embedding_layer,
        use_cross_attention = model_args.use_cross_attention,
        use_hrpo_gate       = True,
        use_dna_gate        = False,
        device              = "cuda",
    ).to("cuda")
    model.text_model.config.use_cache = False

    hidden_size   = model.text_hidden_size
    use_ot_dist   = manifold_gate.mode != "onthefly"
    thinking_gate = ThinkingResidualGate(
        hidden_size=hidden_size, use_ot_dist=use_ot_dist, r_min=0.5, r_max=0.99,
    ).to("cuda")
    print(f"[w8] ThinkingResidualGate: hidden={hidden_size}  "
          f"use_ot_dist={use_ot_dist}  "
          f"params={sum(p.numel() for p in thinking_gate.parameters()):,}")

    _load_sft_checkpoint(model, model_args, merge_lora=True)
    _prep_for_training(model, model_args)

    if hasattr(model, "gate_net") and model.gate_net is not None:
        model.gate_net.requires_grad_(False)
        print("[w8] Froze model.gate_net (unused)")

    patch_model_for_thinking_residual_w8(
        model, thinking_gate, manifold_gate,
        max_think_tokens=script_args.max_think_tokens,
    )
    model._grpo_force_full_trajectory = False
    model._gate_warmup_factor = 0.0
    model = model.to(training_args.device)

    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total       = sum(p.numel() for p in model.parameters())
    gate_params = sum(p.numel() for p in thinking_gate.parameters())
    print(f"Trainable (model): {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"Trainable (gate):  {gate_params:,}")

    data = get_kegg_dataset(
        kegg_csv              = script_args.kegg_csv,
        dataset_name          = getattr(script_args, "dataset_name", None),
        truncate_dna_per_side = model_args.truncate_dna_per_side,
    )

    _registry = {
        **reward_funcs_registry,
        "ot_distance": make_ot_reward_func(model, manifold_gate),
    }
    reward_funcs = [_registry[f] for f in script_args.reward_funcs]
    print(f"[w8] Reward functions: {script_args.reward_funcs}")

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
            SaveGateCallback(thinking_gate),
            GateWarmupCallback(
                model,
                warmup_steps=script_args.gate_warmup_steps,
                ramp_steps=script_args.gate_ramp_steps,
            ),
        ],
        processing_class = model.processor,
    )
    training_args.save_safetensors = False

    resume = training_args.resume_from_checkpoint
    if resume in ("True", "true"):
        checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        resume = str(max(checkpoints, key=os.path.getmtime)) if checkpoints else None
        print(f"Auto-resume: {resume}")

    if resume and isinstance(resume, str):
        gate_pt = os.path.join(resume, "thinking_gate.pt")
        if os.path.exists(gate_pt):
            thinking_gate.load_state_dict(torch.load(gate_pt, map_location="cpu"))
            print(f"[w8] Loaded thinking_gate ← {gate_pt}")
        else:
            print(f"[w8] No thinking_gate.pt in {resume} — fresh init (a_t≈0.99)")

    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
    os.environ.setdefault("WANDB_PROJECT", "dna-grpo-stage3")

    parser = TrlParser((GRPOScriptArguments, DNALLMGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.save_safetensors = False
    training_args.vllm_server_base_url = os.environ.get("VLLM_BASE_URL")

    main(script_args, training_args, model_args)
