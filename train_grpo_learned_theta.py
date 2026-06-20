"""
Stage 3 — Entropy-Conditioned Dual-Mode Reasoning with Learnable θ_low (Option B).

Identical to adaptive_thinking_residual_w9.py except theta_low is an nn.Parameter
trained end-to-end alongside GRPO via REINFORCE:

  p_latent = sigmoid(alpha * (theta_low - entropy))   # alpha=5.0 fixed
  theta_loss = -adv_mean * mean_over_boundaries(log π(decision | entropy, theta))

The warmup schedule sets _warmup_scale (0→1) rather than overwriting theta_low
directly, so gradients flow through theta_low_param from the first active step.

New CLI args vs w9:
  --theta_low_lr     LR for theta_low_param (default 1e-4, ~20x base LR)
  --theta_low_weight Scale factor on theta_loss (default 0.1)
  --theta_low_alpha  Sigmoid temperature (default 5.0, keep fixed)

Usage:
  accelerate launch adaptive_thinking_residual_w9_optB.py \\
      --sft_checkpoint  <stage1.5_ckpt/model.pt> \\
      --output_dir      /scratch/.../stage3_grpo_optB \\
      --theta_low_lr    1e-4 \\
      --theta_low_weight 0.1

  # Resume
  RESUME=1 bash week11tests/sh_stage3_grpo_w9_optB.sh
"""

import os
import pathlib
import types
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

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
from adaptive_latent_grpo_w6 import (
    ManifoldGateW6,
    _build_u_dna,
    make_ot_reward_func,
)
from adaptive_thinking_residual_w6 import (
    GateWarmupCallback,
    ThinkingResidualGRPOTrainer,
    _make_tr_gate_loss,
)
# Import unchanged w9 components.
# _make_dual_mode_generate_w9 closes over latentSp_ctrl and calls
# latentSp_ctrl.is_latent_step() / is_dna_inject() — duck-typed, so our
# LatentSpControllerOptB works without touching the generation loop.
from adaptive_thinking_residual_w9 import (
    MAX_GATE_FACTOR,
    GRPOScriptArgumentsW9,
    DNAHiddenInjector,
    _make_dual_mode_generate_w9,
    _make_dual_mode_forward_w9,
    _make_tr_gate_loss_w9,
    SaveGateAndInjectorCallback,
    make_latent_usage_reward_func,
    patch_model_for_dual_mode_w9,
    _apply_dna_cache,
    _LATENT_START_TOKEN,
    _LATENT_END_TOKEN,
    _LATENT_PAD_TOKEN,
)

register_evo2_tokenizer()


# ── Extended script args ───────────────────────────────────────────────────────

@dataclass
class GRPOScriptArgumentsOptB(GRPOScriptArgumentsW9):
    """Adds REINFORCE fields for learnable theta_low on top of all w9 args."""

    theta_low_lr: float = field(
        default=1e-4,
        metadata={"help": "LR for theta_low_param in its own AdamW param group. "
                          "~20x the base LR is a reasonable starting point since "
                          "the REINFORCE signal is sparse (one update per step)."},
    )
    theta_low_weight: float = field(
        default=0.1,
        metadata={"help": "Coefficient on the REINFORCE theta_loss added to the main "
                          "GRPO loss.  0.1 keeps theta_loss at ~10% of total loss magnitude."},
    )
    theta_low_alpha: float = field(
        default=5.0,
        metadata={"help": "Sigmoid temperature: p_latent = sigmoid(alpha*(theta - entropy)). "
                          "Higher alpha → sharper decision boundary.  Keep fixed during training."},
    )


# ── Learnable LatentSp controller ─────────────────────────────────────────────

class LatentSpControllerOptB(nn.Module):
    """
    Drop-in replacement for w9's LatentSpController with learnable theta_low.

    theta_low is an nn.Parameter (1-D scalar) trained via REINFORCE.
    Effective threshold = clamp(theta_low_param, 0.05, 8.0) * _warmup_scale.

    During generation (inside torch.no_grad()):
      p_latent = sigmoid(alpha * (eff_theta - entropy))
      decision ~ Bernoulli(p_latent)
      decision is stored in _step_decisions for the REINFORCE backward.

    In compute_loss():
      theta_loss = -adv_mean * mean(log π(decisions))
    where log π is re-computed WITH gradient flow through theta_low_param.

    DDP note: theta_low_param lives outside the DDP-wrapped model. Each rank
    updates it from its own completions but using the shared mean advantage.
    The 1-D parameter stays directionally correct across ranks; exact sync is
    not needed for the research experiment.
    """

    def __init__(
        self,
        theta_low_init:  float = 1.0,
        theta_high:      float = 2.5,
        max_consecutive: int   = 3,
        alpha:           float = 5.0,
    ):
        super().__init__()
        self.theta_low_param  = nn.Parameter(torch.tensor(float(theta_low_init)))
        self.theta_high       = theta_high
        self.max_consecutive  = max_consecutive
        self.alpha            = alpha  # fixed sigmoid temperature

        # Set by LatentSpWarmupCallbackOptB at each step.
        self._warmup_scale: float = 0.0

        # (entropy: float, was_latent: bool) per boundary decision this step.
        # Cleared by LatentSpWarmupCallbackOptB.on_step_begin.
        self._step_decisions: List[Tuple[float, bool]] = []

    @property
    def theta_low(self) -> float:
        """Effective theta_low for display / downstream callers."""
        return float(self.theta_low_param.clamp(0.05, 8.0).item()) * self._warmup_scale

    def is_latent_step(self, entropy: float, consecutive: int) -> bool:
        """
        Stochastic latent-step decision; records (entropy, decision) for REINFORCE.

        Falls back to deterministic False when _warmup_scale=0 (warmup phase) so
        generation is well-defined even before the param has been trained at all.
        """
        if self._warmup_scale <= 0.0:
            return False

        eff_theta = float(self.theta_low_param.clamp(0.05, 8.0).item()) * self._warmup_scale
        with torch.no_grad():
            p = torch.sigmoid(
                torch.tensor(self.alpha * (eff_theta - entropy), dtype=torch.float32)
            ).item()
        was_latent = (
            bool(torch.bernoulli(torch.tensor(p)).item())
            and consecutive < self.max_consecutive
        )
        self._step_decisions.append((entropy, was_latent))
        return was_latent

    def is_dna_inject(self, entropy: float) -> bool:
        return entropy > self.theta_high

    def clear_decisions(self):
        self._step_decisions.clear()

    def compute_theta_loss(self, adv_mean: float) -> Optional[torch.Tensor]:
        """
        REINFORCE loss: -adv_mean * mean_over_boundaries(log π(decision | entropy)).

        Gradient flows through theta_low_param via the sigmoid.
          adv_mean > 0 + was_latent=True  → push theta_low UP  (fire more)
          adv_mean < 0 + was_latent=True  → push theta_low DOWN (fire less)
          adv_mean > 0 + was_latent=False → push theta_low DOWN (non-latent rewarded)
        This is the correct REINFORCE direction for making decisions consistent
        with what earned high advantage.

        Uses log(sigmoid(x)) = -softplus(-x) for numerical stability, and
        vectorises the per-decision computation into a single tensor op.
        """
        if not self._step_decisions or self._warmup_scale <= 0.0:
            return None

        dtype  = self.theta_low_param.dtype
        device = self.theta_low_param.device
        eff_theta = self.theta_low_param.clamp(0.05, 8.0) * self._warmup_scale

        entropies   = torch.tensor([h for h, _ in self._step_decisions],
                                   dtype=dtype, device=device)
        was_latent  = torch.tensor([d for _, d in self._step_decisions],
                                   dtype=torch.bool, device=device)
        logits      = self.alpha * (eff_theta - entropies)
        # log p          = log σ(x)       = -softplus(-x)
        # log (1 - p)    = log σ(-x)      = -softplus(x)
        log_pi      = torch.where(
            was_latent, -F.softplus(-logits), -F.softplus(logits)
        )
        return -float(adv_mean) * log_pi.mean()


# ── Warmup callback (sets _warmup_scale, clears buffer) ───────────────────────

class LatentSpWarmupCallbackOptB(TrainerCallback):
    """
    Ramps latentSp_ctrl._warmup_scale from 0 → 1 and clears the REINFORCE
    decision buffer at the start of each training step.

    Unlike w9's LatentSpWarmupCallback (which wrote theta_low directly),
    this one never touches theta_low_param — the optimizer owns that.
    Setting _warmup_scale to 0 during warmup simply gates is_latent_step
    to return False without corrupting the learned parameter value.
    """

    def __init__(
        self,
        ctrl:         LatentSpControllerOptB,
        warmup_steps: int,
        ramp_steps:   int,
    ):
        self._ctrl        = ctrl
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

        # Always clear buffer so each step starts fresh
        self._ctrl.clear_decisions()

        if step < self.warmup_steps:
            scale, phase = 0.0, "warmup"
        elif step < self.warmup_steps + self.ramp_steps:
            frac         = (step - self.warmup_steps) / max(self.ramp_steps, 1)
            scale, phase = frac, "ramp"
        else:
            scale, phase = 1.0, "active"

        self._ctrl._warmup_scale = scale

        if phase != self._prev_phase:
            msgs = {
                "warmup": (
                    f"\n[LatentSpOptB] ── WARMUP  (step {step})\n"
                    f"[LatentSpOptB]    scale=0 for {self.warmup_steps} steps "
                    f"— is_latent_step=False, theta_low_param free to warm up.\n"
                ),
                "ramp": (
                    f"\n[LatentSpOptB] ── RAMP  (step {step})\n"
                    f"[LatentSpOptB]    scale 0→1 over {self.ramp_steps} steps "
                    f"— stochastic latent steps and REINFORCE active.\n"
                ),
                "active": (
                    f"\n[LatentSpOptB] ── ACTIVE  (step {step})\n"
                    f"[LatentSpOptB]    scale=1.0  "
                    f"theta_low_param={self._ctrl.theta_low_param.item():.4f}\n"
                ),
            }
            self._rank0_print(msgs[phase])
            self._prev_phase = phase

        if step % args.logging_steps == 0 and scale > 0:
            self._rank0_print(
                f"[LatentSpOptB] scale={scale:.4f}  "
                f"theta_low_param={self._ctrl.theta_low_param.item():.4f}  "
                f"eff_theta={self._ctrl.theta_low:.4f}  step={step}"
            )


# ── Save callback (gate + injector + theta_low_param) ─────────────────────────

class SaveGateInjectorThetaCallback(TrainerCallback):
    """Save thinking_gate.pt, dna_injector.pt, and theta_low.pt at every checkpoint."""

    def __init__(
        self,
        thinking_gate:  ThinkingResidualGate,
        dna_injector:   DNAHiddenInjector,
        latentSp_ctrl:  LatentSpControllerOptB,
    ):
        self.thinking_gate = thinking_gate
        self.dna_injector  = dna_injector
        self.latentSp_ctrl = latentSp_ctrl

    def on_save(self, args, state, control, **kwargs):
        folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(folder, exist_ok=True)
        torch.save(self.thinking_gate.state_dict(),
                   os.path.join(folder, "thinking_gate.pt"))
        torch.save(self.dna_injector.state_dict(),
                   os.path.join(folder, "dna_injector.pt"))
        torch.save({"theta_low_param": self.latentSp_ctrl.theta_low_param.data},
                   os.path.join(folder, "theta_low.pt"))
        print(
            f"[OptB] thinking_gate + dna_injector + theta_low "
            f"→ {folder}/  (theta_low={self.latentSp_ctrl.theta_low_param.item():.4f})"
        )


# ── Tie-breaking: prefer latest checkpoint when metric is equal ───────────────

class PreferLatestOnTieCallback(TrainerCallback):
    """When eval metric equals the current best, update best_model_checkpoint to
    point at the latest checkpoint *before* _save_checkpoint runs rotation, so
    save_total_limit protects the most recent tied checkpoint instead of the
    older one. Fires inside on_evaluate, ahead of _determine_best_metric and
    _save_checkpoint — the folder need not exist yet (it's just a path string)."""

    def __init__(self, metric_name: str, greater_is_better: bool = True):
        self.metric_name      = metric_name
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


# ── Trainer with REINFORCE theta_loss ─────────────────────────────────────────

class ThinkingResidualGRPOTrainer_OptB(ThinkingResidualGRPOTrainer):
    """
    Extends ThinkingResidualGRPOTrainer with:
      1. theta_low_param as an extra AdamW param group (create_optimizer override)
      2. REINFORCE theta_loss added to the total loss (compute_loss override)
    """

    def __init__(
        self,
        latentSp_ctrl:    LatentSpControllerOptB,
        theta_low_lr:     float = 1e-4,
        theta_low_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latentSp_ctrl    = latentSp_ctrl
        self.theta_low_lr     = theta_low_lr
        self.theta_low_weight = theta_low_weight

        # theta_low_param lives outside the DDP-wrapped model. Without sync,
        # each rank would update it from its own micro-batch and drift apart
        # over thousands of steps. Hook averages the grad across ranks so the
        # parameter stays in lock-step.
        self.latentSp_ctrl.theta_low_param.register_hook(self._sync_theta_grad)

    @staticmethod
    def _sync_theta_grad(grad: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(dist.get_world_size())
        return grad

    def create_optimizer(self):
        """Add theta_low_param as a separate AdamW param group."""
        base_lr = self.args.learning_rate
        param_groups = make_thinking_residual_param_groups(
            model                = self.model,
            thinking_gate        = self.thinking_gate,
            base_lr              = base_lr,
            lr_multiplier_gate   = 20.0,
            lr_multiplier_lambda = 20.0,
            weight_decay         = self.args.weight_decay,
        )
        param_groups.append({
            "params":       [self.latentSp_ctrl.theta_low_param],
            "lr":           self.theta_low_lr,
            "weight_decay": 0.0,
        })
        self.optimizer = torch.optim.AdamW(param_groups)
        print(
            f"[OptB] Optimizer: {len(param_groups)} param groups | "
            f"base_lr={base_lr:.2e}  "
            f"gate_lr={base_lr*20:.2e}  "
            f"theta_low_lr={self.theta_low_lr:.2e}"
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Standard GRPO + manifold loss from parent
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        if loss.item() == 0.0:
            return loss  # anomaly guard zeroed the loss — skip theta_loss too

        # REINFORCE: theta_loss uses decisions recorded during generate_with_hrpo_gate
        decisions = self.latentSp_ctrl._step_decisions
        if decisions and self.latentSp_ctrl._warmup_scale > 0.0:
            adv = inputs.get("advantages", None)
            if adv is not None:
                adv_mean = adv.float().mean().item()
                theta_loss = self.latentSp_ctrl.compute_theta_loss(adv_mean)
                if theta_loss is not None and torch.isfinite(theta_loss).item():
                    # theta_loss is on the same device as theta_low_param
                    # (now CUDA after the .to() in main); broadcast to loss
                    # device just in case.
                    loss = loss + self.theta_low_weight * theta_loss.to(loss.device)
                    mode = "train" if model.training else "eval"
                    self._metrics[mode].setdefault("theta_low_loss", []).append(
                        self.accelerator.gather_for_metrics(theta_loss.detach()).mean().item()
                    )
                    self._metrics[mode].setdefault("theta_low_param", []).append(
                        self.latentSp_ctrl.theta_low_param.detach().item()
                    )
                    self._metrics[mode].setdefault("theta_low_eff", []).append(
                        self.latentSp_ctrl.theta_low
                    )
                    self._metrics[mode].setdefault("n_latent_decisions", []).append(
                        float(sum(1 for _, d in decisions if d))
                    )

        return loss


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

    thinking_gate = ThinkingResidualGate(
        hidden_size = hidden_size,
        use_ot_dist = manifold_gate.mode != "onthefly",
        r_min       = 0.5,
        r_max       = 0.99,
    ).to("cuda")

    dna_injector = DNAHiddenInjector(
        hidden_size = hidden_size,
        r_min       = 0.7,
        r_max       = 0.99,
    ).to("cuda")

    # ── LatentSpControllerOptB (learnable theta_low) ──────────────────────────
    _use_warmup = (
        script_args.latentSp_warmup_steps > 0 or script_args.latentSp_ramp_steps > 0
    )
    latentSp_ctrl = LatentSpControllerOptB(
        theta_low_init  = script_args.latentSp_theta_low,
        theta_high      = script_args.latentSp_theta_high,
        max_consecutive = script_args.latentSp_max_consec,
        alpha           = script_args.theta_low_alpha,
    ).to("cuda")  # theta_low_param must live on the same device as the GRPO loss
    # Scale starts at 0 when warmup is requested (same effective behaviour as w9)
    if _use_warmup:
        latentSp_ctrl._warmup_scale = 0.0
        print(
            f"[OptB] theta_low_param={latentSp_ctrl.theta_low_param.item():.3f}  "
            f"warmup={script_args.latentSp_warmup_steps} ramp={script_args.latentSp_ramp_steps} steps"
        )
    else:
        latentSp_ctrl._warmup_scale = 1.0

    print(
        f"[OptB] ThinkingResidualGate: hidden={hidden_size}  "
        f"params={sum(p.numel() for p in thinking_gate.parameters()):,}"
    )
    print(
        f"[OptB] DNAHiddenInjector:    hidden={hidden_size}  "
        f"params={sum(p.numel() for p in dna_injector.parameters()):,}"
    )
    print(
        f"[OptB] theta_low_param init={script_args.latentSp_theta_low}  "
        f"alpha={script_args.theta_low_alpha}  "
        f"theta_low_lr={script_args.theta_low_lr}  "
        f"theta_low_weight={script_args.theta_low_weight}"
    )

    # ── Load gate/injector checkpoints ────────────────────────────────────────
    gate_ckpt_path = getattr(script_args, "gate_ckpt", None)
    inj_ckpt_path  = getattr(script_args, "injector_ckpt", None)
    if gate_ckpt_path and os.path.exists(gate_ckpt_path):
        _m, _u = thinking_gate.load_state_dict(
            torch.load(gate_ckpt_path, map_location="cpu"), strict=False
        )
        if _m:
            print(f"[OptB] thinking_gate missing keys (fresh init): {_m}")
        if _u:
            print(f"[OptB] thinking_gate unexpected keys (ignored): {_u}")
        print(f"[OptB] Loaded thinking_gate ← {gate_ckpt_path}")
    elif gate_ckpt_path:
        print(f"[OptB] WARNING: gate_ckpt not found: {gate_ckpt_path} — fresh init")
    if inj_ckpt_path and os.path.exists(inj_ckpt_path):
        dna_injector.load_state_dict(
            torch.load(inj_ckpt_path, map_location="cpu"), strict=False
        )
        print(f"[OptB] Loaded dna_injector  ← {inj_ckpt_path}")
    elif inj_ckpt_path:
        print(f"[OptB] WARNING: injector_ckpt not found: {inj_ckpt_path} — fresh init")

    # ── Vocab / latent token setup ────────────────────────────────────────────
    tokenizer = model.processor.tokenizer
    _all_latent = [_LATENT_START_TOKEN, _LATENT_END_TOKEN, _LATENT_PAD_TOKEN]
    _missing = [t for t in _all_latent
                if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if _missing:
        tokenizer.add_special_tokens({"additional_special_tokens": _missing})
        model.text_model.resize_token_embeddings(len(tokenizer))
        print(f"[OptB] Pre-load: added {_missing} → vocab={len(tokenizer)}")
    else:
        print(f"[OptB] Pre-load: latent tokens already in vocab (size={len(tokenizer)})")

    _load_sft_checkpoint(model, model_args, merge_lora=True)
    _prep_for_training(model, model_args)

    if hasattr(model, "gate_net") and model.gate_net is not None:
        model.gate_net.requires_grad_(False)

    latent_start_id = tokenizer.convert_tokens_to_ids(_LATENT_START_TOKEN)
    latent_end_id   = tokenizer.convert_tokens_to_ids(_LATENT_END_TOKEN)
    print(f"[OptB] Latent tokens: {_LATENT_START_TOKEN}={latent_start_id}  "
          f"{_LATENT_END_TOKEN}={latent_end_id}")

    # Patch model: generation loop, forward, gate loss — all from w9 (duck-typed)
    patch_model_for_dual_mode_w9(
        model, thinking_gate, dna_injector, latentSp_ctrl,
        latent_start_id, latent_end_id,
        max_new_tokens = script_args.max_think_tokens + 200,
        lookahead_k    = script_args.latent_lookahead_k,
    )
    model._grpo_force_full_trajectory = False

    # Gate factor init
    _gate_ckpt_loaded = gate_ckpt_path and os.path.isfile(str(gate_ckpt_path))
    if script_args.gate_warmup_steps == 0 and script_args.gate_ramp_steps == 0:
        model._gate_warmup_factor     = MAX_GATE_FACTOR
        model._gate_warmup_min_factor = MAX_GATE_FACTOR
        print(f"[OptB] gate active from step 0 (factor={MAX_GATE_FACTOR})")
    elif _gate_ckpt_loaded:
        model._gate_warmup_factor     = MAX_GATE_FACTOR
        model._gate_warmup_min_factor = MAX_GATE_FACTOR * 0.5
    else:
        model._gate_warmup_factor     = 0.0
        model._gate_warmup_min_factor = 0.0
    model._max_gate_factor = MAX_GATE_FACTOR
    model = model.to(training_args.device)

    if getattr(model_args, "dna_cache", None):
        _apply_dna_cache(model, model_args.dna_cache)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[OptB] Trainable (model):        {trainable:,} / {total:,}")
    print(f"[OptB] Trainable (gate):         "
          f"{sum(p.numel() for p in thinking_gate.parameters()):,}")
    print(f"[OptB] Trainable (injector):     "
          f"{sum(p.numel() for p in dna_injector.parameters()):,}")
    print(f"[OptB] Trainable (theta_low):    1  (nn.Parameter)")

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
    print(f"[OptB] Reward functions: {script_args.reward_funcs}")

    trainer = ThinkingResidualGRPOTrainer_OptB(
        # OptB-specific
        latentSp_ctrl    = latentSp_ctrl,
        theta_low_lr     = script_args.theta_low_lr,
        theta_low_weight = script_args.theta_low_weight,
        # Base trainer args (from ThinkingResidualGRPOTrainer)
        thinking_gate    = thinking_gate,
        manifold_gate    = manifold_gate,
        manifold_weight  = getattr(script_args, "manifold_weight", 0.0),
        max_eval_samples = getattr(script_args, "max_eval_samples", None),
        # Standard GRPOTrainer args
        model            = model,
        reward_funcs     = reward_funcs,
        args             = training_args,
        dna_module       = NucleotideDNAModule(),
        train_dataset    = data["train"],
        eval_dataset     = data["val"] if training_args.eval_strategy != "no" else None,
        peft_config      = None,
        callbacks        = [
            SaveWithPyTorchCallback(),
            SaveGateInjectorThetaCallback(thinking_gate, dna_injector, latentSp_ctrl),
            GateWarmupCallback(
                model,
                warmup_steps = script_args.gate_warmup_steps,
                ramp_steps   = script_args.gate_ramp_steps,
            ),
            LatentSpWarmupCallbackOptB(
                latentSp_ctrl,
                warmup_steps = script_args.latentSp_warmup_steps,
                ramp_steps   = script_args.latentSp_ramp_steps,
            ),
            PreferLatestOnTieCallback(
                metric_name      = training_args.metric_for_best_model,
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
        print(f"[OptB] Auto-resume: {resume}")

    if resume and isinstance(resume, str):
        for fname, obj, key in [
            ("thinking_gate.pt",  thinking_gate,  None),
            ("dna_injector.pt",   dna_injector,   None),
        ]:
            pt = os.path.join(resume, fname)
            if os.path.exists(pt):
                obj.load_state_dict(torch.load(pt, map_location="cpu"))
                print(f"[OptB] Loaded {fname} ← {pt}")
        theta_pt = os.path.join(resume, "theta_low.pt")
        if os.path.exists(theta_pt):
            saved = torch.load(theta_pt, map_location="cpu")
            with torch.no_grad():
                latentSp_ctrl.theta_low_param.copy_(
                    saved["theta_low_param"].to(latentSp_ctrl.theta_low_param.device)
                )
            print(f"[OptB] Loaded theta_low_param={latentSp_ctrl.theta_low_param.item():.4f} ← {theta_pt}")
        else:
            print(f"[OptB] No theta_low.pt in {resume} — keeping init value")

    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
    os.environ.setdefault("WANDB_PROJECT", "dna-grpo-optB")

    parser = TrlParser((GRPOScriptArgumentsOptB, DNALLMGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.save_safetensors = False
    training_args.vllm_server_base_url = os.environ.get("VLLM_BASE_URL")

    main(script_args, training_args, model_args)
