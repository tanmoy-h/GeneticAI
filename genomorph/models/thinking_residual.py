"""
HRPO-style Thinking Residual Gate for GenoMorph.

Replaces the Coconut hard-recurrence latent loop with a single gated residual
connection applied to prompt token embeddings before the LLM forward pass.

The gate blends u_dna (the DNA summary vector) into every prompt token embedding
via a GRU-style rotation:

    a_t  = Lambda(sigmoid(gate_r(embeds)))   # per-dim decay in [r_min, r_max]
    i_t  = sigmoid(gate_i(embeds))           # input gate
    out  = a_t * embeds + sqrt(1−a_t²+ε) · i_t · residual

where residual = u_dna broadcast to [B, T, H].

OT-distance adaptivity
----------------------
ot_dist [B] is the squared cosine distance between the current DNA summary
(u_dna) and its OT-optimal target on the answer manifold (from Stage 2).

  large ot_dist  → sample is far from the manifold → ot_scale drives a_t
                   toward r_min (more blending, more DNA signal injected).
  small ot_dist  → sample is close → a_t stays near r_max (near-identity).

A single scalar ot_scale (init=0) controls the effect:
    ot_factor = tanh(ot_scale · ot_dist)    ∈ [0,1)
    a_t       = a_t − ot_factor · (a_t − r_min)

ot_scale=0 at init → no OT effect for any sample.  As GRPO trains,
ot_scale grows positive and the adaptivity activates.

In onthefly mode (no Stage 2) ot_dist is None and the gate degrades gracefully
to a plain thinking residual with no manifold conditioning.

Differential learning rates (from HRPO patch.py):
  gate_r / gate_i / ot_scale : 20×  base LR  (default 6e-5 when base = 3e-6)
  Lambda                     : 200× base LR  (default 6e-4 when base = 3e-6)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ThinkingResidualLambda(nn.Module):
    """
    Learnable per-dimension decay coefficient a_t ∈ [r_min, r_max].

    Maps the reset gate r_t (already in [0,1] after sigmoid) to a decay
    value via a learned element-wise bias:

        a_t = r_min + (r_max - r_min) * sigmoid(weight + r_t)

    Initialised so a_t ≈ r_max (≈ 0.99) → gate starts as near-identity,
    letting LoRA converge first before the thinking residual activates.
    """

    def __init__(self, hidden_size: int, r_min: float = 0.5, r_max: float = 0.99):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        # One scalar per hidden dim; large positive → sigmoid ≈ 1 → a_t ≈ r_max
        self.weight = nn.Parameter(torch.full((hidden_size,), 4.0))

    def forward(self, r_t: torch.Tensor) -> torch.Tensor:
        # r_t: [..., H]
        return self.r_min + (self.r_max - self.r_min) * torch.sigmoid(
            self.weight + r_t
        )

    def reset_lambda_parameters(self, r_min: float, r_max: float) -> None:
        self.r_min = r_min
        self.r_max = r_max


class ThinkingResidualGate(nn.Module):
    """
    HRPO-style gated residual with optional OT-distance conditioning.

    Args:
        hidden_size: LLM hidden dimension.
        use_ot_dist: If True, adds a scalar ot_scale that reduces a_t for
                     samples far from the answer manifold (default True).
        r_min:       Minimum decay coefficient (default 0.5).
        r_max:       Maximum decay coefficient (default 0.99).
    """

    def __init__(
        self,
        hidden_size: int,
        use_ot_dist: bool = True,
        r_min: float = 0.5,
        r_max: float = 0.99,
    ):
        super().__init__()
        self.use_ot_dist = use_ot_dist
        self.gate_r     = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_i     = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lambda_net = ThinkingResidualLambda(hidden_size, r_min, r_max)

        # OT-distance conditioning: a single learnable scalar.
        #
        #   ot_factor = tanh(ot_scale · ot_dist)   ∈ [0, 1)
        #   a_t       = a_t − ot_factor · (a_t − r_min)
        #
        # ot_scale = 0 at init → ot_factor = 0 for all samples → no OT effect.
        # As ot_scale learns to be positive:
        #   large ot_dist → large ot_factor → more reduction of a_t → more blending.
        #   small ot_dist → small ot_factor → a_t unchanged → near-identity.
        # Monotone in ot_dist by construction.
        if use_ot_dist:
            self.ot_scale = nn.Parameter(torch.zeros(1))

        # gate_r / gate_i: tiny non-zero init so gradients flow from step 0.
        # a_t ≈ r_max still (near-identity) but the gate can learn immediately.
        nn.init.normal_(self.gate_r.weight, std=0.01)
        nn.init.normal_(self.gate_i.weight, std=0.01)

    def forward(
        self,
        embeds:   torch.Tensor,                  # [B, T, H]
        residual: torch.Tensor,                  # [B, T, H]  u_dna broadcast
        ot_dist:  Optional[torch.Tensor] = None, # [B]  squared manifold distance
        eps:      float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out:  [B, T, H]  modified embeddings
            a_t:  [B, T, H]  decay coefficients (for gate loss / logging)
        """
        r_t = torch.sigmoid(self.gate_r(embeds))          # [B, T, H]
        i_t = torch.sigmoid(self.gate_i(embeds))          # [B, T, H]
        a_t = self.lambda_net(r_t)                      # [B, T, H]

        # OT direct modulation: reduce a_t for samples far from the manifold.
        # tanh(0)=0 at init → no effect; grows as ot_scale learns to be positive.
        if self.use_ot_dist and ot_dist is not None:
            ot_factor = torch.tanh(
                self.ot_scale * ot_dist.float().abs().to(a_t.device)
            ).view(-1, 1, 1)                            # [B, 1, 1] ∈ [0, 1)
            a_t = a_t - ot_factor * (a_t - self.lambda_net.r_min)

        out = (
            a_t * embeds
            + torch.sqrt(1.0 - a_t.pow(2) + eps) * (i_t * residual.to(embeds))
        )
        return out, a_t

    def reset_lambda_parameters(self, r_min: float, r_max: float) -> None:
        self.lambda_net.reset_lambda_parameters(r_min, r_max)


def make_thinking_residual_param_groups(
    model: nn.Module,
    thinking_gate: ThinkingResidualGate,
    base_lr: float,
    lr_multiplier_gate: float = 20.0,
    lr_multiplier_lambda: float = 200.0,
    weight_decay: float = 0.01,
):
    """
    Build AdamW param groups with differential LRs (HRPO patch.py style).

    gate_r / gate_i : base_lr * lr_multiplier_gate   (default 20×)
    lambda_net      : base_lr * lr_multiplier_lambda  (default 200×)
    everything else : base_lr

    Returns a list of param-group dicts ready for torch.optim.AdamW.
    """
    gate_ri_params_raw = (
        list(thinking_gate.gate_r.parameters())
        + list(thinking_gate.gate_i.parameters())
        + ([thinking_gate.ot_scale] if thinking_gate.use_ot_dist else [])
    )
    gate_ri_ids = {id(p) for p in gate_ri_params_raw}
    lambda_ids  = {id(p) for p in thinking_gate.lambda_net.parameters()}
    special_ids = gate_ri_ids | lambda_ids

    # Separate decay / no-decay for standard params (mirrors HRPO patch.py)
    standard_decay, standard_nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad or id(p) in special_ids:
            continue
        if p.ndim >= 2:
            standard_decay.append(p)
        else:
            standard_nodecay.append(p)

    gate_ri_params = [p for p in gate_ri_params_raw if p.requires_grad]
    lambda_params = [
        p for p in thinking_gate.lambda_net.parameters() if p.requires_grad
    ]

    param_groups = []
    if standard_decay:
        param_groups.append(
            {"params": standard_decay, "lr": base_lr, "weight_decay": weight_decay}
        )
    if standard_nodecay:
        param_groups.append(
            {"params": standard_nodecay, "lr": base_lr, "weight_decay": 0.0}
        )
    if gate_ri_params:
        param_groups.append(
            {
                "params": gate_ri_params,
                "lr": base_lr * lr_multiplier_gate,
                "weight_decay": 0.0,
            }
        )
    if lambda_params:
        param_groups.append(
            {
                "params": lambda_params,
                "lr": base_lr * lr_multiplier_lambda,
                "weight_decay": 0.0,
            }
        )
    return param_groups
