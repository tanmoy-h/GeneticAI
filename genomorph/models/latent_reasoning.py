"""
Latent Reasoning Mixin for DNALLMModel.

Supports two reasoning modes:
  - Coconut (fixed steps): runs exactly num_latent_steps recurrence steps.
  - HRPO gate (adaptive): runs up to max_latent_steps; a learned GateNet decides
    when to stop. gate_net is trained via L_gate = gate_reg_weight * mean(gate_values).

Week 4 hook: GateNet.forward() accepts an optional u_dna conditioning vector so
DNA embeddings can modulate reasoning depth.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any


class GateNet(nn.Module):
    """
    Small MLP that outputs a scalar gate value g in [0, 1] at each latent step.
    g > gate_threshold  → continue reasoning
    g < gate_threshold  → stop

    Args:
        hidden_size:  LLM hidden dimension (input).
        gate_hidden:  Internal MLP width (default 128).
        dna_size:     If > 0, concatenates a u_dna vector for DNA conditioning
                      (used in Week 4). Set to 0 to disable.
    """

    def __init__(self, hidden_size: int, gate_hidden: int = 128, dna_size: int = 0):
        super().__init__()
        self.dna_size = dna_size
        in_size = hidden_size + dna_size
        self.net = nn.Sequential(
            nn.Linear(in_size, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        # Bias output layer to keep gate closed at init: sigmoid(-5) ≈ 0.007 << threshold.
        # Gate learns to open only when GRPO reward justifies the extra latent steps.
        nn.init.constant_(self.net[2].bias, -2.0)

    def forward(
        self,
        h: torch.Tensor,
        u_dna: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h:     [batch, hidden_size]  last hidden state of the LLM.
            u_dna: [batch, dna_size]     DNA conditioning vector (Week 4).
                   If dna_size > 0 but u_dna is None (LLM-only batch),
                   the DNA slot is zero-padded so the layer dims still match.
        Returns:
            g: [batch]  gate values in [0, 1].
        """
        if self.dna_size > 0:
            if u_dna is not None:
                h = torch.cat([h, u_dna], dim=-1)
            else:
                zeros = torch.zeros(h.shape[0], self.dna_size,
                                    device=h.device, dtype=h.dtype)
                h = torch.cat([h, zeros], dim=-1)
        return self.net(h).squeeze(-1)  # [batch]


class LatentReasoningMixin:
    """
    Mixin that adds latent reasoning to DNALLMModel.

    Two modes:
      generate_with_latent_reasoning() — Coconut fixed-step loop (backward compat).
      generate_with_hrpo_gate()        — HRPO adaptive gate loop (Week 3+).

    After generate_with_hrpo_gate(), call compute_gate_loss() to get the
    differentiable L_gate term for the optimizer.
    """

    # ------------------------------------------------------------------ #
    # Coconut fixed-step loop (unchanged)                                 #
    # ------------------------------------------------------------------ #

    def generate_with_latent_reasoning(
        self,
        input_ids: torch.Tensor,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        num_latent_steps: int = 4,
        **gen_kwargs,
    ) -> torch.Tensor:
        """
        Generate answer tokens after a fixed number of latent reasoning steps.
        """
        inputs_embeds, attention_mask = self.get_prompt_embeddings(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map,
        )
        inputs_embeds = inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        past_key_values = None
        current_embeds = inputs_embeds
        current_mask = attention_mask

        for step in range(num_latent_steps):
            with torch.no_grad():
                out = self.text_model(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
            past_key_values = out.past_key_values
            current_embeds = out.hidden_states[-1][:, -1:, :]
            extra = torch.ones(
                current_mask.size(0), 1,
                dtype=current_mask.dtype, device=current_mask.device,
            )
            current_mask = torch.cat([current_mask, extra], dim=1)

        gen_kwargs.pop("disable_compile", None)
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
            else:
                past_len = past_key_values[0][0].shape[2]
            gen_kwargs["cache_position"] = torch.arange(
                past_len, past_len + current_embeds.shape[1],
                device=current_embeds.device,
            )
        with torch.no_grad():
            answer_ids = self.text_model.generate(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                **gen_kwargs,
            )
        return answer_ids

    # ------------------------------------------------------------------ #
    # HRPO adaptive gate loop (Week 3)                                    #
    # ------------------------------------------------------------------ #

    def generate_with_hrpo_gate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        max_latent_steps: int = 8,
        min_latent_steps: int = 1,
        gate_threshold: float = 0.5,
        u_dna: Optional[torch.Tensor] = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """
        Generate answer tokens using an adaptive HRPO gate loop.

        At each step the gate_net produces g in [0,1].  The loop continues
        while the batch-mean gate value exceeds gate_threshold.  Hidden states
        at each step are stored on self._gate_hidden_states so that
        compute_gate_loss() can recompute gate values WITH gradients cheaply.

        Args:
            input_ids:        [batch, seq_len] prompt token IDs.
            attention_mask:   [batch, seq_len] mask (0 for padding, 1 for real tokens).
                              Pass the collator mask so left-padding is correctly ignored.
            dna_tokenized:    Tokenized DNA sequences (optional).
            batch_idx_map:    DNA-to-batch mapping (optional).
            max_latent_steps: Safety ceiling for the loop.
            gate_threshold:   Stop when mean(g) < gate_threshold.
            u_dna:            [batch, dna_size] DNA conditioning vector (Week 4).
            **gen_kwargs:     Forwarded to text_model.generate().

        Returns:
            answer_ids: generated token IDs.
        """
        assert hasattr(self, "gate_net"), (
            "gate_net not found. Pass use_hrpo_gate=True when constructing DNALLMModel."
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # ── Step 1: build prompt embeddings ──────────────────────────────────
        inputs_embeds, attention_mask = self.get_prompt_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map,
        )
        inputs_embeds  = inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        # ── Step 2: adaptive gate loop ────────────────────────────────────────
        past_key_values = None
        current_embeds  = inputs_embeds
        current_mask    = attention_mask

        # Keep a clean copy of the original prompt embeddings so we can fall
        # back to standard generation if the gate never opens.
        original_embeds = inputs_embeds
        original_mask   = attention_mask

        # Store raw hidden states (detached) for cheap gate loss recomputation
        gate_hidden_states: List[torch.Tensor] = []
        gate_values_nograd: List[torch.Tensor] = []
        steps_taken = 0
        latent_steps_used = 0  # counts steps where gate was open (g >= threshold)

        for step in range(max_latent_steps):
            with torch.no_grad():
                out = self.text_model(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )

            past_key_values = out.past_key_values

            # last hidden state of last token: [batch, hidden_size]
            last_h = out.hidden_states[-1][:, -1, :].detach()
            gate_hidden_states.append(last_h)

            with torch.no_grad():
                g = self.gate_net(last_h, u_dna=u_dna)  # [batch]
            gate_values_nograd.append(g)

            steps_taken += 1

            # Always take the step if we haven't met the minimum yet;
            # only check the gate threshold after min_latent_steps are done.
            if step >= min_latent_steps and g.mean().item() < gate_threshold:
                break

            # Gate open: feed this hidden state as the next "thought token".
            latent_steps_used += 1
            current_embeds = out.hidden_states[-1][:, -1:, :]
            extra = torch.ones(
                current_mask.size(0), 1,
                dtype=current_mask.dtype, device=current_mask.device,
            )
            current_mask = torch.cat([current_mask, extra], dim=1)

        # ── Cache for compute_gate_loss() ─────────────────────────────────────
        self._gate_hidden_states  = gate_hidden_states   # List[[batch, H]]
        self._gate_u_dna          = u_dna
        self._gate_steps_taken    = steps_taken       # gate checks (includes closing step)
        self._gate_latent_steps   = latent_steps_used # actual thought-tokens injected

        # ── Step 3: generate from final latent state ──────────────────────────
        gen_kwargs.pop("disable_compile", None)

        # If the gate never opened (collapsed to 0), fall back to the original
        # prompt embeddings to avoid feeding corrupted hidden states to generate().
        if latent_steps_used == 0:
            past_key_values = None
            current_embeds  = original_embeds
            current_mask    = original_mask

        # Transformers 4.51+ (Qwen3) requires explicit cache_position when
        # generate() is called with inputs_embeds + past_key_values.
        # Without it, cache_position is derived as empty → vmap mask crash.
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()   # DynamicCache
            else:
                past_len = past_key_values[0][0].shape[2]     # legacy tuple
            gen_kwargs["cache_position"] = torch.arange(
                past_len, past_len + current_embeds.shape[1],
                device=current_embeds.device,
            )

        with torch.no_grad():
            answer_ids = self.text_model.generate(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                **gen_kwargs,
            )

        return answer_ids

    def compute_gate_loss(self, gate_reg_weight: float = 0.01) -> torch.Tensor:
        """
        Recompute gate values WITH gradients and return the gate regularization loss.

        Calls gate_net exactly ONCE (batching all steps) so DDP's per-accumulation
        grad hook fires only once per parameter — avoids 'marked ready twice' errors.

        Returns:
            Scalar loss tensor (with grad).
        """
        if not hasattr(self, "_gate_hidden_states") or not self._gate_hidden_states:
            return sum(p.sum() for p in self.gate_net.parameters()) * 0.0

        # Stack: [batch, steps, H] → flatten to [batch*steps, H] for a single forward
        all_h = torch.stack(self._gate_hidden_states, dim=1)  # [B, S, H]
        B, S, H = all_h.shape
        all_h_flat = all_h.reshape(B * S, H)

        u_dna_exp = None
        if self._gate_u_dna is not None:
            u_dna_exp = self._gate_u_dna.repeat_interleave(S, dim=0)

        g_flat = self.gate_net(all_h_flat, u_dna=u_dna_exp)  # [B*S] — single call
        gate_tensor = g_flat.reshape(B, S)                   # [B, steps]
        return gate_reg_weight * gate_tensor.mean()
