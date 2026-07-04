"""
Stage 3 Week 6: GRPO + adaptive HRPO gate guided by OT distance.

Merged from adaptive_latent_grpo.py (Week 5 shared helpers) and
adaptive_latent_grpo_w6.py (Week 6 OT-adaptive gate).

Key Week 6 additions over Week 5:

1. ManifoldGateW6 — three modes (global / nn / onthefly) from stage2_dir contents.
2. GateNetW6 — gate receives (h, u_dna, ot_dist, delta_ot, step_idx).
3. Adaptive gate loop — no fixed gate_threshold; gate LEARNS when to stop.
4. Monkey-patching — replaces model.generate_with_hrpo_gate + compute_gate_loss.

Usage:
  # With Stage 2 Monge map (global mode)
  accelerate launch adaptive_latent_grpo_w6.py --stage2_dir stage2_output_w6 ...

  # No Stage 2 (onthefly mode)
  accelerate launch adaptive_latent_grpo_w6.py ...
"""

import os
import pathlib
import types
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import ModelConfig, ScriptArguments, TrlParser

from genomorph.dataset.kegg import format_kegg_for_dna_llm, load_kegg_from_anon_csv
from genomorph.dataset.utils import truncate_dna
from genomorph.dna_modules import NucleotideDNAModule
from genomorph.models.dna_llm import DNALLMModel, get_target_modules
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from genomorph.trainer import DNALLMGRPOConfig, DNALLMGRPOTrainer

register_evo2_tokenizer()


# ── ManifoldGate (Week 5 — Stage 2 integration) ───────────────────────────────

class ManifoldGate:
    """
    Encodes the precomputed answer manifold from Stage 2 (hiref_offline.py).

    global mode   (stage2_dir provided):
        For each batch DNA embedding u_norm [B, H], finds the nearest Stage 2
        DNA embedding, looks up its Monge-mapped answer embedding, and returns
        those targets for computing L_manifold.

    onthefly mode (no stage2_dir):
        No precomputed manifold. manifold_loss() returns None.
    """

    def __init__(self, stage2_dir: Optional[str]):
        if stage2_dir and os.path.isdir(stage2_dir):
            self.answer_embs = torch.tensor(
                np.load(os.path.join(stage2_dir, "answer_embeddings.npy")),
                dtype=torch.float32,
            )
            self.monge_map = np.load(os.path.join(stage2_dir, "monge_map.npy"))
            self.dna_embs  = torch.tensor(
                np.load(os.path.join(stage2_dir, "dna_embeddings.npy")),
                dtype=torch.float32,
            )
            self.mode = "global"
            print(f"[ManifoldGate] global mode — {len(self.answer_embs)} manifold points loaded")
        else:
            self.answer_embs = None
            self.dna_embs    = None
            self.monge_map   = None
            self.mode        = "onthefly"
            if stage2_dir:
                print(f"[ManifoldGate] WARNING: stage2_dir={stage2_dir!r} not found; "
                      "falling back to on-the-fly mode (no manifold loss)")
            else:
                print("[ManifoldGate] on-the-fly mode — no stage2_dir; manifold loss disabled")

    @torch.no_grad()
    def get_targets(self, u_norm: torch.Tensor) -> Optional[torch.Tensor]:
        if self.mode != "global":
            return None
        device      = u_norm.device
        dna_embs    = self.dna_embs.to(device)
        answer_embs = self.answer_embs.to(device)
        sims        = u_norm @ dna_embs.T
        nearest     = sims.argmax(dim=1)
        return answer_embs[[int(self.monge_map[idx.item()]) for idx in nearest]]

    def manifold_loss(self, u_norm: torch.Tensor) -> Optional[torch.Tensor]:
        targets = self.get_targets(u_norm.detach())
        if targets is None:
            return None
        t_norm = F.normalize(targets.float().to(u_norm.device), dim=-1)
        return ((u_norm.float() - t_norm) ** 2).sum(dim=-1).mean()


# ── ManifoldGRPOTrainer (Week 5) ──────────────────────────────────────────────

class ManifoldGRPOTrainer(DNALLMGRPOTrainer):
    """DNALLMGRPOTrainer extended with L_manifold (Week 5)."""

    def __init__(self, gate: ManifoldGate, manifold_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gate            = gate
        self.manifold_weight = manifold_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        if (self.gate.mode != "global"
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

        u_dna_list = []
        for i in range(batch_size):
            parts = []
            for slot in (i, i + batch_size):
                if slot < len(batch_dna_embeds) and batch_dna_embeds[slot].shape[0] > 0:
                    parts.append(batch_dna_embeds[slot])
            u_dna_list.append(
                torch.cat(parts, dim=0).mean(dim=0) if parts
                else torch.zeros(unwrapped.text_hidden_size, device=_device, dtype=_dtype)
            )
        u_dna  = torch.stack(u_dna_list)
        u_norm = F.normalize(u_dna.float(), dim=-1)

        l_manifold = self.gate.manifold_loss(u_norm)
        if l_manifold is None:
            return loss

        loss = loss + self.manifold_weight * l_manifold
        mode = "train" if model.training else "eval"
        self._metrics[mode]["manifold_loss"].append(
            self.accelerator.gather_for_metrics(l_manifold.detach()).mean().item()
        )
        return loss


# ── Checkpoint saving ─────────────────────────────────────────────────────────

class SaveWithPyTorchCallback(TrainerCallback):
    """Save checkpoints with torch.save instead of safetensors."""

    def on_save(self, args, state, control, **kwargs):
        folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        os.makedirs(folder, exist_ok=True)
        model    = kwargs.get("model")
        unwrapped = model.module if hasattr(model, "module") else model
        torch.save(unwrapped.state_dict(), os.path.join(folder, "pytorch_model.bin"))
        if hasattr(unwrapped, "text_model"):
            cfg = getattr(unwrapped.text_model, "config", None) or getattr(
                getattr(unwrapped.text_model, "base_model", None), "config", None
            )
            if cfg is not None:
                cfg.save_pretrained(folder)
        print(f"Saved checkpoint → {folder}")
        control.should_save = False
        return control


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_kegg_dataset(
    kegg_csv:              Optional[str] = None,
    dataset_name:          str = "wanglab/kegg",
    truncate_dna_per_side: int = 0,
) -> Dataset:
    if kegg_csv:
        data = load_kegg_from_anon_csv(kegg_csv)
    else:
        data = load_dataset(dataset_name, "default")

    if truncate_dna_per_side > 0:
        data = data.map(
            truncate_dna, fn_kwargs={"truncate_dna_per_side": truncate_dna_per_side}
        )
    data = data.map(format_kegg_for_dna_llm, fn_kwargs={"is_sft": False})
    return data


# ── Argument dataclasses ──────────────────────────────────────────────────────

@dataclass
class GRPOModelConfig(ModelConfig):
    text_model_name: str = field(default="Qwen/Qwen3-1.7B", metadata={"help": "LLM model name or path."})
    dna_model_name: Optional[str] = field(default=None, metadata={"help": "DNA encoder model name/path."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HuggingFace model cache directory."})
    max_length_text: int = field(default=6000, metadata={"help": "Max text tokens."})
    max_length_dna:  int = field(default=2048, metadata={"help": "Max DNA tokens."})
    sft_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to SFT checkpoint."})
    lora_r:       int   = field(default=16,  metadata={"help": "LoRA rank."})
    lora_alpha:   int   = field(default=32,  metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout."})
    dna_model_finetune:       bool = field(default=False, metadata={"help": "Fine-tune DNA encoder."})
    dna_projection_finetune:  bool = field(default=True,  metadata={"help": "Fine-tune DNA projection."})
    peft_ckpt: bool = field(default=False, metadata={"help": "Whether sft_checkpoint is a PEFT directory."})
    dna_is_evo2: bool = field(default=False, metadata={"help": "Use Evo2 as DNA encoder."})
    dna_embedding_layer: Optional[str] = field(default=None, metadata={"help": "Evo2 layer to extract embeddings from."})
    truncate_dna_per_side: int = field(default=1024, metadata={"help": "Truncate DNA by this many bp per side."})
    use_cross_attention: bool = field(default=False, metadata={"help": "Use CrossAttentionFusion."})
    dna_cache: Optional[str] = field(default=None, metadata={"help": "Path to precomputed Evo2 embeddings (.pt)."})


@dataclass
class GRPOScriptArguments(ScriptArguments):
    dataset_name: str = field(default="wanglab/kegg", metadata={"help": "HF dataset name."})
    kegg_csv: Optional[str] = field(default=None, metadata={"help": "Local kegg CSV path."})
    full_ckpt: Optional[str] = field(default=None, metadata={"help": "Path to a full pytorch_model.bin checkpoint."})
    reward_funcs: List[str] = field(
        default_factory=lambda: ["xmlcount", "soft_format", "strict_format", "concise", "correctness"],
        metadata={"help": "Reward functions to use."},
    )
    max_think_tokens: int   = field(default=600,   metadata={"help": "Max tokens inside a <think> block."})
    gate_warmup_steps: int  = field(default=100,   metadata={"help": "Steps with gate bypassed."})
    gate_ramp_steps:   int  = field(default=200,   metadata={"help": "Steps to ramp gate 0→1."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Max val samples per evaluate() call."})
    stage2_dir: Optional[str] = field(default=None, metadata={"help": "Path to Stage 2 output directory."})
    manifold_weight: float    = field(default=0.1,  metadata={"help": "Weight of L_manifold."})
    gate_sft_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to gate_net weights (.pt) from Stage 2.5."})


reward_funcs_registry = {
    "xmlcount":           NucleotideDNAModule.xmlcount_reward_func,
    "soft_format":        NucleotideDNAModule.soft_format_reward_func,
    "correctness":        NucleotideDNAModule.correctness_reward_func,
    "completion_quality": NucleotideDNAModule.completion_quality_reward_func,
    "reasoning_quality":  NucleotideDNAModule.reasoning_quality_reward_func,
    "latent_format":      NucleotideDNAModule.latent_format_reward_func,
    # Legacy (kept for checkpoint compat)
    "strict_format":      NucleotideDNAModule.strict_format_reward_func,
    "concise":            NucleotideDNAModule.concise_reward_func,
    "diversity":          NucleotideDNAModule.diversity_reward_func,
    "single_think_close": NucleotideDNAModule.single_think_close_reward_func,
    "non_degenerate":     NucleotideDNAModule.non_degenerate_reward_func,
}


# ── Model preparation ─────────────────────────────────────────────────────────

def _prep_for_training(model: DNALLMModel, model_args: GRPOModelConfig) -> Optional[LoraConfig]:
    """Freeze/unfreeze components and wrap text model with LoRA."""
    if model.dna_model is not None:
        dna_params = (
            model.dna_model.model.parameters()
            if model_args.dna_is_evo2
            else model.dna_model.parameters()
        )
        for p in dna_params:
            p.requires_grad = model_args.dna_model_finetune

    if model.dna_model is not None:
        for p in model.dna_projection.parameters():
            p.requires_grad = model_args.dna_projection_finetune

    if hasattr(model, "gate_net"):
        for p in model.gate_net.parameters():
            p.requires_grad = True

    if model_args.lora_r == 0:
        for p in model.text_model.parameters():
            p.requires_grad = True
        return None

    target_modules = get_target_modules(model)
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights=True,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.text_model = prepare_model_for_kbit_training(model.text_model)
    model.text_model = get_peft_model(model.text_model, lora_config)
    model.text_model.train()
    return lora_config


def _load_sft_checkpoint(
    model:      DNALLMModel,
    model_args: GRPOModelConfig,
    merge_lora: bool = False,
) -> None:
    """Load weights from an SFT checkpoint produced by train_dna_qwen.py."""
    ckpt_path = model_args.sft_checkpoint
    if ckpt_path is None:
        return

    print(f"Loading SFT checkpoint: {ckpt_path}")

    if os.path.isdir(ckpt_path) and model_args.peft_ckpt:
        model.text_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        base       = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True)
        peft_model = PeftModel.from_pretrained(base, ckpt_path, is_trainable=True)
        model.text_model = peft_model.merge_and_unload()
        print("Loaded + merged PEFT checkpoint.")

    elif os.path.isdir(ckpt_path):
        model.text_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        model.text_model     = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True)
        print("Loaded HF directory checkpoint.")

    else:
        raw = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        elif isinstance(raw, dict):
            state = raw
        else:
            raise ValueError(f"Unrecognised checkpoint format: {ckpt_path}")

        def _strip(k: str) -> str:
            for prefix in ("model.", "_forward_module.model.", "_forward_module."):
                if k.startswith(prefix):
                    return k[len(prefix):]
            return k

        state = {_strip(k): v for k, v in state.items()}

        _LORA_TAGS = ("lora_A", "lora_B", "lora_embedding", "lora_magnitude")
        peft_inner = "text_model.base_model.model.model."
        peft_outer = "text_model.base_model.model."
        if any(k.startswith(peft_inner) for k in state):
            if merge_lora:
                print("  Detected PEFT/LoRA structure — merging LoRA deltas into base weights.")
            else:
                print("  Detected PEFT/LoRA structure in checkpoint — extracting base weights.")
            remapped   = {}
            lora_a_map = {}
            lora_b_map = {}

            for k, v in state.items():
                if k.startswith(peft_inner):
                    stripped = k[len(peft_inner):]
                    new_k    = "text_model.model." + stripped
                    if any(tag in stripped for tag in _LORA_TAGS):
                        if merge_lora:
                            if "lora_A" in stripped:
                                lora_a_map[stripped.split(".lora_A.")[0]] = v
                            elif "lora_B" in stripped:
                                lora_b_map[stripped.split(".lora_B.")[0]] = v
                        continue
                    remapped[new_k.replace(".base_layer.", ".")] = v
                elif k.startswith(peft_outer):
                    new_k = "text_model." + k[len(peft_outer):]
                    if any(tag in new_k for tag in _LORA_TAGS):
                        continue
                    remapped[new_k.replace(".base_layer.", ".")] = v
                else:
                    remapped[k] = v

            if merge_lora and lora_a_map:
                lora_r     = getattr(model_args, "lora_r",     16)
                lora_alpha = getattr(model_args, "lora_alpha", 32)
                scaling    = lora_alpha / lora_r
                merged_n   = 0
                for mod_key, A in lora_a_map.items():
                    if mod_key not in lora_b_map:
                        continue
                    B        = lora_b_map[mod_key]
                    full_key = "text_model.model." + mod_key + ".weight"
                    if full_key not in remapped:
                        continue
                    remapped[full_key] = remapped[full_key] + (B.to(remapped[full_key].dtype) @ A.to(remapped[full_key].dtype)) * scaling
                    merged_n += 1
                print(f"  Merged {merged_n} LoRA adapters (α={lora_alpha}, r={lora_r}, scale={scaling:.3f}).")

            state = remapped

        missing, unexpected = model.load_state_dict(state, strict=False)
        expected_missing = {"gate_net.net.0.weight", "gate_net.net.0.bias",
                            "gate_net.net.2.weight", "gate_net.net.2.bias"}
        true_missing = [k for k in missing if k not in expected_missing]
        if true_missing:
            print(f"  Missing keys ({len(true_missing)}): {true_missing[:5]}")
        else:
            print(f"  Missing keys: only gate_net (expected — initialized fresh for GRPO)")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        print(f"Loaded .bin checkpoint ({len(state)} keys).")


# ── ManifoldGateW6 (Week 6) ───────────────────────────────────────────────────

class ManifoldGateW6:
    """
    Three modes — determined automatically from stage2_dir contents:

    global  : answer_embeddings.npy + monge_map.npy present.
    nn      : answer_embeddings.npy present, monge_map.npy absent.
    onthefly: no stage2_dir or missing embeddings.
    """

    def __init__(self, stage2_dir: Optional[str]):
        self.mode        = "onthefly"
        self.answer_embs = None
        self.dna_embs    = None
        self.monge_map   = None

        if not (stage2_dir and os.path.isdir(stage2_dir)):
            msg = (f"WARNING: stage2_dir={stage2_dir!r} not found; onthefly mode"
                   if stage2_dir else "onthefly — no stage2_dir; manifold disabled")
            print(f"[ManifoldGateW6] {msg}")
            return

        ans_path = os.path.join(stage2_dir, "answer_embeddings.npy")
        dna_path = os.path.join(stage2_dir, "dna_embeddings.npy")
        mng_path = os.path.join(stage2_dir, "monge_map.npy")

        if not (os.path.exists(ans_path) and os.path.exists(dna_path)):
            print(f"[ManifoldGateW6] embeddings missing in {stage2_dir}; onthefly mode")
            return

        self.answer_embs = torch.tensor(np.load(ans_path), dtype=torch.float32)
        self.dna_embs    = torch.tensor(np.load(dna_path),  dtype=torch.float32)

        if os.path.exists(mng_path):
            self.monge_map = np.load(mng_path)
            self.mode = "global"
            print(f"[ManifoldGateW6] global mode — {len(self.answer_embs)} points, Monge map loaded")
        else:
            self.mode = "nn"
            print(f"[ManifoldGateW6] nn mode — {len(self.answer_embs)} points, "
                  "no monge_map.npy → identity NN mapping")

    @torch.no_grad()
    def _lookup_targets(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        if self.mode == "onthefly":
            return None
        device      = query.device
        dna_embs    = self.dna_embs.to(device)
        answer_embs = self.answer_embs.to(device)
        nearest     = (query @ dna_embs.T).argmax(dim=1)
        if self.mode == "global":
            indices = [int(self.monge_map[k.item()]) for k in nearest]
        else:
            indices = [k.item() for k in nearest]
        return answer_embs[indices]

    @torch.no_grad()
    def get_ot_dist(
        self,
        h_last: torch.Tensor,
        u_dna:  Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.mode == "onthefly":
            return None
        device = h_last.device
        h_norm = F.normalize(h_last.float().to(device), dim=-1)
        if self.mode == "global" and u_dna is not None:
            u_norm  = F.normalize(u_dna.float().to(device), dim=-1)
            targets = self._lookup_targets(u_norm)
        else:
            targets = self._lookup_targets(h_norm)
        if targets is None:
            return None
        t_norm = F.normalize(targets.float().to(device), dim=-1)
        return ((h_norm - t_norm) ** 2).sum(dim=-1)

    def manifold_loss(self, u_norm: torch.Tensor) -> Optional[torch.Tensor]:
        targets = self._lookup_targets(u_norm.detach())
        if targets is None:
            return None
        t_norm = F.normalize(targets.float().to(u_norm.device), dim=-1)
        return ((u_norm.float() - t_norm) ** 2).sum(dim=-1).mean()


# ── GateNetW6 ─────────────────────────────────────────────────────────────────

class GateNetW6(nn.Module):
    """GateNet extended with OT-distance, Δot trend, and step-index inputs."""

    def __init__(
        self,
        hidden_size:  int,
        gate_hidden:  int  = 128,
        dna_size:     int  = 0,
        ot_size:      int  = 1,
        use_step_idx: bool = True,
        use_delta_ot: bool = True,
    ):
        super().__init__()
        self.dna_size     = dna_size
        self.ot_size      = ot_size
        self.use_step_idx = use_step_idx
        self.use_delta_ot = use_delta_ot
        in_size = (hidden_size + dna_size + ot_size
                   + (1 if use_step_idx else 0)
                   + (1 if use_delta_ot  else 0))
        self.net = nn.Sequential(
            nn.Linear(in_size, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.net[2].bias, -2.0)

    def forward(
        self,
        h:        torch.Tensor,
        u_dna:    Optional[torch.Tensor] = None,
        ot_dist:  Optional[torch.Tensor] = None,
        delta_ot: Optional[torch.Tensor] = None,
        step_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [h]
        if self.dna_size > 0:
            parts.append(
                u_dna if u_dna is not None
                else torch.zeros(h.shape[0], self.dna_size, device=h.device, dtype=h.dtype)
            )
        if self.ot_size > 0:
            if ot_dist is not None:
                d = ot_dist.view(h.shape[0], -1).to(h.device).to(h.dtype)
                if d.shape[-1] < self.ot_size:
                    d = F.pad(d, (0, self.ot_size - d.shape[-1]))
                elif d.shape[-1] > self.ot_size:
                    d = d[:, :self.ot_size]
                parts.append(d)
            else:
                parts.append(torch.zeros(h.shape[0], self.ot_size, device=h.device, dtype=h.dtype))
        if self.use_delta_ot:
            ddt = (delta_ot.view(h.shape[0], 1).to(h.device).to(h.dtype)
                   if delta_ot is not None
                   else torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype))
            parts.append(ddt)
        if self.use_step_idx:
            s = (step_idx.view(h.shape[0], 1).to(h.device).to(h.dtype)
                 if step_idx is not None
                 else torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype))
            parts.append(s)
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


# ── Adaptive gate helpers ─────────────────────────────────────────────────────

@torch.no_grad()
def _build_u_dna(model: DNALLMModel, dna_tokenized, batch_idx_map, batch_size, device):
    """Compute mean-pooled, normalised DNA projection vector for a batch."""
    if dna_tokenized is None:
        return None
    batch_dna_embeds = model.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size)
    _dtype = next(model.dna_projection.parameters()).dtype
    u_list = []
    for i in range(batch_size):
        parts = []
        for slot in (i, i + batch_size):
            if slot < len(batch_dna_embeds) and batch_dna_embeds[slot].shape[0] > 0:
                parts.append(batch_dna_embeds[slot])
        u_list.append(
            torch.cat(parts, dim=0).mean(dim=0) if parts
            else torch.zeros(model.text_hidden_size, device=device, dtype=_dtype)
        )
    return F.normalize(torch.stack(u_list).to(device).float(), dim=-1)


def _make_adaptive_generate(gate_net_w6: GateNetW6, manifold_gate: ManifoldGateW6):
    """Return a replacement for model.generate_with_hrpo_gate (adaptive loop)."""

    def _gen(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   Optional[torch.Tensor] = None,
        dna_tokenized     = None,
        batch_idx_map     = None,
        max_latent_steps: int   = 8,
        min_latent_steps: int   = 0,
        gate_threshold:   float = 0.5,
        u_dna_external:   Optional[torch.Tensor] = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        u_dna = u_dna_external
        if u_dna is None:
            u_dna = _build_u_dna(self, dna_tokenized, batch_idx_map,
                                  input_ids.shape[0], input_ids.device)

        inputs_embeds, attention_mask = self.get_prompt_embeddings(
            input_ids=input_ids, attention_mask=attention_mask,
            dna_tokenized=dna_tokenized, batch_idx_map=batch_idx_map,
        )
        inputs_embeds  = inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        original_embeds = inputs_embeds
        original_mask   = attention_mask
        past_key_values = None
        current_embeds  = inputs_embeds
        current_mask    = attention_mask

        gate_hidden_states: List[torch.Tensor] = []
        gate_ot_dists:      List[torch.Tensor] = []
        gate_delta_ots:     List[torch.Tensor] = []
        steps_taken        = 0
        latent_steps_used  = 0
        prev_ot_dist: Optional[torch.Tensor] = None
        is_training = getattr(self, "_grpo_force_full_trajectory", False)
        dna_alpha   = 0.1

        for step in range(max_latent_steps):
            with torch.no_grad():
                out = self.text_model(
                    inputs_embeds=current_embeds, attention_mask=current_mask,
                    past_key_values=past_key_values, use_cache=True,
                    output_hidden_states=True,
                )
            past_key_values = out.past_key_values
            last_h = out.hidden_states[-1][:, -1, :].detach()
            gate_hidden_states.append(last_h)

            ot_dist = manifold_gate.get_ot_dist(last_h, u_dna)
            gate_ot_dists.append(
                ot_dist if ot_dist is not None
                else torch.zeros(last_h.shape[0], device=last_h.device)
            )
            ot_dist_in = ot_dist.unsqueeze(-1) if ot_dist is not None else None

            if ot_dist is not None:
                delta_ot_in = ((ot_dist - prev_ot_dist).unsqueeze(-1)
                               if prev_ot_dist is not None
                               else torch.zeros(last_h.shape[0], 1,
                                                device=last_h.device, dtype=last_h.dtype))
                prev_ot_dist = ot_dist.detach()
            else:
                delta_ot_in = None
            gate_delta_ots.append(
                delta_ot_in.squeeze(-1) if delta_ot_in is not None
                else torch.zeros(last_h.shape[0], device=last_h.device)
            )

            step_t = torch.full(
                (last_h.shape[0], 1),
                step / max(max_latent_steps - 1, 1),
                device=last_h.device, dtype=last_h.dtype,
            )
            with torch.no_grad():
                g = gate_net_w6(last_h, u_dna=u_dna, ot_dist=ot_dist_in,
                                delta_ot=delta_ot_in, step_idx=step_t)

            steps_taken  += 1
            next_embeds   = out.hidden_states[-1][:, -1:, :]
            if u_dna is not None:
                next_embeds = next_embeds + dna_alpha * u_dna.unsqueeze(1).to(next_embeds)

            if not is_training and step >= min_latent_steps and g.mean().item() < 0.5:
                current_embeds = next_embeds
                extra = torch.ones(current_mask.size(0), 1,
                                   dtype=current_mask.dtype, device=current_mask.device)
                current_mask = torch.cat([current_mask, extra], dim=1)
                break

            latent_steps_used += 1
            current_embeds = next_embeds
            extra = torch.ones(current_mask.size(0), 1,
                               dtype=current_mask.dtype, device=current_mask.device)
            current_mask = torch.cat([current_mask, extra], dim=1)

        self._gate_hidden_states = gate_hidden_states
        self._gate_u_dna         = u_dna
        self._gate_ot_dists      = gate_ot_dists
        self._gate_delta_ots     = gate_delta_ots
        self._gate_steps_taken   = steps_taken
        self._gate_latent_steps  = latent_steps_used

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            print(f"[W6] latent_steps_used={latent_steps_used}/{steps_taken} "
                  f"(force_full={getattr(self, '_grpo_force_full_trajectory', False)})")

        gen_kwargs.pop("disable_compile", None)

        with torch.no_grad():
            if latent_steps_used > 0:
                past_len = (past_key_values.get_seq_length()
                            if hasattr(past_key_values, "get_seq_length")
                            else past_key_values[0][0].shape[2])
                answer_ids = self.text_model.generate(
                    inputs_embeds=current_embeds, attention_mask=current_mask,
                    past_key_values=past_key_values,
                    cache_position=torch.arange(
                        past_len, past_len + current_embeds.shape[1],
                        device=current_embeds.device),
                    **gen_kwargs,
                )
            else:
                answer_ids = self.text_model.generate(
                    inputs_embeds=original_embeds, attention_mask=original_mask,
                    **gen_kwargs,
                )
        return answer_ids

    return _gen


def _make_gate_loss_w6(gate_net_w6: GateNetW6):
    """Return a replacement for model.compute_gate_loss that handles ot_dists."""

    def _gate_loss(self, gate_reg_weight: float = 0.01) -> torch.Tensor:
        if not getattr(self, "_gate_hidden_states", None):
            return sum(p.sum() for p in gate_net_w6.parameters()) * 0.0

        all_h = torch.stack(self._gate_hidden_states, dim=1)
        B, S, H = all_h.shape
        all_h_flat = all_h.reshape(B * S, H)

        u_dna_exp = None
        if self._gate_u_dna is not None:
            u_dna_exp = self._gate_u_dna.repeat_interleave(S, dim=0)

        ot_flat = None
        if getattr(self, "_gate_ot_dists", None):
            ot_flat = torch.stack(self._gate_ot_dists, dim=1).reshape(B * S, 1)

        delta_ot_flat = None
        if getattr(self, "_gate_delta_ots", None):
            delta_ot_flat = torch.stack(self._gate_delta_ots, dim=1).reshape(B * S, 1)

        step_norm     = torch.arange(S, device=all_h_flat.device, dtype=all_h_flat.dtype) / max(S - 1, 1)
        step_idx_flat = step_norm.repeat(B).unsqueeze(-1)

        g_flat = gate_net_w6(all_h_flat, u_dna=u_dna_exp, ot_dist=ot_flat,
                             delta_ot=delta_ot_flat, step_idx=step_idx_flat)

        if ot_flat is not None:
            ot_min = ot_flat.min()
            ot_max = ot_flat.max()
            if (ot_max - ot_min) < 1e-4:
                gate_loss = -(g_flat * (1 - g_flat)).mean()
            else:
                gate_target = ((ot_flat - ot_min) / (ot_max - ot_min + 1e-8)).squeeze(-1).detach()
                bce_w = (1.0 + 2.0 * gate_target).detach()
                gate_loss = F.binary_cross_entropy(g_flat, gate_target, weight=bce_w)
        else:
            gate_loss = -(g_flat * (1 - g_flat)).mean()

        return gate_reg_weight * gate_loss

    return _gate_loss


def make_ot_reward_func(model: DNALLMModel, manifold_gate: ManifoldGateW6):
    """GRPO reward function scoring completions by OT alignment on the answer manifold."""

    def ot_distance_reward_func(completions, **kwargs) -> List[float]:
        if manifold_gate.mode == "onthefly":
            return [0.0] * len(completions)

        u_dna = getattr(model, "_gate_u_dna", None)
        if u_dna is None:
            return [0.0] * len(completions)

        completion_ids = kwargs.get("completion_ids", None)
        if completion_ids is None:
            return [0.0] * len(completions)

        BG = len(completions)
        B  = u_dna.shape[0]
        G  = BG // max(B, 1)
        u_dna_exp = u_dna.repeat_interleave(G, dim=0)

        device    = u_dna.device
        unwrapped = model.module if hasattr(model, "module") else model

        def _len(ids):
            return ids.shape[0] if hasattr(ids, "shape") else len(ids)
        lengths = [_len(ids) for ids in completion_ids]
        max_len = max(lengths)
        pad_id  = (unwrapped.text_tokenizer.pad_token_id
                   if unwrapped.text_tokenizer.pad_token_id is not None else 0)
        padded  = torch.full((BG, max_len), pad_id, dtype=torch.long, device=device)
        for i, ids in enumerate(completion_ids):
            t = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
            t = t.to(device)
            padded[i, :t.shape[0]] = t
        attn_mask = (padded != pad_id).long()

        with torch.no_grad():
            embeds = unwrapped.text_model.get_input_embeddings()(padded)
            out    = unwrapped.text_model(
                inputs_embeds=embeds, attention_mask=attn_mask,
                output_hidden_states=True, use_cache=False,
            )
            last_idx = attn_mask.sum(dim=1) - 1
            h_last   = out.hidden_states[-1][torch.arange(BG, device=device), last_idx]

        ot_dists = manifold_gate.get_ot_dist(h_last, u_dna_exp)
        if ot_dists is None:
            return [0.0] * BG
        return (0.5 * torch.exp(-ot_dists)).tolist()

    ot_distance_reward_func.__name__ = "ot_distance_reward_func"
    return ot_distance_reward_func


def patch_model_for_w6(
    model:         DNALLMModel,
    gate_net_w6:   GateNetW6,
    manifold_gate: ManifoldGateW6,
) -> None:
    """Replace model.generate_with_hrpo_gate and model.compute_gate_loss with Week 6 versions."""
    model.generate_with_hrpo_gate = types.MethodType(
        _make_adaptive_generate(gate_net_w6, manifold_gate), model
    )
    model.compute_gate_loss = types.MethodType(
        _make_gate_loss_w6(gate_net_w6), model
    )


# ── ManifoldGRPOTrainerW6 ─────────────────────────────────────────────────────

class ManifoldGRPOTrainerW6(DNALLMGRPOTrainer):
    """Extends DNALLMGRPOTrainer with L_manifold from ManifoldGateW6."""

    def __init__(self, manifold_gate: ManifoldGateW6, manifold_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.manifold_gate   = manifold_gate
        self.manifold_weight = manifold_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

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
        batch_dna_embeds = unwrapped.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size)
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


# ── Main (Week 6) ─────────────────────────────────────────────────────────────

def main(script_args, training_args, model_args):
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    manifold_gate = ManifoldGateW6(stage2_dir=script_args.stage2_dir)

    use_dna_gate = training_args.use_dna_gate
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
        use_dna_gate        = use_dna_gate,
        device              = "cuda",
    ).to("cuda")
    model.text_model.config.use_cache = False

    hidden_size = model.text_hidden_size
    dna_size    = hidden_size if use_dna_gate else 0
    ot_size     = 1 if manifold_gate.mode != "onthefly" else 0
    gate_net_w6 = GateNetW6(
        hidden_size=hidden_size, gate_hidden=128,
        dna_size=dna_size, ot_size=ot_size,
        use_step_idx=True, use_delta_ot=True,
    ).to("cuda")
    model.gate_net = gate_net_w6

    gate_sft_path = getattr(script_args, "gate_sft_checkpoint", None)
    if gate_sft_path and os.path.exists(gate_sft_path):
        ckpt    = torch.load(gate_sft_path, map_location="cpu")
        weights = ckpt["gate_net"] if isinstance(ckpt, dict) and "gate_net" in ckpt else ckpt
        gate_net_w6.load_state_dict(weights)
        epoch_info = (f"  epoch={ckpt['epoch']}  loss={ckpt['loss']:.4f}"
                      if isinstance(ckpt, dict) and "epoch" in ckpt else "")
        print(f"[W6] Loaded Stage 2.5 gate_net from {gate_sft_path}{epoch_info}")
    else:
        print("[W6] No gate_sft_checkpoint provided — gate_net initialised fresh (bias=-2.0)")

    patch_model_for_w6(model, gate_net_w6, manifold_gate)
    model._grpo_force_full_trajectory = False

    print(f"[W6] GateNetW6: hidden={hidden_size} dna={dna_size} ot={ot_size} "
          f"mode={manifold_gate.mode}")

    _load_sft_checkpoint(model, model_args, merge_lora=True)
    _prep_for_training(model, model_args)

    if script_args.full_ckpt is not None:
        bin_path = os.path.join(script_args.full_ckpt, "pytorch_model.bin")
        if os.path.exists(bin_path):
            ckpt = torch.load(bin_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f"full_ckpt: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model = model.to(training_args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    data = get_kegg_dataset(
        kegg_csv=script_args.kegg_csv,
        dataset_name=script_args.dataset_name,
        truncate_dna_per_side=model_args.truncate_dna_per_side,
    )

    _registry = {**reward_funcs_registry, "ot_distance": make_ot_reward_func(model, manifold_gate)}
    reward_funcs = [_registry[f] for f in script_args.reward_funcs]
    print(f"[W6] Reward functions: {script_args.reward_funcs}")

    trainer = ManifoldGRPOTrainerW6(
        manifold_gate=manifold_gate, manifold_weight=script_args.manifold_weight,
        model=model, reward_funcs=reward_funcs, args=training_args,
        dna_module=NucleotideDNAModule(),
        train_dataset=data["train"],
        eval_dataset=data["val"] if training_args.eval_strategy != "no" else None,
        peft_config=None, callbacks=[SaveWithPyTorchCallback()],
        processing_class=model.processor,
    )
    training_args.save_safetensors = False

    resume = training_args.resume_from_checkpoint
    if resume in ("True", "true"):
        checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        resume = str(max(checkpoints, key=os.path.getmtime)) if checkpoints else None
        print(f"Auto-resume: {resume}")

    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
    os.environ.setdefault("WANDB_PROJECT", "dna-grpo-week6")

    parser = TrlParser((GRPOScriptArguments, DNALLMGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.save_safetensors = False
    training_args.vllm_server_base_url = os.environ.get("VLLM_BASE_URL")

    main(script_args, training_args, model_args)
