import csv
import gc
import glob
import os
import time
import traceback
from argparse import ArgumentParser
from functools import partial
from typing import *

import math

import torch
import torch.nn.functional as F
import wandb
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding

import pytorch_lightning as pl
from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

from genomorph.dataset.kegg import get_format_kegg_function, load_kegg_from_anon_csv, qwen_dna_collate_fn
from genomorph.dataset.utils import truncate_dna
from genomorph.dataset.variant_effect import (
    clean_variant_effect_example,
    clean_variant_effect_non_snv_example,
    get_format_variant_effect_function,
)
from genomorph.models.dl.processing_dl import DLProcessor
from genomorph.models.dna_llm import DNALLMModel, get_target_modules

from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
register_evo2_tokenizer()

# Set start method to 'spawn' for CUDA compatibility with multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DNALLMFineTuner(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning DNA-LLM models.
    """

    def __init__(self, hparams):
        """
        Initialize the DNALLMFineTuner.

        Args:
            hparams: Hyperparameters for the model and training
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        self.text_model_name = self.hparams.text_model_name
        self.dna_model_name = self.hparams.dna_model_name
        self.cache_dir = self.hparams.cache_dir
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay
        self.text_model_finetune = self.hparams.text_model_finetune
        self.dna_model_finetune = self.hparams.dna_model_finetune
        self.lora_rank = self.hparams.lora_rank
        self.lora_alpha = self.hparams.lora_alpha
        self.lora_dropout = self.hparams.lora_dropout
        self.max_length_dna = self.hparams.max_length_dna
        self.max_length_text = self.hparams.max_length_text
        self.dna_is_evo2 = self.hparams.dna_is_evo2
        self.dna_embedding_layer = self.hparams.dna_embedding_layer
        self.return_answer_in_batch = self.hparams.return_answer_in_batch
        self.merge_val_test_set = self.hparams.merge_val_test_set
        self.max_clip_loss_weight = self.hparams.max_clip_loss_weight
        self.clip_temperature = self.hparams.clip_temperature

        # Store dataset configuration
        self.dataset_type = self.hparams.dataset_type

        # Load model
        self.model = DNALLMModel(
            text_model_name=self.text_model_name,
            dna_model_name=self.dna_model_name,
            cache_dir=self.cache_dir,
            max_length_dna=self.max_length_dna,
            max_length_text=self.max_length_text,
            text_model_finetune=self.text_model_finetune,
            dna_model_finetune=self.dna_model_finetune,
            dna_is_evo2=self.dna_is_evo2,
            dna_embedding_layer=self.dna_embedding_layer,
            use_cross_attention=self.hparams.get("use_cross_attention", False),
            use_hrpo_gate=self.hparams.get("use_hrpo_gate", False),
            use_dna_gate=self.hparams.get("use_dna_gate", False),
        )

        self.text_model = self.model.text_model
        self.dna_model = self.model.dna_model
        self.dna_projection = self.model.dna_projection

        # Load tokenizer for target text
        self.tokenizer = self.model.text_tokenizer

        # ── CLIP memory queue (MoCo-style negatives) ──────────────────────────
        # Stores normalized (u_dna, delta_h) from previous steps so the
        # contrastive loss has meaningful negatives even at batch_size=1.
        self.clip_queue_size = self.hparams.clip_queue_size
        H = self.model.text_hidden_size
        self.register_buffer("_clip_q_dna",    F.normalize(torch.randn(self.clip_queue_size, H), dim=-1))
        self.register_buffer("_clip_q_text",   F.normalize(torch.randn(self.clip_queue_size, H), dim=-1))
        self.register_buffer("_clip_q_ptr",    torch.zeros(1, dtype=torch.long))
        self.register_buffer("_clip_q_len",    torch.zeros(1, dtype=torch.long))
        # ── OT queue (Week 5 HiRef) ───────────────────────────────────────────
        # Stores normalized answer hidden states as the target distribution for
        # Sinkhorn OT loss. Shares ptr/len with the CLIP queue so both are
        # always in sync.
        self.register_buffer("_ot_q_answer",   F.normalize(torch.randn(self.clip_queue_size, H), dim=-1))
        self._train_step_count = 0  # manual counter; self._train_step_count stays 0 with Evo2's internal model-parallel

        # Prepare model for training
        self.lora_config = self._prep_for_training()

    def on_train_start(self):
        # Convert --hrpo_warmup_epochs to steps once dataset size is known
        if self.hparams.get("hrpo_warmup_epochs", 0) > 0:
            steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            warmup_steps = int(self.hparams.get("hrpo_warmup_epochs", 0) * steps_per_epoch)
            self.hparams.hrpo_warmup_steps = warmup_steps
            print(f"  [hrpo] warmup_epochs={self.hparams.get('hrpo_warmup_epochs', 0)} → "
                  f"warmup_steps={warmup_steps} ({steps_per_epoch} steps/epoch)")

    def _prep_for_training(self) -> LoraConfig:
        """
        Load and configure the DNALLMModel.
        """

        # Freeze DNA encoder parameters (skip when no DNA model is loaded)
        if self.dna_model is not None and not self.dna_model_finetune:
            if self.dna_is_evo2:
                for param in self.dna_model.model.parameters():
                    param.requires_grad = False
            else:
                for param in self.dna_model.parameters():
                    param.requires_grad = False

        if self.text_model_finetune:
            target_modules = get_target_modules(self)

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=target_modules,
                init_lora_weights="gaussian",
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Prepare text model for training
            self.text_model = prepare_model_for_kbit_training(self.text_model)
            self.text_model = get_peft_model(self.text_model, lora_config)
        else:
            # Freeze text model parameters
            for param in self.text_model.parameters():
                param.requires_grad = False

        # Make projection layer trainable (Identity has no params in LLM-only mode)
        if self.dna_model is not None:
            for param in self.dna_projection.parameters():
                param.requires_grad = True

        return lora_config

    @torch.no_grad()
    def _clip_enqueue(self, u_norm: torch.Tensor, d_norm: torch.Tensor, a_norm: Optional[torch.Tensor] = None):
        """Push current batch embeddings into the circular CLIP and OT queues."""
        B = u_norm.shape[0]
        ptr = int(self._clip_q_ptr)
        for k in range(B):
            slot = (ptr + k) % self.clip_queue_size
            self._clip_q_dna[slot]  = u_norm[k].detach().float()
            self._clip_q_text[slot] = d_norm[k].detach().float()
            if a_norm is not None:
                self._ot_q_answer[slot] = a_norm[k].detach().float()
        self._clip_q_ptr[0] = (ptr + B) % self.clip_queue_size
        self._clip_q_len[0] = min(int(self._clip_q_len) + B, self.clip_queue_size)

    def _compute_clip_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        dna_tokenized,
        batch_idx_map,
    ):
        """
        Compute the CLIP contrastive loss aligning DNA embeddings to
        answer hidden-state shifts (Step 1).

        u_dna   = mean of CrossAttentionFusion DNA token embeddings  [B, H]
        delta_h = mean(h_answer) - mean(h_prompt)                    [B, H]

        Returns (l_clip, mean_sim) — both scalar tensors.
        """
        batch_size = input_ids.shape[0]

        # ── DNA embeddings (gradient flows through dna_projection) ───────────
        batch_dna_embeds = self.model.process_dna_embeddings(
            dna_tokenized, batch_idx_map, batch_size
        )
        _p = next(self.model.dna_projection.parameters())
        u_dna_list = []
        for i in range(batch_size):
            parts = []
            for slot in (i, i + batch_size):
                if slot < len(batch_dna_embeds) and batch_dna_embeds[slot].shape[0] > 0:
                    parts.append(batch_dna_embeds[slot])
            if parts:
                u_dna_list.append(torch.cat(parts, dim=0).mean(dim=0))
            else:
                u_dna_list.append(torch.zeros(self.model.text_hidden_size,
                                              device=_p.device, dtype=_p.dtype))
        u_dna = torch.stack(u_dna_list).to(input_ids.device)   # [B, H]

        # ── Answer hidden-state shift (no gradient — stable target) ──────────
        # prompt positions: labels == -100 AND attention_mask == 1
        # answer positions: labels != -100
        prompt_mask = (labels == -100) & (attention_mask == 1)
        answer_mask = labels != -100

        with torch.no_grad():
            inputs_embeds, attn_mask = self.model.get_prompt_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dna_tokenized=dna_tokenized,
                batch_idx_map=batch_idx_map,
            )
            out = self.model.text_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = out.hidden_states[-1]   # [B, T, H]

        # Mean over valid prompt / answer tokens per batch item
        def masked_mean(h, mask):
            mask_f = mask.unsqueeze(-1).float()
            return (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        prompt_hidden = masked_mean(hidden, prompt_mask)
        answer_hidden = masked_mean(hidden, answer_mask)
        delta_h = (answer_hidden - prompt_hidden).to(u_dna.dtype)

        # ── Symmetric CLIP contrastive loss with memory queue ─────────────────
        u_norm = F.normalize(u_dna.float(), dim=-1)   # [B, H]
        d_norm = F.normalize(delta_h.float(), dim=-1) # [B, H]

        q_len = int(self._clip_q_len)
        if q_len > 0:
            # Augment keys with detached queue entries (negatives from past steps)
            q_dna  = self._clip_q_dna[:q_len].to(u_norm.device)   # [Q, H]
            q_text = self._clip_q_text[:q_len].to(d_norm.device)  # [Q, H]
            all_text = torch.cat([d_norm, q_text], dim=0)          # [B+Q, H]
            all_dna  = torch.cat([u_norm, q_dna],  dim=0)          # [B+Q, H]
        else:
            all_text = d_norm
            all_dna  = u_norm

        lbls  = torch.arange(batch_size, device=input_ids.device)
        sim   = u_norm @ all_text.T / self.clip_temperature  # [B, B+Q]
        sim_t = d_norm @ all_dna.T  / self.clip_temperature  # [B, B+Q]
        l_clip = (F.cross_entropy(sim, lbls) + F.cross_entropy(sim_t, lbls)) / 2

        # Enqueue current batch (also stores a_norm for OT queue)
        a_norm = F.normalize(answer_hidden.float(), dim=-1)  # [B, H] — target for L_OT
        self._clip_enqueue(u_norm, d_norm, a_norm)

        mean_sim = (u_norm * d_norm).sum(dim=-1).mean().detach()
        return l_clip, mean_sim, u_norm, a_norm

    @staticmethod
    def _sinkhorn_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        n_iter: int = 50,
    ) -> torch.Tensor:
        """
        Log-domain Sinkhorn OT distance between point clouds x [N,D] and y [M,D].
        Uniform marginals. Gradients flow through the cost matrix C (i.e. through x).
        y should be detached (stable target distribution).
        """
        N, M = x.shape[0], y.shape[0]
        C = torch.cdist(x.float(), y.float(), p=2).pow(2)  # [N, M] squared-L2 cost

        a = torch.ones(N, device=x.device, dtype=torch.float32) / N
        b = torch.ones(M, device=x.device, dtype=torch.float32) / M
        log_a, log_b = a.log(), b.log()

        # Sinkhorn dual iterations (no grad — plan is treated as fixed)
        with torch.no_grad():
            eps = C.detach().median().clamp(min=1e-2).item()  # median heuristic: auto-scales with embedding distances
            log_K = -C.detach() / eps   # [N, M]
            f = torch.zeros(N, device=x.device, dtype=torch.float32)
            g = torch.zeros(M, device=x.device, dtype=torch.float32)
            for _ in range(n_iter):
                f = log_a - torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
                g = log_b - torch.logsumexp(log_K.T + f.unsqueeze(0), dim=1)
            log_T = f.unsqueeze(1) + g.unsqueeze(0) + log_K   # log-potentials + log_K = (f+g-C)/eps
            T = log_T.exp()             # [N, M] optimal transport plan

        # Loss: T (detached plan) · C (with grad through x).
        # Multiply by N to normalize per-sample gradient magnitude to O(1).
        # Without this, uniform marginals a[i]=1/N make the gradient at each
        # source point O(1/N), ~1000x weaker than CE loss at N=65.
        return (T * C).sum() * N

    def _compute_ot_loss(
        self,
        u_norm: torch.Tensor,   # [B, H] current batch DNA embeddings (with grad)
        a_norm: torch.Tensor,   # [B, H] current batch answer hidden states (detached)
    ) -> torch.Tensor:
        """
        L_OT: Sinkhorn Wasserstein distance between DNA latent distribution
        and answer manifold. Both distributions are built from the current
        batch + MoCo queue so the OT problem is meaningful even at batch_size=1.

        Gradient only flows through x (DNA side); y is fully detached so the
        answer manifold acts as a stable target — analogous to the target
        network in MoCo.
        """
        q_len = int(self._clip_q_len)
        if q_len > 0:
            q_dna    = self._clip_q_dna[:q_len].to(u_norm.device)      # [Q, H] detached
            q_answer = self._ot_q_answer[:q_len].to(a_norm.device)     # [Q, H] detached
            x = torch.cat([u_norm.float(), q_dna], dim=0)              # [B+Q, H] — grad on u_norm only
            y = torch.cat([a_norm.float(), q_answer], dim=0)           # [B+Q, H]
        else:
            x = u_norm.float()
            y = a_norm.float()
        # y fully detached: answer manifold is a stable target, no gradient flows back
        return self._sinkhorn_loss(x, y.detach(),
                                   n_iter=self.hparams.get("ot_n_iter", 50))

    def _step(self, batch: Dict, batch_idx: int, prefix: str) -> torch.Tensor:
        """
        Performs a single step for training, validation, or testing.

        Args:
            batch: Dictionary containing the batch data
            batch_idx: Integer indicating the batch index
            prefix: String indicating the step type ('train', 'val', or 'test')

        Returns:
            torch.Tensor: The computed loss for this batch
        """
        if prefix == "test":
            return {"loss": torch.tensor(0.0, device=self.device)}

        # Get batch data from the collate function
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device) if "labels" in batch else None
        dna_tokenized = batch.get("dna_tokenized")
        if dna_tokenized is not None:
            dna_tokenized = dna_tokenized.to(self.device)
        batch_idx_map = batch.get("batch_idx_map")

        # Forward pass through the model
        # When HRPO gate is active we need hidden states to train gate_net.
        need_hidden = (prefix == "train" and self.hparams.get("use_hrpo_gate", False))
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map,
            labels=labels,
            output_hidden_states=need_hidden,
        )

        # Get the loss from model outputs
        loss = outputs.loss

        # ── L_gate: HRPO gate regularization (Week 3) ────────────────────────
        # Use the last hidden state at the boundary between prompt and answer
        # as a single-step gate probe. This trains gate_net without an extra
        # LLM forward pass. The full multi-step gate loop runs during GRPO.
        if need_hidden and outputs.hidden_states is not None:
            # Find prompt length: first position where labels != -100
            if labels is not None:
                label_mask = (labels != -100)  # [B, T]
                # prompt ends just before first label token
                prompt_ends = label_mask.float().argmax(dim=1) - 1  # [B]
                prompt_ends = prompt_ends.clamp(min=0)
            else:
                prompt_ends = torch.full((input_ids.size(0),), input_ids.size(1) - 1,
                                         device=self.device, dtype=torch.long)

            # Gather last-prompt hidden state per example: [B, hidden_size]
            last_hidden = outputs.hidden_states[-1]  # [B, T, H]
            idx = prompt_ends.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
            h_probe = last_hidden.gather(1, idx).squeeze(1).detach()  # detach from LLM, grad via gate_net only

            # Week 4: compute DNA summary vector for gate conditioning
            u_dna_gate = None
            if (self.hparams.get("use_dna_gate", False)
                    and dna_tokenized is not None
                    and batch_idx_map is not None):
                u_dna_gate = self.model.get_dna_summary(
                    dna_tokenized=dna_tokenized,
                    batch_idx_map=batch_idx_map,
                    batch_size=input_ids.size(0),
                )

            # Populate cache so compute_gate_loss() works
            self.model.gate_net  # ensure it exists
            self.model._gate_hidden_states = [h_probe]
            self.model._gate_u_dna = u_dna_gate

            # Compute gate values (no grad) for logging
            with torch.no_grad():
                gate_vals = self.model.gate_net(h_probe, u_dna=u_dna_gate)  # [B]
            gate_mean = gate_vals.mean().item()
            gate_min  = gate_vals.min().item()
            gate_max  = gate_vals.max().item()

            l_gate = self.model.compute_gate_loss(
                gate_reg_weight=self.hparams.gate_reg_weight
            )
            loss = loss + l_gate

            self.log("train_gate_loss",  l_gate.detach(),  on_step=True, prog_bar=False, logger=True)
            self.log("gate_value_mean",  gate_mean,         on_step=True, prog_bar=True,  logger=True)
            self.log("gate_value_min",   gate_min,          on_step=True, prog_bar=False, logger=True)
            self.log("gate_value_max",   gate_max,          on_step=True, prog_bar=False, logger=True)

            if self._train_step_count % 100 == 0:
                print(f"  [gate] step={self._train_step_count}  mean={gate_mean:.3f}  "
                      f"min={gate_min:.3f}  max={gate_max:.3f}  "
                      f"L_gate={l_gate.item():.4f}")

        # ── L_CLIP + L_OT (Week 5 HiRef) ────────────────────────────────────
        _run_clip = (prefix == "train"
                     and self.max_clip_loss_weight > 0.0
                     and dna_tokenized is not None
                     and labels is not None)
        _run_ot   = (prefix == "train"
                     and self.hparams.get("ot_weight", 0.0) > 0.0
                     and dna_tokenized is not None
                     and labels is not None
                     and self._train_step_count >= self.clip_queue_size)  # wait for full queue
        # Always compute embeddings when OT is configured so the queue fills during warmup.
        # Without this, CLIP=0 + OT-not-started → _compute_clip_loss never called → queue never fills.
        _needs_embeddings = _run_clip or _run_ot or (
            prefix == "train"
            and self.hparams.get("ot_weight", 0.0) > 0.0
            and dna_tokenized is not None
            and labels is not None
        )

        if _needs_embeddings:
            l_clip, mean_sim, u_norm, a_norm = self._compute_clip_loss(
                input_ids, attention_mask, labels, dna_tokenized, batch_idx_map
            )

            if _run_clip:
                total_steps = max(self.trainer.estimated_stepping_batches, 1)
                progress  = self._train_step_count / total_steps
                schedule  = (1.0 + math.cos(math.pi * progress)) / 2.0
                alignment = (1.0 - mean_sim).clamp(0.0, 1.0).item()
                effective_clip_weight = self.max_clip_loss_weight * schedule * alignment
                loss = loss + effective_clip_weight * l_clip
                self.log("train_clip_loss",   l_clip.detach(),      on_step=True, prog_bar=False, logger=True)
                self.log("train_clip_weight", effective_clip_weight, on_step=True, prog_bar=False, logger=True)
                self.log("train_clip_sim",    mean_sim,              on_step=True, prog_bar=False, logger=True)

            if _run_ot:
                l_ot = self._compute_ot_loss(u_norm, a_norm)
                loss = loss + self.hparams.get("ot_weight", 0.0) * l_ot
                self.log("train_ot_loss", l_ot.detach(), on_step=True, prog_bar=True,  logger=True)
                if self._train_step_count % 100 == 0:
                    print(f"  [L_OT] step={self._train_step_count}  l_ot={l_ot.item():.4f}  "
                          f"ot_weight={self.hparams.get('ot_weight', 0.0)}")

        # Occasionally show generations for debugging purposes - ONLY during training/validation
        # You can reduce the frequency of generations by increasing the step size to make the model train faster
        if (prefix == "train" and (self._train_step_count % 3000 == 0)) or (prefix == "val" and (batch_idx % 300 == 0)):
            try:
                # Select first example from batch for demonstration
                example_idx = 0

                print(
                    f"\n=== Sample Generation (step {self._train_step_count} / {self.trainer.estimated_stepping_batches * self.hparams.gradient_accumulation_steps}) ==="
                )

                # Get the tokens that define the assistant pattern
                assistant_start_marker = "<|im_start|>assistant\n"
                assistant_marker_tokens = self.tokenizer.encode(assistant_start_marker, add_special_tokens=False)
                marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
                marker_len = len(assistant_marker_tokens)

                # Find non-padding tokens in input
                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad) > 0:
                    start_idx = non_pad[0].item()  # First non-padding token
                else:
                    start_idx = 0

                # For each position, check if the next marker_len tokens match the pattern
                matches = []
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos : pos + marker_len] == marker_tensor):
                        matches.append(pos)
                        break  # Stop at first match

                assistant_pos = matches[0] if matches else None

                if assistant_pos is not None:
                    # Get input up to and including the assistant marker
                    gen_input_ids = input_ids[
                        example_idx : example_idx + 1, start_idx : assistant_pos + marker_len
                    ]
                    gen_attention_mask = attention_mask[
                        example_idx : example_idx + 1, start_idx : assistant_pos + marker_len
                    ]

                    # Extract DNA data for this example
                    example_dna_data = None
                    example_batch_map = None

                    if dna_tokenized is not None and batch_idx_map is not None:
                        # Find DNA sequences for this example
                        example_indices = [i for i, idx in enumerate(batch_idx_map) if idx == example_idx]

                        if len(example_indices) > 0:
                            # Extract just this example's DNA data
                            example_dna_data = BatchEncoding(
                                {
                                    "input_ids": dna_tokenized.input_ids[example_indices].to(self.device),
                                    "attention_mask": dna_tokenized.attention_mask[example_indices].to(self.device),
                                }
                            )

                            # For generation we need all sequences mapped to index 0
                            example_batch_map = [0] * len(example_indices)

                    # Week 4: compute u_dna for generation gate conditioning
                    gen_u_dna = None
                    if (self.hparams.get("use_dna_gate", False)
                            and example_dna_data is not None
                            and example_batch_map is not None):
                        gen_u_dna = self.model.get_dna_summary(
                            dna_tokenized=example_dna_data,
                            batch_idx_map=example_batch_map,
                            batch_size=1,
                        )

                    # Use HRPO gate in generation only after warmup steps
                    hrpo_ready = (
                        self.hparams.get("use_hrpo_gate", False)
                        and self._train_step_count >= self.hparams.get("hrpo_warmup_steps", 0)
                    )

                    # Generate text
                    with torch.no_grad():
                        if hrpo_ready:
                            generated = self.model.generate_with_hrpo_gate(
                                input_ids=gen_input_ids,
                                dna_tokenized=example_dna_data,
                                batch_idx_map=example_batch_map,
                                max_latent_steps=self.hparams.max_latent_steps,
                                gate_threshold=self.hparams.gate_threshold,
                                u_dna=gen_u_dna,
                                max_new_tokens=2048,
                                temperature=0.6,
                                top_p=0.95,
                                top_k=20,
                                do_sample=True,
                            )
                        else:
                            generated = self.model.generate(
                                input_ids=gen_input_ids,
                                attention_mask=gen_attention_mask,
                                dna_tokenized=example_dna_data,
                                batch_idx_map=example_batch_map,
                                max_new_tokens=2048,
                                temperature=0.6,
                                top_p=0.95,
                                top_k=20,
                                do_sample=True,
                            )

                    # Decode and display
                    user_input = self.tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
                    generation = self.tokenizer.decode(generated[0], skip_special_tokens=False).strip()

                    # Collapse <|dna_start|>...<|dna_pad|>×N...<|dna_end|> into one compact token
                    import re as _re
                    def _compact_dna(text):
                        def _repl(m):
                            n = len(_re.findall(r'<\|dna_pad\|>', m.group(0)))
                            has_end = m.group(2) is not None
                            end_str = '<|dna_end|>' if has_end else ' [NO dna_end!]'
                            return f"<|dna_start|><|dna_pad| ×{n}>{end_str}"
                        return _re.sub(
                            r'<\|dna_start\|>((?:<\|dna_pad\|>)+)(<\|dna_end\|>)?',
                            _repl, text)
                    user_input_display = _compact_dna(user_input)

                    # Free memory early
                    del generated, gen_input_ids, gen_attention_mask, example_dna_data, example_batch_map
                    gc.collect()

                    # Get ground truth if available
                    ground_truth = ""
                    if labels is not None:
                        valid_label_pos = (labels[example_idx] != -100).nonzero(as_tuple=True)[0]
                        if len(valid_label_pos) > 0:
                            if valid_label_pos[0] >= assistant_pos + marker_len:
                                ground_truth = self.tokenizer.decode(
                                    input_ids[example_idx, valid_label_pos], skip_special_tokens=False
                                ).strip()

                    print(f"=====[Sample {prefix} {batch_idx}]=====")
                    print(f"=====[User input]=====\n{user_input_display}")
                    if ground_truth:
                        print(f"=====[Ground truth]=====\n{ground_truth}")
                    print(f"=====[Complete generation]=====\n{generation}")

                    # Log to wandb
                    timestamp = time.time()
                    step_id = f"gen_{self._train_step_count}-{timestamp}"
                    wandb_logger = self.logger.experiment
                    wandb_logger.log(
                        {
                            step_id: wandb.Table(
                                columns=["timestamp", "prefix", "batch_idx", "user_input", "generation", "ground_truth"],
                                data=[[timestamp, prefix, batch_idx, user_input, generation, ground_truth]],
                            )
                        }
                    )

                    # Clean up memory
                    del user_input, user_input_display, generation, ground_truth
                    torch.cuda.empty_cache()
                    gc.collect()

                else:
                    print("No assistant marker found in the input sequence")

            except Exception as e:
                print(f"Error during sample generation: {str(e)}")
                traceback.print_exc()

        # Get current learning rate (skip during test as scheduler might not be available)
        if prefix != "test":
            current_lr = self.lr_schedulers().get_last_lr()[0]
        else:
            current_lr = 0

        # Logging metrics
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{prefix}_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        # Only log learning rate during training/validation
        if prefix != "test":
            self.log(
                "lr",
                current_lr,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        self._train_step_count += 1
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single validation step."""
        return self._step(batch, batch_idx, prefix="val")
    
    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single test step."""
        return self._step(batch, batch_idx, prefix="test")

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Tuple[List, List]: A tuple containing a list of optimizers and schedulers
        """
        proj_params = list(self.dna_projection.parameters())
        proj_param_ids = {id(p) for p in proj_params}
        other_params = [p for p in self.parameters() if id(p) not in proj_param_ids]
        optimizer = AdamW([
            {"params": proj_params,  "lr": self.learning_rate},  # uniform LR (benchmark-matched)
            {"params": other_params, "lr": self.learning_rate},
        ], weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        # Load dataset based on type specified in hyperparameters

        if self.hparams.dataset_type == "kegg":
            if self.hparams.kegg_csv:
                dataset = load_kegg_from_anon_csv(self.hparams.kegg_csv)
            else:
                dataset = load_dataset(self.hparams.kegg_data_dir_huggingface)
            dataset = dataset.map(get_format_kegg_function(self.hparams.model_type))

            labels = []
            for split, data in dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            train_dataset = dataset["train"]

            if self.hparams.truncate_dna_per_side:
                train_dataset = train_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )
            
            processor = DLProcessor(
                tokenizer=self.model.text_tokenizer,
                dna_tokenizer=self.model.dna_tokenizer,
            )

            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
                truncate_for_generation=False,
            )


        elif self.hparams.dataset_type == "variant_effect_coding":
            dataset = load_dataset(self.hparams.variant_effect_coding_data_dir_huggingface)
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.map(get_format_variant_effect_function(self.hparams.model_type))

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            train_dataset = dataset["train"]

            if self.hparams.truncate_dna_per_side:
                train_dataset = train_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )
            
            processor = DLProcessor(
                    tokenizer=self.model.text_tokenizer,
                    dna_tokenizer=self.model.dna_tokenizer,
                )
            
            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
                truncate_for_generation=False,
            )

        elif self.hparams.dataset_type == "variant_effect_non_snv":
            dataset = load_dataset(self.hparams.variant_effect_non_snv_data_dir_huggingface)
            dataset = dataset.map(clean_variant_effect_non_snv_example)
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.rename_column("mutated_sequence", "variant_sequence")

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            train_dataset = dataset["train"]

            if self.hparams.truncate_dna_per_side:
                train_dataset = train_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )
            train_dataset = train_dataset.map(get_format_variant_effect_function(self.hparams.model_type))

            processor = DLProcessor(
                tokenizer=self.model.text_tokenizer,
                dna_tokenizer=self.model.dna_tokenizer,
            )

            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
                truncate_for_generation=False,
            )

        else:
            raise ValueError(f"Unknown dataset type: {self.hparams.dataset_type}")

        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""

        if self.hparams.dataset_type == "kegg":
            if self.hparams.kegg_csv:
                dataset = load_kegg_from_anon_csv(self.hparams.kegg_csv)
            else:
                dataset = load_dataset(self.hparams.kegg_data_dir_huggingface)
            dataset = dataset.map(get_format_kegg_function(self.hparams.model_type))

            if self.hparams.merge_val_test_set:
                val_dataset = concatenate_datasets([dataset['test'], dataset['val']])
            else:
                val_dataset = dataset["val"]

            labels = []
            for split, data in dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            if self.hparams.truncate_dna_per_side:
                val_dataset = val_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

        elif self.hparams.dataset_type == "variant_effect_coding":
            dataset = load_dataset(self.hparams.variant_effect_coding_data_dir_huggingface)
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.map(get_format_variant_effect_function(self.hparams.model_type))

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            val_dataset = dataset["test"]

            if self.hparams.truncate_dna_per_side:
                val_dataset = val_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )
        
        elif self.hparams.dataset_type == "variant_effect_non_snv":
            dataset = load_dataset(self.hparams.variant_effect_non_snv_data_dir_huggingface)
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.map(clean_variant_effect_non_snv_example)

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            dataset = dataset.rename_column("mutated_sequence", "variant_sequence")
            val_dataset = dataset["test"]

            if self.hparams.truncate_dna_per_side:
                val_dataset = val_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )
            val_dataset = val_dataset.map(get_format_variant_effect_function(self.hparams.model_type))
            
        else:
            raise ValueError(f"Unknown dataset type: {self.hparams.dataset_type}")
    
        processor = DLProcessor(
                tokenizer=self.model.text_tokenizer,
                dna_tokenizer=self.model.dna_tokenizer,
            )
        
        # Create partial function with all required arguments except the batch
        collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
                truncate_for_generation=False,
            )

        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        return self.val_dataloader()
    
    def on_test_epoch_end(self):
        """
        Generate text for all test examples and compute accuracy + weighted F1.

        Saves results incrementally after every batch so a crashed run can be
        resumed: if the CSV already exists the processed (batch_idx, example_idx)
        pairs are skipped and counters are restored from the saved rows.
        """
        wandb_logger = self.logger.experiment
        wandb_logger.log({"test_progress": 0.0, "status": "starting test generation"})

        self.model.eval()

        test_dataloader = self.test_dataloader()
        total_batches = len(test_dataloader)

        label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        num_classes = len(self.labels)
        wandb_logger.log({"num_classes": num_classes, "labels": str(self.labels)})

        # ── CSV path (fixed name so resume can find it) ───────────────────────
        model_name = self.hparams.text_model_name.split('/')[-1]
        csv_dir = (os.path.dirname(self.hparams.ckpt_path)
                   if self.hparams.ckpt_path else self.hparams.checkpoint_dir)
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"test_generations_{model_name}.csv")
        fieldnames = ["batch_idx", "example_idx", "user_input", "generation",
                      "ground_truth", "predicted_label", "correct"]

        # ── Resume: load already-processed rows ──────────────────────────────
        done_keys: set = set()
        total_examples = 0
        correct = 0
        all_preds: List[int] = []
        all_targets: List[int] = []

        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    done_keys.add((int(row["batch_idx"]), int(row["example_idx"])))
                    total_examples += 1
                    if str(row["correct"]).lower() == "true":
                        correct += 1
                    all_preds.append(label2id.get(row["predicted_label"].lower(), 0))
                    all_targets.append(label2id.get(row["ground_truth"].lower(), 0))
            print(f"[TEST] Resuming — skipping {total_examples} already-saved examples.")

        # Open CSV in append mode; write header only for a new file
        csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if total_examples == 0:
            csv_writer.writeheader()
            csv_file.flush()

        # ── Generation loop ───────────────────────────────────────────────────
        assistant_start_marker = "<|im_start|>assistant\n"
        assistant_marker_tokens = self.model.text_tokenizer.encode(
            assistant_start_marker, add_special_tokens=False
        )
        marker_len = len(assistant_marker_tokens)

        print(f"\n[TEST] Starting generation for {total_batches} batches...")
        for batch_idx, batch in enumerate(test_dataloader):
            wandb_logger.log({
                "test_progress": batch_idx / total_batches,
                "status": f"processing batch {batch_idx}/{total_batches}",
            })

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            answer = batch["answer"]
            dna_tokenized = batch.get("dna_tokenized")
            if dna_tokenized is not None:
                dna_tokenized = dna_tokenized.to(self.device)
            batch_idx_map = batch.get("batch_idx_map")

            marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
            new_rows: List[Dict] = []

            for example_idx in range(input_ids.size(0)):
                # Skip already-processed examples when resuming
                if (batch_idx, example_idx) in done_keys:
                    continue

                # Find first real (non-padding) token
                non_pad = (input_ids[example_idx] != self.model.text_tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0

                # Locate assistant start marker
                assistant_pos = None
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos:pos + marker_len] == marker_tensor):
                        assistant_pos = pos
                        break

                if assistant_pos is None:
                    continue

                gen_input_ids = input_ids[example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]
                gen_attention_mask = attention_mask[example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]

                example_dna_data = None
                example_batch_map = None
                if dna_tokenized is not None and batch_idx_map is not None:
                    example_indices = [i for i, idx in enumerate(batch_idx_map) if idx == example_idx]
                    if example_indices:
                        example_dna_data = BatchEncoding({
                            "input_ids": dna_tokenized.input_ids[example_indices].to(self.device),
                            "attention_mask": dna_tokenized.attention_mask[example_indices].to(self.device),
                        })
                        example_batch_map = [0] * len(example_indices)

                with torch.no_grad():
                    if self.hparams.get("use_hrpo_gate", False):
                        generated = self.model.generate_with_hrpo_gate(
                            input_ids=gen_input_ids,
                            dna_tokenized=example_dna_data,
                            batch_idx_map=example_batch_map,
                            max_latent_steps=self.hparams.max_latent_steps,
                            gate_threshold=self.hparams.gate_threshold,
                            max_new_tokens=2048,
                            temperature=0.6,
                            top_p=0.95,
                            top_k=20,
                            do_sample=True,
                        )
                    else:
                        generated = self.model.generate(
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            dna_tokenized=example_dna_data,
                            batch_idx_map=example_batch_map,
                            max_new_tokens=2048,
                            temperature=0.6,
                            top_p=0.95,
                            top_k=20,
                            do_sample=True,
                        )

                user_input = self.model.text_tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
                generation = self.model.text_tokenizer.decode(generated[0], skip_special_tokens=False).strip()

                ground_truth = answer[example_idx]
                if ";" in ground_truth:
                    ground_truth = ground_truth.split(";")[0]

                hit = ground_truth.lower() in generation.lower()
                total_examples += 1
                if hit:
                    correct += 1

                # Find best-matching predicted label (longest label found in generation)
                gen_lower = generation.lower()
                matched_label = max(
                    (lbl for lbl in self.labels if lbl.lower() in gen_lower),
                    key=len,
                    default=self.labels[0],
                )
                print(f"[TEST] {total_examples:4d} | {'✓' if hit else '✗'} | "
                      f"gt={ground_truth!r:40s} | pred={matched_label!r}")
                all_preds.append(label2id[matched_label])
                all_targets.append(label2id.get(ground_truth.lower(), 0))

                new_rows.append({
                    "batch_idx": batch_idx,
                    "example_idx": example_idx,
                    "user_input": user_input,
                    "generation": generation,
                    "ground_truth": ground_truth,
                    "predicted_label": matched_label,
                    "correct": hit,
                })

                if total_examples % 10 == 0:
                    wandb_logger.log({
                        "examples_processed": total_examples,
                        "running_accuracy": correct / total_examples,
                    })

                torch.cuda.empty_cache()
                gc.collect()

            # ── Flush new rows to CSV after each batch ────────────────────────
            if new_rows:
                csv_writer.writerows(new_rows)
                csv_file.flush()

            print(f"[TEST] Batch {batch_idx + 1}/{total_batches} done — "
                  f"acc so far: {correct}/{total_examples} "
                  f"({100*correct/max(total_examples,1):.1f}%)")
            wandb_logger.log({
                "batches_processed": batch_idx + 1,
                "examples_processed": total_examples,
                "progress_percentage": (batch_idx + 1) / total_batches * 100,
            })

        csv_file.close()

        # Rename incremental file to a timestamped final copy
        final_csv_path = os.path.join(
            csv_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-test_generations_{model_name}.csv"
        )
        os.rename(csv_path, final_csv_path)
        wandb_logger.log({"csv_saved": True, "csv_path": final_csv_path})

        # ── Final metrics ─────────────────────────────────────────────────────
        accuracy = correct / max(total_examples, 1)

        id2label = {i: lbl for lbl, i in label2id.items()}
        y_true = [id2label.get(t, "") for t in all_targets]
        y_pred = [id2label.get(p, "") for p in all_preds]
        labels = sorted(set(y_true))
        prec_mac = precision_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
        rec_mac  = recall_score(   y_true, y_pred, labels=labels, average="macro",    zero_division=0)
        f1_mac   = sk_f1(          y_true, y_pred, labels=labels, average="macro",    zero_division=0)
        prec_w   = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        rec_w    = recall_score(   y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        f1_w     = sk_f1(          y_true, y_pred, labels=labels, average="weighted", zero_division=0)

        wandb_logger.log({
            "test_accuracy": accuracy,
            "test_precision_macro": prec_mac,
            "test_recall_macro": rec_mac,
            "test_f1_macro": f1_mac,
            "test_precision_weighted": prec_w,
            "test_recall_weighted": rec_w,
            "test_f1_weighted": f1_w,
            "total_examples_processed": total_examples,
            "test_status": "completed",
        })

        # Upload full table to wandb at the end
        with open(final_csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if rows:
            data = [[r.get(c, "") for c in fieldnames] for r in rows]
            wandb_logger.log({
                f"test_generations_{time.strftime('%Y%m%d-%H%M%S')}": wandb.Table(columns=fieldnames, data=data)
            })

        summary = (
            f"Test Results Summary:\n"
            f"Total examples : {total_examples}\n"
            f"Classes        : {len(labels)}\n"
            f"Accuracy       : {accuracy:.4f}\n"
            f"Precision      : {prec_mac:.4f}  (macro)   {prec_w:.4f}  (weighted)\n"
            f"Recall         : {rec_mac:.4f}  (macro)   {rec_w:.4f}  (weighted)\n"
            f"F1             : {f1_mac:.4f}  (macro)   {f1_w:.4f}  (weighted)\n"
            f"CSV            : {final_csv_path}"
        )
        print(summary)
        wandb_logger.log({"test_summary": summary})

        torch.cuda.empty_cache()
        gc.collect()

        return {"test_accuracy": accuracy,
                "test_precision_macro": prec_mac, "test_precision_weighted": prec_w,
                "test_recall_macro": rec_mac,    "test_recall_weighted": rec_w,
                "test_f1_macro": f1_mac,         "test_f1_weighted": f1_w}


def main(args: ArgumentParser):
    """
    Main function to run the DNA-Text fine-tuning process.

    Args:
        args (ArgumentParser): Parsed command-line arguments
    """
    # Set random seed and environment variables
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Setup directories
    run_name = f"{args.wandb_project}-{args.dataset_type}-{args.text_model_name.split('/')[-1]}"
    if args.ckpt_path and not args.test_only:
        # Resuming training: reuse the existing checkpoint directory so new
        # checkpoints sit alongside old ones (avoids nested timestamped folders)
        args.checkpoint_dir = os.path.dirname(args.ckpt_path)
    else:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        flag_file = os.path.join(args.checkpoint_dir, ".active_run")
        if os.path.exists(flag_file):
            with open(flag_file) as f:
                args.checkpoint_dir = f.read().strip()
        else:
            args.checkpoint_dir = f"{args.checkpoint_dir}/{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            with open(flag_file, 'w') as f:
                f.write(args.checkpoint_dir)

    # Initialize model
    model = DNALLMFineTuner(args)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{run_name}-" + "{epoch:02d}-{val_loss_epoch:.4f}",
            save_top_k=2,
            monitor="val_loss_epoch",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Setup logger
    is_resuming = args.ckpt_path is not None
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
        resume="allow" if is_resuming else None,  # Allow resuming existing run
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=(
            DeepSpeedStrategy(stage=2, offload_optimizer=False, allgather_bucket_size=5e8, reduce_bucket_size=5e8)
            if args.strategy == "deepspeed_stage_2"
            else args.strategy
        ),
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        val_check_interval=1 / 3,
        limit_train_batches=args.limit_train_batches,
    )

    # Start the training process (skipped when --test_only is set)
    if not args.test_only:
        trainer.fit(model, ckpt_path=args.ckpt_path)

    # Resolve the checkpoint to test on:
    #   - test_only: use the explicitly provided --ckpt_path
    #   - after training: use the best checkpoint saved by ModelCheckpoint
    if args.test_only:
        test_ckpt = args.ckpt_path
    else:
        checkpoint_callback = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
        test_ckpt = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path

    # DeepSpeed saves checkpoints as directories containing checkpoint files.
    # If the resolved path is a directory, look for the actual weights file inside.
    if test_ckpt and os.path.isdir(test_ckpt):
        candidates = (
            glob.glob(os.path.join(test_ckpt, "*.ckpt")) +
            glob.glob(os.path.join(test_ckpt, "checkpoint", "mp_rank_00_model_states.pt"))
        )
        if candidates:
            test_ckpt = candidates[0]
        else:
            raise FileNotFoundError(
                f"Checkpoint directory {test_ckpt} contains no recognisable checkpoint file."
            )

    print(f"\n[TEST] Using checkpoint: {test_ckpt}\n")

    # Test on rank 0 only — both DDP processes reach this point but running
    # test_trainer on both causes a CSV rename race condition.
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        test_trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        test_trainer.test(model, ckpt_path=test_ckpt)

        flag_file = os.path.join(os.path.dirname(args.checkpoint_dir), ".active_run")
        if os.path.exists(flag_file):
            os.remove(flag_file)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model configuration
    parser.add_argument("--model_type", type=str, choices=["llm", "dna-llm"], default="dna-llm")
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dna_model_name", type=str, default=None,
                        help="DNA encoder model name/path. Leave empty for LLM-only mode (no DNA model loaded).")
    parser.add_argument("--text_model_finetune", type=bool, default=True)
    parser.add_argument("--dna_model_finetune", type=bool, default=False)
    parser.add_argument("--dna_is_evo2", type=bool, default=False)
    parser.add_argument("--dna_embedding_layer", type=str, default=None)
    parser.add_argument("--use_cross_attention", type=bool, default=False,
                        help="Use CrossAttentionFusion instead of plain linear projection.")
    parser.add_argument("--use_hrpo_gate", action="store_true", default=False,
                        help="Enable HRPO adaptive gate (Week 3). Replaces fixed Coconut loop.")
    parser.add_argument("--max_latent_steps", type=int, default=8,
                        help="Max latent steps for HRPO gate (safety ceiling).")
    parser.add_argument("--gate_threshold", type=float, default=0.5,
                        help="Gate value below which HRPO loop stops early.")
    parser.add_argument("--gate_reg_weight", type=float, default=0.01,
                        help="Weight for gate regularization loss L_gate.")
    parser.add_argument("--use_dna_gate", action="store_true", default=False,
                        help="Week 4: condition GateNet on mean-pooled DNA encoder output u_dna.")
    parser.add_argument("--hrpo_warmup_steps", type=int, default=0,
                        help="Steps of pure SFT before HRPO gate loop is used in generation display. "
                             "Gate loss still trains throughout. Default 0 = HRPO from step 1.")
    parser.add_argument("--hrpo_warmup_epochs", type=float, default=0,
                        help="Epochs of pure SFT before HRPO gate loop (alternative to --hrpo_warmup_steps). "
                             "Converted to steps at training start. Overrides --hrpo_warmup_steps if > 0.")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="Fraction or count of batches per epoch. Use small int (e.g. 20) to test checkpoint saving quickly.")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length_dna", type=int, default=1024)
    parser.add_argument("--max_length_text", type=int, default=2048)
    parser.add_argument("--truncate_dna_per_side", type=int, default=1024)
    parser.add_argument("--return_answer_in_batch", type=bool, default=False)
    
    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Infrastructure and paths
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--cache_dir", type=str, default="/model-weights")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--test_only", action="store_true", default=False,
                        help="Skip training and run test only. Requires --ckpt_path.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")
    
    # Dataset configuration
    parser.add_argument("--dataset_type", type=str, choices=["kegg", "variant_effect_coding", "variant_effect_non_snv"], default="kegg")
    parser.add_argument("--use_qwen_dna_collate_fn", type=bool, default=True)
    parser.add_argument("--kegg_data_dir_local", type=str, default="data/kegg")
    parser.add_argument("--kegg_data_dir_huggingface", type=str, default="wanglab/kegg")
    parser.add_argument("--variant_effect_coding_data_dir_huggingface", type=str, default="wanglab/variant_effect_coding")
    parser.add_argument("--variant_effect_non_snv_data_dir_huggingface", type=str, default="wanglab/variant_effect_non_snv")
    parser.add_argument("--merge_val_test_set", type=bool, default=False)
    parser.add_argument("--kegg_csv", type=str, default=None,
                        metavar="CSV_PATH",
                        help="Path to a pre-built KEGG CSV (from build_curriculum.py or "
                             "build_anon_dataset.py). Omit to load directly from HuggingFace.")

    # CLIP loss (Step 1)
    parser.add_argument("--max_clip_loss_weight", type=float, default=0.0,
                        help="Max weight of L_CLIP loss. 0.0 = disabled. Typical: 0.05-0.2.")
    parser.add_argument("--clip_temperature", type=float, default=0.07,
                        help="Temperature for the CLIP contrastive loss.")
    parser.add_argument("--clip_queue_size", type=int, default=64,
                        help="Size of MoCo-style memory queue for CLIP negatives. "
                             "Larger = more negatives = stronger contrastive signal.")
    # OT loss (Week 5 HiRef)
    parser.add_argument("--ot_weight", type=float, default=0.0,
                        help="Weight of L_OT Sinkhorn loss aligning DNA latents to answer manifold. "
                             "0.0 = disabled. Typical: 0.05-0.1.")
    parser.add_argument("--ot_n_iter", type=int, default=50,
                        help="Number of Sinkhorn iterations.")

    # Logging and monitoring
    parser.add_argument("--wandb_project", type=str, default="nt-500m-qwen3-1.7b-finetune")
    parser.add_argument("--wandb_entity", type=str)
    
    args = parser.parse_args()

    main(args)