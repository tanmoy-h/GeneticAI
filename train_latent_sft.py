"""
Stage 1.5 — LatentSp Cold-Start SFT.

Two entropy modes, selected via --entropy_mode:

  global  (default)
      At the start of each curriculum step s, run one full-dataset forward pass
      (torch.no_grad) to record per-step entropy for every training sample.
      All batches within that step use the same fixed assignments.
      Cost: one extra epoch-level forward pass per curriculum step.

  inline
      Inside every training batch:
        1. forward (no_grad) → entropy → select s lowest-entropy step contents
        2. inject <start-latent> [latent]*c <end-latent> markers
        3. forward (with_grad) → masked loss → backward
      Matches LatentSp Algorithm 1 exactly.  ~2x compute per batch.

Progressive curriculum (matches LatentSp Algorithm 1):

  outer loop  s = 1 .. max_latent_steps
      Each s is one full pass over the training set.
      s steps are replaced per sample.
      New data: 10% of training set is re-shuffled into DSFT each outer step
               (controlled by --data_refresh_ratio, default 0.1).

Loss details (from LatentSp paper, Section 4):
  - Latent content tokens: masked (label = -100)
  - <start-latent> and <end-latent> boundary tokens: loss × 4.0
  - All other tokens: loss × 1.0

Usage
-----
  python train_latent_sft.py \\
      --stage1_ckpt   checkpoints/.../epoch=03-val_loss=0.4292.ckpt \\
      --train_csv     genomorph/dataset/kegg_curriculum/global_stage1_anon_genes_mol_keep_chr.csv \\
      --output_dir    /scratch/tanmoyh_iitp/GenoMorph/checkpoints/week9tests/stage1_5_sft \\
      --entropy_mode  global \\
      --max_latent_steps 4

  # Inline mode (slower, matches LatentSp exactly):
  python train_latent_sft.py \\
      --entropy_mode inline ...

Pipeline
--------
  Stage 1 SFT  →  [this script]  →  Stage 2 HiRef / Stage 3 GRPO w9
"""

import os
import re
import argparse
import random
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from datasets import load_dataset
from genomorph.models.dna_llm import DNALLMModel
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from genomorph.dataset.utils import truncate_dna

# Gate modules — imported lazily so train_latent_sft.py works without w9 when --use_gate=False
_GATE_AVAILABLE = False
try:
    from train_grpo_latent_reasoning import (
        ThinkingResidualGate,
        DNAHiddenInjector,
        _make_dual_mode_forward_w9,
        MAX_GATE_FACTOR,
    )
    import types as _types
    _GATE_AVAILABLE = True
except ImportError:
    pass

register_evo2_tokenizer()

LATENT_START        = "<start-latent>"
LATENT_END          = "<end-latent>"
LATENT_PAD          = "<latent>"       # placeholder between markers
BOUNDARY_LOSS_SCALE = 1.0              # 4x caused <start-latent> hallucination at inference


def _compact_dna(text: str) -> str:
    """Collapse consecutive <|dna_pad|>×N → <|dna_pad| ×N>."""
    _tok = '<|dna_pad|>'
    def _repl(m):
        n = len(m.group(0)) // len(_tok)
        return f'<|dna_pad| ×{n}>'
    return re.sub(r'(?:<\|dna_pad\|>){2,}', _repl, text)


# ── Special token setup ───────────────────────────────────────────────────────

def ensure_latent_tokens(tokenizer, model) -> Tuple[int, int, int]:
    """Add latent special tokens if absent; resize model embeddings."""
    new_tokens = []
    for tok in (LATENT_START, LATENT_END, LATENT_PAD):
        if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
            new_tokens.append(tok)
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.text_model.resize_token_embeddings(len(tokenizer))
        print(f"[Stage1.5] Added tokens: {new_tokens}")

    start_id  = tokenizer.convert_tokens_to_ids(LATENT_START)
    end_id    = tokenizer.convert_tokens_to_ids(LATENT_END)
    latent_id = tokenizer.convert_tokens_to_ids(LATENT_PAD)
    print(f"[Stage1.5] {LATENT_START}={start_id}  {LATENT_END}={end_id}  "
          f"{LATENT_PAD}={latent_id}")
    return start_id, end_id, latent_id


def load_start_state_dict_from_dir(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """Load a backbone state_dict from an HF-Trainer/GRPO checkpoint DIRECTORY.

    Mirrors eval_grpo_checkpoint_final.py::_load_llm_weights so RFT can start from a
    train_06b GRPO checkpoint (backbone already w9-native). Priority:
      1. pytorch_model.bin          (train_06b sets save_safetensors=False)
      2. model.safetensors          (HF Trainer default)
      3. model.pt                   (manual torch.save)
      4/5. sharded safetensors / .bin
    Returns the raw state_dict (caller strips any leading 'model.' prefix).
    """
    import glob as _glob

    p = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(p):
        print(f"[Stage1.5] Start weights ← {p}")
        return torch.load(p, map_location="cpu", weights_only=True)

    p = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(p):
        try:
            from safetensors.torch import load_file as _st_load
            print(f"[Stage1.5] Start weights ← {p}")
            return _st_load(p, device="cpu")
        except ImportError:
            print(f"[Stage1.5] Start weights ← {p} (torch.load fallback)")
            return torch.load(p, map_location="cpu", weights_only=True)

    p = os.path.join(ckpt_dir, "model.pt")
    if os.path.exists(p):
        print(f"[Stage1.5] Start weights ← {p}")
        return torch.load(p, map_location="cpu", weights_only=True)

    for pat in ("model-*-of-*.safetensors", "pytorch_model-*-of-*.safetensors"):
        shards = sorted(_glob.glob(os.path.join(ckpt_dir, pat)))
        if shards:
            from safetensors.torch import load_file as _st_load
            merged: Dict[str, torch.Tensor] = {}
            for s in shards:
                merged.update(_st_load(s, device="cpu"))
            print(f"[Stage1.5] Start weights ← {len(shards)} safetensors shards in {ckpt_dir}")
            return merged

    shards = sorted(_glob.glob(os.path.join(ckpt_dir, "pytorch_model-*-of-*.bin")))
    if shards:
        merged = {}
        for s in shards:
            merged.update(torch.load(s, map_location="cpu", weights_only=True))
        print(f"[Stage1.5] Start weights ← {len(shards)} .bin shards in {ckpt_dir}")
        return merged

    raise FileNotFoundError(
        f"No backbone weights (pytorch_model.bin / model.safetensors / model.pt / shards) "
        f"found in checkpoint dir: {ckpt_dir}")


def merge_grpo_lora_state_dict(
    state: Dict[str, torch.Tensor], lora_r: int = 16, lora_alpha: int = 32,
) -> Dict[str, torch.Tensor]:
    """train_06b's _prep_for_training wraps model.text_model in PEFT LoRA
    (adaptive_latent_grpo.py:306, get_peft_model) before training, so a GRPO
    checkpoint's pytorch_model.bin has PEFT-prefixed keys
    ("text_model.base_model.model.model...." + lora_A/lora_B/base_layer), not the
    plain "text_model.model...." names a non-PEFT DNALLMModel.state_dict() expects.
    Loading those keys strict=False into a plain model SILENTLY DROPS the entire
    LoRA fine-tuning (all of it lands in "unexpected", base weights stay at
    untouched pretrained values in "missing") — the checkpoint's actual GRPO
    adaptation never reaches the model. This merges lora_B @ lora_A * (alpha/r)
    into each base weight and strips the PEFT prefixes, mirroring
    adaptive_latent_grpo.py::_load_sft_checkpoint's raw-.bin branch (merge_lora=True)
    exactly so RFT starts from the ACTUAL GRPO-adapted weights.
    No-op (returns state unchanged) if no PEFT-prefixed keys are present."""
    peft_inner = "text_model.base_model.model.model."
    peft_outer = "text_model.base_model.model."
    if not any(k.startswith(peft_inner) for k in state):
        return state   # not a PEFT checkpoint — nothing to merge

    _LORA_TAGS = ("lora_A", "lora_B", "lora_embedding", "lora_magnitude")
    remapped: Dict[str, torch.Tensor] = {}
    lora_a_map: Dict[str, torch.Tensor] = {}
    lora_b_map: Dict[str, torch.Tensor] = {}

    for k, v in state.items():
        if k.startswith(peft_inner):
            stripped = k[len(peft_inner):]
            new_k    = "text_model.model." + stripped
            if any(tag in stripped for tag in _LORA_TAGS):
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

    scaling  = lora_alpha / lora_r
    merged_n = 0
    for mod_key, A in lora_a_map.items():
        if mod_key not in lora_b_map:
            continue
        B        = lora_b_map[mod_key]
        full_key = "text_model.model." + mod_key + ".weight"
        if full_key not in remapped:
            continue
        remapped[full_key] = remapped[full_key] + \
            (B.to(remapped[full_key].dtype) @ A.to(remapped[full_key].dtype)) * scaling
        merged_n += 1
    print(f"  [LoRA merge] Detected PEFT structure — merged {merged_n} LoRA adapters "
          f"(alpha={lora_alpha}, r={lora_r}, scale={scaling:.3f}).")
    return remapped


# ── Step span detection ───────────────────────────────────────────────────────

def find_step_content_spans(
    input_ids: List[int],
    tokenizer,
) -> List[Tuple[int, int, int]]:
    """
    Find the token span of each step's CONTENT (after 'Step N:' header).

    Keeps the 'Step N:' header visible — only the content gets replaced.
    Spans are within the <think>...</think> block only.

    Returns:
        List of (step_number, content_tok_start, content_tok_end) tuples.
        content_tok_end is exclusive.
    """
    text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Locate <think> block
    think_start = text.find("<think>")
    think_end   = text.find("</think>")
    if think_start == -1 or think_end == -1:
        return []

    # If "Answer:" appears inside <think> (e.g. reasoning field embeds it),
    # cap think_end so no span can reach or cross "Answer:".
    answer_in_think = text.find("Answer:", think_start, think_end)
    if answer_in_think != -1:
        think_end = answer_in_think

    think_text = text[think_start:think_end]
    offset     = think_start

    # Find all "Step N:" headers inside <think>
    step_re = re.compile(r'Step\s+(\d+)\s*:')
    matches = list(step_re.finditer(think_text))
    if not matches:
        return []

    # Build char offset mapping: token index → (char_start, char_end)
    enc     = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]  # List[(char_start, char_end)]

    def char_to_tok(char_pos: int, after: bool = False) -> int:
        """Find token index whose span covers char_pos."""
        for idx, (cs, ce) in enumerate(offsets):
            if after:
                if cs >= char_pos:
                    return idx
            else:
                if cs <= char_pos < ce:
                    return idx
        return len(offsets)

    spans = []
    for i, m in enumerate(matches):
        step_num = int(m.group(1))

        # Content starts after the colon; skip the single space separator so
        # "Step N:" stays visible in the token stream (not absorbed into the span).
        content_char_start = offset + m.end()
        while content_char_start < think_end and text[content_char_start] in (' ', '\t'):
            content_char_start += 1

        # Content ends at next step header or </think>
        if i + 1 < len(matches):
            content_char_end = offset + matches[i + 1].start()
        else:
            content_char_end = think_end   # char position of </think> in full text

        # Trim only horizontal whitespace so the trailing \n before the next
        # "Step N+1:" header is preserved in the token stream after <end-latent>.
        content_text = text[content_char_start:content_char_end]
        stripped_len = len(content_text.rstrip(' \t'))
        content_char_end = content_char_start + stripped_len

        if content_char_end <= content_char_start:
            continue

        t_start = char_to_tok(content_char_start, after=True)
        t_end   = char_to_tok(content_char_end,   after=True)

        if t_end > t_start:
            spans.append((step_num, t_start, t_end))

    return spans  # [(step_num, tok_start, tok_end_exclusive), ...]


# ── Entropy computation ───────────────────────────────────────────────────────

def compute_step_entropies(
    logits:     torch.Tensor,   # [T, V]  teacher-forced logits for one sample
    step_spans: List[Tuple[int, int, int]],
) -> List[float]:
    """
    Mean Shannon entropy (nats) per step content span.

    logits[t] predicts token[t+1], so entropy for content tokens [t_start..t_end)
    is taken from logits[t_start-1 .. t_end-1].
    """
    probs   = torch.softmax(logits.float(), dim=-1)           # [T, V]
    token_H = -(probs * (probs + 1e-10).log()).sum(-1)        # [T]

    entropies = []
    for (_, t_start, t_end) in step_spans:
        lo = max(t_start - 1, 0)
        hi = max(t_end   - 1, 0)
        if hi > lo:
            entropies.append(token_H[lo:hi].mean().item())
        else:
            entropies.append(float("inf"))   # skip: infinite entropy = never replace
    return entropies


# ── Latent marker injection ───────────────────────────────────────────────────

def real_content_end(ids: List[int], pad_id: int) -> int:
    """
    Index one past the last real (non-padding) token in a right-padded sequence.

    pad_token_id == eos_token_id == <|im_end|> (151645) in this project, so a
    trailing run of pad_id is padding EXCEPT its first element, which is the
    answer's genuine <|im_end|> terminator (`Answer: X<|im_end|>`). Masking by
    `tok == pad_id` therefore also masks that terminator, so the model never
    learns to stop after the answer. Find the last token that isn't pad_id; the
    real terminator sits exactly one position after it and must be KEPT in loss.
    """
    last_non_pad = -1
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] != pad_id:
            last_non_pad = i
            break
    if last_non_pad == -1:
        return 0  # all padding (should not happen for a real sample)
    return min(last_non_pad + 2, len(ids))


def real_attention_mask(batch_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    [B, T] attention mask that keeps every real token INCLUDING the answer's
    <|im_end|> terminator and masks only trailing padding. Vectorized counterpart
    of real_content_end: since pad_id == <|im_end|>, a plain `batch_ids != pad_id`
    mask wrongly drops the terminator.
    """
    B, T        = batch_ids.shape
    idx         = torch.arange(T, device=batch_ids.device).unsqueeze(0)   # [1, T]
    non_pad     = (batch_ids != pad_id).long()                            # [B, T]
    last_real   = (non_pad * (idx + 1)).max(dim=1).values                 # [B] = last_non_pad_idx + 1
    content_end = (last_real + 1).clamp(max=T).unsqueeze(1)               # keep the terminator
    return (idx < content_end).long()


def inject_latent_markers(
    input_ids:       List[int],
    step_spans:      List[Tuple[int, int, int]],
    replace_indices: List[int],
    start_id:        int,
    end_id:          int,
    latent_id:       int,
    pad_id:          int,
    prompt_end:      int = 0,
    protected_ids:   frozenset = frozenset(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replace the CONTENT of selected steps with latent marker spans.

    Replacement: [start_id] [latent_id]*(c-2) [end_id]
    where c = original content length, preserving sequence length.
    Minimum replacement: [start_id] [end_id] (for content_len == 1 or 2).

    protected_ids: token IDs that must never be replaced (e.g. </think>, Answer:).
    Any span containing a protected token is skipped entirely.

    Returns:
        new_ids      [T]   input_ids with latent markers substituted
        labels       [T]   -100 at latent content, scaled elsewhere
        loss_weights [T]   4.0 at boundary tokens, 1.0 elsewhere, 0.0 at -100
    """
    ids = list(input_ids)
    T   = len(ids)

    labels       = list(ids)          # start as copy of input_ids
    loss_weights = [1.0] * T

    # Mask user-prompt tokens from loss. prompt_end marks the end of the user
    # turn (<|im_start|>user ... <|im_end|>\n). Everything from prompt_end
    # onward is the assistant section and must receive loss.
    # Do NOT mask <|im_start|>assistant, <think>, or Step N: headers — those
    # are part of what the model must learn to generate.
    for t in range(min(prompt_end, T)):
        labels[t]       = -100
        loss_weights[t] = 0.0

    # Sort replace_indices by token start (descending) to splice without offset shift
    to_replace = sorted(replace_indices, key=lambda i: step_spans[i][1], reverse=True)

    for idx in to_replace:
        _, t_start, t_end = step_spans[idx]
        c = t_end - t_start          # original content length in tokens

        if c <= 0:
            continue

        # Never replace a span that contains a protected token (</think>, Answer:, etc.)
        if protected_ids and any(ids[t] in protected_ids for t in range(t_start, min(t_end, len(ids)))):
            continue
        elif c == 1:
            replacement = [start_id, end_id]
        elif c == 2:
            replacement = [start_id, end_id]
        else:
            replacement = [start_id] + [latent_id] * (c - 2) + [end_id]

        # Splice into ids (length preserved when c >= 2, +1 when c == 1)
        ids[t_start:t_end] = replacement

        # Labels: -100 for latent_id positions, boundary tokens kept
        for j, rep_tok in enumerate(replacement):
            abs_j = t_start + j
            if abs_j >= len(labels):
                break
            if rep_tok == latent_id:
                labels[abs_j]       = -100
                loss_weights[abs_j] = 0.0
            elif rep_tok in (start_id, end_id):
                labels[abs_j]       = rep_tok
                loss_weights[abs_j] = BOUNDARY_LOSS_SCALE

    # Mask trailing padding ONLY — by position, not by `tok == pad_id`, because
    # pad_id == <|im_end|> so an id comparison also masks the answer terminator
    # and the model never learns to stop. Keep everything up to the real
    # <|im_end|> (computed from the unmodified input; latent replacement never
    # touches the tail), mask what follows.
    content_end = real_content_end(list(input_ids), pad_id)
    for t in range(content_end, T):
        labels[t]       = -100
        loss_weights[t] = 0.0

    new_ids      = torch.tensor(ids,          dtype=torch.long)
    labels_t     = torch.tensor(labels,       dtype=torch.long)
    weights_t    = torch.tensor(loss_weights, dtype=torch.float)
    return new_ids, labels_t, weights_t


def label_self_adaptive_latents(
    input_ids:       List[int],
    prompt_end:      int,
    pad_id:          int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Label an RFT trace that ALREADY contains latent markers, teacher-forcing the
    ENTIRE completion — reasoning text, <start-latent>/<latent>/<end-latent>, and the
    Answer line — so the model learns to EMIT the latent block itself (self-adaptive).

    Contrast with inject_latent_markers, which (a) INSERTS latent markers by an entropy
    curriculum and (b) masks the <latent> content to -100. Here the markers are already
    present (the GRPO controller placed them when the trace was sampled), and we do NOT
    mask them — the model must learn to predict <start-latent>, the <latent> block, and
    <end-latent> at the right positions. Only the user prompt and trailing padding are
    masked.

    Boundary/latent weight is kept at BOUNDARY_LOSS_SCALE (=1.0). Do NOT boost it — 4x
    previously caused <start-latent> hallucination at inference (see module note at
    BOUNDARY_LOSS_SCALE). Correct placement is taught by the DATA (only correct+short
    traces survive the RFT filter), not by up-weighting the tokens.

    Returns (input_ids_tensor, labels, loss_weights):
      - prompt tokens (< prompt_end)          -> label -100, weight 0
      - trailing padding (after real content) -> label -100, weight 0
      - everything else                       -> teacher-forced, weight 1.0
    """
    ids = list(input_ids)
    T   = len(ids)
    labels       = list(ids)
    loss_weights = [1.0] * T

    # Mask the user prompt (assistant section from prompt_end onward gets loss).
    for t in range(min(prompt_end, T)):
        labels[t]       = -100
        loss_weights[t] = 0.0

    # Mask trailing padding by POSITION (pad_id == <|im_end|>, so an id-compare would
    # also mask the answer's genuine terminator). Keep everything up to real_content_end.
    content_end = real_content_end(list(input_ids), pad_id)
    for t in range(content_end, T):
        labels[t]       = -100
        loss_weights[t] = 0.0

    return (torch.tensor(ids,          dtype=torch.long),
            torch.tensor(labels,       dtype=torch.long),
            torch.tensor(loss_weights, dtype=torch.float))


# ── Weighted SFT loss ─────────────────────────────────────────────────────────

def weighted_sft_loss(
    logits:  torch.Tensor,   # [B, T, V]
    labels:  torch.Tensor,   # [B, T]
    weights: torch.Tensor,   # [B, T]
) -> torch.Tensor:
    """
    Cross-entropy with per-token weights.
    -100 labels are ignored (weight should also be 0 there).
    """
    B, T, V = logits.shape
    shift_logits  = logits[:, :-1, :].contiguous().view(-1, V)
    shift_labels  = labels[:, 1:].contiguous().view(-1)
    shift_weights = weights[:, 1:].contiguous().view(-1)

    # Per-token loss (unreduced)
    per_token = F.cross_entropy(shift_logits, shift_labels,
                                ignore_index=-100, reduction="none")  # [B*(T-1)]

    # Apply weights; guard against all-zero (empty batch)
    weighted = per_token * shift_weights
    denom    = shift_weights[shift_labels != -100].sum().clamp(min=1e-8)
    return weighted.sum() / denom


# ── Dataset ───────────────────────────────────────────────────────────────────

def _make_row(question: str, reasoning: str, answer: str,
              reference_sequence: str = "", variant_sequence: str = "") -> dict:
    """
    Build one training row with a complete ChatML conversation + DNA context.

    Two <|dna_pad|> markers in the user turn will be expanded by DLProcessor
    into the full Evo2 token sequence for the reference and variant DNA.
    Keeping the user turn mirrors Stage 1 and Stage 3 GRPO exactly, so the
    model's learned rule "answer → <|im_end|>" is never disrupted.
    """
    user_text = f"<|im_start|>user\n<|dna_pad|>\n<|dna_pad|>\n{question.strip()}\n<|im_end|>\n"
    asst_text = f"<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n"
    return {
        "user_text":     user_text,
        "text":          user_text + asst_text,
        "answer":        f"Answer: {answer}<|im_end|>",
        "dna_sequences": [reference_sequence, variant_sequence],
    }


def _trunc(ex: dict, per_side: int) -> tuple:
    if per_side > 0:
        ex = truncate_dna(ex, truncate_dna_per_side=per_side)
    return ex["reference_sequence"], ex["variant_sequence"]


def load_kegg_csv(csv_path: str, truncate_dna_per_side: int = 1024):
    """Load KEGG from a local anonymized CSV and return (train_rows, val_rows)."""
    from genomorph.dataset.kegg import load_kegg_from_anon_csv
    ds = load_kegg_from_anon_csv(csv_path)
    train_rows = [_make_row(ex["question"], ex["reasoning"], ex["answer"],
                            *_trunc(ex, truncate_dna_per_side)) for ex in ds["train"]]
    val_rows   = [_make_row(ex["question"], ex["reasoning"], ex["answer"],
                            *_trunc(ex, truncate_dna_per_side)) for ex in ds["val"]]
    return train_rows, val_rows


def load_kegg_hf(dataset_name: str, cache_dir=None, truncate_dna_per_side: int = 1024):
    """Load KEGG from HuggingFace and return (train_rows, val_rows)."""
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    train_rows = [_make_row(ex["question"], ex["reasoning"], ex["answer"],
                            *_trunc(ex, truncate_dna_per_side)) for ex in ds["train"]]
    val_rows   = [_make_row(ex["question"], ex["reasoning"], ex["answer"],
                            *_trunc(ex, truncate_dna_per_side)) for ex in ds["val"]]
    return train_rows, val_rows


def load_eval_rows(dataset_name: str = None, kegg_csv: str = None,
                   cache_dir=None, truncate_dna_per_side: int = 1024) -> List[dict]:
    """Load the combined val+test eval set (290 for wanglab/kegg: val 144 + test 146)
    for the in-training accuracy probe. Falls back to whatever of val/test exists."""
    if kegg_csv:
        from genomorph.dataset.kegg import load_kegg_from_anon_csv
        ds = load_kegg_from_anon_csv(kegg_csv)
    else:
        ds = load_dataset(dataset_name, cache_dir=cache_dir)
    rows: List[dict] = []
    for split in ("val", "validation", "test"):
        if split in ds:
            rows += [_make_row(ex["question"], ex["reasoning"], ex["answer"],
                               *_trunc(ex, truncate_dna_per_side)) for ex in ds[split]]
    return rows


def _make_rft_row(question: str, completion: str,
                  reference_sequence: str = "", variant_sequence: str = "") -> dict:
    """Build one self-adaptive RFT row: the assistant content is the SAMPLED completion
    (reasoning with latent markers + </think> + Answer), not the ground-truth reasoning.

    We reconstruct: user turn + "<|im_start|>assistant\\n<think>\\n" as `text` (the prompt),
    and the completion as `answer`. KeggRawDataset tokenises text+answer; prompt_end (via
    find_assistant_start) masks the user turn, and label_self_adaptive_latents teacher-forces
    the rest INCLUDING the latent markers.

    NOTE: the sampled/gold completion ALSO begins with "<think>\\n" (the GRPO sampler
    re-emitted <think> after the prefill; gold traces are written with it). Since asst_prefix
    already supplies <think>, we STRIP a leading <think> from the completion to avoid a
    DOUBLE "<think>\\n<think>" in the joined sequence.
    """
    completion = completion.rstrip()
    _c = completion.lstrip()
    if _c.startswith("<think>"):                       # drop the redundant leading <think>
        completion = _c[len("<think>"):].lstrip("\n")
    if not completion.endswith("<|im_end|>"):
        completion += "<|im_end|>"
    user_text   = f"<|im_start|>user\n<|dna_pad|>\n<|dna_pad|>\n{question.strip()}\n<|im_end|>\n"
    asst_prefix = "<|im_start|>assistant\n<think>\n"
    return {
        "user_text":     user_text,
        "text":          user_text + asst_prefix,
        "answer":        completion,
        "dna_sequences": [reference_sequence, variant_sequence],
    }


def load_rft_rows(traces_jsonl: str, dataset_name: str = None, kegg_csv: str = None,
                  cache_dir=None, truncate_dna_per_side: int = 1024) -> List[dict]:
    """Build RFT training rows by joining selected completions (by index) with the train
    split (for question + DNA). The traces' `index` must match the train-split order the
    sampler used (eval_grpo_checkpoint_final.py --split train --n_samples -1). Uses the
    "default" config to match that ordering.
    """
    import json as _json
    sel: Dict[int, str] = {}
    with open(traces_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _json.loads(line)
            sel[int(r["index"])] = r["completion"]

    if kegg_csv:
        from genomorph.dataset.kegg import load_kegg_from_anon_csv
        ds_train = load_kegg_from_anon_csv(kegg_csv)["train"]
    else:
        ds_train = load_dataset(dataset_name, "default", cache_dir=cache_dir)["train"]

    rows: List[dict] = []
    for i, ex in enumerate(ds_train):
        if i not in sel:
            continue
        ref, var = _trunc(ex, truncate_dna_per_side)
        rows.append(_make_rft_row(ex["question"], sel[i], ref, var))
    print(f"[RFT] joined {len(rows)} completions with train "
          f"({len(sel)} selected / {len(ds_train)} train samples)")
    return rows


def find_assistant_start(input_ids: torch.Tensor, tokenizer) -> int:
    """Return the token index just after '<|im_start|>assistant\\n'; 0 if not found."""
    marker_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    marker_t   = torch.tensor(marker_ids, dtype=input_ids.dtype)
    marker_len = len(marker_ids)
    L = input_ids.shape[0]
    for i in range(L - marker_len + 1):
        if torch.all(input_ids[i:i + marker_len] == marker_t):
            return i + marker_len
    return 0


class KeggRawDataset(Dataset):
    """
    Tokenises full ChatML conversations with DNA context from KEGG rows.
    Uses DLProcessor so <|dna_pad|> placeholders are expanded with Evo2 tokens.
    Returns (input_ids, dna_tokenized, batch_idx_map, prompt_end) per sample.
    prompt_end marks the start of the assistant turn; tokens before it are
    masked from loss so only the assistant section receives gradient.
    """

    def __init__(self, rows: List[dict], processor, max_length_text: int, max_length_dna: int):
        self.samples: List[Tuple] = []
        tokenizer = processor.tokenizer

        for row in rows:
            text          = row.get("text", "")
            answer        = row.get("answer", "")
            dna_sequences = row.get("dna_sequences", ["", ""])
            full          = text + answer

            batch = processor(
                text                = [full],
                batch_dna_sequences = [dna_sequences],
                return_tensors      = "pt",
                padding             = False,
                add_special_tokens  = False,
                max_length_text     = max_length_text,
                max_length_dna      = max_length_dna,
            )

            ids = batch["input_ids"][0]
            if ids.shape[0] <= 4:
                continue

            # Find where the assistant turn starts (after user + DNA pads are expanded)
            prompt_end = find_assistant_start(ids, tokenizer)

            dna_tok = batch["dna_tokenized"]          # {input_ids: [2,L], attention_mask: [2,L]}
            idx_map = list(batch["batch_idx_map"])    # [0, 0] for a single-sample batch

            self.samples.append((ids, dna_tok, idx_map, prompt_end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (idx,) + self.samples[idx]


def log_latent_sample(
    tokenizer,
    orig_ids: torch.Tensor,   # [T] — original token ids for one sample
    new_ids:  torch.Tensor,   # [T] — modified token ids after latent injection
    s: int,
):
    """Decode and print full original and modified training sequence to verify latent injection."""
    pad_id = tokenizer.pad_token_id or 0

    def _strip_pad(ids):
        lst = ids.tolist()
        while lst and lst[-1] == pad_id:
            lst = lst[:-1]
        return lst

    orig_dec = _compact_dna(tokenizer.decode(_strip_pad(orig_ids), skip_special_tokens=False))
    new_dec  = _compact_dna(tokenizer.decode(_strip_pad(new_ids),  skip_special_tokens=False))

    n_start = new_dec.count("<start-latent>")

    print(f"\n  ── Latent sample (s={s}, replacements={n_start}) ──")
    print(f"  [ORIGINAL]\n{orig_dec}")
    print(f"\n  [MODIFIED]\n{new_dec}")
    print(f"  ────────────────────────────────────────────────────\n")


def generate_samples(
    model,
    processor,
    val_rows: List[dict],
    device: str,
    n_samples: int = 2,
    max_new_tokens: int = 800,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    label: str = "",
    suppress_latent_ids: List[int] = None,
    self_emit_half: bool = False,
) -> List[str]:
    """
    Run DNA-conditioned sampling on the first n_samples val examples (deterministic,
    not random) and print GT vs output. Prompts the model with everything up to and
    including <think>\\n, then lets it generate freely — reveals whether step structure
    and Answer: are preserved. Returns the generated strings for WandB logging.
    model.generate() uses inputs_embeds internally, so output is new tokens only.

    self_emit_half=True (RFT/self-adaptive runs only) splits the n_samples in half: the
    first ceil(n/2) ban latent tokens (fluency check — plain-text reasoning, unchanged
    behavior), the remaining floor(n/2) allow them (self-emission check — does the model
    actually fire <start-latent>/<latent>/<end-latent> when free to). Curriculum SFT
    (self_emit_half=False, the default) was never trained to self-emit, so every sample
    stays latent-free there — showing a "self-emit" half would just be untrained noise.
    """
    model.eval()
    tokenizer = processor.tokenizer
    pad_id    = tokenizer.pad_token_id or 0

    # Stop on both <|endoftext|> and <|im_end|> so Qwen chat termination works
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids  = list({id for id in [tokenizer.eos_token_id, im_end_id]
                      if id is not None and id != tokenizer.unk_token_id})

    think_tag = "<think>\n"
    indices   = list(range(min(n_samples, len(val_rows))))   # first N (deterministic, not random)
    # ceil(n/2): first half latent-free, rest self-emit (only when self_emit_half=True —
    # otherwise n_ban == len(indices) so every sample stays latent-free).
    n_ban     = -(-len(indices) // 2) if self_emit_half else len(indices)

    sep = "=" * 64
    print(f"\n{sep}\n  SAMPLE GENERATIONS  {label}\n{sep}")

    generated_texts: List[str] = []
    for i, idx in enumerate(indices):
        row           = val_rows[idx]
        full_text     = row["text"]
        gt_text       = full_text + row["answer"]
        dna_sequences = row.get("dna_sequences", ["", ""])

        cut         = full_text.find(think_tag)
        prompt_text = full_text[:cut + len(think_tag)] if cut != -1 else full_text[:60]

        batch = processor(
            text                = [prompt_text],
            batch_dna_sequences = [dna_sequences],
            return_tensors      = "pt",
            padding             = False,
            add_special_tokens  = False,
            max_length_text     = model.max_length_text,
            max_length_dna      = model.max_length_dna,
        )
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        dna_tok        = {k: v.to(device) for k, v in batch["dna_tokenized"].items()}
        idx_map        = list(batch["batch_idx_map"])

        _ban_this   = suppress_latent_ids is not None and i < n_ban
        bad_words   = [[t] for t in suppress_latent_ids] if _ban_this else None
        _mode_tag   = "latent-free" if (suppress_latent_ids is None or _ban_this) else "self-emit"
        with torch.no_grad():
            out_ids = model.generate(
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                dna_tokenized        = dna_tok,
                batch_idx_map        = idx_map,
                max_new_tokens       = max_new_tokens,
                do_sample            = True,
                temperature          = temperature,
                top_p                = top_p,
                repetition_penalty   = repetition_penalty,
                no_repeat_ngram_size = 4,
                pad_token_id         = pad_id,
                eos_token_id         = stop_ids,
                bad_words_ids        = bad_words,
            )

        # model.generate uses inputs_embeds → output contains only newly generated tokens
        generated = tokenizer.decode(out_ids[0], skip_special_tokens=False)
        generated_texts.append(generated)

        print(f"\n--- Sample {i + 1} [{_mode_tag}] ---")
        print(f"[GT ]\n{_compact_dna(gt_text)}")
        print(f"\n[GEN]\n<think>\n{generated}")  # prepend <think> (it was in the prompt)

    print(sep + "\n")
    model.train()
    return generated_texts


# ── In-training accuracy probe on val+test (290) ──────────────────────────────
# Mirrors eval_stage1_51_checkpoints.py's extract_answer / is_correct so the
# in-training number is comparable to the standalone eval.
_EVAL_EXPLANATION_DELIMS = (
    ' with ', ' due to', ' caused by', ' characterized by',
    ' resulting from', ' associated with', ' - ',
)


def _extract_answer(generated: str) -> str:
    m = re.search(r'Answer:\s*(.+?)(?:<\|im_end\|>|<\|endoftext\|>|\n|\Z)', generated)
    if not m:
        return ""
    answer = m.group(1).strip().rstrip(".,;")
    lower = answer.lower()
    for delim in _EVAL_EXPLANATION_DELIMS:
        idx = lower.find(delim)
        if idx != -1:
            answer = answer[:idx].strip().rstrip(".,;")
            lower  = answer.lower()
    return answer


def _is_correct(pred: str, gt: str) -> bool:
    if not pred:
        return False
    return gt.lower().strip() in pred.lower().strip()   # gt must appear in prediction


def _metrics_from_details(details: List[Tuple[str, str, bool]]):
    """(prec_mac, rec_mac, f1_mac, prec_w, rec_w, f1_w) from (pred, gt, correct) triples.
    Mirrors eval_stage1_51_checkpoints.py::_f1_from_details exactly (a wrong prediction is
    scored as its own predicted label so it counts as a false positive for that class)."""
    from sklearn.metrics import f1_score as sk_f1, precision_score, recall_score
    if not details:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    y_true = [gt for _, gt, _ in details]
    y_pred = [gt if ok else pred for pred, gt, ok in details]
    labels = sorted(set(y_true))
    prec_mac = float(precision_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0))
    rec_mac  = float(recall_score(   y_true, y_pred, labels=labels, average="macro",    zero_division=0))
    f1_mac   = float(sk_f1(          y_true, y_pred, labels=labels, average="macro",    zero_division=0))
    prec_w   = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    rec_w    = float(recall_score(   y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    f1_w     = float(sk_f1(          y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    return prec_mac, rec_mac, f1_mac, prec_w, rec_w, f1_w


@torch.no_grad()
def evaluate_accuracy(
    model,
    processor,
    rows: List[dict],
    device: str,
    max_new_tokens: int = 800,
    ban_latent_ids: List[int] = None,
    label: str = "",
    step: int = 0,
    dna_cache: Optional[Dict[bytes, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Greedy-decode accuracy on `rows` (the val+test 290 probe). Extracts 'Answer:' and
    scores gt-in-pred, matching eval_stage1_51_checkpoints.py. Prints the standard metric
    block (Accuracy / Precision / Recall / F1 macro+weighted / Mean time) and returns a dict.
    ban_latent_ids=None → latents allowed (self-emit, the RFT target); pass the latent ids
    to force a latent-free number instead.
    dna_cache: preloaded {input_ids.tobytes(): embedding} map (same format as
    eval_stage1_51_checkpoints.py's --dna_cache) to skip live Evo2 for cached sequences —
    scoped to this call only (model._evo2_embed is restored after, so training resumes
    unaffected; dna_model stays on GPU throughout since training needs it live right after)."""
    import time as _time
    model.eval()
    tokenizer = processor.tokenizer
    pad_id    = tokenizer.pad_token_id or 0
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids  = list({t for t in [tokenizer.eos_token_id, im_end_id]
                      if t is not None and t != tokenizer.unk_token_id})
    think_tag = "<think>\n"
    bad_words = [[t] for t in ban_latent_ids] if ban_latent_ids else None

    # Scoped DNA-cache patch: fall back to live Evo2 on a cache miss (RFT rows can
    # include sampled/gold rows not in a val/test-only cache), so a partial cache
    # never breaks the eval — it only saves time where it hits.
    _orig_evo2_embed = model._evo2_embed
    if dna_cache is not None:
        import types as _eval_types
        def _cached_evo2_embed(self, input_ids: torch.Tensor, layer_name: str) -> torch.Tensor:
            key = input_ids.cpu().numpy().tobytes()
            emb = dna_cache.get(key)
            if emb is None:
                return _orig_evo2_embed(input_ids, layer_name)
            _p = next(self.dna_projection.parameters())
            return emb.to(device=_p.device, dtype=_p.dtype)
        model._evo2_embed = _eval_types.MethodType(_cached_evo2_embed, model)

    details: List[Tuple[str, str, bool]] = []
    total_time = 0.0
    n_timed    = 0
    try:
        for row in rows:
            full_text     = row["text"]
            gt            = _extract_answer(row["answer"]) or \
                            row["answer"].replace("Answer:", "").replace("<|im_end|>", "").strip()
            dna_sequences = row.get("dna_sequences", ["", ""])
            cut         = full_text.find(think_tag)
            prompt_text = full_text[:cut + len(think_tag)] if cut != -1 else full_text
            batch = processor(
                text                = [prompt_text],
                batch_dna_sequences = [dna_sequences],
                return_tensors      = "pt",
                padding             = False,
                add_special_tokens  = False,
                max_length_text     = model.max_length_text,
                max_length_dna      = model.max_length_dna,
            )
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            dna_tok   = {k: v.to(device) for k, v in batch["dna_tokenized"].items()}
            idx_map   = list(batch["batch_idx_map"])
            try:
                _t0 = _time.perf_counter()
                out_ids = model.generate(
                    input_ids            = input_ids,
                    attention_mask       = attn,
                    dna_tokenized        = dna_tok,
                    batch_idx_map        = idx_map,
                    max_new_tokens       = max_new_tokens,
                    do_sample            = False,   # greedy → deterministic checkpoint selection
                    pad_token_id         = pad_id,
                    eos_token_id         = stop_ids,
                    bad_words_ids        = bad_words,
                )
                total_time += _time.perf_counter() - _t0
                n_timed    += 1
            except Exception as e:
                print(f"  [eval290] gen error ({e}); scoring as wrong")
                details.append(("", gt, False))
                continue
            gen  = tokenizer.decode(out_ids[0], skip_special_tokens=False)
            pred = _extract_answer(gen)
            details.append((pred, gt, _is_correct(pred, gt)))
    finally:
        if dna_cache is not None:
            model._evo2_embed = _orig_evo2_embed   # restore so training resumes on live Evo2

    n_total   = len(details)
    n_correct = sum(1 for _, _, ok in details if ok)
    acc       = n_correct / max(n_total, 1)
    prec_mac, rec_mac, f1_mac, prec_w, rec_w, f1_w = _metrics_from_details(details)
    mean_time = total_time / max(n_timed, 1)

    sep = "=" * 64
    print(f"\n{sep}\n  EVAL val+test  {label}  "
          f"(latents={'banned' if ban_latent_ids else 'self-emit'})\n{sep}")
    print(f"  Step            : {step}")
    print(f"  Accuracy        : {acc:.4f}  ({n_correct}/{n_total})")
    print(f"  Precision       : {prec_mac:.4f}  (macro)   {prec_w:.4f}  (weighted)")
    print(f"  Recall          : {rec_mac:.4f}  (macro)   {rec_w:.4f}  (weighted)")
    print(f"  F1              : {f1_mac:.4f}  (macro)   {f1_w:.4f}  (weighted)")
    print(f"  Mean time/sample: {mean_time:.2f}s\n{sep}\n")

    model.train()
    return {
        "n_correct": n_correct, "n_total": n_total, "accuracy": acc,
        "precision_macro": prec_mac, "recall_macro": rec_mac, "f1_macro": f1_mac,
        "precision_weighted": prec_w, "recall_weighted": rec_w, "f1_weighted": f1_w,
        "mean_time_per_sample_sec": mean_time,
    }


def collate_fn(batch: List[Tuple], pad_id: int):
    """
    Collate (dataset_idx, ids, dna_tok, idx_map, prompt_end) tuples.
    Returns (padded_ids, dna_tok_batch, idx_map_batch, prompt_ends, sample_indices).
    """
    sample_indices = [item[0] for item in batch]
    ids_list    = [item[1] for item in batch]
    dna_toks    = [item[2] for item in batch]
    idx_maps    = [item[3] for item in batch]
    prompt_ends = [item[4] for item in batch]

    # Pad text token IDs (right-pad; attention mask is built from pad_id != token)
    max_len = max(t.shape[0] for t in ids_list)
    padded  = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(ids_list):
        padded[i, :t.shape[0]] = t

    # Concatenate DNA tokenized: each item has shape [n_seqs, L]
    # Sequences shorter than max_length_dna have varying L — pad to batch max.
    _all_ids   = [d["input_ids"]      for d in dna_toks]
    _all_masks = [d["attention_mask"] for d in dna_toks]
    _max_dna   = max(t.shape[1] for t in _all_ids)
    def _pad_dna(t, val):
        gap = _max_dna - t.shape[1]
        return torch.nn.functional.pad(t, (0, gap), value=val) if gap > 0 else t
    dna_input_ids  = torch.cat([_pad_dna(t, 0) for t in _all_ids],   dim=0)  # [B*n, L]
    dna_attn_mask  = torch.cat([_pad_dna(t, 0) for t in _all_masks], dim=0)  # [B*n, L]
    dna_tok_batch  = {"input_ids": dna_input_ids, "attention_mask": dna_attn_mask}

    # Rebuild batch_idx_map: sample i contributes n_seqs entries all equal to i
    n_seqs        = len(idx_maps[0])
    idx_map_batch = [i for i in range(len(batch)) for _ in range(n_seqs)]

    return padded, dna_tok_batch, idx_map_batch, prompt_ends, sample_indices


# ── Epoch-level entropy computation (global mode) ─────────────────────────────

@torch.no_grad()
def compute_epoch_entropies(
    model:           DNALLMModel,
    dataset:         KeggRawDataset,
    tokenizer,
    device:          str,
    batch_size:      int = 4,
    max_samples:     int = 0,
) -> Dict[int, List[float]]:
    """
    Run one forward pass over the dataset to compute per-step entropies.
    max_samples > 0 limits to a random subset (faster; good entropy estimate).

    Returns:
        {sample_idx: [entropy_step_0, entropy_step_1, ...]}
    """
    model.eval()
    pad_id    = tokenizer.pad_token_id or 0
    if max_samples > 0 and max_samples < len(dataset):
        indices = random.sample(range(len(dataset)), max_samples)
        subset  = torch.utils.data.Subset(dataset, indices)
    else:
        subset  = dataset
    loader    = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    entropy_map: Dict[int, List[float]] = {}
    n_total  = len(subset)
    n_done   = 0
    print(f"  [entropy] 0 / {n_total} samples ...", end="\r", flush=True)

    for batch_ids, dna_tok, idx_map, _, sample_indices in loader:
        batch_ids = batch_ids.to(device)
        dna_tok   = {k: v.to(device) for k, v in dna_tok.items()}
        out       = model(
            input_ids      = batch_ids,
            attention_mask = real_attention_mask(batch_ids, pad_id),
            dna_tokenized  = dna_tok,
            batch_idx_map  = idx_map,
        )
        logits = out.logits   # [B, T, V]

        for b in range(batch_ids.shape[0]):
            ids_b    = batch_ids[b].tolist()
            spans_b  = find_step_content_spans(ids_b, tokenizer)
            ent_b    = compute_step_entropies(logits[b].cpu(), spans_b)
            entropy_map[sample_indices[b]] = ent_b

        n_done += batch_ids.shape[0]
        print(f"  [entropy] {n_done} / {n_total} samples ...", end="\r", flush=True)

    print(f"  [entropy] {n_done} / {n_total} samples — done.    ")
    model.train()
    return entropy_map


# ── Single-batch processing (shared by both modes) ────────────────────────────

def process_batch(
    batch_ids:     torch.Tensor,            # [B, T]  raw token ids
    model:         DNALLMModel,
    tokenizer,
    s:             int,                     # curriculum: replace s steps per sample
    start_id:      int,
    end_id:        int,
    latent_id:     int,
    device:        str,
    entropy_map:   Optional[Dict[int, List[float]]] = None,  # global mode
    batch_offset:  int = 0,                                   # sample index offset
    prompt_ends:   Optional[List[int]] = None,                # user-turn lengths per sample
    dna_tokenized: Optional[Dict[str, torch.Tensor]] = None, # DNA inputs for inline mode
    batch_idx_map: Optional[List[int]] = None,               # DNA batch mapping
    self_adaptive: bool = False,                             # RFT: pre-placed latents
):
    """
    Build modified input_ids, labels, and loss_weights for one batch.

    If entropy_map is provided (global mode), use pre-computed entropies.
    Otherwise (inline mode), compute entropies from current model output.

    Returns:
        new_ids     [B, T]
        labels      [B, T]
        weights     [B, T]
        step_counts [B]    number of steps actually replaced per sample
    """
    pad_id  = tokenizer.pad_token_id or 0
    B, T    = batch_ids.shape

    # Token IDs for </think> and "Answer:" — these must never be replaced by latent markers.
    # Encode both forms ("\nAnswer:" and "Answer:") to cover tokenizer boundary variants.
    _protected: List[int] = []
    for _text in ("</think>", "\n</think>", "Answer:", "\nAnswer:"):
        _protected.extend(tokenizer.encode(_text, add_special_tokens=False))
    protected_ids = frozenset(_protected)

    # RFT / self-adaptive: the completions ALREADY contain their latent markers (placed
    # by the GRPO controller when sampled). Skip the curriculum injection AND the inline
    # entropy forward — just teacher-force the whole completion (latents included) and
    # mask the prompt/padding, so the model learns to EMIT the latents itself.
    if self_adaptive:
        new_ids_list, labels_list, weights_list, step_counts = [], [], [], []
        for b in range(B):
            pe = prompt_ends[b] if prompt_ends is not None else 0
            nid, lbl, wgt = label_self_adaptive_latents(batch_ids[b].tolist(), pe, pad_id)
            new_ids_list.append(nid)
            labels_list.append(lbl)
            weights_list.append(wgt)
            step_counts.append(int((nid == start_id).sum().item()))
        # .to(device): label_self_adaptive_latents builds tensors from .tolist() (CPU), so
        # move them to the model's device (the curriculum path below does the same).
        return (torch.stack(new_ids_list).to(device),
                torch.stack(labels_list).to(device),
                torch.stack(weights_list).to(device), step_counts)

    # Inline entropy: one forward pass (no grad)
    if entropy_map is None:
        with torch.no_grad():
            dna_tok_d = ({k: v.to(device) for k, v in dna_tokenized.items()}
                         if dna_tokenized is not None else None)
            out       = model(
                input_ids      = batch_ids.to(device),
                attention_mask = real_attention_mask(batch_ids, pad_id).to(device),
                dna_tokenized  = dna_tok_d,
                batch_idx_map  = batch_idx_map,
            )
            logits_all = out.logits.cpu()   # [B, T, V]

    new_ids_list  = []
    labels_list   = []
    weights_list  = []
    step_counts   = []

    for b in range(B):
        ids_b   = batch_ids[b].tolist()
        spans_b = find_step_content_spans(ids_b, tokenizer)

        if not spans_b:
            # No steps found: still apply user-prompt mask, keep rest
            pe  = prompt_ends[b] if prompt_ends is not None else 0
            lbl = batch_ids[b].clone()
            wgt = torch.ones(batch_ids[b].shape[0], dtype=torch.float)
            lbl[:pe] = -100
            wgt[:pe] = 0.0
            # Mask trailing padding by position (pad_id == <|im_end|>, so an id
            # comparison would also mask the answer terminator — keep it).
            _ce = real_content_end(ids_b, pad_id)
            lbl[_ce:] = -100
            wgt[_ce:] = 0.0
            new_ids_list.append(batch_ids[b])
            labels_list.append(lbl)
            weights_list.append(wgt)
            step_counts.append(0)
            continue

        # Get entropies
        if entropy_map is not None:
            ent_b = entropy_map.get(batch_offset + b, [float("inf")] * len(spans_b))
            # Pad / trim to match current spans (spans may differ from epoch-entropy pass)
            if len(ent_b) < len(spans_b):
                ent_b = ent_b + [float("inf")] * (len(spans_b) - len(ent_b))
            ent_b = ent_b[:len(spans_b)]
        else:
            ent_b = compute_step_entropies(logits_all[b], spans_b)

        # Select s lowest-entropy step contents.
        # Never replace the last step — it bridges directly to </think>/Answer: and
        # compressing it teaches the model to close the thinking block prematurely.
        eligible        = list(range(len(spans_b) - 1)) if len(spans_b) > 1 else []
        n_replace       = min(s, len(eligible))
        replace_indices = sorted(eligible, key=lambda i: ent_b[i])[:n_replace]

        pe = prompt_ends[b] if prompt_ends is not None else 0
        new_ids_b, labels_b, weights_b = inject_latent_markers(
            ids_b, spans_b, replace_indices,
            start_id, end_id, latent_id, pad_id,
            prompt_end=pe,
            protected_ids=protected_ids,
        )

        # Pad/trim to original length T
        L = new_ids_b.shape[0]
        if L < T:
            new_ids_b  = F.pad(new_ids_b,  (0, T - L), value=pad_id)
            labels_b   = F.pad(labels_b,   (0, T - L), value=-100)
            weights_b  = F.pad(weights_b,  (0, T - L), value=0.0)
        elif L > T:
            new_ids_b  = new_ids_b[:T]
            labels_b   = labels_b[:T]
            weights_b  = weights_b[:T]

        new_ids_list.append(new_ids_b)
        labels_list.append(labels_b)
        weights_list.append(weights_b)
        step_counts.append(n_replace)

    new_ids = torch.stack(new_ids_list).to(device)
    labels  = torch.stack(labels_list).to(device)
    weights = torch.stack(weights_list).to(device)
    return new_ids, labels, weights, step_counts


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[Stage1.5] Seed: {args.seed}")

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    print("[Stage1.5] Loading model ...")
    model = DNALLMModel(
        text_model_name     = args.text_model_name,
        dna_model_name      = args.dna_model_name,
        cache_dir           = args.cache_dir,
        max_length_text     = args.max_length_text,
        max_length_dna      = args.max_length_dna,
        text_model_finetune = True,
        dna_model_finetune  = False,
        dna_is_evo2         = True,
        dna_embedding_layer = args.dna_embedding_layer,
        use_cross_attention = True,
        use_hrpo_gate       = False,
        device              = device,
    ).to(device)
    model.text_model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Load Stage 1 / Stage 1.5 / GRPO start weights.
    # --stage1_ckpt may be EITHER:
    #   * a single file (Stage 1 .ckpt or Stage 1.5 model.pt) — original path, or
    #   * a checkpoint DIRECTORY (e.g. a train_06b GRPO checkpoint-XXXX) holding
    #     pytorch_model.bin / model.safetensors / model.pt (+ optional
    #     thinking_gate.pt / dna_injector.pt) — RFT-on-GRPO start (backbone already
    #     w9-native). The GRPO gate/injector, if present, are loaded further below.
    tokenizer     = model.processor.tokenizer
    _gate_src_dir = None   # dir to load GRPO thinking_gate.pt / dna_injector.pt from

    if os.path.isdir(args.stage1_ckpt):
        # GRPO/HF-Trainer dir: its backbone already carries the latent tokens
        # (vocab 151675), so add them BEFORE loading so embed/lm_head shapes match.
        ensure_latent_tokens(tokenizer, model)
        state = load_start_state_dict_from_dir(args.stage1_ckpt)
        state = merge_grpo_lora_state_dict(
            state, lora_r=args.start_lora_r, lora_alpha=args.start_lora_alpha)
        clean = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(clean, strict=False)
        _gate_src_dir = args.stage1_ckpt
        print(f"[Stage1.5] Loaded start weights from dir: {args.stage1_ckpt} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
        if missing:
            print(f"  Missing keys sample: {missing[:3]}")
    else:
        ckpt  = torch.load(args.stage1_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        clean = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}

        # Pre-resize vocab if the checkpoint already contains latent tokens
        # (Stage 1.5 checkpoints have vocab 151675; base Qwen is 151672).
        _emb = clean.get("text_model.model.embed_tokens.weight")
        if _emb is not None and _emb.shape[0] != len(tokenizer):
            _latent_toks = [LATENT_START, LATENT_END, LATENT_PAD]
            _missing_toks = [t for t in _latent_toks
                             if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
            if _missing_toks:
                tokenizer.add_special_tokens({"additional_special_tokens": _missing_toks})
            model.text_model.resize_token_embeddings(len(tokenizer))
            print(f"[Stage1.5] Pre-load vocab resize → {len(tokenizer)} "
                  f"(ckpt had {_emb.shape[0]})")

        missing, _ = model.load_state_dict(clean, strict=False)
        if missing:
            print(f"  Missing keys (expected if no gate): {missing[:3]}")
        print(f"[Stage1.5] Loaded checkpoint: {args.stage1_ckpt}")

    start_id, end_id, latent_id = ensure_latent_tokens(tokenizer, model)
    pad_id = tokenizer.pad_token_id or 0

    # Freeze DNA encoder. Evo2 wraps the actual nn.Module at .model;
    # dna_llm.py accesses params as dna_model.model.parameters() throughout.
    if hasattr(model, "dna_model") and model.dna_model is not None:
        inner = getattr(model.dna_model, "model", model.dna_model)
        for p in inner.parameters():
            p.requires_grad_(False)

    # ── ThinkingResidualGate + DNAHiddenInjector (optional, --use_gate) ───────
    thinking_gate = None
    dna_injector  = None
    if getattr(args, "use_gate", False):
        if not _GATE_AVAILABLE:
            raise ImportError("--use_gate requires adaptive_thinking_residual_w9 to be importable")
        hidden_size   = model.text_hidden_size
        thinking_gate = ThinkingResidualGate(
            hidden_size = hidden_size,
            use_ot_dist = True,   # must match Stage 3 so ot_scale is saved in checkpoint
            r_min       = 0.5,
            r_max       = 0.99,
        ).to(device)
        dna_injector = DNAHiddenInjector(
            hidden_size = hidden_size,
            r_min       = 0.7,
            r_max       = 0.99,
        ).to(device)
        model.forward = _types.MethodType(
            _make_dual_mode_forward_w9(thinking_gate, dna_injector),
            model,
        )
        model._gate_warmup_factor = MAX_GATE_FACTOR  # fixed, no ramp during SFT
        print(f"[Stage1.5] Gate enabled: ThinkingResidualGate + DNAHiddenInjector "
              f"(factor={MAX_GATE_FACTOR})")

        # RFT-on-GRPO: carry the checkpoint's tuned gate/injector so the whole
        # w9-native state transfers (not just the backbone). Without this the
        # gate/injector would train fresh from random, contradicting the point of
        # starting from a w9-native GRPO checkpoint.
        if _gate_src_dir is not None:
            _gp = os.path.join(_gate_src_dir, "thinking_gate.pt")
            _ip = os.path.join(_gate_src_dir, "dna_injector.pt")
            if os.path.isfile(_gp) and os.path.isfile(_ip):
                thinking_gate.load_state_dict(torch.load(_gp, map_location=device))
                dna_injector.load_state_dict(torch.load(_ip, map_location=device))
                print(f"[Stage1.5] Loaded gate+injector from start dir: {_gate_src_dir}")
            else:
                print(f"[Stage1.5] WARN: no thinking_gate.pt/dna_injector.pt in "
                      f"{_gate_src_dir} — gate/injector will train from fresh init")

    # ── Data ──────────────────────────────────────────────────────────────────
    _rft = getattr(args, "rft_traces", None)
    if _rft:
        print(f"[RFT] Loading self-adaptive RFT traces: {_rft}")
        all_rows = load_rft_rows(
            _rft,
            dataset_name          = None if args.kegg_csv else args.kegg_dataset,
            kegg_csv              = args.kegg_csv,
            cache_dir             = args.cache_dir,
            truncate_dna_per_side = args.truncate_dna_per_side,
        )
        # val: reuse the KEGG val split as a loss proxy during RFT (normal labeling)
        if args.kegg_csv:
            _, val_rows = load_kegg_csv(args.kegg_csv, args.truncate_dna_per_side)
        else:
            _, val_rows = load_kegg_hf(args.kegg_dataset, cache_dir=args.cache_dir,
                                       truncate_dna_per_side=args.truncate_dna_per_side)
    elif args.kegg_csv:
        print(f"[Stage1.5] Loading KEGG from CSV: {args.kegg_csv}")
        all_rows, val_rows = load_kegg_csv(args.kegg_csv, args.truncate_dna_per_side)
    else:
        print(f"[Stage1.5] Loading KEGG dataset: {args.kegg_dataset} ...")
        all_rows, val_rows = load_kegg_hf(args.kegg_dataset, cache_dir=args.cache_dir,
                                           truncate_dna_per_side=args.truncate_dna_per_side)
    print(f"[Stage1.5] train={len(all_rows)}  val={len(val_rows)}")

    val_dataset = KeggRawDataset(val_rows, model.processor, args.max_length_text, args.max_length_dna)

    # ── val+test accuracy probe set (290) for the half-epoch eval ──────────────
    eval290_rows: List[dict] = []
    if getattr(args, "half_epoch_eval", False):
        eval290_rows = load_eval_rows(
            dataset_name          = None if args.kegg_csv else args.kegg_dataset,
            kegg_csv              = args.kegg_csv,
            cache_dir             = args.cache_dir,
            truncate_dna_per_side = args.truncate_dna_per_side,
        )
        print(f"[Stage1.5] half-epoch eval enabled: {len(eval290_rows)} val+test rows "
              f"(latents={'banned' if args.eval_ban_latent else 'self-emit'})")

    eval_dna_cache = None
    if getattr(args, "eval_dna_cache", None):
        eval_dna_cache = torch.load(args.eval_dna_cache, map_location="cpu")
        print(f"[Stage1.5] eval290 DNA cache: {len(eval_dna_cache)} sequences ← {args.eval_dna_cache}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # steps_per_s: one cosine cycle per curriculum level (restart at each s)
    steps_per_s = (
        len(all_rows) // args.batch_size + 1
    ) * args.passes_per_step

    gate_params = []
    if thinking_gate is not None:
        gate_params += list(thinking_gate.parameters())
    if dna_injector is not None:
        gate_params += list(dna_injector.parameters())
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad] + gate_params,
        lr           = args.learning_rate,
        weight_decay = args.weight_decay,
    )
    # scheduler is created fresh per curriculum step (see loop below)
    scheduler = None
    scaler = torch.amp.GradScaler("cuda", enabled=(device != "cpu"))

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = f"stage1_5_{args.entropy_mode}",
            config  = vars(args),
        )

    global_step = 0
    best_val    = float("inf")
    best_acc    = -1.0

    # Checkpoint dir naming: the "sSS_" curriculum prefix is meaningful only for
    # curriculum SFT (s = steps compressed to latents, process_batch:1157). In RFT
    # mode (self_adaptive) the outer s-loop is a single vestigial iteration — s is
    # never read by process_batch's self_adaptive branch — so name by epoch only.
    def _ckpt_name(s: int, inner_pass: int, tag: str = None) -> str:
        base = f"epoch{inner_pass+1:02d}" if _rft else f"s{s:02d}_pass{inner_pass+1:02d}"
        return f"{base}_{tag}" if tag else base

    # ── Half-epoch save + val+test(290) accuracy probe ─────────────────────────
    # Saves a checkpoint (model + gate + injector + tokenizer) and runs the 290
    # accuracy eval, tracking the accuracy-best into <output_dir>/best_acc/. Called
    # at 50% and 100% of every pass when --half_epoch_eval is set.
    def _save_and_eval290(tag: str, s: int, inner_pass: int):
        nonlocal best_acc
        ckpt_dir = os.path.join(args.output_dir, _ckpt_name(s, inner_pass, tag))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        tokenizer.save_pretrained(ckpt_dir)
        if thinking_gate is not None:
            torch.save(thinking_gate.state_dict(), os.path.join(ckpt_dir, "thinking_gate.pt"))
            torch.save(dna_injector.state_dict(),  os.path.join(ckpt_dir, "dna_injector.pt"))
        print(f"  Saved ({tag}) → {ckpt_dir}/model.pt")

        if not eval290_rows:
            return
        ban_ids = [start_id, end_id, latent_id] if args.eval_ban_latent else None
        m = evaluate_accuracy(
            model, model.processor, eval290_rows, device,
            max_new_tokens = args.gen_max_new_tokens,
            ban_latent_ids = ban_ids,
            label          = f"pass={inner_pass+1} [{tag}]" if _rft else f"s={s} pass={inner_pass+1} [{tag}]",
            step           = global_step,
            dna_cache      = eval_dna_cache,
        )
        acc = m["accuracy"]
        if use_wandb:
            import wandb
            wandb.log({f"eval290/{k}": v for k, v in m.items()}, step=global_step)
        if acc > best_acc:
            best_acc = acc
            ba_dir = os.path.join(args.output_dir, "best_acc")
            os.makedirs(ba_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ba_dir, "model.pt"))
            tokenizer.save_pretrained(ba_dir)
            if thinking_gate is not None:
                torch.save(thinking_gate.state_dict(), os.path.join(ba_dir, "thinking_gate.pt"))
                torch.save(dna_injector.state_dict(),  os.path.join(ba_dir, "dna_injector.pt"))
            print(f"  ★ New best eval290 accuracy={best_acc:.4f} → {ba_dir}/")

    # ── Baseline SAMPLE GENERATIONS before any training step ───────────────────
    # Reference point so the 1/3, 2/3, end-of-pass prints later can be compared
    # against the model's behavior prior to this RFT/curriculum run.
    if args.sample_every > 0:
        gens = generate_samples(
            model, model.processor, val_rows, device,
            n_samples           = args.n_gen_samples,
            max_new_tokens      = args.gen_max_new_tokens,
            label               = "before training [step=0]",
            suppress_latent_ids = [start_id, end_id, latent_id],
            self_emit_half      = _rft,
        )
        if use_wandb:
            import wandb
            wandb.log({
                "samples/before_training": wandb.Table(
                    columns=["step", "generated"],
                    data=[[0, g] for g in gens],
                )
            }, step=0)

    # ── Outer curriculum loop: s = start_latent_step .. max_latent_steps ────────
    # RFT mode (self_adaptive): s is never read by process_batch's self_adaptive branch
    # (it teacher-forces the completion as-is, no curriculum). Force a single iteration
    # regardless of --start_latent_step/--max_latent_steps so RFT can't silently repeat
    # the whole epoch set once per curriculum level (the risk if callers relied on those
    # flags being set equal — now the code guarantees it instead).
    _s_range = [args.max_latent_steps] if _rft else range(args.start_latent_step, args.max_latent_steps + 1)
    for s in _s_range:
        if not _rft:
            print(f"\n[Stage1.5] ── Curriculum step s={s} / {args.max_latent_steps} ──")

        # Data refresh: shuffle in 10% new samples each outer step (LatentSp paper)
        n_refresh    = max(1, int(len(all_rows) * args.data_refresh_ratio))
        active_rows  = all_rows.copy()
        random.shuffle(active_rows)
        active_rows  = active_rows[:len(all_rows)]  # keep same size, just reshuffled
        train_dataset = KeggRawDataset(active_rows, model.processor, args.max_length_text, args.max_length_dna)

        train_loader  = DataLoader(
            train_dataset,
            batch_size  = args.batch_size,
            shuffle     = True,
            num_workers = 0,
            collate_fn  = lambda b: collate_fn(b, pad_id),
        )

        # Fresh cosine schedule per curriculum step — prevents LR dying in the
        # cosine tail before the model has had a chance to learn harder compression.
        # Optimizer state (Adam momentum) is preserved across steps.
        for pg in optimizer.param_groups:
            pg["lr"] = args.learning_rate
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = max(1, int(steps_per_s * 0.05)),
            num_training_steps = steps_per_s,
        )
        print(f"  LR restarted: {args.learning_rate:.2e}  steps_this_s={steps_per_s}")

        # Global mode: compute epoch-level entropies ONCE before this curriculum step.
        # Skipped entirely in RFT/self-adaptive mode (no curriculum injection).
        entropy_map: Optional[Dict] = None
        if args.entropy_mode == "global" and not getattr(args, "rft_traces", None):
            print(f"  [global] Computing epoch-level entropies for s={s} ...")
            entropy_map = compute_epoch_entropies(
                model, train_dataset, tokenizer, device,
                batch_size  = args.entropy_batch_size,
                max_samples = args.max_entropy_samples,
            )
            n_with_steps = sum(1 for v in entropy_map.values() if v)
            print(f"  [global] {n_with_steps}/{len(train_dataset)} samples have steps")

        # Inner loop: passes_per_step full passes over the data
        for inner_pass in range(args.passes_per_step):
            model.train()
            total_loss    = 0.0
            total_replace = 0
            total_samples = 0
            n_batches     = 0
            batch_offset  = 0
            logged_sample = False   # print one decoded sample per curriculum step
            _half_at      = max(1, len(train_loader) // 2)   # 50%-of-epoch trigger point
            _did_half     = False
            # SAMPLE GENERATIONS print fires at 1/3 and 2/3 of the pass (3/3 already
            # happens unconditionally in the existing end-of-pass block below).
            _third1_at, _third2_at = (max(1, len(train_loader) * 1 // 3),
                                       max(1, len(train_loader) * 2 // 3))
            _did_third1, _did_third2 = False, False

            for batch_ids, dna_tok, idx_map, prompt_ends, _ in train_loader:
                B = batch_ids.shape[0]
                dna_tok_d = {k: v.to(device) for k, v in dna_tok.items()}

                with torch.amp.autocast("cuda", enabled=(device != "cpu")):
                    # Build modified batch (entropy computed here for inline mode)
                    new_ids, labels, weights, step_counts = process_batch(
                        batch_ids     = batch_ids,
                        model         = model,
                        tokenizer     = tokenizer,
                        s             = s,
                        start_id      = start_id,
                        end_id        = end_id,
                        latent_id     = latent_id,
                        device        = device,
                        entropy_map   = entropy_map,     # None → inline mode
                        batch_offset  = batch_offset,
                        prompt_ends   = prompt_ends,
                        dna_tokenized = dna_tok,         # CPU tensors; moved inside process_batch
                        batch_idx_map = idx_map,
                        self_adaptive = bool(getattr(args, "rft_traces", None)),
                    )

                    # Log one decoded sample per curriculum step to verify injection
                    if not logged_sample:
                        log_latent_sample(tokenizer, batch_ids[0], new_ids[0], s)
                        logged_sample = True

                    # Forward pass with DNA embeddings injected at <|dna_pad|> positions
                    out  = model(
                        input_ids      = new_ids,
                        attention_mask = real_attention_mask(new_ids, pad_id),
                        dna_tokenized  = dna_tok_d,
                        batch_idx_map  = idx_map,
                    )
                    loss = weighted_sft_loss(out.logits, labels, weights)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARNING: NaN/Inf loss at step {global_step}, skipping")
                    optimizer.zero_grad()
                    batch_offset += B
                    continue

                scaler.scale(loss / args.grad_accum).backward()

                if (n_batches + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                total_loss    += loss.item()
                total_replace += sum(step_counts)
                total_samples += B
                n_batches     += 1
                global_step   += 1
                batch_offset  += B

                if global_step % args.log_every == 0:
                    avg_loss = total_loss / max(n_batches, 1)
                    avg_rep  = total_replace / max(total_samples, 1)
                    print(f"  s={s} pass={inner_pass+1} step={global_step}  "
                          f"loss={avg_loss:.4f}  avg_replaced={avg_rep:.2f}")
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss":          avg_loss,
                            "train/avg_replaced":  avg_rep,
                            "train/curriculum_s":  s,
                            "train/lr":            scheduler.get_last_lr()[0],
                        }, step=global_step)

                # SAMPLE GENERATIONS at 1/3 and 2/3 of the pass (3/3 = the existing
                # unconditional end-of-pass block below). Replaces the old fixed
                # step-count cadence, which didn't line up with epoch progress.
                _fire_third = None
                if args.sample_every > 0:
                    if not _did_third1 and n_batches >= _third1_at:
                        _did_third1, _fire_third = True, "1/3"
                    elif not _did_third2 and n_batches >= _third2_at:
                        _did_third2, _fire_third = True, "2/3"
                if _fire_third is not None:
                    gens = generate_samples(
                        model, model.processor, val_rows, device,
                        n_samples           = args.n_gen_samples,
                        max_new_tokens      = args.gen_max_new_tokens,
                        label               = (f"pass={inner_pass+1} [{_fire_third} epoch, step={global_step}]" if _rft
                                                else f"s={s} pass={inner_pass+1} [{_fire_third} epoch, step={global_step}]"),
                        suppress_latent_ids = [start_id, end_id, latent_id],
                        self_emit_half      = _rft,
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "samples/generated": wandb.Table(
                                columns=["step", "generated"],
                                data=[[global_step, g] for g in gens],
                            )
                        }, step=global_step)

                # ── Half-epoch (50%) save + val+test(290) accuracy probe ───────
                if args.half_epoch_eval and not _did_half and n_batches >= _half_at:
                    _did_half = True
                    _save_and_eval290("half", s, inner_pass)

            # ── Validation ────────────────────────────────────────────────────
            model.eval()
            val_loss   = 0.0
            n_val      = 0
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=0, collate_fn=lambda b: collate_fn(b, pad_id),
            )

            with torch.no_grad():
                for val_ids, val_dna_tok, val_idx_map, val_prompt_ends, _ in val_loader:
                    val_dna_tok_d = {k: v.to(device) for k, v in val_dna_tok.items()}
                    new_ids_v, labels_v, weights_v, _ = process_batch(
                        batch_ids     = val_ids,
                        model         = model,
                        tokenizer     = tokenizer,
                        s             = s,
                        start_id      = start_id,
                        end_id        = end_id,
                        latent_id     = latent_id,
                        device        = device,
                        entropy_map   = None,   # always inline for val (no precompute)
                        prompt_ends   = val_prompt_ends,
                        dna_tokenized = val_dna_tok,
                        batch_idx_map = val_idx_map,
                    )
                    out_v = model(
                        input_ids      = new_ids_v,
                        attention_mask = real_attention_mask(new_ids_v, pad_id),
                        dna_tokenized  = val_dna_tok_d,
                        batch_idx_map  = val_idx_map,
                    )
                    loss_v   = weighted_sft_loss(out_v.logits, labels_v, weights_v)
                    val_loss += loss_v.item()
                    n_val    += 1

            avg_val = val_loss / max(n_val, 1)
            print(f"  [val] s={s} pass={inner_pass+1}  val_loss={avg_val:.4f}")

            if use_wandb:
                import wandb
                wandb.log({"val/loss": avg_val, "val/curriculum_s": s}, step=global_step)

            # Generation sample at end of every pass
            gens = generate_samples(
                model, model.processor, val_rows, device,
                n_samples           = args.n_gen_samples,
                max_new_tokens      = args.gen_max_new_tokens,
                label               = (f"pass={inner_pass+1} [end-of-pass]" if _rft
                                        else f"s={s} pass={inner_pass+1} [end-of-pass]"),
                suppress_latent_ids = [start_id, end_id, latent_id],
                self_emit_half      = _rft,
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "samples/end_of_pass": wandb.Table(
                        columns=["s", "pass", "generated"],
                        data=[[s, inner_pass + 1, g] for g in gens],
                    )
                }, step=global_step)

            # Save checkpoint (skipped when half_epoch_eval is on: _save_and_eval290("full", ...)
            # below saves the same end-of-pass state under "..._full" — this avoided a
            # redundant duplicate model.pt save at the same point).
            if not args.half_epoch_eval:
                ckpt_dir = os.path.join(args.output_dir, _ckpt_name(s, inner_pass))
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                tokenizer.save_pretrained(ckpt_dir)
                if thinking_gate is not None:
                    torch.save(thinking_gate.state_dict(), os.path.join(ckpt_dir, "thinking_gate.pt"))
                    torch.save(dna_injector.state_dict(),  os.path.join(ckpt_dir, "dna_injector.pt"))
                print(f"  Saved → {ckpt_dir}/model.pt")

            if avg_val < best_val:
                best_val = avg_val
                best_dir = os.path.join(args.output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
                tokenizer.save_pretrained(best_dir)
                if thinking_gate is not None:
                    torch.save(thinking_gate.state_dict(), os.path.join(best_dir, "thinking_gate.pt"))
                    torch.save(dna_injector.state_dict(),  os.path.join(best_dir, "dna_injector.pt"))
                print(f"  ★ New best val_loss={best_val:.4f} → {best_dir}/")

            # ── End-of-epoch (100%) save + val+test(290) accuracy probe ────────
            if args.half_epoch_eval:
                _save_and_eval290("full", s, inner_pass)

        # After final curriculum step: also refresh data by re-shuffling
        random.shuffle(all_rows)

    # ── Save final ────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    tokenizer.save_pretrained(final_dir)
    if thinking_gate is not None:
        torch.save(thinking_gate.state_dict(), os.path.join(final_dir, "thinking_gate.pt"))
        torch.save(dna_injector.state_dict(),  os.path.join(final_dir, "dna_injector.pt"))
        print(f"\n[Stage1.5] Done. Final model + gate → {final_dir}/")
    else:
        print(f"\n[Stage1.5] Done. Final model → {final_dir}/")

    if use_wandb:
        import wandb
        wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1.5 LatentSp cold-start SFT")
    p.add_argument("--stage1_ckpt",           required=True)
    p.add_argument("--kegg_dataset",          default="wanglab/kegg",
                   help="HuggingFace dataset name (same source as Stage 1 SFT)")
    p.add_argument("--kegg_csv",              default=None,
                   help="Local anonymized KEGG CSV path (overrides --kegg_dataset when set)")
    p.add_argument("--rft_traces",            default=None,
                   help="Self-adaptive RFT mode: path to the selected-traces JSONL from "
                        "train/rft/build_rft_dataset.py. Trains on those completions with "
                        "label_self_adaptive_latents (latents teacher-forced, no curriculum, "
                        "no entropy precompute). Use --start_latent_step == --max_latent_steps "
                        "for a single pass set and --passes_per_step as the epoch count.")
    p.add_argument("--output_dir",            required=True)
    p.add_argument("--entropy_mode",          default="global",
                   choices=["global", "inline"],
                   help="global: epoch-level recompute; inline: per-batch (2x compute)")
    p.add_argument("--max_latent_steps",      type=int,   default=4,
                   help="S in LatentSp Algorithm 1: outer curriculum steps")
    p.add_argument("--start_latent_step",    type=int,   default=1,
                   help="First curriculum step to run (default 1). Set to max_latent_steps "
                        "to fine-tune only at the highest compression level, e.g. when "
                        "resuming from a Stage 1.5 s=4 checkpoint.")
    p.add_argument("--passes_per_step",       type=int,   default=1,
                   help="Full data passes per curriculum step s")
    p.add_argument("--data_refresh_ratio",    type=float, default=0.1,
                   help="Fraction of data re-shuffled between curriculum steps")
    p.add_argument("--batch_size",            type=int,   default=2)
    p.add_argument("--grad_accum",            type=int,   default=4)
    p.add_argument("--learning_rate",         type=float, default=5e-5)
    p.add_argument("--weight_decay",          type=float, default=0.01)
    p.add_argument("--max_length_text",       type=int,   default=6000)
    p.add_argument("--max_length_dna",        type=int,   default=2048)
    p.add_argument("--truncate_dna_per_side", type=int,   default=1024)
    p.add_argument("--entropy_batch_size",    type=int,   default=4,
                   help="Batch size for epoch-level entropy computation (global mode)")
    p.add_argument("--max_entropy_samples",  type=int,   default=0,
                   help="Max samples for entropy estimation (0=full dataset). "
                        "Set to 300 for ~3-4x speedup with minimal accuracy loss.")
    p.add_argument("--log_every",             type=int,   default=20)
    p.add_argument("--text_model_name",       default="Qwen/Qwen3-1.7B")
    p.add_argument("--dna_model_name",        default="evo2_7b_base")
    p.add_argument("--dna_embedding_layer",   default="blocks.28.mlp.l3")
    p.add_argument("--cache_dir",             default="~/.cache/huggingface")
    p.add_argument("--device",                default="cuda")
    p.add_argument("--wandb_project",         default=None)
    p.add_argument("--wandb_entity",          default=None)
    p.add_argument("--sample_every",          type=int,   default=200,
                   help="Enable/disable (>0/0) the periodic SAMPLE GENERATIONS print. "
                        "Fires at 1/3 and 2/3 of every pass (3/3 already happens "
                        "unconditionally at end-of-pass) — no longer a step-count interval.")
    p.add_argument("--n_gen_samples",         type=int,   default=10,
                   help="Number of val examples to generate from (first N, deterministic)")
    p.add_argument("--gen_max_new_tokens",    type=int,   default=800,
                   help="Max new tokens per generation sample")
    p.add_argument("--half_epoch_eval",       action="store_true", default=False,
                   help="At 50%% and 100%% of every pass: save a checkpoint "
                        "(sSS_passPP_half/_full for curriculum SFT, epochPP_half/_full for "
                        "RFT --rft_traces) AND run a greedy accuracy eval on the val+test "
                        "set (290). Tracks the accuracy-best into <output_dir>/best_acc/.")
    p.add_argument("--eval_ban_latent",       action="store_true", default=False,
                   help="Ban latent tokens during the half-epoch 290 eval (latent-free "
                        "number). Default off = latents allowed (self-emit, the RFT target).")
    p.add_argument("--start_lora_r",          type=int, default=16,
                   help="LoRA rank used to train a GRPO-checkpoint --stage1_ckpt DIR "
                        "(train_06b default: 16). Only used to scale the LoRA merge "
                        "(merge_grpo_lora_state_dict) when starting from such a dir.")
    p.add_argument("--start_lora_alpha",      type=int, default=32,
                   help="LoRA alpha used to train a GRPO-checkpoint --stage1_ckpt DIR "
                        "(train_06b default: 32). See --start_lora_r.")
    p.add_argument("--eval_dna_cache",        default=None,
                   help="Path to a precomputed {input_ids_bytes: embedding} DNA cache "
                        "(same format as eval_stage1_51_checkpoints.py --dna_cache) to skip "
                        "live Evo2 during the half-epoch 290 eval. Cache misses fall back to "
                        "live Evo2, so a val/test-only cache is safe. Training itself always "
                        "runs Evo2 live (unaffected — this only scopes the eval calls).")
    p.add_argument("--seed",                  type=int, default=42,
                   help="Random seed for reproducibility (data shuffling, sampling)")
    p.add_argument("--use_gate",             action="store_true", default=False,
                   help="Train Stage 1.5 WITH ThinkingResidualGate + DNAHiddenInjector "
                        "at fixed MAX_GATE_FACTOR. Saves thinking_gate.pt + dna_injector.pt "
                        "at every checkpoint so Stage 3 can load them with gate_warmup_steps=0.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
