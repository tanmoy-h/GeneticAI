"""
Stage 1.5 — LatentSp Cold-Start SFT (cached DNA embeddings).

Add --dna_cache /path/to/cache.pt to skip Evo2 at training time.
Run precompute_dna_embeddings.py first.

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
  python train_latent_sft_cached.py \\
      --stage1_ckpt   checkpoints/.../epoch=03-val_loss=0.4292.ckpt \\
      --train_csv     genomorph/dataset/kegg_curriculum/global_stage1_anon_genes_mol_keep_chr.csv \\
      --output_dir    /scratch/tanmoyh_iitp/GenoMorph/checkpoints/week9tests/stage1_5_sft_cached \\
      --entropy_mode  global \\
      --max_latent_steps 4 \\
      --dna_cache     /scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt

  # Inline mode (slower, matches LatentSp exactly):
  python train_latent_sft_cached.py \\
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

        # Content starts right after the colon
        content_char_start = offset + m.end()

        # Content ends at next step header or </think>
        if i + 1 < len(matches):
            content_char_end = offset + matches[i + 1].start()
        else:
            content_char_end = think_end   # char position of </think> in full text

        # Trim trailing whitespace/newlines from content
        content_text = text[content_char_start:content_char_end]
        stripped_len = len(content_text.rstrip())
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

def inject_latent_markers(
    input_ids:       List[int],
    step_spans:      List[Tuple[int, int, int]],
    replace_indices: List[int],
    start_id:        int,
    end_id:          int,
    latent_id:       int,
    pad_id:          int,
    prompt_end:      int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replace the CONTENT of selected steps with latent marker spans.

    Replacement: [start_id] [latent_id]*(c-2) [end_id]
    where c = original content length, preserving sequence length.
    Minimum replacement: [start_id] [end_id] (for content_len == 1 or 2).

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

    # Mask any remaining positions that are still prompt (before think block)
    # Already handled above; also mask pad tokens
    for t, tok in enumerate(ids):
        if tok == pad_id:
            labels[t]       = -100
            loss_weights[t] = 0.0

    new_ids      = torch.tensor(ids,          dtype=torch.long)
    labels_t     = torch.tensor(labels,       dtype=torch.long)
    weights_t    = torch.tensor(loss_weights, dtype=torch.float)
    return new_ids, labels_t, weights_t


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
) -> List[str]:
    """
    Run DNA-conditioned sampling on a few val examples and print GT vs output.
    Prompts the model with everything up to and including <think>\\n, then lets
    it generate freely — reveals whether step structure and Answer: are preserved.
    Returns the generated strings for WandB logging.
    model.generate() uses inputs_embeds internally, so output is new tokens only.
    """
    model.eval()
    tokenizer = processor.tokenizer
    pad_id    = tokenizer.pad_token_id or 0

    # Stop on both <|endoftext|> and <|im_end|> so Qwen chat termination works
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids  = list({id for id in [tokenizer.eos_token_id, im_end_id]
                      if id is not None and id != tokenizer.unk_token_id})

    think_tag = "<think>\n"
    indices   = random.sample(range(len(val_rows)), min(n_samples, len(val_rows)))

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

        bad_words = [[t] for t in suppress_latent_ids] if suppress_latent_ids else None
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

        print(f"\n--- Sample {i + 1} ---")
        print(f"[GT ]\n{_compact_dna(gt_text)}")
        print(f"\n[GEN]\n<think>\n{generated}")  # prepend <think> (it was in the prompt)

    print(sep + "\n")
    model.train()
    return generated_texts


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
            attention_mask = (batch_ids != pad_id).long(),
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

    # Inline entropy: one forward pass (no grad)
    if entropy_map is None:
        with torch.no_grad():
            dna_tok_d = ({k: v.to(device) for k, v in dna_tokenized.items()}
                         if dna_tokenized is not None else None)
            out       = model(
                input_ids      = batch_ids.to(device),
                attention_mask = (batch_ids != pad_id).long().to(device),
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
            # Mask pad tokens
            for t, tok in enumerate(batch_ids[b].tolist()):
                if tok == pad_id:
                    lbl[t] = -100; wgt[t] = 0.0
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

    # Load Stage 1 weights
    ckpt  = torch.load(args.stage1_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    clean = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, _ = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"  Missing keys (expected if no gate): {missing[:3]}")
    print(f"[Stage1.5] Loaded Stage 1: {args.stage1_ckpt}")

    tokenizer = model.processor.tokenizer
    start_id, end_id, latent_id = ensure_latent_tokens(tokenizer, model)
    pad_id = tokenizer.pad_token_id or 0

    # Freeze DNA encoder. Evo2 wraps the actual nn.Module at .model;
    # dna_llm.py accesses params as dna_model.model.parameters() throughout.
    if hasattr(model, "dna_model") and model.dna_model is not None:
        inner = getattr(model.dna_model, "model", model.dna_model)
        for p in inner.parameters():
            p.requires_grad_(False)

    # ── DNA embedding cache ────────────────────────────────────────────────────
    if args.dna_cache:
        print(f"[Stage1.5] Loading DNA embedding cache: {args.dna_cache}")
        _dna_cache = torch.load(args.dna_cache, map_location="cpu")
        print(f"[Stage1.5] Cache loaded: {len(_dna_cache)} sequences")

        _c = _dna_cache
        def _cached_evo2_embed(self, input_ids: torch.Tensor, layer_name: str) -> torch.Tensor:
            key = input_ids.cpu().numpy().tobytes()
            emb = _c.get(key)
            if emb is None:
                raise KeyError(
                    f"DNA sequence not found in cache (shape={input_ids.shape}). "
                    "Re-run precompute_dna_embeddings.py with the same --max_length_dna."
                )
            _p = next(self.dna_projection.parameters())
            return emb.to(device=_p.device, dtype=_p.dtype)

        import types as _types
        model._evo2_embed = _types.MethodType(_cached_evo2_embed, model)

        if model.dna_is_evo2 and getattr(model, "dna_model", None) is not None:
            model.dna_model.model.cpu()
            torch.cuda.empty_cache()
            print("[Stage1.5] Evo2 offloaded to CPU — GPU VRAM freed (~14 GB for Evo2 7B)")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.kegg_csv:
        print(f"[Stage1.5] Loading KEGG from CSV: {args.kegg_csv}")
        all_rows, val_rows = load_kegg_csv(args.kegg_csv, args.truncate_dna_per_side)
    else:
        print(f"[Stage1.5] Loading KEGG dataset: {args.kegg_dataset} ...")
        all_rows, val_rows = load_kegg_hf(args.kegg_dataset, cache_dir=args.cache_dir,
                                           truncate_dna_per_side=args.truncate_dna_per_side)
    print(f"[Stage1.5] train={len(all_rows)}  val={len(val_rows)}")

    val_dataset = KeggRawDataset(val_rows, model.processor, args.max_length_text, args.max_length_dna)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # steps_per_s: one cosine cycle per curriculum level (restart at each s)
    steps_per_s = (
        len(all_rows) // args.batch_size + 1
    ) * args.passes_per_step

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
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

    # ── Outer curriculum loop: s = 1 .. max_latent_steps ─────────────────────
    for s in range(1, args.max_latent_steps + 1):
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

        # Global mode: compute epoch-level entropies ONCE before this curriculum step
        entropy_map: Optional[Dict] = None
        if args.entropy_mode == "global":
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
                    )

                    # Log one decoded sample per curriculum step to verify injection
                    if not logged_sample:
                        log_latent_sample(tokenizer, batch_ids[0], new_ids[0], s)
                        logged_sample = True

                    # Forward pass with DNA embeddings injected at <|dna_pad|> positions
                    out  = model(
                        input_ids      = new_ids,
                        attention_mask = (new_ids != pad_id).long(),
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

                if args.sample_every > 0 and global_step % args.sample_every == 0:
                    gens = generate_samples(
                        model, model.processor, val_rows, device,
                        n_samples           = args.n_gen_samples,
                        max_new_tokens      = args.gen_max_new_tokens,
                        label               = f"s={s} step={global_step}",
                        suppress_latent_ids = [start_id, end_id, latent_id],
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "samples/generated": wandb.Table(
                                columns=["step", "generated"],
                                data=[[global_step, g] for g in gens],
                            )
                        }, step=global_step)

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
                        attention_mask = (new_ids_v != pad_id).long(),
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
                label               = f"s={s} pass={inner_pass+1} [end-of-pass]",
                suppress_latent_ids = [start_id, end_id, latent_id],
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "samples/end_of_pass": wandb.Table(
                        columns=["s", "pass", "generated"],
                        data=[[s, inner_pass + 1, g] for g in gens],
                    )
                }, step=global_step)

            # Save checkpoint
            ckpt_dir = os.path.join(args.output_dir, f"s{s:02d}_pass{inner_pass+1:02d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Saved → {ckpt_dir}/model.pt")

            if avg_val < best_val:
                best_val = avg_val
                best_dir = os.path.join(args.output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(best_dir, "model.pt"))
                tokenizer.save_pretrained(best_dir)
                print(f"  ★ New best val_loss={best_val:.4f} → {best_dir}/")

        # After final curriculum step: also refresh data by re-shuffling
        random.shuffle(all_rows)

    # ── Save final ────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model.pt"))
    tokenizer.save_pretrained(final_dir)
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
    p.add_argument("--output_dir",            required=True)
    p.add_argument("--entropy_mode",          default="global",
                   choices=["global", "inline"],
                   help="global: epoch-level recompute; inline: per-batch (2x compute)")
    p.add_argument("--max_latent_steps",      type=int,   default=4,
                   help="S in LatentSp Algorithm 1: outer curriculum steps")
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
    p.add_argument("--dna_cache", default=None,
                   help="Path to precomputed Evo2 embeddings (.pt from precompute_dna_embeddings.py). "
                        "When set, Evo2 is never called during training.")
    p.add_argument("--seed",                  type=int,   default=42,
                   help="Random seed for reproducibility (data shuffling, sampling)")
    p.add_argument("--wandb_project",         default=None)
    p.add_argument("--wandb_entity",          default=None)
    p.add_argument("--sample_every",          type=int,   default=200,
                   help="Log sample generations every N steps during training (0=disable)")
    p.add_argument("--n_gen_samples",         type=int,   default=2,
                   help="Number of val examples to generate from")
    p.add_argument("--gen_max_new_tokens",    type=int,   default=800,
                   help="Max new tokens per generation sample")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
