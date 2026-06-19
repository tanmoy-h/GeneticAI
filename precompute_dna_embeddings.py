"""
Precompute Evo2 DNA embeddings for KEGG dataset.

Since Evo2 is frozen during training, its outputs are deterministic.
This script runs Evo2 once over the full dataset and caches the raw
layer activations to disk so train_latent_sft_cached.py can skip the
Evo2 forward pass entirely, freeing ~14 GB GPU VRAM at training time.

Usage
-----
  python precompute_dna_embeddings.py \\
      --output_path /scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt

  # With local anonymized CSV instead of HuggingFace:
  python precompute_dna_embeddings.py \\
      --kegg_csv genomorph/dataset/kegg_curriculum/stage1_anon_genes_mol_keep_chr.csv \\
      --output_path /scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt

Cache format
------------
  dict[bytes, torch.Tensor]
    key:   input_ids[i:i+1].cpu().numpy().tobytes()  — shape [1, L]
    value: [L, H] float32 tensor on CPU

The key is the raw bytes of the [1, L] int64 numpy array, which exactly
matches what _evo2_embed receives inside process_dna_embeddings().
"""

import os
import argparse

import torch

from genomorph.models.dna_llm import DNALLMModel
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from genomorph.dataset.utils import truncate_dna

register_evo2_tokenizer()


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute Evo2 DNA embeddings for the KEGG dataset"
    )
    p.add_argument("--output_path",         required=True,
                   help="Where to save the cache (.pt file)")
    p.add_argument("--kegg_dataset",        default="wanglab/kegg",
                   help="HuggingFace dataset name (used when --kegg_csv is not set)")
    p.add_argument("--kegg_csv",            default=None,
                   help="Local anonymized KEGG CSV path (overrides --kegg_dataset)")
    p.add_argument("--text_model_name",     default="Qwen/Qwen3-1.7B")
    p.add_argument("--dna_model_name",      default="evo2_7b_base")
    p.add_argument("--dna_embedding_layer", default="blocks.28.mlp.l3")
    p.add_argument("--max_length_dna",        type=int, default=2048)
    p.add_argument("--truncate_dna_per_side", type=int, default=1024,
                   help="Remove this many characters from each end of the raw DNA string "
                        "before tokenization — must match the value used in training "
                        "(Stage 3 default: 1024; Stage 1.5 uses 0 until fixed).")
    p.add_argument("--cache_dir",             default="~/.cache/huggingface")
    p.add_argument("--device",                default="cuda")
    return p.parse_args()


def load_dataset_splits(args):
    """Load KEGG dataset and return a flat list of (ref_seq, var_seq) pairs.

    Collects from train + val splits (test is skipped if absent).
    """
    if args.kegg_csv:
        print(f"[precompute] Loading KEGG from CSV: {args.kegg_csv}")
        from genomorph.dataset.kegg import load_kegg_from_anon_csv
        ds = load_kegg_from_anon_csv(args.kegg_csv)
    else:
        print(f"[precompute] Loading KEGG from HuggingFace: {args.kegg_dataset}")
        from datasets import load_dataset
        ds = load_dataset(args.kegg_dataset, cache_dir=args.cache_dir)

    samples = []
    for split in ("train", "val", "test"):
        if split not in ds:
            print(f"[precompute] Split '{split}' not found — skipping")
            continue
        split_data = ds[split]
        print(f"[precompute] {split}: {len(split_data)} samples")
        for ex in split_data:
            samples.append(
                (ex.get("reference_sequence", ""),
                 ex.get("variant_sequence",   ""))
            )

    print(f"[precompute] Total samples collected: {len(samples)}")
    return samples


def main():
    args = parse_args()

    device = args.device
    print(f"[precompute] Device: {device}")

    # ── Build output directory ────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model (Evo2 + processor only; text model loads but stays unused) ─
    print("[precompute] Loading DNALLMModel ...")
    model = DNALLMModel(
        text_model_name     = args.text_model_name,
        dna_model_name      = args.dna_model_name,
        cache_dir           = args.cache_dir,
        max_length_dna      = args.max_length_dna,
        max_length_text     = 256,          # minimal; only processor tokenizer matters
        text_model_finetune = False,
        dna_model_finetune  = False,
        dna_is_evo2         = True,
        dna_embedding_layer = args.dna_embedding_layer,
        use_cross_attention = False,        # not needed for precomputation
        use_hrpo_gate       = False,
        device              = device,
    ).to(device)

    # Freeze everything — we are only doing inference
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    processor   = model.processor
    layer_name  = args.dna_embedding_layer

    # ── Load dataset ──────────────────────────────────────────────────────────
    samples = load_dataset_splits(args)

    # ── Load existing cache (resume if interrupted) ───────────────────────────
    cache: dict = {}
    if os.path.exists(args.output_path):
        print(f"[precompute] Resuming from existing cache: {args.output_path}")
        cache = torch.load(args.output_path, map_location="cpu")
        print(f"[precompute] Existing entries: {len(cache)}")

    # ── Template text that triggers DLProcessor DNA tokenization ─────────────
    # Two <|dna_pad|> placeholders in the user turn — one per DNA sequence —
    # exactly mirrors the format used by _make_row() in train_latent_sft*.py.
    DUMMY_TEXT = (
        "<|im_start|>user\n"
        "<|dna_pad|>\n"
        "<|dna_pad|>\n"
        "dummy\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    n_new     = 0
    n_skipped = 0

    print(f"[precompute] Starting embedding computation for {len(samples)} samples ...")

    for sample_idx, (ref_seq, var_seq) in enumerate(samples):
        # Apply string-level truncation matching Stage 3 (adaptive_latent_grpo.py)
        if args.truncate_dna_per_side > 0:
            ex = truncate_dna(
                {"reference_sequence": ref_seq, "variant_sequence": var_seq},
                truncate_dna_per_side=args.truncate_dna_per_side,
            )
            ref_seq, var_seq = ex["reference_sequence"], ex["variant_sequence"]
        dna_sequences = [ref_seq, var_seq]

        # Tokenize DNA via processor — keys will match training exactly
        batch = processor(
            text                = [DUMMY_TEXT],
            batch_dna_sequences = [dna_sequences],
            return_tensors      = "pt",
            padding             = False,
            add_special_tokens  = False,
            max_length_text     = 256,
            max_length_dna      = args.max_length_dna,
        )

        dna_input_ids  = batch["dna_tokenized"]["input_ids"]       # [2, L]
        dna_attn_mask  = batch["dna_tokenized"]["attention_mask"]   # [2, L]

        for i in range(dna_input_ids.shape[0]):
            seq_ids  = dna_input_ids[i : i + 1]   # [1, L]
            seq_mask = dna_attn_mask[i]             # [L]
            # Trim cross-sequence padding so the key matches what training sees
            actual_len = int(seq_mask.sum().item())
            if actual_len < seq_ids.shape[1]:
                seq_ids = seq_ids[:, :actual_len]  # [1, actual_len]
            key = seq_ids.cpu().numpy().tobytes()

            if key in cache:
                n_skipped += 1
                continue

            # Run Evo2 forward pass
            with torch.no_grad():
                emb = model._evo2_embed(seq_ids.to(device), layer_name)  # [actual_len, H]

            cache[key] = emb.cpu()
            n_new += 1

        if (sample_idx + 1) % 50 == 0:
            print(
                f"  [{sample_idx + 1}/{len(samples)}]  "
                f"new={n_new}  skipped={n_skipped}  "
                f"cache_size={len(cache)}"
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[precompute] Saving cache → {args.output_path}")
    print(f"[precompute] Total entries: {len(cache)}  (new={n_new}, skipped={n_skipped})")
    torch.save(cache, args.output_path)
    print("[precompute] Done.")


if __name__ == "__main__":
    main()
