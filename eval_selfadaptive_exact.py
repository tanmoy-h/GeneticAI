#!/usr/bin/env python3
"""
Exact-reproduction eval for a train_latent_sft.py checkpoint directory (Stage 1.5/1.51,
self-adaptive RFT, or any checkpoint saved by that script -- e.g.
train_07_selfadaptive/best_acc/): calls evaluate_accuracy() directly, the SAME
model.generate() + always-on DNA-injection forward that produced the accuracy/time
numbers reported during training's own half-epoch eval290 probe -- and dumps the
per-row reasoning traces via its trace_path option (evaluate_accuracy otherwise
discards `gen` after scoring).

Deliberately does NOT use eval_grpo_checkpoint_final.py's generate_with_hrpo_gate path:
that is a structurally different generation loop (entropy-conditional DNA injection vs.
evaluate_accuracy's always-on injection) and is not guaranteed to reproduce a specific
checkpoint's already-reported number -- see train/rft/eval_selfadaptive.sh for that path.

Usage:
  python eval_selfadaptive_exact.py \
      --checkpoint /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_07_selfadaptive/best_acc \
      --dataset_name wanglab/kegg \
      --dna_cache /scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt \
      --output_dir test/log --output_prefix rft_exact_best_acc

  # Anon set:
  python eval_selfadaptive_exact.py --checkpoint <...> \
      --kegg_csv genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv ...

  # Latent-free comparison number (bans the latent tokens):
  python eval_selfadaptive_exact.py --checkpoint <...> --ban_latent ...
"""
import argparse
import csv
import json
import os

import torch

from train_latent_sft import (
    DNALLMModel,
    ensure_latent_tokens,
    load_start_state_dict_from_dir,
    merge_grpo_lora_state_dict,
    load_eval_rows,
    evaluate_accuracy,
    _GATE_AVAILABLE,
    MAX_GATE_FACTOR,
)
if _GATE_AVAILABLE:
    from train_grpo_latent_reasoning import ThinkingResidualGate, DNAHiddenInjector, _make_dual_mode_forward_w9
import types as _types


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint DIRECTORY (e.g. train_07_selfadaptive/best_acc) -- "
                        "must contain model.pt/pytorch_model.bin/model.safetensors "
                        "(+ optional thinking_gate.pt/dna_injector.pt).")
    p.add_argument("--dataset_name", default="wanglab/kegg")
    p.add_argument("--kegg_csv", default=None, help="Local anonymized CSV; overrides --dataset_name.")
    p.add_argument("--cache_dir", default="~/.cache/huggingface")
    p.add_argument("--text_model_name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--dna_model_name", default="evo2_7b_base")
    p.add_argument("--dna_embedding_layer", default="blocks.28.mlp.l3")
    p.add_argument("--max_length_text", type=int, default=6000)
    p.add_argument("--max_length_dna", type=int, default=2048)
    p.add_argument("--truncate_dna_per_side", type=int, default=1024)
    p.add_argument("--start_lora_r", type=int, default=16,
                   help="Only matters if --checkpoint turns out to be PEFT-wrapped "
                        "(e.g. a GRPO checkpoint) -- safe no-op on a plain state dict.")
    p.add_argument("--start_lora_alpha", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=800)
    p.add_argument("--ban_latent", action="store_true", default=False,
                   help="Force a latent-free number instead of self-emit.")
    p.add_argument("--dna_cache", default=None,
                   help="Preloaded {input_ids_bytes: embedding} .pt map to skip live "
                        "Evo2 on cache hits (same file/format used during training).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="test/log")
    p.add_argument("--output_prefix", default=None,
                   help="Default: <checkpoint-basename>_exact")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(args.checkpoint):
        raise SystemExit(f"--checkpoint must be a directory (got a file): {args.checkpoint}")

    prefix = args.output_prefix or f"{os.path.basename(os.path.normpath(args.checkpoint))}_exact"
    os.makedirs(args.output_dir, exist_ok=True)
    trace_path   = os.path.join(args.output_dir, f"{prefix}_traces.jsonl")
    csv_path     = os.path.join(args.output_dir, f"{prefix}_predictions.csv")
    metrics_path = os.path.join(args.output_dir, f"{prefix}_metrics.json")

    print(f"[exact-eval] Checkpoint: {args.checkpoint}")
    print("[exact-eval] Loading model ...")
    model = DNALLMModel(
        text_model_name      = args.text_model_name,
        dna_model_name       = args.dna_model_name,
        cache_dir             = args.cache_dir,
        max_length_text       = args.max_length_text,
        max_length_dna        = args.max_length_dna,
        text_model_finetune   = True,
        dna_model_finetune    = False,
        dna_is_evo2           = True,
        dna_embedding_layer   = args.dna_embedding_layer,
        use_cross_attention   = True,
        use_hrpo_gate         = False,
        device                = device,
    ).to(device)
    tokenizer = model.processor.tokenizer

    # Dir-load path mirrors train_latent_sft.py's train() exactly (same purpose there:
    # RFT-on-GRPO / continued-training starts) -- ensure_latent_tokens BEFORE loading so
    # embed_tokens/lm_head shapes match, then merge_grpo_lora_state_dict is a safe no-op
    # on a plain (non-PEFT) state dict like best_acc/model.pt.
    ensure_latent_tokens(tokenizer, model)
    state = load_start_state_dict_from_dir(args.checkpoint)
    state = merge_grpo_lora_state_dict(state, lora_r=args.start_lora_r, lora_alpha=args.start_lora_alpha)
    clean = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[exact-eval] Loaded backbone weights (missing={len(missing)} unexpected={len(unexpected)})")
    if missing:
        print(f"  missing sample:    {missing[:3]}")
    if unexpected:
        print(f"  unexpected sample: {unexpected[:3]}")
    if len(missing) > 20:
        print(f"  WARNING: {len(missing)} missing keys -- this checkpoint likely did NOT "
              f"load correctly. Do not trust this run's results.")

    start_id, end_id, latent_id = ensure_latent_tokens(tokenizer, model)

    if hasattr(model, "dna_model") and model.dna_model is not None:
        inner = getattr(model.dna_model, "model", model.dna_model)
        for p_ in inner.parameters():
            p_.requires_grad_(False)

    # Gate + injector: auto-detected from the checkpoint dir (train_07_selfadaptive
    # always trains --use_gate, so this fires for it automatically).
    gate_pt = os.path.join(args.checkpoint, "thinking_gate.pt")
    inj_pt  = os.path.join(args.checkpoint, "dna_injector.pt")
    if os.path.isfile(gate_pt) and os.path.isfile(inj_pt):
        if not _GATE_AVAILABLE:
            raise ImportError("Checkpoint has a gate but train_grpo_latent_reasoning "
                               "(w9 gate module) isn't importable.")
        hidden_size   = model.text_hidden_size
        thinking_gate = ThinkingResidualGate(
            hidden_size=hidden_size, use_ot_dist=True, r_min=0.5, r_max=0.99).to(device)
        dna_injector  = DNAHiddenInjector(
            hidden_size=hidden_size, r_min=0.7, r_max=0.99).to(device)
        thinking_gate.load_state_dict(torch.load(gate_pt, map_location=device))
        dna_injector.load_state_dict(torch.load(inj_pt, map_location=device))
        model.forward = _types.MethodType(
            _make_dual_mode_forward_w9(thinking_gate, dna_injector), model)
        model._gate_warmup_factor = MAX_GATE_FACTOR
        print(f"[exact-eval] Gate enabled ← {gate_pt}")
    else:
        print(f"[exact-eval] No thinking_gate.pt/dna_injector.pt in {args.checkpoint} "
              f"-- running gate-free")

    model.eval()

    print("[exact-eval] Loading val+test (290) rows ...")
    rows = load_eval_rows(
        dataset_name          = args.dataset_name,
        kegg_csv              = args.kegg_csv,
        cache_dir              = args.cache_dir,
        truncate_dna_per_side  = args.truncate_dna_per_side,
    )
    print(f"[exact-eval] {len(rows)} rows")

    dna_cache = None
    if args.dna_cache and os.path.isfile(args.dna_cache):
        print(f"[exact-eval] Loading DNA cache: {args.dna_cache}")
        dna_cache = torch.load(args.dna_cache, map_location="cpu")

    ban_ids = [start_id, end_id, latent_id] if args.ban_latent else None

    metrics = evaluate_accuracy(
        model, model.processor, rows, device,
        max_new_tokens  = args.max_new_tokens,
        ban_latent_ids  = ban_ids,
        label           = f"exact-eval [{os.path.basename(os.path.normpath(args.checkpoint))}]",
        dna_cache        = dna_cache,
        trace_path       = trace_path,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[exact-eval] metrics → {metrics_path}")

    # Flat predictions.csv from the trace file, matching the shape other eval scripts
    # in this repo produce.
    with open(trace_path, encoding="utf-8") as f, open(csv_path, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["index", "question", "ground_truth", "predicted_answer",
                    "is_correct", "gen_time_sec", "full_generation"])
        for line in f:
            r = json.loads(line)
            w.writerow([r["index"], r.get("question", ""), r["ground_truth"],
                        r["predicted_answer"], r["is_correct"], r.get("gen_time_sec", ""),
                        r.get("full_generation", "")])
    print(f"[exact-eval] predictions → {csv_path}")


if __name__ == "__main__":
    main()
