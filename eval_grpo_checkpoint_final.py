"""
eval_stage3_w9.py — Evaluation of a Stage 3 w9 (or optB) checkpoint.

Loads the w9 checkpoint — supports pytorch_model.bin, model.safetensors, model.pt,
and sharded variants for LLM/LoRA weights, plus thinking_gate.pt and dna_injector.pt —
then runs greedy generation and reports:

  Accuracy   — fraction of correct disease predictions
  Precision  — macro-averaged over classes
  Recall     — macro-averaged over classes
  F1         — macro-averaged over classes
  Per-class  — P / R / F1 / support for every disease

Outputs:
  <output_prefix>_metrics.json   — aggregate + per-class metrics
  <output_prefix>_predictions.csv — one row per sample (question, gt, pred, correct, ...)

Works for both the w9 base script and the optB (learnable theta_low) variant.
For optB, pass --theta_low_pt <checkpoint>/theta_low.pt to restore the learned value.

Usage (single GPU):
  python eval_stage3_w9.py \\
      --checkpoint /scratch/.../stage3_grpo_w9/checkpoint-1158 \\
      --dataset_name wanglab/kegg \\
      --split test

Usage (2 GPUs via accelerate):
  accelerate launch --num_processes 2 eval_stage3_w9.py \\
      --checkpoint /scratch/.../checkpoint-1158 \\
      --split both --n_samples -1

Usage (optB):
  python eval_stage3_w9.py \\
      --checkpoint /scratch/.../stage3_grpo_optB/checkpoint-1158 \\
      --theta_low_pt /scratch/.../stage3_grpo_optB/checkpoint-1158/theta_low.pt \\
      --split test
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BatchEncoding

from genomorph.dataset.kegg import format_kegg_for_dna_llm, load_kegg_from_anon_csv
from genomorph.dataset.utils import truncate_dna
from genomorph.dna_modules import NucleotideDNAModule
from genomorph.models.dna_llm import DNALLMModel, get_target_modules
from genomorph.models.evo2_tokenizer import register_evo2_tokenizer
from genomorph.models.thinking_residual import ThinkingResidualGate
from adaptive_latent_grpo import _load_sft_checkpoint
from adaptive_latent_grpo import ManifoldGateW6, _build_u_dna
from train_grpo_latent_reasoning import (
    DNAHiddenInjector,
    LatentSpController,
    patch_model_for_dual_mode_w9,
    _apply_dna_cache,
    _LATENT_START_TOKEN,
    _LATENT_END_TOKEN,
    _LATENT_PAD_TOKEN,
    MAX_GATE_FACTOR,
)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

register_evo2_tokenizer()
_dna_module = NucleotideDNAModule()


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a Stage 3 w9/optB checkpoint."
    )
    # Data
    p.add_argument("--checkpoint",    required=True,
                   help="Path to checkpoint directory (pytorch_model.bin, model.safetensors, model.pt, or sharded).")
    p.add_argument("--dataset_name",  default="wanglab/kegg",
                   help="HuggingFace dataset name (default: wanglab/kegg).")
    p.add_argument("--kegg_csv",      default=None,
                   help="Local anonymized CSV. Overrides --dataset_name when set.")
    p.add_argument("--split",         default="test", choices=["val", "test", "both"])
    p.add_argument("--n_samples",     type=int, default=-1,
                   help="Max samples to evaluate.  -1 = all.")
    p.add_argument("--seed",          type=int, default=42)

    # Generation
    p.add_argument("--max_new_tokens",     type=int,   default=800)
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="Match training validation: GRPOConfig default is 1.0.")
    p.add_argument("--temperature",        type=float, default=0.0,
                   help="0 = greedy (default for eval).")

    # Model architecture — must match training
    p.add_argument("--text_model_name",       default="Qwen/Qwen3-1.7B")
    p.add_argument("--dna_model_name",        default="evo2_7b_base")
    p.add_argument("--dna_is_evo2",           type=lambda x: x.lower() == "true",
                                              default=True)
    p.add_argument("--dna_embedding_layer",   default="blocks.28.mlp.l3")
    p.add_argument("--use_cross_attention",   type=lambda x: x.lower() == "true",
                                              default=True)
    p.add_argument("--cache_dir",             default="~/.cache/huggingface")
    p.add_argument("--max_length_text",       type=int, default=6000)
    p.add_argument("--max_length_dna",        type=int, default=2048)
    p.add_argument("--truncate_dna_per_side", type=int, default=1024)
    p.add_argument("--lora_r",                type=int, default=16)
    p.add_argument("--lora_alpha",            type=int, default=32)

    # w9 LatentSp thresholds (used if --theta_low_pt not provided)
    p.add_argument("--theta_low",      type=float, default=1.0,
                   help="Entropy threshold for latent steps.")
    p.add_argument("--theta_high",     type=float, default=3.0,
                   help="Entropy threshold for DNA injection.")
    p.add_argument("--max_consec",     type=int,   default=3,
                   help="Max consecutive latent steps.")
    p.add_argument("--lookahead_k",    type=int,   default=3)

    # optB: load learned theta_low from checkpoint
    p.add_argument("--theta_low_pt",   default=None,
                   help="Path to theta_low.pt (optB). Overrides --theta_low when set.")

    # Auxiliary module checkpoints (default: look inside --checkpoint dir)
    p.add_argument("--gate_pt",      default=None,
                   help="Path to thinking_gate.pt. Default: <checkpoint>/thinking_gate.pt")
    p.add_argument("--injector_pt",  default=None,
                   help="Path to dna_injector.pt. Default: <checkpoint>/dna_injector.pt")
    p.add_argument("--dna_cache",    default=None,
                   help="Path to precomputed DNA embedding cache (.pt).")
    p.add_argument("--stage2_dir",   default=None,
                   help="Stage 2 output dir for OT manifold (optional).")

    # Output
    p.add_argument("--output_prefix", default=None,
                   help="Prefix for output files. Auto-generated if not set.")
    p.add_argument("--output_dir",    default="eval_results",
                   help="Directory for output files (default: eval_results/).")

    return p.parse_args()


# ── Model construction (mirrors training setup) ───────────────────────────────

def build_model(args, device):
    model = DNALLMModel(
        text_model_name      = args.text_model_name,
        dna_model_name       = args.dna_model_name,
        cache_dir            = args.cache_dir,
        max_length_text      = args.max_length_text,
        max_length_dna       = args.max_length_dna,
        text_model_finetune  = True,
        dna_model_finetune   = False,
        dna_is_evo2          = args.dna_is_evo2,
        dna_embedding_layer  = args.dna_embedding_layer,
        use_cross_attention  = args.use_cross_attention,
        use_hrpo_gate        = True,
        use_dna_gate         = False,
        device               = "cuda",
    ).to(device)
    target_modules = get_target_modules(model)
    lora_config = LoraConfig(
        r                 = args.lora_r,
        lora_alpha        = args.lora_alpha,
        lora_dropout      = 0.0,
        target_modules    = target_modules,
        init_lora_weights = "gaussian",
        bias              = "none",
        task_type         = "CAUSAL_LM",
    )
    model.text_model = prepare_model_for_kbit_training(model.text_model)
    model.text_model = get_peft_model(model.text_model, lora_config)
    return model


def _load_llm_weights(ckpt: str, model, device):
    """Load LLM/LoRA weights from checkpoint dir.

    Priority:
      1. pytorch_model.bin      — SaveWithPyTorchCallback output
      2. model.safetensors      — HF Trainer default (safetensors)
      3. model.pt               — manual torch.save
      4. model.safetensors shards — pytorch_model-00001-of-NNNNN.safetensors
      5. pytorch_model shards   — pytorch_model-00001-of-NNNNN.bin
    """
    import glob as _glob

    # 1. pytorch_model.bin
    p = os.path.join(ckpt, "pytorch_model.bin")
    if os.path.exists(p):
        sd = torch.load(p, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded LLM weights ← {p}")
        return

    # 2. model.safetensors (single-file safetensors)
    p = os.path.join(ckpt, "model.safetensors")
    if os.path.exists(p):
        try:
            from safetensors.torch import load_file as _st_load
            sd = _st_load(p, device=str(device))
            model.load_state_dict(sd, strict=False)
            print(f"  Loaded LLM weights ← {p}")
            return
        except ImportError:
            pass  # fall through to torch.load attempt
        sd = torch.load(p, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded LLM weights ← {p}")
        return

    # 3. model.pt
    p = os.path.join(ckpt, "model.pt")
    if os.path.exists(p):
        sd = torch.load(p, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded LLM weights ← {p}")
        return

    # 4. sharded safetensors (model-00001-of-NNNNN.safetensors or pytorch_model-…)
    for pat in ("model-*-of-*.safetensors", "pytorch_model-*-of-*.safetensors"):
        shards = sorted(_glob.glob(os.path.join(ckpt, pat)))
        if shards:
            try:
                from safetensors.torch import load_file as _st_load
                merged = {}
                for s in shards:
                    merged.update(_st_load(s, device=str(device)))
                model.load_state_dict(merged, strict=False)
                print(f"  Loaded LLM weights (sharded, {len(shards)} files) ← {ckpt}/{pat}")
                return
            except ImportError:
                break  # safetensors not installed, try .bin shards

    # 5. sharded .bin
    shards = sorted(_glob.glob(os.path.join(ckpt, "pytorch_model-*-of-*.bin")))
    if shards:
        merged = {}
        for s in shards:
            merged.update(torch.load(s, map_location=device, weights_only=True))
        model.load_state_dict(merged, strict=False)
        print(f"  Loaded LLM weights (sharded, {len(shards)} .bin files) ← {ckpt}")
        return

    sys.exit(
        f"ERROR: no model weights found in {ckpt}\n"
        f"  Expected one of: pytorch_model.bin, model.safetensors, model.pt, "
        f"or sharded safetensors/bin files.\n"
        f"  Files present: {os.listdir(ckpt) if os.path.isdir(ckpt) else '<dir not found>'}"
    )


def load_w9_checkpoint(model, thinking_gate, dna_injector, args, device):
    """Load LLM/LoRA weights, thinking_gate.pt and dna_injector.pt from checkpoint dir."""
    ckpt = args.checkpoint

    # LLM / LoRA weights (multi-format)
    _load_llm_weights(ckpt, model, device)

    # ThinkingResidualGate
    gate_pt = args.gate_pt or os.path.join(ckpt, "thinking_gate.pt")
    if os.path.exists(gate_pt):
        thinking_gate.load_state_dict(torch.load(gate_pt, map_location=device), strict=False)
        print(f"  Loaded thinking_gate ← {gate_pt}")
    else:
        print(f"  WARNING: thinking_gate.pt not found ({gate_pt}) — using fresh init")

    # DNAHiddenInjector
    inj_pt = args.injector_pt or os.path.join(ckpt, "dna_injector.pt")
    if os.path.exists(inj_pt):
        dna_injector.load_state_dict(torch.load(inj_pt, map_location=device), strict=False)
        print(f"  Loaded dna_injector  ← {inj_pt}")
    else:
        print(f"  WARNING: dna_injector.pt not found ({inj_pt}) — using fresh init")


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_eval_records(args, rank=0):
    """Return list of formatted samples for the requested split(s)."""
    if args.kegg_csv:
        raw = load_kegg_from_anon_csv(args.kegg_csv)
    else:
        if load_dataset is None:
            sys.exit("ERROR: `datasets` not installed; use --kegg_csv instead.")
        raw = load_dataset(args.dataset_name, "default", cache_dir=args.cache_dir)

    splits_to_use = ["val", "test"] if args.split == "both" else [args.split]
    records = []
    for sp in splits_to_use:
        if sp in raw:
            sp_data = raw[sp]
            if args.truncate_dna_per_side > 0:
                sp_data = sp_data.map(
                    truncate_dna,
                    fn_kwargs={"truncate_dna_per_side": args.truncate_dna_per_side}
                )
            for ex in sp_data:
                item = format_kegg_for_dna_llm(ex, is_sft=False)
                # Keep the raw question text for the CSV
                item["_question_text"] = ex.get("question", "")
                item["_split"] = sp
                records.append(item)
        elif rank == 0:
            print(f"WARNING: split '{sp}' not found in dataset")

    if args.n_samples > 0:
        random.seed(args.seed)
        records = random.sample(records, min(args.n_samples, len(records)))
    return records


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_answer(model, sample, args, device):
    """Run generate_with_hrpo_gate on a single sample; return (text, gen_meta)."""
    prompts_text = _dna_module.prepare_prompt(
        processing_class=model.processor, inputs=[sample]
    )
    batch = model.processor(
        text                = prompts_text,
        batch_dna_sequences = [sample["dna_sequences"]],
        return_tensors      = "pt",
        padding             = False,
        padding_side        = "left",
        add_special_tokens  = False,
        max_length_text     = model.max_length_text,
        max_length_dna      = model.max_length_dna,
    )
    input_ids     = batch["input_ids"].to(device)
    attn_mask     = torch.ones_like(input_ids)
    dna_tokenized = batch.get("dna_tokenized")
    batch_idx_map = batch.get("batch_idx_map", [])
    if dna_tokenized is not None:
        dna_tokenized = BatchEncoding({
            "input_ids":      dna_tokenized["input_ids"].to(device),
            "attention_mask": dna_tokenized["attention_mask"].to(device),
        })

    gen_kwargs = {
        "max_new_tokens":     args.max_new_tokens,
        "do_sample":          False,
        "repetition_penalty": args.repetition_penalty,
    }

    out = model.generate_with_hrpo_gate(
        input_ids     = input_ids,
        attention_mask = attn_mask,
        dna_tokenized  = dna_tokenized,
        batch_idx_map  = batch_idx_map,
        **gen_kwargs,
    )
    text     = model.text_tokenizer.decode(out[0], skip_special_tokens=False).strip()
    gen_meta = getattr(model, "_w9_gen_meta", [])
    return text, gen_meta


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results):
    """Accuracy + macro precision/recall/F1 across all disease classes."""
    total   = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    acc     = correct / total if total > 0 else 0.0

    # Per-class TP / FP / FN counts
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for r in results:
        gt   = r["ground_truth"]
        pred = r["predicted_answer"]
        # Bidirectional substring match (matches is_correct above); empty pred
        # is never a match.
        if pred and (gt in pred or pred in gt):
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pred] = fp.get(pred, 0) + 1

    classes = sorted(set(r["ground_truth"] for r in results))
    per_class = {}
    precisions, recalls, f1s = [], [], []
    for cls in classes:
        p  = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        r  = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[cls] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "support":   tp[cls] + fn[cls],
        }
        precisions.append(p); recalls.append(r); f1s.append(f1)

    from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["ground_truth"] if r["is_correct"] else r["predicted_answer"] for r in results]
    true_labels = sorted(set(y_true))

    n = len(classes)
    return {
        "accuracy":          round(acc, 4),
        "macro_precision":   round(sum(precisions) / n, 4) if n else 0.0,
        "macro_recall":      round(sum(recalls)    / n, 4) if n else 0.0,
        "macro_f1":          round(sum(f1s)        / n, 4) if n else 0.0,
        "weighted_precision": round(float(precision_score(y_true, y_pred, labels=true_labels, average="weighted", zero_division=0)), 4),
        "weighted_recall":    round(float(recall_score(y_true, y_pred, labels=true_labels, average="weighted", zero_division=0)), 4),
        "weighted_f1":        round(float(sk_f1(y_true, y_pred, labels=true_labels, average="weighted", zero_division=0)), 4),
        "correct":           correct,
        "total":             total,
        "n_classes":         n,
        "per_class":         per_class,
    }


# ── CSV writer ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "index", "split", "question", "ground_truth", "predicted_answer",
    "is_correct", "n_latent_steps", "n_dna_injected", "gen_length_tokens",
    "gen_time_sec", "full_generation",
]


def write_csv(results, csv_path):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"  Predictions CSV → {csv_path}")


# ── Distributed gather ────────────────────────────────────────────────────────

def gather_results(local_results, device):
    if not (dist.is_available() and dist.is_initialized()):
        return local_results
    world_size = dist.get_world_size()
    local_bytes = pickle.dumps(local_results)
    size_tensor = torch.tensor([len(local_bytes)], dtype=torch.long, device=device)
    size_list   = [torch.zeros(1, dtype=torch.long, device=device)
                   for _ in range(world_size)]
    dist.all_gather(size_list, size_tensor)
    max_size = max(s.item() for s in size_list)
    buf      = torch.zeros(max_size, dtype=torch.uint8, device=device)
    buf[:len(local_bytes)] = torch.frombuffer(local_bytes, dtype=torch.uint8)
    buf_list = [torch.zeros(max_size, dtype=torch.uint8, device=device)
                for _ in range(world_size)]
    dist.all_gather(buf_list, buf)
    all_results = []
    for b, s in zip(buf_list, size_list):
        all_results += pickle.loads(bytes(b[:s.item()].cpu().tolist()))
    return all_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Distributed init
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()       if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  eval_stage3_w9.py")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Split:      {args.split}")
        print(f"  GPUs:       {world_size}")
        print(f"{'='*60}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    records = load_eval_records(args, rank=rank)
    if rank == 0:
        print(f"Loaded {len(records)} samples from split '{args.split}'")

    # Each rank handles its own shard
    my_records = [records[i] for i in range(rank, len(records), world_size)]
    if rank == 0:
        print(f"Rank distribution: ~{len(my_records)} samples per GPU")

    # ── Build model ───────────────────────────────────────────────────────────
    if rank == 0:
        print("\nBuilding model...")
    model = build_model(args, device)

    # ── Thinking gate + DNA injector ──────────────────────────────────────────
    manifold_gate = ManifoldGateW6(stage2_dir=args.stage2_dir)
    use_ot_dist   = manifold_gate.mode != "onthefly"
    thinking_gate = ThinkingResidualGate(
        hidden_size = model.text_hidden_size,
        use_ot_dist = use_ot_dist,
        r_min       = 0.5,
        r_max       = 0.99,
    ).to(device)
    dna_injector = DNAHiddenInjector(
        hidden_size = model.text_hidden_size,
        r_min       = 0.7,
        r_max       = 0.99,
    ).to(device)

    # ── Latent token vocab — must happen BEFORE loading checkpoint ───────────────
    # Training added 3 latent tokens and called resize_token_embeddings, so the
    # checkpoint's embed_tokens/lm_head have shape [151675, 2048].  If we resize
    # after loading we get a shape mismatch RuntimeError.
    tokenizer    = model.processor.tokenizer
    _all_latent  = [_LATENT_START_TOKEN, _LATENT_END_TOKEN, _LATENT_PAD_TOKEN]
    _missing_tok = [t for t in _all_latent
                    if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if _missing_tok:
        tokenizer.add_special_tokens({"additional_special_tokens": _missing_tok})
        model.text_model.resize_token_embeddings(len(tokenizer))
        if rank == 0:
            print(f"  Resized vocab: +{len(_missing_tok)} latent tokens "
                  f"→ {len(tokenizer)}")
    latent_start_id = tokenizer.convert_tokens_to_ids(_LATENT_START_TOKEN)
    latent_end_id   = tokenizer.convert_tokens_to_ids(_LATENT_END_TOKEN)

    # ── Load checkpoint weights ───────────────────────────────────────────────
    if rank == 0:
        print("\nLoading checkpoint...")
    load_w9_checkpoint(model, thinking_gate, dna_injector, args, device)

    # ── LatentSp controller ───────────────────────────────────────────────────
    theta_low = args.theta_low
    if args.theta_low_pt and os.path.exists(args.theta_low_pt):
        saved = torch.load(args.theta_low_pt, map_location="cpu")
        theta_low = float(saved["theta_low_param"].item())
        if rank == 0:
            print(f"  Loaded learned theta_low={theta_low:.4f} ← {args.theta_low_pt}")

    latentSp_ctrl = LatentSpController(
        theta_low       = theta_low,
        theta_high      = args.theta_high,
        max_consecutive = args.max_consec,
    )
    # Always active during eval (no warmup scale)
    if rank == 0:
        print(f"  LatentSp: theta_low={theta_low:.3f}  "
              f"theta_high={args.theta_high:.1f}  "
              f"max_consec={args.max_consec}")

    # ── Patch model with w9 generation loop ───────────────────────────────────
    patch_model_for_dual_mode_w9(
        model, thinking_gate, dna_injector, latentSp_ctrl,
        latent_start_id, latent_end_id,
        max_new_tokens = args.max_new_tokens,
        lookahead_k    = args.lookahead_k,
    )
    model._gate_warmup_factor     = MAX_GATE_FACTOR
    model._gate_warmup_min_factor = MAX_GATE_FACTOR

    if args.dna_cache and os.path.exists(args.dna_cache):
        _apply_dna_cache(model, args.dna_cache)

    model.eval()
    model.text_model.config.use_cache = True

    # ── Run generation ────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\nRunning generation on {len(my_records)} samples (rank 0)...")
    local_results = []
    for i, sample in enumerate(my_records):
        global_idx = rank + i * world_size
        try:
            t0 = time.perf_counter()
            text, gen_meta = generate_answer(model, sample, args, device)
            gen_time = time.perf_counter() - t0

            extracted  = NucleotideDNAModule._extract_xml_answer(text)
            gt         = sample.get("answer", "").strip().lower()
            _pred      = extracted.lower().strip()
            # Bidirectional substring match (mirrors eval_stage1_5_checkpoints.py
            # is_correct): count correct if gt ⊆ pred OR pred ⊆ gt. Empty pred is
            # never correct (guards against "" ⊆ gt being trivially True).
            is_correct = bool(_pred) and (gt in _pred or _pred in gt)

            # Latent / injection stats from gen_meta
            n_latent   = sum(1 for m in gen_meta if m.get("is_latent", False))
            n_injected = sum(1 for m in gen_meta if m.get("dna_injected", False))
            gen_len    = len(gen_meta)

            mark = "+" if is_correct else "-"
            print(
                f"[rank{rank} {i+1:4d}/{len(my_records)}] [{mark}]  "
                f"gt={repr(gt):<30s}  "
                f"pred={repr(extracted[:50]):<52s}  "
                f"latent={n_latent}  inject={n_injected}  "
                f"time={gen_time:.2f}s  tok={gen_len}",
                flush=True,
            )
            local_results.append({
                "index":             global_idx,
                "split":             sample.get("_split", args.split),
                "question":          sample.get("_question_text", ""),
                "ground_truth":      gt,
                "predicted_answer":  extracted.lower(),
                "is_correct":        is_correct,
                "n_latent_steps":    n_latent,
                "n_dna_injected":    n_injected,
                "gen_length_tokens": gen_len,
                "gen_time_sec":      round(gen_time, 3),
                "full_generation":   text,
            })
        except Exception as exc:
            print(f"[rank{rank} {i+1}/{len(my_records)}] ERROR: {exc}", flush=True)
            local_results.append({
                "index":            global_idx,
                "split":            sample.get("_split", args.split),
                "question":         sample.get("_question_text", ""),
                "ground_truth":     sample.get("answer", "").strip().lower(),
                "predicted_answer": "",
                "is_correct":       False,
                "n_latent_steps":   0,
                "n_dna_injected":   0,
                "gen_length_tokens": 0,
                "gen_time_sec":     0.0,
                "full_generation":  f"ERROR: {exc}",
            })

    # ── Gather across GPUs → rank 0 ───────────────────────────────────────────
    results = gather_results(local_results, device)
    if rank != 0:
        return
    results.sort(key=lambda r: r["index"])

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(results)

    n_latent_total = sum(r["n_latent_steps"]    for r in results)
    n_inject_total = sum(r["n_dna_injected"]    for r in results)
    mean_gen_len   = sum(r["gen_length_tokens"] for r in results) / max(len(results), 1)

    # Timing stats — exclude error samples (gen_length_tokens==0, gen_time_sec==0.0)
    timed_results   = [r for r in results if r["gen_length_tokens"] > 0]
    total_time      = sum(r["gen_time_sec"] for r in timed_results)
    mean_time       = total_time / max(len(timed_results), 1)
    latent_samples  = [r for r in timed_results if r["n_latent_steps"] > 0]
    normal_samples  = [r for r in timed_results if r["n_latent_steps"] == 0]
    mean_time_lat   = (sum(r["gen_time_sec"] for r in latent_samples)
                       / max(len(latent_samples), 1))
    mean_time_norm  = (sum(r["gen_time_sec"] for r in normal_samples)
                       / max(len(normal_samples), 1))
    mean_tok_lat    = (sum(r["gen_length_tokens"] for r in latent_samples)
                       / max(len(latent_samples), 1))
    mean_tok_norm   = (sum(r["gen_length_tokens"] for r in normal_samples)
                       / max(len(normal_samples), 1))
    # sec/token — shows raw throughput independent of output length
    tpt_lat  = mean_time_lat  / max(mean_tok_lat,  1)
    tpt_norm = mean_time_norm / max(mean_tok_norm, 1)
    speedup  = mean_time_norm / mean_time_lat if mean_time_lat > 0 else float("nan")

    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}  |  Samples: {metrics['total']}")
    print(f"{'='*60}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}  "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"  Precision  : {metrics['macro_precision']:.4f}  (macro)   {metrics['weighted_precision']:.4f}  (weighted)")
    print(f"  Recall     : {metrics['macro_recall']:.4f}  (macro)   {metrics['weighted_recall']:.4f}  (weighted)")
    print(f"  F1         : {metrics['macro_f1']:.4f}  (macro)   {metrics['weighted_f1']:.4f}  (weighted)")
    print(f"  Classes    : {metrics['n_classes']}")
    print(f"\n  Generation stats:")
    print(f"    Mean gen length : {mean_gen_len:.1f} tokens")
    print(f"    Total latent    : {n_latent_total}  steps across all samples")
    print(f"    Total DNA inject: {n_inject_total}  steps across all samples")
    print(f"\n  Timing (wall clock, single-sample, rank 0 perspective):")
    print(f"    Total eval time    : {total_time:.1f}s  "
          f"({total_time/60:.1f} min)")
    print(f"    Mean time/sample   : {mean_time:.2f}s")
    print(f"    With latent steps  : {len(latent_samples):4d} samples  "
          f"mean={mean_time_lat:.2f}s  "
          f"mean_tok={mean_tok_lat:.0f}  "
          f"sec/tok={tpt_lat:.4f}")
    print(f"    Without latent     : {len(normal_samples):4d} samples  "
          f"mean={mean_time_norm:.2f}s  "
          f"mean_tok={mean_tok_norm:.0f}  "
          f"sec/tok={tpt_norm:.4f}")
    if len(latent_samples) > 0 and len(normal_samples) > 0:
        saved = mean_time_norm - mean_time_lat
        pct   = 100.0 * saved / max(mean_time_norm, 1e-9)
        print(f"    Latent saves       : {saved:+.2f}s/sample  ({pct:+.1f}%)  "
              f"[speedup={speedup:.2f}x vs non-latent]")
    print(f"\n  Per-class breakdown (top 20 by support):")
    sorted_classes = sorted(
        metrics["per_class"].items(),
        key=lambda kv: kv[1]["support"], reverse=True
    )
    for cls, m in sorted_classes[:20]:
        print(f"    {cls:<40s}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")
    if len(sorted_classes) > 20:
        print(f"    ... ({len(sorted_classes) - 20} more classes)")

    # ── Build output prefix ───────────────────────────────────────────────────
    out_dir    = args.output_dir
    ckpt_name  = os.path.basename(args.checkpoint.rstrip("/"))
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix     = args.output_prefix or f"stage3_w9_{ckpt_name}_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    base_path  = os.path.join(out_dir, prefix)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    json_path = base_path + "_metrics.json"
    payload = {
        "checkpoint":        args.checkpoint,
        "split":             args.split,
        "n_samples":         len(results),
        "timestamp":         datetime.now().isoformat(),
        "theta_low":         theta_low,
        "theta_high":        args.theta_high,
        "metrics":           metrics,
        "generation_stats":  {
            "mean_gen_length_tokens": round(mean_gen_len, 1),
            "total_latent_steps":     n_latent_total,
            "total_dna_injected":     n_inject_total,
        },
        "timing": {
            "total_eval_sec":           round(total_time, 2),
            "mean_time_per_sample_sec": round(mean_time, 3),
            "latent_samples":           len(latent_samples),
            "normal_samples":           len(normal_samples),
            "mean_time_latent_sec":     round(mean_time_lat,  3),
            "mean_time_normal_sec":     round(mean_time_norm, 3),
            "mean_tokens_latent":       round(mean_tok_lat,  1),
            "mean_tokens_normal":       round(mean_tok_norm, 1),
            "sec_per_token_latent":     round(tpt_lat,  5),
            "sec_per_token_normal":     round(tpt_norm, 5),
            "time_saved_per_sample_sec": round(mean_time_norm - mean_time_lat, 3),
            "speedup_vs_normal":        round(speedup, 3) if speedup == speedup else None,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Metrics JSON   → {json_path}")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    csv_path = base_path + "_predictions.csv"
    write_csv(results, csv_path)

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
