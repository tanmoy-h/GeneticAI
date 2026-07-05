"""
Baseline evaluation of the 290 held-out records (test + val splits) using
Meditron (epfl-llm/meditron-7b) via local HuggingFace inference.

Usage:
    python testing/eval_meditron.py \
        --csv  genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv \
        --out  testing/logs/meditron_results.csv \
        --splits test val

Requires a CUDA GPU with ~14 GB VRAM for bfloat16 inference.

Output:
    <out>.csv            — per-sample predictions
    <out>_metrics.json   — accuracy / precision / recall / F1 per class
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    sys.exit("transformers/torch not found. Install with: pip install transformers torch")


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a computational genomics assistant. You will be given:
1. A reference DNA sequence and a variant DNA sequence for a gene.
2. A chromosome number and a gene network pathway definition using anonymized
   gene identifiers (GENE_N) and molecule identifiers (MOL_N).
3. A question asking you to identify the disease caused by the variant allele.

Reason step by step through the pathway logic. Then give your final answer on
its own line in exactly this format:
    Answer: <disease name>

Keep the disease name concise (e.g. "alzheimer's disease", "thyroid dyshormonogenesis").
"""

def build_user_message(question: str, ref_seq: str, var_seq: str,
                        dna_truncate: int = 500) -> str:
    # Center window on the first position where sequences differ.
    # Falls back to the sequence midpoint when they are identical.
    min_len  = min(len(ref_seq), len(var_seq))
    diff_pos = next((i for i in range(min_len) if ref_seq[i] != var_seq[i]), len(ref_seq) // 2)
    half     = dna_truncate // 2
    start    = max(0, diff_pos - half)
    end      = min(len(ref_seq), start + dna_truncate)
    start    = max(0, end - dna_truncate)
    prefix   = f"[+{start}bp] " if start > 0 else ""
    suffix   = "..." if end < len(ref_seq) else ""
    ref_short = prefix + ref_seq[start:end] + suffix
    var_short = prefix + var_seq[start:end] + suffix
    return (
        f"Reference sequence ({dna_truncate} bp around variant): {ref_short}\n"
        f"Variant sequence   ({dna_truncate} bp around variant): {var_short}\n\n"
        f"{question}"
    )

def build_prompt(tokenizer, user_msg: str) -> str:
    # Meditron (LLaMA-2 based) uses the LLaMA-2 chat format.
    # apply_chat_template handles it; fallback uses the Meditron paper format.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Meditron paper format
        prompt = (
            f"<s> ### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Question:\n{user_msg}\n\n"
            f"### Response:\n"
        )
    return prompt


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    # Explicit Answer: tag (last occurrence, in case model repeats it)
    matches = list(re.finditer(r'[Aa]nswer:\s*(.+?)(?:\n|$)', text))
    if matches:
        return matches[-1].group(1).strip()
    # Prose: 'disease "X"' or "disease 'X'"
    m = re.search(r'disease\s+["\x27]([^"\x27\n]+)["\x27]', text, re.I)
    if m:
        return m.group(1).strip()
    # Prose: "cause/contributes to/results in/leads to [the [disease]] X."
    m = re.search(
        r'(?:cause[sd]?|contributes?\s+to|results?\s+in|leads?\s+to|associated\s+with)'
        r'\s+(?:the\s+(?:disease\s+)?)?([a-z][^.\n]{3,80})(?:\.|$)',
        text, re.I
    )
    if m:
        return m.group(1).strip().rstrip('"\'')
    return ""

def is_correct(pred: str, gt: str) -> bool:
    return bool(pred) and gt.lower() in pred.lower()


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_records(csv_path: str, splits: List[str]) -> List[Dict]:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] in splits:
                records.append(row)
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     required=True,
                   help="Path to global_stage1_anon_genes_mol_keep_chr.csv")
    p.add_argument("--out",     default="testing/logs/meditron_results.csv")
    p.add_argument("--splits",  nargs="+", default=["test", "val"])
    p.add_argument("--model",   default="epfl-llm/meditron-7b",
                   help="HuggingFace model ID (also supports epfl-llm/meditron-70b)")
    p.add_argument("--cache_dir", default=None,
                   help="HuggingFace cache directory")
    p.add_argument("--device",  default="cuda",
                   help="Inference device: cuda / cuda:N / cpu (default: cuda)")
    p.add_argument("--dtype",   default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--dna_truncate",   type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--resume",  action="store_true")
    p.add_argument("--limit",   type=int, default=None,
                   help="Evaluate only the first N records (default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {"bfloat16": torch.bfloat16,
                 "float16":  torch.float16,
                 "float32":  torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}  [{args.dtype}]")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=torch_dtype,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device != "cuda":
        model = model.to(args.device)
    model.eval()
    print("Model loaded.")

    records = load_records(args.csv, args.splits)
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} records from splits: {args.splits}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = out_path.parent / (out_path.stem + "_metrics.json")

    done: set = set()
    if args.resume and out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(int(row["idx"]))
        print(f"Resuming: {len(done)} records already done.")

    fieldnames = ["idx", "split", "answer_gt", "answer_pred", "correct",
                  "raw_response", "question"]
    write_header = not (args.resume and out_path.exists())
    out_f = open(out_path, "a" if args.resume else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    n_correct = 0
    n_total   = 0
    per_class: Dict[str, Dict] = {}

    for idx, row in enumerate(records):
        if idx in done:
            n_total += 1
            continue

        question = row["anon_question"]
        ref_seq  = row["reference_sequence"]
        var_seq  = row["variant_sequence"]
        gt       = row["answer"].strip().lower()

        user_msg = build_user_message(question, ref_seq, var_seq, args.dna_truncate)
        prompt   = build_prompt(tokenizer, user_msg)

        t0 = time.time()
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  [idx={idx}] Inference error: {e}")
            raw = ""

        pred    = extract_answer(raw)
        correct = is_correct(pred, gt)
        if correct:
            n_correct += 1
        n_total += 1

        if gt not in per_class:
            per_class[gt] = {"tp": 0, "fp": 0, "fn": 0}
        if correct:
            per_class[gt]["tp"] += 1
        else:
            per_class[gt]["fn"] += 1
            if pred:
                per_class.setdefault(pred.lower(), {"tp": 0, "fp": 0, "fn": 0})
                per_class[pred.lower()]["fp"] += 1

        writer.writerow({
            "idx":         idx,
            "split":       row["split"],
            "answer_gt":   gt,
            "answer_pred": pred,
            "correct":     int(correct),
            "raw_response": raw,
            "question":    question,
        })
        out_f.flush()

        elapsed = time.time() - t0
        print(f"  [{idx+1:3d}/{len(records)}] gt={gt!r:30s}  pred={pred!r:30s}  "
              f"{'✓' if correct else '✗'}  ({elapsed:.1f}s)")

    out_f.close()

    accuracy = n_correct / n_total if n_total else 0.0
    precisions, recalls, f1s = [], [], []
    for cls, s in per_class.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)

    macro_prec   = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls)    / len(recalls)    if recalls    else 0.0
    macro_f1     = sum(f1s)        / len(f1s)        if f1s        else 0.0

    metrics = {
        "model": args.model, "splits": args.splits,
        "n_total": n_total, "n_correct": n_correct,
        "accuracy":        round(accuracy,     4),
        "macro_precision": round(macro_prec,   4),
        "macro_recall":    round(macro_recall, 4),
        "macro_f1":        round(macro_f1,     4),
        "dna_truncate_bp": args.dna_truncate,
        "per_class": {
            cls: {
                "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                "precision": round(s["tp"] / (s["tp"]+s["fp"]) if s["tp"]+s["fp"] else 0, 4),
                "recall":    round(s["tp"] / (s["tp"]+s["fn"]) if s["tp"]+s["fn"] else 0, 4),
            }
            for cls, s in sorted(per_class.items())
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Meditron Evaluation ===")
    print(f"  Model      : {args.model}")
    print(f"  Splits     : {args.splits}  ({n_total} records)")
    print(f"  Accuracy   : {accuracy:.1%}  ({n_correct}/{n_total})")
    print(f"  Macro P/R/F1: {macro_prec:.3f} / {macro_recall:.3f} / {macro_f1:.3f}")
    print(f"  Results CSV : {out_path}")
    print(f"  Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
