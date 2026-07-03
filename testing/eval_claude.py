"""
Baseline evaluation of the 290 held-out records (test + val splits) using
Claude via the Anthropic API.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python testing/eval_claude.py \
        --csv  genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv \
        --out  testing/logs/claude_results.csv \
        --splits test val

The script uses the same correctness metric as the GRPO training reward:
    correct = ground_truth.lower() in extracted_answer.lower()

Output:
    <out>.csv      — per-sample predictions
    <out>_metrics.json — accuracy / precision / recall / F1 per class
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
    import anthropic
except ImportError:
    sys.exit("anthropic package not found. Install with: pip install anthropic")


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
    ref_short = ref_seq[:dna_truncate] + ("..." if len(ref_seq) > dna_truncate else "")
    var_short = var_seq[:dna_truncate] + ("..." if len(var_seq) > dna_truncate else "")
    return (
        f"Reference sequence (first {dna_truncate} bp): {ref_short}\n"
        f"Variant sequence   (first {dna_truncate} bp): {var_short}\n\n"
        f"{question}"
    )


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    # Claude doesn't use </think> tags; scan full text for Answer: pattern.
    # Also handle the case where GenoMorph-style </think> appears in the response.
    for part in text.split("</think>")[1:]:
        if "Answer:" in part or "answer:" in part:
            answer = re.split(r'[Aa]nswer:\s*', part, maxsplit=1)[-1]
            answer = answer.split('\n')[0].strip()
            if answer:
                return answer
    # Fallback: scan full text for Answer: pattern
    m = re.search(r'[Aa]nswer:\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
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
    p.add_argument("--out",     default="testing/logs/claude_results.csv",
                   help="Output CSV path (metrics JSON written alongside)")
    p.add_argument("--splits",  nargs="+", default=["test", "val"],
                   help="Which splits to evaluate (default: test val)")
    p.add_argument("--model",   default="claude-haiku-4-5",
                   help="Anthropic model ID (default: claude-haiku-4-5)")
    p.add_argument("--dna_truncate", type=int, default=500,
                   help="Max bp per DNA sequence sent to the model (default: 500)")
    p.add_argument("--max_tokens",   type=int, default=512,
                   help="Max completion tokens per request (default: 512)")
    p.add_argument("--rpm_limit",    type=int, default=60,
                   help="Requests per minute limit to avoid rate errors (default: 60)")
    p.add_argument("--resume",  action="store_true",
                   help="Skip rows already present in the output CSV")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY environment variable not set.")
    client = anthropic.Anthropic(api_key=api_key)

    records = load_records(args.csv, args.splits)
    print(f"Loaded {len(records)} records from splits: {args.splits}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = out_path.with_suffix("").parent / (out_path.stem + "_metrics.json")

    # Resume: collect already-done indices
    done: set = set()
    if args.resume and out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(int(row["idx"]))
        print(f"Resuming: {len(done)} records already done, skipping.")

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
    min_delay = 60.0 / args.rpm_limit

    for idx, row in enumerate(records):
        if idx in done:
            n_total += 1
            continue

        question = row["anon_question"]
        ref_seq  = row["reference_sequence"]
        var_seq  = row["variant_sequence"]
        gt       = row["answer"].strip().lower()

        user_msg = build_user_message(question, ref_seq, var_seq, args.dna_truncate)

        t0 = time.time()
        try:
            resp = client.messages.create(
                model=args.model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                max_tokens=args.max_tokens,
            )
            raw = resp.content[0].text if resp.content else ""
        except Exception as e:
            print(f"  [idx={idx}] API error: {e} — retrying in 30s")
            time.sleep(30)
            try:
                resp = client.messages.create(
                    model=args.model,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                    max_tokens=args.max_tokens,
                )
                raw = resp.content[0].text if resp.content else ""
            except Exception as e2:
                print(f"  [idx={idx}] Retry failed: {e2} — writing empty")
                raw = ""

        pred    = extract_answer(raw)
        correct = is_correct(pred, gt)
        if correct:
            n_correct += 1
        n_total += 1

        # Per-class stats
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
            "raw_response": raw[:500],
            "question":    question[:200],
        })
        out_f.flush()

        elapsed = time.time() - t0
        print(f"  [{idx+1:3d}/{len(records)}] gt={gt!r:30s}  pred={pred!r:30s}  {'✓' if correct else '✗'}")

        sleep_for = max(0.0, min_delay - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    out_f.close()

    accuracy = n_correct / n_total if n_total else 0.0

    precisions, recalls, f1s = [], [], []
    for cls, s in per_class.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec   = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls)    / len(recalls)    if recalls    else 0.0
    macro_f1     = sum(f1s)        / len(f1s)        if f1s        else 0.0

    metrics = {
        "model":        args.model,
        "splits":       args.splits,
        "n_total":      n_total,
        "n_correct":    n_correct,
        "accuracy":     round(accuracy,     4),
        "macro_precision": round(macro_prec,   4),
        "macro_recall":    round(macro_recall, 4),
        "macro_f1":        round(macro_f1,     4),
        "dna_truncate_bp": args.dna_truncate,
        "per_class":    {
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

    print("\n=== Claude Evaluation ===")
    print(f"  Model      : {args.model}")
    print(f"  Splits     : {args.splits}  ({n_total} records)")
    print(f"  Accuracy   : {accuracy:.1%}  ({n_correct}/{n_total})")
    print(f"  Macro P/R/F1: {macro_prec:.3f} / {macro_recall:.3f} / {macro_f1:.3f}")
    print(f"  Results CSV: {out_path}")
    print(f"  Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
