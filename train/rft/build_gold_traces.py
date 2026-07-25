#!/usr/bin/env python3
"""
Build GOLD RFT traces from the KEGG dataset's own reasoning + answer.

For prompts that NO model answers correctly — e.g. the prion<->creutzfeldt-jakob and
gaucher<->sphingolipidoses synonymy misses, where GRPO and the Stage-1.51 SFT both name a
family/parent term instead of the canonical label — fall back to the dataset's gold chain:

    <think>
    {reasoning}
    </think>
    Answer: {answer}

This is correct by construction, well-formed, latent-free, and uses the EXACT answer text,
so the self-adaptive model gets a clean supervised target for those prompts. `index` is the
position in the unshuffled split order, matching the sampler / RFT filter alignment
(load_eval_records enumerates records the same way).

Typical flow (backfill only what's still uncovered after GRPO + SFT):
  1) filter with --dump_uncovered uncovered.txt   (build_rft_dataset.py)
  2) python train/rft/build_gold_traces.py --split train --dataset_name wanglab/kegg \
        --indices uncovered.txt --out train/rft/samples/rft_gold_traces.jsonl
  3) re-run the filter with rft_gold_traces.jsonl added to --traces

Usage:
  python train/rft/build_gold_traces.py --split train --dataset_name wanglab/kegg \
      [--indices <file>] --out <gold_traces.jsonl>
  python train/rft/build_gold_traces.py --split train --kegg_csv <anon.csv> \
      [--indices <file>] --out <gold_traces.jsonl>
"""
import argparse
import json


def build_rows(raw, splits, want):
    """raw: dict-like split -> iterable of examples (question/reasoning/answer). splits: the
    split order to enumerate. want: set of indices to keep (None = all). Yields gold rows.

    Enumeration mirrors eval_grpo_checkpoint_final.load_eval_records: concatenate the splits
    in order, one index per example, so the emitted `index` aligns with the sampled traces."""
    idx = -1
    for sp in splits:
        if sp not in raw:
            continue
        for ex in raw[sp]:
            idx += 1
            if want is not None and idx not in want:
                continue
            answer = (ex.get("answer", "") or "").strip()
            if not answer:
                continue
            # strip stray tags so the single-<think> / </think> well-formedness holds
            reasoning = (ex.get("reasoning", "") or "").strip() \
                .replace("<think>", "").replace("</think>", "")
            completion = f"<think>\n{reasoning}\n</think>\nAnswer: {answer}"
            yield {
                "index":             idx,
                "pass":              0,
                "split":             sp,
                "question":          ex.get("question", ""),
                "ground_truth":      answer.lower(),
                "predicted_answer":  answer.lower(),
                "is_correct":        True,
                "n_latent_emitted":  0,
                "gen_length_tokens": len(completion.split()),
                "gen_time_sec":      0.0,
                "full_generation":   completion,
                "_gold":             True,
            }


def load_raw(args):
    if args.kegg_csv:
        from genomorph.dataset.kegg import load_kegg_from_anon_csv
        return load_kegg_from_anon_csv(args.kegg_csv)
    from datasets import load_dataset
    return load_dataset(args.dataset_name, "default", cache_dir=args.cache_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out",     required=True, help="Output gold-traces JSONL.")
    p.add_argument("--split",   default="train", help="Split to read (train | val | test | both).")
    p.add_argument("--dataset_name", default="wanglab/kegg")
    p.add_argument("--kegg_csv",     default=None, help="Anon CSV instead of the HF dataset.")
    p.add_argument("--cache_dir",    default="~/.cache/huggingface")
    p.add_argument("--indices", default=None,
                   help="File of indices (one int/line or a JSON list) to backfill. "
                        "Omit to emit a gold trace for EVERY prompt in the split.")
    args = p.parse_args()

    want = None
    if args.indices:
        with open(args.indices, encoding="utf-8") as f:
            txt = f.read().strip()
        want = set(int(x) for x in (json.loads(txt) if txt.startswith("[")
                                    else txt.split()))

    raw = load_raw(args)
    splits = ["val", "test"] if args.split == "both" else [args.split]

    written = 0
    missing = []
    got = set()
    with open(args.out, "w", encoding="utf-8") as f:
        for row in build_rows(raw, splits, want):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            got.add(row["index"])
            written += 1
    if want is not None:
        missing = sorted(want - got)

    print(f"Wrote {written} gold traces -> {args.out}")
    if want is not None:
        print(f"  requested {len(want)} indices; emitted {len(got)}")
        if missing:
            print(f"  WARNING: {len(missing)} requested indices not found / had no answer: "
                  f"{missing[:20]}{' ...' if len(missing) > 20 else ''}")


if __name__ == "__main__":
    main()
