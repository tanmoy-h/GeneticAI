#!/usr/bin/env python3
"""
Build the self-adaptive RFT training set from sampled GRPO traces.

Input : the *_traces.jsonl written by eval_grpo_checkpoint_final.py run in RFT
        sampling mode (--split train --temperature 0.7 --sample_passes N --n_samples -1).
        Each line: {index, pass, question, ground_truth, is_correct,
                    n_latent_emitted, gen_length_tokens, full_generation, ...}

Selection (per prompt index): keep only correct AND well-formed completions, then
prefer a LATENT-using one, and among those the SHORTEST (best-of-N by length). This
bakes in "correct + latent + concise" per example. If no correct latent trace exists
for a prompt, fall back to the shortest correct well-formed text trace (so the prompt's
supervision is not lost) unless --require_latent is set.

Output: a JSONL of selected training rows:
        {index, question, ground_truth, completion, n_latent_emitted, gen_length_tokens}
        The training entry rejoins each row with the dataset (by index) to recover the
        DNA + full prompt. Pass the SAME --split train --n_samples -1 dataset to both.

Usage:
  python train/rft/build_rft_dataset.py \
      --traces <...>_traces.jsonl \
      --out    train/rft/rft_selfadaptive.jsonl \
      [--max_words 350] [--require_latent]
"""
import argparse
import json
import re
from collections import defaultdict


_ANSWER_AFTER_THINK = re.compile(r"</think>[\s\S]*?[Aa]nswer:\s*\S")


def is_well_formed(text: str) -> bool:
    """One <think>, a closing </think>, and a non-empty Answer: after it."""
    if text.count("<think>") != 1:
        return False
    if "</think>" not in text:
        return False
    return bool(_ANSWER_AFTER_THINK.search(text))


def word_count(text: str) -> int:
    return len(text.split())


def select_per_index(records, max_words=None, require_latent=False):
    """records: list of trace dicts sharing one index. Return the chosen dict or None."""
    # correct + well-formed candidates
    cands = [
        r for r in records
        if r.get("is_correct")
        and is_well_formed(r.get("full_generation", ""))
    ]
    if not cands:
        return None, "no_correct_wellformed"

    if max_words is not None:
        capped = [r for r in cands if word_count(r["full_generation"]) <= max_words]
        if capped:                       # only apply the cap if it leaves something
            cands = capped

    latent = [r for r in cands if r.get("n_latent_emitted", 0) >= 1]
    if latent:
        best = min(latent, key=lambda r: r.get("gen_length_tokens", 1 << 30))
        return best, "latent"
    if require_latent:
        return None, "no_correct_latent"
    # text fallback: shortest correct well-formed trace
    best = min(cands, key=lambda r: r.get("gen_length_tokens", 1 << 30))
    return best, "text_fallback"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces",  required=True, help="*_traces.jsonl from the sampler.")
    p.add_argument("--out",     required=True, help="Output selected-traces JSONL.")
    p.add_argument("--max_words", type=int, default=None,
                   help="Prefer completions <= this many words (soft; relaxed if it "
                        "would drop a prompt entirely).")
    p.add_argument("--require_latent", action="store_true",
                   help="Drop prompts with no correct latent-using trace (no text "
                        "fallback). Yields a purely latent training set.")
    args = p.parse_args()

    by_index = defaultdict(list)
    n_lines = 0
    with open(args.traces, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_index[r["index"]].append(r)
            n_lines += 1

    reasons = defaultdict(int)
    selected = []
    for idx, recs in by_index.items():
        best, why = select_per_index(
            recs, max_words=args.max_words, require_latent=args.require_latent
        )
        reasons[why] += 1
        if best is None:
            continue
        selected.append({
            "index":             idx,
            "question":          best.get("question", ""),
            "ground_truth":      best.get("ground_truth", ""),
            "completion":        best["full_generation"],
            "n_latent_emitted":  best.get("n_latent_emitted", 0),
            "gen_length_tokens": best.get("gen_length_tokens", 0),
            "select_reason":     why,
        })

    selected.sort(key=lambda r: r["index"])
    with open(args.out, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_prompts = len(by_index)
    n_latent  = reasons.get("latent", 0)
    n_text    = reasons.get("text_fallback", 0)
    print(f"Read {n_lines} traces over {n_prompts} prompts")
    print(f"Selected {len(selected)} rows -> {args.out}")
    print(f"  latent traces   : {n_latent}")
    print(f"  text fallback   : {n_text}")
    print(f"  dropped         : {reasons.get('no_correct_wellformed', 0) + reasons.get('no_correct_latent', 0)}")
    if selected:
        avg_len = sum(r['gen_length_tokens'] for r in selected) / len(selected)
        avg_lat = sum(r['n_latent_emitted']  for r in selected) / len(selected)
        print(f"  mean gen_length : {avg_len:.1f} tokens")
        print(f"  mean latents    : {avg_lat:.2f} blocks/trace")


if __name__ == "__main__":
    main()
