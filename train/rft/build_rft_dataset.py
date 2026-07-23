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


def select_per_index(records, max_words=None, require_latent=False, is_rare=False):
    """records: list of trace dicts sharing one index. Return the chosen dict or None.

    is_rare relaxes the filters so a rare class never loses its only correct trace:
    the word cap is skipped, and a text fallback is allowed even under require_latent.
    """
    # correct + well-formed candidates
    cands = [
        r for r in records
        if r.get("is_correct")
        and is_well_formed(r.get("full_generation", ""))
    ]
    if not cands:
        return None, "no_correct_wellformed"

    # Rare classes ignore the word cap entirely (keep the correct trace even if long);
    # common classes apply it softly (only if it leaves something).
    if max_words is not None and not is_rare:
        capped = [r for r in cands if word_count(r["full_generation"]) <= max_words]
        if capped:                       # only apply the cap if it leaves something
            cands = capped

    latent = [r for r in cands if r.get("n_latent_emitted", 0) >= 1]
    if latent:
        best = min(latent, key=lambda r: r.get("gen_length_tokens", 1 << 30))
        return best, "latent"
    if require_latent and not is_rare:   # rare classes always keep a text fallback
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
    p.add_argument("--rare_max_prompts", type=int, default=0,
                   help="A disease with <= this many prompts is treated as RARE: its "
                        "correct traces bypass the word cap and require_latent, and its "
                        "kept rows are oversampled (see --rare_oversample). 0 disables.")
    p.add_argument("--rare_oversample", type=int, default=1,
                   help="Duplicate each RARE class's kept rows this many times in the "
                        "output (1 = no oversampling). Counters flat-per-sample dilution "
                        "so the self-adaptive SFT weights rare diseases more.")
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

    # Per-class prompt counts (needed BEFORE selection so rare classes get relaxed
    # filters). Diseases with <= rare_max_prompts prompts are rare.
    prompts_per_disease = defaultdict(int)
    for recs in by_index.values():
        disease = (recs[0].get("ground_truth", "") if recs else "").strip().lower()
        prompts_per_disease[disease] += 1

    def _is_rare(disease):
        return args.rare_max_prompts > 0 and prompts_per_disease[disease] <= args.rare_max_prompts

    reasons  = defaultdict(int)
    selected = []
    n_oversampled = 0
    # Per-disease coverage: prompts seen, and how each resolved. `disease` comes from
    # any trace of the prompt (all passes share the same ground_truth).
    class_stats = defaultdict(lambda: {"prompts": 0, "latent": 0, "text": 0, "dropped": 0})
    for idx, recs in by_index.items():
        disease = (recs[0].get("ground_truth", "") if recs else "").strip().lower()
        rare = _is_rare(disease)
        best, why = select_per_index(
            recs, max_words=args.max_words, require_latent=args.require_latent,
            is_rare=rare,
        )
        reasons[why] += 1
        cs = class_stats[disease]
        cs["prompts"] += 1
        if best is None:
            cs["dropped"] += 1
            continue
        cs["latent" if why == "latent" else "text"] += 1
        row = {
            "index":             idx,
            "question":          best.get("question", ""),
            "ground_truth":      best.get("ground_truth", ""),
            "completion":        best["full_generation"],
            "n_latent_emitted":  best.get("n_latent_emitted", 0),
            "gen_length_tokens": best.get("gen_length_tokens", 0),
            "select_reason":     why,
            "rare":              rare,
        }
        # Oversample rare classes: emit the row `rare_oversample` times so the SFT
        # gradient weights the rare disease more (counters GRPO's flat-per-sample bias).
        copies = args.rare_oversample if (rare and args.rare_oversample > 1) else 1
        for _ in range(copies):
            selected.append(row)
        n_oversampled += copies - 1

    selected.sort(key=lambda r: r["index"])
    with open(args.out, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Per-class coverage report ─────────────────────────────────────────────
    # Full table to a CSV sidecar; problem classes (any drop, or fully text-fallback)
    # to the console, rarest first — this is where rare-disease attention is needed.
    report_path = args.out + ".classreport.csv"
    rows = []
    for dis, cs in class_stats.items():
        kept = cs["latent"] + cs["text"]
        rows.append((dis, cs["prompts"], cs["latent"], cs["text"], cs["dropped"],
                     kept / cs["prompts"] if cs["prompts"] else 0.0))
    # sort: worst coverage first, then rarest (fewest prompts) first
    rows.sort(key=lambda r: (r[5], r[1]))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("disease,prompts,kept_latent,kept_text_fallback,dropped,coverage\n")
        for dis, p, lat, txt, drp, cov in rows:
            f.write(f"\"{dis}\",{p},{lat},{txt},{drp},{cov:.3f}\n")

    n_classes       = len(class_stats)
    zero_cov        = [r for r in rows if r[5] == 0.0]                 # nothing kept
    no_latent_only  = [r for r in rows if r[2] == 0 and (r[3] + r[4]) > 0]  # only text/dropped
    partial         = [r for r in rows if r[4] > 0 and r[5] > 0.0]    # some dropped

    n_prompts = len(by_index)
    n_latent  = reasons.get("latent", 0)
    n_text    = reasons.get("text_fallback", 0)
    n_rare_classes = sum(1 for d in prompts_per_disease if _is_rare(d))
    print(f"Read {n_lines} traces over {n_prompts} prompts")
    print(f"Selected {len(selected)} rows -> {args.out}")
    print(f"  latent traces   : {n_latent}")
    print(f"  text fallback   : {n_text}")
    if args.rare_max_prompts > 0:
        print(f"  rare classes    : {n_rare_classes}  (<= {args.rare_max_prompts} prompts; "
              f"filters relaxed, oversample x{args.rare_oversample})")
        print(f"  rows added by oversampling : {n_oversampled}")
    print(f"  dropped         : {reasons.get('no_correct_wellformed', 0) + reasons.get('no_correct_latent', 0)}")
    if selected:
        avg_len = sum(r['gen_length_tokens'] for r in selected) / len(selected)
        avg_lat = sum(r['n_latent_emitted']  for r in selected) / len(selected)
        print(f"  mean gen_length : {avg_len:.1f} tokens")
        print(f"  mean latents    : {avg_lat:.2f} blocks/trace")

    print(f"\n== Per-class coverage ({n_classes} diseases) -> {report_path} ==")
    print(f"  fully-covered classes : {n_classes - len(zero_cov) - len(partial)}")
    print(f"  ZERO-coverage classes : {len(zero_cov)}   (RFT cannot help these - model never got them right+wellformed)")
    print(f"  latent-less classes   : {len(no_latent_only)}   (kept only text traces - no correct latent demo)")
    if zero_cov:
        print("  zero-coverage (rarest first):")
        for dis, p, lat, txt, drp, cov in zero_cov[:30]:
            print(f"    {dis:<45s} prompts={p} dropped={drp}")
    if no_latent_only:
        print("  latent-less (rarest first):")
        for dis, p, lat, txt, drp, cov in no_latent_only[:30]:
            print(f"    {dis:<45s} prompts={p} text={txt} dropped={drp}")


if __name__ == "__main__":
    main()
