#!/usr/bin/env python3
"""
Pick the prompts the SFT source should cover, from the GRPO sampled traces.

Instead of the frequency-based "rare" proxy, target GRPO's ACTUAL weaknesses:

  MISS  — GRPO produced no correct, well-formed trace for the prompt across all passes.
          The SFT (strong latent-free accuracy) is our best shot at a correct trace.
  SLOW  — GRPO is correct on the prompt, but even its FASTEST correct trace takes longer
          than the average GRPO generation time. We sample the SFT for these too, to
          check whether a latent-free SFT trace is faster (GRPO's latent look-ahead can
          cost wall-time even when it emits fewer tokens). The merge step then keeps
          whichever correct trace is better.

Outputs (given --grpo_traces <...>_traces.jsonl):
  --out_indices  : the SFT target indices (MISS + SLOW), one per line — feed to the
                   sampler as ONLY_INDICES=<this file> (see train/rft/sample_traces.sh).
  --out_marked   : (optional) a copy of the GRPO traces with each output MARKED
                   'slow_output': gen_time_sec > avg, plus 'avg_gen_time_sec', for
                   inspection / the downstream speed comparison.

Usage:
  python train/rft/select_sft_targets.py \
      --grpo_traces train/rft/samples/rft_sample_traces.jsonl \
      --out_indices train/rft/samples/sft_targets.txt \
      [--out_marked train/rft/samples/rft_sample_traces_marked.jsonl] \
      [--max_words 350]
"""
import argparse
import json
import re
from collections import defaultdict

_ANSWER_AFTER_THINK = re.compile(r"</think>[\s\S]*?[Aa]nswer:\s*\S")


def is_well_formed(text: str) -> bool:
    if text.count("<think>") != 1:
        return False
    if "</think>" not in text:
        return False
    return bool(_ANSWER_AFTER_THINK.search(text))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grpo_traces", required=True, help="GRPO *_traces.jsonl from the sampler.")
    p.add_argument("--out_indices", required=True, help="Output target-index file (one int/line).")
    p.add_argument("--out_marked",  default=None,
                   help="Optional: write GRPO traces annotated with slow_output + avg time.")
    p.add_argument("--max_words", type=int, default=None,
                   help="If set, a correct trace must also be <= this many words to count "
                        "(matches the filter's soft cap; a prompt whose only correct traces "
                        "are over-long is treated as a MISS so the SFT can supply a tight one).")
    args = p.parse_args()

    by_index = defaultdict(list)
    all_times = []
    n_lines = 0
    with open(args.grpo_traces, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_index[r["index"]].append(r)
            t = r.get("gen_time_sec")
            if isinstance(t, (int, float)) and t > 0:
                all_times.append(float(t))
            n_lines += 1

    if not all_times:
        raise SystemExit("No gen_time_sec found in traces — cannot compute average time.")
    avg_time = sum(all_times) / len(all_times)

    def correct_wf(recs):
        out = []
        for r in recs:
            if not r.get("is_correct"):
                continue
            if not is_well_formed(r.get("full_generation", "")):
                continue
            if args.max_words is not None and \
               len(r.get("full_generation", "").split()) > args.max_words:
                continue
            out.append(r)
        return out

    miss, slow, ok = [], [], []
    for idx, recs in by_index.items():
        cwf = correct_wf(recs)
        if not cwf:
            miss.append(idx)
            continue
        best_time = min(r.get("gen_time_sec", 1e30) for r in cwf)
        if best_time > avg_time:                 # even the fastest correct trace is slow
            slow.append((idx, best_time))
        else:
            ok.append(idx)

    targets = sorted(set(miss) | set(i for i, _ in slow))   # MISS + SLOW
    with open(args.out_indices, "w", encoding="utf-8") as f:
        for i in targets:
            f.write(f"{i}\n")

    # ── Optional: mark the GRPO outputs slower than average ───────────────────
    if args.out_marked:
        with open(args.out_marked, "w", encoding="utf-8") as f:
            for recs in by_index.values():
                for r in recs:
                    r["avg_gen_time_sec"] = round(avg_time, 3)
                    t = r.get("gen_time_sec")
                    r["slow_output"] = bool(isinstance(t, (int, float)) and t > avg_time)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Report ────────────────────────────────────────────────────────────────
    n_prompts = len(by_index)
    print(f"Read {n_lines} GRPO traces over {n_prompts} prompts")
    print(f"Average GRPO gen_time: {avg_time:.2f}s  (over {len(all_times)} outputs)")
    print(f"  MISS  (no correct well-formed trace) : {len(miss)}")
    print(f"  SLOW  (correct, but fastest > avg)   : {len(slow)}")
    print(f"  OK    (correct & fast, no SFT needed): {len(ok)}")
    print(f"  -> SFT targets (MISS + SLOW)         : {len(targets)}  -> {args.out_indices}")
    if args.out_marked:
        n_slow_out = sum(1 for recs in by_index.values() for r in recs
                         if r.get("gen_time_sec", 0) > avg_time)
        print(f"  marked {n_slow_out} slow outputs         -> {args.out_marked}")
    if slow:
        slow.sort(key=lambda x: -x[1])
        print("  slowest correct prompts (idx: best_correct_time):")
        for idx, t in slow[:15]:
            print(f"    {idx:>5d}: {t:.2f}s")


if __name__ == "__main__":
    main()
