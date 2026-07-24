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

    # ── Per-disease correctness frequency ─────────────────────────────────────
    # How often GRPO gets each disease right — both at the prompt level (fraction of a
    # disease's prompts with >=1 correct well-formed trace) and the pass level (fraction of
    # all its sampled completions that were correct). This is the accuracy-based companion
    # to raw label frequency: a LOW correct-frequency disease needs SFT help even if it is
    # common (e.g. colorectal cancer), which the rare-frequency proxy would miss.
    miss_idx = set(miss)
    slow_idx = set(i for i, _ in slow)
    dis = defaultdict(lambda: {"prompts": set(), "correct_prompts": set(),
                               "passes": 0, "correct_passes": 0,
                               "miss": 0, "slow": 0, "times": []})
    for idx, recs in by_index.items():
        name = (recs[0].get("ground_truth", "") or "").strip().lower()
        st = dis[name]
        st["prompts"].add(idx)
        cwf = correct_wf(recs)
        if cwf:
            st["correct_prompts"].add(idx)
        for r in recs:
            st["passes"] += 1
            if r.get("is_correct") and is_well_formed(r.get("full_generation", "")):
                st["correct_passes"] += 1
            t = r.get("gen_time_sec")
            if isinstance(t, (int, float)) and t > 0:
                st["times"].append(float(t))
        if idx in miss_idx:
            st["miss"] += 1
        if idx in slow_idx:
            st["slow"] += 1
    dis_rows = []
    for name, st in dis.items():
        npr = len(st["prompts"]); ncp = len(st["correct_prompts"])
        prompt_frac = ncp / npr if npr else 0.0
        pass_frac   = st["correct_passes"] / st["passes"] if st["passes"] else 0.0
        avg_t       = sum(st["times"]) / len(st["times"]) if st["times"] else 0.0
        dis_rows.append((name, npr, ncp, prompt_frac, pass_frac, st["miss"], st["slow"], avg_t))
    dis_rows.sort(key=lambda r: (r[3], r[4], r[1]))     # weakest correctness first
    dis_report = args.out_indices + ".diseasereport.csv"
    with open(dis_report, "w", encoding="utf-8") as f:
        f.write("disease,prompts,correct_prompts,prompt_correct_frac,"
                "pass_correct_frac,miss,slow,avg_time_sec\n")
        for name, npr, ncp, pf, paf, ms, sl, at in dis_rows:
            f.write(f"\"{name}\",{npr},{ncp},{pf:.3f},{paf:.3f},{ms},{sl},{at:.2f}\n")

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

    weak = [r for r in dis_rows if r[3] < 1.0]
    print(f"\n  Per-disease correctness -> {dis_report}")
    print(f"  diseases with < 100% prompt correctness: {len(weak)}/{len(dis_rows)}")
    if weak:
        print("  weakest (disease: correct_prompts/prompts, pass_frac, miss/slow):")
        for name, npr, ncp, pf, paf, ms, sl, at in weak[:20]:
            print(f"    {name:<42s} {ncp}/{npr} ({pf:.2f})  pass={paf:.2f}  miss={ms} slow={sl}")


if __name__ == "__main__":
    main()
