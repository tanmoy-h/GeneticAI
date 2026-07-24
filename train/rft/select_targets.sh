#!/bin/bash
## Step 1a.5 of the self-adaptive RFT pipeline: from the GRPO sampled traces, pick the
## prompts the SFT source should cover — GRPO's MISSES (no correct well-formed trace) and
## SLOW prompts (correct, but even the fastest correct trace exceeds the avg gen time) —
## and mark the slow GRPO outputs. CPU-only, fast. Wraps train/rft/select_sft_targets.py.
##
## Output:
##   OUT_INDICES  the SFT target indices (MISS + SLOW) -> feed the SFT run as
##                ONLY_INDICES=<that file> (see train/rft/sample_traces.sh).
##   OUT_MARKED   a copy of the GRPO traces with each output flagged slow_output + avg time.
##
## Usage:
##   GRPO_TRACES=train/rft/samples/rft_sample_traces.jsonl bash train/rft/select_targets.sh
##   GRPO_TRACES=<..> OUT_INDICES=<..> OUT_MARKED=<..> MAX_WORDS=350 \
##     bash train/rft/select_targets.sh

CONDA_ENV=${CONDA_ENV:-dna_env}
GRPO_TRACES=${GRPO_TRACES:-}
OUT_INDICES=${OUT_INDICES:-train/rft/samples/sft_targets.txt}
OUT_MARKED=${OUT_MARKED:-train/rft/samples/rft_sample_traces_marked.jsonl}
## MAX_WORDS: a correct trace must also be <= this many words to count (a prompt whose only
## correct traces are over-long is treated as a MISS so the SFT can supply a tight one).
## Empty = no word cap. Match the filter's MAX_WORDS if you set one there.
MAX_WORDS=${MAX_WORDS:-}

if [ -z "$GRPO_TRACES" ]; then
    echo "ERROR: set GRPO_TRACES=<...>_traces.jsonl (the GRPO output of train/rft/sample_traces.sh)"
    exit 1
fi
if [ ! -f "$GRPO_TRACES" ]; then
    echo "ERROR: GRPO_TRACES not found: $GRPO_TRACES"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
conda activate $CONDA_ENV 2>/dev/null || true
cd "$(dirname "$0")/../.."

ARGS=(--grpo_traces "$GRPO_TRACES" --out_indices "$OUT_INDICES")
[ -n "$OUT_MARKED" ] && ARGS+=(--out_marked "$OUT_MARKED")
[ -n "$MAX_WORDS" ] && ARGS+=(--max_words "$MAX_WORDS")

echo "Selecting SFT targets from $GRPO_TRACES (max_words=${MAX_WORDS:-none})"
python train/rft/select_sft_targets.py "${ARGS[@]}"

echo ""
echo "=== Target selection done -> $OUT_INDICES ==="
echo "=== Next: SFT_BEST=1 ONLY_INDICES=$OUT_INDICES bash train/rft/sample_traces.sh ==="
