#!/bin/bash
## Step 1b of the self-adaptive RFT pipeline: filter sampled traces to
## correct AND well-formed AND (preferably) latent-using AND short, one per prompt.
## CPU-only, fast. Wraps train/rft/build_rft_dataset.py.
##
## Usage:
##   TRACES=train/rft/samples/rft_sample_traces.jsonl bash train/rft/filter_traces.sh
##   TRACES=<..> OUT=<..> MAX_WORDS=350 REQUIRE_LATENT=1 bash train/rft/filter_traces.sh
##   # merge GRPO + Stage-1.51 SFT traces (space-separated) so rare diseases with no
##   # correct GRPO latent trace are covered by an SFT trace:
##   TRACES="train/rft/samples/rft_sample_traces.jsonl train/rft/samples/rft_sample_sft_traces.jsonl" \
##     bash train/rft/filter_traces.sh
##   # PREFER_TIME=1 lets a faster SFT text trace beat a slow GRPO latent trace on GRPO's
##   # slow prompts (pick fastest correct trace, not shortest-by-tokens):
##   PREFER_TIME=1 TRACES="<grpo>_traces.jsonl <sft>_traces.jsonl" bash train/rft/filter_traces.sh

CONDA_ENV=${CONDA_ENV:-dna_env}
TRACES=${TRACES:-}
OUT=${OUT:-train/rft/rft_selfadaptive.jsonl}
MAX_WORDS=${MAX_WORDS:-350}
REQUIRE_LATENT=${REQUIRE_LATENT:-0}
## Rare-disease protection (counters GRPO's flat-per-sample correctness that eroded
## macro-F1): a disease with <= RARE_MAX_PROMPTS prompts bypasses the word cap and
## require_latent (never lose its only correct trace) and is oversampled RARE_OVERSAMPLE
## times so the self-adaptive SFT weights it more. RARE_MAX_PROMPTS=0 disables.
RARE_MAX_PROMPTS=${RARE_MAX_PROMPTS:-2}
RARE_OVERSAMPLE=${RARE_OVERSAMPLE:-3}
## PREFER_TIME=1 selects the FASTEST correct trace per prompt (by gen_time_sec) instead of
## the shortest-by-tokens latent trace, so a faster SFT text trace can beat a slow GRPO
## latent trace on GRPO's slow prompts (see select_sft_targets.py). Default 0 keeps the
## latent-first, shortest-tokens selection.
PREFER_TIME=${PREFER_TIME:-0}

if [ -z "$TRACES" ]; then
    echo "ERROR: set TRACES=<...>_traces.jsonl (output of train/rft/sample_traces.sh)"
    echo "       (space-separate several files to merge sources, e.g. GRPO + SFT traces)"
    exit 1
fi
## TRACES may be several space-separated files; verify each exists.
for _t in $TRACES; do
    if [ ! -f "$_t" ]; then
        echo "ERROR: TRACES file not found: $_t"
        exit 1
    fi
done

module load MLDL/miniconda3 2>/dev/null || true
conda activate $CONDA_ENV 2>/dev/null || true
cd "$(dirname "$0")/../.."

## $TRACES unquoted so multiple space-separated files expand to separate argv entries
## (the builder's --traces takes nargs="+"). File paths must not contain spaces.
ARGS=(--traces $TRACES --out "$OUT")
[ -n "$MAX_WORDS" ] && ARGS+=(--max_words "$MAX_WORDS")
[ "$REQUIRE_LATENT" = "1" ] && ARGS+=(--require_latent)
[ -n "$RARE_MAX_PROMPTS" ] && ARGS+=(--rare_max_prompts "$RARE_MAX_PROMPTS")
[ -n "$RARE_OVERSAMPLE" ] && ARGS+=(--rare_oversample "$RARE_OVERSAMPLE")
[ "$PREFER_TIME" = "1" ] && ARGS+=(--prefer_time)

echo "Filtering $TRACES -> $OUT  (max_words=$MAX_WORDS require_latent=$REQUIRE_LATENT rare<=$RARE_MAX_PROMPTS x$RARE_OVERSAMPLE prefer_time=$PREFER_TIME)"
python train/rft/build_rft_dataset.py "${ARGS[@]}"

echo ""
echo "=== Filter done -> $OUT ==="
echo "=== Next: RFT_TRACES=$OUT bash train/train_07_selfadaptive.sh ==="
