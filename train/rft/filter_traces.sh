#!/bin/bash
## Step 1b of the self-adaptive RFT pipeline: filter sampled traces to
## correct AND well-formed AND (preferably) latent-using AND short, one per prompt.
## CPU-only, fast. Wraps train/rft/build_rft_dataset.py.
##
## Usage:
##   TRACES=train/rft/samples/rft_sample_traces.jsonl bash train/rft/filter_traces.sh
##   TRACES=<..> OUT=<..> MAX_WORDS=350 REQUIRE_LATENT=1 bash train/rft/filter_traces.sh

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

if [ -z "$TRACES" ]; then
    echo "ERROR: set TRACES=<...>_traces.jsonl (output of train/rft/sample_traces.sh)"
    exit 1
fi
if [ ! -f "$TRACES" ]; then
    echo "ERROR: TRACES not found: $TRACES"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
conda activate $CONDA_ENV 2>/dev/null || true
cd "$(dirname "$0")/../.."

ARGS=(--traces "$TRACES" --out "$OUT")
[ -n "$MAX_WORDS" ] && ARGS+=(--max_words "$MAX_WORDS")
[ "$REQUIRE_LATENT" = "1" ] && ARGS+=(--require_latent)
[ -n "$RARE_MAX_PROMPTS" ] && ARGS+=(--rare_max_prompts "$RARE_MAX_PROMPTS")
[ -n "$RARE_OVERSAMPLE" ] && ARGS+=(--rare_oversample "$RARE_OVERSAMPLE")

echo "Filtering $TRACES -> $OUT  (max_words=$MAX_WORDS require_latent=$REQUIRE_LATENT rare<=$RARE_MAX_PROMPTS x$RARE_OVERSAMPLE)"
python train/rft/build_rft_dataset.py "${ARGS[@]}"

echo ""
echo "=== Filter done -> $OUT ==="
echo "=== Next: RFT_TRACES=$OUT bash train/train_07_selfadaptive.sh ==="
