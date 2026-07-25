#!/bin/bash
## Step 1c (optional) of the self-adaptive RFT pipeline: build GOLD traces from the KEGG
## dataset's own reasoning + answer for prompts NO model answers correctly (the synonymy
## misses where GRPO and the SFT both name a family/parent term). CPU-only. Wraps
## train/rft/build_gold_traces.py.
##
## Get INDICES from the filter's --dump_uncovered output (the prompts still uncovered after
## merging the GRPO + SFT traces), then re-run the filter with OUT added to TRACES.
##
## Usage:
##   INDICES=train/rft/samples/uncovered.txt bash train/rft/gold_traces.sh
##   INDICES=<..> OUT=<..> SPLIT=train KEGG_DATASET=wanglab/kegg bash train/rft/gold_traces.sh
##   INDICES=<..> KEGG_CSV=<anon.csv> bash train/rft/gold_traces.sh      # anon

CONDA_ENV=${CONDA_ENV:-dna_env}
INDICES=${INDICES:-}
OUT=${OUT:-train/rft/samples/rft_gold_traces.jsonl}
SPLIT=${SPLIT:-train}
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}

module load MLDL/miniconda3 2>/dev/null || true
conda activate $CONDA_ENV 2>/dev/null || true
cd "$(dirname "$0")/../.."

ARGS=(--out "$OUT" --split "$SPLIT" --cache_dir "$CACHE_DIR")
if [ -n "$KEGG_CSV" ]; then
    ARGS+=(--kegg_csv "$KEGG_CSV")
else
    ARGS+=(--dataset_name "$KEGG_DATASET")
fi
[ -n "$INDICES" ] && ARGS+=(--indices "$INDICES")

echo "Building gold traces (split=$SPLIT indices=${INDICES:-ALL}) -> $OUT"
python train/rft/build_gold_traces.py "${ARGS[@]}"

echo ""
echo "=== Gold traces done -> $OUT ==="
echo "=== Next: add $OUT to TRACES and re-run train/rft/filter_traces.sh ==="
