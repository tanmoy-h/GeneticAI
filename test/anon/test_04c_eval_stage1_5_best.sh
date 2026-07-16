#!/bin/bash
#SBATCH --job-name=w9_eval_stage1_5_best_anon
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=test/anon/logs/test_04c_eval_stage1_5_best_%j.out
#SBATCH --error=test/anon/logs/test_04c_eval_stage1_5_best_%j.err

## Evaluate ONLY the best Stage 1.5.0 (LatentSp curriculum) checkpoint — ANON.
## Auto-selects with the same 3-tier chain as anon train_04_stage1_51.sh, then
## evaluates just that one on the full val set. See test/test_04c_eval_stage1_5_best.sh.
##
## Usage:
##   bash test/anon/test_04c_eval_stage1_5_best.sh
##   BEST_CKPT=/scratch/.../s03_pass02/model.pt bash test/anon/test_04c_eval_stage1_5_best.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
_S15_CKPT_DIR=${CKPT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_03b_stage1_50_cached_anon}
_PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
GPUS=${GPUS:-0}
N_SAMPLES=${N_SAMPLES:-290}   # caps at split size; use a big number for "all"
EVAL_SPLIT=${EVAL_SPLIT:-both}  # anon re-split: val=144, test=146 -> both=290
SEED=${SEED:-42}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-800}
_TS=$(date +%Y%m%d_%H%M%S)
RESULTS_JSON=${RESULTS_JSON:-test/anon/logs/test_04c_eval_stage1_5_best_results_${_TS}.json}
RAW_CSV=${RAW_CSV:-test/anon/logs/test_04c_eval_stage1_5_best_raw_${_TS}.csv}
## ─────────────────────────────────────────────────────────────────────────────

## ── Resolve the best Stage 1.5 checkpoint (mirror anon train_04_stage1_51.sh) ──
BEST_CKPT=${BEST_CKPT:-${STAGE15_CKPT:-}}

## 1) deterministic pointer written by eval_stage1_5_checkpoints.py
if [ -z "${BEST_CKPT:-}" ] && [ -f "$_S15_CKPT_DIR/stage15_best_ckpt.txt" ]; then
    BEST_CKPT=$(cat "$_S15_CKPT_DIR/stage15_best_ckpt.txt")
    echo "BEST_CKPT (from stage15_best_ckpt.txt): $BEST_CKPT"
fi

## 2) parse best from the latest eval results JSON
if [ -z "${BEST_CKPT:-}" ]; then
    _RESULTS_JSON=$(find "$_PROJECT_DIR/test/anon/logs" \
        -not -path "*/.*/*" \
        -name "test_04_eval_stage1_5_results_*.json" 2>/dev/null | sort -V | tail -1)
    if [ -n "$_RESULTS_JSON" ]; then
        BEST_CKPT=$(python3 -c "
import json, sys
try:
    print(json.load(open('$_RESULTS_JSON'))['best']['full_path'])
except: sys.exit(1)
" 2>/dev/null)
        [ -n "$BEST_CKPT" ] && echo "BEST_CKPT (from eval results JSON): $BEST_CKPT" \
            || echo "WARNING: could not parse best checkpoint from $_RESULTS_JSON"
    fi
fi

## 3) fallback: last model.pt by path sort
if [ -z "${BEST_CKPT:-}" ]; then
    BEST_CKPT=$(find "$_S15_CKPT_DIR" -name "model.pt" 2>/dev/null | sort -V | tail -1)
    [ -n "$BEST_CKPT" ] && echo "BEST_CKPT (fallback, last by path): $BEST_CKPT"
fi

if [ -z "${BEST_CKPT:-}" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "ERROR: could not resolve the best Stage 1.5 checkpoint."
    echo "       Run test/anon/test_04_eval_stage1_5.sh first (writes stage15_best_ckpt.txt),"
    echo "       or set BEST_CKPT=<path/to/sNN_passMM/model.pt> explicitly."
    exit 1
fi

## Base Stage 1 SFT checkpoint (eval loads this, then the 1.5 delta on top)
STAGE1_CKPT=${STAGE1_CKPT:-}
if [ -z "${STAGE1_CKPT:-}" ] && [ -f "$_S15_CKPT_DIR/stage1_ckpt.txt" ]; then
    STAGE1_CKPT=$(cat "$_S15_CKPT_DIR/stage1_ckpt.txt")
fi
if [ -z "${STAGE1_CKPT:-}" ]; then
    echo "ERROR: STAGE1_CKPT not set and $_S15_CKPT_DIR/stage1_ckpt.txt not found."
    echo "       Set STAGE1_CKPT=<base .ckpt> explicitly."
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p test/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"

## Isolate the best checkpoint so eval_stage1_5_checkpoints.py
## (which globs CKPT_DIR/s??_pass??/model.pt) scores ONLY this one.
LABEL=$(basename "$(dirname "$BEST_CKPT")")   # e.g. s03_pass02
ISO_DIR=$(mktemp -d)
mkdir -p "$ISO_DIR/$LABEL"
ln -s "$BEST_CKPT" "$ISO_DIR/$LABEL/model.pt"
trap 'rm -rf "$ISO_DIR"' EXIT

LOG=test/anon/logs/test_04c_eval_stage1_5_best_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "Best 1.5 ckpt: $BEST_CKPT  (label=$LABEL)"
echo "Base ckpt:     $STAGE1_CKPT"
echo "Isolated dir:  $ISO_DIR"
echo "GPUs:          $GPUS"
echo "N samples:     $N_SAMPLES   Split: $EVAL_SPLIT   Seed: $SEED"
echo "Results JSON:  $RESULTS_JSON"
nvidia-smi

## Build dataset arg
if [ -n "$KEGG_CSV" ]; then
    KEGG_ARG="--kegg_csv $KEGG_CSV"
else
    KEGG_ARG="--kegg_dataset $KEGG_DATASET"
fi

## Build DNA cache arg
if [ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ]; then
    DNA_CACHE_ARG="--dna_cache $DNA_CACHE"
else
    echo "WARNING: DNA cache not found: $DNA_CACHE — Evo2 will run live (slower, more VRAM)."
    DNA_CACHE_ARG=""
fi

stdbuf -oL -eL python eval_stage1_5_checkpoints.py \
    --stage1_ckpt            "$STAGE1_CKPT" \
    --ckpt_dir               "$ISO_DIR" \
    $KEGG_ARG \
    $DNA_CACHE_ARG \
    --gpus                   "$GPUS" \
    --n_samples              $N_SAMPLES \
    --eval_split             $EVAL_SPLIT \
    --max_new_tokens         $MAX_NEW_TOKENS \
    --seed                   $SEED \
    --cache_dir              "$CACHE_DIR" \
    --results_json           "$RESULTS_JSON" \
    --raw_csv                "$RAW_CSV"

echo ""
echo "=== Best Stage 1.5 checkpoint evaluated: $LABEL ==="
echo "  Accuracy / macro-F1 in the ranking above (single row) and in $RESULTS_JSON"
