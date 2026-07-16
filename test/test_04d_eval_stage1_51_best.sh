#!/bin/bash
#SBATCH --job-name=w9_eval_stage1_51_best
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=test/logs/test_04d_eval_stage1_51_best_%j.out
#SBATCH --error=test/logs/test_04d_eval_stage1_51_best_%j.err

## Evaluate ONLY the best Stage 1.51 (gated) checkpoint.
##
## Auto-selects the best checkpoint with the SAME chain as train_06b_stage3_grpo_optB.sh:
##   1. deterministic pointer stage151_eval_best_ckpt.txt (accuracy-best,
##      written by test_04b_eval_stage1_51.sh)
##   2. best.full_path from the latest test_04b_eval_stage1_51_results_*.json
##   3. fallback: stage151_ckpt.txt (val_loss best/, written by training)
##
## Run test/test_04b_eval_stage1_51.sh first so those artifacts exist, or pass
## BEST_CKPT=<path/to/{sNN_passMM|best}/model.pt> explicitly.
##
## Usage:
##   bash test/test_04d_eval_stage1_51_best.sh                 # 290 records, GPU 0
##   N_SAMPLES=100000 GPUS=0 bash test/test_04d_eval_stage1_51_best.sh   # all val rows
##   BEST_CKPT=/scratch/.../s04_pass02/model.pt bash test/test_04d_eval_stage1_51_best.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-}
CKPT_DIR=${CKPT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51}
_S15_CKPT_DIR=/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_03b_stage1_50_cached
_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
GPUS=${GPUS:-0}
N_SAMPLES=${N_SAMPLES:-290}   # caps at split size; use a big number for "all"
EVAL_SPLIT=${EVAL_SPLIT:-both}  # HF wanglab/kegg re-split too: val=144, test=146 -> both=290
SEED=${SEED:-42}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-800}
_TS=$(date +%Y%m%d_%H%M%S)
RESULTS_JSON=${RESULTS_JSON:-test/logs/test_04d_eval_stage1_51_best_results_${_TS}.json}
RAW_CSV=${RAW_CSV:-test/logs/test_04d_eval_stage1_51_best_raw_${_TS}.csv}
## ─────────────────────────────────────────────────────────────────────────────

## ── Resolve the best Stage 1.51 checkpoint (mirror train_06b_stage3_grpo_optB.sh) ──
BEST_CKPT=${BEST_CKPT:-${STAGE151_CKPT:-}}

## 1) deterministic accuracy-best pointer written by eval_stage1_51_checkpoints.py
if [ -z "${BEST_CKPT:-}" ] && [ -f "$CKPT_DIR/stage151_eval_best_ckpt.txt" ]; then
    BEST_CKPT=$(cat "$CKPT_DIR/stage151_eval_best_ckpt.txt")
    echo "BEST_CKPT (from stage151_eval_best_ckpt.txt): $BEST_CKPT"
fi

## 2) parse best from the latest eval results JSON
if [ -z "${BEST_CKPT:-}" ]; then
    _RESULTS_JSON=$(find "$_PROJECT_DIR/test/logs" \
        -not -path "*/.*/*" \
        -name "test_04b_eval_stage1_51_results_*.json" 2>/dev/null | sort -V | tail -1)
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

## 3) fallback: val_loss best pointer written by training (points at best/model.pt)
if [ -z "${BEST_CKPT:-}" ] && [ -f "$CKPT_DIR/stage151_ckpt.txt" ]; then
    BEST_CKPT=$(cat "$CKPT_DIR/stage151_ckpt.txt")
    echo "BEST_CKPT (fallback, val_loss best from stage151_ckpt.txt): $BEST_CKPT"
fi

if [ -z "${BEST_CKPT:-}" ] || [ ! -f "$BEST_CKPT" ]; then
    echo "ERROR: could not resolve the best Stage 1.51 checkpoint."
    echo "       Run test/test_04b_eval_stage1_51.sh first (writes stage151_eval_best_ckpt.txt),"
    echo "       or set BEST_CKPT=<path/to/{sNN_passMM|best}/model.pt> explicitly."
    exit 1
fi

SRC_DIR=$(dirname "$BEST_CKPT")
for _AUX in thinking_gate.pt dna_injector.pt; do
    if [ ! -f "$SRC_DIR/$_AUX" ]; then
        echo "ERROR: $SRC_DIR is missing $_AUX (Stage 1.51 checkpoint needs model.pt + gate + injector)."
        exit 1
    fi
done

## Base Stage 1 SFT checkpoint (only used to build the model architecture)
STAGE1_CKPT=${STAGE1_CKPT:-}
if [ -z "${STAGE1_CKPT:-}" ]; then
    for _F in "$CKPT_DIR/stage1_ckpt.txt" "$_S15_CKPT_DIR/stage1_ckpt.txt"; do
        if [ -f "$_F" ]; then STAGE1_CKPT=$(cat "$_F"); echo "STAGE1_CKPT (from $_F): $STAGE1_CKPT"; break; fi
    done
fi
if [ -z "${STAGE1_CKPT:-}" ]; then
    echo "ERROR: STAGE1_CKPT not set and no stage1_ckpt.txt found. Set STAGE1_CKPT=<base .ckpt>."
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p test/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"

## Isolate the best checkpoint (model + gate + injector) in a temp dir so
## eval_stage1_51_checkpoints.py (globs CKPT_DIR/s??_pass??/model.pt) scores ONLY
## this one. If the resolved dir is best/ (val_loss fallback), it doesn't match
## the s??_pass?? glob, so use a synthetic matching label.
LABEL=$(basename "$SRC_DIR")
case "$LABEL" in
    s??_pass??) ISO_LABEL="$LABEL" ;;
    *)          ISO_LABEL="s99_pass01" ;;   # synthetic name to satisfy the glob
esac
ISO_DIR=$(mktemp -d)
mkdir -p "$ISO_DIR/$ISO_LABEL"
ln -s "$SRC_DIR/model.pt"         "$ISO_DIR/$ISO_LABEL/model.pt"
ln -s "$SRC_DIR/thinking_gate.pt" "$ISO_DIR/$ISO_LABEL/thinking_gate.pt"
ln -s "$SRC_DIR/dna_injector.pt"  "$ISO_DIR/$ISO_LABEL/dna_injector.pt"
trap 'rm -rf "$ISO_DIR"' EXIT

LOG=test/logs/test_04d_eval_stage1_51_best_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:        bash $0 $*"
echo "Logging to:     $LOG"
echo "Best 1.51 ckpt: $BEST_CKPT  (label=$LABEL -> iso=$ISO_LABEL)"
echo "Base ckpt:      $STAGE1_CKPT"
echo "Isolated dir:   $ISO_DIR"
echo "GPUs:           $GPUS"
echo "N samples:      $N_SAMPLES   Split: $EVAL_SPLIT   Seed: $SEED"
echo "Results JSON:   $RESULTS_JSON"
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

stdbuf -oL -eL python eval_stage1_51_checkpoints.py \
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
echo "=== Best Stage 1.51 checkpoint evaluated: $LABEL ==="
echo "  Accuracy / macro-F1 in the ranking above (single row) and in $RESULTS_JSON"
