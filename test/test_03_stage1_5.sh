#!/bin/bash
#SBATCH --job-name=w9_test_stage1_5
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=test/logs/test_03_stage1_5_%j.out
#SBATCH --error=test/logs/test_03_stage1_5_%j.err

## Evaluate a Stage 1.5 checkpoint (plain model.pt from train_latent_sft.py).
##
## Usage:
##   CKPT_PATH=<path/to/model.pt> bash test/test_03_stage1_5.sh [gpu_id]
##   CKPT_PATH=<path> SPLIT=test bash test/test_03_stage1_5.sh 0

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
SPLIT=${SPLIT:-val}
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-}
## ─────────────────────────────────────────────────────────────────────────────

if [ -z "${CKPT_PATH:-}" ]; then
    echo "ERROR: CKPT_PATH is not set."
    echo "Usage: CKPT_PATH=<path/to/model.pt> bash test/test_03_stage1_5.sh [gpu_id]"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p test/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}

OUTPUT_DIR=$(dirname "$CKPT_PATH")

LOG=test/logs/test_03_stage1_5_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:      bash $0 $*"
echo "Logging to:   $LOG"
echo "CKPT_PATH:    $CKPT_PATH"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "SPLIT:        $SPLIT"
echo "KEGG source:  ${KEGG_CSV:-$KEGG_DATASET}"
nvidia-smi

## Build dataset arg: CSV takes priority over HF dataset name
if [ -n "$KEGG_CSV" ]; then
    KEGG_ARG="--kegg_csv $KEGG_CSV"
else
    KEGG_ARG="--kegg_dataset $KEGG_DATASET"
fi

stdbuf -oL -eL python test_latent_sft.py \
    --ckpt_path              "$CKPT_PATH" \
    --output_dir             "$OUTPUT_DIR" \
    --split                  "$SPLIT" \
    $KEGG_ARG \
    --text_model_name        Qwen/Qwen3-1.7B \
    --dna_model_name         evo2_7b_base \
    --dna_embedding_layer    blocks.28.mlp.l3 \
    --max_length_text        6000 \
    --max_length_dna         2048 \
    --max_new_tokens         800 \
    --cache_dir              "$CACHE_DIR" \
    --device                 cuda
