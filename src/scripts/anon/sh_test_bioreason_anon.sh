#!/bin/bash
#SBATCH --job-name=test_bioreason_anon
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_batch
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160gb
#SBATCH --output=Source/tests/logs/test_bioreason_anon_%j.out
#SBATCH --error=Source/tests/logs/test_bioreason_anon_%j.err

## Test-only for BioReason on anonymised KEGG — loads checkpoint, skips training (--test_only).
##
## Usage:
##   CKPT_PATH=checkpoints/.../last.ckpt bash Source/tests/sh_test_bioreason_anon.sh [gpu_id]
##   CKPT_PATH=... KEGG_DATA_DIR=/path/to/anon_data bash Source/tests/sh_test_bioreason_anon.sh
##   sbatch Source/tests/sh_test_bioreason_anon.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-dna_env}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
WANDB_PROJECT=${WANDB_PROJECT:-asBioReasonE5}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_DATA_DIR=${KEGG_DATA_DIR:-data/kegg}
CKPT_PATH=${CKPT_PATH:-}
## ─────────────────────────────────────────────────────────────────────────────

if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: CKPT_PATH is not set."
    echo "Usage: CKPT_PATH=checkpoints/.../last.ckpt bash $0"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p tests/logs
export CUDA_VISIBLE_DEVICES=${1:-1}
nvidia-smi

LOG=tests/logs/test_bioreason_anon_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
set -x
echo "Command:    bash $0 $*"
echo "Logging:    $LOG"
echo "Checkpoint: $CKPT_PATH"
echo "KEGG data:  $KEGG_DATA_DIR"

if [ ! -d "$KEGG_DATA_DIR" ]; then
    echo "ERROR: KEGG_DATA_DIR not found: $KEGG_DATA_DIR"
    exit 1
fi

stdbuf -oL -eL python train_dna_qwen.py \
    --ckpt_path                 "$CKPT_PATH" \
    --test_only                 \
    --cache_dir                 $CACHE_DIR \
    --wandb_project             $WANDB_PROJECT \
    --wandb_entity              $WANDB_ENTITY \
    --text_model_name           Qwen/Qwen3-1.7B \
    --dna_model_name            evo2_7b_base \
    --dna_is_evo2               True \
    --dna_embedding_layer       blocks.28.mlp.l3 \
    --strategy                  ddp \
    --num_gpus                  1 \
    --batch_size                1 \
    --model_type                dna-llm \
    --dataset_type              kegg \
    --kegg_data_dir_local       "$KEGG_DATA_DIR" \
    --max_length_dna            2048 \
    --truncate_dna_per_side     1024 \
    --max_length_text           6000 \
    --merge_val_test_set        True \
    --return_answer_in_batch    True

echo "=== Test complete ==="
