#!/bin/bash
#SBATCH --job-name=test_bioreason
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_batch
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160gb
#SBATCH --output=Source/tests/logs/test_bioreason_%j.out
#SBATCH --error=Source/tests/logs/test_bioreason_%j.err

## Test-only for BioReason — loads checkpoint, skips training (--test_only).
##
## Usage:
##   CKPT_PATH=checkpoints/.../last.ckpt bash Source/tests/sh_test_bioreason.sh [gpu_id]
##   sbatch Source/tests/sh_test_bioreason.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-dna_env}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
WANDB_PROJECT=${WANDB_PROJECT:-BioReasonE5}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
CKPT_PATH=${CKPT_PATH:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/BioReasonE5-kegg-Qwen3-1.7B-20260531-162003/BioReasonE5-kegg-Qwen3-1.7B-epoch=03-val_loss_epoch=0.4305.ckpt}
## ─────────────────────────────────────────────────────────────────────────────


module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p tests/logs
export CUDA_VISIBLE_DEVICES=${1:-1}
nvidia-smi

LOG=tests/logs/test_bioreason_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
set -x
echo "Command:    bash $0 $*"
echo "Logging:    $LOG"
echo "Checkpoint: $CKPT_PATH"

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
    --max_length_dna            2048 \
    --truncate_dna_per_side     1024 \
    --max_length_text           6000 \
    --merge_val_test_set        True \
    --return_answer_in_batch    True

echo "=== Test complete ==="
