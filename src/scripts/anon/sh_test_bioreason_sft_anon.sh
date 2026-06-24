#!/bin/bash
#SBATCH --job-name=test_bioreason_anon
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_batch
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160gb
#SBATCH --output=src/scripts/logs/test_bioreason_anon_%j.out
#SBATCH --error=src/scripts/logs/test_bioreason_anon_%j.err

## Test-only for BioReason on anonymised KEGG — loads checkpoint, skips training (--test_only).
##
## Usage:
##   bash src/scripts/anon/sh_test_bioreason_anon.sh [gpu_id]
##   CKPT_PATH=checkpoints/.../last.ckpt bash src/scripts/anon/sh_test_bioreason_anon.sh
##   sbatch src/scripts/anon/sh_test_bioreason_anon.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-dna_env}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
WANDB_PROJECT=${WANDB_PROJECT:-asBioReasonE5}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_HF=${KEGG_HF:-iit-patna-cse-ai/kegg-anon-global}
CKPT_PATH=${CKPT_PATH:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/bioreason_anon/asBioReasonE5-kegg-Qwen3-1.7B-20260620-155107/asBioReasonE5-kegg-Qwen3-1.7B-epoch=03-val_loss_epoch=0.4109.ckpt}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p scripts/logs
export CUDA_VISIBLE_DEVICES=${1:-1}
nvidia-smi

LOG=scripts/logs/test_bioreason_anon_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
set -x
echo "Command:    bash $0 $*"
echo "Logging:    $LOG"
echo "Checkpoint: $CKPT_PATH"
echo "KEGG HF:    $KEGG_HF"

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
    --kegg_data_dir_huggingface "$KEGG_HF" \
    --max_length_dna            2048 \
    --truncate_dna_per_side     1024 \
    --max_length_text           6000 \
    --merge_val_test_set        True \
    --return_answer_in_batch    True

echo "=== Test complete ==="
