#!/bin/bash
#SBATCH --job-name=train_bioreason_anon
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_batch
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160gb
#SBATCH --output=src/scripts/logs/train_bioreason_anon_%j.out
#SBATCH --error=src/scripts/logs/train_bioreason_anon_%j.err

## BioReason SFT on anonymised curriculum dataset (Stage 1: anon genes+mol, keep chr).
## Matches "BioReason with Anonymous dataset no-shuffling" cell in training BioReason.ipynb.
##
## Set KEGG_DATA_DIR to the local directory containing the anonymised KEGG data.
## Default points to Source/data/kegg — copy or symlink the curriculum CSVs there.
##
## Usage:
##   bash Source/tests/sh_train_bioreason_anon.sh [gpu_id]
##   KEGG_DATA_DIR=/path/to/anon_data bash Source/tests/sh_train_bioreason_anon.sh
##   CKPT_PATH=checkpoints/.../last.ckpt bash Source/tests/sh_train_bioreason_anon.sh   # resume
##   sbatch Source/tests/sh_train_bioreason_anon.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-dna_env}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
WANDB_PROJECT=${WANDB_PROJECT:-asBioReasonE5}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
CKPT_PATH=${CKPT_PATH:-}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p scripts/logs
export CUDA_VISIBLE_DEVICES=${1:-0}
nvidia-smi

LOG=scripts/logs/train_bioreason_anon_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
set -x
echo "Command:      bash $0 $*"
echo "Logging:      $LOG"
echo "KEGG CSV:     $KEGG_CSV"

if [ ! -f "$KEGG_CSV" ]; then
    echo "ERROR: KEGG_CSV not found: $KEGG_CSV"
    exit 1
fi

CKPT_ARG=""
[ -n "$CKPT_PATH" ] && CKPT_ARG="--ckpt_path $CKPT_PATH" && echo "Resuming from: $CKPT_PATH"

stdbuf -oL -eL python train_dna_qwen.py \
    --cache_dir                 $CACHE_DIR \
    --wandb_project             $WANDB_PROJECT \
    --wandb_entity              $WANDB_ENTITY \
    --text_model_name           Qwen/Qwen3-1.7B \
    --dna_model_name            evo2_7b_base \
    --dna_is_evo2               True \
    --dna_embedding_layer       blocks.28.mlp.l3 \
    --strategy                  ddp \
    --max_epochs                5 \
    --num_gpus                  1 \
    --batch_size                1 \
    --model_type                dna-llm \
    --dataset_type              kegg \
    --kegg_csv                  "$KEGG_CSV" \
    --max_length_dna            2048 \
    --truncate_dna_per_side     1024 \
    --max_length_text           6000 \
    --merge_val_test_set        True \
    --return_answer_in_batch    True \
    --checkpoint_dir            $CHECKPOINT_DIR \
    $CKPT_ARG

echo "=== Training complete ==="
