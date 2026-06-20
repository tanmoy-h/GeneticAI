#!/bin/bash
#SBATCH --job-name=test_llm_only_anon
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_batch
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --output=src/scripts/logs/test_llm_only_anon_%j.out
#SBATCH --error=src/scripts/logs/test_llm_only_anon_%j.err

## Test-only for Normal LLM Qwen3 on the anonymous KEGG dataset.
## Evaluates the base LLM (no DNA) against the anonymized gene names CSV.
##
## Usage:
##   CKPT_PATH=checkpoints/.../last.ckpt bash src/scripts/anon/sh_test_llm_only_anon.sh [gpu_id]
##   sbatch src/scripts/anon/sh_test_llm_only_anon.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-dna_env}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
WANDB_PROJECT=${WANDB_PROJECT:-LLMonlyE5_anon}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
CKPT_PATH=${CKPT_PATH:-}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
## ─────────────────────────────────────────────────────────────────────────────

if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: CKPT_PATH is not set."
    echo "Usage: CKPT_PATH=checkpoints/.../last.ckpt bash $0"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."   # → src/
mkdir -p scripts/logs
export CUDA_VISIBLE_DEVICES=${1:-1}
nvidia-smi

LOG=scripts/logs/test_llm_only_anon_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
set -x
echo "Command:    bash $0 $*"
echo "Logging:    $LOG"
echo "Checkpoint: $CKPT_PATH"
echo "KEGG CSV:   $KEGG_CSV"

## Prefer local CSV; fall back to named KEGG if not found
if [ -n "$KEGG_CSV" ] && [ -f "$KEGG_CSV" ]; then
    echo "Dataset:    CSV → $KEGG_CSV"
    DATASET_ARG="--kegg_csv $KEGG_CSV"
else
    echo "WARNING: KEGG_CSV not found ($KEGG_CSV) — falling back to named wanglab/kegg"
    DATASET_ARG="--dataset_type kegg"
fi

stdbuf -oL -eL python train_dna_qwen.py \
    --ckpt_path                 "$CKPT_PATH" \
    --test_only                 \
    --cache_dir                 $CACHE_DIR \
    --wandb_project             $WANDB_PROJECT \
    --wandb_entity              $WANDB_ENTITY \
    --text_model_name           Qwen/Qwen3-1.7B \
    --strategy                  ddp \
    --num_gpus                  1 \
    --batch_size                1 \
    --model_type                llm \
    $DATASET_ARG \
    --max_length_dna            2048 \
    --truncate_dna_per_side     1024 \
    --max_length_text           6000 \
    --merge_val_test_set        True \
    --return_answer_in_batch    True

echo "=== Test complete ==="
