#!/bin/bash
#SBATCH --job-name=eval_bioreason_grpo
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_batch
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160gb
#SBATCH --output=src/scripts/logs/eval_bioreason_grpo_%j.out
#SBATCH --error=src/scripts/logs/eval_bioreason_grpo_%j.err

## Evaluate GRPO-trained BioReason (real) on KEGG val+test.
##
## Usage:
##   bash src/scripts/real/sh_eval_bioreason_grpo.sh [gpu_id]
##   GRPO_CKPT=/path/to/checkpoint-1000 bash src/scripts/real/sh_eval_bioreason_grpo.sh
##   sbatch src/scripts/real/sh_eval_bioreason_grpo.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-dna_env}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/bioreason_grpo}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/eval_results/bioreason_grpo}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p scripts/logs
export CUDA_VISIBLE_DEVICES=${1:-0}
nvidia-smi

LOG=scripts/logs/eval_bioreason_grpo_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
set -x
echo "Command:      bash $0 $*"
echo "Logging:      $LOG"
echo "CHECKPOINT:   $CHECKPOINT_DIR"
echo "OUTPUT:       $OUTPUT_DIR"

# Auto-detect latest checkpoint if GRPO_CKPT not set
if [ -z "$GRPO_CKPT" ]; then
    GRPO_CKPT=$(ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$GRPO_CKPT" ]; then
        echo "ERROR: No checkpoint-* found under $CHECKPOINT_DIR"
        exit 1
    fi
fi
echo "Using checkpoint: $GRPO_CKPT"

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

stdbuf -oL -eL python eval_grpo.py \
  --grpo_checkpoint             "$GRPO_CKPT" \
  --dataset_name                wanglab/kegg \
  --text_model_name             Qwen/Qwen3-1.7B \
  --dna_model_name              evo2_7b_base \
  --dna_is_evo2                 True \
  --dna_embedding_layer         blocks.28.mlp.l3 \
  --cache_dir                   $CACHE_DIR \
  --truncate_dna_per_side       1024 \
  --lora_r                      32 \
  --lora_alpha                  64 \
  --lora_dropout                0 \
  --max_new_tokens              800 \
  --temperature                 0 \
  --output_dir                  "$OUTPUT_DIR" \
  --tag                         grpo_real

echo "=== GRPO real evaluation complete ==="
