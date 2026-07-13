#!/bin/bash
#SBATCH --job-name=w8_llm_only
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=train/logs/train_01_llm_only_%j.out
#SBATCH --error=train/logs/train_01_llm_only_%j.err

## LLM-only baseline (no DNA fusion) — table row "LLM-only".
##
## --model_type llm  →  format_kegg_for_llm() writes the reference/variant
##   sequences into the QUESTION TEXT and sets dna_sequences=["",""]; no DNA
##   encoder is built (dna_model_name unset → dna_model is None, forward skips
##   DNA fusion). Same Qwen3-1.7B text backbone as GenoMorph, so it is a clean
##   apples-to-apples "LLM without DNA fusion" baseline.
##
## Usage:
##   bash train/train_01_llm_only.sh [gpu_ids]

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-stage1-llm-only}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p train/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0,1}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES  NUM_GPUS: $NUM_GPUS"
nvidia-smi

args=(
    --cache_dir              $CACHE_DIR
    --wandb_project          $WANDB_PROJECT
    --wandb_entity           $WANDB_ENTITY
    --text_model_name        Qwen/Qwen3-1.7B
    --model_type             llm
    --dataset_type           kegg
    --max_epochs             5
    --batch_size             1
    --num_gpus               $NUM_GPUS
    --strategy               ddp
    --max_length_text        6000
    --merge_val_test_set     True
    --return_answer_in_batch True
    --checkpoint_dir         /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_01_llm_only
)

## Resume options (mutually exclusive, CKPT_PATH takes priority).
if [ -n "${CKPT_PATH:-}" ]; then
    echo "Resuming from: $CKPT_PATH"
    args+=(--ckpt_path "$CKPT_PATH")
elif [ "${RESUME:-0}" = "1" ]; then
    LAST_CKPT=$(find /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_01_llm_only -name "last.ckpt" | head -1)
    if [ -z "$LAST_CKPT" ]; then
        echo "ERROR: RESUME=1 but no last.ckpt found in /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_01_llm_only"
        exit 1
    fi
    echo "RESUME=1: resuming from $LAST_CKPT"
    args+=(--ckpt_path "$LAST_CKPT")
fi

LOG=train/logs/train_01_llm_only_$(date +%Y%m%d_%H%M%S).log
mkdir -p train/logs
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to $LOG"
stdbuf -oL -eL python train_dna_qwen.py "${args[@]}"
