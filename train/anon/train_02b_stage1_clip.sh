#!/bin/bash
#SBATCH --job-name=w8_stage1_sft_clip_anon
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=train/anon/logs/train_02b_stage1_clip_%j.out
#SBATCH --error=train/anon/logs/train_02b_stage1_clip_%j.err

## Stage 1 ablation: CrossAttn SFT + CLIP contrastive alignment — ANON.
## Identical to train/anon/train_02_stage1_sft.sh EXCEPT max_clip_loss_weight > 0.
## Table row "— + CLIP (ablated)" (anon columns).
##
## Evaluate with the same script as plain Stage 1 anon:
##   CKPT_PATH=<.ckpt> bash test/anon/test_02_stage1.sh [gpu_id]
##
## Usage:
##   bash train/anon/train_02b_stage1_clip.sh [gpu_ids]
##   CLIP_WEIGHT=0.2 bash train/anon/train_02b_stage1_clip.sh 0,1

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-stage1-sft-clip-anon}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
CLIP_WEIGHT=${CLIP_WEIGHT:-0.1}   # CLIP contrastive loss weight (0.0 = plain Stage 1)
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p train/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0,1}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES  NUM_GPUS: $NUM_GPUS"
echo "CLIP weight: $CLIP_WEIGHT"
nvidia-smi

args=(
    --cache_dir              $CACHE_DIR
    --wandb_project          $WANDB_PROJECT
    --wandb_entity           $WANDB_ENTITY
    --text_model_name        Qwen/Qwen3-1.7B
    --dna_model_name         evo2_7b_base
    --dna_is_evo2            True
    --dna_embedding_layer    blocks.28.mlp.l3
    --model_type             dna-llm
    --dataset_type           kegg
    --kegg_csv               $KEGG_CSV
    --max_epochs             5
    --batch_size             1
    --num_gpus               $NUM_GPUS
    --strategy               ddp
    --max_length_text        6000
    --max_length_dna         2048
    --truncate_dna_per_side  1024
    --merge_val_test_set     True
    --return_answer_in_batch True
    --use_cross_attention    True
    --max_clip_loss_weight   $CLIP_WEIGHT
    --ot_weight              0.0
    --checkpoint_dir         /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02b_stage1_clip_anon
)

if [ -n "${CKPT_PATH:-}" ]; then
    echo "Resuming from: $CKPT_PATH"
    args+=(--ckpt_path "$CKPT_PATH")
elif [ "${RESUME:-0}" = "1" ]; then
    LAST_CKPT=$(find /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02b_stage1_clip_anon -name "last.ckpt" | head -1)
    if [ -z "$LAST_CKPT" ]; then
        echo "ERROR: RESUME=1 but no last.ckpt found in /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02b_stage1_clip_anon"
        exit 1
    fi
    echo "RESUME=1: resuming from $LAST_CKPT"
    args+=(--ckpt_path "$LAST_CKPT")
fi

LOG=train/anon/logs/train_02b_stage1_clip_$(date +%Y%m%d_%H%M%S).log
mkdir -p train/anon/logs
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to $LOG"
stdbuf -oL -eL python train_dna_qwen.py "${args[@]}"
