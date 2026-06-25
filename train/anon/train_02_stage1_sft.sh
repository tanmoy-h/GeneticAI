#!/bin/bash
#SBATCH --job-name=w8_stage1_sft_ca_anon
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=train/anon/logs/train_02_stage1_sft_%j.out
#SBATCH --error=train/anon/logs/train_02_stage1_sft_%j.err

## Stage 1 Week 8: SFT from scratch with cross-attention DNA fusion.
##
## Changes from week5 SFT:
##   - use_cross_attention=True  → DNA embeddings fused via cross-attn (no linear proj)
##   - max_clip_loss_weight=0.0  → no contrastive loss
##   - original HF dataset        → load_dataset("wanglab/kegg"), no kegg_csv
##   - checkpoint_dir            → /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02_stage1_sft_anon
##
## Usage:
##   bash train/train_02_stage1_sft.sh [gpu_ids]
##   e.g. bash train/train_02_stage1_sft.sh 0,1

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-stage1-sft-ca-anon}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
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
    --max_clip_loss_weight   0.0
    --ot_weight              0.0
    --kegg_csv               $KEGG_CSV
    --checkpoint_dir         /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02_stage1_sft_anon
)

## Resume options (mutually exclusive, CKPT_PATH takes priority):
##   CKPT_PATH=<path>  — resume from a specific checkpoint
##   RESUME=1          — auto-find last.ckpt in checkpoint_dir
if [ -n "${CKPT_PATH:-}" ]; then
    echo "Resuming from: $CKPT_PATH"
    args+=(--ckpt_path "$CKPT_PATH")
elif [ "${RESUME:-0}" = "1" ]; then
    LAST_CKPT=$(find /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02_stage1_sft_anon -name "last.ckpt" | head -1)
    if [ -z "$LAST_CKPT" ]; then
        echo "ERROR: RESUME=1 but no last.ckpt found in /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02_stage1_sft_anon"
        exit 1
    fi
    echo "RESUME=1: resuming from $LAST_CKPT"
    args+=(--ckpt_path "$LAST_CKPT")
fi

LOG=train/anon/logs/train_02_stage1_sft_$(date +%Y%m%d_%H%M%S).log
mkdir -p train/anon/logs
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to $LOG"
stdbuf -oL -eL python train_dna_qwen.py "${args[@]}"
