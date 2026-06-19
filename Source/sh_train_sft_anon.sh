#!/bin/bash
#SBATCH --job-name=bioreason_sft_anon
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_batch
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --output=bioreason_sft_anon_%j_%x.out
#SBATCH --error=bioreason_sft_anon_%j_%x.err

## BioReason SFT training on an anonymized KEGG dataset (HF Hub).
##
## Before running:
##   python upload_anon_dataset_to_hf.py \
##       --csv_path /path/to/stage1_anon_genes_mol_keep_chr.csv \
##       --repo_id  iitp-cse/kegg-anon-stage1
##
## Then set HF_DATASET below to the uploaded repo id.
##
## Usage:
##   bash Source/sh_train_sft_anon.sh [gpu_ids]
##   CKPT_PATH=checkpoints/.../last.ckpt bash Source/sh_train_sft_anon.sh   # resume

## ── Configuration ──────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=BioReasonAnon
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}

## HF Hub repo id of the uploaded anonymized dataset
HF_DATASET=iitp-cse/kegg-anon-stage1   # Change to your uploaded repo id

## Resume from checkpoint (optional)
CKPT_PATH=${CKPT_PATH:-}
## ───────────────────────────────────────────────────────────────────────────────

echo "CUDA_HOME: $CUDA_HOME"
echo "which python: $(which python)"

conda activate $CONDA_ENV
cd "$HOME/BioReason"
nvidia-smi

export CUDA_VISIBLE_DEVICES=${1:-0,1}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

LOG=bioreason_sft_anon_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command: bash $0 $*"
echo "HF_DATASET: $HF_DATASET"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES  NUM_GPUS: $NUM_GPUS"

args=(
    --cache_dir                     $CACHE_DIR
    --wandb_project                 $WANDB_PROJECT
    --wandb_entity                  $WANDB_ENTITY
    --text_model_name               Qwen/Qwen3-1.7B
    --dna_model_name                InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
    --strategy                      ddp
    --max_epochs                    5
    --num_gpus                      $NUM_GPUS
    --batch_size                    1
    --model_type                    dna-llm
    --dataset_type                  kegg
    --max_length_dna                2048
    --truncate_dna_per_side         1024
    --max_length_text               6000
    --merge_val_test_set            True
    --return_answer_in_batch        True
    --kegg_data_dir_huggingface     $HF_DATASET
)

if [ -n "$CKPT_PATH" ]; then
    args+=(--ckpt_path "$CKPT_PATH")
fi

stdbuf -oL -eL srun python train_dna_qwen.py "${args[@]}"
