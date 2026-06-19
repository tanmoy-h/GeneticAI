#!/bin/bash
#SBATCH --job-name=bioreason_test_anon
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_batch
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --output=bioreason_test_anon_%j_%x.out
#SBATCH --error=bioreason_test_anon_%j_%x.err

## Test a trained BioReason checkpoint against all anonymized KEGG datasets.
## Iterates from stage1 (easiest) to stage5 (hardest anonymization).
## Results are logged to W&B.
##
## Before running, upload all test CSVs to HF:
##   for CSV in stage1_anon_genes_mol_keep_chr.csv \
##               test2_anon_keep_chr_shuffled.csv \
##               test3_anon_no_chr_shuffled.csv \
##               test4_anon_no_chr_no_pathway_shuffled.csv \
##               test5_anon_no_chr_no_pathway_no_reasoning_shuffled.csv; do
##     python upload_anon_dataset_to_hf.py \
##         --csv_path /path/to/$CSV \
##         --repo_id  iitp-cse/kegg-${CSV%.csv}
##   done
##
## Usage:
##   CKPT_PATH=checkpoints/.../last.ckpt bash Source/sh_test_anon.sh

## ── Configuration ──────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=BioReasonAnon
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}

## Trained checkpoint to evaluate (required)
CKPT_PATH=${CKPT_PATH:-}

## HF Hub repo ids for each anonymization stage (change to your uploaded repos)
declare -A HF_REPOS
HF_REPOS["stage1"]="iitp-cse/kegg-anon-stage1"
HF_REPOS["test2"]="iitp-cse/kegg-anon-test2"
HF_REPOS["test3"]="iitp-cse/kegg-anon-test3"
HF_REPOS["test4"]="iitp-cse/kegg-anon-test4"
HF_REPOS["test5"]="iitp-cse/kegg-anon-test5"
## ───────────────────────────────────────────────────────────────────────────────

if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: CKPT_PATH is not set."
    exit 1
fi

echo "CUDA_HOME: $CUDA_HOME"
conda activate $CONDA_ENV
cd "$HOME/BioReason"
nvidia-smi

export CUDA_VISIBLE_DEVICES=${1:-0}

LOG=bioreason_test_anon_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command: bash $0 $*"
echo "CKPT_PATH: $CKPT_PATH"

for STAGE in stage1 test2 test3 test4 test5; do
    HF_REPO="${HF_REPOS[$STAGE]}"
    echo ""
    echo "=== Testing $STAGE → $HF_REPO ==="

    stdbuf -oL -eL srun python train_dna_qwen.py \
        --ckpt_path                 "$CKPT_PATH" \
        --max_epochs                0 \
        --cache_dir                 $CACHE_DIR \
        --wandb_project             $WANDB_PROJECT \
        --wandb_entity              $WANDB_ENTITY \
        --text_model_name           Qwen/Qwen3-1.7B \
        --dna_model_name            InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
        --strategy                  ddp \
        --num_gpus                  1 \
        --batch_size                1 \
        --model_type                dna-llm \
        --dataset_type              kegg \
        --max_length_dna            2048 \
        --truncate_dna_per_side     1024 \
        --max_length_text           6000 \
        --merge_val_test_set        True \
        --return_answer_in_batch    True \
        --kegg_data_dir_huggingface "$HF_REPO"

    echo "=== Done $STAGE ==="
done

echo ""
echo "=== All stages complete ==="
