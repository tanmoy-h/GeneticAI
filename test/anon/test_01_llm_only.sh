#!/bin/bash
## Evaluate an LLM-only baseline checkpoint (no DNA fusion) — ANON.
## Table row "LLM-only" (anon columns). Mirrors test/test_01_llm_only.sh with
## the anonymized KEGG CSV.
##
## Usage:
##   CKPT_PATH=<path_to_.ckpt> bash test/anon/test_01_llm_only.sh [gpu_id]

CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}

if [ -z "${CKPT_PATH:-}" ]; then
    echo "ERROR: CKPT_PATH is not set."
    echo "Usage: CKPT_PATH=<path> bash test/anon/test_01_llm_only.sh [gpu_id]"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p test/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}

LOG=test/anon/logs/test_01_llm_only_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to $LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint:  $CKPT_PATH"
nvidia-smi

stdbuf -oL -eL python train_dna_qwen.py \
    --cache_dir              $CACHE_DIR \
    --text_model_name        Qwen/Qwen3-1.7B \
    --model_type             llm \
    --dataset_type           kegg \
    --kegg_csv               $KEGG_CSV \
    --batch_size             1 \
    --num_gpus               1 \
    --strategy               ddp \
    --max_length_text        6000 \
    --merge_val_test_set     True \
    --return_answer_in_batch True \
    --ckpt_path              "$CKPT_PATH" \
    --test_only
