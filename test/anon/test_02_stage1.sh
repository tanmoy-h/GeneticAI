#!/bin/bash
## Test a Stage 1 SFT checkpoint with --test_only.
##
## Usage:
##   CKPT_PATH=<path_to_.ckpt> bash week8tests/sh_test_stage1_w8.sh [gpu_id]
##   e.g. CKPT_PATH=checkpoints/.../epoch=03-val_loss_epoch=0.4292.ckpt \
##        bash week8tests/sh_test_stage1_w8.sh 0

CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}

if [ -z "${CKPT_PATH:-}" ]; then
    echo "ERROR: CKPT_PATH is not set."
    echo "Usage: CKPT_PATH=<path> bash week8tests/sh_test_stage1_w8.sh [gpu_id]"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p week8tests/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}

LOG=week8tests/logs/test_stage1_w8_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to $LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint:  $CKPT_PATH"
nvidia-smi

stdbuf -oL -eL python train_dna_qwen.py \
    --cache_dir              $CACHE_DIR \
    --text_model_name        Qwen/Qwen3-1.7B \
    --dna_model_name         evo2_7b_base \
    --dna_is_evo2            True \
    --dna_embedding_layer    blocks.28.mlp.l3 \
    --model_type             dna-llm \
    --kegg_csv               $KEGG_CSV \
    --dataset_type           kegg \
    --batch_size             1 \
    --num_gpus               1 \
    --strategy               ddp \
    --max_length_text        6000 \
    --max_length_dna         2048 \
    --truncate_dna_per_side  1024 \
    --merge_val_test_set     True \
    --return_answer_in_batch True \
    --use_cross_attention    True \
    --ckpt_path              "$CKPT_PATH" \
    --test_only
