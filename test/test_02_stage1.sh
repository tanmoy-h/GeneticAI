#!/bin/bash
## Test a Stage 1 SFT checkpoint with --test_only.
##
## Usage:
##   CKPT_PATH=<path_to_.ckpt> bash test/test_02_stage1.sh [gpu_id]
##   e.g. CKPT_PATH=checkpoints/.../epoch=03-val_loss_epoch=0.4292.ckpt \
##        bash test/test_02_stage1.sh 0

CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface

CKPT_DIR=${CKPT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02_stage1_sft}

## Auto-pick best checkpoint (lowest val_loss_epoch in filename) when CKPT_PATH unset.
## Stage 1 SFT keeps save_top_k=2 by val_loss + last.ckpt; the best is the lowest
## val_loss encoded in the filename. Falls back to last.ckpt.
if [ -z "${CKPT_PATH:-}" ]; then
    CKPT_PATH=$(CKPT_DIR="$CKPT_DIR" python3 -c '
import glob, os, re
d = os.environ["CKPT_DIR"]
best, best_loss = None, float("inf")
for f in glob.glob(os.path.join(d, "**", "*.ckpt"), recursive=True):
    b = os.path.basename(f)
    if b == "last.ckpt":
        continue
    m = re.search(r"val_loss_epoch=([0-9]+\.[0-9]+)", b) or re.search(r"-([0-9]+\.[0-9]+)\.ckpt$", b)
    if not m:
        continue
    v = float(m.group(1))
    if v < best_loss:
        best, best_loss = f, v
print(best or "")
' 2>/dev/null)
    if [ -n "$CKPT_PATH" ]; then
        echo "CKPT_PATH (best val_loss, auto): $CKPT_PATH"
    else
        CKPT_PATH=$(find "$CKPT_DIR" -name last.ckpt 2>/dev/null | head -1)
        [ -n "$CKPT_PATH" ] && echo "CKPT_PATH (fallback, last.ckpt): $CKPT_PATH"
    fi
fi

if [ -z "${CKPT_PATH:-}" ]; then
    echo "ERROR: CKPT_PATH not set and no checkpoint found in $CKPT_DIR."
    echo "Usage: CKPT_PATH=<path> bash test/test_02_stage1.sh [gpu_id]"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p test/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}

LOG=test/logs/test_02_stage1_$(date +%Y%m%d_%H%M%S).log
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
