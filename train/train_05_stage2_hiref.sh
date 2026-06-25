#!/bin/bash
#SBATCH --job-name=w9_stage2_hiref
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=train/logs/train_05_stage2_hiref_%j.out
#SBATCH --error=train/logs/train_05_stage2_hiref_%j.err

## Stage 2 Week 9: Offline HiRef alignment using Stage 1.5 checkpoint.
##
## Accepts plain .pt state dict from train_latent_sft.py (Stage 1.5).
## hiref_kegg_align.py supports both .ckpt (Stage 1) and .pt (Stage 1.5).
##
## Usage:
##   STAGE1_CKPT=<path/to/stage1_5/best/model.pt> bash train/train_05_stage2_hiref.sh
##   STAGE1_CKPT=<path> OUTPUT_DIR=stage2_output_w9 bash train/train_05_stage2_hiref.sh
##
## Multi-GPU: if 2+ GPUs are available, shards run in parallel (shard 0 → cuda:0,
## shard 1 → cuda:1). Falls back to sequential on cuda:0 if only 1 GPU is present.

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_05_stage2_hiref}
STAGE1_CKPT=${STAGE1_CKPT:-}
## ─────────────────────────────────────────────────────────────────────────────

## Auto-detect best Stage 1.51 checkpoint if not set
if [ -z "${STAGE1_CKPT:-}" ]; then
    STAGE1_CKPT=$(find /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51 \
        -name "model.pt" 2>/dev/null | sort -V | tail -1)
    [ -n "$STAGE1_CKPT" ] && echo "Auto-detected STAGE1_CKPT: $STAGE1_CKPT" \
        || echo "WARNING: could not auto-detect STAGE1_CKPT from train_04_stage1_51"
fi

if [ -z "$STAGE1_CKPT" ]; then
    echo "ERROR: STAGE1_CKPT is not set."
    echo "Usage: STAGE1_CKPT=<path/to/stage1_5/best/model.pt> bash train/train_05_stage2_hiref.sh"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p train/logs "$OUTPUT_DIR"
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
[ "$NUM_GPUS" -le 0 ] && NUM_GPUS=1

common_args=(
    --ckpt_path              "$STAGE1_CKPT"
    --kegg_dataset           "$KEGG_DATASET"
    --output_dir             "$OUTPUT_DIR"
    --batch_size             2
    --num_shards             2
    --text_model_name        Qwen/Qwen3-1.7B
    --dna_model_name         evo2_7b_base
    --dna_embedding_layer    blocks.28.mlp.l3
    --max_length_text        6000
    --max_length_dna         2048
    --truncate_dna_per_side  1024
    --cache_dir              "$CACHE_DIR"
)

LOG=train/logs/train_05_stage2_hiref_$(date +%Y%m%d_%H%M%S).log
mkdir -p train/logs
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to:  $LOG"
echo "Stage ckpt:  $STAGE1_CKPT"
echo "Output dir:  $OUTPUT_DIR"
echo "GPUs found:  $NUM_GPUS"
nvidia-smi

LOG0=train/logs/train_05_stage2_hiref_shard0_$(date +%Y%m%d_%H%M%S).log
LOG1=train/logs/train_05_stage2_hiref_shard1_$(date +%Y%m%d_%H%M%S).log

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "=== Stage 2 (Week 9) multi-GPU: shard 0 → cuda:0, shard 1 → cuda:1 (parallel) ==="

    echo "Shard 0 -> cuda:0  (log: $LOG0)"
    stdbuf -oL -eL python -u hiref_kegg_align.py \
        "${common_args[@]}" --shard_idx 0 --device cuda:0 \
        2>&1 | tee "$LOG0" &
    PID0=$!

    echo "Shard 1 -> cuda:1  (log: $LOG1)"
    stdbuf -oL -eL python -u hiref_kegg_align.py \
        "${common_args[@]}" --shard_idx 1 --device cuda:1 \
        2>&1 | tee "$LOG1" &
    PID1=$!

    wait $PID0 || { echo "Shard 0 FAILED"; wait $PID1; exit 1; }
    wait $PID1 || { echo "Shard 1 FAILED"; exit 1; }
else
    echo "=== Stage 2 (Week 9) single-GPU: running 2 shards sequentially on cuda:0 ==="

    echo "Shard 0 -> cuda:0  (log: $LOG0)"
    stdbuf -oL -eL python -u hiref_kegg_align.py \
        "${common_args[@]}" --shard_idx 0 --device cuda:0 \
        2>&1 | tee "$LOG0" || { echo "Shard 0 FAILED"; exit 1; }

    echo "Shard 1 -> cuda:0  (log: $LOG1)"
    stdbuf -oL -eL python -u hiref_kegg_align.py \
        "${common_args[@]}" --shard_idx 1 --device cuda:0 \
        2>&1 | tee "$LOG1" || { echo "Shard 1 FAILED"; exit 1; }
fi

echo "=== Both shards done. Merging and running HiRef ==="
LOG_MERGE=train/logs/train_05_stage2_hiref_merge_$(date +%Y%m%d_%H%M%S).log
stdbuf -oL -eL python -u hiref_kegg_align.py \
    --output_dir "$OUTPUT_DIR" \
    --num_shards 2 \
    --merge \
    2>&1 | tee "$LOG_MERGE"

echo "=== Stage 2 (Week 9) complete. Output in $OUTPUT_DIR ==="
