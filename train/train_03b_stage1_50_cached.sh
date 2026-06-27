#!/bin/bash
#SBATCH --job-name=w9_stage1_50_cached
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=train/logs/train_03b_stage1_50_cached_%j.out
#SBATCH --error=train/logs/train_03b_stage1_50_cached_%j.err

## Stage 1.5 Week 9: LatentSp cold-start SFT with cached DNA embeddings.
##
## Uses precomputed Evo2 embeddings to skip the Evo2 forward pass at
## training time, freeing ~14 GB GPU VRAM.  Run sh_precompute_dna_w9.sh
## first to generate the cache file.
##
## Two entropy modes:
##   ENTROPY_MODE=global  (default) — epoch-level recompute, 1 forward pass per
##                                    curriculum step.  Faster.
##   ENTROPY_MODE=inline            — per-batch entropy, matches LatentSp Algorithm 1
##                                    exactly.  ~2x compute per batch.
##
## Usage:
##   STAGE1_CKPT=<path> bash train/train_03b_stage1_50_cached.sh [gpu_id]
##   STAGE1_CKPT=<path> ENTROPY_MODE=inline bash train/train_03b_stage1_50_cached.sh 0
##   STAGE1_CKPT=<path> PASSES_PER_STEP=2 bash train/train_03b_stage1_50_cached.sh 1
##   STAGE1_CKPT=<path> DNA_CACHE=/custom/path/cache.pt bash train/train_03b_stage1_50_cached.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-dna-sft-week9-stage1-50}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_03b_stage1_50_cached}
ENTROPY_MODE=${ENTROPY_MODE:-global}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
## ─────────────────────────────────────────────────────────────────────────────

STAGE1_CKPT=${STAGE1_CKPT:-}

## Auto-detect best Stage 1 SFT checkpoint if not set
if [ -z "${STAGE1_CKPT:-}" ]; then
    STAGE1_CKPT=$(find /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02_stage1_sft \
        -name "*.ckpt" ! -name "last.ckpt" 2>/dev/null \
        | awk -F'val_loss_epoch=' 'NF>1{print $2, $0}' | sort -n | head -1 | cut -d' ' -f2-)
    [ -n "$STAGE1_CKPT" ] && echo "Auto-detected STAGE1_CKPT: $STAGE1_CKPT" \
        || echo "WARNING: could not auto-detect STAGE1_CKPT from train_02_stage1_sft"
fi

if [ -z "${STAGE1_CKPT:-}" ]; then
    echo "ERROR: STAGE1_CKPT is not set."
    echo "Usage: STAGE1_CKPT=<path> bash train/train_03b_stage1_50_cached.sh [gpu_id]"
    exit 1
fi

if [ ! -f "$DNA_CACHE" ]; then
    echo "WARNING: DNA cache file not found: $DNA_CACHE"
    echo "         Run sh_precompute_dna_w9.sh first to generate it."
    echo "         Continuing without cache — Evo2 will run at training time."
    DNA_CACHE=""
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p train/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}

## Compute train size for --max_entropy_samples
if [ -n "$KEGG_CSV" ]; then
    TRAIN_SIZE=$(python3 -c "
from genomorph.dataset.kegg import load_kegg_from_anon_csv
ds = load_kegg_from_anon_csv('$KEGG_CSV')
print(len(ds['train']))
" 2>/dev/null | grep -E '^[0-9]+$' | tail -1)
else
    TRAIN_SIZE=$(python3 -c "
from datasets import load_dataset
ds = load_dataset('$KEGG_DATASET', 'default', cache_dir='$CACHE_DIR')
print(len(ds['train']))
" 2>/dev/null | grep -E '^[0-9]+$' | tail -1)
fi
if [ -z "$TRAIN_SIZE" ] || [ "$TRAIN_SIZE" -le 0 ] 2>/dev/null; then
    echo "WARNING: Could not read TRAIN_SIZE — falling back to max_entropy_samples=500"
    TRAIN_SIZE=500
fi

LOG=train/logs/train_03b_stage1_50_cached_${ENTROPY_MODE}_$(date +%Y%m%d_%H%M%S).log
mkdir -p train/logs
exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "CUDA:          $CUDA_VISIBLE_DEVICES"
echo "Stage 1 ckpt:  $STAGE1_CKPT"
echo "Entropy mode:  $ENTROPY_MODE"
echo "Passes/step:   ${PASSES_PER_STEP:-1}"
echo "Output dir:    $OUTPUT_DIR"
echo "KEGG dataset:  ${KEGG_CSV:-$KEGG_DATASET}"
echo "DNA cache:     ${DNA_CACHE:-<not set — Evo2 will run live>}"
nvidia-smi

## Build dataset arg: CSV takes priority over HF dataset name
if [ -n "$KEGG_CSV" ]; then
    KEGG_ARG="--kegg_csv $KEGG_CSV"
else
    KEGG_ARG="--kegg_dataset $KEGG_DATASET"
fi

## Build DNA cache arg (empty string → omit flag → Evo2 runs live)
if [ -n "$DNA_CACHE" ]; then
    DNA_CACHE_ARG="--dna_cache $DNA_CACHE"
else
    DNA_CACHE_ARG=""
fi

stdbuf -oL -eL python train_latent_sft_cached.py \
    --stage1_ckpt            "$STAGE1_CKPT" \
    $KEGG_ARG \
    --output_dir             "$OUTPUT_DIR" \
    --text_model_name        Qwen/Qwen3-1.7B \
    --dna_model_name         evo2_7b_base \
    --dna_embedding_layer    blocks.28.mlp.l3 \
    --entropy_mode           "$ENTROPY_MODE" \
    --max_latent_steps       4 \
    --passes_per_step        ${PASSES_PER_STEP:-1} \
    --data_refresh_ratio     0.1 \
    --batch_size             1 \
    --grad_accum             8 \
    --learning_rate          5e-5 \
    --max_length_text        6000 \
    --max_length_dna         2048 \
    --truncate_dna_per_side  1024 \
    --entropy_batch_size     1 \
    --max_entropy_samples    $TRAIN_SIZE \
    --log_every              20 \
    --wandb_project          "$WANDB_PROJECT" \
    --wandb_entity           "$WANDB_ENTITY" \
    $DNA_CACHE_ARG \
    --cache_dir              "$CACHE_DIR" \
    --device                 cuda
