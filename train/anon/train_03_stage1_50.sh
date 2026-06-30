#!/bin/bash
#SBATCH --job-name=w9_stage1_50_anon
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=train/anon/logs/train_03_stage1_50_%j.out
#SBATCH --error=train/anon/logs/train_03_stage1_50_%j.err

## Stage 1.5 Week 9: LatentSp cold-start SFT.
##
## Two entropy modes:
##   ENTROPY_MODE=global  (default) — epoch-level recompute, 1 forward pass per
##                                    curriculum step.  Faster.
##   ENTROPY_MODE=inline            — per-batch entropy, matches LatentSp Algorithm 1
##                                    exactly.  ~2x compute per batch.
##
## Usage:
##   STAGE1_CKPT=<path> bash train/train_03_stage1_50.sh [gpu_id]
##   STAGE1_CKPT=<path> ENTROPY_MODE=inline bash train/train_03_stage1_50.sh 0
##   STAGE1_CKPT=<path> PASSES_PER_STEP=2 bash train/train_03_stage1_50.sh 1

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-dna-sft-week9-stage1-50}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_03_stage1_50_anon}
ENTROPY_MODE=${ENTROPY_MODE:-global}
## ─────────────────────────────────────────────────────────────────────────────

STAGE1_CKPT=${STAGE1_CKPT:-hf://iit-patna-cse-ai/GenoMorph/stage1_sft/dna-sft-week8-ca-kegg-Qwen3-1.7B-epoch=03-val_loss_epoch=0.4292.ckpt}
_S1_LOCAL_DIR=/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02_stage1_sft_anon

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p train/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}

## If hf:// URI: download into local checkpoint dir, then always auto-detect best
if [[ "${STAGE1_CKPT:-}" == hf://* ]]; then
    _HF_REPO=$(echo "$STAGE1_CKPT" | sed 's|hf://||' | cut -d'/' -f1-2)
    _HF_FILE=$(echo "$STAGE1_CKPT" | sed "s|hf://${_HF_REPO}/||")
    _HF_RUNNAME=$(basename "$_HF_FILE" .ckpt | sed 's/-epoch=.*//')
    _HF_LOCAL_SUBDIR="$_S1_LOCAL_DIR/${_HF_RUNNAME}-hf"
    echo "Downloading HF checkpoint into $_HF_LOCAL_SUBDIR: $STAGE1_CKPT"
    mkdir -p "$_HF_LOCAL_SUBDIR"
    if ! HF_HUB_DISABLE_PROGRESS_BARS=1 python3 -c "
from huggingface_hub import hf_hub_download
import sys
try:
    p = hf_hub_download('$_HF_REPO', '$_HF_FILE', local_dir='$_HF_LOCAL_SUBDIR')
    print('Downloaded:', p)
except Exception as e:
    print('WARNING: HF download error:', str(e))
    sys.exit(1)
"; then
        echo "WARNING: HF download failed — will use best existing local checkpoint"
    fi
    STAGE1_CKPT=""
fi

## Auto-detect best checkpoint by val_loss_epoch (always runs for hf:// and empty)
if [ -z "${STAGE1_CKPT:-}" ]; then
    STAGE1_CKPT=$(find "$_S1_LOCAL_DIR" \
        -name "*.ckpt" ! -name "last.ckpt" 2>/dev/null \
        | awk -F'val_loss_epoch=' 'NF>1{print $2, $0}' | sort -n | head -1 | cut -d' ' -f2-)
    [ -n "$STAGE1_CKPT" ] && echo "Auto-detected STAGE1_CKPT: $STAGE1_CKPT" \
        || echo "WARNING: could not auto-detect STAGE1_CKPT from train_02_stage1_sft_anon"
fi

if [ -z "${STAGE1_CKPT:-}" ]; then
    echo "ERROR: STAGE1_CKPT is not set and could not be resolved."
    echo "Usage: STAGE1_CKPT=<path|hf://org/repo/file> bash train/anon/train_03_stage1_50.sh [gpu_id]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "$STAGE1_CKPT" > "$OUTPUT_DIR/stage1_ckpt.txt"

## Compute train size for --max_entropy_samples
if [ -n "$KEGG_CSV" ]; then
    TRAIN_SIZE=$(python3 -c "
from genomorph.dataset.kegg import load_kegg_from_anon_csv
ds = load_kegg_from_anon_csv('$KEGG_CSV')
print(len(ds['train']))
" 2>/dev/null)
else
    TRAIN_SIZE=$(python3 -c "
from datasets import load_dataset
ds = load_dataset('$KEGG_DATASET', 'default', cache_dir='$CACHE_DIR')
print(len(ds['train']))
" 2>/dev/null)
fi
if [ -z "$TRAIN_SIZE" ] || [ "$TRAIN_SIZE" -le 0 ] 2>/dev/null; then
    echo "WARNING: Could not read TRAIN_SIZE — falling back to max_entropy_samples=500"
    TRAIN_SIZE=500
fi

LOG=train/anon/logs/train_03_stage1_50_${ENTROPY_MODE}_$(date +%Y%m%d_%H%M%S).log
mkdir -p train/anon/logs
exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "CUDA:          $CUDA_VISIBLE_DEVICES"
echo "Stage 1 ckpt:  $STAGE1_CKPT"
echo "Entropy mode:  $ENTROPY_MODE"
echo "Passes/step:   ${PASSES_PER_STEP:-1}"
echo "Output dir:    $OUTPUT_DIR"
echo "KEGG dataset:  ${KEGG_CSV:-$KEGG_DATASET}"
nvidia-smi

## Build dataset arg: CSV takes priority over HF dataset name
if [ -n "$KEGG_CSV" ]; then
    KEGG_ARG="--kegg_csv $KEGG_CSV"
else
    KEGG_ARG="--kegg_dataset $KEGG_DATASET"
fi

stdbuf -oL -eL python train_latent_sft.py \
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
    --cache_dir              "$CACHE_DIR" \
    --device                 cuda
