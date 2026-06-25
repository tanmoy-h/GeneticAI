#!/bin/bash
#SBATCH --job-name=w9_stage1_51_anon
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=train/anon/logs/train_04_stage1_51_%j.out
#SBATCH --error=train/anon/logs/train_04_stage1_51_%j.err

## Stage 1.5 Week 9 WITH ThinkingResidualGate + DNAHiddenInjector.
##
## Runs the same LatentSp curriculum SFT as sh_stage1_50_cached_w9.sh but trains
## with the HRPO gate active at fixed MAX_GATE_FACTOR throughout.  The gate
## and injector weights are saved as thinking_gate.pt + dna_injector.pt at
## every checkpoint.
##
## Stage 3 (GRPO) can then load these weights via --gate_ckpt and start with
## gate_warmup_steps=0 — the gate is already adapted, so no log-prob discontinuity.
##
## Usage:
##   STAGE15_CKPT=<path> bash train/train_04_stage1_51.sh [gpu_id]
##   STAGE15_CKPT=<path> OUTPUT_DIR=<dir> bash train/train_04_stage1_51.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-dna-sft-week9-stage1-5-gate}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51_anon}
## ─────────────────────────────────────────────────────────────────────────────

## Start from best Stage 1.5 (no-gate) checkpoint, which already learned latent steps
STAGE15_CKPT=${STAGE15_CKPT:-}

## Auto-detect best Stage 1.5 checkpoint if not set
if [ -z "${STAGE15_CKPT:-}" ]; then
    STAGE15_CKPT=$(find /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_03b_stage1_50_cached_anon \
        -name "model.pt" 2>/dev/null | sort -V | tail -1)
    [ -n "$STAGE15_CKPT" ] && echo "Auto-detected STAGE15_CKPT: $STAGE15_CKPT" \
        || echo "WARNING: could not auto-detect STAGE15_CKPT from train_03b_stage1_50_cached_anon"
fi

if [ -z "${STAGE15_CKPT:-}" ]; then
    echo "ERROR: STAGE15_CKPT is not set."
    echo "Usage: STAGE15_CKPT=<path> bash train/train_04_stage1_51.sh [gpu_id]"
    exit 1
fi

set -e

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p train/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_ALLOC_CONF=expandable_segments:True

LOG=train/anon/logs/train_04_stage1_51_$(date +%Y%m%d_%H%M%S).log
mkdir -p train/anon/logs
exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "CUDA:          $CUDA_VISIBLE_DEVICES"
echo "Stage 1.5 ckpt: $STAGE15_CKPT"
echo "Output dir:    $OUTPUT_DIR"
echo "KEGG dataset:  ${KEGG_CSV:-$KEGG_DATASET}"
nvidia-smi

## Build dataset arg
if [ -n "$KEGG_CSV" ]; then
    KEGG_ARG="--kegg_csv $KEGG_CSV"
else
    KEGG_ARG="--kegg_dataset $KEGG_DATASET"
fi

stdbuf -oL -eL python train_latent_sft.py \
    --stage1_ckpt            "$STAGE15_CKPT" \
    $KEGG_ARG \
    --output_dir             "$OUTPUT_DIR" \
    --text_model_name        Qwen/Qwen3-1.7B \
    --dna_model_name         evo2_7b_base \
    --dna_embedding_layer    blocks.28.mlp.l3 \
    --entropy_mode           global \
    --start_latent_step      4 \
    --max_latent_steps       4 \
    --passes_per_step        ${PASSES_PER_STEP:-2} \
    --data_refresh_ratio     0.1 \
    --batch_size             1 \
    --grad_accum             8 \
    --learning_rate          2e-5 \
    --max_length_text        6000 \
    --max_length_dna         2048 \
    --truncate_dna_per_side  1024 \
    --entropy_batch_size     1 \
    --log_every              20 \
    --wandb_project          "$WANDB_PROJECT" \
    --wandb_entity           "$WANDB_ENTITY" \
    --cache_dir              "$CACHE_DIR" \
    --device                 cuda \
    --use_gate

echo "=== Stage 1.5-with-gate done. Gate weights in $OUTPUT_DIR/best/thinking_gate.pt ==="
echo "=== Run Stage 3 with: GATE_CKPT_DIR=$OUTPUT_DIR/best bash train/train_06_stage3_grpo.sh ==="
