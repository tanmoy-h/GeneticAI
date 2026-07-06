#!/bin/bash
#SBATCH --job-name=w9_eval_stage1_5_anon
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=test/anon/logs/test_04_eval_stage1_5_%j.out
#SBATCH --error=test/anon/logs/test_04_eval_stage1_5_%j.err

## Evaluate Stage 1.5 checkpoints in parallel across GPUs.
##
## Each GPU gets a subset of checkpoints to evaluate concurrently.
## Use --s to restrict which curriculum steps to test.
##
## Usage (direct):
##   bash test/test_04_eval_stage1_5.sh
##
##   # Only evaluate s=1 and s=2 on 2 GPUs:
##   S=1,2 GPUS=0,1 bash test/test_04_eval_stage1_5.sh
##
##   # All steps, 4 GPUs, 100 samples:
##   GPUS=0,1,2,3 N_SAMPLES=100 bash test/test_04_eval_stage1_5.sh
##
##   # Single GPU, s=3 only:
##   S=3 GPUS=0 bash test/test_04_eval_stage1_5.sh
##
## Usage (SLURM):
##   sbatch test/test_04_eval_stage1_5.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}

STAGE1_CKPT=${STAGE1_CKPT:-}

if [ -z "${CKPT_DIR:-}" ]; then
    _CKPT_3B=/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_03b_stage1_50_cached_anon
    _CKPT_3=/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_03_stage1_50_anon
    if [ -f "$_CKPT_3B/stage1_ckpt.txt" ]; then
        CKPT_DIR="$_CKPT_3B"
    elif [ -f "$_CKPT_3/stage1_ckpt.txt" ]; then
        CKPT_DIR="$_CKPT_3"
    else
        CKPT_DIR="$_CKPT_3B"
    fi
fi

DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}

## Comma-separated GPU IDs (e.g. "0,1,2,3"). Auto-detects all available GPUs.
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')}
GPUS=${GPUS:-0}

## Comma-separated curriculum steps to evaluate (e.g. "1,2"). Empty = all steps.
S=${S:-}

N_SAMPLES=${N_SAMPLES:-100}
SEED=${SEED:-42}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-800}
VERBOSE=${VERBOSE:-0}  # set to 1 to print per-sample pred/gt output

## Results JSON / raw-generation CSV paths — auto-named by timestamp if not overridden
_TS=$(date +%Y%m%d_%H%M%S)
RESULTS_JSON=${RESULTS_JSON:-test/anon/logs/test_04_eval_stage1_5_results_${_TS}.json}
RAW_CSV=${RAW_CSV:-test/anon/logs/test_04_eval_stage1_5_raw_${_TS}.csv}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p test/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"

## Use Stage 1 checkpoint recorded by training (written by train_03b_stage1_50_cached.sh)
if [ -z "${STAGE1_CKPT:-}" ]; then
    _TRAIN_CKPT_FILE="$CKPT_DIR/stage1_ckpt.txt"
    if [ -f "$_TRAIN_CKPT_FILE" ]; then
        STAGE1_CKPT=$(cat "$_TRAIN_CKPT_FILE")
        echo "STAGE1_CKPT (from training): $STAGE1_CKPT"
    else
        echo "ERROR: STAGE1_CKPT not set and $CKPT_DIR/stage1_ckpt.txt not found."
        echo "       Run training first, or set STAGE1_CKPT=<path> explicitly."
        exit 1
    fi
fi

## Expose all SLURM-allocated GPUs
## (override with CUDA_VISIBLE_DEVICES env var if needed)

LOG=test/anon/logs/test_04_eval_stage1_5_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "Stage1 ckpt:   $STAGE1_CKPT"
echo "Ckpt dir:      $CKPT_DIR"
echo "DNA cache:     ${DNA_CACHE:-<none>}"
echo "GPUs:          $GPUS"
echo "Steps (--s):   ${S:-<all>}"
echo "N samples:     $N_SAMPLES"
echo "Seed:          $SEED"
echo "Results JSON:  $RESULTS_JSON"
echo "Raw CSV:       $RAW_CSV"
nvidia-smi

## Build dataset arg
if [ -n "$KEGG_CSV" ]; then
    KEGG_ARG="--kegg_csv $KEGG_CSV"
else
    KEGG_ARG="--kegg_dataset $KEGG_DATASET"
fi

## Build DNA cache arg
if [ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ]; then
    DNA_CACHE_ARG="--dna_cache $DNA_CACHE"
else
    echo "WARNING: DNA cache not found: $DNA_CACHE"
    echo "         Evo2 will run live — requires extra VRAM and is much slower."
    DNA_CACHE_ARG=""
fi

## Build --s arg
S_ARG=""
if [ -n "$S" ]; then
    S_ARG="--s $S"
fi

stdbuf -oL -eL python eval_stage1_5_checkpoints.py \
    --stage1_ckpt            "$STAGE1_CKPT" \
    --ckpt_dir               "$CKPT_DIR" \
    $KEGG_ARG \
    $DNA_CACHE_ARG \
    $S_ARG \
    --gpus                   "$GPUS" \
    --n_samples              $N_SAMPLES \
    --max_new_tokens         $MAX_NEW_TOKENS \
    --seed                   $SEED \
    --cache_dir              "$CACHE_DIR" \
    --results_json           "$RESULTS_JSON" \
    --raw_csv                "$RAW_CSV" \
