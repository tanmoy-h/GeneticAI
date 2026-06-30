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
_S1_LOCAL_DIR=/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_02_stage1_sft_anon

## Resolve STAGE1_CKPT: hf:// → download (fallback to local); empty → auto-detect
if [[ "${STAGE1_CKPT:-}" == hf://* ]]; then
    _HF_REPO=$(echo "$STAGE1_CKPT" | sed 's|hf://||' | cut -d'/' -f1-2)
    _HF_FILE=$(echo "$STAGE1_CKPT" | sed "s|hf://${_HF_REPO}/||")
    echo "Resolving HF checkpoint: $STAGE1_CKPT"
    _DL_PATH=$(python3 -c "
from huggingface_hub import hf_hub_download
import sys
try:
    print(hf_hub_download('$_HF_REPO', '$_HF_FILE'))
except Exception as e:
    print(str(e), file=sys.stderr); sys.exit(1)
" 2>/dev/null)
    _DL_EXIT=$?
    if [ $_DL_EXIT -eq 0 ] && [ -n "$_DL_PATH" ] && [ -f "$_DL_PATH" ]; then
        echo "Stage 1 ckpt: $_DL_PATH"
        STAGE1_CKPT="$_DL_PATH"
    else
        echo "WARNING: HF download failed — falling back to local auto-detect"
        STAGE1_CKPT=$(find "$_S1_LOCAL_DIR" \
            -name "*.ckpt" ! -name "last.ckpt" 2>/dev/null \
            | awk -F'val_loss_epoch=' 'NF>1{print $2, $0}' | sort -n | head -1 | cut -d' ' -f2-)
        [ -n "$STAGE1_CKPT" ] && echo "Fallback STAGE1_CKPT: $STAGE1_CKPT" \
            || echo "WARNING: could not auto-detect from $_S1_LOCAL_DIR"
    fi
elif [ -z "${STAGE1_CKPT:-}" ]; then
    STAGE1_CKPT=$(find "$_S1_LOCAL_DIR" \
        -name "*.ckpt" ! -name "last.ckpt" 2>/dev/null \
        | awk -F'val_loss_epoch=' 'NF>1{print $2, $0}' | sort -n | head -1 | cut -d' ' -f2-)
    [ -n "$STAGE1_CKPT" ] && echo "Auto-detected STAGE1_CKPT: $STAGE1_CKPT" \
        || echo "WARNING: could not auto-detect STAGE1_CKPT from train_02_stage1_sft_anon"
fi

CKPT_DIR=${CKPT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_03b_stage1_50_cached_anon}

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

## Results JSON path — auto-named by timestamp if not overridden
RESULTS_JSON=${RESULTS_JSON:-test/anon/logs/test_04_eval_stage1_5_results_$(date +%Y%m%d_%H%M%S).json}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p test/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"

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
