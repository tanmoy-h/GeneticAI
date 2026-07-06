#!/bin/bash
#SBATCH --job-name=w9_eval_stage1_51_anon
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=test/anon/logs/test_04b_eval_stage1_51_%j.out
#SBATCH --error=test/anon/logs/test_04b_eval_stage1_51_%j.err

## Evaluate Stage 1.51 (gated) checkpoints in parallel across GPUs.
##
## Same as test_04_eval_stage1_5.sh but each checkpoint carries an HRPO gate
## (thinking_gate.pt + dna_injector.pt).  Ranks by answer correctness and writes
## a results JSON whose "best" entry Stage 3 / train_06b_stage3_grpo_optB.sh can
## read to pick the peak-accuracy checkpoint instead of lowest-val-loss best/.
##
## Usage (direct):
##   bash test/anon/test_04b_eval_stage1_51.sh
##   GPUS=0,1 N_SAMPLES=100 bash test/anon/test_04b_eval_stage1_51.sh
##   S=4 GPUS=0 bash test/anon/test_04b_eval_stage1_51.sh
##
## Usage (SLURM):
##   sbatch test/anon/test_04b_eval_stage1_51.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}

STAGE1_CKPT=${STAGE1_CKPT:-}

## Stage 1.51 checkpoint dir (written by train_04_stage1_51.sh)
CKPT_DIR=${CKPT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51_anon}

DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}

## Comma-separated GPU IDs (e.g. "0,1,2,3"). Auto-detects all available GPUs.
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')}
GPUS=${GPUS:-0}

## Comma-separated curriculum steps to evaluate (e.g. "4"). Empty = all steps.
S=${S:-}

N_SAMPLES=${N_SAMPLES:-100}
SEED=${SEED:-42}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-800}

## Results JSON / raw-generation CSV paths — auto-named by timestamp if not overridden
_TS=$(date +%Y%m%d_%H%M%S)
RESULTS_JSON=${RESULTS_JSON:-test/anon/logs/test_04b_eval_stage1_51_results_${_TS}.json}
RAW_CSV=${RAW_CSV:-test/anon/logs/test_04b_eval_stage1_51_raw_${_TS}.csv}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p test/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"

## Stage 1 checkpoint: recorded by Stage 1.51 training (stage1_ckpt.txt points at
## the Stage 1.5 best it started from — used only to build the model architecture).
if [ -z "${STAGE1_CKPT:-}" ]; then
    for _F in "$CKPT_DIR/stage1_ckpt.txt" \
              /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_03b_stage1_50_cached_anon/stage1_ckpt.txt; do
        if [ -f "$_F" ]; then
            STAGE1_CKPT=$(cat "$_F")
            echo "STAGE1_CKPT (from $_F): $STAGE1_CKPT"
            break
        fi
    done
fi
if [ -z "${STAGE1_CKPT:-}" ]; then
    echo "ERROR: STAGE1_CKPT not set and no stage1_ckpt.txt found."
    echo "       Set STAGE1_CKPT=<path/to/stage1-sft....ckpt> explicitly."
    exit 1
fi

LOG=test/anon/logs/test_04b_eval_stage1_51_$(date +%Y%m%d_%H%M%S).log
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

stdbuf -oL -eL python eval_stage1_51_checkpoints.py \
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
    --raw_csv                "$RAW_CSV"
