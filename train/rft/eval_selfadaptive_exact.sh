#!/bin/bash
#SBATCH --job-name=rft_eval_sa_exact
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=train/logs/rft_eval_sa_exact_%j.out
#SBATCH --error=train/logs/rft_eval_sa_exact_%j.err

## EXACT-reproduction eval (eval_selfadaptive_exact.py -- wraps train_latent_sft.py's
## own evaluate_accuracy()): use this when you need a checkpoint's reported number to
## match EXACTLY, e.g. reproducing train_07_selfadaptive's 0.9759 acc / 9.75s with full
## reasoning traces. Single-GPU (matches train_07_selfadaptive.sh's own profile) --
## no accelerate/distributed launch, no LoRA-wrap ambiguity (this script never wraps
## in PEFT at all, so there's no --no_lora flag to remember).
##
## Contrast with train/rft/eval_selfadaptive.sh, which uses a DIFFERENT generation
## loop (generate_with_hrpo_gate, entropy-conditional DNA injection) -- convenient for
## multi-GPU throughput and controller-mode baselines, but not guaranteed to reproduce
## a specific checkpoint's already-reported number exactly. Use THIS script when you
## need the exact number; use eval_selfadaptive.sh for everything else.
##
## Usage:
##   CKPT=<...>/train_07_selfadaptive/best_acc SPLIT=both \
##     bash train/rft/eval_selfadaptive_exact.sh [gpu_id]
##   BAN_LATENT=1 CKPT=<...> bash train/rft/eval_selfadaptive_exact.sh   # latent-free comparison number

CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-}
OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/test/log}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-800}
BAN_LATENT=${BAN_LATENT:-0}          # 1 = latent-free comparison number

CKPT=${CKPT:-}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}

if [ -z "$CKPT" ]; then
    echo "ERROR: set CKPT=<checkpoint DIRECTORY to evaluate>"
    exit 1
fi
if [ ! -d "$CKPT" ]; then
    echo "ERROR: CKPT must be a directory (got a file or missing path): $CKPT"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p "$OUTPUT_DIR"
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_ALLOC_CONF=expandable_segments:True

_MODE=$([ "$BAN_LATENT" = "1" ] && echo "latentfree" || echo "selfadaptive")
LOG="${LOG:-$OUTPUT_DIR/rft_exact_${_MODE}_$(basename "$CKPT")_$(date +%Y%m%d_%H%M%S)_run.log}"
exec > >(tee "$LOG") 2>&1
echo "Command:    bash $0 $*"
echo "Checkpoint: $CKPT"
echo "Mode:       $_MODE   ban_latent=$BAN_LATENT"
nvidia-smi

if [ -n "$KEGG_CSV" ]; then
    DATASET_ARGS=(--kegg_csv "$KEGG_CSV")
else
    DATASET_ARGS=(--dataset_name "$KEGG_DATASET")
fi

EXTRA=()
[ "$BAN_LATENT" = "1" ] && EXTRA+=(--ban_latent)
[ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ] && EXTRA+=(--dna_cache "$DNA_CACHE")

stdbuf -oL -eL python eval_selfadaptive_exact.py \
    --checkpoint            "$CKPT" \
    "${DATASET_ARGS[@]}" \
    --max_new_tokens        "$MAX_NEW_TOKENS" \
    --cache_dir             "$CACHE_DIR" \
    --output_dir            "$OUTPUT_DIR" \
    --output_prefix         "rft_exact_${_MODE}_$(basename "$CKPT")" \
    "${EXTRA[@]}"

echo ""
echo "=== Exact eval ($_MODE) done. See $OUTPUT_DIR for _metrics.json / _predictions.csv / _traces.jsonl ==="
