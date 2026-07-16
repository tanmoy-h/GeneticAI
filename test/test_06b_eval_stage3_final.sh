#!/bin/bash
#SBATCH --job-name=w11_eval
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=test/logs/test_06b_eval_stage3_%j.out
#SBATCH --error=test/logs/test_06b_eval_stage3_%j.err

## Full evaluation of a Stage 3 w11 optB checkpoint.
## (Option B: GRPO with learnable theta_low via REINFORCE)
## Reports accuracy, macro precision, recall, F1 and per-class breakdown.
## Writes <prefix>_metrics.json and <prefix>_predictions.csv.
##
## Run sh_stage3_grpo_w9_optB.sh first, then set CKPT to the best checkpoint.
## The best checkpoint is the one with highest eval_correctness in the log.
##
## Usage:
##   bash test/test_06b_eval_stage3.sh [gpu_ids]
##
##   # Override checkpoint:
##   CKPT=/scratch/.../stage3_grpo_optB/checkpoint-N bash test/test_06b_eval_stage3.sh 0,1
##
##   # Quick sanity check (50 samples, val split):
##   SPLIT=val N_SAMPLES=50 bash test/test_06b_eval_stage3.sh 0
##
##   # SLURM:
##   sbatch test/test_06b_eval_stage3.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-}

CKPT=${CKPT:-}

## Auto-detect latest Stage 3 optB checkpoint if not set
if [ -z "${CKPT:-}" ]; then
    CKPT=$(find /scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_06b_stage3_grpo_optB \
        -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    [ -n "$CKPT" ] && echo "Auto-detected CKPT: $CKPT" \
        || echo "WARNING: could not auto-detect CKPT from train_06b_stage3_grpo_optB"
fi

## Split: val | test | both
SPLIT=${SPLIT:-both}

## Number of samples (-1 = all)
N_SAMPLES=${N_SAMPLES:--1}

## DNA embedding cache (skip Evo2 forward, frees ~14 GB VRAM)
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}

## Stage 2 manifold
STAGE2_DIR=${STAGE2_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_05_stage2_hiref}

## optB: load learned theta_low from checkpoint (auto-detected from CKPT dir)
## Use `-` not `:-` so THETA_LOW_PT="" disables auto-load (baseline eval).
THETA_LOW_PT=${THETA_LOW_PT-${CKPT}/theta_low.pt}

## LatentSp thresholds (match optB training defaults)
## NOTE: latent fires when entropy < theta_low (0–4 nats typical).
##   THETA_LOW=0.0  → no latent steps (baseline)
##   THETA_LOW=1.0  → latent for confident tokens (default)
THETA_LOW=${THETA_LOW:-1.0}
THETA_HIGH=${THETA_HIGH:-3.0}

## Generation
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-800}

## Output directory for metrics + CSV
OUTPUT_DIR=${OUTPUT_DIR:-test/logs}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p test/logs "$OUTPUT_DIR"
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0,1}
export PYTORCH_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

LOG=test/logs/test_06b_eval_stage3_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:        bash $0 $*"
echo "Logging to:     $LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES  NUM_GPUS: $NUM_GPUS"
echo "Checkpoint:     $CKPT"
echo "Split:          $SPLIT"
echo "N_samples:      $N_SAMPLES"
echo "Theta_low:      ${THETA_LOW}  (0.0=no latent; override if theta_low.pt not found)"
echo "Theta_high:     $THETA_HIGH"
echo "DNA cache:      ${DNA_CACHE}"
echo "Stage 2 dir:    ${STAGE2_DIR:-<none>}"
echo "Output dir:     $OUTPUT_DIR"
nvidia-smi

## Pre-flight: verify checkpoint exists and has expected files
if [ ! -d "$CKPT" ]; then
    echo "ERROR: checkpoint dir does not exist: $CKPT"
    echo "  List of available optB checkpoints:"
    ls -1d "$(dirname "$CKPT")"/checkpoint-* 2>/dev/null || echo "    (none found)"
    exit 1
fi
echo "=== Checkpoint contents ==="
ls -lh "$CKPT" 2>/dev/null | head -20
for f in pytorch_model.bin model.safetensors model.pt; do
    if [ -f "$CKPT/$f" ]; then
        echo "  LLM weights:    $f"
        break
    fi
done
[ -f "$CKPT/thinking_gate.pt" ] && echo "  Aux:            thinking_gate.pt ✓" || echo "  WARNING: thinking_gate.pt MISSING — gate will run fresh"
[ -f "$CKPT/dna_injector.pt" ] && echo "  Aux:            dna_injector.pt ✓"  || echo "  WARNING: dna_injector.pt MISSING — injector will run fresh"
[ -f "$CKPT/theta_low.pt"   ] && echo "  Aux:            theta_low.pt ✓ (optB learned threshold)" || echo "  INFO:           theta_low.pt absent — using fixed --theta_low $THETA_LOW"

## Dataset arg: CSV takes priority over HF hub
if [ -n "$KEGG_CSV" ] && [ -f "$KEGG_CSV" ]; then
    echo "Dataset:        CSV → $KEGG_CSV"
    DATASET_ARGS=(--kegg_csv "$KEGG_CSV")
else
    echo "Dataset:        HF hub → $KEGG_DATASET"
    DATASET_ARGS=(--dataset_name "$KEGG_DATASET")
fi

## DNA cache arg
if [ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ]; then
    echo "DNA cache:      enabled"
    DNA_CACHE_ARG="--dna_cache $DNA_CACHE"
else
    echo "WARNING: DNA cache not found ($DNA_CACHE) — Evo2 will run live"
    DNA_CACHE_ARG=""
fi

## optB: theta_low.pt — auto-loaded from checkpoint dir if present
THETA_LOW_PT_ARG=""
if [ -n "${THETA_LOW_PT:-}" ] && [ -f "$THETA_LOW_PT" ]; then
    echo "Theta_low.pt:   $THETA_LOW_PT (learned optB value)"
    THETA_LOW_PT_ARG="--theta_low_pt $THETA_LOW_PT"
else
    echo "Theta_low.pt:   not found — using fixed --theta_low $THETA_LOW"
fi

## Build output prefix from checkpoint name + theta_low label
## thlpt  = learned theta_low loaded from theta_low.pt
## thl0.0 = fixed theta_low (e.g. 0.0 = no latent, 1.0 = default)
CKPT_NAME=$(basename "$CKPT")
if [ -n "$THETA_LOW_PT_ARG" ]; then
    THLABEL="thlpt"
else
    THLABEL="thl${THETA_LOW}"
fi
OUTPUT_PREFIX="stage3_w11_${CKPT_NAME}_${SPLIT}_${THLABEL}"

## Write accelerate config (DDP, no DeepSpeed)
## Pick a free port so this eval can run alongside a training job on the same node.
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)" 2>/dev/null || echo 29501)
echo "Using rendezvous port: $FREE_PORT"
ACCEL_CFG=/tmp/accelerate_eval_w11_${$}.yaml
cat > "$ACCEL_CFG" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
machine_rank: 0
main_process_port: $FREE_PORT
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
use_cpu: false
EOF
echo "=== accelerate config ===" && cat "$ACCEL_CFG"
unset ACCELERATE_CONFIG_FILE
DEFAULT_ACCEL=~/.cache/huggingface/accelerate/default_config.yaml
[ -f "$DEFAULT_ACCEL" ] && cp "$DEFAULT_ACCEL" "${DEFAULT_ACCEL}.bak_$$"
cp "$ACCEL_CFG" "$DEFAULT_ACCEL" 2>/dev/null || true

EVAL_START=$SECONDS
ACCELERATE_USE_DEEPSPEED=false \
stdbuf -oL -eL accelerate launch \
    --config_file "$ACCEL_CFG" \
    eval_grpo_checkpoint_final.py \
    --checkpoint            "$CKPT" \
    "${DATASET_ARGS[@]}" \
    --split                 "$SPLIT" \
    --n_samples             "$N_SAMPLES" \
    --max_new_tokens        "$MAX_NEW_TOKENS" \
    --cache_dir             "$CACHE_DIR" \
    --text_model_name       Qwen/Qwen3-1.7B \
    --dna_model_name        evo2_7b_base \
    --dna_is_evo2           True \
    --dna_embedding_layer   blocks.28.mlp.l3 \
    --use_cross_attention   True \
    --max_length_text       6000 \
    --max_length_dna        2048 \
    --truncate_dna_per_side 1024 \
    --lora_r                16 \
    --lora_alpha            32 \
    --theta_low             "$THETA_LOW" \
    --theta_high            "$THETA_HIGH" \
    --lookahead_k           3 \
    --output_dir            "$OUTPUT_DIR" \
    --output_prefix         "$OUTPUT_PREFIX" \
    $DNA_CACHE_ARG \
    $THETA_LOW_PT_ARG \
    ${STAGE2_DIR:+--stage2_dir "$STAGE2_DIR"}

if [ -f "${DEFAULT_ACCEL}.bak_$$" ]; then
    mv "${DEFAULT_ACCEL}.bak_$$" "$DEFAULT_ACCEL"
    echo "Restored original accelerate config"
fi
rm -f "$ACCEL_CFG"

EVAL_ELAPSED=$(( SECONDS - EVAL_START ))
EVAL_MIN=$(( EVAL_ELAPSED / 60 ))
EVAL_SEC=$(( EVAL_ELAPSED % 60 ))

## Fail loudly if the eval produced no metrics — otherwise the summary block
## below reports a confusing "No such file" and hides the real crash.
METRICS_JSON="${OUTPUT_DIR}/${OUTPUT_PREFIX}_metrics.json"
if [ ! -f "$METRICS_JSON" ]; then
    echo ""
    echo "ERROR: evaluation did not write $METRICS_JSON  (after ${EVAL_MIN}m ${EVAL_SEC}s)."
    echo "       eval_grpo_checkpoint_final.py crashed before saving results. The real error is"
    echo "       in the traceback above (or in $LOG). Common causes:"
    echo "         - no GPU / NVIDIA driver on this node (run on a GPU node: sbatch or srun --gres=gpu)"
    echo "         - checkpoint missing files: ls -la \"$CKPT\""
    echo "           (expect model weights + thinking_gate.pt + dna_injector.pt + theta_low.pt)"
    echo "         - wrong --split for this dataset (SPLIT=$SPLIT), OOM, or STAGE2_DIR not found"
    exit 1
fi

echo ""
echo "=== Evaluation complete ==="
echo "  Wall time      : ${EVAL_MIN}m ${EVAL_SEC}s  (${EVAL_ELAPSED}s total)"
echo "  Metrics  → ${OUTPUT_DIR}/${OUTPUT_PREFIX}_metrics.json"
echo "  Predictions CSV → ${OUTPUT_DIR}/${OUTPUT_PREFIX}_predictions.csv"
echo ""
echo "  Latent-reasoning speedup summary (from metrics JSON):"
python3 -c "
import json, sys
try:
    d = json.load(open('${OUTPUT_DIR}/${OUTPUT_PREFIX}_metrics.json'))
    t = d.get('timing', {})
    print(f\"    Samples: {t.get('latent_samples',0)} latent  /  {t.get('normal_samples',0)} normal\")
    print(f\"    Mean time — latent: {t.get('mean_time_latent_sec','?'):.2f}s   normal: {t.get('mean_time_normal_sec','?'):.2f}s\")
    print(f\"    Sec/token  — latent: {t.get('sec_per_token_latent','?'):.4f}   normal: {t.get('sec_per_token_normal','?'):.4f}\")
    saved = t.get('time_saved_per_sample_sec', None)
    spdup = t.get('speedup_vs_normal', None)
    if saved is not None and spdup is not None:
        pct = 100*saved/max(t.get('mean_time_normal_sec',1),1e-9)
        print(f\"    Latent saves: {saved:+.2f}s/sample ({pct:+.1f}%)  speedup={spdup:.2f}x\")
except Exception as e:
    print(f'    (could not parse metrics JSON: {e})')
" 2>/dev/null || true
