#!/bin/bash
#SBATCH --job-name=w9_eval_anon
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=week9tests/logs/eval_stage3_w9_%j.out
#SBATCH --error=week9tests/logs/eval_stage3_w9_%j.err

## Full evaluation of a Stage 3 w9 (or optB) checkpoint.
## Reports accuracy, macro precision, recall, F1 and per-class breakdown.
## Writes <prefix>_metrics.json and <prefix>_predictions.csv.
##
## Default checkpoint: checkpoint-3088
##   (checkpoint-1158 rotated out by save_total_limit=4; checkpoint-3088 is
##   the best surviving checkpoint from the w9 run.)
##
## Works for:
##   w9  — standard fixed-theta_low checkpoint
##   optB — learnable theta_low checkpoint (pass THETA_LOW_PT to restore learned value)
##
## Usage:
##   bash week9tests/sh_eval_stage3_w9.sh [gpu_ids]
##
##   # Override checkpoint:
##   CKPT=/scratch/.../stage3_grpo_w9/checkpoint-386 bash week9tests/sh_eval_stage3_w9.sh 0,1
##
##   # Evaluate optB with learned theta_low:
##   CKPT=/scratch/.../stage3_grpo_optB/checkpoint-N \
##   THETA_LOW_PT=/scratch/.../stage3_grpo_optB/checkpoint-N/theta_low.pt \
##   bash week9tests/sh_eval_stage3_w9.sh 0,1
##
##   # Quick sanity check (50 samples, val split):
##   SPLIT=val N_SAMPLES=50 bash week9tests/sh_eval_stage3_w9.sh 0
##
##   # SLURM:
##   sbatch week9tests/sh_eval_stage3_w9.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}

## Best surviving checkpoint (checkpoint-1158 rotated out by save_total_limit=4)
CKPT=${CKPT:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/week9tests/stage3_grpo_w9/checkpoint-3088}

## Split: val | test | both
SPLIT=${SPLIT:-both}

## Number of samples (-1 = all)
N_SAMPLES=${N_SAMPLES:--1}

## DNA embedding cache (skip Evo2 forward, frees ~14 GB VRAM)
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}

## Stage 2 manifold (training used stage2_output_w9)
STAGE2_DIR=${STAGE2_DIR:-stage2_output_w9}

## optB: path to theta_low.pt inside the checkpoint dir
## For standard w9 this file won't exist — falls back to fixed --theta_low
## Use `-` not `:-` so THETA_LOW_PT="" disables auto-load (baseline eval).
THETA_LOW_PT=${THETA_LOW_PT-${CKPT}/theta_low.pt}

## LatentSp thresholds (latent fires when entropy < theta_low; 0.0 = no latent baseline)
THETA_LOW=${THETA_LOW:-1.0}
THETA_HIGH=${THETA_HIGH:-3.0}

## Generation
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-800}

## Output directory for metrics + CSV
OUTPUT_DIR=${OUTPUT_DIR:-week9tests/logs}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p week9tests/logs "$OUTPUT_DIR"
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0,1}
export PYTORCH_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

LOG=week9tests/logs/eval_stage3_w9_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:        bash $0 $*"
echo "Logging to:     $LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES  NUM_GPUS: $NUM_GPUS"
echo "Checkpoint:     $CKPT"
echo "Split:          $SPLIT"
echo "N_samples:      $N_SAMPLES"
echo "Theta_low:      ${THETA_LOW}  (override if theta_low.pt not found)"
echo "Theta_high:     $THETA_HIGH"
echo "DNA cache:      ${DNA_CACHE}"
echo "Stage 2 dir:    ${STAGE2_DIR:-<none>}"
echo "Output dir:     $OUTPUT_DIR"
nvidia-smi

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

## optB: theta_low.pt arg
THETA_LOW_PT_ARG=""
if [ -n "${THETA_LOW_PT:-}" ] && [ -f "$THETA_LOW_PT" ]; then
    echo "Theta_low.pt:   $THETA_LOW_PT (optB learned value)"
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
OUTPUT_PREFIX="stage3_w9_${CKPT_NAME}_${SPLIT}_${THLABEL}"

## Write accelerate config (DDP, no DeepSpeed)
## Pick a free port so this eval can run alongside a training job on the same node.
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)" 2>/dev/null || echo 29501)
echo "Using rendezvous port: $FREE_PORT"
ACCEL_CFG=/tmp/accelerate_eval_w9_${$}.yaml
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
    eval_grpo_checkpoint.py \
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
