#!/bin/bash
#SBATCH --job-name=rft_sample
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=train/logs/rft_sample_%j.out
#SBATCH --error=train/logs/rft_sample_%j.err

## Step 1a of the self-adaptive RFT pipeline: sample N completions per TRAIN prompt
## from a healthy GRPO checkpoint (CONTROLLER mode — latents inserted by the entropy
## controller), so the completions carry latent demonstrations. Multi-GPU sharded via
## accelerate. Writes <OUTPUT_DIR>/<OUTPUT_PREFIX>_traces.jsonl for the filter.
##
## CRITICAL: uses --split train --n_samples -1 (unshuffled full train) so each trace's
## `index` matches the dataset order that train_latent_sft.py's load_rft_rows re-joins on.
##
## Usage:
##   CKPT=<...>/train_06b_stage3_grpo_optB_planA/checkpoint-772 \
##     bash train/rft/sample_traces.sh [gpu_ids]
##   (theta_low.pt / thinking_gate.pt / dna_injector.pt all auto-load from CKPT's dir;
##    only override THETA_LOW_PT to force a different learned threshold.)
##
## RARE-DISEASE SUPPLEMENT (Stage-1.51 SFT source): GRPO compresses away rare-class
## reasoning, so it produces no correct latent trace for rare diseases. Sample the SFT
## checkpoint too — it keeps rare classes right — and merge in the filter step:
##   CKPT=<...>/train_04_stage1_51/s04_pass02 OUTPUT_PREFIX=rft_sample_sft \
##     THETA_LOW=0.8 bash train/rft/sample_traces.sh [gpu_ids]
##   theta_low.pt won't exist there -> falls back to THETA_LOW for the controller;
##   thinking_gate.pt / dna_injector.pt still auto-load from the SFT dir.
##   Then: TRACES="<grpo>_traces.jsonl <sft>_traces.jsonl" bash train/rft/filter_traces.sh
##
## Cost: 1159 prompts x SAMPLE_PASSES generations. Use 2+ GPUs; drop SAMPLE_PASSES to 4
## to halve time if needed.

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-}
OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/train/rft/samples}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-rft_sample}
SAMPLE_PASSES=${SAMPLE_PASSES:-8}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-800}

## Healthy GRPO checkpoint where latents fire (controller mode). REQUIRED.
CKPT=${CKPT:-}
THETA_LOW_PT=${THETA_LOW_PT:-}          # optional; auto-loads from CKPT/theta_low.pt if unset
THETA_LOW=${THETA_LOW:-0.5}             # used if THETA_LOW_PT not set
THETA_HIGH=${THETA_HIGH:-3.0}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
STAGE2_DIR=${STAGE2_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_05_stage2_hiref}
## ─────────────────────────────────────────────────────────────────────────────

if [ -z "$CKPT" ]; then
    echo "ERROR: set CKPT=<healthy GRPO checkpoint dir where latents fire, e.g. checkpoint-772>"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p "$OUTPUT_DIR"
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0,1}
export PYTORCH_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

## Run-log co-located with the sampled traces (OUTPUT_DIR); in-process tee (no re-exec,
## which would re-run conda activate in a fresh non-interactive bash and hang).
LOG="${LOG:-$OUTPUT_DIR/${OUTPUT_PREFIX}_$(date +%Y%m%d_%H%M%S)_run.log}"
exec > >(tee "$LOG") 2>&1
echo "Command:      bash $0 $*"
echo "Checkpoint:   $CKPT"
echo "Passes:       $SAMPLE_PASSES  temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K"
echo "Output:       $OUTPUT_DIR/${OUTPUT_PREFIX}_traces.jsonl"
echo "GPUs:         $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
nvidia-smi

## Dataset arg
if [ -n "$KEGG_CSV" ]; then
    DATASET_ARGS=(--kegg_csv "$KEGG_CSV")
else
    DATASET_ARGS=(--dataset_name "$KEGG_DATASET")
fi

## Optional args
EXTRA=()
[ -n "$THETA_LOW_PT" ] && [ -f "$THETA_LOW_PT" ] && EXTRA+=(--theta_low_pt "$THETA_LOW_PT")
[ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ] && EXTRA+=(--dna_cache "$DNA_CACHE")
[ -n "$STAGE2_DIR" ] && EXTRA+=(--stage2_dir "$STAGE2_DIR")

## Accelerate DDP config (free port so it can run alongside training)
FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)" 2>/dev/null || echo 29533)
ACCEL_CFG=/tmp/accelerate_rft_sample_${$}.yaml
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
unset ACCELERATE_CONFIG_FILE

ACCELERATE_USE_DEEPSPEED=false \
stdbuf -oL -eL accelerate launch --config_file "$ACCEL_CFG" \
    eval_grpo_checkpoint_final.py \
    --checkpoint            "$CKPT" \
    "${DATASET_ARGS[@]}" \
    --split                 train \
    --n_samples             -1 \
    --sample_passes         "$SAMPLE_PASSES" \
    --temperature           "$TEMPERATURE" \
    --top_p                 "$TOP_P" \
    --top_k                 "$TOP_K" \
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
    "${EXTRA[@]}"

echo ""
echo "=== Sampling done. Traces: $OUTPUT_DIR/${OUTPUT_PREFIX}_traces.jsonl ==="
echo "=== Next: bash train/rft/filter_traces.sh (set TRACES to that file) ==="
