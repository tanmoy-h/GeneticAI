#!/bin/bash
#SBATCH --job-name=w9_test_latentSp
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=week9tests/logs/test_latentSp_%j.out
#SBATCH --error=week9tests/logs/test_latentSp_%j.err

## LatentSp smoke test — runs 50 steps with warmup/ramp disabled so both
## latent steps (low-H) and DNA injection (high-H) fire from step 1.
##
## What to check in the log:
##   train/gate_latent_steps  > 0      → latent step path fires
##   train/gate_steps         > 0      → gate path fires (always should)
##   [w9] dna_injected=True            → DNA injection triggered
##   [w9] latent step                  → latent step triggered
##   No NaN loss, no crash             → mechanisms are stable
##
## Usage:
##   STAGE1_CKPT=<path> bash week9tests/sh_test_latentSp.sh [gpu_ids]

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-dna-grpo-week9-test}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/week9tests/test_latentSp}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
## ─────────────────────────────────────────────────────────────────────────────

## Base model from Stage 1.5.1 (model.pt); gate + injector from Stage 3 checkpoint-386
## (checkpoint-386 is HF format — no model.pt — so we reuse the SFT base weights)
STAGE3_CKPT_DIR=${STAGE3_CKPT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/week9tests/stage3_grpo_w9/checkpoint-386}
STAGE1_CKPT=${STAGE1_CKPT:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/week9tests/stage1_51/s04_pass01/model.pt}
GATE_CKPT=${GATE_CKPT:-${STAGE3_CKPT_DIR}/thinking_gate.pt}
INJECTOR_CKPT=${INJECTOR_CKPT:-${STAGE3_CKPT_DIR}/dna_injector.pt}

if [ -z "${STAGE1_CKPT:-}" ]; then
    echo "ERROR: STAGE1_CKPT is not set."
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p week9tests/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0,1}
export WANDB_PROJECT
export WANDB_ENTITY
export PYTORCH_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

LOG=week9tests/logs/test_latentSp_$(date +%Y%m%d_%H%M%S).log
mkdir -p week9tests/logs
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to:  $LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES  NUM_GPUS: $NUM_GPUS"
echo "Stage1 ckpt: $STAGE1_CKPT"
echo "Output dir:  $OUTPUT_DIR"
nvidia-smi

args=(
    --sft_checkpoint         "$STAGE1_CKPT"
    --dataset_name           "$KEGG_DATASET"
    --output_dir             "$OUTPUT_DIR"
    --cache_dir              "$CACHE_DIR"

    ## Model
    --text_model_name        Qwen/Qwen3-1.7B
    --dna_model_name         evo2_7b_base
    --dna_is_evo2            True
    --dna_embedding_layer    blocks.28.mlp.l3
    --use_cross_attention    True
    --dna_model_finetune     False
    --max_length_text        6000
    --max_length_dna         2048
    --truncate_dna_per_side  1024

    ## HRPO gate — no warmup so it's active from step 1
    --use_hrpo_gate          True
    --use_dna_gate           False
    --gate_reg_weight        0.001
    --gate_warmup_steps      0
    --gate_ramp_steps        0
    --max_latent_steps       1
    --min_latent_steps       0

    ## LatentSp — no warmup/ramp so both paths fire immediately
    --latentSp_theta_low        1.0
    --latentSp_theta_high       3.0
    --latentSp_max_consec       3
    --latent_lookahead_k        3
    --latentSp_warmup_steps     0
    --latentSp_ramp_steps       0

    ## GRPO — minimal run
    --num_generations        8
    --max_completion_length  800
    --temperature            0.7
    --top_p                  0.95
    --top_k                  50
    --beta                   0.02
    --epsilon                0.1
    --max_grad_norm          0.5
    --reward_funcs           xmlcount soft_format correctness completion_quality reasoning_quality ot_distance latent_usage
    --manifold_weight        0.01
    --max_clip_loss_weight   0.0

    ## Training — stop after 50 steps
    --per_device_train_batch_size  1
    --gradient_accumulation_steps 4
    --num_train_epochs       1
    --max_steps              200
    --learning_rate          5e-6
    --lora_r                 16
    --lora_alpha             32
    --eval_strategy          no
    --save_steps             200
    --save_total_limit       1
    --logging_steps          5
    --report_to              wandb
    --bf16                   True
    --use_vllm               False
)

## Gate checkpoint
if [ -n "${GATE_CKPT:-}" ] && [ -f "$GATE_CKPT" ]; then
    echo "Gate ckpt: $GATE_CKPT"
    args+=(--gate_ckpt "$GATE_CKPT")
    [ -f "$INJECTOR_CKPT" ] && args+=(--injector_ckpt "$INJECTOR_CKPT")
else
    echo "WARNING: GATE_CKPT not found ($GATE_CKPT) — gate starts fresh"
fi

## DNA embedding cache
if [ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ]; then
    echo "DNA cache enabled: $DNA_CACHE"
    args+=(--dna_cache "$DNA_CACHE")
fi

ACCEL_CFG=/tmp/accelerate_ddp_test_latentSp_${$}.yaml
cat > "$ACCEL_CFG" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
machine_rank: 0
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

ACCELERATE_USE_DEEPSPEED=false \
stdbuf -oL -eL accelerate launch \
    --config_file "$ACCEL_CFG" \
    adaptive_thinking_residual_w9.py "${args[@]}"

if [ -f "${DEFAULT_ACCEL}.bak_$$" ]; then
    mv "${DEFAULT_ACCEL}.bak_$$" "$DEFAULT_ACCEL"
    echo "Restored original accelerate config"
fi

echo ""
echo "=== LatentSp test complete. Check for: ==="
echo "  train/gate_latent_steps > 0    (latent step path fired)"
echo "  train/gate_steps > 0           (gate path fired)"
echo "  [w9] dna_injected=True         (DNA injection fired)"
echo "  No NaN loss / no crash         (mechanisms stable)"
