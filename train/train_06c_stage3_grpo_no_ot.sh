#!/bin/bash
#SBATCH --job-name=grpo_no_ot
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=train/logs/stage3_grpo_no_ot_%j.out
#SBATCH --error=train/logs/stage3_grpo_no_ot_%j.err

## Ablation: GenoMorph-B without OT reward (HiRef OT disabled).
##
## Identical to train_08_stage3_grpo_optB.sh (learned theta_low via REINFORCE)
## except:
##   - ot_distance removed from --reward_funcs
##   - --manifold_weight 0.0  (Stage 2 manifold not used)
##
## Purpose: isolates the contribution of HiRef OT alignment reward.
##
## Usage:
##   STAGE1_CKPT=<path/to/stage1_51/model.pt> bash train/train_09_stage3_grpo_no_ot.sh [gpu_ids]
##   RESUME=1 STAGE1_CKPT=<path> bash train/train_09_stage3_grpo_no_ot.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-dna-grpo-no-ot}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/stage3_grpo_no_ot}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
## ─────────────────────────────────────────────────────────────────────────────

STAGE1_CKPT=${STAGE1_CKPT:-}
GATE_CKPT_DIR=${GATE_CKPT_DIR:-}

## Use Stage 1.51 checkpoint recorded by training (written by train_04_stage1_51.sh)
if [ -z "${STAGE1_CKPT:-}" ]; then
    _TRAIN_CKPT_FILE=/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51/stage151_ckpt.txt
    if [ -f "$_TRAIN_CKPT_FILE" ]; then
        STAGE1_CKPT=$(cat "$_TRAIN_CKPT_FILE")
        echo "STAGE1_CKPT (from training): $STAGE1_CKPT"
    else
        echo "ERROR: STAGE1_CKPT not set and $_TRAIN_CKPT_FILE not found."
        echo "       Run train_04_stage1_51.sh first, or set STAGE1_CKPT=<path> explicitly."
        exit 1
    fi
fi
if [ -z "${GATE_CKPT_DIR:-}" ] && [ -n "${STAGE1_CKPT:-}" ]; then
    GATE_CKPT_DIR=$(dirname "$STAGE1_CKPT")
fi
GATE_CKPT=${GATE_CKPT:-${GATE_CKPT_DIR:+${GATE_CKPT_DIR}/thinking_gate.pt}}
INJECTOR_CKPT=${INJECTOR_CKPT:-${GATE_CKPT_DIR:+${GATE_CKPT_DIR}/dna_injector.pt}}

if [ -z "${STAGE1_CKPT:-}" ]; then
    echo "ERROR: STAGE1_CKPT is not set."
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p train/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0,1}
export WANDB_PROJECT
export WANDB_ENTITY
export PYTORCH_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

PER_DEVICE_BATCH=1
GRAD_ACCUM=4
NUM_GENERATIONS=8
TRAIN_SIZE=$(python3 -c "
from datasets import load_dataset
ds = load_dataset('$KEGG_DATASET', 'default', cache_dir='$CACHE_DIR')
print(len(ds['train']))
" 2>/dev/null)
if [ -z "$TRAIN_SIZE" ] || [ "$TRAIN_SIZE" -le 0 ] 2>/dev/null; then
    echo "WARNING: Could not read TRAIN_SIZE — falling back to SAVE_STEPS=500"
    SAVE_STEPS=500
    STEPS_PER_EPOCH=1000
else
    STEPS_PER_EPOCH=$(( (TRAIN_SIZE * NUM_GENERATIONS + PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM - 1) / (PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM) ))
    SAVE_STEPS=$(( STEPS_PER_EPOCH / 3 ))
    [ "$SAVE_STEPS" -le 0 ] && SAVE_STEPS=400
fi
TOTAL_STEPS=$(( STEPS_PER_EPOCH * 3 ))

LOG=train/logs/stage3_grpo_no_ot_$(date +%Y%m%d_%H%M%S).log
mkdir -p train/logs
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to:  $LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES  NUM_GPUS: $NUM_GPUS"
echo "Train size: $TRAIN_SIZE  Steps/epoch: $STEPS_PER_EPOCH  Total: $TOTAL_STEPS"
echo "Save every: $SAVE_STEPS steps"
echo "Stage1 ckpt: $STAGE1_CKPT"
echo "Output dir:  $OUTPUT_DIR"
echo "OT reward:   DISABLED (ablation)"
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

    ## HRPO gate
    --use_hrpo_gate          True
    --use_dna_gate           False
    --gate_reg_weight        0.001
    --gate_warmup_steps      100
    --gate_ramp_steps        200
    --max_latent_steps       1
    --min_latent_steps       0

    ## LatentSp — learnable theta_low (Option B), same as GenoMorph-B
    --latentSp_theta_low        1.0
    --latentSp_theta_high       3.0
    --latentSp_max_consec       3
    --latent_lookahead_k        3
    --latentSp_warmup_steps     400
    --latentSp_ramp_steps       400
    --theta_low_lr           1e-4
    --theta_low_weight       0.1
    --theta_low_alpha        5.0

    ## GRPO — OT reward removed for this ablation
    --num_generations        8
    --max_completion_length  800
    --temperature            0.7
    --top_p                  0.95
    --top_k                  50
    --beta                   0.05
    --epsilon                0.1
    --max_grad_norm          0.1
    --reward_funcs           xmlcount soft_format correctness completion_quality reasoning_quality latent_usage
    --manifold_weight        0.0

    ## Training
    --per_device_train_batch_size  1
    --gradient_accumulation_steps 4
    --num_train_epochs       3
    --max_steps              -1
    --learning_rate          2e-6
    --lora_r                 16
    --lora_alpha             32
    --eval_strategy          steps
    --eval_steps             $SAVE_STEPS
    --save_steps             $SAVE_STEPS
    --save_total_limit       4
    --load_best_model_at_end True
    --metric_for_best_model  correctness
    --greater_is_better      True
    --per_device_eval_batch_size 1
    --max_eval_samples       50
    --logging_steps          10
    --report_to              wandb
    --bf16                   True
    --use_vllm               False
)

## Gate/injector checkpoint
if [ -n "${GATE_CKPT:-}" ] && [ -f "$GATE_CKPT" ]; then
    echo "Gate ckpt: $GATE_CKPT"
    args+=(--gate_ckpt "$GATE_CKPT")
    [ -f "${INJECTOR_CKPT:-}" ] && args+=(--injector_ckpt "$INJECTOR_CKPT")
else
    echo "INFO: No GATE_CKPT set — gate/injector start fresh"
fi

## DNA embedding cache
if [ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ]; then
    echo "DNA cache enabled: $DNA_CACHE"
    args+=(--dna_cache "$DNA_CACHE")
elif [ -n "$DNA_CACHE" ]; then
    echo "WARNING: DNA_CACHE not found: $DNA_CACHE — Evo2 will run live"
fi

## Resume from checkpoint
if [ "${RESUME:-0}" = "1" ]; then
    if [ -n "${RESUME_FROM_CKPT:-}" ]; then
        if [ ! -d "$RESUME_FROM_CKPT" ]; then
            echo "ERROR: RESUME_FROM_CKPT not found: $RESUME_FROM_CKPT"
            exit 1
        fi
        echo "RESUME=1: using specific checkpoint $RESUME_FROM_CKPT"
        args+=(--resume_from_checkpoint "$RESUME_FROM_CKPT")
    else
        LAST_CKPT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
        if [ -z "$LAST_CKPT" ]; then
            echo "ERROR: RESUME=1 but no checkpoints found in $OUTPUT_DIR"
            exit 1
        fi
        echo "RESUME=1: resuming from $LAST_CKPT"
        args+=(--resume_from_checkpoint "$LAST_CKPT")
    fi
fi

ACCEL_CFG=/tmp/accelerate_ddp_no_ot_${$}.yaml
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
    train_grpo_learned_theta.py "${args[@]}"

if [ -f "${DEFAULT_ACCEL}.bak_$$" ]; then
    mv "${DEFAULT_ACCEL}.bak_$$" "$DEFAULT_ACCEL"
    echo "Restored original accelerate config"
fi
rm -f "$ACCEL_CFG"

echo ""
echo "=== Ablation (no OT) complete. Check WandB for: ==="
echo "  train/theta_low_param   — learned threshold (should still move without OT)"
echo "  train/theta_low_loss    — REINFORCE signal"
echo "  train/gate_latent_steps — latent steps fired"
echo "  NOTE: ot_distance and manifold_weight were disabled for this run"
