#!/bin/bash
#SBATCH --job-name=w9_stage3_grpo_anon
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=train/anon/logs/train_06_stage3_grpo_%j.out
#SBATCH --error=train/anon/logs/train_06_stage3_grpo_%j.err

## Stage 3 Week 9: GRPO + entropy-conditioned dual-mode reasoning.
##   Engine: train_grpo_latent_reasoning.py
##
## Three mechanisms:
##   1. HRPO gate (always)     — input-embedding DNA conditioning
##   2. LatentSp latent steps     — entropy < theta_low → recycle h_{t-1}, skip token
##   3. DNA hidden injection   — entropy > theta_high → inject u_dna into h_t
##
## Usage:
##   STAGE1_CKPT=<path/to/stage1_5/best/model.pt> bash train/train_06_stage3_grpo.sh [gpu_ids]
##   STAGE1_CKPT=<path> STAGE2_DIR=stage2_output_w9 bash train/train_06_stage3_grpo.sh 0,1
##   RESUME=1 STAGE1_CKPT=<path> bash train/train_06_stage3_grpo.sh
##
## Notes:
##   - STAGE1_CKPT should be the Stage 1.5 SFT best/model.pt
##   - STAGE2_DIR enables manifold alignment loss (optional)

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-dna-grpo-week9}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_06_stage3_grpo_anon}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
STAGE2_DIR=${STAGE2_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_05_stage2_hiref_anon}   # set to "" to skip manifold alignment
## ─────────────────────────────────────────────────────────────────────────────

## Stage 1.5-with-gate best checkpoint (gate-adapted model)
STAGE1_CKPT=${STAGE1_CKPT:-}
## Gate weights from the same Stage 1.5-with-gate run
GATE_CKPT_DIR=${GATE_CKPT_DIR:-}

## Use Stage 1.51 checkpoint recorded by training (written by train_04_stage1_51.sh)
if [ -z "${STAGE1_CKPT:-}" ]; then
    _TRAIN_CKPT_FILE=/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51_anon/stage151_ckpt.txt
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
    echo "Usage: STAGE1_CKPT=<path> bash train/train_06_stage3_grpo.sh [gpu_ids]"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p train/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0,1}
export WANDB_PROJECT
export WANDB_ENTITY
export PYTORCH_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

## Dynamic save_steps: 3 saves per epoch
PER_DEVICE_BATCH=1
GRAD_ACCUM=4
NUM_GENERATIONS=8
TRAIN_SIZE=$(python3 -c "
from genomorph.dataset.kegg import load_kegg_from_anon_csv
ds = load_kegg_from_anon_csv('$KEGG_CSV')
print(len(ds['train']))
" 2>/dev/null)
if [ -z "$TRAIN_SIZE" ] || [ "$TRAIN_SIZE" -le 0 ] 2>/dev/null; then
    echo "WARNING: Could not read TRAIN_SIZE — falling back to SAVE_STEPS=500"
    SAVE_STEPS=500
    STEPS_PER_EPOCH=1000
else
    STEPS_PER_EPOCH=$(( (TRAIN_SIZE * NUM_GENERATIONS + PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM - 1) / (PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM) ))
    SAVE_STEPS=$(( STEPS_PER_EPOCH / 3 ))
    [ "$SAVE_STEPS" -le 0 ] && SAVE_STEPS=500
fi
TOTAL_STEPS=$(( STEPS_PER_EPOCH * 3 ))

LOG=train/anon/logs/train_06_stage3_grpo_$(date +%Y%m%d_%H%M%S).log
mkdir -p train/anon/logs
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to:  $LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES  NUM_GPUS: $NUM_GPUS"
echo "Train size: $TRAIN_SIZE  Steps/epoch: $STEPS_PER_EPOCH  Total: $TOTAL_STEPS"
echo "Save every: $SAVE_STEPS steps"
echo "Stage1 ckpt: $STAGE1_CKPT"
echo "Output dir:  $OUTPUT_DIR"
echo "Stage 2 dir: ${STAGE2_DIR:-<none>}"
echo "DNA cache:   ${DNA_CACHE:-<not set>}"
nvidia-smi

args=(
    --sft_checkpoint         "$STAGE1_CKPT"
    --kegg_csv               "$KEGG_CSV"
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

    ## HRPO gate (input-embedding level)
    --use_hrpo_gate          True
    --use_dna_gate           False
    --gate_reg_weight        0.001
    --gate_warmup_steps      100
    --gate_ramp_steps        200
    --max_latent_steps       1
    --min_latent_steps       0

    ## LatentSp + DNA hidden injection (w9)
    --latentSp_theta_low        1.0
    --latentSp_theta_high       3.0
    --latentSp_max_consec       3
    --latent_lookahead_k     3
    --latentSp_warmup_steps     400
    --latentSp_ramp_steps       400

    ## GRPO
    --num_generations        8
    --max_completion_length  800
    --temperature            0.7
    --top_p                  0.95
    --top_k                  50
    --beta                   0.05
    --epsilon                0.1
    --max_grad_norm          0.1
    --reward_funcs           xmlcount soft_format correctness completion_quality reasoning_quality ot_distance latent_usage
    --manifold_weight        0.01
    --max_clip_loss_weight   0.0

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

## Optional Stage 2 manifold alignment
if [ -n "${STAGE2_DIR:-}" ]; then
    echo "Using Stage 2 manifold from: $STAGE2_DIR"
    args+=(--stage2_dir "$STAGE2_DIR")
fi

## Gate from Stage 1.5-with-gate (gate_warmup_steps=0 → no warmup discontinuity)
if [ -n "${GATE_CKPT:-}" ] && [ -f "$GATE_CKPT" ]; then
    echo "Gate ckpt: $GATE_CKPT"
    args+=(--gate_ckpt "$GATE_CKPT")
    [ -f "$INJECTOR_CKPT" ] && args+=(--injector_ckpt "$INJECTOR_CKPT")
else
    echo "WARNING: GATE_CKPT not found ($GATE_CKPT) — gate starts fresh + gate_warmup_steps=0 means instant activation; set GATE_CKPT_DIR or run Stage 1.5 with --use_gate first"
fi

## Optional DNA embedding cache (skips Evo2 at training time, frees ~14 GB VRAM)
if [ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ]; then
    echo "DNA cache enabled: $DNA_CACHE"
    args+=(--dna_cache "$DNA_CACHE")
elif [ -n "$DNA_CACHE" ]; then
    echo "WARNING: DNA_CACHE set but not found: $DNA_CACHE — Evo2 will run live"
fi

## Resume from checkpoint
## RESUME=1 with RESUME_FROM_CKPT=<path> → use that specific checkpoint
## RESUME=1 alone → use the latest checkpoint in OUTPUT_DIR
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

ACCEL_CFG=/tmp/accelerate_ddp_grpo_w9_${$}.yaml
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
    train_grpo_latent_reasoning.py "${args[@]}"

if [ -f "${DEFAULT_ACCEL}.bak_$$" ]; then
    mv "${DEFAULT_ACCEL}.bak_$$" "$DEFAULT_ACCEL"
    echo "Restored original accelerate config"
fi
