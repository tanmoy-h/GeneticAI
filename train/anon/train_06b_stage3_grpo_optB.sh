#!/bin/bash
#SBATCH --job-name=w11_optB_grpo_anon
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=train/anon/logs/train_06b_stage3_grpo_optB_%j.out
#SBATCH --error=train/anon/logs/train_06b_stage3_grpo_optB_%j.err

## Stage 3 Option B: learnable theta_low via REINFORCE.
##
## Identical to the w9 GRPO run except theta_low is an nn.Parameter trained
## end-to-end alongside the main GRPO objective:
##
##   p_latent = sigmoid(alpha * (theta_low - entropy))
##   theta_loss = -adv_mean * mean(log π(decisions))
##
## What to watch in the log / WandB:
##   train/theta_low_param   — should move away from init (1.0) over training
##   train/theta_low_eff     — effective theta = param * warmup_scale
##   train/theta_low_loss    — REINFORCE signal (non-zero after step ~800)
##   train/n_latent_decisions — decisions per step (non-zero after warmup)
##   train/gate_latent_steps — latent steps fired in generation
##   No NaN loss / no crash  — mechanisms stable
##
## Usage:
##   STAGE1_CKPT=<path/to/stage1_51/model.pt> bash train/train_06b_stage3_grpo_optB.sh [gpu_ids]
##   RESUME=1 STAGE1_CKPT=<path> bash train/train_06b_stage3_grpo_optB.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-dna-grpo-optB}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_06b_stage3_grpo_optB_anon}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
STAGE2_DIR=${STAGE2_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_05_stage2_hiref_anon}    # manifold loss source
## ─────────────────────────────────────────────────────────────────────────────

## Use Stage 1.5.1 SFT weights as the starting point
STAGE1_CKPT=${STAGE1_CKPT:-}
## Gate/injector from Stage 1.5.1
GATE_CKPT_DIR=${GATE_CKPT_DIR:-}

## Pick Stage 1.51 starting checkpoint.
## Priority: (1) explicit STAGE1_CKPT, (2) deterministic pointer from the 1.51 eval,
## (3) best-ACCURACY checkpoint from the latest test_04b eval JSON, (4) fallback to
## stage151_ckpt.txt (best val_loss).
_S151_DIR=/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51_anon
_PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

## Prefer the deterministic pointer written by eval_stage1_51_checkpoints.py
if [ -z "${STAGE1_CKPT:-}" ] && [ -f "$_S151_DIR/stage151_eval_best_ckpt.txt" ]; then
    STAGE1_CKPT=$(cat "$_S151_DIR/stage151_eval_best_ckpt.txt")
    echo "STAGE1_CKPT (from stage151_eval_best_ckpt.txt): $STAGE1_CKPT"
fi

if [ -z "${STAGE1_CKPT:-}" ]; then
    _RESULTS_JSON=$(find "$_PROJECT_DIR/test/anon/logs" \
        -not -path "*/.*/*" \
        -name "test_04b_eval_stage1_51_results_*.json" 2>/dev/null | sort -V | tail -1)
    if [ -n "$_RESULTS_JSON" ]; then
        STAGE1_CKPT=$(python3 -c "
import json, sys
try:
    d = json.load(open('$_RESULTS_JSON'))
    print(d['best']['full_path'])
except Exception:
    sys.exit(1)
" 2>/dev/null)
        [ -n "$STAGE1_CKPT" ] && echo "STAGE1_CKPT (best accuracy from eval): $STAGE1_CKPT" \
            || echo "WARNING: could not parse best checkpoint from $_RESULTS_JSON"
    fi
fi

if [ -z "${STAGE1_CKPT:-}" ]; then
    _TRAIN_CKPT_FILE="$_S151_DIR/stage151_ckpt.txt"
    if [ -f "$_TRAIN_CKPT_FILE" ]; then
        STAGE1_CKPT=$(cat "$_TRAIN_CKPT_FILE")
        echo "STAGE1_CKPT (fallback, best val_loss): $STAGE1_CKPT"
    else
        echo "ERROR: STAGE1_CKPT not set, no eval JSON, and $_TRAIN_CKPT_FILE not found."
        echo "       Run train_04_stage1_51.sh (+ test_04b_eval_stage1_51.sh), or set STAGE1_CKPT=<path>."
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
    echo "Usage: STAGE1_CKPT=<path> bash train/train_06b_stage3_grpo_optB.sh [gpu_ids]"
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
" 2>/dev/null | grep -E '^[0-9]+$' | tail -1)
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

LOG=train/anon/logs/train_06b_stage3_grpo_optB_$(date +%Y%m%d_%H%M%S).log
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

    ## HRPO gate
    --use_hrpo_gate          True
    --use_dna_gate           False
    --gate_reg_weight        0.001
    --gate_warmup_steps      100
    --gate_ramp_steps        200
    --max_latent_steps       1
    --min_latent_steps       0

    ## LatentSp — learnable theta_low (Option B)
    ## theta_low_init: starting value; will be learned via REINFORCE after warmup
    ## theta_low_lr:   dedicated LR for theta_low_param (~20x base LR)
    ## theta_low_weight: REINFORCE loss scale (0.05 → ~5% of total loss; lowered
    ##                    from 0.1 so the learned threshold settles instead of
    ##                    climbing to ~0.97 and inflating latent firing/time)
    ## theta_low_alpha: sigmoid temperature (keep fixed)
    --latentSp_theta_low        0.8
    --latentSp_theta_high       3.0
    --latentSp_max_consec       2
    --latent_lookahead_k        3
    --latentSp_warmup_steps     400
    --latentSp_ramp_steps       400
    --theta_low_lr           1e-4
    --theta_low_weight       0.05
    --theta_low_alpha        5.0
    ## theta_low is optimised by sliding-window SPSA on the reward (fixes the
    ## zero-signal REINFORCE + removes theta grad-sync/deadlock). Search starts
    ## after the latent warmup+ramp. --latentSp_theta_low is the search START value.
    --theta_search           True
    --theta_search_delta     0.1
    --theta_search_lr        0.02
    --theta_search_window    30

    ## GRPO
    --num_generations        8
    --max_completion_length  800
    --temperature            0.7
    --top_p                  0.95
    --top_k                  50
    --beta                   0.15
    --epsilon                0.1
    --max_grad_norm          0.1
    ## Reward shaping: correctness DOMINATES (weight 2.0 → max +6.0); all shaping
    ## terms are gentle nudges (≤0.5) so a correct answer always beats style/brevity
    ## points. length_penalty is the total-completion time lever. Weights MUST stay
    ## aligned 1:1 with reward_funcs order (trainer errors otherwise).
    --reward_funcs           format correctness completion_quality reasoning_quality latent_format ot_distance latent_usage conciseness
    --reward_weights         0.5    2.0         0.5                0.3               0.5           0.5         0.75         0.5
    --manifold_weight        0.01
    --max_clip_loss_weight   0.0

    ## Training
    --per_device_train_batch_size  1
    --gradient_accumulation_steps 4
    --num_train_epochs       3
    --max_steps              -1
    --learning_rate          1e-6
    --lora_r                 16
    --lora_alpha             32
    --eval_strategy          steps
    --eval_steps             $SAVE_STEPS
    --save_steps             $SAVE_STEPS
    --load_best_model_at_end True
    --metric_for_best_model  correctness
    --greater_is_better      True
    --per_device_eval_batch_size 1
    --max_eval_samples       290
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

ACCEL_CFG=/tmp/accelerate_ddp_optB_${$}.yaml
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

echo ""
echo "=== Option B complete. Check WandB for: ==="
echo "  train/theta_low_param   — learned threshold (init=1.0, should move)"
echo "  train/theta_low_eff     — effective threshold after warmup scale"
echo "  train/theta_low_loss    — REINFORCE signal (non-zero after step ~800)"
echo "  train/n_latent_decisions — decisions per step (non-zero after warmup)"
echo "  train/gate_latent_steps — latent steps fired per generation"
