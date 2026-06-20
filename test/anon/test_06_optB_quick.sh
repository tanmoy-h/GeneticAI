#!/bin/bash
#SBATCH --job-name=w11_optB_quick_anon
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --output=week11tests/logs/test_optB_quick_%j.out
#SBATCH --error=week11tests/logs/test_optB_quick_%j.err

## Quick smoke test for adaptive theta_low (Option B) training.
##
## Runs 15 steps with all warmups disabled so REINFORCE is active
## from step 1. Verifies:
##   - No crash, no NaN loss
##   - theta_low_param receives gradients and moves from its init value
##   - Gate and injector activate correctly
##   - DNA cache is loaded and used
##
## Usage:
##   bash week11tests/sh_test_optB_quick.sh [gpu_ids]
##   sbatch week11tests/sh_test_optB_quick.sh

## ── Configuration (mirrors sh_stage3_grpo_w9_optB.sh) ────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
STAGE1_CKPT=${STAGE1_CKPT:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/week9tests/stage1_51/s04_pass01/model.pt}
GATE_CKPT_DIR=${GATE_CKPT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/week9tests/stage1_51/s04_pass01}
GATE_CKPT=${GATE_CKPT:-${GATE_CKPT_DIR}/thinking_gate.pt}
INJECTOR_CKPT=${INJECTOR_CKPT:-${GATE_CKPT_DIR}/dna_injector.pt}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
STAGE2_DIR=${STAGE2_DIR:-stage2_output_w9}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/week11tests/test_optB_quick}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p week11tests/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0,1}
export PYTORCH_ALLOC_CONF=expandable_segments:True
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

LOG=week11tests/logs/test_optB_quick_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to:  $LOG"
echo "CUDA:        $CUDA_VISIBLE_DEVICES  NUM_GPUS: $NUM_GPUS"
echo "Stage1 ckpt: $STAGE1_CKPT"
echo "DNA cache:   $DNA_CACHE"
nvidia-smi

args=(
    --sft_checkpoint         "$STAGE1_CKPT"
    --kegg_csv               "$KEGG_CSV"
    --output_dir             "$OUTPUT_DIR"
    --cache_dir              "$CACHE_DIR"

    ## Model (same as full run)
    --text_model_name        Qwen/Qwen3-1.7B
    --dna_model_name         evo2_7b_base
    --dna_is_evo2            True
    --dna_embedding_layer    blocks.28.mlp.l3
    --use_cross_attention    True
    --dna_model_finetune     False
    --max_length_text        6000
    --max_length_dna         2048
    --truncate_dna_per_side  1024

    ## Gate — no warmup so it activates from step 1
    --use_hrpo_gate          True
    --use_dna_gate           False
    --gate_reg_weight        0.001
    --gate_warmup_steps      0
    --gate_ramp_steps        0
    --max_latent_steps       1
    --min_latent_steps       0

    ## LatentSp — no warmup so REINFORCE fires from step 1
    --latentSp_theta_low        1.0
    --latentSp_theta_high       3.0
    --latentSp_max_consec       3
    --latent_lookahead_k        3
    --latentSp_warmup_steps     0
    --latentSp_ramp_steps       0
    --theta_low_lr           1e-4
    --theta_low_weight       0.1
    --theta_low_alpha        5.0

    ## GRPO (same as full run)
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

    ## Training — short run
    --per_device_train_batch_size  1
    --gradient_accumulation_steps 4
    --max_steps              15
    --learning_rate          2e-6
    --lora_r                 16
    --lora_alpha             32
    --eval_strategy          steps
    --eval_steps             10
    --save_steps             15
    --save_total_limit       1
    --load_best_model_at_end False
    --per_device_eval_batch_size 1
    --max_eval_samples       10
    --logging_steps          1
    --report_to              none
    --bf16                   True
    --use_vllm               False
)

## Stage 2 manifold
if [ -n "${STAGE2_DIR:-}" ] && [ -d "$STAGE2_DIR" ]; then
    echo "Stage 2 manifold: $STAGE2_DIR"
    args+=(--stage2_dir "$STAGE2_DIR")
else
    echo "INFO: stage2_dir not found ($STAGE2_DIR) — manifold loss off for quick test"
fi

## Gate checkpoint
if [ -n "${GATE_CKPT:-}" ] && [ -f "$GATE_CKPT" ]; then
    echo "Gate ckpt: $GATE_CKPT"
    args+=(--gate_ckpt "$GATE_CKPT")
    [ -f "${INJECTOR_CKPT:-}" ] && args+=(--injector_ckpt "$INJECTOR_CKPT")
else
    echo "WARNING: GATE_CKPT not found — gate starts fresh"
fi

## DNA cache
if [ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ]; then
    echo "DNA cache: enabled"
    args+=(--dna_cache "$DNA_CACHE")
else
    echo "WARNING: DNA cache not found — Evo2 will run live (slower)"
fi

ACCEL_CFG=/tmp/accelerate_ddp_optB_quick_${$}.yaml
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
unset ACCELERATE_CONFIG_FILE
DEFAULT_ACCEL=~/.cache/huggingface/accelerate/default_config.yaml
[ -f "$DEFAULT_ACCEL" ] && cp "$DEFAULT_ACCEL" "${DEFAULT_ACCEL}.bak_$$"
cp "$ACCEL_CFG" "$DEFAULT_ACCEL" 2>/dev/null || true

ACCELERATE_USE_DEEPSPEED=false \
stdbuf -oL -eL accelerate launch \
    --config_file "$ACCEL_CFG" \
    train_grpo_learned_theta.py "${args[@]}"

EXIT_CODE=$?

if [ -f "${DEFAULT_ACCEL}.bak_$$" ]; then
    mv "${DEFAULT_ACCEL}.bak_$$" "$DEFAULT_ACCEL"
fi
rm -f "$ACCEL_CFG"

echo ""
echo "=== Quick test complete (exit=$EXIT_CODE) ==="
echo "Check the log for:"
echo "  [OptB] theta_low_param=<value>  — should differ from 1.0 after 15 steps"
echo "  train/theta_low_loss            — should be non-zero from step 1"
echo "  train/n_latent_decisions        — should be non-zero (REINFORCE firing)"
echo "  No 'NaN' or 'inf' in loss       — stability check"
echo "  [DNA cache] hits                — cache working"
echo ""

## Extract theta_low movement from log
python3 - "$LOG" <<'PYEOF'
import re, sys
log = open(sys.argv[1]).read()
# matches both: theta_low_param=1.0000  and  'train/theta_low_param': 1.0
params = re.findall(r"theta_low_param[=': ]+([0-9.]+)", log)
if params:
    print(f"theta_low_param: {params[0]} (init) -> {params[-1]} (final)")
    moved = abs(float(params[-1]) - float(params[0])) > 1e-5
    print(f"Moved: {'YES - REINFORCE is working' if moved else 'NO - check gradients'}")
else:
    print("No theta_low_param values found in log")
PYEOF
