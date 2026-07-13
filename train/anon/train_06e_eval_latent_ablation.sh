#!/bin/bash
#SBATCH --job-name=eval_latent_ablation_anon
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=train/anon/logs/eval_latent_ablation_%j.out
#SBATCH --error=train/anon/logs/eval_latent_ablation_%j.err

## Latent ablation — ZERO-training evaluation of a Stage 3 Option-B checkpoint (ANON).
## See train/train_06e_eval_latent_ablation.sh for the full rationale.
##
## Scores one frozen GRPO checkpoint on the fixed val set twice:
##   (1) baseline : latents ON  (learned theta_low from the checkpoint)
##   (2) ablation : latents OFF (--disable_latents; theta_low.pt not reloaded)
## No optimizer steps run — the delta in eval_correctness is the latent contribution.
## Runs on a SINGLE GPU so both arms score the identical records.
##
## Usage:
##   CKPT=<.../train_06b_stage3_grpo_optB_anon/checkpoint-N> bash train/anon/train_06e_eval_latent_ablation.sh [gpu_id]

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_06b_stage3_grpo_optB_anon}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-50}
MODE=${MODE:-both}   # both | baseline | ablation
## ─────────────────────────────────────────────────────────────────────────────

CKPT=${CKPT:-${RESUME_FROM_CKPT:-}}
if [ -z "${CKPT:-}" ]; then
    echo "ERROR: CKPT is not set. Point it at a Stage 3 anon checkpoint dir, e.g."
    echo "  CKPT=$OUTPUT_DIR/checkpoint-1158 bash train/anon/train_06e_eval_latent_ablation.sh"
    exit 1
fi
if [ ! -d "$CKPT" ]; then
    echo "ERROR: CKPT is not a directory: $CKPT"
    exit 1
fi

## Base Stage 1.51 SFT weights (anon).
_S151_DIR=/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51_anon
STAGE1_CKPT=${STAGE1_CKPT:-}
if [ -z "${STAGE1_CKPT:-}" ] && [ -f "$_S151_DIR/stage151_eval_best_ckpt.txt" ]; then
    STAGE1_CKPT=$(cat "$_S151_DIR/stage151_eval_best_ckpt.txt")
    echo "STAGE1_CKPT (from stage151_eval_best_ckpt.txt): $STAGE1_CKPT"
fi
if [ -z "${STAGE1_CKPT:-}" ] && [ -f "$_S151_DIR/stage151_ckpt.txt" ]; then
    STAGE1_CKPT=$(cat "$_S151_DIR/stage151_ckpt.txt")
    echo "STAGE1_CKPT (fallback, best val_loss): $STAGE1_CKPT"
fi
if [ -z "${STAGE1_CKPT:-}" ]; then
    echo "ERROR: STAGE1_CKPT not set and no 1.51 anon pointer found. Set STAGE1_CKPT=<base model.pt>."
    exit 1
fi

GATE_CKPT_DIR=${GATE_CKPT_DIR:-$(dirname "$STAGE1_CKPT")}
GATE_CKPT=${GATE_CKPT:-${GATE_CKPT_DIR}/thinking_gate.pt}
INJECTOR_CKPT=${INJECTOR_CKPT:-${GATE_CKPT_DIR}/dna_injector.pt}

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p train/anon/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=${WANDB_MODE:-disabled}

LOG=train/anon/logs/eval_latent_ablation_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Command:     bash $0 $*"
echo "Logging to:  $LOG"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Base ckpt:   $STAGE1_CKPT"
echo "GRPO ckpt:   $CKPT"
echo "Anon CSV:    $KEGG_CSV"
echo "Eval size:   $MAX_EVAL_SAMPLES records   MODE=$MODE"
nvidia-smi

base_args=(
    --sft_checkpoint         "$STAGE1_CKPT"
    --dataset_name           "$KEGG_DATASET"
    --kegg_csv               "$KEGG_CSV"
    --output_dir             "$OUTPUT_DIR"
    --cache_dir              "$CACHE_DIR"
    --resume_from_checkpoint "$CKPT"
    --eval_only              True

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
    --max_latent_steps       1
    --min_latent_steps       0

    ## LatentSp (init only; baseline reloads learned theta_low.pt from the ckpt)
    --latentSp_theta_low     1.0
    --latentSp_theta_high    3.0
    --latentSp_max_consec    3
    --latent_lookahead_k     3
    --latentSp_warmup_steps  0
    --latentSp_ramp_steps    0
    --theta_low_alpha        5.0

    ## GRPO construction (rewards unused for correctness scoring)
    --num_generations        8
    --max_completion_length  800
    --temperature            0.7
    --top_p                  0.95
    --top_k                  50
    --reward_funcs           format correctness completion_quality reasoning_quality latent_format latent_usage

    ## Eval
    --eval_strategy          steps
    --per_device_eval_batch_size 1
    --max_eval_samples       "$MAX_EVAL_SAMPLES"
    --logging_steps          10
    --report_to              none
    --bf16                   True
    --use_vllm               False
)

[ -f "$GATE_CKPT" ]     && base_args+=(--gate_ckpt "$GATE_CKPT")
[ -f "$INJECTOR_CKPT" ] && base_args+=(--injector_ckpt "$INJECTOR_CKPT")
[ -f "$DNA_CACHE" ]     && base_args+=(--dna_cache "$DNA_CACHE")

run_arm () {
    local name="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ARM: $name"
    echo "════════════════════════════════════════════════════════════════"
    python train_grpo_learned_theta.py "${base_args[@]}" "$@"
}

if [ "$MODE" = "both" ] || [ "$MODE" = "baseline" ]; then
    run_arm "baseline (latents ON, learned theta_low)"
fi
if [ "$MODE" = "both" ] || [ "$MODE" = "ablation" ]; then
    run_arm "ablation (latents OFF)" --disable_latents True
fi

echo ""
echo "=== Done. Compare eval_correctness across arms: ==="
echo "    grep -E 'ARM:|eval_only metrics' $LOG"
