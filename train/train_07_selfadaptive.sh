#!/bin/bash
#SBATCH --job-name=w9_rft_selfadaptive
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=train/logs/train_07_selfadaptive_%j.out
#SBATCH --error=train/logs/train_07_selfadaptive_%j.err

## Step 2 of the self-adaptive latent plan: RFT-SFT on sampled GRPO traces so the
## model learns to EMIT <start-latent>/<end-latent> itself (no controller at inference).
##
## Pipeline (run these first):
##   1. Sample traces from a healthy GRPO checkpoint on the TRAIN split:
##        python eval_grpo_checkpoint_final.py --checkpoint <ckpt-386> \
##            --split train --temperature 0.7 --sample_passes 8 --n_samples -1 \
##            --output_prefix rft_sample
##   2. Filter to correct+latent+short:
##        python train/rft/build_rft_dataset.py \
##            --traces <...>rft_sample_traces.jsonl --out train/rft/rft_selfadaptive.jsonl
##   3. This script trains on those completions with label_self_adaptive_latents.
##
## Usage:
##   STAGE1_CKPT=<start weights> RFT_TRACES=<selected.jsonl> bash train/train_07_selfadaptive.sh [gpu_id]
##
## STAGE1_CKPT accepts EITHER:
##   * a single file  — a Stage 1.5.1 model.pt / Stage 1 .ckpt (the original path), or
##   * a checkpoint DIRECTORY — e.g. a train_06b GRPO checkpoint-XXXX. The loader
##     detects pytorch_model.bin / model.safetensors / model.pt (+ shards) and also
##     loads that dir's thinking_gate.pt + dna_injector.pt, so the FULL w9-native
##     state transfers (backbone + gate + injector), not just the backbone.
##   RFT-on-GRPO example (w9-native start — best odds the self-emission doesn't
##   degenerate into empty-latent / "Step N:" loops under teacher forcing):
##     STAGE1_CKPT=/scratch/.../train_06b_stage3_grpo_optB_spsa/checkpoint-1544 \
##       RFT_TRACES=train/rft/rft_selfadaptive.jsonl bash train/train_07_selfadaptive.sh 0

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
WANDB_PROJECT=${WANDB_PROJECT:-dna-rft-selfadaptive}
WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_07_selfadaptive}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-2}                 # passes over the RFT data
LR=${LR:-1e-5}                      # below Stage 1.51's 2e-5 (fine-tuning a strong model)
## HALF_EPOCH_EVAL=1: at 50% and 100% of every epoch, save a checkpoint
## (epochPP_half / _full — RFT drops the meaningless "sSS_" curriculum prefix) AND run a
## greedy accuracy eval on val+test (290), tracking the accuracy-best into
## $OUTPUT_DIR/best_acc/. EVAL_BAN_LATENT=1 forces a latent-free number; default 0 =
## latents allowed (self-emit, the RFT target).
HALF_EPOCH_EVAL=${HALF_EPOCH_EVAL:-1}
EVAL_BAN_LATENT=${EVAL_BAN_LATENT:-0}
## DNA_CACHE: precomputed {input_ids_bytes: embedding} map (same as sample_traces.sh /
## test_04d's cache) to skip live Evo2 during the half-epoch 290 eval only. Cache misses
## fall back to live Evo2, so a val/test-only cache is safe. Training itself is unaffected
## (always live Evo2). Empty/missing = eval always runs live (slower but still correct).
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
## ─────────────────────────────────────────────────────────────────────────────

## Start weights (file OR GRPO checkpoint dir — see Usage above). Default: the best
## Stage 1.5.1 correctness checkpoint (s04_pass02). For self-EMISSION, prefer a
## train_06b GRPO checkpoint dir instead: its backbone is already w9-native (trained
## through the free-generation loop), which the teacher-forced SFT base is not — that
## mismatch is what made the SFT-init run degenerate when self-emitting.
STAGE1_CKPT=${STAGE1_CKPT:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51/s04_pass02/model.pt}

if [ -z "${RFT_TRACES:-}" ]; then
    echo "ERROR: RFT_TRACES is not set (path to the selected-traces JSONL)."
    echo "       Build it with train/rft/build_rft_dataset.py first (see header)."
    exit 1
fi
if [ ! -f "$RFT_TRACES" ]; then
    echo "ERROR: RFT_TRACES not found: $RFT_TRACES"
    exit 1
fi
## STAGE1_CKPT may be a file (Stage 1/1.5 ckpt) or a dir (GRPO checkpoint) — accept both.
if [ ! -f "$STAGE1_CKPT" ] && [ ! -d "$STAGE1_CKPT" ]; then
    echo "ERROR: STAGE1_CKPT not found (need a model file or a checkpoint dir): $STAGE1_CKPT"
    exit 1
fi

set -e
module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p train/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_ALLOC_CONF=expandable_segments:True

## In-process tee (no re-exec, which would re-run conda activate in a fresh bash and hang).
LOG="${LOG:-train/logs/train_07_selfadaptive_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "CUDA:          $CUDA_VISIBLE_DEVICES"
echo "Start ckpt:    $STAGE1_CKPT"
echo "RFT traces:    $RFT_TRACES"
echo "Output dir:    $OUTPUT_DIR"
echo "Half-epoch eval: $HALF_EPOCH_EVAL (ban_latent=$EVAL_BAN_LATENT dna_cache=${DNA_CACHE:-none})"
echo "Epochs:        $EPOCHS   LR: $LR"
nvidia-smi

## Build dataset arg (for the val loss proxy + index-join in load_rft_rows)
if [ -n "$KEGG_CSV" ]; then
    KEGG_ARG="--kegg_csv $KEGG_CSV"
else
    KEGG_ARG="--kegg_dataset $KEGG_DATASET"
fi

## Optional half-epoch save + val+test(290) accuracy eval flags
EXTRA_EVAL=()
[ "$HALF_EPOCH_EVAL" = "1" ] && EXTRA_EVAL+=(--half_epoch_eval)
[ "$EVAL_BAN_LATENT" = "1" ] && EXTRA_EVAL+=(--eval_ban_latent)
[ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ] && EXTRA_EVAL+=(--eval_dna_cache "$DNA_CACHE")

## No --start_latent_step/--max_latent_steps: RFT mode (--rft_traces) forces the outer
## curriculum loop to a single iteration in train_latent_sft.py itself (s is never read by
## the self_adaptive branch). passes_per_step is the epoch count.
stdbuf -oL -eL python train_latent_sft.py \
    --stage1_ckpt            "$STAGE1_CKPT" \
    --rft_traces             "$RFT_TRACES" \
    $KEGG_ARG \
    --output_dir             "$OUTPUT_DIR" \
    --text_model_name        Qwen/Qwen3-1.7B \
    --dna_model_name         evo2_7b_base \
    --dna_embedding_layer    blocks.28.mlp.l3 \
    --entropy_mode           global \
    --passes_per_step        "$EPOCHS" \
    --batch_size             1 \
    --grad_accum             8 \
    --learning_rate          "$LR" \
    --max_length_text        6000 \
    --max_length_dna         2048 \
    --truncate_dna_per_side  1024 \
    --entropy_batch_size     1 \
    --log_every              20 \
    --wandb_project          "$WANDB_PROJECT" \
    --wandb_entity           "$WANDB_ENTITY" \
    --cache_dir              "$CACHE_DIR" \
    --device                 cuda \
    --seed                   "$SEED" \
    --use_gate \
    "${EXTRA_EVAL[@]}"

echo ""
echo "=== Self-adaptive RFT done. Weights in $OUTPUT_DIR/best/ ==="
echo "=== Eval (native, self-adaptive) with: ==="
echo "  python eval_grpo_checkpoint_final.py --checkpoint $OUTPUT_DIR/best/model.pt \\"
echo "      --split both --n_samples -1 --self_adaptive"
