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
##   SFT_BEST=1 bash train/rft/sample_traces.sh [gpu_ids]
##   SFT_BEST=1 auto-resolves the ACCURACY-best Stage-1.51 checkpoint via the SAME chain
##   as train_06b_stage3_grpo_optB.sh (stage151_eval_best_ckpt.txt -> test_04b eval JSON
##   -> stage151_ckpt.txt), hands the sampler its directory, and forces NO_LATENT=1 so we
##   capture the SFT's strong latent-free accuracy on rare classes (not the latent-inserted
##   output that erodes them). Override with an explicit CKPT=<sNN_passMM dir> if needed.
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
## NO_LATENT=1 -> latent-free sampling (--self_adaptive skips the entropy controller;
## a non-self-emitting SFT then generates pure text). Use for the Stage-1.51 SFT run:
## it reproduces the SFT's latent-free native accuracy (the rare-disease strength we
## want) instead of re-inserting the latents that erode rare cases. Leave 0 for GRPO.
NO_LATENT=${NO_LATENT:-0}
## RARE_ONLY=1 -> sample ONLY the rare tail (diseases with <= RARE_MAX_PROMPTS prompts in
## the split), preserving original indices. The common classes come from the GRPO run, so
## the SFT source needs only the rare prompts -> ~1159 prompts drops to ~the rare handful,
## an ~8x shorter run. Auto-enabled by SFT_BEST=1. RARE_MAX_PROMPTS matches the filter's.
RARE_ONLY=${RARE_ONLY:-0}
RARE_MAX_PROMPTS=${RARE_MAX_PROMPTS:-2}
## ONLY_INDICES=<file> -> sample EXACTLY those split indices (one int/line). Use the GRPO
## miss+slow targets from select_sft_targets.py so the SFT covers GRPO's actual failures
## and slow prompts, not a frequency proxy. Takes precedence over RARE_ONLY.
ONLY_INDICES=${ONLY_INDICES:-}
## RESUME=1 -> resume a dropped run: records stream to <prefix>_traces.part*.jsonl as they
## complete, so a crash keeps all progress; RESUME=1 skips the (index,pass) already done and
## finishes the rest. Re-run the SAME command with RESUME=1 and the SAME OUTPUT_PREFIX.
RESUME=${RESUME:-0}

## Healthy GRPO checkpoint where latents fire (controller mode). REQUIRED — unless
## SFT_BEST=1, which auto-resolves the ACCURACY-best Stage-1.51 checkpoint (the same
## one train_06b_stage3_grpo_optB.sh starts from) for the rare-disease SFT source run.
CKPT=${CKPT:-}
SFT_BEST=${SFT_BEST:-0}
THETA_LOW_PT=${THETA_LOW_PT:-}          # optional; auto-loads from CKPT/theta_low.pt if unset
THETA_LOW=${THETA_LOW:-0.5}             # used if THETA_LOW_PT not set
THETA_HIGH=${THETA_HIGH:-3.0}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
STAGE2_DIR=${STAGE2_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_05_stage2_hiref}
## ─────────────────────────────────────────────────────────────────────────────

## SFT_BEST=1: mirror train_06b_stage3_grpo_optB.sh's exact resolution chain to pick the
## accuracy-best Stage-1.51 checkpoint, then hand the sampler its DIRECTORY (gate/injector
## live alongside model.pt). Also flips on NO_LATENT (SFT should be sampled latent-free).
if [ "$SFT_BEST" = "1" ] && [ -z "$CKPT" ]; then
    _S151_DIR=${S151_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51}
    _PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
    _STAGE1_CKPT=""
    ## (2) deterministic accuracy-best pointer written by eval_stage1_51_checkpoints.py
    if [ -f "$_S151_DIR/stage151_eval_best_ckpt.txt" ]; then
        _STAGE1_CKPT=$(cat "$_S151_DIR/stage151_eval_best_ckpt.txt")
        echo "SFT_BEST (from stage151_eval_best_ckpt.txt): $_STAGE1_CKPT"
    fi
    ## (3) best-accuracy from the latest test_04b eval JSON
    if [ -z "$_STAGE1_CKPT" ]; then
        _RESULTS_JSON=$(find "$_PROJECT_DIR/test/logs" -not -path "*/.*/*" \
            -name "test_04b_eval_stage1_51_results_*.json" 2>/dev/null | sort -V | tail -1)
        if [ -n "$_RESULTS_JSON" ]; then
            _STAGE1_CKPT=$(python3 -c "
import json, sys
try:
    d = json.load(open('$_RESULTS_JSON')); print(d['best']['full_path'])
except Exception:
    sys.exit(1)
" 2>/dev/null)
            [ -n "$_STAGE1_CKPT" ] && echo "SFT_BEST (best accuracy from eval JSON): $_STAGE1_CKPT"
        fi
    fi
    ## (4) fallback: val_loss best pointer
    if [ -z "$_STAGE1_CKPT" ] && [ -f "$_S151_DIR/stage151_ckpt.txt" ]; then
        _STAGE1_CKPT=$(cat "$_S151_DIR/stage151_ckpt.txt")
        echo "SFT_BEST (fallback, best val_loss): $_STAGE1_CKPT"
    fi
    if [ -z "$_STAGE1_CKPT" ]; then
        echo "ERROR: SFT_BEST=1 but no Stage-1.51 pointer/eval JSON found under $_S151_DIR."
        echo "       Run test_04b_eval_stage1_51.sh first, or set CKPT=<sNN_passMM dir> explicitly."
        exit 1
    fi
    ## the pointers store .../model.pt; the sampler wants the containing directory
    CKPT=$(dirname "$_STAGE1_CKPT")
    NO_LATENT=1                                   # SFT source is always sampled latent-free
    RARE_ONLY=${RARE_ONLY:-1}                     # and only the rare tail (common = GRPO's job)
    OUTPUT_PREFIX=${OUTPUT_PREFIX:-rft_sample_sft}
    echo "SFT_BEST resolved -> CKPT=$CKPT (NO_LATENT=1, RARE_ONLY=$RARE_ONLY, prefix=$OUTPUT_PREFIX)"
fi

if [ -z "$CKPT" ]; then
    echo "ERROR: set CKPT=<healthy GRPO checkpoint dir where latents fire, e.g. checkpoint-772>"
    echo "       or SFT_BEST=1 to auto-resolve the accuracy-best Stage-1.51 SFT checkpoint."
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
if [ -n "$ONLY_INDICES" ]; then
    echo "Mode:         NO_LATENT=$NO_LATENT  ONLY_INDICES=$ONLY_INDICES ($(wc -l < "$ONLY_INDICES" 2>/dev/null) targets)"
else
    echo "Mode:         NO_LATENT=$NO_LATENT  RARE_ONLY=$RARE_ONLY (<=$RARE_MAX_PROMPTS prompts/disease)"
fi
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
[ "$NO_LATENT" = "1" ] && EXTRA+=(--self_adaptive)   # latent-free (SFT rare-disease traces)
if [ -n "$ONLY_INDICES" ]; then                      # explicit targets win over rare filter
    EXTRA+=(--only_indices "$ONLY_INDICES")
elif [ "$RARE_ONLY" = "1" ]; then
    EXTRA+=(--rare_max_prompts "$RARE_MAX_PROMPTS")   # rare tail only
fi
[ "$RESUME" = "1" ] && EXTRA+=(--resume)             # skip (index,pass) already streamed
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
