#!/bin/bash
#SBATCH --job-name=rft_eval_sa
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=train/logs/rft_eval_sa_%j.out
#SBATCH --error=train/logs/rft_eval_sa_%j.err

## Native SELF-ADAPTIVE eval (eval_grpo_checkpoint_final.py --self_adaptive): the model
## emits <start-latent> itself, the entropy look-ahead/controller is skipped. Serves BOTH:
##   - Step 0 probe: CKPT=<healthy GRPO checkpoint> (does it self-emit? fast?)
##   - Step 3      : CKPT=<train_07_selfadaptive/best/model.pt> (final RFT model)
##
## Look for: latent=<ctrl>/<emitted> — emitted>0 means self-adaptive latents fire.
## Compare time= vs the controller path (run without CONTROLLER=0 / drop --self_adaptive).
##
## Usage:
##   CKPT=<...>/train_07_selfadaptive/best/model.pt SPLIT=both bash train/rft/eval_selfadaptive.sh [gpu_ids]
##   CONTROLLER=1 CKPT=<...> bash train/rft/eval_selfadaptive.sh   # baseline (controller, no self_adaptive)

CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-}
OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/test/log}
SPLIT=${SPLIT:-both}
N_SAMPLES=${N_SAMPLES:--1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-800}
CONTROLLER=${CONTROLLER:-0}          # 1 = baseline (controller mode, no --self_adaptive)

CKPT=${CKPT:-}
THETA_LOW_PT=${THETA_LOW_PT:-}
THETA_LOW=${THETA_LOW:-0.5}
THETA_HIGH=${THETA_HIGH:-3.0}
DNA_CACHE=${DNA_CACHE:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_2048.pt}
STAGE2_DIR=${STAGE2_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_05_stage2_hiref}

if [ -z "$CKPT" ]; then
    echo "ERROR: set CKPT=<checkpoint dir or model.pt to evaluate>"
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

_MODE=$([ "$CONTROLLER" = "1" ] && echo "controller" || echo "selfadaptive")
## Run-log co-located with the eval outputs (OUTPUT_DIR); tee keeps terminal + file.
## In-process redirect (NOT a re-exec — a re-exec re-runs conda activate in a fresh
## non-interactive bash where the `conda` shell function isn't defined, which hangs).
LOG="${LOG:-$OUTPUT_DIR/rft_${_MODE}_$(basename "$CKPT" .pt)_$(date +%Y%m%d_%H%M%S)_run.log}"
exec > >(tee "$LOG") 2>&1
echo "Command:    bash $0 $*"
echo "Checkpoint: $CKPT"
echo "Mode:       $_MODE   split=$SPLIT  n=$N_SAMPLES"
nvidia-smi

if [ -n "$KEGG_CSV" ]; then
    DATASET_ARGS=(--kegg_csv "$KEGG_CSV")
else
    DATASET_ARGS=(--dataset_name "$KEGG_DATASET")
fi

EXTRA=()
[ "$CONTROLLER" != "1" ] && EXTRA+=(--self_adaptive)
[ -n "$THETA_LOW_PT" ] && [ -f "$THETA_LOW_PT" ] && EXTRA+=(--theta_low_pt "$THETA_LOW_PT")
[ -n "$DNA_CACHE" ] && [ -f "$DNA_CACHE" ] && EXTRA+=(--dna_cache "$DNA_CACHE")
[ -n "$STAGE2_DIR" ] && EXTRA+=(--stage2_dir "$STAGE2_DIR")

FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)" 2>/dev/null || echo 29534)
ACCEL_CFG=/tmp/accelerate_rft_eval_${$}.yaml
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
    --split                 "$SPLIT" \
    --n_samples             "$N_SAMPLES" \
    --temperature           0.0 \
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
    --output_prefix         "rft_${_MODE}_$(basename "$CKPT" .pt)" \
    "${EXTRA[@]}"

echo ""
echo "=== Eval ($_MODE) done. See $OUTPUT_DIR for _metrics.json / _predictions.csv ==="
