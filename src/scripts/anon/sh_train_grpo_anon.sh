#!/bin/bash
#SBATCH --job-name=train_bioreason_grpo_anon
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=80G
#SBATCH --output=src/scripts/logs/train_bioreason_grpo_anon_%j.out
#SBATCH --error=src/scripts/logs/train_bioreason_grpo_anon_%j.err

## BioReason GRPO on anonymised KEGG dataset (HuggingFace).
## Requires a completed SFT checkpoint from sh_train_bioreason_anon.sh.
## 2 GPUs: one for training, one for vLLM colocate inference.
##
## Usage:
##   bash src/scripts/anon/sh_train_grpo_anon.sh [hf_dataset]
##   SFT_CKPT=/path/to/sft.ckpt bash src/scripts/anon/sh_train_grpo_anon.sh
##   sbatch src/scripts/anon/sh_train_grpo_anon.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=${CONDA_ENV:-dna_env}
export CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
export WANDB_PROJECT=${WANDB_PROJECT:-BioReasonGRPOAnonE5}
export WANDB_ENTITY=${WANDB_ENTITY:-iitp-cse}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/bioreason_grpo_anon}
export KEGG_HF=${1:-${KEGG_HF:-iit-patna-cse-ai/kegg-anon-global}}
export SFT_CKPT=${SFT_CKPT:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/bioreason_anon/asBioReasonE5-kegg-Qwen3-1.7B-20260620-155107/asBioReasonE5-kegg-Qwen3-1.7B-epoch=03-val_loss_epoch=0.4109.ckpt}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p scripts/logs
nvidia-smi -L || true

LOG=scripts/logs/train_bioreason_grpo_anon_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
set -x
echo "Command:    bash $0 $*"
echo "Logging:    $LOG"
echo "KEGG HF:    $KEGG_HF"
echo "SFT_CKPT:  $SFT_CKPT"
echo "OUTPUT:     $CHECKPOINT_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((12000 + ${SLURM_JOB_ID:-0} % 20000))

: "${SLURM_NTASKS:=2}"

srun --ntasks="$SLURM_NTASKS" \
     --cpu-bind=cores \
     --gpu-bind=single:1 \
     --label \
     bash -s <<'SRUN_PAYLOAD'
set -euo pipefail
echo "[rank ${SLURM_PROCID}] host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-unset}"

export WORLD_SIZE="${SLURM_NTASKS}"
export RANK="${SLURM_PROCID}"
export LOCAL_RANK=0

python -u train_grpo_anon.py \
  --dataset_name                "$KEGG_HF" \
  --text_model_name             Qwen/Qwen3-1.7B \
  --dna_model_name              evo2_7b_base \
  --dna_is_evo2                 True \
  --dna_embedding_layer         blocks.28.mlp.l3 \
  --cache_dir                   "$CACHE_DIR" \
  --sft_checkpoint              "$SFT_CKPT" \
  --peft_ckpt                   False \
  --truncate_dna_per_side       1024 \
  --lora_r                      16 \
  --lora_alpha                  32 \
  --lora_dropout                0 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing      True \
  --max_steps                   1000 \
  --max_completion_length       800 \
  --num_generations             8 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size  1 \
  --beta                        0.0 \
  --run_name                    bioreason-grpo-anon \
  --learning_rate               1e-5 \
  --logging_steps               1 \
  --temperature                 1 \
  --top_p                       0.95 \
  --top_k                       20 \
  --output_dir                  "$CHECKPOINT_DIR" \
  --save_strategy               steps \
  --save_steps                  100 \
  --save_total_limit            2 \
  --log_completions             True \
  --use_vllm                    True \
  --vllm_mode                   colocate \
  --vllm_tensor_parallel_size   1 \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_max_model_len          3000 \
  --bf16                        True \
  --resume_from_checkpoint      True
SRUN_PAYLOAD

echo "=== GRPO training complete ==="
