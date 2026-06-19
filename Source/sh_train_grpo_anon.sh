#!/bin/bash -l
#SBATCH -J bioreason_grpo_anon
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=80G
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

## BioReason GRPO training on an anonymized KEGG dataset (HF Hub).
## Requires a completed SFT checkpoint from sh_train_sft_anon.sh.
##
## Before running:
##   python upload_anon_dataset_to_hf.py \
##       --csv_path /path/to/stage1_anon_genes_mol_keep_chr.csv \
##       --repo_id  iitp-cse/kegg-anon-stage1
##
## Usage:
##   SFT_CHECKPOINT=checkpoints/.../last.ckpt bash Source/sh_train_grpo_anon.sh

## ── Configuration ──────────────────────────────────────────────────────────────
USER=USERNAME                              # Change to your username
ENV_NAME="bio"

export CACHE_DIR=CACHE_DIR                 # Change to your HuggingFace cache dir
export SFT_CHECKPOINT=SFT_CHECKPOINT       # Change to your SFT checkpoint path
export OUTPUT_DIR=OUTPUT_DIR               # Change to your output dir
export WANDB_PROJECT="bioreason-grpo-anon"

## HF Hub repo id of the uploaded anonymized dataset
HF_DATASET=iitp-cse/kegg-anon-stage1      # Change to your uploaded repo id
## ───────────────────────────────────────────────────────────────────────────────

export PATH="/home/$USER/miniconda/envs/$ENV_NAME/bin:$PATH"
source "/home/$USER/miniconda/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

set -u

: "${SLURM_NTASKS:=2}"
: "${SLURM_CPUS_PER_TASK:=8}"

cd "$HOME/BioReason"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))

echo "[driver] host=$(hostname) MASTER=${MASTER_ADDR}:${MASTER_PORT} tasks=${SLURM_NTASKS}"
echo "HF_DATASET: $HF_DATASET"
nvidia-smi -L || true

srun --ntasks="$SLURM_NTASKS" \
     --cpu-bind=cores \
     --gpu-bind=single:1 \
     --label \
     --output=rank-%j-%t.log \
     bash -s <<'SRUN_PAYLOAD'
set -euo pipefail
echo "[rank ${SLURM_PROCID}] host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-unset}"

export WORLD_SIZE="${SLURM_NTASKS}"
export RANK="${SLURM_PROCID}"
export LOCAL_RANK=0

python -u train_grpo_anon.py \
  --dataset_name               "$HF_DATASET" \
  --text_model_name            "Qwen/Qwen3-1.7B" \
  --dna_model_name             "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species" \
  --cache_dir                  "$CACHE_DIR" \
  --sft_checkpoint             "$SFT_CHECKPOINT" \
  --peft_ckpt                  False \
  --dna_is_evo2                False \
  --truncate_dna_per_side      1024 \
  --deepspeed                  grpo_trainer_lora_model/ds_config_stage2.json \
  --lora_r                     16 \
  --lora_alpha                 32 \
  --lora_dropout               0 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing     True \
  --max_steps                  1000 \
  --max_completion_length      800 \
  --num_generations            8 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size  8 \
  --beta                       0.0 \
  --run_name                   bioreason-grpo-anon \
  --learning_rate              1e-5 \
  --logging_steps              1 \
  --temperature                1 \
  --top_p                      0.95 \
  --top_k                      20 \
  --output_dir                 "$OUTPUT_DIR" \
  --save_strategy              steps \
  --save_steps                 100 \
  --save_total_limit           2 \
  --log_completions            True \
  --use_vllm                   True \
  --vllm_mode                  colocate \
  --vllm_tensor_parallel_size  1 \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_max_model_len         3000 \
  --bf16                       True \
  --resume_from_checkpoint     True
SRUN_PAYLOAD
