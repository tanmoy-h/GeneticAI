#!/bin/bash
#SBATCH --job-name=train_dna_qwen    # Name of the job
#SBATCH --time=12:00:00    # Time limit
#SBATCH --partition=gpu_batch    # Partition
#SBATCH --gpus=1    # Number of GPUs
#SBATCH --ntasks=1    # Number of tasks
#SBATCH --cpus-per-task=8    # Number of cores
#SBATCH --mem=128gb    # Memory limit
#SBATCH --output=train_dna_qwen_%j_%x.out    # Output file
#SBATCH --error=train_dna_qwen_%j_%x.err    # Error file

## Environment Setup
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "which python: $(which python)"

## Configuration Variables
# Change these to match your setup
CONDA_ENV=CONDA_ENV                     # Change to your conda environment name
CACHE_DIR=CACHE_DIR                     # Change to your HuggingFace cache directory
OUTPUT_DIR=OUTPUT_DIR                   # Change to your output/log directory
WANDB_PROJECT=WANDB_PROJECT             # Change to your W&B project name

## Setup Environment
conda activate $CONDA_ENV               # Change to your conda environment
cd .../BioReason/    # Change to the directory containing the script
nvidia-smi                             # Check GPU status


## =============================================================================
## KEGG Dataset Training
## =============================================================================

# NT-500M + Qwen3-1.7B on KEGG
stdbuf -oL -eL srun python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type dna-llm \
    --dataset_type kegg \
    --merge_val_test_set True \
    --return_answer_in_batch True

# EVO2-1B + Qwen3-1.7B on KEGG
stdbuf -oL -eL srun python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type dna-llm \
    --dataset_type kegg \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --merge_val_test_set True \
    --return_answer_in_batch True

# Qwen3-4B on KEGG (LLM-only)
stdbuf -oL -eL srun python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type llm \
    --dataset_type kegg \
    --max_length_dna 4 \
    --max_length_text 8192 \
    --truncate_dna_per_side 1024 \
    --merge_val_test_set True \
    --return_answer_in_batch True

## =============================================================================
## Variant Effect Prediction (VEP) Training
## =============================================================================

# NT-500M + Qwen3-4B on VEP
stdbuf -oL -eL srun python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding \
    --return_answer_in_batch True

# EVO2-1B + Qwen3-1.7B on VEP
stdbuf -oL -eL srun python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --text_model_name Qwen/Qwen3-1.7B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_coding \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True

# Qwen3-4B on VEP (LLM-only) - Testing max length text
stdbuf -oL -eL srun python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type llm \
    --dataset_type variant_effect_coding \
    --max_length_dna 4 \
    --max_length_text 4096 \
    --truncate_dna_per_side 1024 \
    --return_answer_in_batch True

## =============================================================================
## Variant Effect Prediction Non-SNV Training
## =============================================================================

# NT-500M + Qwen3-4B on VEP Non-SNV
stdbuf -oL -eL srun python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 1 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_non_snv \
    --return_answer_in_batch True

# EVO2-1B + Qwen3-4B on VEP Non-SNV
stdbuf -oL -eL srun python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name evo2_1b_base \
    --strategy deepspeed_stage_2 \
    --max_epochs 3 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type dna-llm \
    --dataset_type variant_effect_non_snv \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --dna_is_evo2 True \
    --dna_embedding_layer blocks.20.mlp.l3 \
    --return_answer_in_batch True

# Qwen3-4B on VEP Non-SNV (LLM-only) - Testing max length text
stdbuf -oL -eL srun python train_dna_qwen.py \
    --cache_dir $CACHE_DIR \
    --wandb_project $WANDB_PROJECT \
    --text_model_name Qwen/Qwen3-4B \
    --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --strategy deepspeed_stage_2 \
    --max_epochs 1 \
    --num_gpus 1 \
    --batch_size 2 \
    --model_type llm \
    --dataset_type variant_effect_non_snv \
    --max_length_dna 4 \
    --max_length_text 4096 \
    --truncate_dna_per_side 1024 \
    --return_answer_in_batch True