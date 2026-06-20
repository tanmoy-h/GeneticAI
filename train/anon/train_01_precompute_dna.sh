#!/bin/bash
#SBATCH --job-name=w9_precompute_dna_anon
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=week9tests/logs/precompute_dna_%j.out
#SBATCH --error=week9tests/logs/precompute_dna_%j.err

## Week 9: Precompute Evo2 DNA embeddings for KEGG dataset.
##
## Evo2 7B is frozen during training, so its outputs are deterministic.
## Running this once caches the raw layer activations to disk so
## train_latent_sft_cached.py can skip the Evo2 forward pass entirely,
## freeing ~14 GB GPU VRAM at training time.
##
## Covers train + val + test splits (precompute_dna_embeddings.py loops
## over all three).  If the output file already exists the script
## appends only missing sequences — already-cached keys are skipped.
##
## Usage:
##   bash week9tests/sh_precompute_dna_w9.sh [gpu_id]
##   KEGG_CSV=/path/to/kegg.csv bash week9tests/sh_precompute_dna_w9.sh 0
##   OUTPUT_PATH=/custom/path/cache.pt bash week9tests/sh_precompute_dna_w9.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
CACHE_DIR=~/.cache/huggingface
KEGG_DATASET=${KEGG_DATASET:-wanglab/kegg}
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
OUTPUT_PATH=${OUTPUT_PATH:-/scratch/tanmoyh_iitp/GenoMorph/cache/dna_embeddings_kegg_anon_2048.pt}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/../.."
mkdir -p week9tests/logs
export TMPDIR=$(pwd)/tmp && mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=${1:-0}

LOG=week9tests/logs/precompute_dna_$(date +%Y%m%d_%H%M%S).log
mkdir -p week9tests/logs
exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "CUDA:          $CUDA_VISIBLE_DEVICES"
echo "Output path:   $OUTPUT_PATH"
echo "KEGG dataset:  ${KEGG_CSV:-$KEGG_DATASET}"
nvidia-smi

## Show existing cache size (create or append)
if [ -f "$OUTPUT_PATH" ]; then
    EXISTING=$(python3 -c "
import torch
c = torch.load('$OUTPUT_PATH', map_location='cpu')
print(len(c))
" 2>/dev/null)
    echo "Existing cache: ${EXISTING:-?} entries — will append missing sequences"
else
    echo "No existing cache at $OUTPUT_PATH — creating from scratch"
fi

## Build dataset arg: CSV takes priority over HF dataset name
if [ -n "$KEGG_CSV" ]; then
    KEGG_ARG="--kegg_csv $KEGG_CSV"
else
    KEGG_ARG="--kegg_dataset $KEGG_DATASET"
fi

stdbuf -oL -eL python precompute_dna_embeddings.py \
    --output_path "$OUTPUT_PATH" \
    $KEGG_ARG \
    --max_length_dna        2048 \
    --truncate_dna_per_side 1024 \
    --dna_embedding_layer   blocks.28.mlp.l3 \
    --cache_dir             "$CACHE_DIR" \
    --device                cuda

## Show new cache size
NEW=$(python3 -c "
import torch
c = torch.load('$OUTPUT_PATH', map_location='cpu')
print(len(c))
" 2>/dev/null)
echo ""
echo "=== Done === cache entries: ${NEW:-?}  →  $OUTPUT_PATH"
