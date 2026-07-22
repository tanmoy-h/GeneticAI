#!/bin/bash
## ANON wrapper for Step 1a RFT sampling. Defaults KEGG_CSV + anon output paths and
## delegates to train/rft/sample_traces.sh (which does the actual work). Set CKPT to an
## anon GRPO checkpoint where latents fire (train_06b_stage3_grpo_optB_anon/checkpoint-*).
##
## Usage:
##   CKPT=<...>/train_06b_stage3_grpo_optB_anon/checkpoint-772 \
##     bash train/anon/rft/sample_traces.sh [gpu_ids]
##   (theta_low.pt / gate / injector all auto-load from CKPT's dir.)
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

export KEGG_CSV="${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-$_REPO/train/rft/samples_anon}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-rft_sample_anon}"

exec bash "$_REPO/train/rft/sample_traces.sh" "$@"
