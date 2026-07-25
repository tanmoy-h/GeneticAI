#!/bin/bash
## ANON wrapper for Step 1 RFT sampling. Defaults KEGG_CSV + anon output paths + anon Stage-2
## manifold, and delegates to train/rft/sample_traces.sh. Set CKPT to an anon GRPO checkpoint
## where latents fire (train_06b_stage3_grpo_optB_anon/checkpoint-*).
##
## Usage:
##   CKPT=<...>/train_06b_stage3_grpo_optB_anon/checkpoint-772 \
##     bash train/anon/rft/sample_traces.sh [gpu_ids]
##   (theta_low.pt / gate / injector all auto-load from CKPT's dir.)
## All other env vars (RESUME, SAMPLE_PASSES, ...) pass straight through.
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

export KEGG_CSV="${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-$_REPO/train/rft/samples_anon}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-rft_sample_anon}"
## Stage-2 manifold must match what the anon model trained with (anon variant). DNA_CACHE is
## shared — DNA embeddings aren't anonymized, only the question text is — so it isn't overridden.
export STAGE2_DIR="${STAGE2_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_05_stage2_hiref_anon}"

exec bash "$_REPO/train/rft/sample_traces.sh" "$@"
