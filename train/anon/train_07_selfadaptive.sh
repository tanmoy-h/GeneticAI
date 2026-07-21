#!/bin/bash
## ANON wrapper for Step 2 RFT training. Defaults KEGG_CSV, the anon Stage-1.51 base
## checkpoint, and an anon output dir, then delegates to train/train_07_selfadaptive.sh.
## Set RFT_TRACES to the anon selected-traces JSONL.
##
## Usage:
##   RFT_TRACES=train/rft/rft_selfadaptive_anon.jsonl bash train/anon/train_07_selfadaptive.sh [gpu_id]
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../.." && pwd)"

export KEGG_CSV="${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}"
export STAGE1_CKPT="${STAGE1_CKPT:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_04_stage1_51_anon/s04_pass01/model.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/tanmoyh_iitp/GenoMorph/checkpoints/train_07_selfadaptive_anon}"
export WANDB_PROJECT="${WANDB_PROJECT:-dna-rft-selfadaptive-anon}"

exec bash "$_REPO/train/train_07_selfadaptive.sh" "$@"
