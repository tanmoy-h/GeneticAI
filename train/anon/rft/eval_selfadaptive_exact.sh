#!/bin/bash
## ANON wrapper for the EXACT-reproduction self-adaptive eval (evaluate_accuracy()
## reused directly from train_latent_sft.py -- see train/rft/eval_selfadaptive_exact.sh
## for why this differs from eval_selfadaptive.sh's generate_with_hrpo_gate path).
## Defaults KEGG_CSV and routes outputs to test/anon/log (kept separate from the named
## run's test/log, since both checkpoints are typically named "best_acc").
##
## Usage:
##   CKPT=<...>/train_07_selfadaptive_anon/best_acc bash train/anon/rft/eval_selfadaptive_exact.sh [gpu_id]
##   BAN_LATENT=1 CKPT=<...> bash train/anon/rft/eval_selfadaptive_exact.sh   # latent-free comparison number
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

export KEGG_CSV="${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-$_REPO/test/anon/log}"

exec bash "$_REPO/train/rft/eval_selfadaptive_exact.sh" "$@"
