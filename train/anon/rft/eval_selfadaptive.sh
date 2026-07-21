#!/bin/bash
## ANON wrapper for the native self-adaptive eval (Step 0 probe AND Step 3 final).
## Defaults KEGG_CSV and delegates to train/rft/eval_selfadaptive.sh. Set CKPT to the
## checkpoint to evaluate (anon GRPO checkpoint for the probe, or the anon RFT output).
## CONTROLLER=1 flips to the controller baseline for A/B.
##
## Usage:
##   CKPT=<...>/train_07_selfadaptive_anon/best/model.pt bash train/anon/rft/eval_selfadaptive.sh [gpu_ids]
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

export KEGG_CSV="${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}"

exec bash "$_REPO/train/rft/eval_selfadaptive.sh" "$@"
