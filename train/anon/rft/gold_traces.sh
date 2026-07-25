#!/bin/bash
## ANON wrapper for Step 1c RFT gold-trace backfill. Defaults the anon KEGG_CSV + output
## path and delegates to train/rft/gold_traces.sh.
##
## Usage:
##   INDICES=train/rft/samples_anon/uncovered_anon.txt bash train/anon/rft/gold_traces.sh
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

export KEGG_CSV="${KEGG_CSV:-$_REPO/genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}"
export OUT="${OUT:-$_REPO/train/rft/samples_anon/rft_gold_anon_traces.jsonl}"

exec bash "$_REPO/train/rft/gold_traces.sh" "$@"
