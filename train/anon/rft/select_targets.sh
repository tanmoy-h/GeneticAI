#!/bin/bash
## ANON wrapper for Step 1a.5 RFT target selection. Defaults the anon input/output paths
## and delegates to train/rft/select_targets.sh (which does the actual work). The selector
## is dataset-agnostic (operates on the sampled traces JSONL); this wrapper just keeps anon
## outputs separate.
##
## Usage:
##   GRPO_TRACES=train/rft/samples_anon/rft_sample_anon_traces.jsonl \
##     bash train/anon/rft/select_targets.sh
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

export OUT_INDICES="${OUT_INDICES:-$_REPO/train/rft/samples_anon/sft_targets_anon.txt}"
export OUT_MARKED="${OUT_MARKED:-$_REPO/train/rft/samples_anon/rft_sample_anon_traces_marked.jsonl}"

exec bash "$_REPO/train/rft/select_targets.sh" "$@"
