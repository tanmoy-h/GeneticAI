#!/bin/bash
## ANON wrapper for Step 1b RFT filtering. Defaults the output path and delegates to
## train/rft/filter_traces.sh. The filter itself is dataset-agnostic (operates on the
## sampled traces JSONL); this wrapper just keeps anon outputs separate.
##
## Usage:
##   TRACES=train/rft/samples_anon/rft_sample_anon_traces.jsonl \
##     bash train/anon/rft/filter_traces.sh
set -euo pipefail
_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

export OUT="${OUT:-$_REPO/train/rft/rft_selfadaptive_anon.jsonl}"

exec bash "$_REPO/train/rft/filter_traces.sh" "$@"
