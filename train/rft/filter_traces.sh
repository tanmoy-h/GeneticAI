#!/bin/bash
## Step 1b of the self-adaptive RFT pipeline: filter sampled traces to
## correct AND well-formed AND (preferably) latent-using AND short, one per prompt.
## CPU-only, fast. Wraps train/rft/build_rft_dataset.py.
##
## Usage:
##   TRACES=train/rft/samples/rft_sample_traces.jsonl bash train/rft/filter_traces.sh
##   TRACES=<..> OUT=<..> MAX_WORDS=350 REQUIRE_LATENT=1 bash train/rft/filter_traces.sh
##   # merge GRPO + Stage-1.51 SFT traces (space-separated) so rare diseases with no
##   # correct GRPO latent trace are covered by an SFT trace:
##   TRACES="train/rft/samples/rft_sample_traces.jsonl train/rft/samples/rft_sample_sft_traces.jsonl" \
##     bash train/rft/filter_traces.sh
##   # PREFER_TIME=1 lets a faster SFT text trace beat a slow GRPO latent trace on GRPO's
##   # slow prompts (pick fastest correct trace, not shortest-by-tokens):
##   PREFER_TIME=1 TRACES="<grpo>_traces.jsonl <sft>_traces.jsonl" bash train/rft/filter_traces.sh
##   # dump the still-uncovered (synonymy) prompts to backfill from KEGG gold reasoning:
##   DUMP_UNCOVERED=train/rft/samples/uncovered.txt \
##     TRACES="<grpo>_traces.jsonl <sft>_traces.jsonl" bash train/rft/filter_traces.sh

CONDA_ENV=${CONDA_ENV:-dna_env}
TRACES=${TRACES:-}
OUT=${OUT:-train/rft/rft_selfadaptive.jsonl}
MAX_WORDS=${MAX_WORDS:-350}
REQUIRE_LATENT=${REQUIRE_LATENT:-0}
## Rare-disease protection (counters GRPO's flat-per-sample correctness that eroded
## macro-F1): a disease with <= RARE_MAX_PROMPTS prompts bypasses the word cap and
## require_latent (never lose its only correct trace) and is oversampled RARE_OVERSAMPLE
## times so the self-adaptive SFT weights it more. RARE_MAX_PROMPTS=0 disables.
RARE_MAX_PROMPTS=${RARE_MAX_PROMPTS:-2}
RARE_OVERSAMPLE=${RARE_OVERSAMPLE:-3}
## CONFIDENCE_OVERSAMPLE=k (>0) oversamples FRAGILE prompts: copies = round(1 + k*(1-pass_frac)),
## where pass_frac is the fraction of a prompt's traces that were correct+well-formed. Firms up
## prompts the model barely rescued via best-of-N (e.g. CJD pass_frac 0.16). Composes with rare
## oversampling via max(). 0 disables. k=3 -> up to 4 copies at 0 confidence.
CONFIDENCE_OVERSAMPLE=${CONFIDENCE_OVERSAMPLE:-0}
## GOLD_OVERSAMPLE=N -> flat copy count for GOLD-backfilled prompts (the hardest 0/4 cases the
## model never got right). Default 1 (x1). Raise to reinforce them; they're exempt from the
## confidence rule (single hand-written reasoning -> memorization risk if over-duplicated).
GOLD_OVERSAMPLE=${GOLD_OVERSAMPLE:-1}
## PREFER_TIME=1 selects the FASTEST correct trace per prompt (by gen_time_sec) instead of
## the shortest-by-tokens latent trace, so among a prompt's passes the quickest correct one
## wins. Default 0 keeps the latent-first, shortest-tokens selection.
PREFER_TIME=${PREFER_TIME:-0}
## DUMP_UNCOVERED=<file> writes the indices of prompts with NO correct trace from any source
## (the synonymy zero-coverage prompts) -> feed to train/rft/gold_traces.sh to backfill them
## with the KEGG dataset's own reasoning + answer, then re-run this filter with the gold file
## added to TRACES. Empty = don't dump.
DUMP_UNCOVERED=${DUMP_UNCOVERED:-}

if [ -z "$TRACES" ]; then
    echo "ERROR: set TRACES=<...>_traces.jsonl (output of train/rft/sample_traces.sh)"
    echo "       (space-separate several files to merge sources, e.g. GRPO + SFT traces)"
    exit 1
fi
## TRACES may be several space-separated files; verify each exists.
for _t in $TRACES; do
    if [ ! -f "$_t" ]; then
        echo "ERROR: TRACES file not found: $_t"
        exit 1
    fi
done

module load MLDL/miniconda3 2>/dev/null || true
conda activate $CONDA_ENV 2>/dev/null || true
cd "$(dirname "$0")/../.."

## $TRACES unquoted so multiple space-separated files expand to separate argv entries
## (the builder's --traces takes nargs="+"). File paths must not contain spaces.
ARGS=(--traces $TRACES --out "$OUT")
[ -n "$MAX_WORDS" ] && ARGS+=(--max_words "$MAX_WORDS")
[ "$REQUIRE_LATENT" = "1" ] && ARGS+=(--require_latent)
[ -n "$RARE_MAX_PROMPTS" ] && ARGS+=(--rare_max_prompts "$RARE_MAX_PROMPTS")
[ -n "$RARE_OVERSAMPLE" ] && ARGS+=(--rare_oversample "$RARE_OVERSAMPLE")
[ -n "$CONFIDENCE_OVERSAMPLE" ] && ARGS+=(--confidence_oversample "$CONFIDENCE_OVERSAMPLE")  # 0 = no-op
[ -n "$GOLD_OVERSAMPLE" ] && ARGS+=(--gold_oversample "$GOLD_OVERSAMPLE")                    # 1 = x1
[ "$PREFER_TIME" = "1" ] && ARGS+=(--prefer_time)
[ -n "$DUMP_UNCOVERED" ] && ARGS+=(--dump_uncovered "$DUMP_UNCOVERED")

echo "Filtering $TRACES -> $OUT  (max_words=$MAX_WORDS require_latent=$REQUIRE_LATENT rare<=$RARE_MAX_PROMPTS x$RARE_OVERSAMPLE prefer_time=$PREFER_TIME dump_uncovered=${DUMP_UNCOVERED:-none})"
python train/rft/build_rft_dataset.py "${ARGS[@]}"

echo ""
echo "=== Filter done -> $OUT ==="
if [ -n "$DUMP_UNCOVERED" ] && [ -s "$DUMP_UNCOVERED" ]; then
    echo "=== $(wc -l < "$DUMP_UNCOVERED") uncovered prompts -> $DUMP_UNCOVERED ==="
    echo "=== Backfill: INDICES=$DUMP_UNCOVERED bash train/rft/gold_traces.sh, then re-run this filter with the gold file added to TRACES ==="
fi
echo "=== Next: RFT_TRACES=$OUT bash train/train_07_selfadaptive.sh ==="
