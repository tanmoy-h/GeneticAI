#!/bin/bash
#SBATCH --job-name=gpt4o_mini_eval_anon
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=testing/logs/run_eval_gpt4o_mini_anon_%j.out
#SBATCH --error=testing/logs/run_eval_gpt4o_mini_anon_%j.err

## Baseline evaluation of the 290 held-out records (test + val) with GPT-4o-mini.
## Dataset: iit-patna-cse-ai/kegg-anon-global (anonymized, HuggingFace)
##
## Usage:
##   OPENAI_API_KEY=sk-... bash testing/run_eval_gpt4o_mini_anon.sh
##   RESUME=1 OPENAI_API_KEY=sk-... bash testing/run_eval_gpt4o_mini_anon.sh
##   sbatch testing/run_eval_gpt4o_mini_anon.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
HF_DATASET="iit-patna-cse-ai/kegg-anon-global"
KEGG_CSV=${KEGG_CSV:-testing/logs/kegg_anon_hf_cache.csv}
SPLITS=${SPLITS:-"test val"}
MODEL=${MODEL:-gpt-4o-mini}
DNA_TRUNCATE=${DNA_TRUNCATE:-500}
MAX_TOKENS=${MAX_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0}
RPM_LIMIT=${RPM_LIMIT:-500}
LIMIT=${LIMIT:-}                      # max records to evaluate (empty = all 290)
## ─────────────────────────────────────────────────────────────────────────────

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p testing/logs

## Download HF dataset once and cache as CSV
if [ ! -f "$KEGG_CSV" ]; then
    echo "Downloading $HF_DATASET → $KEGG_CSV"
    python3 - <<PYEOF
from datasets import load_dataset
import csv

ds = load_dataset("$HF_DATASET")
fieldnames = ["split", "answer", "anon_question", "anon_reasoning",
              "reference_sequence", "variant_sequence"]
split_map = {"validation": "val", "train": "train", "test": "test"}

with open("$KEGG_CSV", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for split_name, split_ds in ds.items():
        mapped = split_map.get(split_name, split_name)
        for row in split_ds:
            row = dict(row)
            row["split"] = mapped
            writer.writerow(row)
print(f"Saved {sum(len(s) for s in ds.values())} records to $KEGG_CSV")
PYEOF
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_CSV=testing/logs/gpt4o_mini_anon_${TIMESTAMP}.csv
LOG=testing/logs/run_eval_gpt4o_mini_anon_${TIMESTAMP}.log

exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "Dataset:       $HF_DATASET"
echo "Model:         $MODEL"
echo "Splits:        $SPLITS"
echo "CSV cache:     $KEGG_CSV"
echo "Output CSV:    $OUT_CSV"

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    LAST_CSV=$(ls -t testing/logs/gpt4o_mini_anon_*.csv 2>/dev/null | head -1)
    if [ -n "$LAST_CSV" ]; then
        OUT_CSV="$LAST_CSV"
        RESUME_FLAG="--resume"
        echo "Resuming from: $OUT_CSV"
    else
        echo "WARNING: RESUME=1 but no existing CSV found — starting fresh"
    fi
fi

LIMIT_FLAG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

python testing/eval_gpt4o_mini.py \
    --csv          "$KEGG_CSV" \
    --out          "$OUT_CSV" \
    --splits       $SPLITS \
    --model        "$MODEL" \
    --dna_truncate "$DNA_TRUNCATE" \
    --max_tokens   "$MAX_TOKENS" \
    --temperature  "$TEMPERATURE" \
    --rpm_limit    "$RPM_LIMIT" \
    $RESUME_FLAG
    $LIMIT_FLAG

echo ""
echo "=== Done. Results: ==="
echo "  CSV:     $OUT_CSV"
echo "  Metrics: ${OUT_CSV%.csv}_metrics.json"
