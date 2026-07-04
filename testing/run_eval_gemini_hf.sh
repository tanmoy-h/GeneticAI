#!/bin/bash
#SBATCH --job-name=gemini_eval_hf
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=testing/logs/run_eval_gemini_hf_%j.out
#SBATCH --error=testing/logs/run_eval_gemini_hf_%j.err

## Baseline evaluation of the 290 held-out records (test + val) with Gemini.
## Dataset: wanglab/kegg (real gene names, HuggingFace)
##
## Usage:
##   GEMINI_API_KEY=AIza... bash testing/run_eval_gemini_hf.sh
##   RESUME=1 GEMINI_API_KEY=AIza... bash testing/run_eval_gemini_hf.sh
##   sbatch testing/run_eval_gemini_hf.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
HF_DATASET="wanglab/kegg"
KEGG_CSV=${KEGG_CSV:-testing/logs/kegg_real_hf_cache.csv}
SPLITS=${SPLITS:-"test val"}
MODEL=${MODEL:-gemini-2.5-flash}
DNA_TRUNCATE=${DNA_TRUNCATE:-500}
MAX_TOKENS=${MAX_TOKENS:-512}
RPM_LIMIT=${RPM_LIMIT:-60}
LIMIT=${LIMIT:-}                      # max records to evaluate (empty = all 290)
## ─────────────────────────────────────────────────────────────────────────────

if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY is not set."
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
            # Map real-gene-name columns to the format expected by eval scripts
            if "anon_question" not in row:
                row["anon_question"] = row.get("question", "")
            if "anon_reasoning" not in row:
                row["anon_reasoning"] = row.get("reasoning", "")
            writer.writerow(row)
print(f"Saved {sum(len(s) for s in ds.values())} records to $KEGG_CSV")
PYEOF
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_CSV=testing/logs/gemini_hf_${TIMESTAMP}.csv
LOG=testing/logs/run_eval_gemini_hf_${TIMESTAMP}.log

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
    LAST_CSV=$(ls -t testing/logs/gemini_hf_*.csv 2>/dev/null | head -1)
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

python testing/eval_gemini.py \
    --csv          "$KEGG_CSV" \
    --out          "$OUT_CSV" \
    --splits       $SPLITS \
    --model        "$MODEL" \
    --dna_truncate "$DNA_TRUNCATE" \
    --max_tokens   "$MAX_TOKENS" \
    --rpm_limit    "$RPM_LIMIT" \
    $RESUME_FLAG $LIMIT_FLAG

echo ""
echo "=== Done. Results: ==="
echo "  CSV:     $OUT_CSV"
echo "  Metrics: ${OUT_CSV%.csv}_metrics.json"
