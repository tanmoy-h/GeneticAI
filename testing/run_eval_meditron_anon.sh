#!/bin/bash
#SBATCH --job-name=meditron_eval_anon
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=testing/logs/run_eval_meditron_anon_%j.out
#SBATCH --error=testing/logs/run_eval_meditron_anon_%j.err

## Baseline evaluation of the 290 held-out records (test + val) with Meditron.
## Dataset: iit-patna-cse-ai/kegg-anon-global (anonymized, HuggingFace)
##
## Usage:
##   bash testing/run_eval_meditron_anon.sh
##   RESUME=1 bash testing/run_eval_meditron_anon.sh
##   MODEL=epfl-llm/meditron-70b bash testing/run_eval_meditron_anon.sh
##   sbatch testing/run_eval_meditron_anon.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
HF_DATASET="iit-patna-cse-ai/kegg-anon-global"
KEGG_CSV=${KEGG_CSV:-testing/logs/kegg_anon_hf_cache.csv}
SPLITS=${SPLITS:-"test val"}
MODEL=${MODEL:-epfl-llm/meditron-7b}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
DNA_TRUNCATE=${DNA_TRUNCATE:-500}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
DTYPE=${DTYPE:-bfloat16}
LIMIT=${LIMIT:-}                      # max records to evaluate (empty = all 290)
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
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
            # upload_anon_dataset.py renames anon_question->question, anon_reasoning
            # ->reasoning on push (drop-in replacement for wanglab/kegg) -- map back.
            row["anon_question"]  = row.get("question", "")
            row["anon_reasoning"] = row.get("reasoning", "")
            writer.writerow(row)
print(f"Saved {sum(len(s) for s in ds.values())} records to $KEGG_CSV")
PYEOF
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_CSV=testing/logs/meditron_anon_${TIMESTAMP}.csv
LOG=testing/logs/run_eval_meditron_anon_${TIMESTAMP}.log

exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "Dataset:       $HF_DATASET"
echo "Model:         $MODEL"
echo "Splits:        $SPLITS"
echo "CSV cache:     $KEGG_CSV"
echo "dtype:         $DTYPE"
echo "Output CSV:    $OUT_CSV"
nvidia-smi 2>/dev/null || true

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    LAST_CSV=$(ls -t testing/logs/meditron_anon_*.csv 2>/dev/null | head -1)
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

python testing/eval_meditron.py \
    --csv            "$KEGG_CSV" \
    --out            "$OUT_CSV" \
    --splits         $SPLITS \
    --model          "$MODEL" \
    --cache_dir      "$CACHE_DIR" \
    --dtype          "$DTYPE" \
    --dna_truncate   "$DNA_TRUNCATE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    $RESUME_FLAG $LIMIT_FLAG

echo ""
echo "=== Done. Results: ==="
echo "  CSV:     $OUT_CSV"
echo "  Metrics: ${OUT_CSV%.csv}_metrics.json"
