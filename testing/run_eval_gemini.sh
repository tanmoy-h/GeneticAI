#!/bin/bash
#SBATCH --job-name=gemini_eval
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=testing/logs/run_eval_gemini_%j.out
#SBATCH --error=testing/logs/run_eval_gemini_%j.err

## Baseline evaluation of the 290 held-out records (test + val) with Gemini.
##
## Requires GEMINI_API_KEY to be set in the environment.
##
## Usage:
##   GEMINI_API_KEY=AIza... bash testing/run_eval_gemini.sh
##
##   # Only test split:
##   SPLITS="test" bash testing/run_eval_gemini.sh
##
##   # Resume an interrupted run:
##   RESUME=1 GEMINI_API_KEY=AIza... bash testing/run_eval_gemini.sh
##
##   # Different model (e.g. gemini-2.0-flash-thinking-exp or gemini-1.5-pro):
##   MODEL=gemini-1.5-pro GEMINI_API_KEY=AIza... bash testing/run_eval_gemini.sh
##
##   # SLURM (set GEMINI_API_KEY in your env before sbatch):
##   sbatch testing/run_eval_gemini.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
SPLITS=${SPLITS:-"test val"}
MODEL=${MODEL:-gemini-2.0-flash}
DNA_TRUNCATE=${DNA_TRUNCATE:-500}     # bp sent per sequence (keeps cost low)
MAX_TOKENS=${MAX_TOKENS:-512}
RPM_LIMIT=${RPM_LIMIT:-60}            # requests/min; Flash free-tier limit
## ─────────────────────────────────────────────────────────────────────────────

if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo "ERROR: GEMINI_API_KEY is not set."
    echo "Usage: GEMINI_API_KEY=AIza... bash testing/run_eval_gemini.sh"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p testing/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_CSV=testing/logs/gemini_${TIMESTAMP}.csv
LOG=testing/logs/run_eval_gemini_${TIMESTAMP}.log

exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "Model:         $MODEL"
echo "Splits:        $SPLITS"
echo "CSV:           $KEGG_CSV"
echo "DNA truncate:  ${DNA_TRUNCATE} bp"
echo "RPM limit:     $RPM_LIMIT"
echo "Output CSV:    $OUT_CSV"

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    LAST_CSV=$(ls -t testing/logs/gemini_*.csv 2>/dev/null | head -1)
    if [ -n "$LAST_CSV" ]; then
        OUT_CSV="$LAST_CSV"
        RESUME_FLAG="--resume"
        echo "Resuming from: $OUT_CSV"
    else
        echo "WARNING: RESUME=1 but no existing CSV found — starting fresh"
    fi
fi

python testing/eval_gemini.py \
    --csv          "$KEGG_CSV" \
    --out          "$OUT_CSV" \
    --splits       $SPLITS \
    --model        "$MODEL" \
    --dna_truncate "$DNA_TRUNCATE" \
    --max_tokens   "$MAX_TOKENS" \
    --rpm_limit    "$RPM_LIMIT" \
    $RESUME_FLAG

echo ""
echo "=== Done. Results: ==="
echo "  CSV:     $OUT_CSV"
echo "  Metrics: ${OUT_CSV%.csv}_metrics.json"
