#!/bin/bash
#SBATCH --job-name=gpt4o_mini_eval
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=testing/logs/run_eval_gpt4o_mini_%j.out
#SBATCH --error=testing/logs/run_eval_gpt4o_mini_%j.err

## Baseline evaluation of the 290 held-out records (test + val) with GPT-4o-mini.
##
## Requires OPENAI_API_KEY to be set in the environment.
##
## Usage:
##   OPENAI_API_KEY=sk-... bash testing/run_eval_gpt4o_mini.sh
##
##   # Only test split:
##   SPLITS="test" bash testing/run_eval_gpt4o_mini.sh
##
##   # Resume an interrupted run:
##   RESUME=1 OPENAI_API_KEY=sk-... bash testing/run_eval_gpt4o_mini.sh
##
##   # SLURM (set OPENAI_API_KEY in your env before sbatch):
##   sbatch testing/run_eval_gpt4o_mini.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
SPLITS=${SPLITS:-"test val"}
MODEL=${MODEL:-gpt-4o-mini}
DNA_TRUNCATE=${DNA_TRUNCATE:-500}     # bp sent per sequence (keeps cost low)
MAX_TOKENS=${MAX_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0}
RPM_LIMIT=${RPM_LIMIT:-500}           # requests/min; gpt-4o-mini tier-1 limit
## ─────────────────────────────────────────────────────────────────────────────

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Usage: OPENAI_API_KEY=sk-... bash testing/run_eval_gpt4o_mini.sh"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p testing/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_CSV=testing/logs/gpt4o_mini_${TIMESTAMP}.csv
LOG=testing/logs/run_eval_gpt4o_mini_${TIMESTAMP}.log

exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "Model:         $MODEL"
echo "Splits:        $SPLITS"
echo "CSV:           $KEGG_CSV"
echo "DNA truncate:  ${DNA_TRUNCATE} bp"
echo "Output CSV:    $OUT_CSV"

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    LAST_CSV=$(ls -t testing/logs/gpt4o_mini_*.csv 2>/dev/null | head -1)
    if [ -n "$LAST_CSV" ]; then
        OUT_CSV="$LAST_CSV"
        RESUME_FLAG="--resume"
        echo "Resuming from: $OUT_CSV"
    else
        echo "WARNING: RESUME=1 but no existing CSV found — starting fresh"
    fi
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

echo ""
echo "=== Done. Results: ==="
echo "  CSV:     $OUT_CSV"
echo "  Metrics: ${OUT_CSV%.csv}_metrics.json"
