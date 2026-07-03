#!/bin/bash
#SBATCH --job-name=claude_eval_local
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=testing/logs/run_eval_claude_local_%j.out
#SBATCH --error=testing/logs/run_eval_claude_local_%j.err

## Baseline evaluation of the 290 held-out records (test + val) with Claude.
##
## Requires ANTHROPIC_API_KEY to be set in the environment.
##
## Usage:
##   ANTHROPIC_API_KEY=sk-ant-... bash testing/run_eval_claude_local.sh
##
##   # Only test split:
##   SPLITS="test" bash testing/run_eval_claude_local.sh
##
##   # Resume an interrupted run:
##   RESUME=1 ANTHROPIC_API_KEY=sk-ant-... bash testing/run_eval_claude_local.sh
##
##   # Different model (e.g. claude-opus-4-8):
##   MODEL=claude-opus-4-8 ANTHROPIC_API_KEY=sk-ant-... bash testing/run_eval_claude_local.sh
##
##   # SLURM (set ANTHROPIC_API_KEY in your env before sbatch):
##   sbatch testing/run_eval_claude_local.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
SPLITS=${SPLITS:-"test val"}
MODEL=${MODEL:-claude-haiku-4-5}
DNA_TRUNCATE=${DNA_TRUNCATE:-500}     # bp sent per sequence (keeps cost low)
MAX_TOKENS=${MAX_TOKENS:-512}
RPM_LIMIT=${RPM_LIMIT:-60}            # requests/min; Haiku standard tier limit
LIMIT=${LIMIT:-}                      # max records to evaluate (empty = all 290)
## ─────────────────────────────────────────────────────────────────────────────

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    echo "Usage: ANTHROPIC_API_KEY=sk-ant-... bash testing/run_eval_claude_local.sh"
    exit 1
fi

module load MLDL/miniconda3 2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p testing/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_CSV=testing/logs/claude_local_${TIMESTAMP}.csv
LOG=testing/logs/run_eval_claude_local_${TIMESTAMP}.log

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
    LAST_CSV=$(ls -t testing/logs/claude_local_*.csv 2>/dev/null | head -1)
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

python testing/eval_claude.py \
    --csv          "$KEGG_CSV" \
    --out          "$OUT_CSV" \
    --splits       $SPLITS \
    --model        "$MODEL" \
    --dna_truncate "$DNA_TRUNCATE" \
    --max_tokens   "$MAX_TOKENS" \
    --rpm_limit    "$RPM_LIMIT" \
    $RESUME_FLAG \
    $LIMIT_FLAG

echo ""
echo "=== Done. Results: ==="
echo "  CSV:     $OUT_CSV"
echo "  Metrics: ${OUT_CSV%.csv}_metrics.json"
