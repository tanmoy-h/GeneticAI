#!/bin/bash
#SBATCH --job-name=meditron_eval_local
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=testing/logs/run_eval_meditron_local_%j.out
#SBATCH --error=testing/logs/run_eval_meditron_local_%j.err

## Baseline evaluation of the 290 held-out records (test + val) with Meditron.
## Runs local HuggingFace inference — no API key required.
##
## Usage:
##   bash testing/run_eval_meditron_local.sh
##
##   # Only test split:
##   SPLITS="test" bash testing/run_eval_meditron_local.sh
##
##   # Resume an interrupted run:
##   RESUME=1 bash testing/run_eval_meditron_local.sh
##
##   # Use the 70B variant (needs multi-GPU or CPU offload):
##   MODEL=epfl-llm/meditron-70b bash testing/run_eval_meditron_local.sh
##
##   # SLURM:
##   sbatch testing/run_eval_meditron_local.sh

## ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV=dna_env
KEGG_CSV=${KEGG_CSV:-genomorph/dataset/global_stage1_anon_genes_mol_keep_chr.csv}
SPLITS=${SPLITS:-"test val"}
MODEL=${MODEL:-epfl-llm/meditron-7b}
CACHE_DIR=${CACHE_DIR:-~/.cache/huggingface}
DNA_TRUNCATE=${DNA_TRUNCATE:-500}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
DTYPE=${DTYPE:-bfloat16}
## ─────────────────────────────────────────────────────────────────────────────

module load MLDL/miniconda3 2>/dev/null || true
module load cuda/12.8        2>/dev/null || true
conda activate $CONDA_ENV
cd "$(dirname "$0")/.."
mkdir -p testing/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_CSV=testing/logs/meditron_local_${TIMESTAMP}.csv
LOG=testing/logs/run_eval_meditron_local_${TIMESTAMP}.log

exec > >(tee "$LOG") 2>&1
echo "Command:       bash $0 $*"
echo "Logging to:    $LOG"
echo "Model:         $MODEL"
echo "Splits:        $SPLITS"
echo "CSV:           $KEGG_CSV"
echo "DNA truncate:  ${DNA_TRUNCATE} bp"
echo "dtype:         $DTYPE"
echo "Output CSV:    $OUT_CSV"
nvidia-smi 2>/dev/null || true

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    LAST_CSV=$(ls -t testing/logs/meditron_local_*.csv 2>/dev/null | head -1)
    if [ -n "$LAST_CSV" ]; then
        OUT_CSV="$LAST_CSV"
        RESUME_FLAG="--resume"
        echo "Resuming from: $OUT_CSV"
    else
        echo "WARNING: RESUME=1 but no existing CSV found — starting fresh"
    fi
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
    $RESUME_FLAG

echo ""
echo "=== Done. Results: ==="
echo "  CSV:     $OUT_CSV"
echo "  Metrics: ${OUT_CSV%.csv}_metrics.json"
