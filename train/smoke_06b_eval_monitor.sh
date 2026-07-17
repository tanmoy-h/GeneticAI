#!/bin/bash
#SBATCH --job-name=smoke_06b_eval_monitor
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=train/logs/smoke_06b_eval_monitor_%j.out
#SBATCH --error=train/logs/smoke_06b_eval_monitor_%j.err

## ─────────────────────────────────────────────────────────────────────────────
## Smoke test for the Stage 3 GRPO training eval monitor.
##
## Purpose: confirm ThinkingResidualGRPOTrainer.evaluate() runs end-to-end and
## prints the Accuracy / Precision / Recall / F1 / Mean-time block BEFORE
## committing to a multi-hour GRPO run — without editing the real train script.
##
## What it does:
##   - copies the real train script to a temp file
##   - shrinks it: num_generations 2, grad_accum 2, max_steps 2, eval_steps 1,
##     save_steps 1, max_eval_samples 8   (one eval fires in ~2-3 min on 1 GPU)
##     (grad_accum=2 keeps the GRPO generation batch divisible by num_generations)
##   - writes checkpoints to a throwaway OUTPUT_DIR, disables wandb
##   - hard-fails if any override did not apply (never runs the full config)
##
## Usage:
##   bash train/smoke_06b_eval_monitor.sh [gpu_id]
##   TARGET=train/anon/train_06b_stage3_grpo_optB.sh bash train/smoke_06b_eval_monitor.sh 0
##   TARGET=train/train_06c_stage3_grpo_no_ot.sh     bash train/smoke_06b_eval_monitor.sh 0
##
## Pass = you see the metric block with an (k/8) accuracy fraction, nonzero
## P/R/F1, and NO "sklearn metrics failed" warning and NO KeyError.
## ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU=${1:-0}
TARGET=${TARGET:-train/train_06b_stage3_grpo_optB.sh}

cd "$(dirname "$0")/.."   # repo root

if [ ! -f "$TARGET" ]; then
    echo "ERROR: target train script not found: $TARGET"
    exit 1
fi

mkdir -p train/logs
## Create the temp copy ALONGSIDE the target script (same directory), NOT in
## tmp/. The train scripts derive _PROJECT_DIR and their cd-to-repo-root from
## $0's location ("$(dirname "$0")/.." for named, "/../.." for anon). A copy run
## from tmp/ breaks that relative resolution for the anon script (two levels up
## from tmp/ overshoots the repo root), so its priority-3 STAGE1_CKPT lookup
## silently misses and falls back to best/model.pt. Same dir as the target ->
## identical $0 depth -> correct resolution.
_TDIR=$(dirname "$TARGET")
TMP=$(mktemp "$_TDIR/smoke_$(basename "$TARGET" .sh)_XXXXXX.sh")
trap 'rm -f "$TMP"' EXIT

## Shrink the real script. Anchor each sed on the flag name so only the intended
## value changes; flexible whitespace so it survives reformatting of the source.
##
## NOTE on grad_accum=2: GRPO requires the generation batch
## (per_device_train_batch_size * num_processes * gradient_accumulation_steps) to
## be divisible by num_generations. With num_generations=2 on 1 GPU and
## per_device=1, grad_accum MUST be 2 (1*1*2=2). grad_accum=1 -> 1, not divisible
## by 2 -> TRL aborts at init (ChildFailedError from the accelerate launcher).
sed -E \
    -e 's/--num_generations[[:space:]]+[0-9]+/--num_generations 2/' \
    -e 's/--gradient_accumulation_steps[[:space:]]+[0-9]+/--gradient_accumulation_steps 2/' \
    -e 's/--max_steps[[:space:]]+-?[0-9]+/--max_steps 2/' \
    -e 's/--eval_steps[[:space:]]+\$SAVE_STEPS/--eval_steps 1/' \
    -e 's/--save_steps[[:space:]]+\$SAVE_STEPS/--save_steps 1/' \
    -e 's/--max_eval_samples[[:space:]]+[0-9]+/--max_eval_samples 8/' \
    "$TARGET" > "$TMP"

## Fail loudly if any override silently missed (source drifted) — otherwise the
## smoke test would quietly launch the real multi-hour config.
_fail=0
for pat in \
    '--num_generations 2' \
    '--gradient_accumulation_steps 2' \
    '--max_steps 2' \
    '--eval_steps 1' \
    '--save_steps 1' \
    '--max_eval_samples 8' ; do
    if ! grep -qF -- "$pat" "$TMP"; then
        echo "ERROR: override not applied in $TARGET : expected '$pat'"
        _fail=1
    fi
done
[ "$_fail" -eq 0 ] || { echo "Aborting — script structure changed, update the sed patterns."; exit 1; }

## Throwaway output + no wandb so the smoke run cannot touch real checkpoints/logs.
export OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/tmp/smoke_out_$$}"
export WANDB_MODE=disabled
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo " SMOKE TEST — Stage 3 GRPO eval monitor"
echo "   Target script : $TARGET"
echo "   Temp script   : $TMP"
echo "   Overrides     : max_steps=2 eval_steps=1 max_eval_samples=8 num_gen=2 grad_accum=2"
echo "   OUTPUT_DIR    : $OUTPUT_DIR   (throwaway — safe to delete)"
echo "   wandb         : disabled"
echo "   GPU           : $GPU"
echo "======================================================================"
echo ""
echo ">>> Expect within ~2-3 min:"
echo ">>>   [OptB] Eval monitor: val+test = 290 records"
echo ">>>   Accuracy / Precision / Recall / F1 / Mean time block with (k/8)"
echo ""

bash "$TMP" "$GPU"

echo ""
echo "=== SMOKE TEST DONE — if you saw the metric block above with (k/8), the monitor works. ==="
echo "    Clean up throwaway output:  rm -rf \"$OUTPUT_DIR\""
