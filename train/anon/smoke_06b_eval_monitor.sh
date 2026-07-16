#!/bin/bash
#SBATCH --job-name=smoke_06b_eval_monitor_anon
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=train/anon/logs/smoke_06b_eval_monitor_%j.out
#SBATCH --error=train/anon/logs/smoke_06b_eval_monitor_%j.err

## ─────────────────────────────────────────────────────────────────────────────
## ANON smoke test for the Stage 3 GRPO training eval monitor.
##
## Thin wrapper: defaults TARGET to the anon train_06b script and delegates to
## train/smoke_06b_eval_monitor.sh (which does all the shrink/run/verify work).
## See that script's header for the full description and pass criteria.
##
## Usage:
##   bash train/anon/smoke_06b_eval_monitor.sh [gpu_id]
## ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU=${1:-0}
_REPO="$(cd "$(dirname "$0")/../.." && pwd)"

export TARGET="${TARGET:-train/anon/train_06b_stage3_grpo_optB.sh}"
exec bash "$_REPO/train/smoke_06b_eval_monitor.sh" "$GPU"
