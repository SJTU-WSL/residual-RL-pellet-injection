#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate Nuclear

RUN_NAME="residual_interval_baseline_10s_cpu_$(date +%Y%m%d_%H%M%S)"

cd "${PROJECT_ROOT}"

python3 rl_residual_lab/eval_residual.py \
  --torax-config config/ITER.py \
  --device cuda:0 \
  --seed 42 \
  --batch-size 1 \
  --max-steps 10000 \
  --eval-steps 10000 \
  --warmup-steps 2000 \
  --inject-every 130 \
  --inject-duration 1 \
  --baseline-velocity 300 \
  --baseline-thickness-mm 2.0 \
  --residual-velocity-max 50.0 \
  --residual-thickness-mm-max 0.5 \
  --append-schedule-features \
  --normalize-actions \
  --print-every 200 \
  --save-dir eval_logs \
  --run-name "${RUN_NAME}"
