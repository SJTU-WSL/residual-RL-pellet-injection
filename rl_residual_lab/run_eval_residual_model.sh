#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash rl_residual_lab/run_eval_residual_model.sh <MODEL_PATH> [DEVICE]"
  exit 1
fi

MODEL_PATH="$1"
DEVICE="${2:-cpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate Nuclear

RUN_NAME="residual_model_eval_10s_$(basename "${MODEL_PATH%.zip}")_$(date +%Y%m%d_%H%M%S)"

cd "${PROJECT_ROOT}"

python3 rl_residual_lab/eval_residual.py \
  --model-path "${MODEL_PATH}" \
  --torax-config config/ITER.py \
  --device "${DEVICE}" \
  --seed 42 \
  --batch-size 1 \
  --max-steps 10000 \
  --eval-steps 10000 \
  --warmup-steps 2000 \
  --inject-every 100 \
  --inject-duration 1 \
  --baseline-velocity 300 \
  --baseline-thickness-mm 2.0 \
  --residual-velocity-max 50.0 \
  --residual-thickness-mm-max 0.5 \
  --append-schedule-features \
  --normalize-actions \
  --deterministic \
  --print-every 200 \
  --save-dir eval_logs \
  --run-name "${RUN_NAME}"
