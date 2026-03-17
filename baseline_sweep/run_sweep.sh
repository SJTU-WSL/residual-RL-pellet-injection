#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate Nuclear

RUNTIME_CACHE_DIR="${PROJECT_ROOT}/.runtime_cache"
MPL_CACHE_DIR="${RUNTIME_CACHE_DIR}/matplotlib"
JAX_CACHE_DIR="${RUNTIME_CACHE_DIR}/jax_cache"

mkdir -p "${MPL_CACHE_DIR}" "${JAX_CACHE_DIR}"

export MPLCONFIGDIR="${MPL_CACHE_DIR}"
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${PROJECT_ROOT}"

# ── Phase 1: Run grid search (CPU) ──
python3 baseline_sweep/sweep_eval.py \
  --torax-config config/ITER.py \
  --batch-size 16 \
  --warmup-steps 2000 \
  --control-steps 8000 \
  --device cpu \
  --seed 0 \
  --log-dir baseline_sweep/eval_logs \
  --log-interval 500 \
  --resume

# ── Phase 2: Visualize results ──
python3 baseline_sweep/sweep_visualize.py \
  --csv baseline_sweep/eval_logs/sweep_summary.csv \
  --output-dir baseline_sweep/eval_logs \
  --top-n 20
