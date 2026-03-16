#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate Nuclear

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${PROJECT_ROOT}/rl_logs/${RUN_TS}"
RUN_MODEL_DIR="${PROJECT_ROOT}/rl_models/${RUN_TS}"
RUNTIME_CACHE_DIR="${PROJECT_ROOT}/.runtime_cache"
MPL_CACHE_DIR="${RUNTIME_CACHE_DIR}/matplotlib"
JAX_CACHE_DIR="${RUNTIME_CACHE_DIR}/jax_cache"

mkdir -p "${RUN_LOG_DIR}" "${RUN_MODEL_DIR}" "${MPL_CACHE_DIR}" "${JAX_CACHE_DIR}"

export MPLCONFIGDIR="${MPL_CACHE_DIR}"
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE_DIR}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CMD=(
  python3 rl_full_residual_lab/train_full_residual.py
  --torax-config config/ITER.py
  --device cuda:0
  --seed 0
  --batch-size 128
  --max-steps 10000
  --warmup-steps 2000
  --sim-steps-per-rl-step 1
  --base-interval-steps 100
  --inject-duration 1
  --min-interval-steps 20
  --max-interval-steps 200
  --baseline-velocity 300
  --baseline-thickness-mm 2.0
  --residual-interval-max 80.0
  --residual-velocity-max 50.0
  --residual-thickness-mm-max 1.0
  --append-scheduler-features
  --normalize-actions
  --total-timesteps 20_000_000
  --ppo-steps 64
  --ppo-batch-size 1024
  --ppo-epochs 4
  --learning-rate 1e-4
  --gamma 0.999
  --clip-range 0.2
  --ent-coef 0.001
  --initial-residual-std 0.2
  --log-dir rl_logs
  --save-dir rl_models
  --run-name "${RUN_TS}"
  --checkpoint-freq 50000
  --num-stack 1
  --log-interval 1
)

{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'cd %q\n' "${PROJECT_ROOT}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "${RUN_LOG_DIR}/launch_command.sh"
chmod +x "${RUN_LOG_DIR}/launch_command.sh"

env | sort > "${RUN_LOG_DIR}/env.txt"

cd "${PROJECT_ROOT}"
"${CMD[@]}" 2>&1 | tee "${RUN_LOG_DIR}/train.stdout.log"
