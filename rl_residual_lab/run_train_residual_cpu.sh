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
export JAX_PLATFORMS=cpu

CMD=(
  python3 rl_residual_lab/train_residual.py
  --torax-config config/ITER.py
  --device cpu
  --seed 0
  --batch-size 4
  --max-steps 1000
  --warmup-steps 2000
  --inject-every 100
  --inject-duration 1
  --baseline-velocity 300
  --baseline-thickness-mm 2.0
  --residual-velocity-max 50.0
  --residual-thickness-mm-max 0.5
  --append-schedule-features
  --normalize-actions
  --total-timesteps 500000
  --ppo-steps 256
  --ppo-batch-size 256
  --ppo-epochs 4
  --learning-rate 1e-4
  --gamma 0.999
  --clip-range 0.2
  --ent-coef 0.001
  --initial-residual-std 0.25
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
