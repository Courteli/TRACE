#!/usr/bin/env bash
set -uo pipefail

SOURCE_ROOT=/home/dingxukai/TRACE/trace_role_bridge_native_v9
ROT_PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
RUN_ROOT=${1:?usage: wait_for_current_run_strict_validation.sh RUN_ROOT}
STATE_ROOT=${RUN_ROOT}/state
LOG_FILE=${RUN_ROOT}/strict_validation_supervisor.log
LOCK_FILE=/tmp/trace_role_native_v9_strict_validation.lock
SCAN_SECONDS=${STRICT_VALIDATION_SCAN_SECONDS:-30}

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another strict native-v9 validation supervisor is active." >&2
  exit 3
fi

while true; do
  if [[ -f "${STATE_ROOT}/strict_evaluation.done" ]]; then
    exit 0
  fi

  # The already-running pipeline was launched before the strict evaluator was
  # installed.  Wait for its four-card training/evaluation process to release
  # the GPUs, then append the exact 747-question evidence without disturbing
  # any Stage0/1/2 checkpoint or completion marker.
  if [[ ! -f "${STATE_ROOT}/complete.done" ]]; then
    sleep "${SCAN_SECONDS}"
    continue
  fi
  if [[ ! -s "${STATE_ROOT}/stage2_best.txt" ]]; then
    echo "$(date '+%F %T') complete.done exists without stage2_best.txt" \
      >> "${LOG_FILE}"
    sleep "${SCAN_SECONDS}"
    continue
  fi

  GPU_STATE=$(nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>>"${LOG_FILE}") || {
      sleep "${SCAN_SECONDS}"
      continue
    }
  FREE_GPU=$(printf '%s\n' "${GPU_STATE}" | awk -F',' \
    '{gsub(/ /, "", $0); if ($2 < 5000 && $3 < 25) {print $1; exit}}')
  if [[ -z "${FREE_GPU}" ]]; then
    sleep "${SCAN_SECONDS}"
    continue
  fi

  STAGE2_CKPT=$(<"${STATE_ROOT}/stage2_best.txt")
  if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "$(date '+%F %T') missing Stage2 checkpoint ${STAGE2_CKPT}" \
      >> "${LOG_FILE}"
    sleep "${SCAN_SECONDS}"
    continue
  fi

  echo "$(date '+%F %T') strict validation starting on GPU ${FREE_GPU}" \
    | tee -a "${LOG_FILE}"
  if CUDA_VISIBLE_DEVICES=${FREE_GPU} "${ROT_PYTHON}" \
      "${SOURCE_ROOT}/tools/run_native_v9_strict_validation.py" \
      --model-root "${RUN_ROOT}/code_snapshot" \
      --run-root "${RUN_ROOT}" \
      --checkpoint "${STAGE2_CKPT}" \
      --output-dir "${RUN_ROOT}/strict_validation" \
      --replications 5 \
      --seed 1701 \
      --expected-questions 747 \
      >> "${LOG_FILE}" 2>&1; then
    [[ -s "${RUN_ROOT}/strict_validation/strict_validation_summary.json" ]] || {
      echo "$(date '+%F %T') evaluator returned without summary" \
        >> "${LOG_FILE}"
      sleep "${SCAN_SECONDS}"
      continue
    }
    touch "${STATE_ROOT}/strict_evaluation.done"
    echo "$(date '+%F %T') strict 747x5 validation complete" \
      | tee -a "${LOG_FILE}"
    exit 0
  fi
  echo "$(date '+%F %T') strict validation failed; retrying when a GPU is free" \
    | tee -a "${LOG_FILE}"
  sleep "${SCAN_SECONDS}"
done
