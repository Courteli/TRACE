#!/usr/bin/env bash
set -uo pipefail

RUN_ROOT=${1:?usage: watch_native_v9_runtime.sh RUN_ROOT}
SESSION_NAME=${SESSION_NAME:-trace_role_native_v9_full_20260828}
SUPERVISOR=/home/dingxukai/TRACE/trace_role_bridge_native_v9/scripts/wait_for_four_and_run.sh
SCAN_SECONDS=${WATCH_SCAN_SECONDS:-60}
RECHECK_SECONDS=${WATCH_RECHECK_SECONDS:-5}
LOG_FILE=${RUN_ROOT}.minute_watchdog.log
LOCK_FILE=/tmp/trace_role_native_v9_minute_watchdog.lock

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another native-v9 minute watchdog is already active." >&2
  exit 3
fi

stage2_is_active() {
  local pid ppid
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    ppid=$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d ' ')
    # DDP workers left behind after rank 0 exits are re-parented to PID 1.
    # They are not a live training job and must not suppress recovery.
    if [[ -n "${ppid}" && "${ppid}" != 1 ]]; then
      return 0
    fi
  done < <(
    pgrep -u "$(id -u)" -f \
      '[r]un.py --model trace_role_native_qwen3_instruct.*--log_suffix native_v9_stage2_joint_rl' \
      2>/dev/null || true
  )
  return 1
}

supervisor_is_active() {
  pgrep -u "$(id -u)" -f \
    "[w]ait_for_four_and_run.sh ${RUN_ROOT}" \
    >/dev/null 2>&1
}

workload_is_active() {
  if [[ -f "${RUN_ROOT}/state/stage2.done" ]]; then
    pgrep -u "$(id -u)" -f \
      "[r]un_full_native_v9.sh ${RUN_ROOT}|[r]un_native_v9_strict_validation.py.*${RUN_ROOT}" \
      >/dev/null 2>&1
  else
    stage2_is_active
  fi
}

start_supervisor() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    sleep 1
  fi
  tmux new-session -d -s "${SESSION_NAME}" \
    "/bin/bash -lc 'RETRY_SECONDS=10 HEALTH_SCAN_SECONDS=60 ${SUPERVISOR} ${RUN_ROOT}'"
  echo "$(date '+%F %T') action=supervisor_restarted session=${SESSION_NAME}" \
    >> "${LOG_FILE}"
}

while true; do
  if [[ -f "${RUN_ROOT}/state/complete.done" ]]; then
    echo "$(date '+%F %T') status=complete" >> "${LOG_FILE}"
    exit 0
  fi

  ACTIVE=false
  if workload_is_active; then
    ACTIVE=true
  fi
  {
    echo "$(date '+%F %T') status=scan stage2_active=${ACTIVE}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  } >> "${LOG_FILE}" 2>&1

  if [[ "${ACTIVE}" == true ]]; then
    sleep "${SCAN_SECONDS}"
    continue
  fi

  # Avoid reacting to a brief DDP respawn or stage transition.
  sleep "${RECHECK_SECONDS}"
  if workload_is_active || [[ -f "${RUN_ROOT}/state/complete.done" ]]; then
    sleep "${SCAN_SECONDS}"
    continue
  fi

  {
    echo "$(date '+%F %T') fault=stage2_process_missing"
    for file in "${RUN_ROOT}/stage2.log" "${RUN_ROOT}.failure.log"; do
      if [[ -f "${file}" ]]; then
        echo "log=${file}"
        tail -n 120 "${file}"
      fi
    done
  } >> "${LOG_FILE}" 2>&1

  # A healthy supervisor already polls GPU availability and retries failed
  # launches.  Do not replace it here: doing so would reset the consecutive
  # idle-GPU stability scans used to avoid racing newly started external jobs.
  if supervisor_is_active && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "$(date '+%F %T') status=supervisor_waiting_or_retrying" >> "${LOG_FILE}"
    sleep "${SCAN_SECONDS}"
    continue
  fi

  # Remove only a stale tmux shell before reconstructing the supervisor.
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    sleep 2
  fi
  start_supervisor
  sleep "${SCAN_SECONDS}"
done
