#!/usr/bin/env bash
set -uo pipefail

PIPELINE=/home/dingxukai/TRACE/trace_role_bridge_native_v9/scripts/run_full_native_v9.sh
# Only observe contention. Never signal another task's GPU process.
LEASE_GUARD=/home/dingxukai/TRACE/trace_role_bridge_native_v9/scripts/observe_gpu_lease.py
RESERVATION_RUNNER=/home/dingxukai/TRACE/trace_role_bridge_native_v9/scripts/run_gpu_reservation_experiment.sh
RUN_ROOT=${1:-/disk1/dingxukai/TRACE/role_native_v9_$(date +%Y%m%d-%H%M%S)}
MONITOR_LOG=${RUN_ROOT}.gpu_wait.log
HEALTH_LOG=${RUN_ROOT}.runtime_health.log
FAILURE_LOG=${RUN_ROOT}.failure.log
RESERVATION_ROOT=${RUN_ROOT}.gpu_reservations
RESERVATION_STATE_DIR=${RESERVATION_ROOT}/state
RESERVATION_LOG=${RESERVATION_ROOT}/supervisor.log
LOCK_FILE=/tmp/trace_role_native_v9_four_gpu.lock
RETRY_SECONDS=${RETRY_SECONDS:-10}
WAIT_SCAN_SECONDS=${WAIT_SCAN_SECONDS:-10}
HEALTH_SCAN_SECONDS=${HEALTH_SCAN_SECONDS:-2400}
MAX_IDLE_MEMORY_MIB=${MAX_IDLE_MEMORY_MIB:-128}
MAX_IDLE_UTIL_PERCENT=${MAX_IDLE_UTIL_PERCENT:-5}
RESERVATION_STOP_TIMEOUT=${RESERVATION_STOP_TIMEOUT:-45}
NEXT_HEALTH_SECONDS=0

mkdir -p "${RESERVATION_STATE_DIR}"

pid_identity() {
  local pid=$1
  ps -o lstart= -p "${pid}" 2>/dev/null | \
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

pid_group() {
  local pid=$1
  ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' '
}

pid_matches() {
  local pid=$1
  local identity=$2
  [[ "${pid}" =~ ^[0-9]+$ ]] && \
    [[ -n "${identity}" ]] && \
    [[ "$(pid_identity "${pid}")" == "${identity}" ]]
}

guard_stop_exact() {
  local pid=$1
  local identity=$2
  if pid_matches "${pid}" "${identity}"; then
    kill -TERM "${pid}" 2>/dev/null || true
  fi
}

pid_is_descendant() {
  local pid=$1 ancestor=$2 parent depth=0
  while [[ "${pid}" =~ ^[0-9]+$ ]] && (( pid > 1 && depth < 64 )); do
    [[ "${pid}" == "${ancestor}" ]] && return 0
    parent=$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d ' ')
    [[ "${parent}" =~ ^[0-9]+$ && "${parent}" != "${pid}" ]] || return 1
    pid=${parent}
    depth=$((depth + 1))
  done
  return 1
}

health_if_due() {
  (( SECONDS >= NEXT_HEALTH_SECONDS )) || return 0
  local gpu child file
  {
    date '+%F %T'
    echo "health_interval_seconds=${HEALTH_SCAN_SECONDS} phase=${1:-waiting} pipeline_pid=${PIPELINE_PID:-none} reserved=${RESERVED_GPUS[*]:-none}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    ps -u "$(id -u)" -o pid,ppid,etime,comm
    df -h "${RUN_ROOT}"
    for file in "${RUN_ROOT}/stage2/recovery/last.ckpt" "${RUN_ROOT}/stage2.log"; do
      [[ ! -f "${file}" ]] || stat -c 'progress_file=%n size=%s modified=%y' "${file}"
    done
    for gpu in "${RESERVED_GPUS[@]}"; do
      if [[ -f "${RESERVATION_ROOT}/gpu${gpu}/child.pid" ]]; then
        child=$(<"${RESERVATION_ROOT}/gpu${gpu}/child.pid")
        echo "reservation_gpu=${gpu} child_pid=${child}"
        ps -p "${child}" -o pid,ppid,etime,comm
      fi
    done
  } >> "${HEALTH_LOG}" 2>&1
  NEXT_HEALTH_SECONDS=$((SECONDS + HEALTH_SCAN_SECONDS))
}

reservations_are_ready() {
  local index gpu uuid table row_uuid pid found
  table=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits) || return 1
  for index in "${!RESERVED_GPUS[@]}"; do
    gpu=${RESERVED_GPUS[${index}]}
    uuid=$(awk -F'\t' 'NR == 1 {print $2}' "${RESERVATION_ROOT}/gpu${gpu}/selected_gpu_uuids.tsv")
    found=false
    while IFS=',' read -r row_uuid pid; do
      row_uuid=${row_uuid// /}
      pid=${pid// /}
      [[ "${row_uuid}" == "${uuid}" ]] || continue
      pid_is_descendant "${pid}" "${RESERVED_RUNNER_PIDS[${index}]}" || return 1
      found=true
    done <<< "${table}"
    [[ "${found}" == true ]] || return 1
  done
}

snapshot_one_gpu() {
  local physical_gpu=$1
  local uuid_file=$2
  local baseline_file=$3
  local uuid pid identity

  : > "${uuid_file}"
  : > "${baseline_file}"
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | \
    awk -F',' -v gpu="${physical_gpu}" '
      {gsub(/ /, "", $1); gsub(/ /, "", $2); if ($1 == gpu) print $2}
    ')
  [[ -n "${uuid}" ]] || return 1
  printf '%s\t%s\n' "${physical_gpu}" "${uuid}" > "${uuid_file}"

  while IFS=',' read -r process_uuid pid _process_name _used_memory; do
    process_uuid=$(printf '%s' "${process_uuid}" | xargs)
    pid=$(printf '%s' "${pid}" | xargs)
    [[ "${process_uuid}" == "${uuid}" && "${pid}" =~ ^[0-9]+$ ]] || continue
    identity=$(pid_identity "${pid}")
    [[ -n "${identity}" ]] || continue
    printf '%s\t%s\t%s\n' "${process_uuid}" "${pid}" "${identity}" \
      >> "${baseline_file}"
  done < <(
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
}

write_reservation_state() {
  local physical_gpu=$1
  local runner_pid=$2
  local uuid_file=$3
  local baseline_file=$4
  local runner_identity runner_pgid guard_pid guard_identity uuid state_file temp_file

  runner_identity=$(pid_identity "${runner_pid}")
  runner_pgid=$(pid_group "${runner_pid}")
  [[ -n "${runner_identity}" && "${runner_pgid}" =~ ^[0-9]+$ ]] || return 1
  uuid=$(awk -F'\t' 'NR == 1 {print $2}' "${uuid_file}")
  [[ -n "${uuid}" ]] || return 1

  nohup env GPU_LEASE_SCAN_SECONDS=2 "${LEASE_GUARD}" \
    "${RESERVATION_ROOT}/gpu${physical_gpu}" "${runner_pid}" \
    "${uuid_file}" "${baseline_file}" 9>&- >/dev/null 2>&1 &
  guard_pid=$!
  guard_identity=$(pid_identity "${guard_pid}")
  [[ -n "${guard_identity}" ]] || return 1

  state_file=${RESERVATION_STATE_DIR}/gpu${physical_gpu}.tsv
  temp_file=${state_file}.tmp.$$
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${physical_gpu}" "${uuid}" "${runner_pid}" "${runner_pgid}" \
    "${runner_identity}" "${guard_pid}" "${guard_identity}" > "${temp_file}"
  mv "${temp_file}" "${state_file}"
}

start_reservation() {
  local physical_gpu=$1 gpu_pids
  local gpu_line memory_used utilization runner_pid reserve_dir uuid_file baseline_file

  # Reconfirm immediately before launch so an old scan cannot claim a card
  # that another job took in the meantime.
  gpu_line=$(nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | \
    awk -F',' -v gpu="${physical_gpu}" '
      {gsub(/ /, "", $0); if ($1 == gpu) print $0}
    ')
  IFS=',' read -r _ memory_used utilization <<< "${gpu_line}"
  [[ "${memory_used}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || return 1
  (( memory_used < MAX_IDLE_MEMORY_MIB && utilization < MAX_IDLE_UTIL_PERCENT )) || return 1
  gpu_pids=$(nvidia-smi -i "${physical_gpu}" --query-compute-apps=pid --format=csv,noheader,nounits) || return 1
  [[ -z "${gpu_pids}" ]] || return 1

  reserve_dir=${RESERVATION_ROOT}/gpu${physical_gpu}
  uuid_file=${reserve_dir}/selected_gpu_uuids.tsv
  baseline_file=${reserve_dir}/baseline_compute_processes.tsv
  mkdir -p "${reserve_dir}"
  snapshot_one_gpu "${physical_gpu}" "${uuid_file}" "${baseline_file}" || return 1

  nohup setsid "${RESERVATION_RUNNER}" "${RUN_ROOT}" "${physical_gpu}" \
    9>&- >/dev/null 2>&1 &
  runner_pid=$!
  sleep 0.2 9>&-
  if ! pid_matches "${runner_pid}" "$(pid_identity "${runner_pid}")"; then
    return 1
  fi
  if ! write_reservation_state \
    "${physical_gpu}" "${runner_pid}" "${uuid_file}" "${baseline_file}"; then
    kill -TERM "${runner_pid}" 2>/dev/null || true
    return 1
  fi
  echo "$(date '+%F %T') action=reservation_started physical_gpu=${physical_gpu} runner_pid=${runner_pid}" \
    | tee -a "${MONITOR_LOG}" "${RESERVATION_LOG}"
}

mark_stale_state() {
  local state_file=$1
  local guard_pid=${2:-}
  local guard_identity=${3:-}
  guard_stop_exact "${guard_pid}" "${guard_identity}"
  mv "${state_file}" "${state_file}.stale.$(date +%Y%m%d-%H%M%S)"
}

refresh_reservations() {
  local physical_gpu uuid runner_pid runner_pgid runner_identity guard_pid guard_identity state_file
  RESERVED_GPUS=()
  RESERVED_RUNNER_PIDS=()
  RESERVED_RUNNER_PGIDS=()
  RESERVED_RUNNER_IDENTITIES=()
  RESERVED_GUARD_PIDS=()
  RESERVED_GUARD_IDENTITIES=()

  for physical_gpu in {0..7}; do
    state_file=${RESERVATION_STATE_DIR}/gpu${physical_gpu}.tsv
    [[ -f "${state_file}" ]] || continue
    IFS=$'\t' read -r physical_gpu uuid runner_pid runner_pgid runner_identity guard_pid guard_identity \
      < "${state_file}"
    if ! pid_matches "${runner_pid}" "${runner_identity}"; then
      mark_stale_state "${state_file}" "${guard_pid}" "${guard_identity}"
      continue
    fi
    if ! pid_matches "${guard_pid}" "${guard_identity}"; then
      if write_reservation_state \
        "${physical_gpu}" "${runner_pid}" \
        "${RESERVATION_ROOT}/gpu${physical_gpu}/selected_gpu_uuids.tsv" \
        "${RESERVATION_ROOT}/gpu${physical_gpu}/baseline_compute_processes.tsv"; then
        IFS=$'\t' read -r physical_gpu uuid runner_pid runner_pgid runner_identity guard_pid guard_identity \
          < "${state_file}"
        echo "$(date '+%F %T') action=reservation_guard_restarted physical_gpu=${physical_gpu} guard_pid=${guard_pid}" \
          | tee -a "${MONITOR_LOG}" "${RESERVATION_LOG}"
      else
        echo "$(date '+%F %T') action=reservation_guard_restart_failed physical_gpu=${physical_gpu}" \
          | tee -a "${MONITOR_LOG}" "${RESERVATION_LOG}"
      fi
    fi
    RESERVED_GPUS+=("${physical_gpu}")
    RESERVED_RUNNER_PIDS+=("${runner_pid}")
    RESERVED_RUNNER_PGIDS+=("${runner_pgid}")
    RESERVED_RUNNER_IDENTITIES+=("${runner_identity}")
    RESERVED_GUARD_PIDS+=("${guard_pid}")
    RESERVED_GUARD_IDENTITIES+=("${guard_identity}")
  done
}

adopt_untracked_reservations() {
  local physical_gpu reserve_dir runner_pid runner_identity state_file uuid_file baseline_file
  for physical_gpu in {0..7}; do
    state_file=${RESERVATION_STATE_DIR}/gpu${physical_gpu}.tsv
    [[ -f "${state_file}" ]] && continue
    reserve_dir=${RESERVATION_ROOT}/gpu${physical_gpu}
    [[ -s "${reserve_dir}/runner.pid" && -s "${reserve_dir}/runner.identity" ]] || continue
    runner_pid=$(<"${reserve_dir}/runner.pid")
    runner_identity=$(<"${reserve_dir}/runner.identity")
    pid_matches "${runner_pid}" "${runner_identity}" || continue
    uuid_file=${reserve_dir}/selected_gpu_uuids.tsv
    baseline_file=${reserve_dir}/baseline_compute_processes.tsv
    snapshot_one_gpu "${physical_gpu}" "${uuid_file}" "${baseline_file}" || continue
    if write_reservation_state \
      "${physical_gpu}" "${runner_pid}" "${uuid_file}" "${baseline_file}"; then
      echo "$(date '+%F %T') action=reservation_adopted physical_gpu=${physical_gpu} runner_pid=${runner_pid}" \
        | tee -a "${MONITOR_LOG}" "${RESERVATION_LOG}"
    fi
  done
}

gpu_is_reserved() {
  local wanted=$1
  local physical_gpu
  for physical_gpu in "${RESERVED_GPUS[@]}"; do
    [[ "${physical_gpu}" == "${wanted}" ]] && return 0
  done
  return 1
}

stop_reservations_for_handoff() {
  local index pid pgid identity guard_pid guard_identity state_file deadline

  echo "$(date '+%F %T') action=reservation_handoff_start physical_gpus=${SELECTED}" \
    | tee -a "${MONITOR_LOG}" "${RESERVATION_LOG}"
  for index in "${!RESERVED_GPUS[@]}"; do
    pid=${RESERVED_RUNNER_PIDS[${index}]}
    pgid=${RESERVED_RUNNER_PGIDS[${index}]}
    identity=${RESERVED_RUNNER_IDENTITIES[${index}]}
    if pid_matches "${pid}" "${identity}" && [[ "$(pid_group "${pid}")" == "${pgid}" ]]; then
      kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  deadline=$((SECONDS + RESERVATION_STOP_TIMEOUT))
  while (( SECONDS < deadline )); do
    local any_alive=false
    for index in "${!RESERVED_GPUS[@]}"; do
      if pid_matches "${RESERVED_RUNNER_PIDS[${index}]}" "${RESERVED_RUNNER_IDENTITIES[${index}]}"; then
        any_alive=true
        break
      fi
    done
    [[ "${any_alive}" == false ]] && break
    sleep 0.5 9>&-
  done

  for index in "${!RESERVED_GPUS[@]}"; do
    pid=${RESERVED_RUNNER_PIDS[${index}]}
    pgid=${RESERVED_RUNNER_PGIDS[${index}]}
    identity=${RESERVED_RUNNER_IDENTITIES[${index}]}
    guard_pid=${RESERVED_GUARD_PIDS[${index}]}
    guard_identity=${RESERVED_GUARD_IDENTITIES[${index}]}
    if pid_matches "${pid}" "${identity}" && [[ "$(pid_group "${pid}")" == "${pgid}" ]]; then
      kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
    guard_stop_exact "${guard_pid}" "${guard_identity}"
    state_file=${RESERVATION_STATE_DIR}/gpu${RESERVED_GPUS[${index}]}.tsv
    if [[ -f "${state_file}" ]]; then
      mv "${state_file}" "${state_file}.handed_over.$(date +%Y%m%d-%H%M%S)"
    fi
  done
  echo "$(date '+%F %T') action=reservation_handoff_complete physical_gpus=${SELECTED}" \
    | tee -a "${MONITOR_LOG}" "${RESERVATION_LOG}"
}

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another native-v9 four-GPU supervisor is already active." >&2
  exit 3
fi

while true; do
  if [[ -f "${RUN_ROOT}/state/complete.done" ]]; then
    echo "$(date '+%F %T') native v9 is complete" | tee -a "${MONITOR_LOG}"
    exit 0
  fi

  # Keep every card already acquired by this run, including reservation
  # experiments that survived a supervisor restart. The first launch after a
  # code update may also adopt a manually started reservation.
  refresh_reservations
  adopt_untracked_reservations
  refresh_reservations
  health_if_due waiting

  GPU_STATE=$(nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)
  printf '%s reserved=%s\n%s\n' \
    "$(date '+%F %T')" "${RESERVED_GPUS[*]:-none}" "${GPU_STATE}" \
    >> "${MONITOR_LOG}"
  mapfile -t FREE_GPUS < <(
    printf '%s\n' "${GPU_STATE}" | \
      awk -F',' \
        -v max_mem="${MAX_IDLE_MEMORY_MIB}" \
        -v max_util="${MAX_IDLE_UTIL_PERCENT}" \
        '{gsub(/ /, "", $0); if ($2 < max_mem && $3 < max_util) print $1}'
  )

  # Acquire cards one at a time. A real, isolated single-GPU Stage-2
  # experiment occupies each card continuously while the supervisor waits for
  # the remaining cards.
  if (( ${#RESERVED_GPUS[@]} < 4 )); then
    for physical_gpu in "${FREE_GPUS[@]}"; do
      gpu_is_reserved "${physical_gpu}" && continue
      start_reservation "${physical_gpu}" || continue
      refresh_reservations
      (( ${#RESERVED_GPUS[@]} >= 4 )) && break
    done
  fi

  if (( ${#RESERVED_GPUS[@]} < 4 )); then
    sleep "${WAIT_SCAN_SECONDS}" 9>&-
    continue
  fi

  SELECTED=$(IFS=,; echo "${RESERVED_GPUS[*]:0:4}")
  # Do not stop the first three jobs until CUDA is really running on all
  # four cards and every resident process belongs to our reservation jobs.
  if ! reservations_are_ready; then
    echo "$(date '+%F %T') action=waiting_for_four_owned_cuda_processes" >> "${MONITOR_LOG}"
    sleep "${WAIT_SCAN_SECONDS}" 9>&-
    continue
  fi
  stop_reservations_for_handoff
  echo "$(date '+%F %T') launching/resuming on physical GPUs ${SELECTED}" \
    | tee -a "${MONITOR_LOG}"

  LEASE_DIR=${RUN_ROOT}.gpu_lease
  UUID_FILE=${LEASE_DIR}/selected_gpu_uuids.tsv
  BASELINE_FILE=${LEASE_DIR}/baseline_compute_processes.tsv
  mkdir -p "${LEASE_DIR}"
  : > "${UUID_FILE}"
  : > "${BASELINE_FILE}"
  while IFS=',' read -r physical_index uuid; do
    physical_index=$(printf '%s' "${physical_index}" | xargs)
    uuid=$(printf '%s' "${uuid}" | xargs)
    if [[ ",${SELECTED}," == *",${physical_index},"* ]]; then
      printf '%s\t%s\n' "${physical_index}" "${uuid}" >> "${UUID_FILE}"
    fi
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)
  if [[ $(wc -l < "${UUID_FILE}") -ne 4 ]]; then
    echo "$(date '+%F %T') lease snapshot failed: expected 4 selected GPU UUIDs" \
      | tee -a "${MONITOR_LOG}"
    sleep "${RETRY_SECONDS}" 9>&-
    continue
  fi
  while IFS=',' read -r uuid pid _process_name _used_memory; do
    uuid=$(printf '%s' "${uuid}" | xargs)
    pid=$(printf '%s' "${pid}" | xargs)
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue
    if awk -F'\t' -v uuid="${uuid}" '$2 == uuid {found=1} END {exit(found ? 0 : 1)}' "${UUID_FILE}"; then
      identity=$(ps -o lstart= -p "${pid}" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
      [[ -n "${identity}" ]] || continue
      printf '%s\t%s\t%s\n' "${uuid}" "${pid}" "${identity}" >> "${BASELINE_FILE}"
    fi
  done < <(
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true
  )

  PHYSICAL_GPUS=${SELECTED} "${PIPELINE}" "${RUN_ROOT}" 9>&- &
  PIPELINE_PID=$!
  GPU_LEASE_SCAN_SECONDS=2 "${LEASE_GUARD}" \
    "${RUN_ROOT}" "${PIPELINE_PID}" "${UUID_FILE}" "${BASELINE_FILE}" 9>&- &
  LEASE_GUARD_PID=$!
  while kill -0 "${PIPELINE_PID}" 2>/dev/null; do
    health_if_due training
    sleep "${WAIT_SCAN_SECONDS}" 9>&-
  done
  kill "${LEASE_GUARD_PID}" 2>/dev/null || true
  wait "${LEASE_GUARD_PID}" 2>/dev/null || true
  if wait "${PIPELINE_PID}"; then
    echo "$(date '+%F %T') pipeline completed successfully" | tee -a "${MONITOR_LOG}"
    exit 0
  else
    STATUS=$?
    {
      echo "$(date '+%F %T') pipeline failed status=${STATUS}; preserving checkpoints and retrying"
      for LOG in "${RUN_ROOT}"/stage0.log "${RUN_ROOT}"/stage1.log "${RUN_ROOT}"/stage2.log "${RUN_ROOT}"/stage2_evaluation.log; do
        if [[ -f "${LOG}" ]]; then
          echo "log=${LOG}"
          tail -n 80 "${LOG}"
        fi
      done
    } >> "${FAILURE_LOG}" 2>&1
    echo "$(date '+%F %T') failure captured; next retry in ${RETRY_SECONDS}s" | tee -a "${MONITOR_LOG}"
    sleep "${RETRY_SECONDS}" 9>&-
  fi
done
