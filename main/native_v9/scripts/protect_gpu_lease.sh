#!/usr/bin/env bash
set -uo pipefail

RUN_ROOT=${1:?usage: protect_gpu_lease.sh RUN_ROOT PIPELINE_PID UUID_FILE BASELINE_FILE}
PIPELINE_PID=${2:?usage: protect_gpu_lease.sh RUN_ROOT PIPELINE_PID UUID_FILE BASELINE_FILE}
UUID_FILE=${3:?usage: protect_gpu_lease.sh RUN_ROOT PIPELINE_PID UUID_FILE BASELINE_FILE}
BASELINE_FILE=${4:?usage: protect_gpu_lease.sh RUN_ROOT PIPELINE_PID UUID_FILE BASELINE_FILE}
SCAN_SECONDS=${GPU_LEASE_SCAN_SECONDS:-2}
TERM_GRACE_SECONDS=${GPU_LEASE_TERM_GRACE_SECONDS:-2}
RETRY_DENIED_SECONDS=${GPU_LEASE_RETRY_DENIED_SECONDS:-10}
LEASE_LOG=${RUN_ROOT}.gpu_lease.log
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}

declare -A PROTECTED_UUIDS=()
declare -A BASELINE_IDENTITIES=()
declare -A LAST_ATTEMPT_SECONDS=()
declare -A TERMINATED_IDENTITIES=()

while IFS=$'\t' read -r physical_index uuid; do
  [[ -n "${physical_index}" && -n "${uuid}" ]] || continue
  PROTECTED_UUIDS["${uuid}"]=${physical_index}
done < "${UUID_FILE}"

while IFS=$'\t' read -r uuid pid identity; do
  [[ -n "${uuid}" && -n "${pid}" && -n "${identity}" ]] || continue
  BASELINE_IDENTITIES["${uuid}|${pid}|${identity}"]=1
done < "${BASELINE_FILE}"

pid_identity() {
  local pid=$1
  ps -o lstart= -p "${pid}" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

pid_is_pipeline_descendant() {
  local pid=$1
  local parent
  local depth=0
  while [[ "${pid}" =~ ^[0-9]+$ ]] && (( pid > 1 )) && (( depth < 64 )); do
    if [[ "${pid}" == "${PIPELINE_PID}" ]]; then
      return 0
    fi
    parent=$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d ' ')
    [[ "${parent}" =~ ^[0-9]+$ ]] || break
    [[ "${parent}" != "${pid}" ]] || break
    pid=${parent}
    depth=$((depth + 1))
  done
  return 1
}

pid_is_still_on_protected_gpu() {
  local expected_uuid=$1
  local expected_pid=$2
  "${NVIDIA_SMI_BIN}" \
    --query-compute-apps=gpu_uuid,pid \
    --format=csv,noheader,nounits 2>/dev/null | \
    awk -F',' -v expected_uuid="${expected_uuid}" -v expected_pid="${expected_pid}" '
      {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        if ($1 == expected_uuid && $2 == expected_pid) found=1
      }
      END {exit(found ? 0 : 1)}
    '
}

log_process_details() {
  local action=$1
  local uuid=$2
  local pid=$3
  local used_memory=$4
  local physical_index=${PROTECTED_UUIDS[${uuid}]:-unknown}
  local details
  details=$(ps -o user=,pid=,ppid=,lstart=,cmd= -p "${pid}" 2>/dev/null || true)
  {
    echo "$(date '+%F %T') action=${action} physical_gpu=${physical_index} uuid=${uuid} pid=${pid} used_memory_mib=${used_memory}"
    echo "process=${details}"
  } >> "${LEASE_LOG}"
}

terminate_confirmed_invader() {
  local uuid=$1
  local pid=$2
  local identity=$3
  local used_memory=$4

  # Resolve the exact target again immediately before signalling it.  This
  # prevents PID reuse or a process that already released the protected GPU
  # from turning a stale observation into a destructive action.
  [[ "$(pid_identity "${pid}")" == "${identity}" ]] || return 2
  pid_is_still_on_protected_gpu "${uuid}" "${pid}" || return 2
  pid_is_pipeline_descendant "${pid}" && return 2

  log_process_details terminate_attempt "${uuid}" "${pid}" "${used_memory}"
  if ! kill -TERM "${pid}" 2>> "${LEASE_LOG}"; then
    log_process_details permission_denied "${uuid}" "${pid}" "${used_memory}"
    return 1
  fi

  local deadline=$((SECONDS + TERM_GRACE_SECONDS))
  while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 0.2
  done
  if kill -0 "${pid}" 2>/dev/null; then
    # Reconfirm identity after the grace period before escalating to SIGKILL.
    if [[ "$(pid_identity "${pid}")" == "${identity}" ]]; then
      kill -KILL "${pid}" 2>> "${LEASE_LOG}" || true
    fi
  fi
  log_process_details terminate_complete "${uuid}" "${pid}" "${used_memory}"
}

{
  echo "$(date '+%F %T') action=lease_guard_started pipeline_pid=${PIPELINE_PID} scan_seconds=${SCAN_SECONDS} retry_denied_seconds=${RETRY_DENIED_SECONDS}"
  echo "protected_gpu_map=$(tr '\n' ';' < "${UUID_FILE}")"
  echo "baseline_process_count=$(wc -l < "${BASELINE_FILE}")"
} >> "${LEASE_LOG}"

while kill -0 "${PIPELINE_PID}" 2>/dev/null; do
  while IFS=',' read -r uuid pid process_name used_memory; do
    uuid=$(printf '%s' "${uuid}" | xargs)
    pid=$(printf '%s' "${pid}" | xargs)
    process_name=$(printf '%s' "${process_name}" | xargs)
    used_memory=$(printf '%s' "${used_memory}" | xargs)
    [[ -n "${PROTECTED_UUIDS[${uuid}]+x}" ]] || continue
    [[ "${pid}" =~ ^[0-9]+$ ]] || continue

    identity=$(pid_identity "${pid}")
    [[ -n "${identity}" ]] || continue
    [[ -z "${BASELINE_IDENTITIES[${uuid}|${pid}|${identity}]+x}" ]] || continue
    pid_is_pipeline_descendant "${pid}" && continue

    attempt_key="${uuid}|${pid}|${identity}"
    [[ -z "${TERMINATED_IDENTITIES[${attempt_key}]+x}" ]] || continue

    # A cross-user process may reject the first signal.  Keep retrying at a
    # bounded cadence instead of permanently treating a failed attempt as
    # handled.  PID identity and GPU residency are revalidated inside the
    # termination function on every attempt.
    now=${SECONDS}
    last_attempt=${LAST_ATTEMPT_SECONDS[${attempt_key}]:--1000000}
    (( now - last_attempt >= RETRY_DENIED_SECONDS )) || continue
    if [[ -z "${LAST_ATTEMPT_SECONDS[${attempt_key}]+x}" ]]; then
      log_process_details invader_detected "${uuid}" "${pid}" "${used_memory}"
    else
      log_process_details terminate_retry "${uuid}" "${pid}" "${used_memory}"
    fi
    LAST_ATTEMPT_SECONDS["${attempt_key}"]=${now}

    terminate_confirmed_invader "${uuid}" "${pid}" "${identity}" "${used_memory}"
    terminate_status=$?
    if (( terminate_status == 0 )); then
      TERMINATED_IDENTITIES["${attempt_key}"]=1
    fi
  done < <(
    "${NVIDIA_SMI_BIN}" \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
  sleep "${SCAN_SECONDS}"
done

echo "$(date '+%F %T') action=lease_guard_stopped pipeline_pid=${PIPELINE_PID}" \
  >> "${LEASE_LOG}"
