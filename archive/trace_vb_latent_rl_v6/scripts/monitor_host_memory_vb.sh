#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <tmux-session> <run-tag> <run-directory>" >&2
  exit 2
fi

session_name=$1
run_tag=$2
run_directory=$3
interval_seconds=${TRACE_VB_HOST_WATCH_INTERVAL_SECONDS:-10}
maximum_rank_rss_gib=${TRACE_VB_MAXIMUM_RANK_RSS_GIB:-20}
minimum_host_available_gib=${TRACE_VB_MINIMUM_HOST_AVAILABLE_GIB:-192}

[[ "${interval_seconds}" =~ ^[1-9][0-9]*$ ]] || {
  echo "interval must be a positive integer" >&2
  exit 2
}
[[ "${maximum_rank_rss_gib}" =~ ^[1-9][0-9]*$ ]] || {
  echo "maximum rank RSS must be a positive integer GiB value" >&2
  exit 2
}
[[ "${minimum_host_available_gib}" =~ ^[1-9][0-9]*$ ]] || {
  echo "minimum host available memory must be a positive integer GiB value" >&2
  exit 2
}

maximum_rank_rss_kib=$((maximum_rank_rss_gib * 1024 * 1024))
minimum_host_available_kib=$((minimum_host_available_gib * 1024 * 1024))
mkdir -p "${run_directory}"
watch_log="${run_directory}/host_memory_watchdog.tsv"
if [[ ! -s "${watch_log}" ]]; then
  printf 'timestamp\tstatus\tmax_rank_rss_gib\ttotal_rank_rss_gib\thost_available_gib\tpids\n'     > "${watch_log}"
fi

while tmux has-session -t "${session_name}" 2>/dev/null; do
  mapfile -t rank_pids < <(
    pgrep -f "[r]un.py.*--log_suffix ${run_tag}" || true
  )
  if [[ "${#rank_pids[@]}" -eq 0 ]]; then
    sleep "${interval_seconds}"
    continue
  fi

  maximum_rss_kib=0
  total_rss_kib=0
  live_pids=()
  for pid in "${rank_pids[@]}"; do
    [[ -r "/proc/${pid}/status" ]] || continue
    rss_kib=$(awk '/^VmRSS:/ {print $2}' "/proc/${pid}/status")
    [[ -n "${rss_kib}" ]] || continue
    live_pids+=("${pid}")
    total_rss_kib=$((total_rss_kib + rss_kib))
    if (( rss_kib > maximum_rss_kib )); then
      maximum_rss_kib=${rss_kib}
    fi
  done
  host_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  timestamp=$(date --iso-8601=seconds)
  pid_csv=$(IFS=,; echo "${live_pids[*]}")
  awk -v timestamp="${timestamp}"       -v status="OK"       -v max_kib="${maximum_rss_kib}"       -v total_kib="${total_rss_kib}"       -v available_kib="${host_available_kib}"       -v pids="${pid_csv}"       'BEGIN {
        printf "%s\t%s\t%.3f\t%.3f\t%.3f\t%s\n",
          timestamp, status,
          max_kib / 1048576.0,
          total_kib / 1048576.0,
          available_kib / 1048576.0,
          pids
      }' >> "${watch_log}"

  if ((
    maximum_rss_kib >= maximum_rank_rss_kib
    || host_available_kib <= minimum_host_available_kib
  )); then
    reason="max_rss_kib=${maximum_rss_kib},host_available_kib=${host_available_kib}"
    printf '%s\tCONTROLLED_STOP\t%s\n' "${timestamp}" "${reason}"       >> "${watch_log}"
    tmux send-keys -t "${session_name}" C-c
    sleep 5
    for pid in "${live_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}"
      fi
    done
    exit 99
  fi
  sleep "${interval_seconds}"
done

printf '%s\tSESSION_FINISHED\n' "$(date --iso-8601=seconds)"   >> "${watch_log}"

