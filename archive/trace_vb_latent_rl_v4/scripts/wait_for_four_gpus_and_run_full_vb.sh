#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

POLL_SECONDS=${POLL_SECONDS:-30}
TRACE_VB_MAX_IDLE_UTILIZATION=${TRACE_VB_MAX_IDLE_UTILIZATION:-10}
TRAIN_SEED=${TRAIN_SEED:-0}
PIPELINE_TAG=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v4_full_seed${TRAIN_SEED}}

[[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] && (( POLL_SECONDS >= 10 )) || \
  trace_vb_die "POLL_SECONDS must be an integer of at least 10"
[[ "${TRACE_VB_MAX_IDLE_UTILIZATION}" =~ ^[0-9]+$ ]] \
  && (( TRACE_VB_MAX_IDLE_UTILIZATION <= 100 )) || \
  trace_vb_die "TRACE_VB_MAX_IDLE_UTILIZATION must be between 0 and 100"

supervisor_dir=${TRACE_VB_ARTIFACT_ROOT}/supervisor
mkdir -p "${supervisor_dir}"
exec 9>"${supervisor_dir}/.trace_vb_v4_full_pipeline.lock"
if ! flock -n 9; then
  trace_vb_die "a TRACE-VB-v4 full-pipeline waiter or run is already active"
fi

supervisor_log=${supervisor_dir}/${PIPELINE_TAG}_wait.log
exec > >(tee -a "${supervisor_log}") 2>&1
echo "$(date --iso-8601=seconds) waiting for four GPUs with at least ${TRACE_VB_MIN_FREE_GPU_MIB} MiB free and utilization at most ${TRACE_VB_MAX_IDLE_UTILIZATION}%"

last_signature=
while true; do
  eligible_gpus=()
  status_parts=()
  while IFS=',' read -r raw_index raw_free raw_utilization; do
    index=${raw_index//[[:space:]]/}
    free=${raw_free//[[:space:]]/}
    utilization=${raw_utilization//[[:space:]]/}
    status_parts+=("${index}:${free}MiB:${utilization}%")
    if (( free >= TRACE_VB_MIN_FREE_GPU_MIB )) \
      && (( utilization <= TRACE_VB_MAX_IDLE_UTILIZATION )); then
      eligible_gpus+=("${index}")
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )

  signature="eligible=${eligible_gpus[*]:-none};status=${status_parts[*]}"
  if [[ "${signature}" != "${last_signature}" ]]; then
    echo "$(date --iso-8601=seconds) ${signature}"
    last_signature=${signature}
  fi

  if (( ${#eligible_gpus[@]} >= 4 )); then
    selected=$(IFS=,; echo "${eligible_gpus[*]:0:4}")
    echo "$(date --iso-8601=seconds) launching TRACE-VB-v4 full pipeline on ${selected}"
    exec env PIPELINE_TAG="${PIPELINE_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
      bash "${SCRIPT_DIR}/run_full_pipeline_vb.sh" \
        "${selected}" "${eligible_gpus[0]}"
  fi

  sleep "${POLL_SECONDS}"
done
