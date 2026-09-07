#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

POLL_SECONDS=${POLL_SECONDS:-30}
TRAIN_SEED=${TRAIN_SEED:-0}
PIPELINE_TAG=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v8_any4_seed${TRAIN_SEED}}
SCAN_GPUS=${TRACE_VB_SCAN_GPUS:-0,1,2,3,4,5,6,7}

trace_vb_require_safe_tag "${PIPELINE_TAG}"
[[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] && (( POLL_SECONDS >= 10 )) || \
  trace_vb_die "POLL_SECONDS must be an integer of at least 10"
[[ -n "${V7_STAGE1_DIR:-}" && -d "${V7_STAGE1_DIR}" ]] || \
  trace_vb_die "V7_STAGE1_DIR must point to the completed formal v7 Stage 1"
[[ "${SCAN_GPUS}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*)){7}$ ]] || \
  trace_vb_die "TRACE_VB_SCAN_GPUS must contain exactly eight numeric GPU IDs"

IFS=',' read -r -a scan_gpus <<< "${SCAN_GPUS}"
declare -A seen=()
for gpu in "${scan_gpus[@]}"; do
  [[ -z "${seen[${gpu}]+x}" ]] || \
    trace_vb_die "TRACE_VB_SCAN_GPUS contains duplicate GPU ${gpu}"
  seen[${gpu}]=1
done

trace_vb_assert_no_test_tokens "${BASH_SOURCE[0]}"
trace_vb_assert_no_test_tokens \
  "${SCRIPT_DIR}/wait_for_four_gpus_and_run_train_only_v8.sh"

supervisor_dir=${TRACE_VB_ARTIFACT_ROOT}/supervisor
mkdir -p "${supervisor_dir}"
resource_log=${supervisor_dir}/${PIPELINE_TAG}_resource_wait.log
exec > >(tee -a "${resource_log}") 2>&1

echo "$(date --iso-8601=seconds) waiting for any four launch-eligible GPUs from ${SCAN_GPUS}"
last_signature=
while true; do
  eligible=()
  status_parts=()
  query_failed=false
  for gpu in "${scan_gpus[@]}"; do
    row=
    if ! row=$(nvidia-smi -i "${gpu}" \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits); then
      status_parts+=("${gpu}:query_error")
      query_failed=true
      continue
    fi
    read -r free utilization < <(
      awk -F',' \
        '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}' \
        <<< "${row}"
    )
    if [[ ! "${free}" =~ ^[0-9]+$ || ! "${utilization}" =~ ^[0-9]+$ ]]; then
      status_parts+=("${gpu}:invalid")
      query_failed=true
      continue
    fi
    status_parts+=("${gpu}:${free}MiB:${utilization}%")
    if (( free >= TRACE_VB_MIN_FREE_GPU_MIB )) \
      && (( utilization <= TRACE_VB_MAX_IDLE_UTILIZATION )); then
      eligible+=("${gpu}")
    fi
  done

  signature="${status_parts[*]} eligible=${eligible[*]:-none}"
  if [[ "${signature}" != "${last_signature}" ]]; then
    echo "$(date --iso-8601=seconds) ${signature}"
    last_signature=${signature}
  fi

  if [[ "${query_failed}" == false && "${#eligible[@]}" -ge 4 ]]; then
    selected=("${eligible[@]:0:4}")
    selected_csv=$(IFS=,; printf '%s' "${selected[*]}")
    echo "$(date --iso-8601=seconds) selected GPUs ${selected_csv}; handing off to the audited fixed-set waiter"
    export TRACE_VB_FORMAL_GPUS="${selected_csv}"
    export TRACE_VB_FIXED_GPUS="${selected_csv}"
    export PIPELINE_TAG TRAIN_SEED POLL_SECONDS V7_STAGE1_DIR
    exec bash "${SCRIPT_DIR}/wait_for_four_gpus_and_run_train_only_v8.sh"
  fi
  sleep "${POLL_SECONDS}"
done
