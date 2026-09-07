#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

POLL_SECONDS=${POLL_SECONDS:-30}
TRACE_VB_MAX_IDLE_UTILIZATION=${TRACE_VB_MAX_IDLE_UTILIZATION:-10}
TRACE_VB_FIXED_GPUS=${TRACE_VB_FIXED_GPUS:-}
PIPELINE_RESUME_STAGE1_CKPT=${PIPELINE_RESUME_STAGE1_CKPT:-}
TRACE_VB_SUPERVISOR_ATTEMPT=${TRACE_VB_SUPERVISOR_ATTEMPT:-}
TRAIN_SEED=${TRAIN_SEED:-0}
PIPELINE_TAG=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v7_full_seed${TRAIN_SEED}}

[[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] && (( POLL_SECONDS >= 10 )) || \
  trace_vb_die "POLL_SECONDS must be an integer of at least 10"
[[ "${TRACE_VB_MAX_IDLE_UTILIZATION}" =~ ^[0-9]+$ ]] \
  && (( TRACE_VB_MAX_IDLE_UTILIZATION <= 100 )) || \
  trace_vb_die "TRACE_VB_MAX_IDLE_UTILIZATION must be between 0 and 100"
if [[ -n "${TRACE_VB_FIXED_GPUS}" ]]; then
  IFS=',' read -r -a fixed_gpu_array <<< "${TRACE_VB_FIXED_GPUS}"
  [[ "${#fixed_gpu_array[@]}" -eq 4 ]] || \
    trace_vb_die "TRACE_VB_FIXED_GPUS must contain exactly four GPUs"
  [[ "$(printf '%s\n' "${fixed_gpu_array[@]}" | sort -u | wc -l)" -eq 4 ]] || \
    trace_vb_die "TRACE_VB_FIXED_GPUS entries must be unique"
  for gpu in "${fixed_gpu_array[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || \
      trace_vb_die "invalid fixed physical GPU ID: ${gpu}"
  done
fi
if [[ -n "${PIPELINE_RESUME_STAGE1_CKPT}" ]]; then
  [[ -f "${PIPELINE_RESUME_STAGE1_CKPT}" ]] || \
    trace_vb_die "missing Stage-1 recovery checkpoint: ${PIPELINE_RESUME_STAGE1_CKPT}"
  [[ -n "${TRACE_VB_FIXED_GPUS}" ]] || \
    trace_vb_die "formal recovery requires TRACE_VB_FIXED_GPUS"
  [[ -n "${TRACE_VB_SUPERVISOR_ATTEMPT}" ]] || \
    trace_vb_die "formal recovery requires a unique TRACE_VB_SUPERVISOR_ATTEMPT"
fi
[[ -z "${TRACE_VB_SUPERVISOR_ATTEMPT}" \
  || "${TRACE_VB_SUPERVISOR_ATTEMPT}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  trace_vb_die "TRACE_VB_SUPERVISOR_ATTEMPT contains unsafe characters"

supervisor_dir=${TRACE_VB_ARTIFACT_ROOT}/supervisor
mkdir -p "${supervisor_dir}"
exec 9>"${supervisor_dir}/.trace_vb_v7_full_pipeline.lock"
if ! flock -n 9; then
  trace_vb_die "a TRACE-VB-v7 full-pipeline waiter or run is already active"
fi

supervisor_stem=${PIPELINE_TAG}
if [[ -n "${TRACE_VB_SUPERVISOR_ATTEMPT}" ]]; then
  supervisor_stem=${PIPELINE_TAG}_${TRACE_VB_SUPERVISOR_ATTEMPT}
fi
supervisor_log=${supervisor_dir}/${supervisor_stem}_wait.log
exit_status_file=${supervisor_dir}/${supervisor_stem}.exit_status
[[ ! -e "${exit_status_file}" ]] || \
  trace_vb_die "refusing to overwrite supervisor status: ${exit_status_file}"
exec > >(tee -a "${supervisor_log}") 2>&1
echo "$(date --iso-8601=seconds) waiting for four GPUs with at least ${TRACE_VB_MIN_FREE_GPU_MIB} MiB free and utilization at most ${TRACE_VB_MAX_IDLE_UTILIZATION}%; fixed=${TRACE_VB_FIXED_GPUS:-auto}"

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

  selected=
  if [[ -n "${TRACE_VB_FIXED_GPUS}" ]]; then
    all_fixed_ready=true
    for fixed_gpu in "${fixed_gpu_array[@]}"; do
      if [[ ! " ${eligible_gpus[*]:-} " =~ " ${fixed_gpu} " ]]; then
        all_fixed_ready=false
        break
      fi
    done
    if [[ "${all_fixed_ready}" == true ]]; then
      selected=${TRACE_VB_FIXED_GPUS}
    fi
  elif (( ${#eligible_gpus[@]} >= 4 )); then
    selected=$(IFS=,; echo "${eligible_gpus[*]:0:4}")
  fi

  if [[ -n "${selected}" ]]; then
    echo "$(date --iso-8601=seconds) launching TRACE-VB-v7 full pipeline on ${selected}"
    set +e
    if [[ -n "${PIPELINE_RESUME_STAGE1_CKPT}" ]]; then
      env TRAIN_SEED="${TRAIN_SEED}" \
        bash "${SCRIPT_DIR}/resume_full_pipeline_vb.sh" \
          "${selected}" "${selected%%,*}" "${PIPELINE_TAG}" \
          "${PIPELINE_RESUME_STAGE1_CKPT}"
    else
      env PIPELINE_TAG="${PIPELINE_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
        bash "${SCRIPT_DIR}/run_full_pipeline_vb.sh" \
          "${selected}" "${selected%%,*}"
    fi
    run_status=$?
    set -e
    printf '%s\n' "${run_status}" > "${exit_status_file}"
    echo "$(date --iso-8601=seconds) full pipeline exited with status ${run_status}"
    exit "${run_status}"
  fi

  sleep "${POLL_SECONDS}"
done
