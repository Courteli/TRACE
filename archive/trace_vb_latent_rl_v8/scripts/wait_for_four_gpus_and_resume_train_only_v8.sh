#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -ne 4 ]]; then
  echo "Usage: V7_BEST_CKPT=/path/to/v7-best.ckpt $0 <gpus> <pipeline-tag> <stage1|stage2> <validated-boundary-checkpoint>" >&2
  exit 2
fi
physical_gpus=$1
PIPELINE_TAG=$2
resume_phase=$3
recovery_checkpoint=$4
POLL_SECONDS=${POLL_SECONDS:-30}
TRAIN_SEED=${TRAIN_SEED:-0}
trace_vb_require_safe_tag "${PIPELINE_TAG}"
trace_vb_require_formal_gpu_set "${physical_gpus}" "resume waiter"
[[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] && (( POLL_SECONDS >= 10 )) || \
  trace_vb_die "POLL_SECONDS must be an integer of at least 10"
[[ "${resume_phase}" == stage1 || "${resume_phase}" == stage2 ]] || \
  trace_vb_die "resume phase must be stage1 or stage2"
[[ -f "${recovery_checkpoint}" ]] || \
  trace_vb_die "missing validated boundary checkpoint: ${recovery_checkpoint}"
[[ -n "${V7_BEST_CKPT:-}" && -f "${V7_BEST_CKPT}" ]] || \
  trace_vb_die "V7_BEST_CKPT is required for recovery continuity"
[[ -n "${V7_STAGE1_DIR:-}" && -d "${V7_STAGE1_DIR}" ]] || \
  trace_vb_die "V7_STAGE1_DIR is required for recovery continuity"
trace_vb_require_metric_artifacts
trace_vb_assert_no_test_tokens "${BASH_SOURCE[0]}"
trace_vb_assert_no_test_tokens "${SCRIPT_DIR}/resume_train_only_v8.sh"

PIPELINE_DIR=${TRACE_VB_ARTIFACT_ROOT}/pipelines/${PIPELINE_TAG}
STAGE1_TAG=${PIPELINE_TAG}_stage1
STAGE2_TAG=${PIPELINE_TAG}_stage2
STAGE1_DIR=${TRACE_VB_ARTIFACT_ROOT}/training/${STAGE1_TAG}
STAGE2_DIR=${TRACE_VB_ARTIFACT_ROOT}/training/${STAGE2_TAG}
if [[ "${resume_phase}" == stage1 ]]; then
  candidate_dir=${STAGE1_DIR}/candidates
else
  candidate_dir=${STAGE2_DIR}/candidates
fi
supervisor_dir=${TRACE_VB_ARTIFACT_ROOT}/supervisor
mkdir -p "${supervisor_dir}"
exec 9>"${supervisor_dir}/.trace_vb_v8_train_only.lock"
if ! flock -n 9; then
  trace_vb_die "a v8 train-only waiter or run is already active"
fi

attempt_id=${TRACE_VB_SUPERVISOR_ATTEMPT_ID:-resume_${resume_phase}_$(date +%Y%m%d-%H%M%S)}
trace_vb_require_safe_tag "${attempt_id}"
supervisor_session=${TRACE_VB_SUPERVISOR_SESSION:-}
if [[ -z "${supervisor_session}" && -n "${TMUX:-}" ]]; then
  supervisor_session=$(tmux display-message -p '#S')
fi
[[ -n "${supervisor_session}" ]] || \
  trace_vb_die "TRACE_VB_SUPERVISOR_SESSION is required outside tmux"
trace_vb_require_safe_tag "${supervisor_session}"
supervisor_log=${supervisor_dir}/${PIPELINE_TAG}_wait.log
exec > >(tee -a "${supervisor_log}") 2>&1

resume_binding=${PIPELINE_DIR}/resume_binding_${resume_phase}_${attempt_id}.json
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/audit_pipeline_resume_binding_v8.py" \
  --manifest "${PIPELINE_DIR}/manifest.txt" \
  --pipeline-tag "${PIPELINE_TAG}" \
  --stage1-tag "${STAGE1_TAG}" \
  --stage2-tag "${STAGE2_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --physical-gpus "${physical_gpus}" \
  --v7-best-checkpoint "${V7_BEST_CKPT}" \
  --recovery-checkpoint "${recovery_checkpoint}" \
  --phase "${resume_phase}" \
  --candidate-dir "${candidate_dir}" \
  --output "${resume_binding}"
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" start \
  --supervisor-dir "${supervisor_dir}" \
  --pipeline-tag "${PIPELINE_TAG}" \
  --attempt-id "${attempt_id}" \
  --attempt-kind "resume_${resume_phase}" \
  --resume-phase "${resume_phase}" \
  --physical-gpus "${physical_gpus}" \
  --formal-gpus "${TRACE_VB_FORMAL_GPUS}" \
  --fixed-gpus "${TRACE_VB_FIXED_GPUS}" \
  --tmux-session "${supervisor_session}" \
  --stage1-train-log "${STAGE1_DIR}/train.log" \
  --stage2-train-log "${STAGE2_DIR}/train.log" \
  --validated-boundary-contract "${resume_binding}"
attempt_finalized=false
finalize_unexpected_exit() {
  local status=$?
  if [[ "${attempt_finalized}" != true ]]; then
    local failure_status=${status}
    (( failure_status > 0 && failure_status <= 255 )) || failure_status=2
    set +e
    "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" transition \
      --supervisor-dir "${supervisor_dir}" \
      --pipeline-tag "${PIPELINE_TAG}" \
      --attempt-id "${attempt_id}" \
      --status FAILED \
      --exit-status "${failure_status}"
    set -e
  fi
}
trap finalize_unexpected_exit EXIT

echo "$(date --iso-8601=seconds) waiting to resume ${resume_phase} on fixed GPUs ${physical_gpus}"
IFS=',' read -r -a fixed_gpus <<< "${physical_gpus}"
last_signature=
while true; do
  ready=true
  status_parts=()
  v7_processes=none
  if trace_vb_v7_training_is_active; then
    ready=false
    v7_processes=active
  fi
  for gpu in "${fixed_gpus[@]}"; do
    read -r free utilization < <(
      nvidia-smi -i "${gpu}" \
        --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
    )
    status_parts+=("${gpu}:${free}MiB:${utilization}%")
    if (( free < TRACE_VB_MIN_FREE_GPU_MIB )) \
      || (( utilization > TRACE_VB_MAX_IDLE_UTILIZATION )); then
      ready=false
    fi
  done
  signature="${status_parts[*]} v7_processes=${v7_processes} boundary=validated"
  if [[ "${signature}" != "${last_signature}" ]]; then
    echo "$(date --iso-8601=seconds) ${signature}"
    last_signature=${signature}
  fi
  if [[ "${ready}" == true ]]; then
    "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" transition \
      --supervisor-dir "${supervisor_dir}" \
      --pipeline-tag "${PIPELINE_TAG}" \
      --attempt-id "${attempt_id}" \
      --status RUNNING
    set +e
    env \
      TRACE_VB_FORMAL_GPUS="${TRACE_VB_FORMAL_GPUS}" \
      TRACE_VB_FIXED_GPUS="${TRACE_VB_FIXED_GPUS}" \
      TRACE_VB_RESUME_ATTEMPT="${attempt_id}" \
      V7_BEST_CKPT="${V7_BEST_CKPT}" \
      V7_STAGE1_DIR="${V7_STAGE1_DIR}" \
      TRAIN_SEED="${TRAIN_SEED}" \
      bash "${SCRIPT_DIR}/resume_train_only_v8.sh" \
        "${physical_gpus}" "${PIPELINE_TAG}" \
        "${resume_phase}" "${recovery_checkpoint}"
    status=$?
    set -e
    if (( status == 0 )); then
      "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" transition \
        --supervisor-dir "${supervisor_dir}" \
        --pipeline-tag "${PIPELINE_TAG}" \
        --attempt-id "${attempt_id}" \
        --status SUCCEEDED
    else
      "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" transition \
        --supervisor-dir "${supervisor_dir}" \
        --pipeline-tag "${PIPELINE_TAG}" \
        --attempt-id "${attempt_id}" \
        --status FAILED \
        --exit-status "${status}"
    fi
    attempt_finalized=true
    echo "$(date --iso-8601=seconds) resume pipeline exited with status ${status}"
    exit "${status}"
  fi
  sleep "${POLL_SECONDS}"
done
