#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

POLL_SECONDS=${POLL_SECONDS:-30}
TRAIN_SEED=${TRAIN_SEED:-0}
PIPELINE_TAG=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v8_train_seed${TRAIN_SEED}}
trace_vb_require_formal_gpu_set "${TRACE_VB_FIXED_GPUS}" "waiter fixed"
[[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] && (( POLL_SECONDS >= 10 )) || \
  trace_vb_die "POLL_SECONDS must be an integer of at least 10"
[[ -n "${V7_STAGE1_DIR:-}" ]] || \
  trace_vb_die "V7_STAGE1_DIR is required; the waiter resolves its formal best"
trace_vb_require_safe_tag "${PIPELINE_TAG}"
trace_vb_assert_no_test_tokens "${BASH_SOURCE[0]}"
trace_vb_assert_no_test_tokens "${SCRIPT_DIR}/run_train_only_v8.sh"

supervisor_dir=${TRACE_VB_ARTIFACT_ROOT}/supervisor
mkdir -p "${supervisor_dir}"
exec 9>"${supervisor_dir}/.trace_vb_v8_train_only.lock"
if ! flock -n 9; then
  trace_vb_die "a v8 train-only waiter or run is already active"
fi
supervisor_log=${supervisor_dir}/${PIPELINE_TAG}_wait.log
trigger_report=${supervisor_dir}/${PIPELINE_TAG}.v7_trigger.json
stage1_train_log=${TRACE_VB_ARTIFACT_ROOT}/training/${PIPELINE_TAG}_stage1/train.log
stage2_train_log=${TRACE_VB_ARTIFACT_ROOT}/training/${PIPELINE_TAG}_stage2/train.log
exec > >(tee -a "${supervisor_log}") 2>&1
supervisor_attempt=${TRACE_VB_SUPERVISOR_ATTEMPT_ID:-initial_$(date +%Y%m%d-%H%M%S)}
trace_vb_require_safe_tag "${supervisor_attempt}"
supervisor_session=${TRACE_VB_SUPERVISOR_SESSION:-}
if [[ -z "${supervisor_session}" && -n "${TMUX:-}" ]]; then
  supervisor_session=$(tmux display-message -p '#S')
fi
[[ -n "${supervisor_session}" ]] || \
  trace_vb_die "TRACE_VB_SUPERVISOR_SESSION is required outside tmux"
trace_vb_require_safe_tag "${supervisor_session}"
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" start \
  --supervisor-dir "${supervisor_dir}" \
  --pipeline-tag "${PIPELINE_TAG}" \
  --attempt-id "${supervisor_attempt}" \
  --attempt-kind initial \
  --physical-gpus "${TRACE_VB_FIXED_GPUS}" \
  --formal-gpus "${TRACE_VB_FORMAL_GPUS}" \
  --fixed-gpus "${TRACE_VB_FIXED_GPUS}" \
  --tmux-session "${supervisor_session}" \
  --stage1-train-log "${stage1_train_log}" \
  --stage2-train-log "${stage2_train_log}"
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
      --attempt-id "${supervisor_attempt}" \
      --status FAILED \
      --exit-status "${failure_status}"
    set -e
  fi
}
trap finalize_unexpected_exit EXIT
echo "$(date --iso-8601=seconds) waiting for fixed GPUs ${TRACE_VB_FIXED_GPUS}"

IFS=',' read -r -a fixed_gpus <<< "${TRACE_VB_FIXED_GPUS}"
last_signature=
resolved_v7_best=
while true; do
  ready=true
  status_parts=()
  v7_processes=none
  if trace_vb_v7_training_is_active; then
    ready=false
    v7_processes=active
  fi
  trigger_state=pending
  if [[ -n "${resolved_v7_best}" ]]; then
    trigger_state=pass
  else
    v7_manifest=${V7_STAGE1_DIR}/manifest.txt
    v7_best_record=${V7_STAGE1_DIR}/best_checkpoint.txt
    v7_index=
    if [[ -s "${v7_manifest}" ]]; then
      v7_index=$(awk -F= '$1 == "validation_summary_index" {print substr($0, index($0, "=") + 1)}' "${v7_manifest}")
    fi
    if [[ -s "${v7_manifest}" \
      && -s "${v7_best_record}" \
      && -n "${v7_index}" \
      && -s "${v7_index}" ]] \
      && grep -Eq '^finished_at=.+' "${v7_manifest}"; then
      set +e
      trace_vb_audit_v7_trigger \
        "${V7_STAGE1_DIR}" "" "${trigger_report}"
      trigger_status=$?
      set -e
      if (( trigger_status == 0 )); then
        resolved_v7_best=$("${TRACE_VB_PYTHON}" - "${trigger_report}" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit("v7 trigger report is not PASS")
checkpoint = Path(report["checkpoint"]).resolve()
if not checkpoint.is_file():
    raise SystemExit("resolved v7 formal best is missing")
print(checkpoint)
PY
        )
        export V7_BEST_CKPT="${resolved_v7_best}"
        trigger_state=pass
        echo "$(date --iso-8601=seconds) resolved formal v7 best: ${resolved_v7_best}"
      elif (( trigger_status == 3 )); then
        echo "$(date --iso-8601=seconds) terminal trigger_false: formal v7 best is already above 70%; v8 will not launch"
        "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" transition \
          --supervisor-dir "${supervisor_dir}" \
          --pipeline-tag "${PIPELINE_TAG}" \
          --attempt-id "${supervisor_attempt}" \
          --status SUCCEEDED
        attempt_finalized=true
        exit 0
      elif [[ "${v7_processes}" == none ]]; then
        echo "$(date --iso-8601=seconds) completed v7 artifact failed the formal trigger audit"
        "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" transition \
          --supervisor-dir "${supervisor_dir}" \
          --pipeline-tag "${PIPELINE_TAG}" \
          --attempt-id "${supervisor_attempt}" \
          --status FAILED \
          --exit-status 2
        attempt_finalized=true
        exit 2
      fi
    fi
  fi
  if [[ "${trigger_state}" != pass ]]; then
    ready=false
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
  signature="${status_parts[*]} v7_processes=${v7_processes} trigger=${trigger_state}"
  if [[ "${signature}" != "${last_signature}" ]]; then
    echo "$(date --iso-8601=seconds) ${signature}"
    last_signature=${signature}
  fi
  if [[ "${ready}" == true ]]; then
    echo "$(date --iso-8601=seconds) launching v8 train-only pipeline"
    "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" transition \
      --supervisor-dir "${supervisor_dir}" \
      --pipeline-tag "${PIPELINE_TAG}" \
      --attempt-id "${supervisor_attempt}" \
      --status RUNNING
    set +e
    env \
      TRACE_VB_FORMAL_GPUS="${TRACE_VB_FORMAL_GPUS}" \
      TRACE_VB_FIXED_GPUS="${TRACE_VB_FIXED_GPUS}" \
      V7_BEST_CKPT="${V7_BEST_CKPT}" \
      V7_STAGE1_DIR="${V7_STAGE1_DIR}" \
      PIPELINE_TAG="${PIPELINE_TAG}" \
      TRAIN_SEED="${TRAIN_SEED}" \
      bash "${SCRIPT_DIR}/run_train_only_v8.sh" "${TRACE_VB_FIXED_GPUS}"
    status=$?
    set -e
    if (( status == 0 )); then
      "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" transition \
        --supervisor-dir "${supervisor_dir}" \
        --pipeline-tag "${PIPELINE_TAG}" \
        --attempt-id "${supervisor_attempt}" \
        --status SUCCEEDED
    else
      "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/supervisor_attempt_v8.py" transition \
        --supervisor-dir "${supervisor_dir}" \
        --pipeline-tag "${PIPELINE_TAG}" \
        --attempt-id "${supervisor_attempt}" \
        --status FAILED \
        --exit-status "${status}"
    fi
    attempt_finalized=true
    echo "$(date --iso-8601=seconds) pipeline exited with status ${status}"
    exit "${status}"
  fi
  sleep "${POLL_SECONDS}"
done
