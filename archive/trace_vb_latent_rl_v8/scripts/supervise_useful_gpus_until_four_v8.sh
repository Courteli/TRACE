#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

CONTROL_TAG=${CONTROL_TAG:?CONTROL_TAG is required}
RESTART_SECONDS=${TRACE_VB_CONTROLLER_RESTART_SECONDS:-10}
trace_vb_require_safe_tag "${CONTROL_TAG}"
[[ "${RESTART_SECONDS}" =~ ^[0-9]+$ ]] && (( RESTART_SECONDS >= 10 )) || \
  trace_vb_die "TRACE_VB_CONTROLLER_RESTART_SECONDS must be at least 10"

control_root=${TRACE_VB_ARTIFACT_ROOT}/useful_controls/${CONTROL_TAG}
handoff_record=${control_root}/formal_gpu_handoff.txt
mkdir -p "${control_root}"

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "$(date --iso-8601=seconds) starting resource-controller attempt ${attempt}"
  set +e
  bash "${SCRIPT_DIR}/use_three_gpus_until_four_v8.sh"
  status=$?
  set -e
  if [[ -s "${handoff_record}" ]]; then
    echo "$(date --iso-8601=seconds) formal handoff recorded; controller exited status=${status}"
    exit "${status}"
  fi
  echo "$(date --iso-8601=seconds) resource controller exited before handoff status=${status}; restarting in ${RESTART_SECONDS}s"
  sleep "${RESTART_SECONDS}"
done
