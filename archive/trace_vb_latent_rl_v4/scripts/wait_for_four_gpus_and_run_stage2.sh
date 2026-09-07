#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_ROOT=${DATA_ROOT:-/disk1/dingxukai/TRACE}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/disk1/dingxukai/TRACE/role_semantic_runs}
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MAX_FREE_MEMORY_MIB=${MAX_FREE_MEMORY_MIB:-4096}
MAX_FREE_UTILIZATION=${MAX_FREE_UTILIZATION:-10}
POLL_SECONDS=${POLL_SECONDS:-60}

if [[ "$#" -ne 1 || ! -f "$1" ]]; then
  echo "Usage: $0 <new-stage1-best-checkpoint>" >&2
  exit 2
fi
stage1_checkpoint=$1

mkdir -p "${ARTIFACT_ROOT}/supervisor"
exec 9>"${ARTIFACT_ROOT}/supervisor/.role_stage2_monitor.lock"
if ! flock -n 9; then
  echo "A role-semantic Stage 2 monitor is already active" >&2
  exit 2
fi

cd "${CODE_ROOT}"
TRACE_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" tools/data_contract_audit.py >/dev/null
TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${DATA_ROOT}" \
  "${PYTHON}" tools/role_pipeline_contract_audit.py >/dev/null

last_available=-1
while true; do
  free_gpus=()
  while IFS=',' read -r raw_index raw_memory raw_utilization; do
    index=${raw_index//[[:space:]]/}
    memory=${raw_memory//[[:space:]]/}
    utilization=${raw_utilization//[[:space:]]/}
    if [[ "${memory}" -le "${MAX_FREE_MEMORY_MIB}" ]] \
      && [[ "${utilization}" -le "${MAX_FREE_UTILIZATION}" ]]; then
      free_gpus+=("${index}")
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )

  if [[ "${#free_gpus[@]}" -ne "${last_available}" ]]; then
    echo "$(date --iso-8601=seconds) free_gpus=${free_gpus[*]:-none}"
    last_available=${#free_gpus[@]}
  fi
  if [[ "${#free_gpus[@]}" -ge 4 ]]; then
    selected=$(IFS=,; echo "${free_gpus[*]:0:4}")
    echo "$(date --iso-8601=seconds) launching role Stage 2 on ${selected}"
    exec "${SCRIPT_DIR}/run_stage2_and_evidence.sh" \
      "${selected}" "${stage1_checkpoint}"
  fi
  sleep "${POLL_SECONDS}"
done
