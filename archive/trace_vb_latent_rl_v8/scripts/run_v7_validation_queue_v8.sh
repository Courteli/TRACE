#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
V7_ROOT=/home/dingxukai/TRACE/trace_vb_latent_rl_v7
V7_RUN_DIR=/disk1/dingxukai/TRACE/trace_vb_v7_runs/training/20260818-213000_trace_vb_v7_full_seed0_stage1
V7_MANIFEST=${V7_RUN_DIR}/manifest.txt
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <physical-gpu> <early|middle|late> <control-tag> <output-root>" >&2
  exit 2
fi
physical_gpu=$1
queue=$2
control_tag=$3
output_root=$4
[[ "${physical_gpu}" =~ ^[0-9]+$ ]] || { echo "invalid GPU" >&2; exit 2; }
[[ "${control_tag}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe tag" >&2; exit 2; }
case "${queue}" in early|middle|late) ;; *) echo "invalid queue" >&2; exit 2 ;; esac
[[ -s "${V7_MANIFEST}" ]] || { echo "missing v7 manifest" >&2; exit 2; }

worker_root=${output_root}/gpu${physical_gpu}_${queue}
mkdir -p "${worker_root}/logs" "${worker_root}/tmp"
exec > >(tee -a "${worker_root}/worker.log") 2>&1
echo "$(date --iso-8601=seconds) starting ${queue} validation queue on physical GPU ${physical_gpu}"

child_pid=
stop_child() {
  local status=$?
  trap - INT TERM HUP EXIT
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill -INT "${child_pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${child_pid}" 2>/dev/null || break
      sleep 1
    done
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  echo "$(date --iso-8601=seconds) queue ${queue} stopped status=${status}"
  exit "${status}"
}
trap stop_child INT TERM HUP EXIT

resolve_epoch_checkpoint() {
  local epoch=$1 checkpoint
  checkpoint=$(awk -F= -v key="epoch_${epoch}_last_checkpoint" \
    '$1 == key {print substr($0, index($0, "=") + 1)}' "${V7_MANIFEST}")
  [[ -f "${checkpoint}" ]] || { echo "missing epoch ${epoch} checkpoint" >&2; exit 2; }
  printf '%s\n' "${checkpoint}"
}

run_validation() {
  local label=$1 checkpoint=$2 validation_path=$3
  local result=${worker_root}/${label}_${validation_path}.json
  local run_tag=${control_tag}_gpu${physical_gpu}_${label}_${validation_path}
  if [[ -s "${result}" ]]; then
    echo "$(date --iso-8601=seconds) reusing completed ${result}"
    return
  fi
  local completed_summary
  completed_summary=$(find "${output_root}" -type f \
    -path "*_${control_tag}_gpu*_${label}_${validation_path}/validation_epoch_000.json" \
    -print | sort | sed -n '1p')
  if [[ -n "${completed_summary}" && -s "${completed_summary}" ]]; then
    cp "${completed_summary}" "${result}"
    echo "$(date --iso-8601=seconds) promoted completed strict summary ${completed_summary}"
    return
  fi
  echo "$(date --iso-8601=seconds) validating ${label} path=${validation_path} checkpoint=${checkpoint}"
  env \
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    TRACE_LOG_ROOT="${worker_root}/logs" \
    TMPDIR="${worker_root}/tmp" \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRACE_VB_STARTUP_GUARD_MIB=4096 \
    "${PYTHON}" "${CODE_ROOT}/tools/run_v7_validation_only.py" \
      --v7-root "${V7_ROOT}" \
      --checkpoint "${checkpoint}" \
      --validation-path "${validation_path}" \
      --run-tag "${run_tag}" \
      --output "${result}" \
      --seed 0 &
  child_pid=$!
  printf '%s\n' "${child_pid}" > "${worker_root}/active_child.pid"
  wait "${child_pid}"
  child_pid=
  : > "${worker_root}/active_child.pid"
  echo "$(date --iso-8601=seconds) completed ${label} path=${validation_path}"
}

case "${queue}" in
  early)
    for epoch in 1 2 3 4; do
      run_validation "epoch${epoch}" "$(resolve_epoch_checkpoint "${epoch}")" student_commit
    done
    for epoch in 1 2 3 4; do
      run_validation "epoch${epoch}" "$(resolve_epoch_checkpoint "${epoch}")" capability_teacher_all_roles
    done
    ;;
  middle)
    for epoch in 5 6 7; do
      run_validation "epoch${epoch}" "$(resolve_epoch_checkpoint "${epoch}")" student_commit
    done
    for epoch in 5 6 7; do
      run_validation "epoch${epoch}" "$(resolve_epoch_checkpoint "${epoch}")" capability_teacher_all_roles
    done
    ;;
  late)
    for epoch in 8 9 10; do
      run_validation "epoch${epoch}" "$(resolve_epoch_checkpoint "${epoch}")" student_commit
    done
    for epoch in 8 9 10; do
      run_validation "epoch${epoch}" "$(resolve_epoch_checkpoint "${epoch}")" capability_teacher_all_roles
    done
    best_checkpoint=$(<"${V7_RUN_DIR}/best_checkpoint.txt")
    [[ -f "${best_checkpoint}" ]] || { echo "missing formal v7 best checkpoint" >&2; exit 2; }
    run_validation best "${best_checkpoint}" capability_teacher_all_roles
    ;;
esac

trap - INT TERM HUP EXIT
echo "$(date --iso-8601=seconds) queue ${queue} completed; entering managed CUDA lease"
rm -f "${worker_root}/lease_ready.json"
env \
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  "${PYTHON}" "${CODE_ROOT}/tools/hold_cuda_lease.py" \
    --reserve-mib 16384 \
    --poll-seconds 5 \
    --ready-file "${worker_root}/lease_ready.json" &
child_pid=$!
printf '%s\n' "${child_pid}" > "${worker_root}/active_child.pid"
trap stop_child INT TERM HUP EXIT
wait "${child_pid}"
child_pid=
: > "${worker_root}/active_child.pid"
trap - INT TERM HUP EXIT
echo "$(date --iso-8601=seconds) queue ${queue} lease stopped"
