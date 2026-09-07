#!/usr/bin/env bash
set -eu

if [[ "$*" == *"--query-compute-apps=gpu_uuid,pid,process_name,used_memory"* ]]; then
  printf '%s, %s, /usr/bin/sleep, 512\n' "${FAKE_GPU_UUID}" "${FAKE_GPU_PID}"
elif [[ "$*" == *"--query-compute-apps=gpu_uuid,pid"* ]]; then
  printf '%s, %s\n' "${FAKE_GPU_UUID}" "${FAKE_GPU_PID}"
else
  echo "unsupported mock nvidia-smi query: $*" >&2
  exit 2
fi
