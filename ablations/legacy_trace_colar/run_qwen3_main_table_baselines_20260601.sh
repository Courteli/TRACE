#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dingxukai/colar origin"
OUT_DIR="${ROOT}/run_outputs/qwen3_main_table_baselines_20260601"
mkdir -p "${OUT_DIR}"

launch_worker() {
  local method="$1"
  local gpu="$2"
  local session="origin_${method}_qwen3_main_table_gpu${gpu}"

  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "bash '${ROOT}/run_qwen3_main_table_baseline_worker_20260601.sh' '${method}' '${gpu}' 2>&1 | tee -a '${OUT_DIR}/${method}_worker.log'"
  echo "Started ${session}"
}

launch_worker icot 1
launch_worker coconut 2
launch_worker distill 3

echo "Logs: ${OUT_DIR}"
