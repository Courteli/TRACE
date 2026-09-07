#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dingxukai/colar origin"
OUT_DIR="${ROOT}/run_outputs/qwen3_early_epoch_retrain_eval_20260602"
mkdir -p "${OUT_DIR}"

launch_worker() {
  local method="$1"
  local gpu="$2"
  local max_epochs="$3"
  local session="origin_${method}_qwen3_early_epoch_gpu${gpu}"

  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "bash '${ROOT}/run_qwen3_early_epoch_retrain_eval_worker_20260602.sh' '${method}' '${gpu}' '${max_epochs}' 2>&1 | tee -a '${OUT_DIR}/${method}_worker.log'"
  echo "Started ${session}"
}

# Lightning checkpoint filenames are zero-indexed. These correspond to paper
# Epoch 1 for iCoT, and paper Epochs 1-2 for Coconut and CODI.
launch_worker icot 1 1
launch_worker coconut 2 2
launch_worker distill 3 2

echo "Logs: ${OUT_DIR}"
