#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

launch_one() {
  local session="$1"
  local mode="$2"
  local gpu="$3"
  local tag="$4"
  local log="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/${tag}.outer.log"
  mkdir -p "$(dirname "${log}")"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[skip] tmux session already exists: ${session}"
    return
  fi
  tmux new-session -d -s "${session}" \
    "cd /disk1/dingxukai/trace_colar && MODE=${mode} GPU=${gpu} RUN_TAG=${tag} TEST_TIMES=1 bash run_trace_bridge_pipeline_20260707.sh 2>&1 | tee -a ${log}"
  echo "[launched] ${session}: MODE=${mode} GPU=${gpu} RUN_TAG=${tag}"
}

launch_one trace_bridge_baseline_gpu7_0707 baseline 7 20260707_trace_bridge_baseline_gpu7
launch_one trace_bridge_stage1_v2_gpu2_0707 stage1 2 20260707_trace_bridge_stage1_v2_gpu2
launch_one trace_bridge_stage2_conservative_v2_gpu4_0707 stage2_conservative 4 20260707_trace_bridge_stage2_conservative_v2_gpu4
launch_one trace_bridge_stage2_strong_v2_gpu6_0707 stage2_strong 6 20260707_trace_bridge_stage2_strong_v2_gpu6

tmux ls | grep trace_bridge || true
