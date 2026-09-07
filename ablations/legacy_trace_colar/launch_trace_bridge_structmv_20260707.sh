#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

TRACE_MODEL="${TRACE_MODEL:-trace_bridge_qwen3_instruct_structmv}"

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
    "cd /disk1/dingxukai/trace_colar && TRACE_MODEL=${TRACE_MODEL} MODE=${mode} GPU=${gpu} RUN_TAG=${tag} TEST_TIMES=1 bash run_trace_bridge_pipeline_snapshot_20260707.sh 2>&1 | tee -a ${log}"
  echo "[launched] ${session}: MODEL=${TRACE_MODEL} MODE=${mode} GPU=${gpu} RUN_TAG=${tag}"
}

launch_one trace_bridge_structmv_stage1_gpu2_0707 stage1 "${GPU_STAGE1:-2}" 20260707_trace_bridge_structmv_stage1_gpu${GPU_STAGE1:-2}
launch_one trace_bridge_structmv_stage2_conservative_gpu4_0707 stage2_conservative "${GPU_CONS:-4}" 20260707_trace_bridge_structmv_stage2_conservative_gpu${GPU_CONS:-4}
launch_one trace_bridge_structmv_stage2_strong_gpu6_0707 stage2_strong "${GPU_STRONG:-6}" 20260707_trace_bridge_structmv_stage2_strong_gpu${GPU_STRONG:-6}

tmux ls | grep trace_bridge || true
