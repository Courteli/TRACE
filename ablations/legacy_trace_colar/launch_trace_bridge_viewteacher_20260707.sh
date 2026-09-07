#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

TRACE_MODEL="${TRACE_MODEL:-trace_bridge_qwen3_instruct_viewteacher}"
MODE="${MODE:-stage1}"
GPU="${GPU:-1}"
RUN_TAG="${RUN_TAG:-20260707_trace_bridge_viewteacher_${MODE}_gpu${GPU}}"
SESSION="${SESSION:-trace_bridge_viewteacher_${MODE}_gpu${GPU}_0707}"
LOG="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/${RUN_TAG}.outer.log"

mkdir -p "$(dirname "${LOG}")"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd /disk1/dingxukai/trace_colar && TRACE_MODEL=${TRACE_MODEL} MODE=${MODE} GPU=${GPU} RUN_TAG=${RUN_TAG} TEST_TIMES=1 bash run_trace_bridge_pipeline_snapshot_20260707.sh 2>&1 | tee -a ${LOG}"

echo "[launched] ${SESSION}: MODEL=${TRACE_MODEL} MODE=${MODE} GPU=${GPU} RUN_TAG=${RUN_TAG}"
tmux ls | grep trace_bridge || true
