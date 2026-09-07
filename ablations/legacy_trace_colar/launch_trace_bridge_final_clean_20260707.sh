#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

SNAPSHOT="${SNAPSHOT:-run_trace_bridge_pipeline_snapshot_20260707.sh}"

launch_one() {
  local model="$1"
  local mode="$2"
  local gpu="$3"
  local session="$4"
  local tag="$5"
  local log="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/${tag}.outer.log"

  mkdir -p "$(dirname "${log}")"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[skip] tmux session already exists: ${session}"
    return
  fi

  tmux new-session -d -s "${session}" \
    "cd /disk1/dingxukai/trace_colar && TRACE_MODEL=${model} MODE=${mode} GPU=${gpu} RUN_TAG=${tag} TEST_TIMES=1 bash ${SNAPSHOT} 2>&1 | tee -a ${log}"
  echo "[launched] ${session}: MODEL=${model} MODE=${mode} GPU=${gpu} RUN_TAG=${tag}"
}

# Version A: residual-path multi-view perturbation.
launch_one trace_bridge_qwen3_instruct_structmv stage1 2 \
  trace_bridge_final_structmv_stage1_gpu2_0707 \
  20260707_trace_bridge_final_structmv_stage1_gpu2

launch_one trace_bridge_qwen3_instruct_structmv stage2_conservative 4 \
  trace_bridge_final_structmv_stage2_conservative_gpu4_0707 \
  20260707_trace_bridge_final_structmv_stage2_conservative_gpu4

launch_one trace_bridge_qwen3_instruct_structmv stage2_strong 6 \
  trace_bridge_final_structmv_stage2_strong_gpu6_0707 \
  20260707_trace_bridge_final_structmv_stage2_strong_gpu6

# Version B: same structure plus weak multi-view compression teacher.
launch_one trace_bridge_qwen3_instruct_viewteacher stage1 7 \
  trace_bridge_final_viewteacher_stage1_gpu7_0707 \
  20260707_trace_bridge_final_viewteacher_stage1_gpu7

launch_one trace_bridge_qwen3_instruct_viewteacher stage2_conservative 5 \
  trace_bridge_final_viewteacher_stage2_conservative_gpu5_0707 \
  20260707_trace_bridge_final_viewteacher_stage2_conservative_gpu5

launch_one trace_bridge_qwen3_instruct_viewteacher stage2_strong 1 \
  trace_bridge_final_viewteacher_stage2_strong_gpu1_0707 \
  20260707_trace_bridge_final_viewteacher_stage2_strong_gpu1

WATCH_SESSIONS="trace_bridge_final_structmv_stage1_gpu2_0707 trace_bridge_final_structmv_stage2_conservative_gpu4_0707 trace_bridge_final_structmv_stage2_strong_gpu6_0707 trace_bridge_final_viewteacher_stage1_gpu7_0707 trace_bridge_final_viewteacher_stage2_conservative_gpu5_0707 trace_bridge_final_viewteacher_stage2_strong_gpu1_0707"
RUN_REGEX="20260707_trace_bridge_final_(structmv|viewteacher)_(stage1|stage2_conservative|stage2_strong)_gpu(1|2|4|5|6|7)"
SESSION=trace_bridge_final_audit_waiter_0707 \
LOG=/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_final_20260707/audit_waiter.log \
OUT_DIR=/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_final_20260707 \
WATCH_SESSIONS="${WATCH_SESSIONS}" \
RUN_REGEX="${RUN_REGEX}" \
bash launch_trace_bridge_audit_waiter_20260707.sh

tmux ls | grep trace_bridge || true
