#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

SNAPSHOT="${SNAPSHOT:-run_trace_bridge_pipeline_snapshot_20260707.sh}"
RUN_MARKER="${RUN_MARKER:-20260708_trace_bridge_bridgefull_}"
AUDIT_OUT="${AUDIT_OUT:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_bridgefull_20260708}"
RUN_REGEX="${RUN_REGEX:-20260708_trace_bridge_bridgefull_(bridge_baseline_gpu2|(viewteacher|vizstrong)_(stage1|stage2_conservative|stage2_strong)_gpu(1|2|4|5|6|7))}"
ROOT="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge"

mkdir -p "${AUDIT_OUT}"

launch_one() {
  local model="$1"
  local mode="$2"
  local gpu="$3"
  local session="$4"
  local tag="$5"
  local log="${ROOT}/${tag}.outer.log"

  mkdir -p "$(dirname "${log}")"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[skip] tmux session already exists: ${session}"
    return
  fi

  tmux new-session -d -s "${session}" \
    "cd /disk1/dingxukai/trace_colar && TRACE_MODEL=${model} MODE=${mode} GPU=${gpu} RUN_TAG=${tag} TEST_TIMES=1 bash ${SNAPSHOT} 2>&1 | tee -a ${log}"
  echo "[launched] ${session}: MODEL=${model} MODE=${mode} GPU=${gpu} RUN_TAG=${tag}"
}

queue_one() {
  local wait_for="$1"
  local model="$2"
  local mode="$3"
  local gpu="$4"
  local session="$5"
  local tag="$6"
  WAIT_FOR_SESSION="${wait_for}" \
  SESSION="${session}" \
  TRACE_MODEL="${model}" \
  MODE="${mode}" \
  GPU="${gpu}" \
  RUN_TAG="${tag}" \
  SNAPSHOT="${SNAPSHOT}" \
  AUDIT_OUT="${AUDIT_OUT}" \
  AUDIT_REGEX="${RUN_REGEX}" \
  bash launch_trace_bridge_vizstrong_queue_20260707.sh
}

launch_one trace_bridge_qwen3_instruct_viewteacher stage1 2 \
  trace_bridge_bridgefull_viewteacher_stage1_gpu2_0708 \
  20260708_trace_bridge_bridgefull_viewteacher_stage1_gpu2

launch_one trace_bridge_qwen3_instruct_viewteacher stage2_conservative 4 \
  trace_bridge_bridgefull_viewteacher_stage2_conservative_gpu4_0708 \
  20260708_trace_bridge_bridgefull_viewteacher_stage2_conservative_gpu4

launch_one trace_bridge_qwen3_instruct_viewteacher stage2_strong 6 \
  trace_bridge_bridgefull_viewteacher_stage2_strong_gpu6_0708 \
  20260708_trace_bridge_bridgefull_viewteacher_stage2_strong_gpu6

launch_one trace_bridge_qwen3_instruct_vizstrong stage1 1 \
  trace_bridge_bridgefull_vizstrong_stage1_gpu1_0708 \
  20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1

launch_one trace_bridge_qwen3_instruct_vizstrong stage2_conservative 5 \
  trace_bridge_bridgefull_vizstrong_stage2_conservative_gpu5_0708 \
  20260708_trace_bridge_bridgefull_vizstrong_stage2_conservative_gpu5

launch_one trace_bridge_qwen3_instruct_vizstrong stage2_strong 7 \
  trace_bridge_bridgefull_vizstrong_stage2_strong_gpu7_0708 \
  20260708_trace_bridge_bridgefull_vizstrong_stage2_strong_gpu7

queue_one trace_bridge_bridgefull_viewteacher_stage1_gpu2_0708 \
  bridge_qwen3_instruct_hybrid_compact_anchor_gate baseline 2 \
  trace_bridge_bridgefull_bridge_baseline_gpu2_0708 \
  20260708_trace_bridge_bridgefull_bridge_baseline_gpu2

WATCH_SESSIONS="trace_bridge_bridgefull_viewteacher_stage1_gpu2_0708 trace_bridge_bridgefull_viewteacher_stage2_conservative_gpu4_0708 trace_bridge_bridgefull_viewteacher_stage2_strong_gpu6_0708 trace_bridge_bridgefull_vizstrong_stage1_gpu1_0708 trace_bridge_bridgefull_vizstrong_stage2_conservative_gpu5_0708 trace_bridge_bridgefull_vizstrong_stage2_strong_gpu7_0708 trace_bridge_bridgefull_bridge_baseline_gpu2_0708"

SESSION=trace_bridge_bridgefull_audit_waiter_0708 \
LOG="${AUDIT_OUT}/audit_waiter.log" \
OUT_DIR="${AUDIT_OUT}" \
WATCH_SESSIONS="${WATCH_SESSIONS}" \
RUN_REGEX="${RUN_REGEX}" \
bash launch_trace_bridge_audit_waiter_20260707.sh

SESSION=trace_bridge_bridgefull_status_watch_0708 \
OUT_DIR="${AUDIT_OUT}" \
WATCH_SESSIONS="${WATCH_SESSIONS}" \
STATUS_RUN_REGEX="${RUN_REGEX}" \
bash launch_trace_bridge_final_status_watch_20260707.sh

if tmux has-session -t trace_bridge_bridgefull_ckpt_snapshot_watch_0708 2>/dev/null; then
  echo "[skip] tmux session already exists: trace_bridge_bridgefull_ckpt_snapshot_watch_0708"
else
  tmux new-session -d -s trace_bridge_bridgefull_ckpt_snapshot_watch_0708 "
    set -euo pipefail
    cd /disk1/dingxukai/trace_colar
    echo '[snapshot-watch] started at '\$(date '+%F %T') | tee -a ${AUDIT_OUT}/ckpt_snapshot_watcher.log
    /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_bridge_ckpt_snapshot_watcher.py \
      --run-marker ${RUN_MARKER} \
      --interval-seconds 300 \
      --min-age-seconds 120 \
      --state ${AUDIT_OUT}/ckpt_snapshot_state.json \
      --manifest ${AUDIT_OUT}/ckpt_snapshots.jsonl \
      --watch-sessions ${WATCH_SESSIONS} \
      2>&1 | tee -a ${AUDIT_OUT}/ckpt_snapshot_watcher.log
  "
  echo "[launched] trace_bridge_bridgefull_ckpt_snapshot_watch_0708"
fi

SESSION=trace_bridge_bridgefull_snapshot_eval_after_0708 \
ROOT="${ROOT}" \
AUDIT_OUT="${AUDIT_OUT}" \
RUN_REGEX="${RUN_REGEX}" \
WATCH_SESSIONS="${WATCH_SESSIONS} trace_bridge_bridgefull_ckpt_snapshot_watch_0708" \
MANIFEST_GLOB="${ROOT}/20260708_trace_bridge_bridgefull_*/manifest.txt" \
SNAPSHOT_EVAL_MAX_PER_RUN=12 \
bash launch_trace_bridge_snapshot_eval_after_20260707.sh

{
  echo "# TRACE-BRIDGE Full Run 20260708"
  echo
  echo "- marker: ${RUN_MARKER}"
  echo "- audit: ${AUDIT_OUT}"
  echo "- regex: ${RUN_REGEX}"
  echo "- variants: viewteacher, vizstrong"
  echo "- modes: stage1, stage2_conservative, stage2_strong"
  echo "- baseline: bridge_qwen3_instruct_hybrid_compact_anchor_gate"
  echo "- test_times: 1"
  echo "- outputs: /disk1/dingxukai/trace_colar/run_outputs/trace_bridge"
  echo "- roots: /disk1/dingxukai/trace_colar/run_roots/trace_bridge"
} > /disk1/dingxukai/trace_colar/run_outputs/trace_bridge/BRIDGEFULL_RUN_20260708.md

tmux ls | grep trace_bridge_bridgefull || true
