#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

SESSION="${SESSION:-trace_bridge_final_full_ckpt_snapshot_watch_0707}"
OUT_DIR="${OUT_DIR:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_final_full_20260707}"
LOG="${LOG:-${OUT_DIR}/ckpt_snapshot_watcher.log}"
PYTHON="${PYTHON:-/home/dingxukai/miniconda3/envs/ROT/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MIN_AGE_SECONDS="${MIN_AGE_SECONDS:-120}"
WATCH_SESSIONS="${WATCH_SESSIONS:-trace_bridge_final_full_structmv_stage1_gpu2_0707 trace_bridge_final_full_structmv_stage2_conservative_gpu4_0707 trace_bridge_final_full_structmv_stage2_strong_gpu6_0707 trace_bridge_final_full_viewteacher_stage1_gpu7_0707 trace_bridge_final_full_viewteacher_stage2_conservative_gpu5_0707 trace_bridge_final_full_viewteacher_stage2_strong_gpu1_0707 trace_bridge_final_full_bridge_baseline_gpu7_0707 trace_bridge_final_full_vizstrong_stage1_gpu2_0707}"

mkdir -p "${OUT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  set -euo pipefail
  cd /disk1/dingxukai/trace_colar
  echo '[snapshot-watch] started at '\$(date '+%F %T') | tee -a ${LOG}
  ${PYTHON} tools/trace_bridge_ckpt_snapshot_watcher.py \
    --interval-seconds ${INTERVAL_SECONDS} \
    --min-age-seconds ${MIN_AGE_SECONDS} \
    --watch-sessions ${WATCH_SESSIONS} \
    2>&1 | tee -a ${LOG}
"

echo "[launched] ${SESSION}"
tmux ls | grep trace_bridge_final || true
