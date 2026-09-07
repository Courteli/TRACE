#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-trace_bridge_audit_waiter_0707}"
LOG="${LOG:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_20260707/audit_waiter.log}"
OUT_DIR="${OUT_DIR:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_20260707}"
RUN_REGEX="${RUN_REGEX:-20260707_trace_bridge_(baseline_gpu7|stage1_v2_gpu2|stage2_conservative_v2_gpu4|stage2_strong_v2_gpu6|structmv_stage1_gpu1|structmv_stage2_from_stage1_gpu1|viewteacher_gpu7|viewteacher_gpu7_stage2_from_stage1)}"
WATCH_SESSIONS="${WATCH_SESSIONS:-trace_bridge_baseline_gpu7_0707 trace_bridge_stage1_v2_gpu2_0707 trace_bridge_stage2_conservative_v2_gpu4_0707 trace_bridge_stage2_strong_v2_gpu6_0707 trace_bridge_structmv_stage1_gpu1_0707 trace_bridge_structmv_stage2_from_stage1_gpu1_0707 trace_bridge_viewteacher_queue_gpu7_0707 trace_bridge_recovery_waiter_0707}"

mkdir -p "${OUT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  set -euo pipefail
  echo '[wait] audit waiter started at '\$(date '+%F %T') | tee -a ${LOG}
  while true; do
    alive=0
    for s in ${WATCH_SESSIONS}; do
      if tmux has-session -t \${s} 2>/dev/null; then alive=1; fi
    done
    if [[ \${alive} -eq 0 ]]; then break; fi
    /home/dingxukai/miniconda3/envs/ROT/bin/python /disk1/dingxukai/trace_colar/tools/trace_bridge_collect_results.py --run-regex '${RUN_REGEX}' --out ${OUT_DIR} >/dev/null 2>&1 || true
    sleep 600
  done
  echo '[audit] experiments finished at '\$(date '+%F %T') | tee -a ${LOG}
  /home/dingxukai/miniconda3/envs/ROT/bin/python /disk1/dingxukai/trace_colar/tools/trace_bridge_collect_results.py --run-regex '${RUN_REGEX}' --out ${OUT_DIR} 2>&1 | tee -a ${LOG}
"

echo "[launched-waiter] ${SESSION}"
tmux ls | grep trace_bridge || true
