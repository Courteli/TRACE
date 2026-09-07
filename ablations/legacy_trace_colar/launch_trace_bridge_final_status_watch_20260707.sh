#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-trace_bridge_final_status_watch_0707}"
OUT_DIR="${OUT_DIR:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_final_20260707}"
LOG="${LOG:-${OUT_DIR}/final_status_watch.log}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
STATUS_TOOL="${STATUS_TOOL:-/disk1/dingxukai/trace_colar/tools/trace_bridge_final_status.py}"
PYTHON="${PYTHON:-/home/dingxukai/miniconda3/envs/ROT/bin/python}"
WATCH_SESSIONS="${WATCH_SESSIONS:-trace_bridge_final_structmv_stage1_gpu2_0707 trace_bridge_final_structmv_stage2_conservative_gpu4_0707 trace_bridge_final_structmv_stage2_strong_gpu6_0707 trace_bridge_final_viewteacher_stage1_gpu7_0707 trace_bridge_final_viewteacher_stage2_conservative_gpu5_0707 trace_bridge_final_viewteacher_stage2_strong_gpu1_0707 trace_bridge_final_vizstrong_stage1_gpu2_0707}"
STATUS_ARGS="${STATUS_ARGS:-}"
STATUS_RUN_REGEX="${STATUS_RUN_REGEX:-}"
STATUS_ARGS_CMD="${STATUS_ARGS}"
if [[ -n "${STATUS_RUN_REGEX}" ]]; then
  STATUS_ARGS_CMD="--run-regex '${STATUS_RUN_REGEX}'"
fi

mkdir -p "${OUT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  set -euo pipefail
  while true; do
    {
      echo
      echo '===== final status '\$(date '+%F %T')' ====='
      ${PYTHON} ${STATUS_TOOL} ${STATUS_ARGS_CMD}
    } 2>&1 | tee -a ${LOG}

    alive=0
    for s in ${WATCH_SESSIONS}; do
      if tmux has-session -t \${s} 2>/dev/null; then
        alive=1
      fi
    done
    if [[ \${alive} -eq 0 ]]; then
      echo '[watch] all final experiment sessions ended at '\$(date '+%F %T') | tee -a ${LOG}
      break
    fi
    sleep ${INTERVAL_SECONDS}
  done
"

echo "[launched-watcher] ${SESSION}"
tmux ls | grep trace_bridge_final || true
