#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

SESSION="${SESSION:-trace_bridge_viewteacher_queue_gpu7_0707}"
WAIT_SESSION="${WAIT_SESSION:-trace_bridge_baseline_gpu7_0707}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/20260707_trace_bridge_baseline_gpu7/summary_bridge_baseline.md}"
RECOVERY_SESSION="${RECOVERY_SESSION:-trace_bridge_recovery_waiter_0707}"
GPU="${GPU:-7}"
TRACE_MODEL="${TRACE_MODEL:-trace_bridge_qwen3_instruct_viewteacher}"
RUN_TAG="${RUN_TAG:-20260707_trace_bridge_viewteacher_gpu${GPU}}"
OUT_DIR="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/${RUN_TAG}"
LOG="${OUT_DIR}.outer.log"
STAGE1_BEST_FILE="${OUT_DIR}/${RUN_TAG}_stage1_best_ckpt.txt"

mkdir -p "$(dirname "${LOG}")"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  set -euo pipefail
  cd /disk1/dingxukai/trace_colar
  echo '[wait] waiting for ${WAIT_SESSION} before launching viewteacher on GPU=${GPU}' | tee -a ${LOG}
  while tmux has-session -t ${WAIT_SESSION} 2>/dev/null; do sleep 300; done
  echo '[wait] waiting for baseline summary or recovery completion before using GPU=${GPU}' | tee -a ${LOG}
  while [[ ! -s ${BASELINE_SUMMARY} ]] && tmux has-session -t ${RECOVERY_SESSION} 2>/dev/null; do sleep 300; done
  echo '[stage1] launching viewteacher Stage1 at '\$(date '+%F %T') | tee -a ${LOG}
  TRACE_MODEL=${TRACE_MODEL} MODE=stage1 GPU=${GPU} RUN_TAG=${RUN_TAG} TEST_TIMES=1 bash run_trace_bridge_pipeline_snapshot_20260707.sh 2>&1 | tee -a ${LOG}
  while [[ ! -s ${STAGE1_BEST_FILE} ]]; do sleep 60; done
  STAGE1_CKPT=\$(cat ${STAGE1_BEST_FILE})
  echo \"[stage2] launching viewteacher Stage2 both from \${STAGE1_CKPT}\" | tee -a ${LOG}
  TRACE_MODEL=${TRACE_MODEL} MODE=stage2_both_from_ckpt GPU=${GPU} RUN_TAG=${RUN_TAG}_stage2_from_stage1 TEST_TIMES=1 STAGE1_CKPT=\"\${STAGE1_CKPT}\" bash run_trace_bridge_pipeline_snapshot_20260707.sh 2>&1 | tee -a ${LOG}
"

echo "[launched-waiter] ${SESSION}: waits for ${WAIT_SESSION}, then runs viewteacher Stage1+Stage2 on GPU=${GPU}"
tmux ls | grep trace_bridge || true
