#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

TRACE_MODEL="${TRACE_MODEL:-trace_bridge_qwen3_instruct_structmv}"
GPU="${GPU:-1}"
STAGE1_SESSION="${STAGE1_SESSION:-trace_bridge_structmv_stage1_gpu1_0707}"
STAGE1_OUT_DIR="${STAGE1_OUT_DIR:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/20260707_trace_bridge_structmv_stage1_gpu1}"
STAGE1_BEST_FILE="${STAGE1_BEST_FILE:-${STAGE1_OUT_DIR}/20260707_trace_bridge_structmv_stage1_gpu1_stage1_best_ckpt.txt}"
RUN_TAG="${RUN_TAG:-20260707_trace_bridge_structmv_stage2_from_stage1_gpu${GPU}}"
SESSION="${SESSION:-trace_bridge_structmv_stage2_from_stage1_gpu${GPU}_0707}"
LOG="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/${RUN_TAG}.outer.log"

mkdir -p "$(dirname "${LOG}")"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  set -euo pipefail
  cd /disk1/dingxukai/trace_colar
  echo '[wait] waiting for ${STAGE1_SESSION} to finish and ${STAGE1_BEST_FILE} to exist' | tee -a ${LOG}
  while tmux has-session -t ${STAGE1_SESSION} 2>/dev/null; do sleep 300; done
  while [[ ! -s ${STAGE1_BEST_FILE} ]]; do sleep 300; done
  STAGE1_CKPT=\$(cat ${STAGE1_BEST_FILE})
  echo \"[stage2] using Stage1 ckpt: \${STAGE1_CKPT}\" | tee -a ${LOG}
  TRACE_MODEL=${TRACE_MODEL} MODE=stage2_both_from_ckpt GPU=${GPU} RUN_TAG=${RUN_TAG} TEST_TIMES=1 STAGE1_CKPT=\"\${STAGE1_CKPT}\" bash run_trace_bridge_pipeline_snapshot_20260707.sh 2>&1 | tee -a ${LOG}
"

echo "[launched-waiter] ${SESSION}: waits for ${STAGE1_SESSION}, then runs Stage2 both on GPU=${GPU}"
tmux ls | grep trace_bridge || true
