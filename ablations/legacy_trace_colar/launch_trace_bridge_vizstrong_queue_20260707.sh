#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

SESSION="${SESSION:-trace_bridge_final_vizstrong_stage1_gpu2_0707}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-trace_bridge_final_structmv_stage1_gpu2_0707}"
TRACE_MODEL="${TRACE_MODEL:-trace_bridge_qwen3_instruct_vizstrong}"
MODE="${MODE:-stage1}"
GPU="${GPU:-2}"
RUN_TAG="${RUN_TAG:-20260707_trace_bridge_final_vizstrong_stage1_gpu2}"
SNAPSHOT="${SNAPSHOT:-run_trace_bridge_pipeline_snapshot_20260707.sh}"
OUT_DIR="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/${RUN_TAG}"
LOG="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/${RUN_TAG}.outer.log"
AUDIT_OUT="${AUDIT_OUT:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_final_20260707}"
AUDIT_REGEX="${AUDIT_REGEX:-20260707_trace_bridge_final_(structmv|viewteacher|vizstrong)_(stage1|stage2_conservative|stage2_strong)_gpu(1|2|4|5|6|7)}"

mkdir -p "$(dirname "${LOG}")" "${OUT_DIR}" "${AUDIT_OUT}"

if [[ ! -s "${OUT_DIR}/manifest.txt" ]]; then
  {
    printf 'mode=queued_%s\n' "${MODE}"
    printf 'gpu=%s\n' "${GPU}"
    printf 'run_tag=%s\n' "${RUN_TAG}"
    printf 'trace_model=%s\n' "${TRACE_MODEL}"
    printf 'wait_for_session=%s\n' "${WAIT_FOR_SESSION}"
    printf 'test_times=1\n'
    printf 'created_at=%s\n' "$(date '+%F %T')"
  } > "${OUT_DIR}/manifest.txt"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  set -euo pipefail
  cd /disk1/dingxukai/trace_colar
  echo '[queue] waiting for ${WAIT_FOR_SESSION} before launching ${RUN_TAG} at '\$(date '+%F %T') | tee -a ${LOG}
  while tmux has-session -t ${WAIT_FOR_SESSION} 2>/dev/null; do
    sleep 600
  done
  echo '[queue] launching ${RUN_TAG} on GPU=${GPU} at '\$(date '+%F %T') | tee -a ${LOG}
  TRACE_MODEL=${TRACE_MODEL} MODE=${MODE} GPU=${GPU} RUN_TAG=${RUN_TAG} TEST_TIMES=1 bash ${SNAPSHOT} 2>&1 | tee -a ${LOG}
  /home/dingxukai/miniconda3/envs/ROT/bin/python /disk1/dingxukai/trace_colar/tools/trace_bridge_collect_results.py --run-regex '${AUDIT_REGEX}' --out ${AUDIT_OUT} 2>&1 | tee -a ${LOG}
"

echo "[launched-queue] ${SESSION}: waits for ${WAIT_FOR_SESSION}, then MODEL=${TRACE_MODEL} MODE=${MODE} GPU=${GPU} RUN_TAG=${RUN_TAG}"
tmux ls | grep trace_bridge_final || true
