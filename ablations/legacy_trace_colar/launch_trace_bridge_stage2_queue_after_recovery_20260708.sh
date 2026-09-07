#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

GPU="${GPU:-7}"
ROOT="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge"
SNAPSHOT="${SNAPSHOT:-run_trace_bridge_pipeline_snapshot_20260707.sh}"
TRACE_RL_EXP_BATCH_SIZE="${TRACE_RL_EXP_BATCH_SIZE:-1}"

mkdir -p "${ROOT}"

launch_waiting_stage2() {
  local wait_for="$1"
  local session="$2"
  local model="$3"
  local mode="$4"
  local tag="$5"
  local ckpt="$6"
  local log="${ROOT}/${tag}.outer.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[skip] queued session already exists: ${session}"
    return
  fi

  tmux new-session -d -s "${session}" "
    set -euo pipefail
    cd /disk1/dingxukai/trace_colar
    echo '[stage2-queue] waiting for ${wait_for} before ${tag} at '\$(date '+%F %T') | tee -a ${log}
    while tmux has-session -t ${wait_for} 2>/dev/null; do
      sleep 600
    done
    echo '[stage2-queue] launching ${tag} on GPU=${GPU} at '\$(date '+%F %T') | tee -a ${log}
    STAGE1_CKPT=${ckpt} \
    TRACE_MODEL=${model} \
    MODE=${mode} \
    GPU=${GPU} \
    RUN_TAG=${tag} \
    TEST_TIMES=1 \
    TRACE_RL_EXP_BATCH_SIZE=${TRACE_RL_EXP_BATCH_SIZE} \
    bash ${SNAPSHOT} 2>&1 | tee -a ${log}
  "
  echo "[queued] ${session}: waits for ${wait_for}, then ${tag}"
}

FIRST="trace_bridge_bridgefull_recover_vizstrong_stage2_conservative_gpu7_0708"

launch_waiting_stage2 "${FIRST}" \
  trace_bridge_bridgefull_queue_vizstrong_stage2_strong_gpu7_0708 \
  trace_bridge_qwen3_instruct_vizstrong \
  stage2_strong_from_ckpt \
  20260708_trace_bridge_bridgefull_queue_vizstrong_stage2_strong_gpu7 \
  /disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_284260_20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1_stage1/checkpoints/epoch1__step13452__monitor0.661.ckpt

launch_waiting_stage2 trace_bridge_bridgefull_queue_vizstrong_stage2_strong_gpu7_0708 \
  trace_bridge_bridgefull_queue_viewteacher_stage2_conservative_gpu7_0708 \
  trace_bridge_qwen3_instruct_viewteacher \
  stage2_conservative_from_ckpt \
  20260708_trace_bridge_bridgefull_queue_viewteacher_stage2_conservative_gpu7 \
  /disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_viewteacher/qsa-gsm/20260708-003948_666327_20260708_trace_bridge_bridgefull_viewteacher_stage1_gpu2_stage1/checkpoints/epoch0__step6726__monitor0.653.ckpt

launch_waiting_stage2 trace_bridge_bridgefull_queue_viewteacher_stage2_conservative_gpu7_0708 \
  trace_bridge_bridgefull_queue_viewteacher_stage2_strong_gpu7_0708 \
  trace_bridge_qwen3_instruct_viewteacher \
  stage2_strong_from_ckpt \
  20260708_trace_bridge_bridgefull_queue_viewteacher_stage2_strong_gpu7 \
  /disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_viewteacher/qsa-gsm/20260708-003948_666327_20260708_trace_bridge_bridgefull_viewteacher_stage1_gpu2_stage1/checkpoints/epoch0__step6726__monitor0.653.ckpt

tmux ls | rg 'trace_bridge_bridgefull_(recover|queue)_' || true
