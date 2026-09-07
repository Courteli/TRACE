#!/usr/bin/env bash
set -euo pipefail

POST_GPU="${1:-7}"
POST_SAMPLES="${2:-64}"
DIAG_GROUP_SIZE="${DIAG_GROUP_SIZE:-8}"
TEST_TIMES="${TEST_TIMES:-5}"
TRACE_OOD_MAX_SAMPLES="${TRACE_OOD_MAX_SAMPLES:-0}"
ROOT="/disk1/dingxukai/trace_colar"
RUN_OUTPUTS="${ROOT}/run_outputs/trace"
WATCH_SESSION="${WATCH_SESSION:-trace_auto_post_after_rl}"
V2_SESSION="${V2_SESSION:-trace_v2_rl_qwen3_c5_gpu0}"
ANSWER_SESSION="${ANSWER_SESSION:-trace_answer_only_rl_qwen3_c5_gpu4}"

mkdir -p "${RUN_OUTPUTS}"

if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  echo "${WATCH_SESSION} already exists."
  exit 0
fi

tmux new-session -d -s "${WATCH_SESSION}" \
  "set -euo pipefail; \
   cd '${ROOT}'; \
   echo '[TRACE auto-post] waiting for RL sessions to appear' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'; \
   until tmux has-session -t '${V2_SESSION}' 2>/dev/null && tmux has-session -t '${ANSWER_SESSION}' 2>/dev/null; do sleep 120; done; \
   echo '[TRACE auto-post] waiting for RL sessions to finish' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'; \
   while tmux has-session -t '${V2_SESSION}' 2>/dev/null || tmux has-session -t '${ANSWER_SESSION}' 2>/dev/null; do sleep 300; done; \
   echo '[TRACE auto-post] launching post-v2 validation' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'; \
   V2_RUN='${V2_SESSION}' ANSWER_RUN='${ANSWER_SESSION}' DIAG_GROUP_SIZE='${DIAG_GROUP_SIZE}' TEST_TIMES='${TEST_TIMES}' TRACE_OOD_MAX_SAMPLES='${TRACE_OOD_MAX_SAMPLES}' POST_SESSION='trace_post_v2_validate_${WATCH_SESSION}_gpu${POST_GPU}' bash scripts/trace_post_v2_validate.sh '${POST_GPU}' '${POST_SAMPLES}'; \
   echo '[TRACE auto-post] post-v2 validation watcher launched' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'"

echo "Started ${WATCH_SESSION}"
