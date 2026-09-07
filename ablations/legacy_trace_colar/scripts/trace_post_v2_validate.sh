#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
SAMPLES="${2:-64}"
ROOT="/disk1/dingxukai/trace_colar"
RUN_OUTPUTS="${ROOT}/run_outputs/trace"
SESSION="${POST_SESSION:-trace_post_v2_validate_gpu${GPU}}"
V0_RUN="${V0_RUN:-trace_v0_sft_qwen3_c5_gpu0}"
V2_RUN="${V2_RUN:-trace_v2_rl_qwen3_c5_gpu0}"
ANSWER_RUN="${ANSWER_RUN:-trace_answer_only_rl_qwen3_c5_gpu4}"
DIAG_GROUP_SIZE="${DIAG_GROUP_SIZE:-8}"
TEST_TIMES="${TEST_TIMES:-5}"
TRACE_OOD_MAX_SAMPLES="${TRACE_OOD_MAX_SAMPLES:-0}"

mkdir -p "${RUN_OUTPUTS}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && V0_RUN='${V0_RUN}' V2_RUN='${V2_RUN}' ANSWER_RUN='${ANSWER_RUN}' DIAG_GROUP_SIZE='${DIAG_GROUP_SIZE}' TEST_TIMES='${TEST_TIMES}' TRACE_OOD_MAX_SAMPLES='${TRACE_OOD_MAX_SAMPLES}' bash scripts/trace_post_v2_validate_worker.sh '${GPU}' '${SAMPLES}' 2>&1 | tee -a '${RUN_OUTPUTS}/${SESSION}.log'"

echo "Started ${SESSION}"
