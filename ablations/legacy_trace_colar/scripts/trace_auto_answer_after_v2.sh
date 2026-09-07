#!/usr/bin/env bash
set -euo pipefail

V2_SESSION="${1:?Usage: bash scripts/trace_auto_answer_after_v2.sh <v2_session> <answer_gpu> <sft_ckpt> [post_gpu] [samples]}"
ANSWER_GPU="${2:?Usage: bash scripts/trace_auto_answer_after_v2.sh <v2_session> <answer_gpu> <sft_ckpt> [post_gpu] [samples]}"
SFT_CKPT="${3:?Usage: bash scripts/trace_auto_answer_after_v2.sh <v2_session> <answer_gpu> <sft_ckpt> [post_gpu] [samples]}"
POST_GPU="${4:-${ANSWER_GPU}}"
POST_SAMPLES="${5:-64}"
GROUP_SIZE="${GROUP_SIZE:-8}"
EXP_BATCH_SIZE="${EXP_BATCH_SIZE:-8}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-512}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
MAX_N_LATENT_FORWARD="${MAX_N_LATENT_FORWARD:-64}"
LR="${LR:-3e-5}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-1.0}"
TRACE_FILTER_MIXED="${TRACE_FILTER_MIXED:-False}"
TRACE_FILTER_CANDIDATE_FACTOR="${TRACE_FILTER_CANDIDATE_FACTOR:-1.0}"
TRACE_FILTER_CANDIDATE_COUNT="${TRACE_FILTER_CANDIDATE_COUNT:-0}"
TRACE_FILTER_BATCH_SIZE="${TRACE_FILTER_BATCH_SIZE:-1}"
TRACE_FILTER_MIXED_FILL_FRACTION="${TRACE_FILTER_MIXED_FILL_FRACTION:-0.5}"
ANSWER_TRACE_RESAMPLE_MIXED="${ANSWER_TRACE_RESAMPLE_MIXED:-False}"
ANSWER_TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS="${ANSWER_TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS:-4}"
ANSWER_TRACE_RESAMPLE_MIXED_TARGET_FRAC="${ANSWER_TRACE_RESAMPLE_MIXED_TARGET_FRAC:-1.0}"
TEST_TIMES="${TEST_TIMES:-5}"
TRACE_OOD_MAX_SAMPLES="${TRACE_OOD_MAX_SAMPLES:-0}"

ROOT="/disk1/dingxukai/trace_colar"
RUN_OUTPUTS="${ROOT}/run_outputs/trace"
WATCH_SESSION="trace_auto_answer_after_${V2_SESSION}"
ANSWER_SESSION="trace_answer_only_rl_qwen3_c5_gpu${ANSWER_GPU}"

mkdir -p "${RUN_OUTPUTS}"

if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  echo "${WATCH_SESSION} already exists."
  exit 0
fi

tmux new-session -d -s "${WATCH_SESSION}" \
  "set -euo pipefail; \
   cd '${ROOT}'; \
   echo '[TRACE answer-after-v2] waiting for ${V2_SESSION}' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'; \
   until tmux has-session -t '${V2_SESSION}' 2>/dev/null; do sleep 60; done; \
   while tmux has-session -t '${V2_SESSION}' 2>/dev/null; do sleep 300; done; \
   echo '[TRACE answer-after-v2] launching answer-only on GPU ${ANSWER_GPU}' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'; \
   GROUP_SIZE='${GROUP_SIZE}' EXP_BATCH_SIZE='${EXP_BATCH_SIZE}' N_TRAIN_SAMPLES='${N_TRAIN_SAMPLES}' MAX_EPOCHS='${MAX_EPOCHS}' MAX_N_LATENT_FORWARD='${MAX_N_LATENT_FORWARD}' LR='${LR}' LATENT_TEMPERATURE='${LATENT_TEMPERATURE}' TRACE_FILTER_MIXED='${TRACE_FILTER_MIXED}' TRACE_FILTER_CANDIDATE_FACTOR='${TRACE_FILTER_CANDIDATE_FACTOR}' TRACE_FILTER_CANDIDATE_COUNT='${TRACE_FILTER_CANDIDATE_COUNT}' TRACE_FILTER_BATCH_SIZE='${TRACE_FILTER_BATCH_SIZE}' TRACE_FILTER_MIXED_FILL_FRACTION='${TRACE_FILTER_MIXED_FILL_FRACTION}' TRACE_RESAMPLE_MIXED='${ANSWER_TRACE_RESAMPLE_MIXED}' TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS='${ANSWER_TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS}' TRACE_RESAMPLE_MIXED_TARGET_FRAC='${ANSWER_TRACE_RESAMPLE_MIXED_TARGET_FRAC}' bash scripts/trace_train_answer_only_rl.sh '${ANSWER_GPU}' '${SFT_CKPT}'; \
   until tmux has-session -t '${ANSWER_SESSION}' 2>/dev/null; do sleep 30; done; \
   while tmux has-session -t '${ANSWER_SESSION}' 2>/dev/null; do sleep 300; done; \
   echo '[TRACE answer-after-v2] launching post validation' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'; \
   V2_RUN='${V2_SESSION}' ANSWER_RUN='${ANSWER_SESSION}' DIAG_GROUP_SIZE='${GROUP_SIZE}' TEST_TIMES='${TEST_TIMES}' TRACE_OOD_MAX_SAMPLES='${TRACE_OOD_MAX_SAMPLES}' POST_SESSION='trace_post_v2_validate_${WATCH_SESSION}_gpu${POST_GPU}' bash scripts/trace_post_v2_validate.sh '${POST_GPU}' '${POST_SAMPLES}'; \
   echo '[TRACE answer-after-v2] done' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'"

echo "Started ${WATCH_SESSION}"
