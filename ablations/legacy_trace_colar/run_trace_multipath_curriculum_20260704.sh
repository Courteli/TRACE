#!/usr/bin/env bash
set -euo pipefail

cd "/disk1/dingxukai/trace_colar"

GPU="${GPU:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TEST_TIMES="${TEST_TIMES:-5}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-512}"
MIN_L="${MIN_L:-0}"
EPOCHS_L8="${EPOCHS_L8:-1}"
EPOCHS_L16="${EPOCHS_L16:-1}"
EPOCHS_L40="${EPOCHS_L40:-3}"
FILTER_CANDIDATE_COUNT="${FILTER_CANDIDATE_COUNT:-0}"
FILTER_MAX_BATCHES="${FILTER_MAX_BATCHES:-0}"
RESAMPLE_INFORMATIVE_GROUPS="${RESAMPLE_INFORMATIVE_GROUPS:-True}"
RESAMPLE_MAX_ATTEMPTS="${RESAMPLE_MAX_ATTEMPTS:-3}"
DO_TEST_INTERMEDIATE="${DO_TEST_INTERMEDIATE:-False}"
DO_TEST_FINAL="${DO_TEST_FINAL:-False}"
LIMIT_VAL_BATCHES_INTERMEDIATE="${LIMIT_VAL_BATCHES_INTERMEDIATE:-0}"
LIMIT_VAL_BATCHES_FINAL="${LIMIT_VAL_BATCHES_FINAL:-0}"
INIT_CKPT="${INIT_CKPT:-logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260525-172642_884973_origin_colar_qwen3_c5_cotinit_rawgsm_lr3e-5_50epoch_gpu5_20260525/checkpoints/epoch15__step107616__monitor0.246.ckpt}"

find_ckpt() {
  local run_contains="$1"
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_find_checkpoint.py \
    --log_root logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl \
    --run_contains "${run_contains}" \
    --prefer best
}

run_stage() {
  local name="$1"
  local max_l="$2"
  local epochs="$3"
  local init_ckpt="$4"
  local do_test="$5"
  local limit_val_batches="$6"
  local suffix="trace_multipath_${name}_qwen3_c5_L${max_l}_g${GROUP_SIZE}_gpu${GPU}"
  echo "[TRACE MultiPath curriculum] stage=${name} #L=${max_l} init=${init_ckpt}"
  GPU="${GPU}" \
  MAX_L="${max_l}" \
  MIN_L="${MIN_L}" \
  GROUP_SIZE="${GROUP_SIZE}" \
  MAX_EPOCHS="${epochs}" \
  TEST_TIMES="${TEST_TIMES}" \
  N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES}" \
  FILTER_CANDIDATE_COUNT="${FILTER_CANDIDATE_COUNT}" \
  FILTER_MAX_BATCHES="${FILTER_MAX_BATCHES}" \
  RESAMPLE_INFORMATIVE_GROUPS="${RESAMPLE_INFORMATIVE_GROUPS}" \
  RESAMPLE_MAX_ATTEMPTS="${RESAMPLE_MAX_ATTEMPTS}" \
  DO_TEST="${do_test}" \
  LIMIT_VAL_BATCHES="${limit_val_batches}" \
  INIT_CKPT="${init_ckpt}" \
  LOG_SUFFIX="${suffix}" \
    bash run_trace_multipath_rl_qwen3_c5_gpu4.sh
}

run_stage full 8 "${EPOCHS_L8}" "${INIT_CKPT}" "${DO_TEST_INTERMEDIATE}" "${LIMIT_VAL_BATCHES_INTERMEDIATE}"
CKPT_L8="$(find_ckpt "trace_multipath_full_qwen3_c5_L8_g${GROUP_SIZE}_gpu${GPU}")"
echo "[TRACE MultiPath curriculum] L8 ckpt=${CKPT_L8}"

run_stage full 16 "${EPOCHS_L16}" "${CKPT_L8}" "${DO_TEST_INTERMEDIATE}" "${LIMIT_VAL_BATCHES_INTERMEDIATE}"
CKPT_L16="$(find_ckpt "trace_multipath_full_qwen3_c5_L16_g${GROUP_SIZE}_gpu${GPU}")"
echo "[TRACE MultiPath curriculum] L16 ckpt=${CKPT_L16}"

run_stage full 40 "${EPOCHS_L40}" "${CKPT_L16}" "${DO_TEST_FINAL}" "${LIMIT_VAL_BATCHES_FINAL}"
CKPT_L40="$(find_ckpt "trace_multipath_full_qwen3_c5_L40_g${GROUP_SIZE}_gpu${GPU}")"
echo "[TRACE MultiPath curriculum] L40 ckpt=${CKPT_L40}"

/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_collect_results.py
