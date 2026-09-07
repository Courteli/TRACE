#!/usr/bin/env bash
set -euo pipefail

cd "/disk1/dingxukai/trace_colar"

TARGET="${1:-all}"
GPU="${GPU:-4}"
TEST_TIMES="${TEST_TIMES:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
GROUP_SIZE="${GROUP_SIZE:-8}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-512}"
MIN_L="${MIN_L:-0}"

run_variant() {
  local name="$1"
  local max_l="$2"
  local trace_w="$3"
  local mode_w="$4"
  local hard_w="$5"
  local step_w="$6"
  local noncollapse_w="$7"
  echo "[TRACE MultiPath] ${name}: GPU=${GPU} #L=${max_l} group=${GROUP_SIZE}"
  GPU="${GPU}" \
  MAX_L="${max_l}" \
  MIN_L="${MIN_L}" \
  GROUP_SIZE="${GROUP_SIZE}" \
  N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES}" \
  MAX_EPOCHS="${MAX_EPOCHS}" \
  TEST_TIMES="${TEST_TIMES}" \
  TRACE_REWARD_WEIGHT="${trace_w}" \
  MODE_DIVERSITY_WEIGHT="${mode_w}" \
  HARD_REPULSION_WEIGHT="${hard_w}" \
  STEP_COHERENCE_WEIGHT="${step_w}" \
  NONCOLLAPSE_WEIGHT="${noncollapse_w}" \
  LOG_SUFFIX="trace_multipath_${name}_qwen3_c5_L${max_l}_g${GROUP_SIZE}_gpu${GPU}" \
    bash run_trace_multipath_rl_qwen3_c5_gpu4.sh
}

case "${TARGET}" in
  full)
    run_variant full 40 0.25 0.25 1.0 0.05 0.05
    ;;
  answer_only)
    run_variant answer_only 40 0.0 0.0 0.0 0.0 0.0
    ;;
  no_mode)
    run_variant no_mode 40 0.25 0.0 1.0 0.05 0.05
    ;;
  no_hard)
    run_variant no_hard 40 0.25 0.25 0.0 0.05 0.05
    ;;
  no_step)
    run_variant no_step 40 0.25 0.25 1.0 0.0 0.0
    ;;
  L8)
    run_variant L8 8 0.25 0.25 1.0 0.05 0.05
    ;;
  L16)
    run_variant L16 16 0.25 0.25 1.0 0.05 0.05
    ;;
  L40)
    run_variant L40 40 0.25 0.25 1.0 0.05 0.05
    ;;
  all)
    run_variant full 40 0.25 0.25 1.0 0.05 0.05
    run_variant answer_only 40 0.0 0.0 0.0 0.0 0.0
    run_variant no_mode 40 0.25 0.0 1.0 0.05 0.05
    run_variant no_hard 40 0.25 0.25 0.0 0.05 0.05
    run_variant no_step 40 0.25 0.25 1.0 0.0 0.0
    run_variant L8 8 0.25 0.25 1.0 0.05 0.05
    run_variant L16 16 0.25 0.25 1.0 0.05 0.05
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    exit 2
    ;;
esac
