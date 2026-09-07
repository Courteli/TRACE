#!/usr/bin/env bash
set -euo pipefail

cd "/disk1/dingxukai/trace_colar"

# Usage examples:
#   GPU=7 ./run_trace_trajectory_ablation_20260704.sh full
#   GPU=7 ./run_trace_trajectory_ablation_20260704.sh all
#
# This matrix keeps the method story clean:
#   full        = answer + state + transition
#   no_state    = answer + transition
#   no_trans    = answer + state
#   answer_only = fixed latent answer training without trajectory alignment
#   k4/k16/k40  = latent-budget sensitivity for the same objective

TARGET="${1:-all}"
GPU="${GPU:-7}"
TEST_TIMES="${TEST_TIMES:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"

run_variant() {
  local name="$1"
  local k="$2"
  local ans_w="$3"
  local state_w="$4"
  local trans_w="$5"
  echo "[TRACE trajectory] launching ${name} on GPU ${GPU} with K=${k}"
  GPU="${GPU}" \
  TRACE_STEPS="${k}" \
  ANSWER_WEIGHT="${ans_w}" \
  STATE_WEIGHT="${state_w}" \
  TRANSITION_WEIGHT="${trans_w}" \
  ROLE_WEIGHT=0.0 \
  MAX_EPOCHS="${MAX_EPOCHS}" \
  TEST_TIMES="${TEST_TIMES}" \
  LOG_SUFFIX="trace_trajectory_${name}_qwen3_c5_k${k}_gpu${GPU}" \
    bash run_trace_trajectory_core_qwen3_c5_gpu7.sh
}

case "${TARGET}" in
  full)
    run_variant full 8 1.0 1.0 1.0
    ;;
  no_state)
    run_variant no_state 8 1.0 0.0 1.0
    ;;
  no_trans)
    run_variant no_trans 8 1.0 1.0 0.0
    ;;
  answer_only)
    run_variant answer_only 8 1.0 0.0 0.0
    ;;
  k4)
    run_variant k4 4 1.0 1.0 1.0
    ;;
  k16)
    run_variant k16 16 1.0 1.0 1.0
    ;;
  k40)
    run_variant k40 40 1.0 1.0 1.0
    ;;
  all)
    run_variant full 8 1.0 1.0 1.0
    run_variant no_state 8 1.0 0.0 1.0
    run_variant no_trans 8 1.0 1.0 0.0
    run_variant answer_only 8 1.0 0.0 0.0
    run_variant k4 4 1.0 1.0 1.0
    run_variant k16 16 1.0 1.0 1.0
    run_variant k40 40 1.0 1.0 1.0
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    exit 2
    ;;
esac
