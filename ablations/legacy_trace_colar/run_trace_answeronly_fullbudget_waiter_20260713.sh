#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
OUT=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_answeronly_fullbudget_control
STAGE1_CKPT=/disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_284260_20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1_stage1/checkpoints/epoch1__step13452__monitor0.661.ckpt
POLL_SECONDS="${POLL_SECONDS:-300}"
FREE_MEMORY_MIB="${FREE_MEMORY_MIB:-500}"

mkdir -p "${OUT}"
cd "${ROOT}"

if [[ -f "${OUT}/pipeline_done.txt" ]]; then
  exit 0
fi

while true; do
  mapfile -t free_gpus < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v threshold="${FREE_MEMORY_MIB}" '
          {
            gsub(/ /, "", $1);
            gsub(/ /, "", $2);
            if ($1 != 6 && $2 + 0 < threshold) print $1;
          }
        '
  )
  {
    printf '%s free_allowed_gpus=' "$(date '+%F %T')"
    printf '%s ' "${free_gpus[@]:-none}"
    printf '\n'
  } >> "${OUT}/waiter.log"
  if (( ${#free_gpus[@]} >= 4 )); then
    selected=("${free_gpus[@]:0:4}")
    selected_csv="$(IFS=,; printf '%s' "${selected[*]}")"
    printf '%s\n' "${selected_csv}" > "${OUT}/selected_physical_gpus.txt"
    break
  fi
  sleep "${POLL_SECONDS}"
done

env \
  MODE=stage2_answer_only_from_ckpt \
  RUN_TAG=20260713_trace_bridge_answeronly_fullbudget_control \
  OUT_DIR="${OUT}" \
  ROOT_DIR="${ROOT}/run_roots/trace_bridge/20260713_trace_bridge_answeronly_fullbudget_control" \
  TRACE_MODEL=trace_bridge_qwen3_instruct_vizstrong \
  STAGE1_CKPT="${STAGE1_CKPT}" \
  GPU="${selected_csv}" \
  TRACE_TRAIN_DEVICES=0,1,2,3 \
  TRACE_TRAIN_STRATEGY=ddp_find_unused_parameters_true \
  TRACE_TRAIN_BATCH_SIZE=4 \
  TRACE_RL_EXP_BATCH_SIZE=1 \
  TRACE_RL_WARMUP_STEPS=75 \
  TRACE_RL_TRAINING_STEPS=5120 \
  TRACE_SAVE_TOP_K=-1 \
  TRACE_CHECKPOINT_FILENAME='epoch{epoch}__step{step}__monitor{monitor:.6f}' \
  TEST_TIMES=1 \
  bash "${ROOT}/run_trace_bridge_pipeline_snapshot_20260707.sh" \
  2>&1 | tee "${OUT}/pipeline.log"

printf 'Full-budget answer-only control completed.\n' > "${OUT}/pipeline_done.txt"
