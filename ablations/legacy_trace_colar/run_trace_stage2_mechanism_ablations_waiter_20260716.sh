#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
OUT=${ROOT}/run_outputs/trace/20260716_stage2_mechanism_ablations
RUN_ROOT=${ROOT}/run_roots/trace/20260716_stage2_mechanism_ablations
PIPELINE=${ROOT}/run_trace_bridge_pipeline_snapshot_20260707.sh
FULL_STAGE1_CKPT=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_284260_20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1_stage1/checkpoints/epoch1__step13452__monitor0.661.ckpt
NO_PATH_BEST_FILE=${ROOT}/run_outputs/trace/20260716_stage1_component_ablations_gpu0/no_path_consistency/best_checkpoint.txt
POLL_SECONDS=${POLL_SECONDS:-300}
FREE_MEMORY_MIB=${FREE_MEMORY_MIB:-500}
MAX_EPOCHS=${STAGE2_MAX_EPOCHS:-5}

mkdir -p "${OUT}" "${RUN_ROOT}"
cd "${ROOT}"

exec 9>"${OUT}/pipeline.lock"
if ! flock -n 9; then
  printf '[skip] another Stage 2 mechanism-ablation waiter is active\n'
  exit 0
fi
printf '%s\n' "$$" > "${OUT}/pipeline.pid"

timestamp() {
  date '+%F %T'
}

wait_for_four_gpus() {
  local free_gpus=()
  while true; do
    mapfile -t free_gpus < <(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' -v threshold="${FREE_MEMORY_MIB}" '
            {
              gsub(/ /, "", $1);
              gsub(/ /, "", $2);
              if ($1 != 6 && $2 + 0 <= threshold) print $1;
            }
          '
    )
    printf '%s [wait-gpus] free=' "$(timestamp)" >> "${OUT}/pipeline_status.log"
    if (( ${#free_gpus[@]} )); then
      printf '%s' "${free_gpus[0]}" >> "${OUT}/pipeline_status.log"
      for gpu in "${free_gpus[@]:1}"; do
        printf ',%s' "${gpu}" >> "${OUT}/pipeline_status.log"
      done
    else
      printf 'none' >> "${OUT}/pipeline_status.log"
    fi
    printf '\n' >> "${OUT}/pipeline_status.log"
    if (( ${#free_gpus[@]} >= 4 )); then
      SELECTED_GPUS=("${free_gpus[@]:0:4}")
      return
    fi
    sleep "${POLL_SECONDS}"
  done
}

wait_for_checkpoint_file() {
  local path="$1"
  while [[ ! -s "${path}" ]] || [[ ! -f "$(<"${path}" 2>/dev/null || true)" ]]; do
    printf '%s [wait-checkpoint] file=%s\n' "$(timestamp)" "${path}" \
      >> "${OUT}/pipeline_status.log"
    sleep "${POLL_SECONDS}"
  done
}

run_ablation() {
  local label="$1"
  local stage1_ckpt="$2"
  local replay_weight="$3"
  local variant_out="${OUT}/${label}"
  local variant_root="${RUN_ROOT}/${label}"
  if [[ -f "${variant_out}/pipeline_done.txt" ]]; then
    printf '%s [skip] completed variant=%s\n' "$(timestamp)" "${label}" \
      >> "${OUT}/pipeline_status.log"
    return
  fi

  wait_for_four_gpus
  local selected_csv
  selected_csv="$(IFS=,; printf '%s' "${SELECTED_GPUS[*]}")"
  mkdir -p "${variant_out}"
  {
    printf 'experiment=TRACE_Stage2_mechanism_ablation\n'
    printf 'variant=%s\n' "${label}"
    printf 'stage1_checkpoint_sha256=%s\n' "$(sha256sum "${stage1_ckpt}" | awk '{print $1}')"
    printf 'stage1_replay_weight=%s\n' "${replay_weight}"
    printf 'training_questions_per_epoch=2048\n'
    printf 'max_epochs=%s\n' "${MAX_EPOCHS}"
    printf 'validation_schedule=final_epoch_only\n'
    printf 'validation_every_n_epoch=%s\n' "${MAX_EPOCHS}"
    printf 'test_times=1\n'
    printf 'budget_revision=2026-07-18_user_requested_short_ablation\n'
    printf 'physical_gpus=%s\n' "${selected_csv}"
    printf 'started_at=%s\n' "$(timestamp)"
  } > "${variant_out}/PUBLIC_PROTOCOL.txt"
  printf '%s [start] variant=%s gpus=%s replay=%s\n' \
    "$(timestamp)" "${label}" "${selected_csv}" "${replay_weight}" \
    >> "${OUT}/pipeline_status.log"

  env \
    MODE=stage2_answer_only_from_ckpt \
    RUN_TAG="20260716_trace_stage2_${label}" \
    OUT_DIR="${variant_out}/internal_run" \
    ROOT_DIR="${variant_root}" \
    TRACE_MODEL=trace_bridge_qwen3_instruct_vizstrong \
    STAGE1_CKPT="${stage1_ckpt}" \
    GPU="${selected_csv}" \
    TRACE_TRAIN_DEVICES=0,1,2,3 \
    TRACE_TRAIN_STRATEGY=ddp_find_unused_parameters_true \
    TRACE_TRAIN_BATCH_SIZE=4 \
    TRACE_RL_EXP_BATCH_SIZE=1 \
    TRACE_RL_WARMUP_STEPS=75 \
    TRACE_RL_TRAINING_STEPS=5120 \
    TRACE_MAX_EPOCHS="${MAX_EPOCHS}" \
    TRACE_VAL_EVERY_N_EPOCH="${MAX_EPOCHS}" \
    TRACE_SAVE_TOP_K=-1 \
    TRACE_CHECKPOINT_FILENAME='epoch{epoch}__step{step}__monitor{monitor:.6f}' \
    TRACE_STAGE2_SFT_REPLAY_WEIGHT="${replay_weight}" \
    TEST_TIMES=1 \
    bash "${PIPELINE}" \
    > "${variant_out}/pipeline.log" 2>&1

  cp "${variant_out}/internal_run/stage2_answer_only_best_ckpt.txt" \
    "${variant_out}/best_checkpoint.txt"
  printf '%s\n' "$(timestamp)" > "${variant_out}/pipeline_done.txt"
  printf '%s [done] variant=%s\n' "$(timestamp)" "${label}" \
    >> "${OUT}/pipeline_status.log"
}

run_ablation no_stage1_replay "${FULL_STAGE1_CKPT}" 0.0
wait_for_checkpoint_file "${NO_PATH_BEST_FILE}"
run_ablation no_path_initialization "$(<"${NO_PATH_BEST_FILE}")" 0.05

printf '%s\n' "$(timestamp)" > "${OUT}/pipeline_done.txt"
printf '%s [done] all Stage 2 mechanism ablations complete\n' "$(timestamp)" \
  >> "${OUT}/pipeline_status.log"
