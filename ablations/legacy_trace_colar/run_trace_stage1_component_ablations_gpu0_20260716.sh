#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
PY=/home/dingxukai/miniconda3/envs/ROT/bin/python
GPU=${GPU:-0}
PARETO_DONE=${ROOT}/run_outputs/trace/20260716_accuracy_length_pareto_gpu0/pipeline_done.txt
OUT=${ROOT}/run_outputs/trace/20260716_stage1_component_ablations_gpu0
RUN_ROOT=${ROOT}/run_roots/trace/20260716_stage1_component_ablations_gpu0
STAGE0_CKPT=/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/checkpoints/epoch0__step6726__monitor0.871.ckpt
TRAIN_DATA=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_EPOCHS=${STAGE1_MAX_EPOCHS:-5}
VAL_EVERY_N_EPOCH=${STAGE1_VAL_EVERY_N_EPOCH:-${MAX_EPOCHS}}

declare -A EVAL_DATASETS=(
  [gsm8k]="${TRAIN_DATA}"
  [gsmhard]="/home/dingxukai/RoT/data/GSM8k-Hard/readcot_qsa_test"
  [svamp]="/home/dingxukai/RoT/data/SVAMP/readcot_qsa_test"
  [multiarith]="/home/dingxukai/RoT/data/Multiarith/readcot_qsa_test"
)

mkdir -p "${OUT}" "${RUN_ROOT}"
cd "${ROOT}"

exec 9>"${OUT}/pipeline.lock"
if ! flock -n 9; then
  printf '[skip] another Stage 1 ablation pipeline is active\n'
  exit 0
fi

timestamp() {
  date '+%F %T'
}

wait_for_pareto() {
  while [[ ! -f "${PARETO_DONE}" ]]; do
    printf '%s [wait] accuracy-length Pareto is still active\n' "$(timestamp)" | tee -a "${OUT}/pipeline_status.log"
    sleep "${POLL_SECONDS}"
  done
}

wait_for_gpu() {
  local used
  while true; do
    used="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( used <= 500 )); then
      return
    fi
    printf '%s [wait] gpu=%s used=%sMiB\n' "$(timestamp)" "${GPU}" "${used}" | tee -a "${OUT}/pipeline_status.log"
    sleep "${POLL_SECONDS}"
  done
}

variant_overrides() {
  local variant="$1"
  case "${variant}" in
    no_path_consistency)
      printf '%s\n' \
        'model.model_kwargs.trace_bridge_config.stage1_path_weight=0.0' \
        'model.model_kwargs.trace_bridge_config.multiview_path_weight=0.0'
      ;;
    no_progress_anchor)
      printf '%s\n' \
        'model.model_kwargs.trace_bridge_config.stage1_progress_anchor_mix=0.0' \
        'model.model_kwargs.trace_bridge_config.multiview_teacher_progress_bias=0.0'
      ;;
    no_multiview)
      printf '%s\n' \
        'model.model_kwargs.trace_bridge_config.stage1_multiview_weight=0.0' \
        'model.model_kwargs.trace_bridge_config.use_multiview_teacher=false'
      ;;
    *)
      printf '[error] unknown variant: %s\n' "${variant}" >&2
      return 1
      ;;
  esac
}

train_variant() {
  local variant="$1"
  local base="${OUT}/${variant}"
  local logger="${base}/train_logs"
  local actual_logger=""
  local best_file="${base}/best_checkpoint.txt"
  local overrides=()
  mkdir -p "${base}"
  if [[ -s "${best_file}" ]] && [[ -f "$(<"${best_file}")" ]]; then
    printf '%s [skip] trained variant=%s\n' "$(timestamp)" "${variant}" | tee -a "${OUT}/pipeline_status.log"
    return
  fi
  mapfile -t overrides < <(variant_overrides "${variant}")
  wait_for_gpu
  printf '%s [train] variant=%s full_data=true max_epochs=%s validation=final_epoch_only\n' \
    "$(timestamp)" "${variant}" "${MAX_EPOCHS}" | tee -a "${OUT}/pipeline_status.log"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" run.py \
      --model trace_bridge_qwen3_instruct_vizstrong \
      --dataset qsa \
      --trainer trace_stage1_patience4 \
      --devices 0 \
      --workspace_path /home/dingxukai \
      --load_ckpt_path "${STAGE0_CKPT}" \
      --disable_early_stopping \
      --do_test \
      --test_times 1 \
      --seed 0 \
      dataset_dir="${TRAIN_DATA}" \
      tiny_dataset=false \
      batch_size=1 \
      val_batch_size=1 \
      num_workers=4 \
      persistent_workers=false \
      trainer.max_epochs="${MAX_EPOCHS}" \
      trainer.max_steps=-1 \
      trainer.val_check_interval=1.0 \
      trainer.check_val_every_n_epoch="${VAL_EVERY_N_EPOCH}" \
      trainer.num_sanity_val_steps=0 \
      trainer.default_root_dir="${RUN_ROOT}/${variant}/train" \
      trainer.logger.save_dir="${logger}" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.training_kwargs.scheduler.num_training_steps=67260 \
      model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=false \
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
      "${overrides[@]}" \
      2>&1 | tee "${base}/train.log"
  actual_logger="$(
    for candidate in "${ROOT}"/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20*; do
      [[ -f "${candidate}/hparams.yaml" ]] || continue
      if rg -q -F "${RUN_ROOT}/${variant}/train" "${candidate}/hparams.yaml"; then
        printf '%s\n' "${candidate}"
      fi
    done | sort | tail -n 1
  )"
  if [[ -n "${actual_logger}" ]]; then
    mkdir -p "${logger}/tb"
    if [[ ! -e "${logger}/tb/run" ]]; then
      ln -s "${actual_logger}" "${logger}/tb/run"
    fi
  fi
  local best
  best="$(find "${logger}" ${actual_logger:+"${actual_logger}"} -type f -name 'epoch*.ckpt' ! -name 'last.ckpt' -print 2>/dev/null | sort -V | tail -n 1)"
  if [[ -z "${best}" ]]; then
    printf '%s [error] no best checkpoint variant=%s\n' "$(timestamp)" "${variant}" | tee -a "${OUT}/pipeline_status.log"
    return 1
  fi
  printf '%s\n' "${best}" > "${best_file}"
  printf '%s [trained] variant=%s checkpoint=%s\n' "$(timestamp)" "${variant}" "$(basename "${best}")" | tee -a "${OUT}/pipeline_status.log"
}

result_json() {
  local base="$1"
  find "${base}" -type f -name 'test_*_gsm_pid*.json' -print 2>/dev/null | sort | tail -n 1
}

eval_variant() {
  local variant="$1"
  local ckpt
  ckpt="$(<"${OUT}/${variant}/best_checkpoint.txt")"
  for dataset in gsm8k gsmhard svamp multiarith; do
    local base="${OUT}/${variant}/eval/${dataset}"
    local visual_args=(
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=false
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0
      model.model_kwargs.trace_bridge_config.trace_visual_group_views=1
      model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true
    )
    if [[ "${dataset}" == "gsm8k" ]]; then
      visual_args=(
        model.model_kwargs.trace_bridge_config.save_trace_visual_info=true
        model.model_kwargs.trace_bridge_config.trace_visual_record_limit=200
        model.model_kwargs.trace_bridge_config.trace_visual_group_views=8
        model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=false
      )
    fi
    mkdir -p "${base}/logs"
    if [[ -n "$(result_json "${base}")" ]]; then
      printf '%s [skip] evaluated variant=%s dataset=%s\n' "$(timestamp)" "${variant}" "${dataset}" | tee -a "${OUT}/pipeline_status.log"
      continue
    fi
    wait_for_gpu
    printf '%s [eval] variant=%s dataset=%s test_times=1\n' "$(timestamp)" "${variant}" "${dataset}" | tee -a "${OUT}/pipeline_status.log"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" run.py \
        --model trace_bridge_qwen3_instruct_vizstrong \
        --dataset qsa \
        --trainer default \
        --devices 0 \
        --workspace_path /home/dingxukai \
        --test_ckpt_path "${ckpt}" \
        --test_times 1 \
        --seed 0 \
        dataset_dir="${EVAL_DATASETS[$dataset]}" \
        tiny_dataset=false \
        batch_size=1 \
        val_batch_size=1 \
        num_workers=2 \
        persistent_workers=false \
        trainer.num_sanity_val_steps=0 \
        trainer.strategy=auto \
        trainer.default_root_dir="${RUN_ROOT}/${variant}/eval/${dataset}" \
        trainer.logger.save_dir="${base}/logs" \
        trainer.logger.name=tb \
        trainer.logger.version=run \
        model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
        "${visual_args[@]}" \
        2>&1 | tee "${base}/eval.log"
  done
  local record="${OUT}/${variant}/eval/gsm8k/logs/tb/run/trace_bridge_visual_test.pt"
  if [[ -f "${record}" ]] && [[ ! -f "${OUT}/${variant}/geometry_200/.done" ]]; then
    "${PY}" tools/trace_bridge_geometry_summary.py \
      --records "${record}" \
      --out_dir "${OUT}/${variant}/geometry_200" \
      --max_records 200 \
      > "${OUT}/${variant}/geometry_200.log" 2>&1
    touch "${OUT}/${variant}/geometry_200/.done"
  fi
}

{
  printf 'experiment=TRACE_Stage1_component_ablations\n'
  printf 'variants=no_path_consistency,no_progress_anchor,no_multiview\n'
  printf 'full_training_examples_per_epoch=6726\n'
  printf 'max_epochs=%s\n' "${MAX_EPOCHS}"
  printf 'budget_revision=2026-07-18_user_requested_short_ablation\n'
  printf 'validation_schedule=final_epoch_only\n'
  printf 'validation_every_n_epoch=%s\n' "${VAL_EVERY_N_EPOCH}"
  printf 'early_stopping=false\n'
  printf 'test_times=1\n'
  printf 'started_at=%s\n' "$(timestamp)"
} > "${OUT}/manifest.txt"

wait_for_pareto
for variant in no_path_consistency no_progress_anchor no_multiview; do
  train_variant "${variant}"
  eval_variant "${variant}"
done
touch "${OUT}/pipeline_done.txt"
printf '%s [done] Stage 1 component ablations complete\n' "$(timestamp)" | tee -a "${OUT}/pipeline_status.log"
