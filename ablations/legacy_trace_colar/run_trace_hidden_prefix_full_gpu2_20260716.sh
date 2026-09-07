#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
PY=/home/dingxukai/miniconda3/envs/ROT/bin/python
GPU=2
DATASET=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc
OUT=${ROOT}/run_outputs/trace/20260716_answeronly_hidden_prefix_gpu2
RUN_ROOT=${ROOT}/run_roots/trace/20260716_answeronly_hidden_prefix_gpu2
ANALYZER=${ROOT}/tools/trace_hidden_prefix_analysis.py
POLL_SECONDS=${POLL_SECONDS:-30}

FINAL_CKPT=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260714-003100_749268_20260713_trace_bridge_answeronly_fullbudget_control_stage2_answer_only/checkpoints/epoch4__step2560__monitor0.725936.ckpt
STAGE1_CKPT=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_284260_20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1_stage1/checkpoints/epoch1__step13452__monitor0.661.ckpt
declare -A CKPTS=(
  [final]="${FINAL_CKPT}"
  [stage1]="${STAGE1_CKPT}"
)

mkdir -p "${OUT}/full" "${OUT}/analysis" "${RUN_ROOT}"
cd "${ROOT}"

exec 9>"${OUT}/full_pipeline.lock"
if ! flock -n 9; then
  printf '[skip] another hidden-prefix pipeline holds %s\n' "${OUT}/full_pipeline.lock"
  exit 0
fi

timestamp() {
  date '+%F %T'
}

result_json() {
  local base="$1"
  find "${base}" -type f -name 'test_*_gsm_pid*.json' -print 2>/dev/null | sort | tail -n 1
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

run_attempt() {
  local model="$1"
  local k="$2"
  local batch_size="$3"
  local base="$4"
  local ckpt="${CKPTS[$model]}"
  local attempt="${base}/attempt_b${batch_size}"
  mkdir -p "${attempt}/logs"
  printf '%s [run] model=%s k=%s batch=%s\n' "$(timestamp)" "${model}" "${k}" "${batch_size}" | tee -a "${OUT}/pipeline_status.log"
  set +e
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
      dataset_dir="${DATASET}" \
      tiny_dataset=false \
      batch_size="${batch_size}" \
      val_batch_size="${batch_size}" \
      num_workers=2 \
      persistent_workers=false \
      trainer.num_sanity_val_steps=0 \
      trainer.strategy=auto \
      trainer.default_root_dir="${RUN_ROOT}/full/${model}/k${k}/b${batch_size}" \
      trainer.logger.save_dir="${attempt}/logs" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.target=src.models.trace_bridge.LitTRACEBridge \
      model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
      model.model_kwargs.trace_bridge_config.trace_eval_hidden_prefix_k="${k}" \
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=false \
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
      model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true \
      2>&1 | tee "${attempt}/eval.log"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

run_point() {
  local model="$1"
  local k="$2"
  local base="${OUT}/full/${model}/k${k}"
  local result
  mkdir -p "${base}"
  result="$(result_json "${base}")"
  if [[ -n "${result}" ]]; then
    printf '%s\n' "${result}" > "${base}/completed_result.txt"
    printf '%s [skip] completed model=%s k=%s\n' "$(timestamp)" "${model}" "${k}" | tee -a "${OUT}/pipeline_status.log"
    return
  fi

  wait_for_gpu
  for batch_size in 8 4 2 1; do
    if run_attempt "${model}" "${k}" "${batch_size}" "${base}"; then
      result="$(result_json "${base}")"
      if [[ -n "${result}" ]]; then
        printf '%s\n' "${result}" > "${base}/completed_result.txt"
        printf '%s [done] model=%s k=%s batch=%s result=%s\n' \
          "$(timestamp)" "${model}" "${k}" "${batch_size}" "${result}" | tee -a "${OUT}/pipeline_status.log"
        return
      fi
    fi
    if ! grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "${base}/attempt_b${batch_size}/eval.log"; then
      printf '%s [error] non-OOM failure model=%s k=%s batch=%s\n' \
        "$(timestamp)" "${model}" "${k}" "${batch_size}" | tee -a "${OUT}/pipeline_status.log"
      return 1
    fi
    printf '%s [retry] OOM model=%s k=%s old_batch=%s\n' \
      "$(timestamp)" "${model}" "${k}" "${batch_size}" | tee -a "${OUT}/pipeline_status.log"
    sleep 5
  done
  printf '%s [error] all batch sizes failed model=%s k=%s\n' "$(timestamp)" "${model}" "${k}" | tee -a "${OUT}/pipeline_status.log"
  return 1
}

if [[ ! -f "${OUT}/smoke/prefix8_parity_b8.json" ]] || \
   ! grep -q '"parity": true' "${OUT}/smoke/prefix8_parity_b8.json" || \
   ! find "${OUT}/smoke/prefix0_b8" -type f -name 'test_*_gsm_pid*.json' -print -quit | grep -q .; then
  printf '%s [error] smoke parity or k=0 preflight is incomplete\n' "$(timestamp)" | tee -a "${OUT}/pipeline_status.log"
  exit 1
fi

{
  printf 'experiment=hidden_prefix_causal_access\n'
  printf 'gpu=%s\n' "${GPU}"
  printf 'dataset=%s\n' "${DATASET}"
  printf 'test_times=1\n'
  printf 'n_latents=8\n'
  printf 'position_control=fixed_original_absolute_positions\n'
  printf 'final_sha256=%s\n' "$(sha256sum "${FINAL_CKPT}" | awk '{print $1}')"
  printf 'stage1_sha256=%s\n' "$(sha256sum "${STAGE1_CKPT}" | awk '{print $1}')"
  printf 'started_at=%s\n' "$(timestamp)"
} > "${OUT}/manifest.txt"

# Endpoints are first so the most decisive paired effect is available early;
# intermediate k values then establish whether information accrues progressively.
for model in final stage1; do
  for k in 0 8 2 4 6 1 3 5 7; do
    run_point "${model}" "${k}"
    "${PY}" "${ANALYZER}" --root "${OUT}" >> "${OUT}/analysis_updates.log" 2>&1
  done
done

"${PY}" "${ANALYZER}" --root "${OUT}" | tee "${OUT}/analysis_final.log"
touch "${OUT}/pipeline_done.txt"
printf '%s [done] full hidden-prefix experiment complete\n' "$(timestamp)" | tee -a "${OUT}/pipeline_status.log"
