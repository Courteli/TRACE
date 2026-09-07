#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
PY=/home/dingxukai/miniconda3/envs/ROT/bin/python
GPU=${GPU:-0}
DATASET=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc
RAW_OUT=${ROOT}/run_outputs/trace/20260716_accuracy_length_pareto_gpu0
PAPER_OUT=${ROOT}/run_outputs/trace/20260716_paper_evidence_suite/accuracy_length_pareto
RUN_ROOT=${ROOT}/run_roots/trace/20260716_accuracy_length_pareto_gpu0
ANALYZER=${ROOT}/tools/trace_accuracy_length_pareto.py
POLL_SECONDS=${POLL_SECONDS:-30}

declare -A CKPTS=(
  [stage1]="${ROOT}/checkpoints/trace/stage1_run/checkpoints/model.ckpt"
  [final]="${ROOT}/checkpoints/trace/final_epoch4_run/checkpoints/model.ckpt"
)

mkdir -p "${RAW_OUT}" "${PAPER_OUT}" "${RUN_ROOT}"
cd "${ROOT}"

exec 9>"${RAW_OUT}/pipeline.lock"
if ! flock -n 9; then
  printf '[skip] another accuracy-length pipeline is already active\n'
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
    printf '%s [wait] gpu=%s used=%sMiB\n' "$(timestamp)" "${GPU}" "${used}" | tee -a "${RAW_OUT}/pipeline_status.log"
    sleep "${POLL_SECONDS}"
  done
}

run_attempt() {
  local stage="$1"
  local budget="$2"
  local batch_size="$3"
  local base="$4"
  local attempt="${base}/attempt_b${batch_size}"
  mkdir -p "${attempt}/logs"
  printf '%s [run] stage=%s budget=%s batch=%s\n' \
    "$(timestamp)" "${stage}" "${budget}" "${batch_size}" | tee -a "${RAW_OUT}/pipeline_status.log"
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
      --test_ckpt_path "${CKPTS[$stage]}" \
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
      trainer.default_root_dir="${RUN_ROOT}/${stage}/budget${budget}/b${batch_size}" \
      trainer.logger.save_dir="${attempt}/logs" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.target=src.models.trace_bridge.LitTRACEBridge \
      model.model_kwargs.hybrid_generation_config.max_new_tokens="${budget}" \
      model.model_kwargs.trace_bridge_config.trace_eval_hidden_prefix_k=-1 \
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=false \
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
      model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true \
      2>&1 | tee "${attempt}/eval.log"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

run_point() {
  local stage="$1"
  local budget="$2"
  local base="${RAW_OUT}/raw/${stage}/budget${budget}"
  local result
  mkdir -p "${base}"
  result="$(result_json "${base}")"
  if [[ -n "${result}" ]]; then
    printf '%s [skip] complete stage=%s budget=%s\n' \
      "$(timestamp)" "${stage}" "${budget}" | tee -a "${RAW_OUT}/pipeline_status.log"
    return
  fi

  wait_for_gpu
  for batch_size in 8 4 2 1; do
    if run_attempt "${stage}" "${budget}" "${batch_size}" "${base}"; then
      result="$(result_json "${base}")"
      if [[ -n "${result}" ]]; then
        printf '%s\n' "${result}" > "${base}/completed_result.txt"
        printf '%s [done] stage=%s budget=%s batch=%s\n' \
          "$(timestamp)" "${stage}" "${budget}" "${batch_size}" | tee -a "${RAW_OUT}/pipeline_status.log"
        return
      fi
    fi
    if ! grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "${base}/attempt_b${batch_size}/eval.log"; then
      printf '%s [error] non-OOM failure stage=%s budget=%s batch=%s\n' \
        "$(timestamp)" "${stage}" "${budget}" "${batch_size}" | tee -a "${RAW_OUT}/pipeline_status.log"
      return 1
    fi
    printf '%s [retry] OOM stage=%s budget=%s old_batch=%s\n' \
      "$(timestamp)" "${stage}" "${budget}" "${batch_size}" | tee -a "${RAW_OUT}/pipeline_status.log"
    sleep 5
  done
  printf '%s [error] all batch sizes failed stage=%s budget=%s\n' \
    "$(timestamp)" "${stage}" "${budget}" | tee -a "${RAW_OUT}/pipeline_status.log"
  return 1
}

{
  printf 'experiment=matched_accuracy_length_pareto\n'
  printf 'dataset=GSM8K_test_full\n'
  printf 'test_times=1\n'
  printf 'latent_slots=8\n'
  printf 'budgets=24,32,40,48,64\n'
  printf 'stage1_checkpoint_sha256=%s\n' "$(sha256sum "${CKPTS[stage1]}" | awk '{print $1}')"
  printf 'final_checkpoint_sha256=%s\n' "$(sha256sum "${CKPTS[final]}" | awk '{print $1}')"
  printf 'started_at=%s\n' "$(timestamp)"
} > "${RAW_OUT}/manifest.txt"

# Start with the paper operating point, then cover shorter and longer budgets.
for budget in 48 32 64 24 40; do
  for stage in stage1 final; do
    run_point "${stage}" "${budget}"
  done
  "${PY}" "${ANALYZER}" \
    --raw-root "${RAW_OUT}/raw" \
    --output-dir "${PAPER_OUT}" \
    >> "${RAW_OUT}/analysis_updates.log" 2>&1
done

"${PY}" "${ANALYZER}" \
  --raw-root "${RAW_OUT}/raw" \
  --output-dir "${PAPER_OUT}" \
  | tee "${RAW_OUT}/analysis_final.log"
touch "${RAW_OUT}/pipeline_done.txt"
printf '%s [done] accuracy-length Pareto pipeline complete\n' "$(timestamp)" | tee -a "${RAW_OUT}/pipeline_status.log"
