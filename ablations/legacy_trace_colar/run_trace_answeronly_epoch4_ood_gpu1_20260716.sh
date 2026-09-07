#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
PY=/home/dingxukai/miniconda3/envs/ROT/bin/python
GPU="${GPU:-1}"
OUT="${OUT:-${ROOT}/run_outputs/trace_bridge/20260716_trace_bridge_answeronly_epoch4_final_evidence_gpu7}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/run_roots/trace_bridge/20260716_trace_bridge_answeronly_epoch4_final_evidence_gpu7}"
CKPT="${CKPT:-${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260714-003100_749268_20260713_trace_bridge_answeronly_fullbudget_control_stage2_answer_only/checkpoints/epoch4__step2560__monitor0.725936.ckpt}"
MODEL=trace_bridge_qwen3_instruct_vizstrong

declare -A DATASETS=(
  [gsmhard]=/home/dingxukai/RoT/data/GSM8k-Hard/readcot_qsa_test
  [svamp]=/home/dingxukai/RoT/data/SVAMP/readcot_qsa_test
  [multiarith]=/home/dingxukai/RoT/data/Multiarith/readcot_qsa_test
)

mkdir -p "${OUT}/benchmarks" "${RUN_ROOT}"
cd "${ROOT}"
exec 8>"${OUT}/ood_gpu1.lock"
if ! flock -n 8; then
  exit 0
fi

result_json() {
  local log_dir="$1"
  find "${log_dir}/tb/run" -maxdepth 1 -type f -name 'test_*_gsm_pid*.json' -print 2>/dev/null | sort | tail -n 1
}

for key in gsmhard svamp multiarith; do
  base="${OUT}/benchmarks/${key}"
  log_dir="${base}/logs"
  log_file="${base}/eval_gpu${GPU}.log"
  mkdir -p "${log_dir}"
  if [[ -n "$(result_json "${log_dir}")" ]]; then
    continue
  fi
  printf '%s [run] auxiliary OOD key=%s physical_gpu=%s\n' "$(date '+%F %T')" "${key}" "${GPU}" | tee -a "${OUT}/ood_gpu1_status.log"
  TMPDIR=/tmp \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PY}" run.py \
      --model "${MODEL}" \
      --dataset qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /home/dingxukai \
      --test_ckpt_path "${CKPT}" \
      --test_times 1 \
      dataset_dir="${DATASETS[$key]}" \
      batch_size=1 \
      val_batch_size=1 \
      num_workers=2 \
      persistent_workers=false \
      do_trace_rl=true \
      trainer.num_sanity_val_steps=0 \
      trainer.strategy=auto \
      trainer.default_root_dir="${RUN_ROOT}/${key}_none_gpu${GPU}" \
      trainer.logger.save_dir="${log_dir}" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
      model.model_kwargs.trace_bridge_config.trace_eval_intervention=none \
      model.model_kwargs.trace_bridge_config.trace_eval_intervention_seed=0 \
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=false \
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
      model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true \
      2>&1 | tee -a "${log_file}"
  if [[ -z "$(result_json "${log_dir}")" ]]; then
    printf '%s [error] no result key=%s\n' "$(date '+%F %T')" "${key}" | tee -a "${OUT}/ood_gpu1_status.log"
    exit 1
  fi
  printf 'gpu%s\n' "${GPU}" > "${base}/completed_attempt.txt"
done

printf '%s\n' "$(date '+%F %T')" > "${OUT}/ood_gpu1_done.txt"
