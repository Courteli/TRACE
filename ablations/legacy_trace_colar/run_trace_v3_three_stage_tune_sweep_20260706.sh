#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

CKPT="${CKPT:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260706-032913_586957_trace_v3_three_stage_full_20260705_stage1solid_stage2_trace_multipath_rl/checkpoints/epoch4__step2560__monitor0.347.ckpt}"
GPU="${GPU:-2}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706}"
VAL_N="${VAL_N:-256}"
TEST_TIMES="${TEST_TIMES:-1}"
mkdir -p "${ARTIFACT_DIR}/datasets" "${ARTIFACT_DIR}/eval_json" "${ARTIFACT_DIR}/logs"

VAL_SUBSET="${ARTIFACT_DIR}/datasets/gsm8k_val_stride_${VAL_N}.jsonl"
if [[ ! -s "${VAL_SUBSET}" ]]; then
  awk -v n="${VAL_N}" 'NR % 3 == 1 { print; c += 1 } c >= n { exit }' \
    /home/dingxukai/RoT/data/GSM8k-Aug-NL/gsm8k_val_processed.jsonl > "${VAL_SUBSET}"
fi

SUMMARY_CSV="${ARTIFACT_DIR}/sweep_summary.csv"
echo "tag,json_path,dataset,test_file,n_items,n_predictions,acc,avg_L,avg_output_len,max_L,min_L,latent_temperature,eol_temperature,compression_factor,max_new_tokens,ckpt" > "${SUMMARY_CSV}"

run_eval() {
  local tag="$1"
  local dataset_dir="$2"
  local test_file="$3"
  local dataset_name="$4"
  local max_l="$5"
  local min_l="$6"
  local latent_temp="$7"
  local eol_temp="$8"
  local compression_factor="$9"
  local max_new_tokens="${10}"
  local log_dir="${ARTIFACT_DIR}/logs/${tag}"
  mkdir -p "${log_dir}"
  echo "[sweep] start tag=${tag} max_l=${max_l} min_l=${min_l} latent_temp=${latent_temp} eol_temp=${eol_temp} cf=${compression_factor} max_new=${max_new_tokens}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
    --model=trace_multipath_qwen3_instruct \
    --dataset=gsm8k_aug_nl \
    --devices=0 \
    --test_ckpt_path="${CKPT}" \
    --test_times="${TEST_TIMES}" \
    --seed=0 \
    --workspace_path=/home/dingxukai \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=0 \
    persistent_workers=False \
    num_sanity_val_steps=0 \
    trainer.logger.save_dir="${log_dir}" \
    trainer.logger.name="tb" \
    trainer.logger.version="run" \
    trainer.default_root_dir="${ARTIFACT_DIR}/roots/${tag}" \
    dataset_name="${dataset_name}" \
    dataset_dir="${dataset_dir}" \
    test_file="${test_file}" \
    max_n_latent_forward="${max_l}" \
    min_n_latent_forward="${min_l}" \
    latent_temperature="${latent_temp}" \
    eol_temperature="${eol_temp}" \
    compression_factor="${compression_factor}" \
    max_new_tokens="${max_new_tokens}" \
    2>&1 | tee "${ARTIFACT_DIR}/logs/${tag}.log"

  local json_path
  json_path="$(find "${log_dir}/tb/run" -maxdepth 1 -type f -name 'test_*.json' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "${json_path}" ]]; then
    echo "[sweep] ERROR: no test json for ${tag}" >&2
    exit 3
  fi
  local copied="${ARTIFACT_DIR}/eval_json/${tag}.json"
  cp -f "${json_path}" "${copied}"
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_eval_json_summary.py --csv "${copied}" \
    | tail -n +2 \
    | sed "s|^|${tag},|" >> "${SUMMARY_CSV}"
  tail -1 "${SUMMARY_CSV}"
}

SUBSET_DIR="${ARTIFACT_DIR}/datasets"
SUBSET_FILE="$(basename "${VAL_SUBSET}")"

run_eval "val256_base_m0_t1_e1_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 0 1.0 1.0 5 16
run_eval "val256_m8_t1_e1_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 8 1.0 1.0 5 16
run_eval "val256_m12_t1_e1_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 12 1.0 1.0 5 16
run_eval "val256_m16_t1_e1_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 16 1.0 1.0 5 16
run_eval "val256_m24_t1_e1_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 24 1.0 1.0 5 16
run_eval "val256_m12_t07_e1_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 12 0.7 1.0 5 16
run_eval "val256_m16_t07_e1_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 16 0.7 1.0 5 16
run_eval "val256_m12_t05_e1_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 12 0.5 1.0 5 16
run_eval "val256_m12_t1_e07_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 12 1.0 0.7 5 16
run_eval "val256_m12_t1_e13_cf5" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 12 1.0 1.3 5 16
run_eval "val256_m12_t1_e1_cf4" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 12 1.0 1.0 4 16
run_eval "val256_m16_t1_e1_cf4" "${SUBSET_DIR}" "${SUBSET_FILE}" "gsm8k_val_sweep256" 40 16 1.0 1.0 4 16

echo "[sweep] finished subset sweep. Summary: ${SUMMARY_CSV}"
