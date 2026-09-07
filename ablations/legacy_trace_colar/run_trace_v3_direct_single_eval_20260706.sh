#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

CKPT="${CKPT:?CKPT is required}"
TAG="${TAG:?TAG is required}"
GPU="${GPU:-0}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706/direct_single}"
MAX_L="${MAX_L:-40}"
MIN_L="${MIN_L:-8}"
LATENT_TEMP="${LATENT_TEMP:-1.0}"
EOL_TEMP="${EOL_TEMP:-1.0}"
COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEST_TIMES="${TEST_TIMES:-1}"
EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS:-}"
EXTRA_ARGS=()
if [[ -n "${EXTRA_RUN_ARGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_RUN_ARGS}"
fi

mkdir -p "${ARTIFACT_DIR}"
SUMMARY="${ARTIFACT_DIR}/single_eval_summary.csv"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "tag,json_path,dataset,test_file,n_items,n_predictions,acc,avg_L,avg_output_len,max_L,min_L,latent_temperature,eol_temperature,compression_factor,max_new_tokens,ckpt" > "${SUMMARY}"
fi

LOG_DIR="${ARTIFACT_DIR}/logs/${TAG}"
mkdir -p "${LOG_DIR}"
echo "[direct-single] start tag=${TAG} ckpt=${CKPT} max_l=${MAX_L} min_l=${MIN_L} latent_temp=${LATENT_TEMP} eol_temp=${EOL_TEMP} cf=${COMPRESSION_FACTOR}"
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
  trainer.logger.save_dir="${LOG_DIR}" \
  trainer.logger.name="tb" \
  trainer.logger.version="run" \
  trainer.default_root_dir="${ARTIFACT_DIR}/roots/${TAG}" \
  dataset_name="gsm8k_aug_nl" \
  dataset_dir="/home/dingxukai/RoT/data/GSM8k-Aug-NL" \
  test_file="gsm8k_test_processed.jsonl" \
  max_n_latent_forward="${MAX_L}" \
  min_n_latent_forward="${MIN_L}" \
  latent_temperature="${LATENT_TEMP}" \
  eol_temperature="${EOL_TEMP}" \
  compression_factor="${COMPRESSION_FACTOR}" \
  max_new_tokens="${MAX_NEW_TOKENS}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${ARTIFACT_DIR}/${TAG}.log"

JSON_PATH="$(find "${LOG_DIR}/tb/run" -maxdepth 1 -type f -name 'test_*.json' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "${JSON_PATH}" ]]; then
  echo "[direct-single] ERROR: no test json for ${TAG}" >&2
  exit 3
fi
COPIED="${ARTIFACT_DIR}/${TAG}.json"
cp -f "${JSON_PATH}" "${COPIED}"
ROW="$(/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_eval_json_summary.py --csv "${COPIED}" | tail -n +2)"
echo "${TAG},${ROW}" >> "${SUMMARY}"
tail -1 "${SUMMARY}"
