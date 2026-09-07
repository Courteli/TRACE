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
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706/followup_${TAG}}"
DATASETS="${DATASETS:-gsm8k_aug_nl gsmhard svamp multiarith}"
MAX_L="${MAX_L:-40}"
MIN_L="${MIN_L:-12}"
LATENT_TEMP="${LATENT_TEMP:-0.7}"
EOL_TEMP="${EOL_TEMP:-1.0}"
COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEST_TIMES="${TEST_TIMES:-1}"
RUN_OOD="${RUN_OOD:-1}"
RUN_VIS="${RUN_VIS:-1}"
RUN_GEOMETRY="${RUN_GEOMETRY:-1}"
LP_DETERMINISTIC="${LP_DETERMINISTIC:-0}"
VIS_CANDIDATE_COUNT="${VIS_CANDIDATE_COUNT:-200}"
VIS_NUM_QUESTIONS="${VIS_NUM_QUESTIONS:-6}"
GEOMETRY_NUM_QUESTIONS="${GEOMETRY_NUM_QUESTIONS:-200}"
GROUP_SIZE="${GROUP_SIZE:-8}"
SEED="${SEED:-0}"
RUN_EXTRA_ARGS=()
TOOL_EXTRA_ARGS=()
if [[ "${LP_DETERMINISTIC}" == "1" || "${LP_DETERMINISTIC}" == "true" || "${LP_DETERMINISTIC}" == "True" ]]; then
  RUN_EXTRA_ARGS+=(lp_determinisitc=true)
  TOOL_EXTRA_ARGS+=(--lp_deterministic)
fi

mkdir -p "${ARTIFACT_DIR}"
SUMMARY="${ARTIFACT_DIR}/candidate_main_ood_summary.csv"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "tag,json_path,dataset,test_file,n_items,n_predictions,acc,avg_L,avg_output_len,max_L,min_L,latent_temperature,eol_temperature,compression_factor,max_new_tokens,ckpt" > "${SUMMARY}"
fi

dataset_spec() {
  local dataset="$1"
  case "${dataset}" in
    gsm8k_aug_nl)
      echo "gsm8k_aug_nl /home/dingxukai/RoT/data/GSM8k-Aug-NL gsm8k_test_processed.jsonl"
      ;;
    gsmhard)
      echo "gsmhard /home/dingxukai/RoT/data/GSM8k-Hard gsmhard_test_processed.jsonl"
      ;;
    svamp)
      echo "svamp /home/dingxukai/RoT/data/SVAMP svamp_test_processed.jsonl"
      ;;
    multiarith)
      echo "multiarith /home/dingxukai/RoT/data/Multiarith multiarith_test_processed.jsonl"
      ;;
    *)
      echo "unknown dataset: ${dataset}" >&2
      return 2
      ;;
  esac
}

eval_dataset() {
  local dataset="$1"
  local dataset_name dataset_dir data_file
  read -r dataset_name dataset_dir data_file < <(dataset_spec "${dataset}")
  local ds_tag="${TAG}_${dataset_name}"
  local log_dir="${ARTIFACT_DIR}/logs/${ds_tag}"
  mkdir -p "${log_dir}"

  if awk -F, -v needle="${ds_tag}" '$1 == needle { found = 1 } END { exit(found ? 0 : 1) }' "${SUMMARY}"; then
    echo "[followup] skip existing ${ds_tag}"
    return 0
  fi

  echo "[followup] eval ${dataset_name} tag=${ds_tag} max_l=${MAX_L} min_l=${MIN_L} latent_temp=${LATENT_TEMP} eol_temp=${EOL_TEMP} cf=${COMPRESSION_FACTOR}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
    --model=trace_multipath_qwen3_instruct \
    --dataset="${dataset_name}" \
    --devices=0 \
    --test_ckpt_path="${CKPT}" \
    --test_times="${TEST_TIMES}" \
    --seed="${SEED}" \
    --workspace_path=/home/dingxukai \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=0 \
    persistent_workers=False \
    num_sanity_val_steps=0 \
    trainer.logger.save_dir="${log_dir}" \
    trainer.logger.name="tb" \
    trainer.logger.version="run" \
    trainer.default_root_dir="${ARTIFACT_DIR}/roots/${ds_tag}" \
    dataset_name="${dataset_name}" \
    dataset_dir="${dataset_dir}" \
    test_file="${data_file}" \
    max_n_latent_forward="${MAX_L}" \
    min_n_latent_forward="${MIN_L}" \
    latent_temperature="${LATENT_TEMP}" \
    eol_temperature="${EOL_TEMP}" \
    compression_factor="${COMPRESSION_FACTOR}" \
    max_new_tokens="${MAX_NEW_TOKENS}" \
    "${RUN_EXTRA_ARGS[@]}" \
    2>&1 | tee "${ARTIFACT_DIR}/${ds_tag}.log"

  local json_path copied row
  json_path="$(find "${log_dir}/tb/run" -maxdepth 1 -type f -name 'test_*.json' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "${json_path}" ]]; then
    echo "[followup] ERROR: no test json for ${ds_tag}" >&2
    exit 3
  fi
  copied="${ARTIFACT_DIR}/${ds_tag}.json"
  cp -f "${json_path}" "${copied}"
  row="$(/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_eval_json_summary.py --csv "${copied}" | tail -n +2)"
  echo "${ds_tag},${row}" >> "${SUMMARY}"
  tail -1 "${SUMMARY}"
}

echo "[followup] start $(date '+%F %T')"
echo "[followup] ckpt=${CKPT}"
echo "[followup] artifact_dir=${ARTIFACT_DIR}"

if [[ "${RUN_OOD}" == "1" ]]; then
  for dataset in ${DATASETS}; do
    eval_dataset "${dataset}"
  done
fi

if [[ "${RUN_VIS}" == "1" ]]; then
  VIS_DIR="${ARTIFACT_DIR}/visual_auto${VIS_NUM_QUESTIONS}_global"
  echo "[followup] visualization -> ${VIS_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/trace_multipath_visualize.py \
    --ckpt "${CKPT}" \
    --auto_select \
    --candidate_count "${VIS_CANDIDATE_COUNT}" \
    --num_questions "${VIS_NUM_QUESTIONS}" \
    --seed "${SEED}" \
    --group_size "${GROUP_SIZE}" \
    --max_l "${MAX_L}" \
    --min_l "${MIN_L}" \
    --latent_temperature "${LATENT_TEMP}" \
    --eol_temperature "${EOL_TEMP}" \
    --compression_factor "${COMPRESSION_FACTOR}" \
    "${TOOL_EXTRA_ARGS[@]}" \
    --device cuda:0 \
    --trajectory_space residual \
    --pca_scope global \
    --out_dir "${VIS_DIR}" \
    2>&1 | tee "${ARTIFACT_DIR}/visual_auto${VIS_NUM_QUESTIONS}_global.log"
fi

if [[ "${RUN_GEOMETRY}" == "1" ]]; then
  GEOM_DIR="${ARTIFACT_DIR}/geometry_${GEOMETRY_NUM_QUESTIONS}"
  echo "[followup] geometry -> ${GEOM_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/trace_multipath_geometry_summary.py \
    --ckpt "${CKPT}" \
    --out_dir "${GEOM_DIR}" \
    --num_questions "${GEOMETRY_NUM_QUESTIONS}" \
    --seed "${SEED}" \
    --group_size "${GROUP_SIZE}" \
    --max_l "${MAX_L}" \
    --min_l "${MIN_L}" \
    --latent_temperature "${LATENT_TEMP}" \
    --eol_temperature "${EOL_TEMP}" \
    --compression_factor "${COMPRESSION_FACTOR}" \
    "${TOOL_EXTRA_ARGS[@]}" \
    --device cuda:0 \
    2>&1 | tee "${ARTIFACT_DIR}/geometry_${GEOMETRY_NUM_QUESTIONS}.log"
fi

echo "[followup] done $(date '+%F %T')"
