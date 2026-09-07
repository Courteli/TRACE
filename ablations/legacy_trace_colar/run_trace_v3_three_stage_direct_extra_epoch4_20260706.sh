#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

CKPT="${CKPT:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260706-032913_586957_trace_v3_three_stage_full_20260705_stage1solid_stage2_trace_multipath_rl/checkpoints/epoch4__step2560__monitor0.347.ckpt}"
GPU="${GPU:-6}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706/direct_extra_epoch4}"
TARGET_ACC="${TARGET_ACC:-0.30}"
TEST_TIMES="${TEST_TIMES:-1}"
mkdir -p "${ARTIFACT_DIR}"

SUMMARY="${ARTIFACT_DIR}/direct_extra_summary.csv"
echo "tag,json_path,dataset,test_file,n_items,n_predictions,acc,avg_L,avg_output_len,max_L,min_L,latent_temperature,eol_temperature,compression_factor,max_new_tokens,ckpt" > "${SUMMARY}"

run_full_eval() {
  local tag="$1"
  local max_l="$2"
  local min_l="$3"
  local latent_temp="$4"
  local eol_temp="$5"
  local compression_factor="$6"
  local max_new_tokens="$7"
  local log_dir="${ARTIFACT_DIR}/logs/${tag}"
  mkdir -p "${log_dir}"
  echo "[direct-extra] start tag=${tag} max_l=${max_l} min_l=${min_l} latent_temp=${latent_temp} eol_temp=${eol_temp} cf=${compression_factor}"
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
    dataset_name="gsm8k_aug_nl" \
    dataset_dir="/home/dingxukai/RoT/data/GSM8k-Aug-NL" \
    test_file="gsm8k_test_processed.jsonl" \
    max_n_latent_forward="${max_l}" \
    min_n_latent_forward="${min_l}" \
    latent_temperature="${latent_temp}" \
    eol_temperature="${eol_temp}" \
    compression_factor="${compression_factor}" \
    max_new_tokens="${max_new_tokens}" \
    2>&1 | tee "${ARTIFACT_DIR}/${tag}.log"

  local json_path
  json_path="$(find "${log_dir}/tb/run" -maxdepth 1 -type f -name 'test_*.json' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "${json_path}" ]]; then
    echo "[direct-extra] ERROR: no test json for ${tag}" >&2
    exit 3
  fi
  local copied="${ARTIFACT_DIR}/${tag}.json"
  cp -f "${json_path}" "${copied}"
  local row
  row="$(/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_eval_json_summary.py --csv "${copied}" | tail -n +2)"
  echo "${tag},${row}" >> "${SUMMARY}"
  tail -1 "${SUMMARY}"
  local full_acc
  full_acc="$(echo "${row}" | /home/dingxukai/miniconda3/envs/ROT/bin/python -c 'import csv,sys; print(next(csv.DictReader(sys.stdin, fieldnames=["path","dataset","test_file","n_items","n_predictions","acc","avg_L","avg_output_len","max_L","min_L","latent_temperature","eol_temperature","compression_factor","max_new_tokens","ckpt"]))["acc"])')"
  /home/dingxukai/miniconda3/envs/ROT/bin/python - <<PY
import sys
sys.exit(0 if float("${full_acc}") >= float("${TARGET_ACC}") else 1)
PY
}

if run_full_eval "epoch4_full_m16_t07_e1_cf5" 40 16 0.7 1.0 5 16; then
  echo "[direct-extra] target reached at epoch4_full_m16_t07_e1_cf5"
  exit 0
fi
if run_full_eval "epoch4_full_m20_t07_e1_cf5" 40 20 0.7 1.0 5 16; then
  echo "[direct-extra] target reached at epoch4_full_m20_t07_e1_cf5"
  exit 0
fi
if run_full_eval "epoch4_full_m12_t05_e1_cf5" 40 12 0.5 1.0 5 16; then
  echo "[direct-extra] target reached at epoch4_full_m12_t05_e1_cf5"
  exit 0
fi

echo "[direct-extra] target not reached by extra candidates. Summary: ${SUMMARY}"
