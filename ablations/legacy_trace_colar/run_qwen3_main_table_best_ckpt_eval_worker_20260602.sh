#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?Usage: $0 METHOD GPU CKPT}"
GPU="${2:?Usage: $0 METHOD GPU CKPT}"
CKPT="${3:?Usage: $0 METHOD GPU CKPT}"

ROOT="/home/dingxukai/colar origin"
OUT_DIR="${ROOT}/run_outputs/qwen3_main_table_best_ckpt_eval_20260602"
MODEL="${METHOD}_qwen3_instruct"

mkdir -p "${OUT_DIR}" "${ROOT}/run_roots"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

run_eval() {
  local dataset="$1"
  local data_dir="$2"
  local test_file="$3"
  local log_file="${OUT_DIR}/${METHOD}_${dataset}.log"

  echo "[$(date '+%F %T')] START method=${METHOD} dataset=${dataset} gpu=${GPU} ckpt=${CKPT}" | tee -a "${log_file}"
  cd "${ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model="${MODEL}" \
    --dataset=gsm8k_aug_nl \
    --devices=0 \
    --workspace_path=/home/dingxukai \
    --test_ckpt_path="${CKPT}" \
    --test_times=5 \
    --log_suffix="origin_${METHOD}_qwen3_instruct_${dataset}_best_ckpt_eval_gpu${GPU}" \
    dataset_name="${dataset}" \
    dataset_dir="${data_dir}" \
    test_file="${test_file}" \
    max_new_tokens=16 \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    persistent_workers=false \
    trainer.default_root_dir="${ROOT}/run_roots/origin_${METHOD}_qwen3_instruct_${dataset}_best_ckpt_eval_gpu${GPU}" \
    2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE method=${METHOD} dataset=${dataset}" | tee -a "${log_file}"
}

run_eval gsm8k_aug_nl "/home/dingxukai/RoT/data/GSM8k-Aug-NL" "gsm8k_test_processed.jsonl"
run_eval gsmhard "/home/dingxukai/RoT/data/GSM8k-Hard" "gsmhard_test_processed.jsonl"
run_eval svamp "/home/dingxukai/RoT/data/SVAMP" "svamp_test_processed.jsonl"
run_eval multiarith "/home/dingxukai/RoT/data/Multiarith" "multiarith_test_processed.jsonl"

echo "[$(date '+%F %T')] ALL DONE method=${METHOD}" | tee -a "${OUT_DIR}/${METHOD}_done.log"
