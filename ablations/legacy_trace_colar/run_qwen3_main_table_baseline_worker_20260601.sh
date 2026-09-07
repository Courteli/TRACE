#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?Usage: $0 METHOD GPU}"
GPU="${2:?Usage: $0 METHOD GPU}"

ROOT="/home/dingxukai/colar origin"
OUT_DIR="${ROOT}/run_outputs/qwen3_main_table_baselines_20260601"
COT_CKPT="${ROOT}/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260426-134652_725405_origin_cot_qwen3_instruct_rawgsm_lr3e-5_3epoch_gpus1_2/checkpoints/epoch1__step6726__monitor0.857.ckpt"
MODEL="${METHOD}_qwen3_instruct"
SUFFIX="origin_${METHOD}_qwen3_instruct_cotinit_lr3e-5_50epoch_gpu${GPU}"
TRAIN_LOG="${OUT_DIR}/${METHOD}_train.log"

mkdir -p "${OUT_DIR}" "${ROOT}/run_roots"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

if [[ ! -f "${COT_CKPT}" ]]; then
  echo "Missing CoT-SFT initialization checkpoint: ${COT_CKPT}" >&2
  exit 1
fi

echo "[$(date '+%F %T')] START train method=${METHOD} gpu=${GPU}" | tee -a "${TRAIN_LOG}"
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
  --model="${MODEL}" \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  --load_ckpt_path="${COT_CKPT}" \
  --do_test \
  --test_times=5 \
  --workspace_path=/home/dingxukai \
  --log_suffix="${SUFFIX}" \
  batch_size=1 \
  val_batch_size=1 \
  max_epochs=50 \
  num_sanity_val_steps=0 \
  max_new_tokens=16 \
  lr=3e-5 \
  trainer.default_root_dir="${ROOT}/run_roots/${SUFFIX}" \
  2>&1 | tee -a "${TRAIN_LOG}"

LOG_DIR="$(find "${ROOT}/logs/${MODEL}/gsm8k_aug_nl-gsm8k_aug_nl" -mindepth 1 -maxdepth 1 -type d -name "*_${SUFFIX}" | sort | tail -n 1)"
if [[ -z "${LOG_DIR}" ]]; then
  echo "Unable to find training log directory for ${METHOD}" >&2
  exit 1
fi

BEST_CKPT="$(python - "${LOG_DIR}/checkpoints" <<'PY'
import re
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
candidates = []
for path in checkpoint_dir.glob("epoch*__monitor*.ckpt"):
    match = re.search(r"monitor(-?[0-9]+(?:\\.[0-9]+)?)", path.name)
    if match:
        candidates.append((float(match.group(1)), path))
if not candidates:
    raise SystemExit(f"No monitored checkpoint found in {checkpoint_dir}")
print(max(candidates, key=lambda item: item[0])[1])
PY
)"
printf '%s\n' "${BEST_CKPT}" > "${OUT_DIR}/${METHOD}_best_ckpt.txt"
echo "[$(date '+%F %T')] BEST method=${METHOD} ckpt=${BEST_CKPT}" | tee -a "${TRAIN_LOG}"

run_ood_eval() {
  local dataset="$1"
  local data_dir="$2"
  local test_file="$3"
  local log_file="${OUT_DIR}/${METHOD}_${dataset}.log"

  echo "[$(date '+%F %T')] START test method=${METHOD} dataset=${dataset} gpu=${GPU}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model="${MODEL}" \
    --dataset=gsm8k_aug_nl \
    --devices=0 \
    --workspace_path=/home/dingxukai \
    --test_ckpt_path="${BEST_CKPT}" \
    --test_times=5 \
    --log_suffix="origin_${METHOD}_qwen3_instruct_${dataset}_ood_gpu${GPU}" \
    dataset_name="${dataset}" \
    dataset_dir="${data_dir}" \
    test_file="${test_file}" \
    max_new_tokens=16 \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    persistent_workers=false \
    trainer.default_root_dir="${ROOT}/run_roots/origin_${METHOD}_qwen3_instruct_${dataset}_ood_gpu${GPU}" \
    2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE test method=${METHOD} dataset=${dataset}" | tee -a "${log_file}"
}

run_ood_eval gsmhard "/home/dingxukai/RoT/data/GSM8k-Hard" "gsmhard_test_processed.jsonl"
run_ood_eval svamp "/home/dingxukai/RoT/data/SVAMP" "svamp_test_processed.jsonl"
run_ood_eval multiarith "/home/dingxukai/RoT/data/Multiarith" "multiarith_test_processed.jsonl"

echo "[$(date '+%F %T')] ALL DONE method=${METHOD}" | tee -a "${OUT_DIR}/${METHOD}_done.log"
