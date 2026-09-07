#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?Usage: $0 METHOD GPU MAX_EPOCHS}"
GPU="${2:?Usage: $0 METHOD GPU MAX_EPOCHS}"
MAX_EPOCHS="${3:?Usage: $0 METHOD GPU MAX_EPOCHS}"

ROOT="/home/dingxukai/colar origin"
OUT_DIR="${ROOT}/run_outputs/qwen3_early_epoch_retrain_eval_20260602"
COT_CKPT="${ROOT}/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260426-134652_725405_origin_cot_qwen3_instruct_rawgsm_lr3e-5_3epoch_gpus1_2/checkpoints/epoch1__step6726__monitor0.857.ckpt"
MODEL="${METHOD}_qwen3_instruct"
SUFFIX="origin_${METHOD}_qwen3_early${MAX_EPOCHS}_cotinit_lr3e-5_gpu${GPU}"
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

echo "[$(date '+%F %T')] START train method=${METHOD} epochs=${MAX_EPOCHS} gpu=${GPU}" | tee -a "${TRAIN_LOG}"
cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
  --model="${MODEL}" \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  --load_ckpt_path="${COT_CKPT}" \
  --workspace_path=/home/dingxukai \
  --log_suffix="${SUFFIX}" \
  batch_size=1 \
  val_batch_size=1 \
  max_epochs="${MAX_EPOCHS}" \
  num_sanity_val_steps=0 \
  max_new_tokens=16 \
  lr=3e-5 \
  callbacks.save_top_k=-1 \
  callbacks.save_last=false \
  trainer.default_root_dir="${ROOT}/run_roots/${SUFFIX}" \
  2>&1 | tee -a "${TRAIN_LOG}"

LOG_DIR="$(find "${ROOT}/logs/${MODEL}/gsm8k_aug_nl-gsm8k_aug_nl" -mindepth 1 -maxdepth 1 -type d -name "*_${SUFFIX}" | sort | tail -n 1)"
if [[ -z "${LOG_DIR}" ]]; then
  echo "Unable to find training log directory for ${METHOD}" >&2
  exit 1
fi

mapfile -t CKPTS < <(find "${LOG_DIR}/checkpoints" -maxdepth 1 -type f -name 'epoch*__monitor*.ckpt' | sort -V)
if [[ "${#CKPTS[@]}" -ne "${MAX_EPOCHS}" ]]; then
  echo "Expected ${MAX_EPOCHS} monitored checkpoints for ${METHOD}, found ${#CKPTS[@]}" >&2
  printf '  %s\n' "${CKPTS[@]}" >&2
  exit 1
fi

MANIFEST="${OUT_DIR}/${METHOD}_checkpoints.tsv"
printf 'paper_epoch\tlightning_checkpoint\n' > "${MANIFEST}"
for index in "${!CKPTS[@]}"; do
  paper_epoch="$((index + 1))"
  printf '%s\t%s\n' "${paper_epoch}" "${CKPTS[$index]}" >> "${MANIFEST}"
done

run_eval() {
  local paper_epoch="$1"
  local ckpt="$2"
  local dataset="$3"
  local data_dir="$4"
  local test_file="$5"
  local log_file="${OUT_DIR}/${METHOD}_epoch${paper_epoch}_${dataset}.log"

  echo "[$(date '+%F %T')] START test method=${METHOD} paper_epoch=${paper_epoch} dataset=${dataset} gpu=${GPU} ckpt=${ckpt}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model="${MODEL}" \
    --dataset=gsm8k_aug_nl \
    --devices=0 \
    --workspace_path=/home/dingxukai \
    --test_ckpt_path="${ckpt}" \
    --test_times=5 \
    --log_suffix="origin_${METHOD}_qwen3_epoch${paper_epoch}_${dataset}_eval_gpu${GPU}" \
    dataset_name="${dataset}" \
    dataset_dir="${data_dir}" \
    test_file="${test_file}" \
    max_new_tokens=16 \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    persistent_workers=false \
    trainer.default_root_dir="${ROOT}/run_roots/origin_${METHOD}_qwen3_epoch${paper_epoch}_${dataset}_eval_gpu${GPU}" \
    2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] DONE test method=${METHOD} paper_epoch=${paper_epoch} dataset=${dataset}" | tee -a "${log_file}"
}

for index in "${!CKPTS[@]}"; do
  paper_epoch="$((index + 1))"
  ckpt="${CKPTS[$index]}"
  run_eval "${paper_epoch}" "${ckpt}" gsm8k_aug_nl "/home/dingxukai/RoT/data/GSM8k-Aug-NL" "gsm8k_test_processed.jsonl"
  run_eval "${paper_epoch}" "${ckpt}" gsmhard "/home/dingxukai/RoT/data/GSM8k-Hard" "gsmhard_test_processed.jsonl"
  run_eval "${paper_epoch}" "${ckpt}" svamp "/home/dingxukai/RoT/data/SVAMP" "svamp_test_processed.jsonl"
  run_eval "${paper_epoch}" "${ckpt}" multiarith "/home/dingxukai/RoT/data/Multiarith" "multiarith_test_processed.jsonl"
done

echo "[$(date '+%F %T')] ALL DONE method=${METHOD}" | tee -a "${OUT_DIR}/${METHOD}_done.log"
