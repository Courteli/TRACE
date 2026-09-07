#!/usr/bin/env bash
set -euo pipefail

cd "/disk1/dingxukai/trace_colar"
set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

RUN_CONTAINS="${1:-trace_multipath_full_qwen3_c5_L40}"
GPU="${GPU:-4}"
TEST_TIMES="${TEST_TIMES:-1}"
CKPT="${CKPT:-}"

if [[ -z "${CKPT}" ]]; then
  CKPT="$(/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_find_checkpoint.py \
    --log_root logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl \
    --run_contains "${RUN_CONTAINS}" \
    --prefer best)"
fi

echo "[TRACE MultiPath OOD] checkpoint: ${CKPT}"

for DATASET in gsm8k_aug_nl gsmhard svamp multiarith; do
  case "${DATASET}" in
    gsm8k_aug_nl)
      DATASET_DIR="/home/dingxukai/RoT/data/GSM8k-Aug-NL"
      DATA_FILE="gsm8k_test_processed.jsonl"
      ;;
    gsmhard)
      DATASET_DIR="/home/dingxukai/RoT/data/GSM8k-Hard"
      DATA_FILE="gsmhard_test_processed.jsonl"
      ;;
    svamp)
      DATASET_DIR="/home/dingxukai/RoT/data/SVAMP"
      DATA_FILE="svamp_test_processed.jsonl"
      ;;
    multiarith)
      DATASET_DIR="/home/dingxukai/RoT/data/Multiarith"
      DATA_FILE="multiarith_test_processed.jsonl"
      ;;
  esac
  echo "[TRACE MultiPath OOD] dataset=${DATASET}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
    --model=trace_multipath_qwen3_instruct \
    --dataset="${DATASET}" \
    --devices=0 \
    --test_ckpt_path="${CKPT}" \
    --test_times="${TEST_TIMES}" \
    --workspace_path=/home/dingxukai \
    batch_size=1 \
    val_batch_size=1 \
    num_sanity_val_steps=0 \
    dataset_name="${DATASET}" \
    dataset_dir="${DATASET_DIR}" \
    test_file="${DATA_FILE}"
done
