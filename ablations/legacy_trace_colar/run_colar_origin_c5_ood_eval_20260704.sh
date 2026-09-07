#!/usr/bin/env bash
set -euo pipefail

cd "/disk1/dingxukai/trace_colar"
set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

GPU="${GPU:-4}"
TEST_TIMES="${TEST_TIMES:-5}"
MAX_L="${MAX_L:-40}"
ORIGIN_CKPT="${ORIGIN_CKPT:-logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260427-002951_286878_origin_colar_qwen3_instruct_r5_cotinit_rawgsm_lr3e-5_50epoch_gpu3/checkpoints/epoch49__step336300__monitor0.256.ckpt}"
OUT_DIR="${OUT_DIR:-run_outputs/trace/baselines}"

mkdir -p "${OUT_DIR}"

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

  VERSION="trace_baseline_colar_origin_r5_full50_L${MAX_L}_${DATASET}_gpu${GPU}"
  echo "[CoLaR origin baseline] dataset=${DATASET} checkpoint=${ORIGIN_CKPT}" | tee -a "${OUT_DIR}/${VERSION}.log"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
    --model=colar_qwen3_instruct \
    --dataset=gsm8k_aug_nl \
    --devices=0 \
    --workspace_path=/home/dingxukai \
    --test_ckpt_path="${ORIGIN_CKPT}" \
    --test_times="${TEST_TIMES}" \
    dataset_name="${DATASET}" \
    dataset_dir="${DATASET_DIR}" \
    test_file="${DATA_FILE}" \
    max_compression_factor=5 \
    compression_factor=5 \
    max_n_latent_forward="${MAX_L}" \
    max_new_tokens=16 \
    batch_size=1 \
    val_batch_size=1 \
    num_sanity_val_steps=0 \
    trainer.logger.name=trace_baselines \
    trainer.logger.version="${VERSION}" \
    trainer.default_root_dir="/disk1/dingxukai/trace_colar/run_roots/${VERSION}" \
    2>&1 | tee -a "${OUT_DIR}/${VERSION}.log"
done
