#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT="/disk1/dingxukai/trace_colar"
COT_CKPT="${ROOT}/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260426-134652_725405_origin_cot_qwen3_instruct_rawgsm_lr3e-5_3epoch_gpus1_2/checkpoints/epoch1__step6726__monitor0.857.ckpt"

cd "${ROOT}"
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT

CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
  --model=trace_colar_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  --load_ckpt_path="${COT_CKPT}" \
  --workspace_path=/home/dingxukai \
  --no_log \
  tiny_dataset=True \
  batch_size=1 \
  val_batch_size=1 \
  max_epochs=1 \
  max_steps=2 \
  num_sanity_val_steps=0 \
  num_workers=0 \
  persistent_workers=False \
  max_compression_factor=5 \
  compression_factor=5 \
  max_new_tokens=8 \
  lr=3e-5
