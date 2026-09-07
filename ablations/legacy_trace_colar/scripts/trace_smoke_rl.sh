#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
CKPT="${2:-}"
ROOT="/disk1/dingxukai/trace_colar"
FALLBACK_CKPT="${ROOT}/logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260525-172642_884973_origin_colar_qwen3_c5_cotinit_rawgsm_lr3e-5_50epoch_gpu5_20260525/checkpoints/epoch15__step107616__monitor0.246.ckpt"
if [[ -z "${CKPT}" ]]; then
  CKPT="${FALLBACK_CKPT}"
fi

cd "${ROOT}"
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT

CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
  --model=trace_colar_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  --load_ckpt_path="${CKPT}" \
  --workspace_path=/home/dingxukai \
  --no_log \
  tiny_dataset=True \
  batch_size=1 \
  val_batch_size=1 \
  max_epochs=1 \
  max_steps=1 \
  num_sanity_val_steps=0 \
  gradient_clip_val=0 \
  num_workers=0 \
  persistent_workers=False \
  do_rl=True \
  enable_trace_reward=True \
  group_size=4 \
  exp_batch_size=4 \
  n_train_samples_per_epoch=4 \
  trace_reward_weight=0.1 \
  max_compression_factor=5 \
  compression_factor=5 \
  max_new_tokens=8 \
  lr=3e-5
