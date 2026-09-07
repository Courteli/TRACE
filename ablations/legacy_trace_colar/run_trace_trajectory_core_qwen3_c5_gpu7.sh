#!/usr/bin/env bash
set -euo pipefail

cd "/disk1/dingxukai/trace_colar"
set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

GPU="${GPU:-7}"
TRACE_STEPS="${TRACE_STEPS:-8}"
ANSWER_WEIGHT="${ANSWER_WEIGHT:-1.0}"
STATE_WEIGHT="${STATE_WEIGHT:-1.0}"
TRANSITION_WEIGHT="${TRANSITION_WEIGHT:-1.0}"
ROLE_WEIGHT="${ROLE_WEIGHT:-0.0}"
USE_MEAN_LATENTS_TRAIN="${USE_MEAN_LATENTS_TRAIN:-True}"
USE_MEAN_LATENTS_EVAL="${USE_MEAN_LATENTS_EVAL:-True}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
TEST_TIMES="${TEST_TIMES:-5}"
LOG_SUFFIX="${LOG_SUFFIX:-trace_trajectory_core_qwen3_c5_k${TRACE_STEPS}_gpu${GPU}}"
ROOT_DIR="${ROOT_DIR:-/disk1/dingxukai/trace_colar/run_roots/${LOG_SUFFIX}}"
INIT_CKPT="${INIT_CKPT:-logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260525-172642_884973_origin_colar_qwen3_c5_cotinit_rawgsm_lr3e-5_50epoch_gpu5_20260525/checkpoints/epoch15__step107616__monitor0.246.ckpt}"
CACHE_DIR="${CACHE_DIR:-run_outputs/trace/teacher_cache/k${TRACE_STEPS}}"
BUILD_TEACHER_CACHE="${BUILD_TEACHER_CACHE:-1}"

mkdir -p run_outputs/trace
mkdir -p "${CACHE_DIR}"

if [[ "${BUILD_TEACHER_CACHE}" == "1" ]]; then
  for SPLIT in train val test; do
    case "${SPLIT}" in
      train) STEM="gsm8k_train_processed" ;;
      val) STEM="gsm8k_val_processed" ;;
      test) STEM="gsm8k_test_processed" ;;
    esac
    CACHE_FILE="${CACHE_DIR}/${STEM}_trace_teacher_k${TRACE_STEPS}.pt"
    if [[ ! -f "${CACHE_FILE}" ]]; then
      echo "[stage1] Building frozen teacher cache: ${CACHE_FILE}"
      CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/trace_build_teacher_cache.py \
        --model trace_trajectory_qwen3_instruct \
        --dataset gsm8k_aug_nl \
        --split "${SPLIT}" \
        --trace_steps "${TRACE_STEPS}" \
        --device cuda:0 \
        --load_ckpt_path "${INIT_CKPT}" \
        --out_dir "${CACHE_DIR}" \
        --batch_size 1
    else
      echo "[stage1] Reusing frozen teacher cache: ${CACHE_FILE}"
    fi
  done
fi

CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
  --model=trace_trajectory_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  --load_ckpt_path="${INIT_CKPT}" \
  --do_test \
  --test_times="${TEST_TIMES}" \
  --workspace_path=/home/dingxukai \
  --log_suffix="${LOG_SUFFIX}" \
  batch_size=1 \
  val_batch_size=1 \
  max_epochs="${MAX_EPOCHS}" \
  num_sanity_val_steps=0 \
  gradient_clip_val=0 \
  max_n_latent_forward="${TRACE_STEPS}" \
  min_n_latent_forward="${TRACE_STEPS}" \
  trace_steps="${TRACE_STEPS}" \
  teacher_speed="${TRACE_STEPS}" \
  student_speed="${TRACE_STEPS}" \
  answer_weight="${ANSWER_WEIGHT}" \
  state_weight="${STATE_WEIGHT}" \
  transition_weight="${TRANSITION_WEIGHT}" \
  role_weight="${ROLE_WEIGHT}" \
  use_mean_latents_train="${USE_MEAN_LATENTS_TRAIN}" \
  use_mean_latents_eval="${USE_MEAN_LATENTS_EVAL}" \
  trace_teacher_cache_dir="${CACHE_DIR}" \
  max_new_tokens=16 \
  lr=1e-6 \
  trainer.default_root_dir="${ROOT_DIR}" \
  2>&1 | tee -a "run_outputs/trace/${LOG_SUFFIX}.log"
