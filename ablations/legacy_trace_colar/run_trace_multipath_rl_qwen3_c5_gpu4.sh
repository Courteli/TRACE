#!/usr/bin/env bash
set -euo pipefail

cd "/disk1/dingxukai/trace_colar"
set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

GPU="${GPU:-4}"
MAX_L="${MAX_L:-40}"
MIN_L="${MIN_L:-0}"
GROUP_SIZE="${GROUP_SIZE:-8}"
EXP_BATCH_SIZE="${EXP_BATCH_SIZE:-1}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-512}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
TEST_TIMES="${TEST_TIMES:-5}"
DO_TEST="${DO_TEST:-True}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
LR="${LR:-1e-6}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-1.0}"
TRACE_REWARD_WEIGHT="${TRACE_REWARD_WEIGHT:-0.35}"
TRACE_BONUS_CLIP="${TRACE_BONUS_CLIP:-2.0}"
SIGNATURE_MEAN_WEIGHT="${SIGNATURE_MEAN_WEIGHT:-0.5}"
SIGNATURE_LAST_WEIGHT="${SIGNATURE_LAST_WEIGHT:-0.5}"
SIGNATURE_TREND_WEIGHT="${SIGNATURE_TREND_WEIGHT:-1.0}"
SIGNATURE_DELTA_WEIGHT="${SIGNATURE_DELTA_WEIGHT:-1.0}"
CENTER_PATH_SIGNATURE="${CENTER_PATH_SIGNATURE:-True}"
PATH_SIGNATURE_RAW_MIX="${PATH_SIGNATURE_RAW_MIX:-0.25}"
MAX_MODES="${MAX_MODES:-3}"
MIN_POSITIVE_MODES="${MIN_POSITIVE_MODES:-2}"
TARGET_POSITIVE_MODES="${TARGET_POSITIVE_MODES:-3}"
MODE_MERGE_THRESHOLD="${MODE_MERGE_THRESHOLD:-0.65}"
POS_MODE_FIT_WEIGHT="${POS_MODE_FIT_WEIGHT:-0.5}"
POS_PAIR_REPULSION_WEIGHT="${POS_PAIR_REPULSION_WEIGHT:-0.35}"
POS_PAIR_MARGIN="${POS_PAIR_MARGIN:-0.75}"
MODE_DIVERSITY_WEIGHT="${MODE_DIVERSITY_WEIGHT:-0.45}"
MODE_BALANCE_WEIGHT="${MODE_BALANCE_WEIGHT:-0.20}"
NEG_REPULSION_WEIGHT="${NEG_REPULSION_WEIGHT:-0.8}"
HARD_REPULSION_WEIGHT="${HARD_REPULSION_WEIGHT:-1.4}"
HARD_POS_REPULSION_WEIGHT="${HARD_POS_REPULSION_WEIGHT:-0.25}"
NEG_MARGIN="${NEG_MARGIN:-0.15}"
HARD_MARGIN="${HARD_MARGIN:--0.05}"
HARD_THRESHOLD="${HARD_THRESHOLD:-0.25}"
STEP_COHERENCE_WEIGHT="${STEP_COHERENCE_WEIGHT:-0.05}"
NONCOLLAPSE_WEIGHT="${NONCOLLAPSE_WEIGHT:-0.05}"
ENABLE_TRACE_MULTIPATH_REWARD="${ENABLE_TRACE_MULTIPATH_REWARD:-True}"
FILTER_INFORMATIVE="${FILTER_INFORMATIVE:-True}"
FILTER_CANDIDATE_FACTOR="${FILTER_CANDIDATE_FACTOR:-1.0}"
FILTER_CANDIDATE_COUNT="${FILTER_CANDIDATE_COUNT:-0}"
FILTER_SCORE_GEOMETRY="${FILTER_SCORE_GEOMETRY:-True}"
FILTER_MAX_BATCHES="${FILTER_MAX_BATCHES:-0}"
RESAMPLE_INFORMATIVE_GROUPS="${RESAMPLE_INFORMATIVE_GROUPS:-True}"
RESAMPLE_MAX_ATTEMPTS="${RESAMPLE_MAX_ATTEMPTS:-3}"
LOG_SUFFIX="${LOG_SUFFIX:-trace_multipath_rl_qwen3_c5_L${MAX_L}_g${GROUP_SIZE}_gpu${GPU}}"
ROOT_DIR="${ROOT_DIR:-/disk1/dingxukai/trace_colar/run_roots/${LOG_SUFFIX}}"
INIT_CKPT="${INIT_CKPT:-logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260525-172642_884973_origin_colar_qwen3_c5_cotinit_rawgsm_lr3e-5_50epoch_gpu5_20260525/checkpoints/epoch15__step107616__monitor0.246.ckpt}"

mkdir -p run_outputs/trace_multipath

TEST_ARGS=()
if [[ "${DO_TEST}" == "True" ]]; then
  TEST_ARGS+=(--do_test)
fi

EXTRA_TRAINER_ARGS=()
if [[ -n "${LIMIT_VAL_BATCHES}" ]]; then
  EXTRA_TRAINER_ARGS+=(trainer.limit_val_batches="${LIMIT_VAL_BATCHES}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
  --model=trace_multipath_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  --load_ckpt_path="${INIT_CKPT}" \
  "${TEST_ARGS[@]}" \
  --test_times="${TEST_TIMES}" \
  --workspace_path=/home/dingxukai \
  --log_suffix="${LOG_SUFFIX}" \
  batch_size=1 \
  val_batch_size=1 \
  max_epochs="${MAX_EPOCHS}" \
  num_sanity_val_steps=0 \
  gradient_clip_val=0 \
  max_n_latent_forward="${MAX_L}" \
  min_n_latent_forward="${MIN_L}" \
  latent_temperature="${LATENT_TEMPERATURE}" \
  eol_temperature=1.0 \
  max_new_tokens=16 \
  do_rl=True \
  group_size="${GROUP_SIZE}" \
  exp_batch_size="${EXP_BATCH_SIZE}" \
  n_train_samples_per_epoch="${N_TRAIN_SAMPLES}" \
  enable_trace_multipath_reward="${ENABLE_TRACE_MULTIPATH_REWARD}" \
  filter_informative_train_indices="${FILTER_INFORMATIVE}" \
  filter_candidate_factor="${FILTER_CANDIDATE_FACTOR}" \
  filter_candidate_count="${FILTER_CANDIDATE_COUNT}" \
  filter_score_geometry="${FILTER_SCORE_GEOMETRY}" \
  filter_max_batches="${FILTER_MAX_BATCHES}" \
  trace_reward_weight="${TRACE_REWARD_WEIGHT}" \
  trace_bonus_clip="${TRACE_BONUS_CLIP}" \
  signature_mean_weight="${SIGNATURE_MEAN_WEIGHT}" \
  signature_last_weight="${SIGNATURE_LAST_WEIGHT}" \
  signature_trend_weight="${SIGNATURE_TREND_WEIGHT}" \
  signature_delta_weight="${SIGNATURE_DELTA_WEIGHT}" \
  center_path_signature="${CENTER_PATH_SIGNATURE}" \
  path_signature_raw_mix="${PATH_SIGNATURE_RAW_MIX}" \
  max_modes="${MAX_MODES}" \
  min_positive_modes="${MIN_POSITIVE_MODES}" \
  target_positive_modes="${TARGET_POSITIVE_MODES}" \
  mode_merge_threshold="${MODE_MERGE_THRESHOLD}" \
  pos_mode_fit_weight="${POS_MODE_FIT_WEIGHT}" \
  pos_pair_repulsion_weight="${POS_PAIR_REPULSION_WEIGHT}" \
  pos_pair_margin="${POS_PAIR_MARGIN}" \
  mode_diversity_weight="${MODE_DIVERSITY_WEIGHT}" \
  mode_balance_weight="${MODE_BALANCE_WEIGHT}" \
  neg_repulsion_weight="${NEG_REPULSION_WEIGHT}" \
  hard_repulsion_weight="${HARD_REPULSION_WEIGHT}" \
  hard_pos_repulsion_weight="${HARD_POS_REPULSION_WEIGHT}" \
  neg_margin="${NEG_MARGIN}" \
  hard_margin="${HARD_MARGIN}" \
  hard_threshold="${HARD_THRESHOLD}" \
  step_coherence_weight="${STEP_COHERENCE_WEIGHT}" \
  noncollapse_weight="${NONCOLLAPSE_WEIGHT}" \
  resample_informative_groups="${RESAMPLE_INFORMATIVE_GROUPS}" \
  resample_max_attempts="${RESAMPLE_MAX_ATTEMPTS}" \
  lr="${LR}" \
  trainer.default_root_dir="${ROOT_DIR}" \
  "${EXTRA_TRAINER_ARGS[@]}" \
  2>&1 | tee -a "run_outputs/trace_multipath/${LOG_SUFFIX}.log"
