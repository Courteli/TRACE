#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

INIT_CKPT="${INIT_CKPT:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260706-032913_586957_trace_v3_three_stage_full_20260705_stage1solid_stage2_trace_multipath_rl/checkpoints/epoch4__step2560__monitor0.347.ckpt}"
GPU="${GPU:-2}"
RUN_SUFFIX="${RUN_SUFFIX:-trace_v3_three_stage_tuned_stage2_from_epoch4_20260706}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706/stage2_tuned}"
MAX_EPOCHS="${MAX_EPOCHS:-4}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-512}"
GROUP_SIZE="${GROUP_SIZE:-8}"
EXP_BATCH_SIZE="${EXP_BATCH_SIZE:-8}"
LR="${LR:-5e-7}"
MAX_L="${MAX_L:-40}"
MIN_L="${MIN_L:-8}"
LATENT_TEMP="${LATENT_TEMP:-1.0}"
EOL_TEMP="${EOL_TEMP:-1.0}"
COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-5}"
EVAL_MAX_L="${EVAL_MAX_L:-${MAX_L}}"
EVAL_MIN_L="${EVAL_MIN_L:-${MIN_L}}"
EVAL_LATENT_TEMP="${EVAL_LATENT_TEMP:-${LATENT_TEMP}}"
EVAL_EOL_TEMP="${EVAL_EOL_TEMP:-${EOL_TEMP}}"
EVAL_COMPRESSION_FACTOR="${EVAL_COMPRESSION_FACTOR:-${COMPRESSION_FACTOR}}"
TEST_TIMES="${TEST_TIMES:-1}"
RESAMPLE_MAX_ATTEMPTS="${RESAMPLE_MAX_ATTEMPTS:-2}"
RESAMPLE_TARGET_SCORE="${RESAMPLE_TARGET_SCORE:-0.75}"
FILTER_CANDIDATE_FACTOR="${FILTER_CANDIDATE_FACTOR:-1.0}"
FILTER_CANDIDATE_COUNT="${FILTER_CANDIDATE_COUNT:-0}"
FILTER_INFORMATIVE_TRAIN_INDICES="${FILTER_INFORMATIVE_TRAIN_INDICES:-True}"
FILTER_SCORE_GEOMETRY="${FILTER_SCORE_GEOMETRY:-True}"
FILTER_MIXED_FRACTION="${FILTER_MIXED_FRACTION:-0.7}"
FILTER_POSITIVE_FRACTION="${FILTER_POSITIVE_FRACTION:-0.9}"
RESAMPLE_INFORMATIVE_GROUPS="${RESAMPLE_INFORMATIVE_GROUPS:-True}"
TRACE_REWARD_WEIGHT="${TRACE_REWARD_WEIGHT:-0.20}"
MODE_DIVERSITY_WEIGHT="${MODE_DIVERSITY_WEIGHT:-0.25}"
MODE_BALANCE_WEIGHT="${MODE_BALANCE_WEIGHT:-0.10}"
NEG_REPULSION_WEIGHT="${NEG_REPULSION_WEIGHT:-0.50}"
HARD_REPULSION_WEIGHT="${HARD_REPULSION_WEIGHT:-0.80}"
HARD_POS_REPULSION_WEIGHT="${HARD_POS_REPULSION_WEIGHT:-0.10}"
POS_PAIR_REPULSION_WEIGHT="${POS_PAIR_REPULSION_WEIGHT:-0.20}"
POS_MODE_FIT_WEIGHT="${POS_MODE_FIT_WEIGHT:-0.50}"
STEP_COHERENCE_WEIGHT="${STEP_COHERENCE_WEIGHT:-0.05}"
NONCOLLAPSE_WEIGHT="${NONCOLLAPSE_WEIGHT:-0.05}"
TRACE_BONUS_CLIP="${TRACE_BONUS_CLIP:-2.0}"
CLIP_EPS="${CLIP_EPS:-0.2}"
USE_LATENT_LOSS="${USE_LATENT_LOSS:-True}"
USE_ANSWER_LOSS="${USE_ANSWER_LOSS:-True}"
AVERAGE_PER_TOKEN_LOSS="${AVERAGE_PER_TOKEN_LOSS:-False}"
PUNISH_LATENT_LENGTH="${PUNISH_LATENT_LENGTH:-False}"
FREEZE_LLM_STAGE2="${FREEZE_LLM_STAGE2:-False}"
STAGE2_SFT_REPLAY_WEIGHT="${STAGE2_SFT_REPLAY_WEIGHT:-0.0}"
STAGE2_ALL_WRONG_REPLAY_MULTIPLIER="${STAGE2_ALL_WRONG_REPLAY_MULTIPLIER:-0.0}"
ALL_WRONG_NUMERIC_REWARD_WEIGHT="${ALL_WRONG_NUMERIC_REWARD_WEIGHT:-0.0}"
ALL_WRONG_NUMERIC_REWARD_CLIP="${ALL_WRONG_NUMERIC_REWARD_CLIP:-2.0}"
NUMERIC_DENSE_REWARD_WEIGHT="${NUMERIC_DENSE_REWARD_WEIGHT:-0.0}"
NUMERIC_DENSE_REWARD_CLIP="${NUMERIC_DENSE_REWARD_CLIP:-2.0}"
NEAR_MISS_PATH_BOOTSTRAP_WEIGHT="${NEAR_MISS_PATH_BOOTSTRAP_WEIGHT:-0.0}"
NEAR_MISS_PATH_BOOTSTRAP_CLIP="${NEAR_MISS_PATH_BOOTSTRAP_CLIP:-2.0}"
NEAR_MISS_PATH_MIN_SPAN="${NEAR_MISS_PATH_MIN_SPAN:-0.02}"
NEAR_MISS_PATH_MIN_SCORE="${NEAR_MISS_PATH_MIN_SCORE:-0.0}"
NEAR_MISS_PATH_NEG_MARGIN="${NEAR_MISS_PATH_NEG_MARGIN:-0.15}"

mkdir -p "${ARTIFACT_DIR}"
PIPELINE_LOG="${ARTIFACT_DIR}/stage2_tuned.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "[stage2-tune] start $(date '+%F %T')"
echo "[stage2-tune] init_ckpt=${INIT_CKPT}"
echo "[stage2-tune] gpu=${GPU} epochs=${MAX_EPOCHS} n_train=${N_TRAIN_SAMPLES} group=${GROUP_SIZE} lr=${LR} min_l=${MIN_L} max_l=${MAX_L} latent_temp=${LATENT_TEMP} eol_temp=${EOL_TEMP} cf=${COMPRESSION_FACTOR}"
echo "[stage2-tune] eval min_l=${EVAL_MIN_L} max_l=${EVAL_MAX_L} latent_temp=${EVAL_LATENT_TEMP} eol_temp=${EVAL_EOL_TEMP} cf=${EVAL_COMPRESSION_FACTOR}"
echo "[stage2-tune] trace_weight=${TRACE_REWARD_WEIGHT} mode_div=${MODE_DIVERSITY_WEIGHT} hard_rep=${HARD_REPULSION_WEIGHT} pos_pair=${POS_PAIR_REPULSION_WEIGHT} resample_target=${RESAMPLE_TARGET_SCORE}"
echo "[stage2-tune] filter_enabled=${FILTER_INFORMATIVE_TRAIN_INDICES} filter_factor=${FILTER_CANDIDATE_FACTOR} filter_count=${FILTER_CANDIDATE_COUNT} filter_score_geometry=${FILTER_SCORE_GEOMETRY} mixed_frac=${FILTER_MIXED_FRACTION} positive_frac=${FILTER_POSITIVE_FRACTION} resample_groups=${RESAMPLE_INFORMATIVE_GROUPS}"
echo "[stage2-tune] pos_fit=${POS_MODE_FIT_WEIGHT} step=${STEP_COHERENCE_WEIGHT} noncollapse=${NONCOLLAPSE_WEIGHT} bonus_clip=${TRACE_BONUS_CLIP} clip_eps=${CLIP_EPS} use_latent=${USE_LATENT_LOSS} use_answer=${USE_ANSWER_LOSS} avg_token_loss=${AVERAGE_PER_TOKEN_LOSS} punish_latent_length=${PUNISH_LATENT_LENGTH} freeze_llm_stage2=${FREEZE_LLM_STAGE2} stage2_sft_replay_weight=${STAGE2_SFT_REPLAY_WEIGHT} stage2_all_wrong_replay_multiplier=${STAGE2_ALL_WRONG_REPLAY_MULTIPLIER} all_wrong_numeric_reward_weight=${ALL_WRONG_NUMERIC_REWARD_WEIGHT} all_wrong_numeric_reward_clip=${ALL_WRONG_NUMERIC_REWARD_CLIP} numeric_dense_reward_weight=${NUMERIC_DENSE_REWARD_WEIGHT} numeric_dense_reward_clip=${NUMERIC_DENSE_REWARD_CLIP} near_miss_path_bootstrap_weight=${NEAR_MISS_PATH_BOOTSTRAP_WEIGHT} near_miss_path_bootstrap_clip=${NEAR_MISS_PATH_BOOTSTRAP_CLIP} near_miss_path_min_span=${NEAR_MISS_PATH_MIN_SPAN} near_miss_path_min_score=${NEAR_MISS_PATH_MIN_SCORE} near_miss_path_neg_margin=${NEAR_MISS_PATH_NEG_MARGIN}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
  --model=trace_multipath_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  --load_ckpt_path="${INIT_CKPT}" \
  --test_times="${TEST_TIMES}" \
  --workspace_path=/home/dingxukai \
  --log_suffix="${RUN_SUFFIX}" \
  batch_size=1 \
  val_batch_size=1 \
  max_epochs="${MAX_EPOCHS}" \
  num_sanity_val_steps=0 \
  trainer.limit_val_batches=1.0 \
  gradient_clip_val=0 \
  max_n_latent_forward="${MAX_L}" \
  min_n_latent_forward="${MIN_L}" \
  latent_temperature="${LATENT_TEMP}" \
  eol_temperature="${EOL_TEMP}" \
  compression_factor="${COMPRESSION_FACTOR}" \
  max_new_tokens=16 \
  do_rl=True \
  group_size="${GROUP_SIZE}" \
  exp_batch_size="${EXP_BATCH_SIZE}" \
  n_train_samples_per_epoch="${N_TRAIN_SAMPLES}" \
  clip_eps="${CLIP_EPS}" \
  use_latent_loss="${USE_LATENT_LOSS}" \
  use_answer_loss="${USE_ANSWER_LOSS}" \
  average_per_token_loss="${AVERAGE_PER_TOKEN_LOSS}" \
  punish_latent_length="${PUNISH_LATENT_LENGTH}" \
  enable_trace_path_pretraining=True \
  path_use_policy_mean=True \
  path_anchor_jitter=1 \
  path_multiview_bootstrap_mix=0.20 \
  stage2_sft_replay_weight="${STAGE2_SFT_REPLAY_WEIGHT}" \
  stage2_all_wrong_replay_multiplier="${STAGE2_ALL_WRONG_REPLAY_MULTIPLIER}" \
  all_wrong_numeric_reward_weight="${ALL_WRONG_NUMERIC_REWARD_WEIGHT}" \
  all_wrong_numeric_reward_clip="${ALL_WRONG_NUMERIC_REWARD_CLIP}" \
  numeric_dense_reward_weight="${NUMERIC_DENSE_REWARD_WEIGHT}" \
  numeric_dense_reward_clip="${NUMERIC_DENSE_REWARD_CLIP}" \
  near_miss_path_bootstrap_weight="${NEAR_MISS_PATH_BOOTSTRAP_WEIGHT}" \
  near_miss_path_bootstrap_clip="${NEAR_MISS_PATH_BOOTSTRAP_CLIP}" \
  near_miss_path_min_span="${NEAR_MISS_PATH_MIN_SPAN}" \
  near_miss_path_min_score="${NEAR_MISS_PATH_MIN_SCORE}" \
  near_miss_path_neg_margin="${NEAR_MISS_PATH_NEG_MARGIN}" \
  freeze_llm_stage2="${FREEZE_LLM_STAGE2}" \
  enable_trace_multipath_reward=True \
  filter_informative_train_indices="${FILTER_INFORMATIVE_TRAIN_INDICES}" \
  filter_candidate_factor="${FILTER_CANDIDATE_FACTOR}" \
  filter_candidate_count="${FILTER_CANDIDATE_COUNT}" \
  filter_score_geometry="${FILTER_SCORE_GEOMETRY}" \
  filter_max_batches=0 \
  filter_mixed_fraction="${FILTER_MIXED_FRACTION}" \
  filter_positive_fraction="${FILTER_POSITIVE_FRACTION}" \
  resample_informative_groups="${RESAMPLE_INFORMATIVE_GROUPS}" \
  resample_max_attempts="${RESAMPLE_MAX_ATTEMPTS}" \
  resample_target_score="${RESAMPLE_TARGET_SCORE}" \
  trace_bonus_clip="${TRACE_BONUS_CLIP}" \
  trace_reward_weight="${TRACE_REWARD_WEIGHT}" \
  pos_mode_fit_weight="${POS_MODE_FIT_WEIGHT}" \
  mode_diversity_weight="${MODE_DIVERSITY_WEIGHT}" \
  mode_balance_weight="${MODE_BALANCE_WEIGHT}" \
  neg_repulsion_weight="${NEG_REPULSION_WEIGHT}" \
  hard_repulsion_weight="${HARD_REPULSION_WEIGHT}" \
  hard_pos_repulsion_weight="${HARD_POS_REPULSION_WEIGHT}" \
  pos_pair_repulsion_weight="${POS_PAIR_REPULSION_WEIGHT}" \
  step_coherence_weight="${STEP_COHERENCE_WEIGHT}" \
  noncollapse_weight="${NONCOLLAPSE_WEIGHT}" \
  lr="${LR}" \
  trainer.default_root_dir="${ROOT}/run_roots/${RUN_SUFFIX}"

BEST_CKPT="$(/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_find_checkpoint.py \
  --log_root logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl \
  --run_contains "${RUN_SUFFIX}" \
  --prefer best)"
echo "[stage2-tune] best_ckpt=${BEST_CKPT}"
printf '%s\n' "${BEST_CKPT}" > "${ARTIFACT_DIR}/best_ckpt.txt"

echo "[stage2-tune] full GSM8K eval with tuned inference"
CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
  --model=trace_multipath_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  --test_ckpt_path="${BEST_CKPT}" \
  --test_times="${TEST_TIMES}" \
  --seed=0 \
  --workspace_path=/home/dingxukai \
  batch_size=1 \
  val_batch_size=1 \
  num_workers=0 \
  persistent_workers=False \
  num_sanity_val_steps=0 \
  trainer.logger.save_dir="${ARTIFACT_DIR}/eval_logs" \
  trainer.logger.name="tb" \
  trainer.logger.version="gsm8k_min${EVAL_MIN_L}" \
  trainer.default_root_dir="${ARTIFACT_DIR}/eval_root" \
  dataset_name="gsm8k_aug_nl" \
  dataset_dir="/home/dingxukai/RoT/data/GSM8k-Aug-NL" \
  test_file="gsm8k_test_processed.jsonl" \
  max_n_latent_forward="${EVAL_MAX_L}" \
  min_n_latent_forward="${EVAL_MIN_L}" \
  latent_temperature="${EVAL_LATENT_TEMP}" \
  eol_temperature="${EVAL_EOL_TEMP}" \
  compression_factor="${EVAL_COMPRESSION_FACTOR}" \
  max_new_tokens=16

JSON_PATH="$(find "${ARTIFACT_DIR}/eval_logs/tb/gsm8k_min${EVAL_MIN_L}" -maxdepth 1 -type f -name 'test_*.json' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "${JSON_PATH}" ]]; then
  cp -f "${JSON_PATH}" "${ARTIFACT_DIR}/gsm8k_min${EVAL_MIN_L}.json"
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_eval_json_summary.py --csv "${ARTIFACT_DIR}/gsm8k_min${EVAL_MIN_L}.json" > "${ARTIFACT_DIR}/gsm8k_min${EVAL_MIN_L}_summary.csv"
  cat "${ARTIFACT_DIR}/gsm8k_min${EVAL_MIN_L}_summary.csv"
fi

echo "[stage2-tune] done $(date '+%F %T')"
