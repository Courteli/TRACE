#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

TRAIN_FILE="/home/dingxukai/RoT/data/GSM8k-Aug-NL/gsm8k_train_processed.jsonl"
VAL_FILE="/home/dingxukai/RoT/data/GSM8k-Aug-NL/gsm8k_val_processed.jsonl"
TRAIN_COUNT="$(wc -l < "${TRAIN_FILE}")"
VAL_COUNT="$(wc -l < "${VAL_FILE}")"

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-trace_v3_three_stage_full_${STAMP}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/${RUN_NAME}}"
PIPELINE_LOG="${PIPELINE_LOG:-${ARTIFACT_DIR}/pipeline.log}"
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

STAGE0_GPUS="${STAGE0_GPUS:-1,2}"
STAGE0_DEVICES="${STAGE0_DEVICES:-0,1}"
STAGE0_BATCH_SIZE="${STAGE0_BATCH_SIZE:-2}"
STAGE0_MAX_EPOCHS="${STAGE0_MAX_EPOCHS:-3}"
STAGE0_LR="${STAGE0_LR:-3e-5}"
STAGE0_CKPT="${STAGE0_CKPT:-${ROOT}/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260426-134652_725405_origin_cot_qwen3_instruct_rawgsm_lr3e-5_3epoch_gpus1_2/checkpoints/epoch1__step6726__monitor0.857.ckpt}"
RUN_STAGE0="${RUN_STAGE0:-False}"

STAGE1_GPU="${STAGE1_GPU:-4}"
STAGE1_TRAINER="${STAGE1_TRAINER:-trace_stage1_patience4}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-1}"
STAGE1_MAX_EPOCHS="${STAGE1_MAX_EPOCHS:-50}"
STAGE1_LR="${STAGE1_LR:-3e-5}"
STAGE1_N_TRAIN_SAMPLES="${STAGE1_N_TRAIN_SAMPLES:-${TRAIN_COUNT}}"
RUN_STAGE1_TEST="${RUN_STAGE1_TEST:-False}"
PATH_PRETRAINING_WEIGHT="${PATH_PRETRAINING_WEIGHT:-0.15}"
PATH_ANCHOR_JITTER="${PATH_ANCHOR_JITTER:-1}"
PATH_MULTIVIEW_BOOTSTRAP_MIX="${PATH_MULTIVIEW_BOOTSTRAP_MIX:-0.20}"

STAGE2_GPU="${STAGE2_GPU:-4}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-1}"
STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-10}"
STAGE2_N_TRAIN_SAMPLES="${STAGE2_N_TRAIN_SAMPLES:-512}"
STAGE2_GROUP_SIZE="${STAGE2_GROUP_SIZE:-8}"
STAGE2_EXP_BATCH_SIZE="${STAGE2_EXP_BATCH_SIZE:-8}"
STAGE2_LR="${STAGE2_LR:-1e-6}"
MAX_L="${MAX_L:-40}"
MIN_L="${MIN_L:-0}"
TEST_TIMES="${TEST_TIMES:-1}"
VIS_CANDIDATE_COUNT="${VIS_CANDIDATE_COUNT:-200}"
RUN_COLAR_BASELINE="${RUN_COLAR_BASELINE:-True}"

echo "[TRACE v3 three-stage] start $(date '+%F %T')"
echo "[TRACE v3 three-stage] run_name=${RUN_NAME}"
echo "[TRACE v3 three-stage] train_count=${TRAIN_COUNT} val_count=${VAL_COUNT}"
echo "[TRACE v3 three-stage] Stage0 RUN_STAGE0=${RUN_STAGE0} epochs=${STAGE0_MAX_EPOCHS} batch=${STAGE0_BATCH_SIZE}"
echo "[TRACE v3 three-stage] Stage1 epochs=${STAGE1_MAX_EPOCHS} n_train=${STAGE1_N_TRAIN_SAMPLES} batch=${STAGE1_BATCH_SIZE} full_val=1.0 trainer=${STAGE1_TRAINER} patience=4 stage1_test=${RUN_STAGE1_TEST}"
echo "[TRACE v3 three-stage] Stage2 epochs=${STAGE2_MAX_EPOCHS} n_train=${STAGE2_N_TRAIN_SAMPLES} group=${STAGE2_GROUP_SIZE} exp_batch=${STAGE2_EXP_BATCH_SIZE} full_val=1.0"
echo "[TRACE v3 three-stage] test_times=${TEST_TIMES}"

if [[ "${STAGE1_N_TRAIN_SAMPLES}" -lt "${TRAIN_COUNT}" ]]; then
  echo "[TRACE v3 three-stage] ERROR: Stage1 must see the full train pool. STAGE1_N_TRAIN_SAMPLES=${STAGE1_N_TRAIN_SAMPLES}, train_count=${TRAIN_COUNT}" >&2
  exit 2
fi
if [[ "${STAGE2_N_TRAIN_SAMPLES}" -ne 512 ]]; then
  echo "[TRACE v3 three-stage] ERROR: Stage2 is expected to mimic CoLaR RL budget with 512 samples per epoch. Got ${STAGE2_N_TRAIN_SAMPLES}" >&2
  exit 2
fi

find_best_ckpt() {
  local log_root="$1"
  local run_contains="$2"
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_find_checkpoint.py \
    --log_root "${log_root}" \
    --run_contains "${run_contains}" \
    --prefer best
}

if [[ "${RUN_STAGE0}" == "True" ]]; then
  STAGE0_SUFFIX="${RUN_NAME}_stage0_cot"
  echo "[TRACE v3 three-stage] Stage0 training CoT-SFT"
  CUDA_VISIBLE_DEVICES="${STAGE0_GPUS}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
    --model=cot_qwen3_instruct \
    --dataset=gsm8k_aug_nl \
    --devices="${STAGE0_DEVICES}" \
    --do_test \
    --test_times="${TEST_TIMES}" \
    --workspace_path=/home/dingxukai \
    --log_suffix="${STAGE0_SUFFIX}" \
    batch_size="${STAGE0_BATCH_SIZE}" \
    val_batch_size=1 \
    max_epochs="${STAGE0_MAX_EPOCHS}" \
    num_sanity_val_steps=0 \
    trainer.limit_val_batches=1.0 \
    max_new_tokens=256 \
    lr="${STAGE0_LR}" \
    trainer.default_root_dir="${ROOT}/run_roots/${STAGE0_SUFFIX}"
  STAGE0_CKPT="$(find_best_ckpt logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl "${STAGE0_SUFFIX}")"
else
  echo "[TRACE v3 three-stage] Stage0 using verified existing CoT-SFT checkpoint: ${STAGE0_CKPT}"
fi
printf '%s\n' "${STAGE0_CKPT}" > "${ARTIFACT_DIR}/stage0_cot_ckpt.txt"

STAGE1_SUFFIX="${RUN_NAME}_stage1_trace_path_sft"
echo "[TRACE v3 three-stage] Stage1 TRACE path-scaffold supervised latent training"
STAGE1_TEST_ARGS=()
if [[ "${RUN_STAGE1_TEST}" == "True" ]]; then
  STAGE1_TEST_ARGS+=(--do_test --test_times="${TEST_TIMES}")
fi
CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
  --model=trace_multipath_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --trainer="${STAGE1_TRAINER}" \
  --devices=0 \
  --load_ckpt_path="${STAGE0_CKPT}" \
  "${STAGE1_TEST_ARGS[@]}" \
  --workspace_path=/home/dingxukai \
  --log_suffix="${STAGE1_SUFFIX}" \
  batch_size="${STAGE1_BATCH_SIZE}" \
  val_batch_size=1 \
  max_epochs="${STAGE1_MAX_EPOCHS}" \
  num_sanity_val_steps=0 \
  trainer.limit_val_batches=1.0 \
  max_compression_factor=5 \
  compression_factor=5 \
  max_n_latent_forward="${MAX_L}" \
  min_n_latent_forward="${MIN_L}" \
  max_new_tokens=16 \
  do_rl=False \
  n_train_samples_per_epoch="${STAGE1_N_TRAIN_SAMPLES}" \
  enable_trace_path_pretraining=True \
  path_pretraining_weight="${PATH_PRETRAINING_WEIGHT}" \
  path_use_policy_mean=True \
  path_anchor_jitter="${PATH_ANCHOR_JITTER}" \
  path_multiview_bootstrap_mix="${PATH_MULTIVIEW_BOOTSTRAP_MIX}" \
  lr="${STAGE1_LR}" \
  trainer.default_root_dir="${ROOT}/run_roots/${STAGE1_SUFFIX}"

STAGE1_CKPT="$(find_best_ckpt logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl "${STAGE1_SUFFIX}")"
echo "[TRACE v3 three-stage] Stage1 best checkpoint: ${STAGE1_CKPT}"
printf '%s\n' "${STAGE1_CKPT}" > "${ARTIFACT_DIR}/stage1_trace_path_sft_ckpt.txt"

STAGE2_SUFFIX="${RUN_NAME}_stage2_trace_multipath_rl"
echo "[TRACE v3 three-stage] Stage2 TRACE multi-path mode-aware RL"
CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
  --model=trace_multipath_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  --load_ckpt_path="${STAGE1_CKPT}" \
  --test_times="${TEST_TIMES}" \
  --workspace_path=/home/dingxukai \
  --log_suffix="${STAGE2_SUFFIX}" \
  batch_size="${STAGE2_BATCH_SIZE}" \
  val_batch_size=1 \
  max_epochs="${STAGE2_MAX_EPOCHS}" \
  num_sanity_val_steps=0 \
  trainer.limit_val_batches=1.0 \
  gradient_clip_val=0 \
  max_n_latent_forward="${MAX_L}" \
  min_n_latent_forward="${MIN_L}" \
  latent_temperature=1.0 \
  eol_temperature=1.0 \
  max_new_tokens=16 \
  do_rl=True \
  group_size="${STAGE2_GROUP_SIZE}" \
  exp_batch_size="${STAGE2_EXP_BATCH_SIZE}" \
  n_train_samples_per_epoch="${STAGE2_N_TRAIN_SAMPLES}" \
  enable_trace_path_pretraining=True \
  path_use_policy_mean=True \
  path_anchor_jitter="${PATH_ANCHOR_JITTER}" \
  path_multiview_bootstrap_mix="${PATH_MULTIVIEW_BOOTSTRAP_MIX}" \
  enable_trace_multipath_reward=True \
  filter_informative_train_indices=True \
  filter_candidate_factor=1.0 \
  filter_candidate_count=0 \
  filter_score_geometry=True \
  filter_max_batches=0 \
  resample_informative_groups=True \
  resample_max_attempts=3 \
  trace_reward_weight=0.35 \
  mode_diversity_weight=0.45 \
  mode_balance_weight=0.20 \
  neg_repulsion_weight=0.8 \
  hard_repulsion_weight=1.4 \
  hard_pos_repulsion_weight=0.25 \
  lr="${STAGE2_LR}" \
  trainer.default_root_dir="${ROOT}/run_roots/${STAGE2_SUFFIX}"

STAGE2_CKPT="$(find_best_ckpt logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl "${STAGE2_SUFFIX}")"
STAGE2_RUN_DIR="$(dirname "$(dirname "${STAGE2_CKPT}")")"
echo "[TRACE v3 three-stage] Stage2 monitored best checkpoint: ${STAGE2_CKPT}"
printf '%s\n' "${STAGE2_CKPT}" > "${ARTIFACT_DIR}/stage2_trace_best_ckpt.txt"
printf '%s\n' "${STAGE2_RUN_DIR}" > "${ARTIFACT_DIR}/stage2_trace_run_dir.txt"
cp --reflink=auto -f "${STAGE2_CKPT}" "${ARTIFACT_DIR}/best.ckpt" || cp -f "${STAGE2_CKPT}" "${ARTIFACT_DIR}/best.ckpt"

echo "[TRACE v3 three-stage] TRACE Stage2 best ckpt GSM8K/OOD benchmark"
CKPT="${STAGE2_CKPT}" GPU="${STAGE2_GPU}" TEST_TIMES="${TEST_TIMES}" \
  bash run_trace_multipath_ood_eval_20260704.sh "${STAGE2_SUFFIX}"

/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_collect_results.py \
  --run_contains "${RUN_NAME}" \
  --out_dir "${ARTIFACT_DIR}" \
  --out_name trace_three_stage_results

if [[ "${RUN_COLAR_BASELINE}" == "True" ]]; then
  echo "[TRACE v3 three-stage] CoLaR origin baseline GSM8K/OOD benchmark"
  OUT_DIR="${ARTIFACT_DIR}/colar_baseline" GPU="${STAGE2_GPU}" TEST_TIMES="${TEST_TIMES}" MAX_L="${MAX_L}" \
    bash run_colar_origin_c5_ood_eval_20260704.sh
fi

echo "[TRACE v3 three-stage] fixed-question global-PCA 3D paths and heatmap"
CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_visualize.py \
  --ckpt "${STAGE2_CKPT}" \
  --indices 0 1 2 \
  --group_size "${STAGE2_GROUP_SIZE}" \
  --max_l "${MAX_L}" \
  --min_l "${MIN_L}" \
  --device cuda:0 \
  --trajectory_space residual \
  --pca_scope global \
  --out_dir "${ARTIFACT_DIR}/visualizations/fixed_q012_global"

echo "[TRACE v3 three-stage] auto-selected global-PCA 3D paths and heatmap"
CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_visualize.py \
  --ckpt "${STAGE2_CKPT}" \
  --auto_select \
  --candidate_count "${VIS_CANDIDATE_COUNT}" \
  --num_questions 3 \
  --seed 0 \
  --group_size "${STAGE2_GROUP_SIZE}" \
  --max_l "${MAX_L}" \
  --min_l "${MIN_L}" \
  --device cuda:0 \
  --trajectory_space residual \
  --pca_scope global \
  --out_dir "${ARTIFACT_DIR}/visualizations/auto${VIS_CANDIDATE_COUNT}_global"

echo "[TRACE v3 three-stage] 200-question geometry summary"
CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_geometry_summary.py \
  --ckpt "${STAGE2_CKPT}" \
  --num_questions 200 \
  --group_size "${STAGE2_GROUP_SIZE}" \
  --max_l "${MAX_L}" \
  --min_l "${MIN_L}" \
  --device cuda:0 \
  --out_dir "${ARTIFACT_DIR}/geometry_200"

echo "[TRACE v3 three-stage] done $(date '+%F %T')"
echo "[TRACE v3 three-stage] artifacts=${ARTIFACT_DIR}"
