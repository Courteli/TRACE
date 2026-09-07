#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-trace_v3_three_stage_full_${STAMP}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/${RUN_NAME}}"
PIPELINE_LOG="${PIPELINE_LOG:-${ARTIFACT_DIR}/pipeline_continue_from_stage1.log}"
mkdir -p "${ARTIFACT_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

STAGE1_SUFFIX="${STAGE1_SUFFIX:-${RUN_NAME}_stage1_trace_path_sft}"
STAGE1_CKPT="${STAGE1_CKPT:-}"

STAGE2_GPU="${STAGE2_GPU:-4}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-1}"
STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-10}"
STAGE2_N_TRAIN_SAMPLES="${STAGE2_N_TRAIN_SAMPLES:-512}"
STAGE2_GROUP_SIZE="${STAGE2_GROUP_SIZE:-8}"
STAGE2_EXP_BATCH_SIZE="${STAGE2_EXP_BATCH_SIZE:-8}"
STAGE2_LR="${STAGE2_LR:-1e-6}"
STAGE2_RESUME_CKPT="${STAGE2_RESUME_CKPT:-}"
STAGE2_INIT_CKPT="${STAGE2_INIT_CKPT:-}"
MAX_L="${MAX_L:-40}"
MIN_L="${MIN_L:-0}"
TEST_TIMES="${TEST_TIMES:-1}"
VIS_CANDIDATE_COUNT="${VIS_CANDIDATE_COUNT:-200}"
PATH_ANCHOR_JITTER="${PATH_ANCHOR_JITTER:-1}"
PATH_MULTIVIEW_BOOTSTRAP_MIX="${PATH_MULTIVIEW_BOOTSTRAP_MIX:-0.20}"
RUN_COLAR_BASELINE="${RUN_COLAR_BASELINE:-False}"

find_best_ckpt() {
  local log_root="$1"
  local run_contains="$2"
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_find_checkpoint.py \
    --log_root "${log_root}" \
    --run_contains "${run_contains}" \
    --prefer best
}

if [[ -z "${STAGE1_CKPT}" ]]; then
  STAGE1_CKPT="$(find_best_ckpt logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl "${STAGE1_SUFFIX}")"
fi

echo "[TRACE v3 continue] start $(date '+%F %T')"
echo "[TRACE v3 continue] run_name=${RUN_NAME}"
echo "[TRACE v3 continue] Stage1 best checkpoint: ${STAGE1_CKPT}"
if [[ -n "${STAGE2_RESUME_CKPT}" ]]; then
  echo "[TRACE v3 continue] Stage2 resume checkpoint: ${STAGE2_RESUME_CKPT}"
fi
if [[ -n "${STAGE2_INIT_CKPT}" ]]; then
  echo "[TRACE v3 continue] Stage2 init checkpoint: ${STAGE2_INIT_CKPT}"
fi
echo "[TRACE v3 continue] Stage2 epochs=${STAGE2_MAX_EPOCHS} n_train=${STAGE2_N_TRAIN_SAMPLES} group=${STAGE2_GROUP_SIZE} exp_batch=${STAGE2_EXP_BATCH_SIZE} full_val=1.0"
echo "[TRACE v3 continue] test_times=${TEST_TIMES}"
printf '%s\n' "${STAGE1_CKPT}" > "${ARTIFACT_DIR}/stage1_trace_path_sft_ckpt.txt"

if [[ "${STAGE2_N_TRAIN_SAMPLES}" -ne 512 ]]; then
  echo "[TRACE v3 continue] ERROR: Stage2 is expected to mimic CoLaR RL budget with 512 samples per epoch. Got ${STAGE2_N_TRAIN_SAMPLES}" >&2
  exit 2
fi

STAGE2_SUFFIX="${RUN_NAME}_stage2_trace_multipath_rl"
echo "[TRACE v3 continue] Stage2 TRACE multi-path mode-aware RL"
if [[ -n "${STAGE2_RESUME_CKPT}" ]]; then
  STAGE2_CKPT_ARGS=(--resume_ckpt_path="${STAGE2_RESUME_CKPT}")
elif [[ -n "${STAGE2_INIT_CKPT}" ]]; then
  STAGE2_CKPT_ARGS=(--load_ckpt_path="${STAGE2_INIT_CKPT}")
else
  STAGE2_CKPT_ARGS=(--load_ckpt_path="${STAGE1_CKPT}")
fi
CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
  --model=trace_multipath_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --devices=0 \
  "${STAGE2_CKPT_ARGS[@]}" \
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
echo "[TRACE v3 continue] Stage2 monitored best checkpoint: ${STAGE2_CKPT}"
printf '%s\n' "${STAGE2_CKPT}" > "${ARTIFACT_DIR}/stage2_trace_best_ckpt.txt"
printf '%s\n' "${STAGE2_RUN_DIR}" > "${ARTIFACT_DIR}/stage2_trace_run_dir.txt"
cp --reflink=auto -f "${STAGE2_CKPT}" "${ARTIFACT_DIR}/best.ckpt" || cp -f "${STAGE2_CKPT}" "${ARTIFACT_DIR}/best.ckpt"

echo "[TRACE v3 continue] TRACE Stage2 best ckpt GSM8K/OOD benchmark"
CKPT="${STAGE2_CKPT}" GPU="${STAGE2_GPU}" TEST_TIMES="${TEST_TIMES}" \
  bash run_trace_multipath_ood_eval_20260704.sh "${STAGE2_SUFFIX}"

/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_collect_results.py \
  --run_contains "${RUN_NAME}" \
  --out_dir "${ARTIFACT_DIR}" \
  --out_name trace_three_stage_results

if [[ "${RUN_COLAR_BASELINE}" == "True" ]]; then
  echo "[TRACE v3 continue] CoLaR origin baseline GSM8K/OOD benchmark"
  OUT_DIR="${ARTIFACT_DIR}/colar_baseline" GPU="${STAGE2_GPU}" TEST_TIMES="${TEST_TIMES}" MAX_L="${MAX_L}" \
    bash run_colar_origin_c5_ood_eval_20260704.sh
fi

echo "[TRACE v3 continue] fixed-question global-PCA 3D paths and heatmap"
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

echo "[TRACE v3 continue] auto-selected global-PCA 3D paths and heatmap"
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

echo "[TRACE v3 continue] 200-question geometry summary"
CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_geometry_summary.py \
  --ckpt "${STAGE2_CKPT}" \
  --num_questions 200 \
  --group_size "${STAGE2_GROUP_SIZE}" \
  --max_l "${MAX_L}" \
  --min_l "${MIN_L}" \
  --device cuda:0 \
  --out_dir "${ARTIFACT_DIR}/geometry_200"

echo "[TRACE v3 continue] done $(date '+%F %T')"
echo "[TRACE v3 continue] artifacts=${ARTIFACT_DIR}"
