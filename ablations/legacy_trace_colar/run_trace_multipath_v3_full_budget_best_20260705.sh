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

GPU="${GPU:-4}"
MAX_L="${MAX_L:-40}"
MIN_L="${MIN_L:-0}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-${TRAIN_COUNT}}"
TEST_TIMES="${TEST_TIMES:-1}"
VIS_CANDIDATE_COUNT="${VIS_CANDIDATE_COUNT:-200}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-trace_multipath_v3_fullbudget_L${MAX_L}_g${GROUP_SIZE}_e${MAX_EPOCHS}_n${N_TRAIN_SAMPLES}_fullval_origininit_${STAMP}_gpu${GPU}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/${RUN_NAME}}"
PIPELINE_LOG="${PIPELINE_LOG:-${ARTIFACT_DIR}/pipeline.log}"
ROOT_DIR="${ROOT_DIR:-${ROOT}/run_roots/${RUN_NAME}}"
INIT_CKPT="${INIT_CKPT:-${ROOT}/logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260525-172642_884973_origin_colar_qwen3_c5_cotinit_rawgsm_lr3e-5_50epoch_gpu5_20260525/checkpoints/epoch15__step107616__monitor0.246.ckpt}"

mkdir -p "${ARTIFACT_DIR}" "${ROOT_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "[TRACE v3 full-budget] start $(date '+%F %T')"
echo "[TRACE v3 full-budget] run_name=${RUN_NAME}"
echo "[TRACE v3 full-budget] gpu=${GPU}"
echo "[TRACE v3 full-budget] init_ckpt=${INIT_CKPT}"
echo "[TRACE v3 full-budget] train_file=${TRAIN_FILE}"
echo "[TRACE v3 full-budget] train_count=${TRAIN_COUNT}"
echo "[TRACE v3 full-budget] val_file=${VAL_FILE}"
echo "[TRACE v3 full-budget] val_count=${VAL_COUNT}"
echo "[TRACE v3 full-budget] max_epochs=${MAX_EPOCHS}"
echo "[TRACE v3 full-budget] n_train_samples_per_epoch=${N_TRAIN_SAMPLES}"
echo "[TRACE v3 full-budget] validation=full every epoch (trainer.limit_val_batches=1.0, check_val_every_n_epoch=1)"
echo "[TRACE v3 full-budget] test_times=${TEST_TIMES}"

if [[ "${N_TRAIN_SAMPLES}" -lt "${TRAIN_COUNT}" ]]; then
  echo "[TRACE v3 full-budget] ERROR: N_TRAIN_SAMPLES=${N_TRAIN_SAMPLES} is smaller than train_count=${TRAIN_COUNT}" >&2
  exit 2
fi

MAX_EPOCHS="${MAX_EPOCHS}" \
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES}" \
TEST_TIMES="${TEST_TIMES}" \
DO_TEST=False \
LIMIT_VAL_BATCHES=1.0 \
FILTER_INFORMATIVE=True \
FILTER_CANDIDATE_FACTOR=1.0 \
FILTER_CANDIDATE_COUNT=0 \
FILTER_MAX_BATCHES=0 \
FILTER_SCORE_GEOMETRY=True \
RESAMPLE_INFORMATIVE_GROUPS=True \
RESAMPLE_MAX_ATTEMPTS=3 \
GPU="${GPU}" \
MAX_L="${MAX_L}" \
MIN_L="${MIN_L}" \
GROUP_SIZE="${GROUP_SIZE}" \
LOG_SUFFIX="${RUN_NAME}" \
ROOT_DIR="${ROOT_DIR}" \
INIT_CKPT="${INIT_CKPT}" \
bash run_trace_multipath_rl_qwen3_c5_gpu4.sh

BEST_CKPT="$(
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_find_checkpoint.py \
    --log_root logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl \
    --run_contains "${RUN_NAME}" \
    --prefer best
)"
RUN_DIR="$(dirname "$(dirname "${BEST_CKPT}")")"

echo "[TRACE v3 full-budget] monitored_best=${BEST_CKPT}"
echo "[TRACE v3 full-budget] run_dir=${RUN_DIR}"
printf '%s\n' "${BEST_CKPT}" > "${ARTIFACT_DIR}/best_ckpt.txt"
printf '%s\n' "${RUN_DIR}" > "${ARTIFACT_DIR}/run_dir.txt"
cp --reflink=auto -f "${BEST_CKPT}" "${ARTIFACT_DIR}/best.ckpt" || cp -f "${BEST_CKPT}" "${ARTIFACT_DIR}/best.ckpt"

echo "[TRACE v3 full-budget] OOD/main table evaluation with the same best checkpoint"
CKPT="${BEST_CKPT}" \
GPU="${GPU}" \
TEST_TIMES="${TEST_TIMES}" \
bash run_trace_multipath_ood_eval_20260704.sh "${RUN_NAME}"

/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_collect_results.py \
  --run_contains "${RUN_NAME}" \
  --out_dir "${ARTIFACT_DIR}" \
  --out_name main_table_best_ckpt

echo "[TRACE v3 full-budget] fixed-question global-PCA 3D paths and heatmap"
CUDA_VISIBLE_DEVICES="${GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_visualize.py \
  --ckpt "${BEST_CKPT}" \
  --indices 0 1 2 \
  --group_size "${GROUP_SIZE}" \
  --max_l "${MAX_L}" \
  --min_l "${MIN_L}" \
  --device cuda:0 \
  --trajectory_space residual \
  --pca_scope global \
  --out_dir "${ARTIFACT_DIR}/visualizations/fixed_q012_global"

echo "[TRACE v3 full-budget] auto-selected global-PCA 3D paths and heatmap from ${VIS_CANDIDATE_COUNT} candidates"
CUDA_VISIBLE_DEVICES="${GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_visualize.py \
  --ckpt "${BEST_CKPT}" \
  --auto_select \
  --candidate_count "${VIS_CANDIDATE_COUNT}" \
  --num_questions 3 \
  --seed 0 \
  --group_size "${GROUP_SIZE}" \
  --max_l "${MAX_L}" \
  --min_l "${MIN_L}" \
  --device cuda:0 \
  --trajectory_space residual \
  --pca_scope global \
  --out_dir "${ARTIFACT_DIR}/visualizations/auto${VIS_CANDIDATE_COUNT}_global"

echo "[TRACE v3 full-budget] 200-question geometry summary with the same best checkpoint"
CUDA_VISIBLE_DEVICES="${GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_geometry_summary.py \
  --ckpt "${BEST_CKPT}" \
  --num_questions 200 \
  --group_size "${GROUP_SIZE}" \
  --max_l "${MAX_L}" \
  --min_l "${MIN_L}" \
  --device cuda:0 \
  --out_dir "${ARTIFACT_DIR}/geometry_200"

echo "[TRACE v3 full-budget] done $(date '+%F %T')"
echo "[TRACE v3 full-budget] artifacts=${ARTIFACT_DIR}"
