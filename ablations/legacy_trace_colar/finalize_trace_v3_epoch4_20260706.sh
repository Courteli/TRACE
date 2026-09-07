#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

RUN_NAME="${RUN_NAME:-trace_v3_three_stage_full_20260705_stage1solid}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/${RUN_NAME}}"
STAGE2_SUFFIX="${RUN_NAME}_stage2_trace_multipath_rl"
STAGE2_RUN_DIR="${STAGE2_RUN_DIR:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260706-032913_586957_trace_v3_three_stage_full_20260705_stage1solid_stage2_trace_multipath_rl}"
STAGE2_CKPT="${STAGE2_CKPT:-${STAGE2_RUN_DIR}/checkpoints/epoch4__step2560__monitor0.347.ckpt}"
STAGE2_GPU="${STAGE2_GPU:-5}"
STAGE2_GROUP_SIZE="${STAGE2_GROUP_SIZE:-8}"
MAX_L="${MAX_L:-40}"
MIN_L="${MIN_L:-0}"
TEST_TIMES="${TEST_TIMES:-1}"
VIS_CANDIDATE_COUNT="${VIS_CANDIDATE_COUNT:-200}"

mkdir -p "${ARTIFACT_DIR}"

echo "[TRACE v3 epoch4 finalize] start $(date '+%F %T')"
echo "[TRACE v3 epoch4 finalize] ckpt=${STAGE2_CKPT}"
echo "[TRACE v3 epoch4 finalize] run_dir=${STAGE2_RUN_DIR}"

printf '%s\n' "${STAGE2_CKPT}" > "${ARTIFACT_DIR}/stage2_trace_best_ckpt.txt"
printf '%s\n' "${STAGE2_RUN_DIR}" > "${ARTIFACT_DIR}/stage2_trace_run_dir.txt"
cp --reflink=auto -f "${STAGE2_CKPT}" "${ARTIFACT_DIR}/best.ckpt" || cp -f "${STAGE2_CKPT}" "${ARTIFACT_DIR}/best.ckpt"

echo "[TRACE v3 epoch4 finalize] TRACE epoch4 GSM8K/OOD benchmark"
CKPT="${STAGE2_CKPT}" GPU="${STAGE2_GPU}" TEST_TIMES="${TEST_TIMES}" \
  bash run_trace_multipath_ood_eval_20260704.sh "${STAGE2_SUFFIX}"

/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_collect_results.py \
  --run_contains "${RUN_NAME}" \
  --out_dir "${ARTIFACT_DIR}" \
  --out_name trace_three_stage_results

echo "[TRACE v3 epoch4 finalize] fixed-question global-PCA 3D paths and heatmap"
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

echo "[TRACE v3 epoch4 finalize] auto-selected global-PCA 3D paths and heatmap"
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

echo "[TRACE v3 epoch4 finalize] 200-question geometry summary"
CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_geometry_summary.py \
  --ckpt "${STAGE2_CKPT}" \
  --num_questions 200 \
  --group_size "${STAGE2_GROUP_SIZE}" \
  --max_l "${MAX_L}" \
  --min_l "${MIN_L}" \
  --device cuda:0 \
  --out_dir "${ARTIFACT_DIR}/geometry_200"

/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_full_experiment_audit.py \
  --run-name "${RUN_NAME}" \
  --write-json "${ARTIFACT_DIR}/experiment_audit_latest.json" || true

echo "[TRACE v3 epoch4 finalize] done $(date '+%F %T')"
echo "[TRACE v3 epoch4 finalize] artifacts=${ARTIFACT_DIR}"
