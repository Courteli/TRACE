#!/usr/bin/env bash
set -euo pipefail

cd "/disk1/dingxukai/trace_colar"

GPU="${GPU:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TEST_TIMES="${TEST_TIMES:-5}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-128}"
FILTER_CANDIDATE_COUNT="${FILTER_CANDIDATE_COUNT:-128}"
FILTER_MAX_BATCHES="${FILTER_MAX_BATCHES:-128}"
EPOCHS_L8="${EPOCHS_L8:-1}"
EPOCHS_L16="${EPOCHS_L16:-1}"
EPOCHS_L40="${EPOCHS_L40:-3}"
RUN_OOD="${RUN_OOD:-True}"
RUN_VIS="${RUN_VIS:-True}"
RUN_ABLATIONS="${RUN_ABLATIONS:-True}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-1}"

export GPU GROUP_SIZE TEST_TIMES N_TRAIN_SAMPLES FILTER_CANDIDATE_COUNT FILTER_MAX_BATCHES
export EPOCHS_L8 EPOCHS_L16 EPOCHS_L40 MIN_L="${MIN_L:-0}"

echo "[TRACE MultiPath suite] main curriculum"
bash run_trace_multipath_curriculum_20260704.sh

CKPT_L40="$(/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_find_checkpoint.py \
  --log_root logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl \
  --run_contains "trace_multipath_full_qwen3_c5_L40_g${GROUP_SIZE}_gpu${GPU}" \
  --prefer best)"
echo "[TRACE MultiPath suite] final checkpoint: ${CKPT_L40}"

if [[ "${RUN_OOD}" == "True" ]]; then
  echo "[TRACE MultiPath suite] OOD evaluation"
  CKPT="${CKPT_L40}" bash run_trace_multipath_ood_eval_20260704.sh "trace_multipath_full_qwen3_c5_L40_g${GROUP_SIZE}_gpu${GPU}"
fi

if [[ "${RUN_VIS}" == "True" ]]; then
  echo "[TRACE MultiPath suite] SemCoT-style 3D visualization"
  CUDA_VISIBLE_DEVICES="${GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_visualize.py \
    --ckpt "${CKPT_L40}" \
    --auto_select \
    --candidate_count 24 \
    --num_questions 3 \
    --group_size "${GROUP_SIZE}" \
    --max_l 40 \
    --min_l "${MIN_L}" \
    --device cuda:0 \
    --out_dir "run_outputs/trace_multipath/visualizations/full_L40_g${GROUP_SIZE}_gpu${GPU}"
fi

if [[ "${RUN_ABLATIONS}" == "True" ]]; then
  echo "[TRACE MultiPath suite] ablations"
  MAX_EPOCHS="${ABLATION_EPOCHS}" bash run_trace_multipath_ablation_20260704.sh answer_only
  MAX_EPOCHS="${ABLATION_EPOCHS}" bash run_trace_multipath_ablation_20260704.sh no_mode
  MAX_EPOCHS="${ABLATION_EPOCHS}" bash run_trace_multipath_ablation_20260704.sh no_hard
  MAX_EPOCHS="${ABLATION_EPOCHS}" bash run_trace_multipath_ablation_20260704.sh no_step
fi

/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_multipath_collect_results.py \
  --out_name trace_multipath_full_suite_summary

echo "[TRACE MultiPath suite] done"
