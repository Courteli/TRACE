#!/usr/bin/env bash
set -euo pipefail

cd "/disk1/dingxukai/trace_colar"

# Queue the remaining TRACE trajectory ablations after the main full run.
# This keeps GPU memory use single-process while preserving a reproducible run order.

GPU="${GPU:-4}"
TEST_TIMES="${TEST_TIMES:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-trace_trajectory_full_k8_gpu4}"
VARIANTS="${VARIANTS:-no_state no_trans answer_only k4 k16 k40}"
RUN_VIS_FULL="${RUN_VIS_FULL:-1}"
RUN_OOD_FULL="${RUN_OOD_FULL:-1}"
RUN_ORIGIN_BASELINE="${RUN_ORIGIN_BASELINE:-1}"
RUN_COLLECT_AFTER_EACH="${RUN_COLLECT_AFTER_EACH:-1}"
ORIGIN_CKPT="${ORIGIN_CKPT:-logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260525-172642_884973_origin_colar_qwen3_c5_cotinit_rawgsm_lr3e-5_50epoch_gpu5_20260525/checkpoints/epoch15__step107616__monitor0.246.ckpt}"

collect_results() {
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_collect_results.py \
    --run_contains trace_trajectory_ \
    --out_name trace_trajectory_summary
}

collect_origin_results() {
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_collect_results.py \
    --log_root logs/colar_qwen3_instruct/trace_baselines \
    --run_contains trace_baseline_colar_origin_c5_ \
    --out_name trace_baseline_colar_origin_c5_summary
}

find_full_ckpt() {
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_find_checkpoint.py \
    --log_root logs/trace_trajectory_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl \
    --run_contains "trace_trajectory_full_qwen3_c5_k8_gpu${GPU}" \
    --prefer best
}

if tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; then
  echo "[queue] waiting for tmux session ${WAIT_FOR_SESSION} to finish"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 120
  done
fi

if [[ "${RUN_VIS_FULL}" == "1" ]]; then
  echo "[queue] rendering 3D trajectory visualization for full K=8"
  FULL_CKPT="$(find_full_ckpt)"
  CUDA_VISIBLE_DEVICES="${GPU}" /home/dingxukai/miniconda3/envs/ROT/bin/python \
    tools/trace_trajectory_visualize.py \
    --ckpt "${FULL_CKPT}" \
    --indices 0 1 2 \
    --trace_steps 8 \
    --device cuda:0 \
    --origin_ckpt "${ORIGIN_CKPT}" \
    --origin_speed 5 \
    --out_dir "run_outputs/trace/trajectory_visualizations/full_k8_gpu${GPU}_q012"
fi

if [[ "${RUN_OOD_FULL}" == "1" ]]; then
  echo "[queue] running OOD eval for full K=8"
  GPU="${GPU}" TEST_TIMES="${TEST_TIMES}" \
    bash run_trace_trajectory_ood_eval_20260704.sh "trace_trajectory_full_qwen3_c5_k8_gpu${GPU}"
  collect_results
fi

echo "[queue] starting remaining TRACE trajectory ablations on GPU ${GPU}"
for variant in ${VARIANTS}; do
  echo "[queue] running ${variant}"
  GPU="${GPU}" TEST_TIMES="${TEST_TIMES}" MAX_EPOCHS="${MAX_EPOCHS}" \
    bash run_trace_trajectory_ablation_20260704.sh "${variant}"
  if [[ "${RUN_COLLECT_AFTER_EACH}" == "1" ]]; then
    collect_results
  fi
done

if [[ "${RUN_ORIGIN_BASELINE}" == "1" ]]; then
  echo "[queue] running CoLaR origin c=5 baseline OOD eval"
  GPU="${GPU}" TEST_TIMES="${TEST_TIMES}" ORIGIN_CKPT="${ORIGIN_CKPT}" \
    bash run_colar_origin_c5_ood_eval_20260704.sh
  collect_origin_results
fi

collect_results
echo "[queue] all requested TRACE trajectory ablations finished"
