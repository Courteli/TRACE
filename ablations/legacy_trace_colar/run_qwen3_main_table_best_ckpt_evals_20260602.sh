#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dingxukai/colar origin"
OUT_DIR="${ROOT}/run_outputs/qwen3_main_table_best_ckpt_eval_20260602"
mkdir -p "${OUT_DIR}"

launch_eval() {
  local method="$1"
  local gpu="$2"
  local ckpt="$3"
  local session="origin_${method}_qwen3_best_ckpt_eval_gpu${gpu}"

  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "bash '${ROOT}/run_qwen3_main_table_best_ckpt_eval_worker_20260602.sh' '${method}' '${gpu}' '${ckpt}' 2>&1 | tee -a '${OUT_DIR}/${method}_worker.log'"
  echo "Started ${session}"
}

launch_eval icot 1 "${ROOT}/logs/icot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260601-012437_672050_origin_icot_qwen3_instruct_cotinit_lr3e-5_50epoch_gpu1/checkpoints/epoch2__step20178__monitor0.380.ckpt"
launch_eval coconut 2 "${ROOT}/logs/coconut_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260601-012437_283276_origin_coconut_qwen3_instruct_cotinit_lr3e-5_50epoch_gpu2/checkpoints/epoch3__step26904__monitor0.356.ckpt"
launch_eval distill 3 "${ROOT}/logs/distill_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260601-012437_585574_origin_distill_qwen3_instruct_cotinit_lr3e-5_50epoch_gpu3/checkpoints/epoch3__step26904__monitor0.129.ckpt"

echo "Logs: ${OUT_DIR}"
