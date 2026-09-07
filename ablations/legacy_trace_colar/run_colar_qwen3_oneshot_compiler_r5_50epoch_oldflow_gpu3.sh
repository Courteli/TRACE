#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dingxukai/colar origin"
RUN_OUTPUTS="${ROOT}/run_outputs"
COT_CKPT="${ROOT}/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260426-134652_725405_origin_cot_qwen3_instruct_rawgsm_lr3e-5_3epoch_gpus1_2/checkpoints/epoch1__step6726__monitor0.857.ckpt"
OLDFLOW_DATA="/home/dingxukai/RoT/data/GSM8k-Aug-NL"
SESSION="origin_colar_qwen3_oneshot_compiler_r5_cotinit_50epoch_exactoldflow_gpu3"

mkdir -p "${RUN_OUTPUTS}" "${ROOT}/run_roots"

if [[ ! -f "${COT_CKPT}" ]]; then
  echo "CoT-SFT checkpoint not found: ${COT_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${OLDFLOW_DATA}" ]]; then
  echo "Oldflow data directory not found: ${OLDFLOW_DATA}" >&2
  exit 1
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && conda activate ROT && CUDA_VISIBLE_DEVICES=3 python run.py --model=colar_qwen3_oneshot_compiler --dataset=gsm8k_aug_nl --devices=0 --load_ckpt_path='${COT_CKPT}' --do_test --test_times=5 --workspace_path=/home/dingxukai --log_suffix=origin_colar_qwen3_oneshot_compiler_r5_cotinit_exactoldflow_lr3e-5_50epoch_gpu3 batch_size=1 val_batch_size=1 max_epochs=50 num_sanity_val_steps=0 max_compression_factor=5 compression_factor=5 max_new_tokens=16 lr=3e-5 data_module.dataset_dir='${OLDFLOW_DATA}' trainer.default_root_dir='${ROOT}/run_roots/origin_colar_qwen3_oneshot_compiler_r5_cotinit_50epoch_exactoldflow_gpu3' 2>&1 | tee -a '${RUN_OUTPUTS}/origin_colar_qwen3_oneshot_compiler_r5_cotinit_50epoch_exactoldflow_gpu3.log'"

echo "Started ${SESSION}"
