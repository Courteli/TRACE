#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dingxukai/colar origin"
RUN_OUTPUTS="${ROOT}/run_outputs"
COT_CKPT="${ROOT}/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260426-134652_725405_origin_cot_qwen3_instruct_rawgsm_lr3e-5_3epoch_gpus1_2/checkpoints/epoch1__step6726__monitor0.857.ckpt"

mkdir -p "${RUN_OUTPUTS}" "${ROOT}/run_roots"

if [[ ! -f "${COT_CKPT}" ]]; then
  echo "CoT-SFT checkpoint not found: ${COT_CKPT}" >&2
  exit 1
fi

R2_SESSION="origin_colar_qwen3_instruct_r2_cotinit_50epoch_gpu2"
R5_SESSION="origin_colar_qwen3_instruct_r5_cotinit_50epoch_gpu3"

tmux kill-session -t "${R2_SESSION}" 2>/dev/null || true
tmux kill-session -t "${R5_SESSION}" 2>/dev/null || true

tmux new-session -d -s "${R2_SESSION}" \
  "cd '${ROOT}' && source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && conda activate ROT && CUDA_VISIBLE_DEVICES=2 python run.py --model=colar_qwen3_instruct --dataset=gsm8k_aug_nl --devices=0 --load_ckpt_path='${COT_CKPT}' --do_test --test_times=5 --workspace_path=/home/dingxukai --log_suffix=origin_colar_qwen3_instruct_r2_cotinit_rawgsm_lr3e-5_50epoch_gpu2 batch_size=1 val_batch_size=1 max_epochs=50 num_sanity_val_steps=0 max_compression_factor=2 compression_factor=2 max_new_tokens=16 lr=3e-5 trainer.default_root_dir='${ROOT}/run_roots/origin_colar_qwen3_instruct_r2_cotinit_50epoch_gpu2' 2>&1 | tee -a '${RUN_OUTPUTS}/origin_colar_qwen3_instruct_r2_cotinit_50epoch_gpu2.log'"

tmux new-session -d -s "${R5_SESSION}" \
  "cd '${ROOT}' && source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && conda activate ROT && CUDA_VISIBLE_DEVICES=3 python run.py --model=colar_qwen3_instruct --dataset=gsm8k_aug_nl --devices=0 --load_ckpt_path='${COT_CKPT}' --do_test --test_times=5 --workspace_path=/home/dingxukai --log_suffix=origin_colar_qwen3_instruct_r5_cotinit_rawgsm_lr3e-5_50epoch_gpu3 batch_size=1 val_batch_size=1 max_epochs=50 num_sanity_val_steps=0 max_compression_factor=5 compression_factor=5 max_new_tokens=16 lr=3e-5 trainer.default_root_dir='${ROOT}/run_roots/origin_colar_qwen3_instruct_r5_cotinit_50epoch_gpu3' 2>&1 | tee -a '${RUN_OUTPUTS}/origin_colar_qwen3_instruct_r5_cotinit_50epoch_gpu3.log'"

echo "Started ${R2_SESSION} and ${R5_SESSION}"
