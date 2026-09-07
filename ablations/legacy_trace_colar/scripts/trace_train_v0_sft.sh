#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT="/disk1/dingxukai/trace_colar"
RUN_OUTPUTS="${ROOT}/run_outputs/trace"
SESSION="trace_v0_sft_qwen3_c5_gpu${GPU}"
COT_CKPT="${ROOT}/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260426-134652_725405_origin_cot_qwen3_instruct_rawgsm_lr3e-5_3epoch_gpus1_2/checkpoints/epoch1__step6726__monitor0.857.ckpt"

mkdir -p "${RUN_OUTPUTS}" "${ROOT}/run_roots/${SESSION}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && \
   source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && \
   conda activate ROT && \
   CUDA_VISIBLE_DEVICES=${GPU} python run.py \
     --model=trace_colar_qwen3_instruct \
     --dataset=gsm8k_aug_nl \
     --devices=0 \
     --load_ckpt_path='${COT_CKPT}' \
     --do_test \
     --test_times=5 \
     --workspace_path=/home/dingxukai \
     --log_suffix=${SESSION} \
     batch_size=1 \
     val_batch_size=1 \
     max_epochs=50 \
     num_sanity_val_steps=0 \
     max_compression_factor=5 \
     compression_factor=5 \
     max_new_tokens=16 \
     lr=3e-5 \
     trainer.default_root_dir='${ROOT}/run_roots/${SESSION}' \
     2>&1 | tee -a '${RUN_OUTPUTS}/${SESSION}.log'"

echo "Started ${SESSION}"
