#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
SFT_CKPT="${2:?Usage: bash scripts/trace_train_answer_only_rl.sh <gpu> <sft_ckpt>}"
ROOT="/disk1/dingxukai/trace_colar"
RUN_OUTPUTS="${ROOT}/run_outputs/trace"
SESSION="${SESSION_NAME:-trace_answer_only_rl_qwen3_c5_gpu${GPU}}"
GROUP_SIZE="${GROUP_SIZE:-8}"
EXP_BATCH_SIZE="${EXP_BATCH_SIZE:-8}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-512}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
MAX_N_LATENT_FORWARD="${MAX_N_LATENT_FORWARD:-64}"
LR="${LR:-3e-5}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-1.0}"
DO_TEST="${DO_TEST:-True}"
TEST_TIMES="${TEST_TIMES:-5}"
TRACE_FILTER_MIXED="${TRACE_FILTER_MIXED:-False}"
TRACE_FILTER_CANDIDATE_FACTOR="${TRACE_FILTER_CANDIDATE_FACTOR:-1.0}"
TRACE_FILTER_CANDIDATE_COUNT="${TRACE_FILTER_CANDIDATE_COUNT:-0}"
TRACE_FILTER_BATCH_SIZE="${TRACE_FILTER_BATCH_SIZE:-1}"
TRACE_FILTER_MIXED_FILL_FRACTION="${TRACE_FILTER_MIXED_FILL_FRACTION:-0.5}"
TRACE_RESAMPLE_MIXED="${TRACE_RESAMPLE_MIXED:-False}"
TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS="${TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS:-4}"
TRACE_RESAMPLE_MIXED_TARGET_FRAC="${TRACE_RESAMPLE_MIXED_TARGET_FRAC:-1.0}"
DO_TEST_FLAG=""
if [[ "${DO_TEST}" == "True" || "${DO_TEST}" == "true" || "${DO_TEST}" == "1" ]]; then
  DO_TEST_FLAG="--do_test"
fi

mkdir -p "${RUN_OUTPUTS}" "${ROOT}/run_roots/${SESSION}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && \
   source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && \
   conda activate ROT && \
	   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${GPU} python run.py \
     --model=trace_colar_qwen3_instruct \
     --dataset=gsm8k_aug_nl \
     --devices=0 \
     --load_ckpt_path='${SFT_CKPT}' \
     ${DO_TEST_FLAG} \
     --test_times=${TEST_TIMES} \
     --workspace_path=/home/dingxukai \
     --log_suffix=${SESSION} \
     batch_size=1 \
     val_batch_size=1 \
     max_epochs=${MAX_EPOCHS} \
     num_sanity_val_steps=0 \
	     gradient_clip_val=0 \
	     max_compression_factor=5 \
	     compression_factor=5 \
	     max_n_latent_forward=${MAX_N_LATENT_FORWARD} \
	     latent_temperature=${LATENT_TEMPERATURE} \
	     max_new_tokens=16 \
	     do_rl=True \
	     enable_trace_reward=False \
	     group_size=${GROUP_SIZE} \
	     exp_batch_size=${EXP_BATCH_SIZE} \
	     n_train_samples_per_epoch=${N_TRAIN_SAMPLES} \
	     trace_filter_mixed_train_indices=${TRACE_FILTER_MIXED} \
	     trace_filter_candidate_factor=${TRACE_FILTER_CANDIDATE_FACTOR} \
	     trace_filter_candidate_count=${TRACE_FILTER_CANDIDATE_COUNT} \
	     trace_filter_batch_size=${TRACE_FILTER_BATCH_SIZE} \
	     trace_filter_mixed_fill_fraction=${TRACE_FILTER_MIXED_FILL_FRACTION} \
	     trace_resample_mixed_rollout=${TRACE_RESAMPLE_MIXED} \
	     trace_resample_mixed_max_attempts=${TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS} \
	     trace_resample_mixed_target_frac=${TRACE_RESAMPLE_MIXED_TARGET_FRAC} \
	     lr=${LR} \
     trainer.default_root_dir='${ROOT}/run_roots/${SESSION}' \
     2>&1 | tee -a '${RUN_OUTPUTS}/${SESSION}.log'"

echo "Started ${SESSION}"
