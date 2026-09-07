#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
OUT_ROOT="${OUT_ROOT:-/home/dingxukai/run_outputs/trace_multipath/requested_20260705}"
mkdir -p "${OUT_ROOT}"

TRACE_V3_CKPT="${TRACE_V3_CKPT:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260705-043057_316626_trace_multipath_v3_fast_L40_g8_gpu4/checkpoints/last.ckpt}"
TRACE_VIS_CKPT="${TRACE_VIS_CKPT:-${TRACE_V3_CKPT}}"
ANSWER_ONLY_CKPT="${ANSWER_ONLY_CKPT:-${ROOT}/logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260704-112325_182870_trace_answer_only_rl_qwen3_c5_gpu5/checkpoints/last.ckpt}"
ORIGIN_CKPT="${ORIGIN_CKPT:-${ROOT}/logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260525-172642_884973_origin_colar_qwen3_c5_cotinit_rawgsm_lr3e-5_50epoch_gpu5_20260525/checkpoints/epoch15__step107616__monitor0.246.ckpt}"
ABLATION_INIT_CKPT="${ABLATION_INIT_CKPT:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260705-001103_910691_trace_multipath_full_qwen3_c5_L40_g8_gpu4/checkpoints/last.ckpt}"

TRACE_BENCH_GPU="${TRACE_BENCH_GPU:-0}"
ANSWER_BENCH_GPU="${ANSWER_BENCH_GPU:-1}"
ABLATION_GPU="${ABLATION_GPU:-2}"
VIS_GPU="${VIS_GPU:-0}"
SUMMARY_GPU="${SUMMARY_GPU:-0}"
TEST_TIMES="${TEST_TIMES:-1}"

start_trace_v3_ood() {
  local session="${1:-trace_v3_t1_ood_20260705}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && \
     TEST_TIMES=${TEST_TIMES} GPU=${TRACE_BENCH_GPU} CKPT='${TRACE_V3_CKPT}' \
       bash run_trace_multipath_ood_eval_20260704.sh trace_multipath_v3_fast_L40_g8_gpu4 \
       2>&1 | tee -a '${OUT_ROOT}/${session}.log'"
  echo "Started ${session}"
}

start_answer_only_ood() {
  local session="${1:-answer_only_t1_ood_20260705}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && \
     TEST_TIMES=${TEST_TIMES} MAX_N_LATENT_FORWARD=40 MIN_N_LATENT_FORWARD=0 \
       TRACE_OOD_DATASETS='gsm8k_aug_nl gsmhard svamp multiarith' \
       bash scripts/trace_test_ood.sh ${ANSWER_BENCH_GPU} '${ANSWER_ONLY_CKPT}' answer_only_rl_L40_t1_20260705 \
       2>&1 | tee -a '${OUT_ROOT}/${session}.log'"
  echo "Started ${session}"
}

start_ablation_queue() {
  local session="${1:-trace_v3_ablation_t1_20260705}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && \
     source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && conda activate ROT && \
     export TEST_TIMES=${TEST_TIMES} GPU=${ABLATION_GPU} INIT_CKPT='${ABLATION_INIT_CKPT}' && \
     export MAX_L=40 MIN_L=0 GROUP_SIZE=8 MAX_EPOCHS=1 N_TRAIN_SAMPLES=32 FILTER_CANDIDATE_COUNT=64 FILTER_MAX_BATCHES=64 && \
     export DO_TEST=False LIMIT_VAL_BATCHES=0 LR=1e-6 && \
     for VARIANT in no_hard no_mode no_filter; do \
       case \${VARIANT} in \
         no_hard) \
           export LOG_SUFFIX=trace_multipath_v3_ablate_no_hard_qwen3_c5_L40_g8_gpu${ABLATION_GPU}_20260705; \
           export HARD_REPULSION_WEIGHT=0 HARD_POS_REPULSION_WEIGHT=0 NEG_REPULSION_WEIGHT=0.8 MODE_DIVERSITY_WEIGHT=0.45 MODE_BALANCE_WEIGHT=0.20 POS_PAIR_REPULSION_WEIGHT=0.35 FILTER_INFORMATIVE=True RESAMPLE_INFORMATIVE_GROUPS=True ;; \
         no_mode) \
           export LOG_SUFFIX=trace_multipath_v3_ablate_no_mode_qwen3_c5_L40_g8_gpu${ABLATION_GPU}_20260705; \
           export HARD_REPULSION_WEIGHT=1.4 HARD_POS_REPULSION_WEIGHT=0.25 NEG_REPULSION_WEIGHT=0.8 MODE_DIVERSITY_WEIGHT=0 MODE_BALANCE_WEIGHT=0 POS_PAIR_REPULSION_WEIGHT=0 FILTER_INFORMATIVE=True RESAMPLE_INFORMATIVE_GROUPS=True ;; \
         no_filter) \
           export LOG_SUFFIX=trace_multipath_v3_ablate_no_filter_qwen3_c5_L40_g8_gpu${ABLATION_GPU}_20260705; \
           export HARD_REPULSION_WEIGHT=1.4 HARD_POS_REPULSION_WEIGHT=0.25 NEG_REPULSION_WEIGHT=0.8 MODE_DIVERSITY_WEIGHT=0.45 MODE_BALANCE_WEIGHT=0.20 POS_PAIR_REPULSION_WEIGHT=0.35 FILTER_INFORMATIVE=False RESAMPLE_INFORMATIVE_GROUPS=False FILTER_CANDIDATE_COUNT=0 FILTER_MAX_BATCHES=0 ;; \
       esac; \
       echo \"[ablation] training \${VARIANT} LOG_SUFFIX=\${LOG_SUFFIX}\"; \
       bash run_trace_multipath_rl_qwen3_c5_gpu4.sh; \
       CKPT=\$(/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_find_checkpoint.py --log_root logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl --run_contains \"\${LOG_SUFFIX}\" --prefer best); \
       echo \"[ablation] testing \${VARIANT} CKPT=\${CKPT}\"; \
       CUDA_VISIBLE_DEVICES=${ABLATION_GPU} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
         --model=trace_multipath_qwen3_instruct --dataset=gsm8k_aug_nl --devices=0 \
         --test_ckpt_path=\"\${CKPT}\" --test_times=${TEST_TIMES} --workspace_path=/home/dingxukai \
         batch_size=1 val_batch_size=1 num_sanity_val_steps=0 \
         dataset_name=gsm8k_aug_nl dataset_dir=/home/dingxukai/RoT/data/GSM8k-Aug-NL test_file=gsm8k_test_processed.jsonl; \
     done \
     2>&1 | tee -a '${OUT_ROOT}/${session}.log'"
  echo "Started ${session}"
}

start_global_pca() {
  local session="${1:-trace_v3_global_pca_20260705}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "cd '/home/dingxukai' && \
     source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && conda activate ROT && \
     CUDA_VISIBLE_DEVICES=${VIS_GPU} python trace_requested_experiments_20260705/trace_geometry_tools.py global-pca \
       --origin_ckpt '${ORIGIN_CKPT}' \
       --answer_ckpt '${ANSWER_ONLY_CKPT}' \
       --trace_ckpt '${TRACE_VIS_CKPT}' \
       --indices 808 101 3 \
       --group_size 8 --max_l 40 --device cuda:0 \
       --out_dir '${OUT_ROOT}/global_pca_same_question' \
       2>&1 | tee -a '${OUT_ROOT}/${session}.log'"
  echo "Started ${session}"
}

start_geometry_summary() {
  local session="${1:-trace_v3_geometry_200_20260705}"
  local method="${2:-TRACE_v3}"
  local ckpt="${3:-${TRACE_V3_CKPT}}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "cd '/home/dingxukai' && \
     source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && conda activate ROT && \
     CUDA_VISIBLE_DEVICES=${SUMMARY_GPU} python trace_requested_experiments_20260705/trace_geometry_tools.py summary \
       --method '${method}' --ckpt '${ckpt}' \
       --num_questions 200 --candidate_seed 0 \
       --group_size 8 --max_l 40 --device cuda:0 \
       --out_dir '${OUT_ROOT}/geometry_summary_${method}' \
       2>&1 | tee -a '${OUT_ROOT}/${session}.log'"
  echo "Started ${session}"
}

case "${1:-all}" in
  trace_ood) start_trace_v3_ood ;;
  answer_ood) start_answer_only_ood ;;
  ablation) start_ablation_queue ;;
  global_pca) start_global_pca ;;
  summary) start_geometry_summary "${2:-trace_v3_geometry_200_20260705}" "${3:-TRACE_v3}" "${4:-${TRACE_V3_CKPT}}" ;;
  all)
    start_trace_v3_ood
    start_answer_only_ood
    start_ablation_queue
    start_global_pca
    start_geometry_summary
    ;;
  *)
    echo "Usage: $0 {all|trace_ood|answer_ood|ablation|global_pca|summary}" >&2
    exit 2
    ;;
esac
