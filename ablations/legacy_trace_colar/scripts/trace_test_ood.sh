#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
CKPT="${2:?Usage: bash scripts/trace_test_ood.sh <gpu> <ckpt> [tag]}"
TAG="${3:-$(basename "$(dirname "$(dirname "${CKPT}")")")}"
ROOT="/disk1/dingxukai/trace_colar"
RUN_OUTPUTS="${ROOT}/run_outputs/trace"
SESSION="trace_test_ood_${TAG}_gpu${GPU}"
TEST_TIMES="${TEST_TIMES:-5}"
TRACE_OOD_MAX_SAMPLES="${TRACE_OOD_MAX_SAMPLES:-0}"
TRACE_OOD_DATASETS="${TRACE_OOD_DATASETS:-gsm8k_aug_nl gsmhard svamp multiarith}"
MAX_N_LATENT_FORWARD="${MAX_N_LATENT_FORWARD:-64}"
MIN_N_LATENT_FORWARD="${MIN_N_LATENT_FORWARD:-0}"
EOL_TEMPERATURE="${EOL_TEMPERATURE:-1.0}"
OOD_SUBSET_DIR="${RUN_OUTPUTS}/ood_subsets/${TAG}_max${TRACE_OOD_MAX_SAMPLES}"

mkdir -p "${RUN_OUTPUTS}"
if [ "${TRACE_OOD_MAX_SAMPLES}" -gt 0 ]; then
  mkdir -p "${OOD_SUBSET_DIR}"
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && \
   source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && \
   conda activate ROT && \
   for DATASET in ${TRACE_OOD_DATASETS}; do \
     case \${DATASET} in \
       gsm8k_aug_nl) DATASET_DIR='/home/dingxukai/RoT/data/GSM8k-Aug-NL'; DATA_FILE='gsm8k_test_processed.jsonl' ;; \
       gsmhard) DATASET_DIR='/home/dingxukai/RoT/data/GSM8k-Hard'; DATA_FILE='gsmhard_test_processed.jsonl' ;; \
       svamp) DATASET_DIR='/home/dingxukai/RoT/data/SVAMP'; DATA_FILE='svamp_test_processed.jsonl' ;; \
       multiarith) DATASET_DIR='/home/dingxukai/RoT/data/Multiarith'; DATA_FILE='multiarith_test_processed.jsonl' ;; \
     esac; \
     if [ '${TRACE_OOD_MAX_SAMPLES}' -gt 0 ]; then \
       SUBSET_FILE=\"\${DATASET}_max${TRACE_OOD_MAX_SAMPLES}.jsonl\"; \
       head -n '${TRACE_OOD_MAX_SAMPLES}' \"\${DATASET_DIR}/\${DATA_FILE}\" > '${OOD_SUBSET_DIR}'/\"\${SUBSET_FILE}\"; \
       DATASET_DIR='${OOD_SUBSET_DIR}'; \
       DATA_FILE=\"\${SUBSET_FILE}\"; \
     fi; \
     echo \"[TRACE OOD] testing \${DATASET} test_times=${TEST_TIMES} max_samples=${TRACE_OOD_MAX_SAMPLES}\" | tee -a '${RUN_OUTPUTS}/${SESSION}.log'; \
     CUDA_VISIBLE_DEVICES=${GPU} python run.py \
       --model=trace_colar_qwen3_instruct \
       --dataset=\${DATASET} \
       --devices=0 \
       --test_ckpt_path='${CKPT}' \
       --test_times=${TEST_TIMES} \
       --workspace_path=/home/dingxukai \
       batch_size=1 \
       val_batch_size=1 \
       num_workers=4 \
       persistent_workers=False \
       dataset_name=\${DATASET} \
       dataset_dir=\${DATASET_DIR} \
       train_file=\${DATA_FILE} \
       val_file=\${DATA_FILE} \
       test_file=\${DATA_FILE} \
       model.model_kwargs.latent_generation_config.max_n_latent_forward=${MAX_N_LATENT_FORWARD} \
       model.model_kwargs.latent_generation_config.min_n_latent_forward=${MIN_N_LATENT_FORWARD} \
       model.model_kwargs.latent_generation_config.eol_temperature=${EOL_TEMPERATURE} \
       max_new_tokens=16 \
       compression_factor=5 \
       2>&1 | tee -a '${RUN_OUTPUTS}/${SESSION}.log'; \
   done; \
   echo '[TRACE OOD] done' | tee -a '${RUN_OUTPUTS}/${SESSION}.log'"

echo "Started ${SESSION}"
