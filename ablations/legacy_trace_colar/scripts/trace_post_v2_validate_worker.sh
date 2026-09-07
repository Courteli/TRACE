#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
SAMPLES="${2:-64}"
ROOT="/disk1/dingxukai/trace_colar"
RUN_OUTPUTS="${ROOT}/run_outputs/trace"
DIAG_DIR="${RUN_OUTPUTS}/diagnostics"
STAMP="$(date +%Y%m%d-%H%M%S)"

V0_RUN="${V0_RUN:-trace_v0_sft_qwen3_c5_gpu0}"
V2_RUN="${V2_RUN:-trace_v2_rl_qwen3_c5_gpu0}"
ANSWER_RUN="${ANSWER_RUN:-trace_answer_only_rl_qwen3_c5_gpu4}"
DIAG_GROUP_SIZE="${DIAG_GROUP_SIZE:-8}"
TEST_TIMES="${TEST_TIMES:-5}"
TRACE_OOD_MAX_SAMPLES="${TRACE_OOD_MAX_SAMPLES:-0}"
OOD_SUBSET_DIR="${RUN_OUTPUTS}/ood_subsets/post_${STAMP}_max${TRACE_OOD_MAX_SAMPLES}"

source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
mkdir -p "${DIAG_DIR}"
if [ "${TRACE_OOD_MAX_SAMPLES}" -gt 0 ]; then
  mkdir -p "${OOD_SUBSET_DIR}"
fi

find_ckpt() {
  python tools/trace_find_checkpoint.py --run_contains "$1"
}

V0_CKPT="$(find_ckpt "${V0_RUN}")"
V2_CKPT="$(find_ckpt "${V2_RUN}")"
ANSWER_CKPT="$(find_ckpt "${ANSWER_RUN}")"

echo "[TRACE post] v0=${V0_CKPT}"
echo "[TRACE post] v2=${V2_CKPT}"
echo "[TRACE post] answer_only=${ANSWER_CKPT}"

run_diag() {
  local name="$1"
  local ckpt="$2"
	  CUDA_VISIBLE_DEVICES="${GPU}" python tools/trace_group_diagnostics.py \
	    --ckpt "${ckpt}" \
	    --num_samples "${SAMPLES}" \
	    --group_size "${DIAG_GROUP_SIZE}" \
	    --output "${DIAG_DIR}/${STAMP}_${name}_diagnostics.json"
}

run_test() {
  local name="$1"
  local ckpt="$2"
  local dataset="$3"
  local dataset_dir="$4"
  local data_file="$5"
  local test_dataset_dir="${dataset_dir}"
  local test_data_file="${data_file}"
  if [ "${TRACE_OOD_MAX_SAMPLES}" -gt 0 ]; then
    test_data_file="${dataset}_${name}_max${TRACE_OOD_MAX_SAMPLES}.jsonl"
    head -n "${TRACE_OOD_MAX_SAMPLES}" "${dataset_dir}/${data_file}" > "${OOD_SUBSET_DIR}/${test_data_file}"
    test_dataset_dir="${OOD_SUBSET_DIR}"
  fi
  echo "[TRACE post] testing ${name} on ${dataset} test_times=${TEST_TIMES} max_samples=${TRACE_OOD_MAX_SAMPLES}"
  CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model=trace_colar_qwen3_instruct \
    --dataset="${dataset}" \
    --devices=0 \
    --test_ckpt_path="${ckpt}" \
    --test_times="${TEST_TIMES}" \
    --workspace_path=/home/dingxukai \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    persistent_workers=False \
    dataset_name="${dataset}" \
    dataset_dir="${test_dataset_dir}" \
    train_file="${test_data_file}" \
    val_file="${test_data_file}" \
    test_file="${test_data_file}" \
    max_new_tokens=16 \
    compression_factor=5
}

run_diag "v0" "${V0_CKPT}"
run_diag "v2" "${V2_CKPT}"
run_diag "answer_only" "${ANSWER_CKPT}"

for model_name in v2 answer_only; do
  if [[ "${model_name}" == "v2" ]]; then
    ckpt="${V2_CKPT}"
  else
    ckpt="${ANSWER_CKPT}"
  fi
  run_test "${model_name}" gsm8k_aug_nl /home/dingxukai/RoT/data/GSM8k-Aug-NL gsm8k_test_processed.jsonl
  run_test "${model_name}" gsmhard /home/dingxukai/RoT/data/GSM8k-Hard gsmhard_test_processed.jsonl
  run_test "${model_name}" svamp /home/dingxukai/RoT/data/SVAMP svamp_test_processed.jsonl
  run_test "${model_name}" multiarith /home/dingxukai/RoT/data/Multiarith multiarith_test_processed.jsonl
done

python tools/trace_summarize_runs.py \
  --log_root logs/trace_colar_qwen3_instruct \
  --latest 40 \
  --json_out "${RUN_OUTPUTS}/${STAMP}_trace_summary.json" \
  | tee "${RUN_OUTPUTS}/${STAMP}_trace_summary.md"

echo "[TRACE post] done"
