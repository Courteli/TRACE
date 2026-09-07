#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <physical-gpu>" >&2
  exit 2
fi
physical_gpu=$1
[[ "${physical_gpu}" =~ ^[0-9]+$ ]] || trace_vb_die "invalid cache GPU: ${physical_gpu}"
trace_vb_require_stage0

cache_dir=$(dirname "${TRACE_VB_SUFFICIENCY_CACHE}")
pilot_cache=${cache_dir}/gsm8k_prefix_sufficiency_pilot256.pt
mkdir -p "${cache_dir}" "${TRACE_VB_ARTIFACT_ROOT}/tmp"
cd "${CODE_ROOT}"

if [[ ! -s "${TRACE_VB_SUFFICIENCY_CACHE}" ]]; then
  if [[ ! -s "${pilot_cache}" ]]; then
    env \
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      TRACE_PROJECT_ROOT="${CODE_ROOT}" \
      TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
      TMPDIR="${TRACE_VB_ARTIFACT_ROOT}/tmp" \
      TOKENIZERS_PARALLELISM=false \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES="${physical_gpu}" \
      "${TRACE_VB_PYTHON}" tools/build_trace_vb_sufficiency_cache.py \
        --data "${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL/gsm8k_train_processed.jsonl" \
        --teacher_checkpoint "${TRACE_VB_REGISTERED_STAGE0}" \
        --base_model "${TRACE_VB_BASE_MODEL}" \
        --output "${pilot_cache}" \
        --batch_size 4 \
        --device cuda:0 \
        --dtype bfloat16 \
        --limit 256 \
        2>&1 | tee "${cache_dir}/pilot256_build.log"
  fi
  "${TRACE_VB_PYTHON}" tools/build_trace_vb_sufficiency_cache.py \
    --audit_only \
    --output "${pilot_cache}" \
    --min_valid_row_fraction 0.50 \
    --min_valid_prefix_fraction 0.20 \
    --min_nonzero_gain_fraction 0.30 \
    --min_score_span_mean 0.05 \
    | tee "${cache_dir}/pilot256_audit.json"

  env \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TRACE_PROJECT_ROOT="${CODE_ROOT}" \
    TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
    TMPDIR="${TRACE_VB_ARTIFACT_ROOT}/tmp" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    "${TRACE_VB_PYTHON}" tools/build_trace_vb_sufficiency_cache.py \
      --data "${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL/gsm8k_train_processed.jsonl" \
      --teacher_checkpoint "${TRACE_VB_REGISTERED_STAGE0}" \
      --base_model "${TRACE_VB_BASE_MODEL}" \
      --output "${TRACE_VB_SUFFICIENCY_CACHE}" \
      --batch_size 4 \
      --device cuda:0 \
      --dtype bfloat16 \
      2>&1 | tee "${cache_dir}/build.log"
fi

"${TRACE_VB_PYTHON}" tools/build_trace_vb_sufficiency_cache.py \
  --audit_only \
  --output "${TRACE_VB_SUFFICIENCY_CACHE}" \
  --min_valid_row_fraction 0.50 \
  --min_valid_prefix_fraction 0.20 \
  --min_nonzero_gain_fraction 0.30 \
  --min_score_span_mean 0.05 \
  | tee "${cache_dir}/audit.json"
