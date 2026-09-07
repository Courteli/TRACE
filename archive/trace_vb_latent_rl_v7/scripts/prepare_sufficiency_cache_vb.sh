#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
TRACE_VB_SAFE_RUNNER='import runpy, sys; import src.utils.safe_checkpoint; script = sys.argv[1]; sys.argv = sys.argv[1:]; runpy.run_path(script, run_name="__main__")'

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

TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
  tools/data_contract_audit.py > "${cache_dir}/registered_data_contract.json"
"${TRACE_VB_PYTHON}" - \
  "${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL/gsm8k_val_processed.jsonl" \
  "${cache_dir}/trace_vb_v7_validation_source_contract.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
rows = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 747:
    raise SystemExit(f"registered validation has {len(rows)} rows, expected 747")
source_ids = []
for index, row in enumerate(rows):
    if "id" not in row:
        raise SystemExit(f"registered validation row {index} has no source id")
    source_ids.append(int(row["id"]))
if len(set(source_ids)) != 747:
    raise SystemExit("registered validation source ids are not unique")
output.write_text(
    json.dumps(
        {
            "schema_version": "trace_vb_v7_validation_source_id_contract_v1",
            "rows": 747,
            "unique_source_ids": 747,
            "deduplication_key": "source_id",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

if [[ ! -s "${TRACE_VB_SUFFICIENCY_CACHE}" ]]; then
  if [[ ! -s "${pilot_cache}" ]]; then
    env \
      TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
      TRACE_PROJECT_ROOT="${CODE_ROOT}" \
      TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
      TMPDIR="${TRACE_VB_ARTIFACT_ROOT}/tmp" \
      TOKENIZERS_PARALLELISM=false \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES="${physical_gpu}" \
      "${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
        tools/build_trace_vb_sufficiency_cache.py \
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
  "${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
    tools/build_trace_vb_sufficiency_cache.py \
    --audit_only \
    --output "${pilot_cache}" \
    --min_valid_row_fraction 0.50 \
    --min_valid_prefix_fraction 0.20 \
    --min_nonzero_gain_fraction 0.30 \
    --min_score_span_mean 0.05 \
    | tee "${cache_dir}/pilot256_audit.json"

  env \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
    TRACE_PROJECT_ROOT="${CODE_ROOT}" \
    TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
    TMPDIR="${TRACE_VB_ARTIFACT_ROOT}/tmp" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    "${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
      tools/build_trace_vb_sufficiency_cache.py \
      --data "${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL/gsm8k_train_processed.jsonl" \
      --teacher_checkpoint "${TRACE_VB_REGISTERED_STAGE0}" \
      --base_model "${TRACE_VB_BASE_MODEL}" \
      --output "${TRACE_VB_SUFFICIENCY_CACHE}" \
      --batch_size 4 \
      --device cuda:0 \
      --dtype bfloat16 \
      2>&1 | tee "${cache_dir}/build.log"
fi

"${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
  tools/build_trace_vb_sufficiency_cache.py \
  --audit_only \
  --output "${TRACE_VB_SUFFICIENCY_CACHE}" \
  --min_valid_row_fraction 0.50 \
  --min_valid_prefix_fraction 0.20 \
  --min_nonzero_gain_fraction 0.30 \
  --min_score_span_mean 0.05 \
  | tee "${cache_dir}/audit.json"
