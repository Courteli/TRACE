#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export TRACE_DATA_ROOT=${TRACE_DATA_ROOT:-${REPO_ROOT}/data}
export TRACE_MODEL_PATH=${TRACE_MODEL_PATH:-${REPO_ROOT}/models/base_reference}
cd "${REPO_ROOT}/main/native_v9"
exec "${TRACE_PYTHON:-python}" -B -m unittest discover -s tests -p 'test_*.py' -v
