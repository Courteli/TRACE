#!/usr/bin/env bash
set -euo pipefail
TRACE_STRUCTURED_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${TRACE_STRUCTURED_CODE}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${TRACE_STRUCTURED_CODE}"
exec "${TRACE_PYTHON:-python}" -B -m unittest discover -s tests -p 'test_*.py' -v
