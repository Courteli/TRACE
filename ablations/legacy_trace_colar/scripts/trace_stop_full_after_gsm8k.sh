#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-trace_test_ood_v2_g8_best0325_full_gpu4}" \
DATASET="${DATASET:-gsm8k_aug_nl}" \
MIN_ITEMS="${MIN_ITEMS:-1000}" \
MIN_TIMES="${MIN_TIMES:-5}" \
LOG="${LOG:-/disk1/dingxukai/trace_colar/run_outputs/trace/trace_stop_full_after_gsm8k.log}" \
  bash /disk1/dingxukai/trace_colar/scripts/trace_stop_session_after_dataset.sh
