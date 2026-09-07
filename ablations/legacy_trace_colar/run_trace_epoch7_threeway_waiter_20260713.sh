#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
OUT=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence
ENV=/home/dingxukai/miniconda3/envs/ROT
BRIDGE_RECORD=${OUT}/matched_bridge_rollouts/logs/tb/run/trace_bridge_visual_test.pt
STAGE1_RECORD=${OUT}/matched_stage1_rollouts/logs/tb/run/trace_bridge_visual_test.pt
BRIDGE=${OUT}/matched_bridge_rollouts/geometry_stage2_metric_null1024/trace_bridge_geometry_summary.json
STAGE1=${OUT}/matched_stage1_rollouts/geometry_stage2_metric_null1024/trace_bridge_geometry_summary.json
TRACE=${OUT}/geometry_epoch7_stage2_metric_null1024/trace_bridge_geometry_summary.json
TARGET=${OUT}/threeway_geometry_bridge_stage1_epoch7_null1024

mkdir -p "${TARGET}"
while [[ ! -s "${BRIDGE_RECORD}" || ! -s "${STAGE1_RECORD}" || ! -s "${TRACE}" ]]; do
  printf '%s waiting for BRIDGE/Stage1 records and epoch7 high-precision geometry\n' "$(date '+%F %T')" >> "${TARGET}/waiter.log"
  sleep 300
done

cd "${ROOT}"
if [[ ! -s "${STAGE1}" ]]; then
  "${ENV}/bin/python" tools/trace_bridge_geometry_summary.py \
    --records "${STAGE1_RECORD}" \
    --out_dir "$(dirname "${STAGE1}")" \
    --max_records 200 \
    --signature_representation stage2_centered \
    --signature_raw_mix 0.25 \
    --permutation_null_trials 1024 \
    > "${OUT}/matched_stage1_rollouts/geometry_stage2_metric_null1024.log"
fi
if [[ ! -s "${BRIDGE}" ]]; then
  "${ENV}/bin/python" tools/trace_bridge_geometry_summary.py \
    --records "${BRIDGE_RECORD}" \
    --out_dir "$(dirname "${BRIDGE}")" \
    --max_records 200 \
    --signature_representation stage2_centered \
    --signature_raw_mix 0.25 \
    --permutation_null_trials 1024 \
    > "${OUT}/matched_bridge_rollouts/geometry_stage2_metric_null1024.log"
fi
"${ENV}/bin/python" tools/trace_bridge_threeway_geometry.py \
  --summary "BRIDGE=${BRIDGE}" \
  --summary "Stage1=${STAGE1}" \
  --summary "TRACE-epoch7=${TRACE}" \
  --target_label TRACE-epoch7 \
  --out_dir "${TARGET}" \
  --bootstrap_trials 10000 \
  --seed 0 \
  2>&1 | tee "${TARGET}/threeway_geometry.log"

printf 'Three-way 200-question high-precision geometry scorecard completed.\n' > "${TARGET}/done.txt"
