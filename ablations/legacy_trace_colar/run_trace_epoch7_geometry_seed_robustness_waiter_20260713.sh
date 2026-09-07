#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
ENV=/home/dingxukai/miniconda3/envs/ROT
EVIDENCE=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence
SEED0_RAW=${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_final_guarded_saveall_v6_matched_gpu3457/visual_stage2_trace_guarded_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json
SEED0_CENTERED=${EVIDENCE}/geometry_epoch7_stage2_metric_null1024/trace_bridge_geometry_summary.json
SEED1_RAW=${EVIDENCE}/geometry_replication_seed1/geometry_raw/trace_bridge_geometry_summary.json
SEED1_CENTERED=${EVIDENCE}/geometry_replication_seed1/geometry_stage2_metric_null1024/trace_bridge_geometry_summary.json
OUT=${EVIDENCE}/geometry_seed_robustness

mkdir -p "${OUT}"
while [[ ! -s "${SEED0_RAW}" || ! -s "${SEED0_CENTERED}" || ! -s "${SEED1_RAW}" || ! -s "${SEED1_CENTERED}" ]]; do
  printf '%s waiting for seed0/seed1 raw and centered summaries\n' "$(date '+%F %T')" >> "${OUT}/waiter.log"
  sleep 180
done

cd "${ROOT}"
"${ENV}/bin/python" tools/trace_bridge_geometry_seed_robustness.py \
  --seed0_raw "${SEED0_RAW}" \
  --seed1_raw "${SEED1_RAW}" \
  --seed0_centered "${SEED0_CENTERED}" \
  --seed1_centered "${SEED1_CENTERED}" \
  --out_dir "${OUT}" \
  --bootstrap_trials 10000 \
  --seed 0 \
  > "${OUT}/geometry_seed_robustness.log"

printf 'TRACE epoch7 independent-seed geometry robustness completed.\n' > "${OUT}/done.txt"
