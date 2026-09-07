#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
PY=/home/dingxukai/miniconda3/envs/ROT/bin/python
TECTONIC=/home/dingxukai/.local/bin/tectonic
ABLATION_ROOT=${ROOT}/run_outputs/trace/20260716_stage1_component_ablations_gpu0
DONE=${ABLATION_ROOT}/pipeline_done.txt
PUBLIC_OUT=${ROOT}/run_outputs/trace/20260716_paper_evidence_suite/component_ablations
PAPER=${ROOT}/paper/kdd2026_trace_direct
POLL_SECONDS=${POLL_SECONDS:-300}

BASELINE_ROOT=${ROOT}/run_outputs/trace_bridge/20260708_trace_bridge_bridgefull_stage1eval_vizstrong_stage1_epoch1_m661_gpu1
BASELINE_GSM8K=${BASELINE_ROOT}/eval_vizstrong_stage1_epoch1_m661_gsm8k_aug_logs/tb/run/test_20260709-035947_gsm_pid40123.json
BASELINE_GSMHARD=${BASELINE_ROOT}/eval_vizstrong_stage1_epoch1_m661_gsmhard_logs/tb/run/test_20260709-045259_gsm_pid84498.json
BASELINE_SVAMP=${BASELINE_ROOT}/eval_vizstrong_stage1_epoch1_m661_svamp_logs/tb/run/test_20260709-055736_gsm_pid18783.json
BASELINE_MULTIARITH=${BASELINE_ROOT}/eval_vizstrong_stage1_epoch1_m661_multiarith_logs/tb/run/test_20260709-062654_gsm_pid40289.json
BASELINE_RECORDS=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/matched_stage1_rollouts/logs/tb/run/trace_bridge_visual_test.pt

mkdir -p "${PUBLIC_OUT}"
exec 9>"${PUBLIC_OUT}/report_waiter.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "${PUBLIC_OUT}/report_waiter.pid"

while [[ ! -f "${DONE}" ]]; do
  printf '%s [wait] full-budget component ablations are active\n' "$(date '+%F %T')" \
    >> "${PUBLIC_OUT}/report_waiter.log"
  sleep "${POLL_SECONDS}"
done

cd "${ROOT}"
"${PY}" tools/trace_stage1_ablation_report.py \
  --root "${ABLATION_ROOT}" \
  --output-dir "${PUBLIC_OUT}" \
  --baseline-gsm8k "${BASELINE_GSM8K}" \
  --baseline-gsmhard "${BASELINE_GSMHARD}" \
  --baseline-svamp "${BASELINE_SVAMP}" \
  --baseline-multiarith "${BASELINE_MULTIARITH}" \
  --baseline-records "${BASELINE_RECORDS}" \
  --paper-table "${PAPER}/tables/ablation_tbd.tex" \
  --bootstrap-trials 10000 \
  --permutations 1024 \
  --seed 20260716 \
  > "${PUBLIC_OUT}/report.log" 2>&1

cp "${PUBLIC_OUT}/fig_stage1_component_ablation.pdf" \
  "${PAPER}/figures/fig_stage1_component_ablation.pdf"

cd "${PAPER}"
"${TECTONIC}" --keep-logs --keep-intermediates --outdir build main.tex \
  > "${PUBLIC_OUT}/paper_rebuild.log" 2>&1
printf '%s\n' "$(date '+%F %T')" > "${PUBLIC_OUT}/report_done.txt"
