#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
ENV=/home/dingxukai/miniconda3/envs/ROT
TRACE_OUT=${ROOT}/run_outputs/trace_bridge/20260710_trace_bridge_final_streaming_modecontrast_gpu0123
ANSWER_OUT=${ROOT}/run_outputs/trace_bridge/20260710_trace_bridge_final_vizstrong_answeronly_gpu457
STAGE1_OUT=${ROOT}/run_outputs/trace_bridge/20260710_trace_bridge_final_stage1_rollout_audit_gpu0
AUDIT_OUT=${ROOT}/run_outputs/trace_bridge/20260710_trace_bridge_final_goal_audit
STAGE1_CKPT=/disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_284260_20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1_stage1/checkpoints/epoch1__step13452__monitor0.661.ckpt
BASELINE_SUMMARY=${ROOT}/run_outputs/trace_bridge/20260708_trace_bridge_bridgefull_bridge_baseline_gpu2/summary_snapshot_epoch0__step6726__monitor0_655__mtime1783503244088392417__size674673616_.md
OLD_STAGE1_SUMMARY=${ROOT}/run_outputs/trace_bridge/20260708_trace_bridge_bridgefull_stage1eval_vizstrong_stage1_epoch1_m661_gpu1/summary_vizstrong_stage1_epoch1_m661.md

TRACE_GEOMETRY=${TRACE_OUT}/visual_stage2_trace_modecontrast_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json
ANSWER_GEOMETRY=${ANSWER_OUT}/visual_stage2_answer_only_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json

mkdir -p "${AUDIT_OUT}"
cd "${ROOT}"

while [[ ! -f "${TRACE_GEOMETRY}" || ! -f "${ANSWER_GEOMETRY}" ]]; do
  if [[ ! -f "${TRACE_GEOMETRY}" ]] && ! tmux has-session -t trace_final_streaming_mode_gpu0123_0710 2>/dev/null; then
    printf 'TRACE pipeline ended before geometry output was produced.\n' > "${AUDIT_OUT}/watcher_failed.txt"
    exit 1
  fi
  if [[ ! -f "${ANSWER_GEOMETRY}" ]] && ! tmux has-session -t trace_final_vizstrong_answer_gpu457_0710 2>/dev/null; then
    printf 'Answer-only pipeline ended before geometry output was produced.\n' > "${AUDIT_OUT}/watcher_failed.txt"
    exit 1
  fi
  sleep 300
done

env \
  MODE=eval_trace_from_ckpt \
  RUN_TAG=20260710_trace_bridge_final_stage1_rollout_audit_gpu0 \
  TRACE_MODEL=trace_bridge_qwen3_instruct_vizstrong \
  GPU=0 \
  CKPT="${STAGE1_CKPT}" \
  EVAL_LABEL=stage1_rollout_baseline \
  TEST_TIMES=1 \
  bash "${ROOT}/run_trace_bridge_pipeline_snapshot_20260707.sh"

STAGE1_RECORD=${STAGE1_OUT}/eval_stage1_rollout_baseline_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
ANSWER_RECORD=${ANSWER_OUT}/eval_stage2_answer_only_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
TRACE_RECORD=${TRACE_OUT}/eval_stage2_trace_modecontrast_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
STAGE1_GEOMETRY=${STAGE1_OUT}/visual_stage1_rollout_baseline_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json

"${ENV}/bin/python" tools/trace_bridge_compare_rollouts.py \
  --record "Stage1=${STAGE1_RECORD}" \
  --record "answer-only=${ANSWER_RECORD}" \
  --record "TRACE=${TRACE_RECORD}" \
  --out_dir "${AUDIT_OUT}/global_pca" \
  --max_records 200 \
  --pca_fit_records 200

"${ENV}/bin/python" tools/trace_bridge_final_audit.py \
  --baseline_summary "${BASELINE_SUMMARY}" \
  --stage1_summary "${OLD_STAGE1_SUMMARY}" \
  --answer_summary "${ANSWER_OUT}/summary_stage2_answer_only.md" \
  --trace_summary "${TRACE_OUT}/summary_stage2_trace_modecontrast.md" \
  --stage1_geometry "${STAGE1_GEOMETRY}" \
  --answer_geometry "${ANSWER_GEOMETRY}" \
  --trace_geometry "${TRACE_GEOMETRY}" \
  --out_dir "${AUDIT_OUT}"

printf 'Final TRACE audit completed.\n' > "${AUDIT_OUT}/watcher_done.txt"
