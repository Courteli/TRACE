#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -ne 4 ]]; then
  echo "Usage: V7_BEST_CKPT=/path/to/v7-best.ckpt $0 <gpus> <pipeline-tag> <stage1|stage2> <recovery-checkpoint>" >&2
  exit 2
fi
physical_gpus=$1
PIPELINE_TAG=$2
resume_phase=$3
recovery_checkpoint=$4
trace_vb_require_formal_gpu_set "${physical_gpus}" "recovery"
trace_vb_require_safe_tag "${PIPELINE_TAG}"
[[ -f "${recovery_checkpoint}" ]] || \
  trace_vb_die "missing recovery checkpoint: ${recovery_checkpoint}"
[[ -n "${V7_BEST_CKPT:-}" && -f "${V7_BEST_CKPT}" ]] || \
  trace_vb_die "V7_BEST_CKPT is required for recovery continuity"
[[ -n "${V7_STAGE1_DIR:-}" && -d "${V7_STAGE1_DIR}" ]] || \
  trace_vb_die "V7_STAGE1_DIR is required for recovery continuity"

TRAIN_SEED=${TRAIN_SEED:-0}
PIPELINE_DIR=${TRACE_VB_ARTIFACT_ROOT}/pipelines/${PIPELINE_TAG}
STAGE1_TAG=${PIPELINE_TAG}_stage1
STAGE2_TAG=${PIPELINE_TAG}_stage2
STAGE1_DIR=${TRACE_VB_ARTIFACT_ROOT}/training/${STAGE1_TAG}
STAGE2_DIR=${TRACE_VB_ARTIFACT_ROOT}/training/${STAGE2_TAG}
[[ -s "${PIPELINE_DIR}/manifest.txt" ]] || \
  trace_vb_die "recovery requires the original pipeline manifest"
if [[ "${resume_phase}" == stage1 ]]; then
  resume_candidate_dir=${STAGE1_DIR}/candidates
elif [[ "${resume_phase}" == stage2 ]]; then
  resume_candidate_dir=${STAGE2_DIR}/candidates
else
  trace_vb_die "resume phase must be stage1 or stage2"
fi
trace_vb_assert_no_test_tokens "${BASH_SOURCE[0]}"
trace_vb_require_metric_artifacts
resume_binding_attempt=${TRACE_VB_RESUME_ATTEMPT:-$(date +%Y%m%d-%H%M%S)}
trace_vb_require_safe_tag "${resume_binding_attempt}"
resume_binding_path=${PIPELINE_DIR}/resume_binding_${resume_phase}_${resume_binding_attempt}.json
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/audit_pipeline_resume_binding_v8.py" \
  --manifest "${PIPELINE_DIR}/manifest.txt" \
  --pipeline-tag "${PIPELINE_TAG}" \
  --stage1-tag "${STAGE1_TAG}" \
  --stage2-tag "${STAGE2_TAG}" \
  --train-seed "${TRAIN_SEED}" \
  --physical-gpus "${physical_gpus}" \
  --v7-best-checkpoint "${V7_BEST_CKPT}" \
  --recovery-checkpoint "${recovery_checkpoint}" \
  --phase "${resume_phase}" \
  --candidate-dir "${resume_candidate_dir}" \
  --output "${resume_binding_path}"
trace_vb_require_no_v7_training
trace_vb_require_four_gpus "${physical_gpus}"
trace_vb_audit_v7_trigger \
  "${V7_STAGE1_DIR}" "${V7_BEST_CKPT}" \
  "${PIPELINE_DIR}/v7_trigger_contract.recovery.json"

case "${resume_phase}" in
  stage1)
    TRACE_VB_FORMAL_GPUS="${TRACE_VB_FORMAL_GPUS}" \
      TRACE_VB_FIXED_GPUS="${TRACE_VB_FIXED_GPUS}" \
      RUN_TAG="${STAGE1_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
      TRACE_VB_RESUME_ATTEMPT="${resume_binding_attempt}" \
      STAGE1_RESUME_CKPT="${recovery_checkpoint}" \
      V7_STAGE1_DIR="${V7_STAGE1_DIR}" \
      bash "${SCRIPT_DIR}/run_stage1_vb.sh" \
        "${physical_gpus}" "${V7_BEST_CKPT}"
    stage1_checkpoint=$(<"${STAGE1_DIR}/best_checkpoint.txt")
    TRACE_VB_FORMAL_GPUS="${TRACE_VB_FORMAL_GPUS}" \
      TRACE_VB_FIXED_GPUS="${TRACE_VB_FIXED_GPUS}" \
      RUN_TAG="${STAGE2_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
      bash "${SCRIPT_DIR}/run_stage2_vb.sh" \
        "${physical_gpus}" \
        "${stage1_checkpoint}" \
        "${STAGE1_DIR}/candidate_index.json"
    ;;
  stage2)
    [[ -s "${STAGE1_DIR}/best_checkpoint.txt" ]] || \
      trace_vb_die "Stage-2 recovery is missing the Stage-1 best record"
    stage1_checkpoint=$(<"${STAGE1_DIR}/best_checkpoint.txt")
    TRACE_VB_FORMAL_GPUS="${TRACE_VB_FORMAL_GPUS}" \
      TRACE_VB_FIXED_GPUS="${TRACE_VB_FIXED_GPUS}" \
      RUN_TAG="${STAGE2_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
      TRACE_VB_RESUME_ATTEMPT="${resume_binding_attempt}" \
      STAGE2_RESUME_CKPT="${recovery_checkpoint}" \
      bash "${SCRIPT_DIR}/run_stage2_vb.sh" \
        "${physical_gpus}" \
        "${stage1_checkpoint}" \
        "${STAGE1_DIR}/candidate_index.json"
    ;;
  *) trace_vb_die "resume phase must be stage1 or stage2" ;;
esac

[[ -s "${STAGE1_DIR}/best_checkpoint.txt" ]] || \
  trace_vb_die "recovery did not publish the Stage-1 best record"
[[ -s "${STAGE1_DIR}/candidate_index.json" ]] || \
  trace_vb_die "recovery did not publish the Stage-1 candidate index"
[[ -s "${STAGE2_DIR}/best_checkpoint.txt" ]] || \
  trace_vb_die "recovery did not publish the final best record"
[[ -s "${STAGE2_DIR}/candidate_index.json" ]] || \
  trace_vb_die "recovery did not publish the final candidate index"
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/publish_pipeline_completion_v8.py" \
  --pipeline-dir "${PIPELINE_DIR}" \
  --stage1-dir "${STAGE1_DIR}" \
  --stage2-dir "${STAGE2_DIR}" \
  --physical-gpus "${physical_gpus}" \
  --recovery-phase "${resume_phase}" \
  --recovery-checkpoint "${recovery_checkpoint}" \
  --resume-attempt "${resume_binding_attempt}" \
  --resume-binding "${resume_binding_path}"
