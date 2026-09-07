#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: V7_BEST_CKPT=/path/to/best.ckpt $0 <four-physical-gpu-csv>" >&2
  exit 2
fi
physical_gpus=$1
trace_vb_require_formal_gpu_set "${physical_gpus}" "pipeline"
[[ -n "${V7_BEST_CKPT:-}" && -f "${V7_BEST_CKPT}" ]] || \
  trace_vb_die "V7_BEST_CKPT must point to the completed formal v7 best"
[[ -n "${V7_STAGE1_DIR:-}" && -d "${V7_STAGE1_DIR}" ]] || \
  trace_vb_die "V7_STAGE1_DIR must point to the completed formal v7 Stage 1"

TRAIN_SEED=${TRAIN_SEED:-0}
PIPELINE_TAG=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v8_train_seed${TRAIN_SEED}}
trace_vb_require_safe_tag "${PIPELINE_TAG}"
PIPELINE_DIR=${TRACE_VB_ARTIFACT_ROOT}/pipelines/${PIPELINE_TAG}
STAGE1_TAG=${PIPELINE_TAG}_stage1
STAGE2_TAG=${PIPELINE_TAG}_stage2
STAGE1_DIR=${TRACE_VB_ARTIFACT_ROOT}/training/${STAGE1_TAG}
STAGE2_DIR=${TRACE_VB_ARTIFACT_ROOT}/training/${STAGE2_TAG}
METRIC_SAFE_BASELINE_ARTIFACT=${PIPELINE_DIR}/metric_safe_baseline.json
REGISTERED_CAPABILITY_VALIDATION_ARTIFACT=${PIPELINE_DIR}/registered_capability_validation.json

trace_vb_assert_no_test_tokens "${BASH_SOURCE[0]}"
trace_vb_assert_no_test_tokens "${SCRIPT_DIR}/run_stage1_vb.sh"
trace_vb_assert_no_test_tokens "${SCRIPT_DIR}/run_stage2_vb.sh"
trace_vb_require_no_v7_training
trace_vb_require_metric_artifacts
trace_vb_require_four_gpus "${physical_gpus}"
[[ ! -e "${PIPELINE_DIR}" ]] || \
  trace_vb_die "refusing to reuse pipeline tag: ${PIPELINE_TAG}"
V7_TRIGGER_AUDIT_TMP=$(mktemp /tmp/trace_vb_v8_pipeline_trigger_XXXXXX.json)
trap 'rm -f "${V7_TRIGGER_AUDIT_TMP}"' EXIT
trace_vb_audit_v7_trigger \
  "${V7_STAGE1_DIR}" "${V7_BEST_CKPT}" "${V7_TRIGGER_AUDIT_TMP}"
mkdir -p "${PIPELINE_DIR}"
cp "${V7_TRIGGER_AUDIT_TMP}" "${PIPELINE_DIR}/v7_trigger_contract.json"
cp "${TRACE_VB_METRIC_SAFE_BASELINE}" \
  "${METRIC_SAFE_BASELINE_ARTIFACT}"
cp "${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION}" \
  "${REGISTERED_CAPABILITY_VALIDATION_ARTIFACT}"
trace_vb_require_sha256 \
  "pipeline metric-safe baseline" \
  "${METRIC_SAFE_BASELINE_ARTIFACT}" \
  "${TRACE_VB_METRIC_SAFE_BASELINE_SHA256}"
trace_vb_require_sha256 \
  "pipeline registered capability validation" \
  "${REGISTERED_CAPABILITY_VALIDATION_ARTIFACT}" \
  "${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_SHA256}"
cat > "${PIPELINE_DIR}/manifest.txt" <<EOF
model=${TRACE_VB_VERSION}
pipeline=train_and_validation_only
pipeline_tag=${PIPELINE_TAG}
project_root=${TRACE_VB_CODE_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
v7_best_checkpoint=${V7_BEST_CKPT}
v7_best_checkpoint_sha256=$(sha256sum "${V7_BEST_CKPT}" | awk '{print $1}')
v7_stage1_dir=${V7_STAGE1_DIR}
v7_trigger_contract=${PIPELINE_DIR}/v7_trigger_contract.json
registered_capability=${TRACE_VB_REGISTERED_CAPABILITY}
registered_capability_sha256=${TRACE_VB_REGISTERED_CAPABILITY_SHA256}
registered_capability_payload_tensors=${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_TENSORS}
registered_capability_payload_sha256=${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}
cot_encoder_checkpoint_sha256=${TRACE_VB_COT_ENCODER_SHA256}
metric_safe_baseline_source=student_commit
metric_safe_baseline_origin=${TRACE_VB_METRIC_SAFE_BASELINE}
metric_safe_baseline_artifact=${METRIC_SAFE_BASELINE_ARTIFACT}
metric_safe_baseline_sha256=${TRACE_VB_METRIC_SAFE_BASELINE_SHA256}
metric_safe_baseline_correct=${TRACE_VB_METRIC_SAFE_BASELINE_CORRECT}
metric_safe_baseline_questions=${TRACE_VB_METRIC_SAFE_BASELINE_QUESTIONS}
registered_capability_validation_source=capability_teacher_all_roles
registered_capability_validation_origin=${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION}
registered_capability_validation_artifact=${REGISTERED_CAPABILITY_VALIDATION_ARTIFACT}
registered_capability_validation_sha256=${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_SHA256}
registered_capability_validation_correct=${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_CORRECT}
registered_capability_validation_questions=${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_QUESTIONS}
stage1_tag=${STAGE1_TAG}
stage2_tag=${STAGE2_TAG}
physical_gpus=${physical_gpus}
formal_gpus=${TRACE_VB_FORMAL_GPUS}
fixed_gpus=${TRACE_VB_FIXED_GPUS}
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF

TRACE_VB_FORMAL_GPUS="${TRACE_VB_FORMAL_GPUS}" \
TRACE_VB_FIXED_GPUS="${TRACE_VB_FIXED_GPUS}" \
RUN_TAG="${STAGE1_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
  V7_STAGE1_DIR="${V7_STAGE1_DIR}" \
  bash "${SCRIPT_DIR}/run_stage1_vb.sh" \
    "${physical_gpus}" "${V7_BEST_CKPT}"
[[ -s "${STAGE1_DIR}/best_checkpoint.txt" ]] || \
  trace_vb_die "Stage 1 did not publish a selected checkpoint"
[[ -s "${STAGE1_DIR}/candidate_index.json" ]] || \
  trace_vb_die "Stage 1 did not publish its five-candidate index"
stage1_checkpoint=$(<"${STAGE1_DIR}/best_checkpoint.txt")

TRACE_VB_FORMAL_GPUS="${TRACE_VB_FORMAL_GPUS}" \
TRACE_VB_FIXED_GPUS="${TRACE_VB_FIXED_GPUS}" \
RUN_TAG="${STAGE2_TAG}" TRAIN_SEED="${TRAIN_SEED}" \
  bash "${SCRIPT_DIR}/run_stage2_vb.sh" \
    "${physical_gpus}" \
    "${stage1_checkpoint}" \
    "${STAGE1_DIR}/candidate_index.json"
[[ -s "${STAGE2_DIR}/best_checkpoint.txt" ]] || \
  trace_vb_die "Stage 2 did not publish a selected checkpoint"
[[ -s "${STAGE2_DIR}/candidate_index.json" ]] || \
  trace_vb_die "Stage 2 did not publish its five-candidate index"
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/publish_pipeline_completion_v8.py" \
  --pipeline-dir "${PIPELINE_DIR}" \
  --stage1-dir "${STAGE1_DIR}" \
  --stage2-dir "${STAGE2_DIR}" \
  --physical-gpus "${physical_gpus}"
