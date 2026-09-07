#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <capability|student>" >&2
  exit 2
fi
physical_gpus=$1
mode=$2
case "${mode}" in
  capability)
    validation_path=capability_teacher_all_roles
    exact_correct=540
    minimum_correct=540
    ;;
  student)
    validation_path=student_commit
    exact_correct=-1
    minimum_correct=449
    ;;
  *) trace_vb_die "unknown preflight mode: ${mode}" ;;
esac

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v7_${mode}_preflight}
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  trace_vb_die "RUN_TAG contains unsafe characters: ${RUN_TAG}"
DATASET_DIR=${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL
LOG_ROOT=${TRACE_VB_ARTIFACT_ROOT}/preflight_logs
TMP_ROOT=${TRACE_VB_ARTIFACT_ROOT}/tmp
OUT_DIR=${TRACE_VB_ARTIFACT_ROOT}/preflight/${RUN_TAG}
log_parent=${LOG_ROOT}/${TRACE_VB_MODEL_CONFIG}/gsm8k_aug_nl-gsm8k_aug_nl

trace_vb_require_four_gpus "${physical_gpus}"
trace_vb_require_capability
trace_vb_require_stage0
trace_vb_require_cache
[[ ! -e "${OUT_DIR}" ]] || trace_vb_die "preflight tag already exists: ${RUN_TAG}"
mkdir -p "${OUT_DIR}" "${TMP_ROOT}" "${log_parent}"
cd "${CODE_ROOT}"

env \
  TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  TRACE_LOG_ROOT="${LOG_ROOT}" \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${TRACE_VB_PYTHON}" run.py \
    --model "${TRACE_VB_MODEL_CONFIG}" \
    --dataset gsm8k_aug_nl \
    --trainer trace_vb_stage1_v2 \
    --devices 0,1,2,3 \
    --workspace_path "${CODE_ROOT}" \
    --load_ckpt_path "${TRACE_VB_REGISTERED_CAPABILITY}" \
    --cot_encoder_ckpt_path "${TRACE_VB_REGISTERED_STAGE0}" \
    --test_times 1 \
    --seed 271828 \
    --disable_early_stopping \
    --log_suffix "${RUN_TAG}" \
    data_module.dataset_dir="${DATASET_DIR}" \
    data_module.enforce_registered_source=true \
    data_module.tiny_dataset=false \
    data_module.epoch_scaling=1 \
    batch_size=4 \
    val_batch_size=1 \
    num_workers=0 \
    pin_memory=false \
    persistent_workers=false \
    trainer.max_epochs=1 \
    trainer.max_steps=-1 \
    trainer.limit_train_batches=1 \
    trainer.limit_val_batches=1.0 \
    trainer.num_sanity_val_steps=0 \
    trainer.log_every_n_steps=1 \
    save_top_k=0 \
    save_last=false \
    model.model_kwargs.do_trace_rl=false \
    model.model_kwargs.trace_policy_config.validation_path="${validation_path}" \
    model.model_kwargs.trace_policy_config.stage1_capability_lora_lr=0.0 \
    model.model_kwargs.trace_policy_config.stage1_role_lr=0.0 \
    model.model_kwargs.trace_policy_config.stage1_recovery_checkpoint_interval=0 \
    model.model_kwargs.trace_policy_config.stage1_host_memory_guard_interval=0 \
    model.model_kwargs.trace_policy_config.visual_record_limit=0 \
    model.training_kwargs.scheduler.warmup_steps=0 \
    model.training_kwargs.scheduler.num_training_steps=1 \
    2>&1 | tee "${OUT_DIR}/run.log"

mapfile -t logger_dirs < <(
  find "${log_parent}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${RUN_TAG}" -print
)
[[ "${#logger_dirs[@]}" -eq 1 ]] || \
  trace_vb_die "expected one preflight logger directory, found ${#logger_dirs[@]}"
summary=${logger_dirs[0]}/validation_epoch_000.json
[[ -s "${summary}" ]] || trace_vb_die "missing strict preflight validation summary"

"${TRACE_VB_PYTHON}" - \
  "${summary}" "${OUT_DIR}/validation_gate.json" \
  "${validation_path}" "${exact_correct}" "${minimum_correct}" <<'PY'
import json
import math
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
expected_path = sys.argv[3]
exact_correct = int(sys.argv[4])
minimum_correct = int(sys.argv[5])
summary = json.loads(source.read_text(encoding="utf-8"))
failures = []
if summary.get("schema_version") != "trace_vb_v7_validation_behavior_v1":
    failures.append("wrong validation schema")
if summary.get("validation_path") != expected_path:
    failures.append("wrong validation path")
if int(summary.get("world_size", -1)) != 4:
    failures.append("world_size is not four")
if int(summary.get("unique_questions", -1)) != 747:
    failures.append("validation is not the strict 747-question set")
correct = int(summary.get("correct_count", -1))
accuracy = float(summary.get("accuracy", float("nan")))
if not math.isfinite(accuracy) or not math.isclose(
    accuracy, correct / 747, rel_tol=0.0, abs_tol=1e-12
):
    failures.append("accuracy is inconsistent with integer correct_count")
if exact_correct >= 0 and correct != exact_correct:
    failures.append(f"capability parity is {correct}/747, expected {exact_correct}/747")
if correct < minimum_correct:
    failures.append(f"accuracy is {correct}/747, requires at least {minimum_correct}/747")
report = {
    **summary,
    "status": "FAIL" if failures else "PASS",
    "exact_correct_required": exact_correct,
    "minimum_correct_required": minimum_correct,
    "failures": failures,
}
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if failures:
    raise SystemExit("TRACE-VB-v7 preflight failed: " + "; ".join(failures))
PY

printf '%s\n' "${logger_dirs[0]}" > "${OUT_DIR}/logger_dir.txt"
