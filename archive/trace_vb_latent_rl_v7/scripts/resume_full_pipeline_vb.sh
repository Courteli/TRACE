#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <cache-and-evidence-physical-gpu> <pipeline-tag> <stage1-last-checkpoint>" >&2
  exit 2
fi

physical_gpus=$1
evidence_gpu=$2
pipeline_tag=$3
stage1_resume_checkpoint=$4
TRAIN_SEED=${TRAIN_SEED:-0}
TRACE_VB_RECOVERY_PREFLIGHT_ONLY=${TRACE_VB_RECOVERY_PREFLIGHT_ONLY:-0}

[[ "${evidence_gpu}" =~ ^[0-9]+$ ]] || \
  trace_vb_die "invalid cache/evidence GPU: ${evidence_gpu}"
case ",${physical_gpus}," in
  *",${evidence_gpu},"*) ;;
  *) trace_vb_die "cache/evidence GPU must be one of the four training GPUs" ;;
esac
[[ "${TRAIN_SEED}" =~ ^[0-9]+$ ]] || \
  trace_vb_die "TRAIN_SEED must be non-negative"
[[ "${TRACE_VB_RECOVERY_PREFLIGHT_ONLY}" =~ ^[01]$ ]] || \
  trace_vb_die "TRACE_VB_RECOVERY_PREFLIGHT_ONLY must be 0 or 1"
[[ "${pipeline_tag}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  trace_vb_die "pipeline tag contains unsafe path or glob characters"
[[ -f "${stage1_resume_checkpoint}" ]] || \
  trace_vb_die "missing Stage-1 recovery checkpoint: ${stage1_resume_checkpoint}"

pipeline_dir=${TRACE_VB_ARTIFACT_ROOT}/pipelines/${pipeline_tag}
training_root=${TRACE_VB_ARTIFACT_ROOT}/training
evidence_root=${TRACE_VB_ARTIFACT_ROOT}/evidence
stage1_tag=${pipeline_tag}_stage1
stage2_tag=${pipeline_tag}_stage2
evidence_tag=${pipeline_tag}_evidence
stage1_dir=${training_root}/${stage1_tag}
stage2_dir=${training_root}/${stage2_tag}
evidence_dir=${evidence_root}/${evidence_tag}

if [[ "${TRACE_VB_RECOVERY_PREFLIGHT_ONLY}" == 1 ]]; then
  trace_vb_require_four_gpus "${physical_gpus}" 0
else
  trace_vb_require_four_gpus "${physical_gpus}"
fi
trace_vb_require_capability
trace_vb_require_stage0
trace_vb_require_cache
[[ -d "${pipeline_dir}" ]] || \
  trace_vb_die "missing interrupted pipeline directory: ${pipeline_dir}"
[[ -d "${stage1_dir}" ]] || \
  trace_vb_die "missing interrupted Stage-1 directory: ${stage1_dir}"
for forbidden in \
  best_checkpoint.txt \
  last_checkpoint.txt \
  validation_summaries; do
  [[ ! -e "${stage1_dir}/${forbidden}" ]] || \
    trace_vb_die "refusing recovery because Stage 1 already published ${forbidden}"
done
[[ ! -e "${stage2_dir}" ]] || \
  trace_vb_die "refusing recovery because Stage-2 output already exists: ${stage2_dir}"
[[ ! -e "${evidence_dir}" ]] || \
  trace_vb_die "refusing recovery because evidence output already exists: ${evidence_dir}"
[[ ! -e "${pipeline_dir}/COMPLETE.json" ]] || \
  trace_vb_die "pipeline is already complete"

for required in \
  manifest.txt \
  capability_parity_gate.json \
  student_initial_gate.json \
  validation_source_id_contract.json \
  data_contract_audit.json \
  unit_tests.log; do
  [[ -s "${pipeline_dir}/${required}" ]] || \
    trace_vb_die "interrupted pipeline is missing ${required}"
done

cd "${CODE_ROOT}"
for script in \
  scripts/trace_vb_common.sh \
  scripts/run_stage1_vb.sh \
  scripts/run_stage2_vb.sh \
  scripts/run_evidence_vb.sh \
  scripts/wait_for_four_gpus_and_run_full_vb.sh \
  scripts/resume_full_pipeline_vb.sh; do
  bash -n "${script}"
done

# Bind recovery to the original preflight results and to the exact Stage-1
# run encoded in the full-state checkpoint. This prevents using a healthy
# checkpoint from a different experiment merely because its tensor shapes fit.
env TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" - \
    "${pipeline_dir}" \
    "${pipeline_tag}" \
    "${stage1_tag}" \
    "${stage1_resume_checkpoint}" \
    "${TRAIN_SEED}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

from src.utils.safe_checkpoint import safe_load_checkpoint

pipeline_dir = Path(sys.argv[1]).resolve()
pipeline_tag = sys.argv[2]
stage1_tag = sys.argv[3]
checkpoint_path = Path(sys.argv[4]).resolve()
train_seed = int(sys.argv[5])

manifest = {}
for raw in (pipeline_dir / "manifest.txt").read_text(encoding="utf-8").splitlines():
    if "=" in raw:
        key, value = raw.split("=", 1)
        manifest[key] = value
if manifest.get("model") != "TRACE-VB-v7":
    raise SystemExit("interrupted pipeline manifest has the wrong model")
if manifest.get("pipeline_tag") != pipeline_tag:
    raise SystemExit("interrupted pipeline manifest has a different tag")
if int(manifest.get("train_seed", -1)) != train_seed:
    raise SystemExit("recovery seed differs from the interrupted pipeline")

capability = json.loads(
    (pipeline_dir / "capability_parity_gate.json").read_text(encoding="utf-8")
)
student = json.loads(
    (pipeline_dir / "student_initial_gate.json").read_text(encoding="utf-8")
)
if capability.get("status") != "PASS" or capability.get("validation_path") != (
    "capability_teacher_all_roles"
):
    raise SystemExit("original capability preflight did not pass")
if int(capability.get("correct_count", -1)) != 540:
    raise SystemExit("original capability preflight is not exact 540/747")
if int(capability.get("unique_questions", -1)) != 747:
    raise SystemExit("original capability preflight did not cover 747 questions")
if student.get("status") != "PASS" or student.get("validation_path") != (
    "student_commit"
):
    raise SystemExit("original q+COMMIT preflight did not pass")
student_correct = int(student.get("correct_count", -1))
student_accuracy = float(student.get("accuracy", float("nan")))
if student_correct < 449 or int(student.get("unique_questions", -1)) != 747:
    raise SystemExit("original q+COMMIT preflight violates its formal gate")
if not math.isfinite(student_accuracy) or not math.isclose(
    student_accuracy, student_correct / 747.0, abs_tol=1e-12, rel_tol=0.0
):
    raise SystemExit("original q+COMMIT preflight has inconsistent accuracy")

checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu")
if checkpoint_path.name != "last.ckpt":
    raise SystemExit("formal recovery requires a complete last.ckpt")
if checkpoint.get("trace_policy_training_stage") != 1:
    raise SystemExit("recovery checkpoint is not TRACE-VB Stage 1")
if checkpoint.get("trace_vb_schema_version") != "trace_vb_v7":
    raise SystemExit("recovery checkpoint has the wrong TRACE-VB schema")
config = checkpoint.get("hyper_parameters", {}).get("all_config")
if config is None or str(config.model.target) != "src.models.trace_vb.LitTRACEVB":
    raise SystemExit("recovery checkpoint has the wrong model config")
if str(config.args.log_suffix) != stage1_tag:
    raise SystemExit("recovery checkpoint belongs to a different Stage-1 tag")
if int(config.args.seed) != train_seed:
    raise SystemExit("recovery checkpoint seed differs from the formal seed")
fit_loop = checkpoint.get("loops", {}).get("fit_loop", {})
epoch_progress = fit_loop.get("epoch_progress", {}).get("total", {})
completed_epochs = int(epoch_progress.get("processed", -1))
if not 1 <= completed_epochs < 10:
    raise SystemExit(
        f"recovery checkpoint has {completed_epochs} completed epochs; expected 1..9"
    )
if int(checkpoint.get("global_step", -1)) <= 0:
    raise SystemExit("recovery checkpoint has no optimizer progress")
if len(checkpoint.get("optimizer_states", [])) != 1:
    raise SystemExit("recovery checkpoint does not contain one optimizer state")
if len(checkpoint.get("lr_schedulers", [])) != 1:
    raise SystemExit("recovery checkpoint does not contain one scheduler state")
monitor_states = [
    state
    for state in checkpoint.get("callbacks", {}).values()
    if state.get("monitor") == "monitor"
]
if len(monitor_states) != 1:
    raise SystemExit("recovery checkpoint has an ambiguous validation callback")
monitor_state = monitor_states[0]
canonical_best = Path(str(monitor_state.get("best_model_path", ""))).resolve()
if not canonical_best.is_file() or re.match(
    r"^epoch\d+__step\d+__monitor[-+0-9.eE]+\.ckpt$", canonical_best.name
) is None:
    raise SystemExit("recovery checkpoint does not reference a canonical validation best")
best_epoch = int(canonical_best.name.split("__", 1)[0][5:])
best_summary_path = canonical_best.parent.parent / f"validation_epoch_{best_epoch:03d}.json"
if not best_summary_path.is_file():
    raise SystemExit("canonical recovery best has no full-validation summary")
best_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
best_score = float(monitor_state.get("best_model_score", float("nan")))
if not math.isfinite(best_score) or not math.isclose(
    best_score,
    float(best_summary.get("accuracy", float("nan"))),
    abs_tol=5e-8,
    rel_tol=0.0,
):
    raise SystemExit("canonical recovery best disagrees with exact validation")
print(
    json.dumps(
        {
            "pipeline_tag": pipeline_tag,
            "stage1_tag": stage1_tag,
            "resume_checkpoint": str(checkpoint_path),
            "completed_epochs": completed_epochs,
            "global_step": int(checkpoint["global_step"]),
            "student_preflight_correct": student_correct,
            "canonical_validation_best": str(canonical_best),
            "canonical_validation_best_score": best_score,
        },
        indent=2,
    )
)
PY

# Re-run code and data checks under the repaired loader. The original reports
# remain immutable; these recovery-specific records document the code that
# actually continues optimization.
env TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" -m unittest discover -s tests \
  > "${pipeline_dir}/recovery_unit_tests.log" 2>&1
env TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" tools/data_contract_audit.py \
  > "${pipeline_dir}/recovery_data_contract_audit.json"
sha256sum \
  run.py \
  src/utils/safe_checkpoint.py \
  src/models/trace_vb.py \
  src/modules/trace_vb.py \
  src/modules/trace_policy.py \
  src/datasets/gsm8k_aug_nl.py \
  src/configs/models/trace_vb_policy_qwen3_instruct.yaml \
  src/configs/datasets/gsm8k_aug_nl.yaml \
  src/configs/trainer/trace_vb_stage1_v2.yaml \
  src/configs/trainer/default.yaml \
  scripts/trace_vb_common.sh \
  scripts/prepare_sufficiency_cache_vb.sh \
  scripts/run_preflight_validation_v7.sh \
  scripts/run_stage1_vb.sh \
  scripts/run_stage2_vb.sh \
  scripts/run_evidence_vb.sh \
  scripts/run_full_pipeline_vb.sh \
  scripts/wait_for_four_gpus_and_run_full_vb.sh \
  scripts/resume_full_pipeline_vb.sh \
  tools/build_trace_vb_sufficiency_cache.py \
  tools/trace_policy_task_summary.py \
  tools/trace_policy_geometry_summary.py \
  tools/trace_policy_stage_comparison.py \
  tools/trace_policy_causal_summary.py \
  tools/verify_evidence_complete.py \
  tests/test_trace_policy.py \
  tests/test_trace_vb_model_contract.py \
  tests/test_safe_checkpoint.py \
  > "${pipeline_dir}/recovery_source_sha256.txt"

"${TRACE_VB_PYTHON}" - \
  "${pipeline_dir}/source_sha256.txt" \
  "${pipeline_dir}/recovery_source_continuity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

original_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
allowed_changes = {
    "run.py": "explicit restricted-loader handoff before Lightning resume",
    "src/utils/safe_checkpoint.py": "persistent narrow safe-global registration",
    "scripts/run_stage1_vb.sh": "recovery provenance and canonical best publication",
    "tests/test_trace_vb_model_contract.py": "recovery contract regression coverage",
}
original = {}
for raw in original_path.read_text(encoding="utf-8").splitlines():
    digest, relative = raw.split(maxsplit=1)
    original[relative] = digest
changed = {}
unchanged = []
unexpected = []
for relative, old_digest in original.items():
    path = Path(relative)
    if not path.is_file():
        unexpected.append(f"missing:{relative}")
        continue
    new_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if new_digest == old_digest:
        unchanged.append(relative)
    else:
        changed[relative] = {
            "before": old_digest,
            "after": new_digest,
            "reason": allowed_changes.get(relative),
        }
        if relative not in allowed_changes:
            unexpected.append(f"unapproved-change:{relative}")
required_unchanged = {
    "src/models/trace_vb.py",
    "src/modules/trace_vb.py",
    "src/modules/trace_policy.py",
    "src/datasets/gsm8k_aug_nl.py",
    "src/configs/models/trace_vb_policy_qwen3_instruct.yaml",
    "src/configs/trainer/trace_vb_stage1_v2.yaml",
    "scripts/run_stage2_vb.sh",
    "scripts/run_evidence_vb.sh",
}
for relative in sorted(required_unchanged):
    if relative not in unchanged:
        unexpected.append(f"training-contract-drift:{relative}")
report = {
    "schema_version": "trace_vb_v7_recovery_source_continuity_v1",
    "status": "FAIL" if unexpected else "PASS",
    "original_source_manifest": str(original_path.resolve()),
    "unchanged_files": sorted(unchanged),
    "approved_changes": changed,
    "unexpected_changes": unexpected,
}
output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if unexpected:
    raise SystemExit("recovery source continuity failed: " + "; ".join(unexpected))
PY

if [[ "${TRACE_VB_RECOVERY_PREFLIGHT_ONLY}" == 1 ]]; then
  echo "TRACE-VB recovery preflight passed; no training was launched"
  exit 0
fi

cat >> "${pipeline_dir}/manifest.txt" <<EOF
recovery_started_at=$(date --iso-8601=seconds)
recovery_reason=safe_weights_only_allowlist_lifecycle_fixed
recovery_checkpoint=${stage1_resume_checkpoint}
recovery_checkpoint_sha256=$(sha256sum "${stage1_resume_checkpoint}" | awk '{print $1}')
recovery_protocol=skip_completed_epochs_then_original_full_validation_and_stage_gates
recovery_source_sha256=${pipeline_dir}/recovery_source_sha256.txt
recovery_unit_tests=${pipeline_dir}/recovery_unit_tests.log
recovery_data_contract_audit=${pipeline_dir}/recovery_data_contract_audit.json
recovery_source_continuity=${pipeline_dir}/recovery_source_continuity.json
EOF

RUN_TAG="${stage1_tag}" TRAIN_SEED="${TRAIN_SEED}" \
  STAGE1_RESUME_CKPT="${stage1_resume_checkpoint}" \
  bash "${SCRIPT_DIR}/run_stage1_vb.sh" \
    "${physical_gpus}" "${TRACE_VB_REGISTERED_CAPABILITY}"

stage1_record=${stage1_dir}/best_checkpoint.txt
[[ -s "${stage1_record}" ]] || \
  trace_vb_die "Stage 1 recovery did not publish a validation-selected checkpoint"
stage1_checkpoint=$(<"${stage1_record}")
[[ -f "${stage1_checkpoint}" ]] || \
  trace_vb_die "Stage-1 best-checkpoint record points to a missing file"

# Stage 2 independently revalidates all ten Stage-1 747-question summaries,
# exact-count threshold, and behavior metrics before any RL optimizer step.
RUN_TAG="${stage2_tag}" TRAIN_SEED="${TRAIN_SEED}" \
  bash "${SCRIPT_DIR}/run_stage2_vb.sh" \
    "${physical_gpus}" "${stage1_checkpoint}"
stage2_record=${stage2_dir}/best_checkpoint.txt
[[ -s "${stage2_record}" ]] || \
  trace_vb_die "Stage 2 did not publish a validation-selected checkpoint"
stage2_checkpoint=$(<"${stage2_record}")
[[ -f "${stage2_checkpoint}" ]] || \
  trace_vb_die "Stage-2 best-checkpoint record points to a missing file"

cp "${stage2_dir}/stage1_validation_gate.json" \
  "${pipeline_dir}/stage1_validation_gate.json"
cp "${stage2_dir}/validation_contract.json" \
  "${pipeline_dir}/stage2_validation_contract.json"

RUN_TAG="${evidence_tag}" OUT="${evidence_dir}" \
  bash "${SCRIPT_DIR}/run_evidence_vb.sh" \
    "${evidence_gpu}" "${stage1_checkpoint}" "${stage2_checkpoint}"

complete_record=${evidence_dir}/COMPLETE.json
strict_validation_summary=${evidence_dir}/validation747_summary.json
[[ -s "${complete_record}" ]] || \
  trace_vb_die "complete evidence gate did not produce COMPLETE.json"
[[ -s "${strict_validation_summary}" ]] || \
  trace_vb_die "evidence did not publish paired 747 source-id validation"
env TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  "${TRACE_VB_PYTHON}" tools/verify_evidence_complete.py \
    --evidence-root "${evidence_dir}" \
    --stage1-checkpoint "${stage1_checkpoint}" \
    --final-checkpoint "${stage2_checkpoint}" \
    > "${pipeline_dir}/evidence_completeness_recheck.json"
cp "${complete_record}" "${pipeline_dir}/COMPLETE.json"
cp "${strict_validation_summary}" \
  "${pipeline_dir}/validation747_summary.json"

cat >> "${pipeline_dir}/manifest.txt" <<EOF
stage1_checkpoint=${stage1_checkpoint}
stage1_checkpoint_sha256=$(sha256sum "${stage1_checkpoint}" | awk '{print $1}')
stage1_gate_report=${pipeline_dir}/stage1_validation_gate.json
stage2_checkpoint=${stage2_checkpoint}
stage2_checkpoint_sha256=$(sha256sum "${stage2_checkpoint}" | awk '{print $1}')
stage2_validation_contract=${pipeline_dir}/stage2_validation_contract.json
strict_paired_validation=${pipeline_dir}/validation747_summary.json
evidence_dir=${evidence_dir}
complete_record=${pipeline_dir}/COMPLETE.json
recovery_finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${TRACE_VB_REGISTERED_CAPABILITY}" \
  > "${pipeline_dir}/capability_checkpoint.txt"
printf '%s\n' "${stage1_checkpoint}" > "${pipeline_dir}/stage1_best.txt"
printf '%s\n' "${stage2_checkpoint}" > "${pipeline_dir}/stage2_best.txt"
