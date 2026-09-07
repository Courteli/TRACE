#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"
EVIDENCE_ROOT=${TRACE_VB_ARTIFACT_ROOT}/evidence
TMP_ROOT=${TRACE_VB_ARTIFACT_ROOT}/tmp
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
TRACE_VB_SAFE_RUNNER='import runpy, sys; import src.utils.safe_checkpoint; script = sys.argv[1]; sys.argv = sys.argv[1:]; runpy.run_path(script, run_name="__main__")'
REGISTERED_DATASET_DIR=${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL
REGISTERED_VALIDATION_FILE=${REGISTERED_DATASET_DIR}/gsm8k_val_processed.jsonl

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <physical-gpu> <stage1-best-checkpoint> <stage2-best-checkpoint>" >&2
  exit 2
fi
physical_gpu=$1
stage1_checkpoint=$2
stage2_checkpoint=$3
[[ "${physical_gpu}" =~ ^[0-9]+$ ]] || trace_vb_die "invalid evidence GPU: ${physical_gpu}"

validate_checkpoint() {
  local phase=$1 checkpoint=$2 expected_rl=$3
  [[ -f "${checkpoint}" ]] || trace_vb_die "missing ${phase} checkpoint: ${checkpoint}"
  local hparams
  hparams="$(dirname "$(dirname "${checkpoint}")")/hparams.yaml"
  [[ -f "${hparams}" ]] || trace_vb_die "missing ${phase} hparams: ${hparams}"
  rg -q "src.models.trace_vb.LitTRACEVB" "${hparams}" || \
    trace_vb_die "${phase} is not a TRACE-VB policy checkpoint"
  rg -Fq "workspace_path: ${CODE_ROOT}" "${hparams}" || \
    trace_vb_die "${phase} was not produced by the isolated code root"
  rg -qi "do_trace_rl: ${expected_rl}" "${hparams}" || \
    trace_vb_die "${phase} has the wrong training phase"
  rg -Fq "answer_context_mode: question_and_commit" "${hparams}" || \
    trace_vb_die "${phase} does not preserve the question+COMMIT bridge"
  "${TRACE_VB_PYTHON}" - \
    "${checkpoint}" \
    "${expected_rl}" \
    "${REGISTERED_DATASET_DIR}" <<'PY'
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from src.utils.safe_checkpoint import safe_load_checkpoint

checkpoint_path = Path(sys.argv[1]).resolve()
checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu")
expected_stage = 2 if sys.argv[2].lower() == "true" else 1
if int(checkpoint.get("trace_policy_training_stage", -1)) != expected_stage:
    raise SystemExit(f"checkpoint phase marker is not Stage {expected_stage}")
if checkpoint.get("trace_vb_schema_version") != "trace_vb_v7":
    raise SystemExit("evidence requires a trace_vb_v7 checkpoint")
expected_objective = {
    1: "capability_preserving_role_semantic_cot_sft",
    2: "capability_anchored_role_local_latent_rl",
}[expected_stage]
if checkpoint.get("trace_vb_stage2_objective") != expected_objective:
    raise SystemExit(
        f"Stage {expected_stage} objective marker is not {expected_objective}"
    )
keys = tuple(checkpoint.get("state_dict", {}))
for fragment in (
    "trajectory_policy.",
    "trajectory_posterior.",
    "posterior_context_norm.",
    "plan_forecaster",
    "solve_text_decoder.",
    "sufficiency_head.",
    "capability_state_norm.weight",
    "capability_state_norm.bias",
    "capability_latent_bridge.0.weight",
    "capability_latent_bridge.2.weight",
    "capability_latent_queries",
    ".trace_capability.",
    ".trace_cot_encoder.",
):
    if not any(fragment in key for key in keys):
        raise SystemExit(f"checkpoint is missing {fragment}")

all_config = checkpoint.get("hyper_parameters", {}).get("all_config")
if all_config is None:
    raise SystemExit("checkpoint is missing its embedded configuration")
if str(all_config.model.target) != "src.models.trace_vb.LitTRACEVB":
    raise SystemExit("checkpoint does not instantiate LitTRACEVB")
model_kwargs = all_config.model.model_kwargs
if bool(model_kwargs.do_trace_rl) != (expected_stage == 2):
    raise SystemExit("embedded configuration has the wrong training phase")
policy = model_kwargs.trace_policy_config
if str(policy.answer_context_mode) != "question_and_commit":
    raise SystemExit("embedded configuration changed the question+COMMIT bridge")
data = all_config.data_module
if Path(str(data.dataset_dir)).resolve() != Path(sys.argv[3]).resolve():
    raise SystemExit("checkpoint used a different dataset directory")
if str(data.val_file) != "gsm8k_val_processed.jsonl":
    raise SystemExit("checkpoint used a different validation file")
if not bool(data.enforce_registered_source) or bool(data.tiny_dataset):
    raise SystemExit("checkpoint bypassed the registered full-data contract")

if expected_stage == 2:
    rl = model_kwargs.trace_rl_config
    if not bool(rl.use_trajectory_policy_loss):
        raise SystemExit("Stage 2 did not optimize the latent policy")
    if bool(rl.use_answer_policy_loss):
        raise SystemExit("Stage 2 unexpectedly optimized answer-token actions")
    if not bool(rl.use_terminal_exact_reward):
        raise SystemExit("Stage 2 changed the terminal exact-answer objective")
    if not bool(rl.use_evidence_gated_group_rl):
        raise SystemExit("Stage 2 changed the evidence-gated group objective")
    if float(rl.step_reward_weight) <= 0.0:
        raise SystemExit("Stage 2 has no positive role-local process signal")
    reference = checkpoint.get("trace_stage1_policy_reference")
    if not isinstance(reference, Mapping) or not reference:
        raise SystemExit("Stage-2 checkpoint is missing its immutable Stage-1 prior")
    for prefix in (
        "policy_step_embedding.",
        "policy_trunk.",
        "mean_heads.plan.",
        "mean_heads.solve.",
        "mean_heads.check.",
        "log_std_heads.plan.",
        "log_std_heads.solve.",
        "log_std_heads.check.",
    ):
        if not any(str(name).startswith(prefix) for name in reference):
            raise SystemExit(f"Stage-1 policy prior is missing {prefix}")
else:
    if "trace_stage1_policy_reference" in checkpoint:
        raise SystemExit("Stage-1 checkpoint unexpectedly publishes a policy prior")

match = re.fullmatch(
    r"epoch(?P<epoch>\d+)__step\d+__monitor(?P<monitor>[-+0-9.eE]+)\.ckpt",
    checkpoint_path.name,
)
if match is None:
    raise SystemExit("evidence requires a validation-selected checkpoint filename")
report_path = (
    checkpoint_path.parent.parent
    / f"validation_epoch_{int(match.group('epoch')):03d}.json"
)
if not report_path.is_file():
    raise SystemExit(f"missing full-validation report: {report_path}")
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("schema_version") != "trace_vb_v7_validation_behavior_v1":
    raise SystemExit("checkpoint validation report has the wrong schema")
if int(report.get("world_size", -1)) != 4:
    raise SystemExit("checkpoint was not selected by four-rank validation")
if int(report.get("unique_questions", -1)) != 747:
    raise SystemExit("checkpoint was not selected on exactly 747 unique questions")
correct = int(report.get("correct_count", -1))
accuracy = float(report.get("accuracy", float("nan")))
if not 0 <= correct <= 747 or not math.isclose(
    accuracy,
    correct / 747.0,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit("validation accuracy is inconsistent with correct_count/747")
if not math.isclose(
    float(match.group("monitor")),
    accuracy,
    rel_tol=0.0,
    abs_tol=5.1e-7,
):
    raise SystemExit("checkpoint monitor does not match its strict validation report")
PY
}
validate_checkpoint "Stage 1" "${stage1_checkpoint}" false
validate_checkpoint "Stage 2" "${stage2_checkpoint}" true

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_complete_evidence}
OUT=${OUT:-${EVIDENCE_ROOT}/${RUN_TAG}}
mkdir -p "${OUT}" "${TMP_ROOT}"
cd "${CODE_ROOT}"

TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
  tools/data_contract_audit.py > "${OUT}/registered_data_contract.json"
"${TRACE_VB_PYTHON}" - \
  "${REGISTERED_VALIDATION_FILE}" \
  "${OUT}/validation_source_id_contract.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
rows = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 747:
    raise SystemExit(f"registered validation has {len(rows)} rows, expected 747")
source_ids = []
for index, row in enumerate(rows):
    if "id" not in row:
        raise SystemExit(f"registered validation row {index} has no source id")
    source_ids.append(int(row["id"]))
if len(set(source_ids)) != 747:
    raise SystemExit("registered validation source ids are not unique")
payload = {
    "schema_version": "trace_vb_v7_validation_source_id_contract_v1",
    "path": str(source),
    "rows": 747,
    "unique_source_ids": 747,
    "source_id_sha256": hashlib.sha256(
        json.dumps(source_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest(),
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

run_test() {
  local name=$1 checkpoint=$2
  shift 2
  mkdir -p "${OUT}/${name}"
  env \
    TRACE_PROJECT_ROOT="${CODE_ROOT}" \
    TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    "${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" run.py \
      --model "${TRACE_VB_MODEL_CONFIG}" \
      --dataset gsm8k_aug_nl \
      --trainer default \
      --devices 0 \
      --workspace_path "${CODE_ROOT}" \
      --test_ckpt_path "${checkpoint}" \
      --test_times 1 \
      --seed 0 \
      trainer.logger.save_dir="${OUT}" \
      trainer.logger.name="${name}" \
      trainer.logger.version=run \
      val_batch_size=1 \
      num_workers=4 \
      persistent_workers=false \
      data_module.dataset_dir="${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL" \
      data_module.enforce_registered_source=true \
      "$@" \
      2>&1 | tee "${OUT}/${name}/test.log"
}

cat > "${OUT}/manifest.txt" <<EOF
model=TRACE-VB-v7
checkpoint_schema=trace_vb_v7
model_config=${TRACE_VB_MODEL_CONFIG}
stage1_checkpoint=${stage1_checkpoint}
stage2_checkpoint=${stage2_checkpoint}
physical_gpu=${physical_gpu}
test_times=1
validation_questions=747
validation_source_id_contract=strict_unique_and_paired
validation_checkpoint_contract=four_rank_full_747
gsm8k_questions=1319
geometry_questions=200
geometry_rollouts_per_question=8
geometry_rollout_schema=iid_role_conditioned_gaussian_with_deterministic_COMMIT
projection_contract=global_unlabeled_PCA_no_manual_offsets_no_path_rescaling
causal_bootstrap_draws=10000
all_transition_interval=familywise_95_percent_Bonferroni_bootstrap
causal_primary=equal_norm_single_transition_replacement_with_suffix_and_COMMIT_recomputation
causal_co_primary=terminal_outcome_value_calibration
prefix_curve_interpretation=COMMIT_bottleneck_sanity_only_not_stepwise_contribution
ood_datasets=GSMHard,SVAMP,MultiArith
comparison=paired_stage1_vs_stage2
stage2_optimization=actor_only_role_local_policy_refinement
stage2_trainable_policy=step_embedding_plus_shared_role_trunk_plus_PLAN_SOLVE_REFINE_heads
stage2_terminal_objective=exact_answer_correctness
stage2_process_signal=discounted_role_local_CoT_semantic_agreement
total_L_definition=8_latent_states_plus_all_generated_answer_tokens
gsm8k_source=${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL
generated_cots=false
latent_roles=PLAN,SOLVE1,SOLVE2,SOLVE3,SOLVE4,SOLVE5,REFINE,COMMIT
record_role_schema=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT
answer_context=question_plus_COMMIT_latent_only
answer_decoder_question_attention=true
private_latent_answer_access=false
deployment_policy_CoT_conditioning=false
posthoc_text_CoT_usage=semantic_alignment_and_frozen_decoder_audit_only
deployment_solve_text_decoder=false
started_at=$(date --iso-8601=seconds)
EOF

"${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
  tools/trace_prepare_pca_fit_set.py \
  --source-file "${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL/gsm8k_train_processed.jsonl" \
  --output-dir "${OUT}/pca_fit_dataset" \
  --count 200

run_evidence_suite() {
  local prefix=$1 checkpoint=$2 rl_mode=$3
  local phase_override="model.model_kwargs.do_trace_rl=${rl_mode}"

  # Re-evaluate the already selected checkpoint once on the registered 747.
  # The paired summary below uses the registered source id as its key.
  run_test "${prefix}_gsm8k_val747" "${checkpoint}" \
    "${phase_override}" \
    data_module.test_file=gsm8k_val_processed.jsonl \
    model.model_kwargs.trace_policy_config.visual_record_limit=0

  run_test "${prefix}_pca_fit_train200" "${checkpoint}" \
    "${phase_override}" \
    data_module.dataset_dir="${OUT}/pca_fit_dataset" \
    data_module.enforce_registered_source=false \
    data_module.test_file=gsm8k_test_processed.jsonl \
    model.model_kwargs.trace_policy_config.visual_record_limit=200 \
    model.model_kwargs.trace_policy_config.visual_group_size=8 \
    model.model_kwargs.trace_policy_config.visual_seed=314159

  # Full 1,319-question IID test; visual_record_limit limits only the paired
  # geometry cache, not evaluation rows.
  run_test "${prefix}_gsm8k_geometry200" "${checkpoint}" \
    "${phase_override}" \
    model.model_kwargs.trace_policy_config.visual_record_limit=200 \
    model.model_kwargs.trace_policy_config.visual_group_size=8 \
    model.model_kwargs.trace_policy_config.visual_seed=271828

  run_test "${prefix}_gsmhard" "${checkpoint}" \
    "${phase_override}" \
    data_module.target=src.datasets.gsm8k_aug_nl.GSM8KAugNLDataModule \
    data_module.dataset_name=gsmhard \
    data_module.dataset_dir="${TRACE_VB_DATA_ROOT}/data/raw/GSM8K-Hard" \
    data_module.train_file=gsmhard_test_processed.jsonl \
    data_module.val_file=gsmhard_test_processed.jsonl \
    data_module.test_file=gsmhard_test_processed.jsonl \
    data_module.enforce_registered_source=false \
    model.model_kwargs.trace_policy_config.visual_record_limit=0

  run_test "${prefix}_svamp" "${checkpoint}" \
    "${phase_override}" \
    data_module.target=src.datasets.gsm8k_aug_nl.GSM8KAugNLDataModule \
    data_module.dataset_name=svamp \
    data_module.dataset_dir="${TRACE_VB_DATA_ROOT}/data/raw/SVAMP" \
    data_module.train_file=svamp_test_processed.jsonl \
    data_module.val_file=svamp_test_processed.jsonl \
    data_module.test_file=svamp_test_processed.jsonl \
    data_module.enforce_registered_source=false \
    model.model_kwargs.trace_policy_config.visual_record_limit=0

  run_test "${prefix}_multiarith" "${checkpoint}" \
    "${phase_override}" \
    data_module.target=src.datasets.gsm8k_aug_nl.GSM8KAugNLDataModule \
    data_module.dataset_name=multiarith \
    data_module.dataset_dir="${TRACE_VB_DATA_ROOT}/data/raw/MultiArith" \
    data_module.train_file=multiarith_test_processed.jsonl \
    data_module.val_file=multiarith_test_processed.jsonl \
    data_module.test_file=multiarith_test_processed.jsonl \
    data_module.enforce_registered_source=false \
    model.model_kwargs.trace_policy_config.visual_record_limit=0
}

run_evidence_suite stage1 "${stage1_checkpoint}" false
run_evidence_suite final "${stage2_checkpoint}" true

"${TRACE_VB_PYTHON}" - \
  "${REGISTERED_VALIDATION_FILE}" \
  "${OUT}/stage1_gsm8k_val747" \
  "${OUT}/final_gsm8k_val747" \
  "${stage1_checkpoint}" \
  "${stage2_checkpoint}" \
  "${OUT}/validation747_summary.json" <<'PY'
import json
import sys
from pathlib import Path

source_file = Path(sys.argv[1]).resolve()
stage1_dir = Path(sys.argv[2]).resolve()
final_dir = Path(sys.argv[3]).resolve()
stage1_checkpoint = Path(sys.argv[4]).resolve()
final_checkpoint = Path(sys.argv[5]).resolve()
output = Path(sys.argv[6]).resolve()
source_rows = [
    json.loads(line)
    for line in source_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(source_rows) != 747:
    raise SystemExit("strict validation source no longer contains 747 rows")
source_ids = [int(row["id"]) for row in source_rows]
if len(set(source_ids)) != 747:
    raise SystemExit("strict validation source ids are not unique")

def scalar(value, *, label):
    if not isinstance(value, list) or len(value) != 1:
        raise SystemExit(f"{label} was not evaluated exactly once")
    return float(value[0])

def load_phase(directory, checkpoint):
    files = sorted(directory.rglob("test_*.json"))
    if len(files) != 1:
        raise SystemExit(
            f"{directory} must contain one test JSON, found {len(files)}"
        )
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    metadata = payload.get("test_metadata", {})
    if int(metadata.get("test_times", -1)) != 1:
        raise SystemExit(f"{files[0]} changed the single-evaluation contract")
    if Path(str(metadata.get("ckpt_path", ""))).resolve() != checkpoint:
        raise SystemExit(f"{files[0]} used the wrong checkpoint")
    data = metadata.get("data_module", {})
    if str(data.get("test_file")) != "gsm8k_val_processed.jsonl":
        raise SystemExit(f"{files[0]} did not evaluate the registered validation")
    by_source_id = {}
    for key, record in payload.items():
        if not str(key).isdigit():
            continue
        dataset_index = int(key)
        if not 0 <= dataset_index < 747:
            raise SystemExit(f"{files[0]} has invalid dataset index {dataset_index}")
        source_id = source_ids[dataset_index]
        if source_id in by_source_id:
            raise SystemExit(f"{files[0]} repeats source id {source_id}")
        by_source_id[source_id] = scalar(
            record["acc"],
            label=f"{files[0]} source id {source_id}",
        )
    if len(by_source_id) != 747 or set(by_source_id) != set(source_ids):
        raise SystemExit(f"{files[0]} is not complete after source-id deduplication")
    return by_source_id, files[0]

stage1, stage1_file = load_phase(stage1_dir, stage1_checkpoint)
final, final_file = load_phase(final_dir, final_checkpoint)
if set(stage1) != set(final):
    raise SystemExit("strict Stage-1 and final validations are not source-id paired")
ordered = sorted(stage1)
stage1_correct = sum(int(stage1[key] > 0.5) for key in ordered)
final_correct = sum(int(final[key] > 0.5) for key in ordered)
rescued = sum(
    int(stage1[key] <= 0.5 and final[key] > 0.5) for key in ordered
)
regressed = sum(
    int(stage1[key] > 0.5 and final[key] <= 0.5) for key in ordered
)
report = {
    "schema_version": "trace_vb_v7_source_id_paired_validation_v1",
    "questions": 747,
    "deduplication_key": "source_id",
    "test_times": 1,
    "stage1_checkpoint": str(stage1_checkpoint),
    "final_checkpoint": str(final_checkpoint),
    "stage1_test_json": str(stage1_file),
    "final_test_json": str(final_file),
    "stage1_correct": stage1_correct,
    "stage1_accuracy": stage1_correct / 747.0,
    "final_correct": final_correct,
    "final_accuracy": final_correct / 747.0,
    "rescued": rescued,
    "regressed": regressed,
}
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY

"${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
  tools/trace_policy_task_summary.py \
  --evidence-root "${OUT}" \
  --stage1-checkpoint "${stage1_checkpoint}" \
  --final-checkpoint "${stage2_checkpoint}" \
  --output-dir "${OUT}/task_summary"

stage1_fit_records="${OUT}/stage1_pca_fit_train200/run/trace_policy_visual_test.pt"
final_fit_records="${OUT}/final_pca_fit_train200/run/trace_policy_visual_test.pt"
[[ -f "${stage1_fit_records}" && -f "${final_fit_records}" ]] || \
  trace_vb_die "missing one or both shared-PCA fit caches"
for prefix in stage1 final; do
  records="${OUT}/${prefix}_gsm8k_geometry200/run/trace_policy_visual_test.pt"
  [[ -f "${records}" ]] || trace_vb_die "missing ${prefix} geometry cache"
  "${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
    tools/trace_policy_geometry_summary.py \
    --fit-records "${stage1_fit_records}" \
    --additional-fit-records "${final_fit_records}" \
    --records "${records}" \
    --output-dir "${OUT}/${prefix}_geometry_summary"
done

stage1_pca="${OUT}/stage1_geometry_summary/global_train_fit_pca.pt"
final_pca="${OUT}/final_geometry_summary/global_train_fit_pca.pt"
"${TRACE_VB_PYTHON}" - "${stage1_pca}" "${final_pca}" <<'PY'
import sys
import torch
left = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
right = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
for key in ("mean", "components", "explained_ratio"):
    if not torch.allclose(left[key], right[key], atol=1e-7, rtol=1e-6):
        raise SystemExit("Stage-1 and Stage-2 geometry used different PCA bases")
PY

stage1_records="${OUT}/stage1_gsm8k_geometry200/run/trace_policy_visual_test.pt"
final_records="${OUT}/final_gsm8k_geometry200/run/trace_policy_visual_test.pt"
"${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
  tools/trace_policy_stage_comparison.py \
  --stage1-records "${stage1_records}" \
  --final-records "${final_records}" \
  --stage1-geometry "${OUT}/stage1_geometry_summary/question_geometry.csv" \
  --final-geometry "${OUT}/final_geometry_summary/question_geometry.csv" \
  --shared-pca "${stage1_pca}" \
  --output-dir "${OUT}/stage_comparison"

mkdir -p "${OUT}/causal_summary"
env \
  TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  "${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
    tools/trace_policy_causal_summary.py \
    --checkpoint "${stage2_checkpoint}" \
    --records "${final_records}" \
    --output-dir "${OUT}/causal_summary" \
    --device cuda:0 \
    --count 200 \
    --bootstrap 10000 \
    2>&1 | tee "${OUT}/causal_summary/run.log"

cat >> "${OUT}/manifest.txt" <<EOF
stage1_records=${stage1_records}
final_records=${final_records}
shared_pca_stage1_fit_records=${stage1_fit_records}
shared_pca_final_fit_records=${final_fit_records}
shared_pca_state=${stage1_pca}
stage1_geometry_summary=${OUT}/stage1_geometry_summary
final_geometry_summary=${OUT}/final_geometry_summary
paired_stage_comparison=${OUT}/stage_comparison
task_summary=${OUT}/task_summary
strict_validation_summary=${OUT}/validation747_summary.json
causal_summary=${OUT}/causal_summary
finished_at=$(date --iso-8601=seconds)
EOF

"${TRACE_VB_PYTHON}" -c "${TRACE_VB_SAFE_RUNNER}" \
  tools/verify_evidence_complete.py \
  --evidence-root "${OUT}" \
  --stage1-checkpoint "${stage1_checkpoint}" \
  --final-checkpoint "${stage2_checkpoint}" \
  --write-complete \
  > "${OUT}/completeness_gate.json"
