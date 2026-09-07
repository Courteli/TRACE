#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage1-best-checkpoint> <stage1-candidate-index>" >&2
  exit 2
fi
physical_gpus=$1
stage1_checkpoint=$2
stage1_candidate_index=$3
TRAIN_SEED=${TRAIN_SEED:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v8_stage2_seed${TRAIN_SEED}}
STAGE2_RESUME_CKPT=${STAGE2_RESUME_CKPT:-}
trace_vb_require_safe_tag "${RUN_TAG}"
[[ "${TRAIN_SEED}" =~ ^[0-9]+$ ]] || trace_vb_die "TRAIN_SEED must be non-negative"

CODE_ROOT=${TRACE_VB_CODE_ROOT}
DATASET_DIR=${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL
LOG_ROOT=${TRACE_VB_ARTIFACT_ROOT}/logs
TMP_ROOT=${TRACE_VB_ARTIFACT_ROOT}/tmp
OUT_DIR=${TRACE_VB_ARTIFACT_ROOT}/training/${RUN_TAG}
LOG_PARENT=${LOG_ROOT}/${TRACE_VB_MODEL_CONFIG}/gsm8k_aug_nl-gsm8k_aug_nl
CANDIDATE_DIR=${OUT_DIR}/candidates
STAGE1_RUN_DIR=$(dirname "$(readlink -f "${stage1_candidate_index}")")
METRIC_SAFE_BASELINE_ARTIFACT=${STAGE1_RUN_DIR}/metric_safe_baseline.json
REGISTERED_CAPABILITY_VALIDATION_ARTIFACT=${STAGE1_RUN_DIR}/registered_capability_validation.json
ROLLOUT_BATCHES_PER_INTERVAL=256
STAGE2_RECOVERY_CHECKPOINT_INTERVAL=64
QUESTIONS_PER_INTERVAL=1024
FORMAL_INTERVALS=4
POLICY_UPDATES_PER_ROLLOUT=2
TOTAL_OPTIMIZER_STEPS=2048
EXPECTED_ROLLOUT_BATCHES=0,256,512,768,1024
MINIMUM_STAGE1_CORRECT=527

trace_vb_require_four_gpus "${physical_gpus}"
trace_vb_require_capability
trace_vb_require_metric_artifacts
trace_vb_require_cot_encoder
[[ -f "${stage1_checkpoint}" ]] || \
  trace_vb_die "missing Stage-1 selected checkpoint: ${stage1_checkpoint}"
[[ -s "${stage1_candidate_index}" ]] || \
  trace_vb_die "missing Stage-1 candidate index: ${stage1_candidate_index}"
trace_vb_require_sha256 \
  "Stage-1 run-local metric-safe baseline" \
  "${METRIC_SAFE_BASELINE_ARTIFACT}" \
  "${TRACE_VB_METRIC_SAFE_BASELINE_SHA256}"
trace_vb_require_sha256 \
  "Stage-1 run-local registered capability validation" \
  "${REGISTERED_CAPABILITY_VALIDATION_ARTIFACT}" \
  "${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_SHA256}"
[[ -z "${RESUME_CKPT_PATH:-}" ]] || \
  trace_vb_die "use STAGE2_RESUME_CKPT; generic RESUME_CKPT_PATH is forbidden"

cd "${CODE_ROOT}"

V7_SOURCE_SHA256=$("${TRACE_VB_PYTHON}" - "${stage1_checkpoint}" <<'PY'
import sys
from src.utils.safe_checkpoint import safe_load_checkpoint
checkpoint = safe_load_checkpoint(sys.argv[1], map_location="cpu")
value = str(checkpoint.get("trace_vb_v7_source_checkpoint_sha256", "")).strip().lower()
if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
    raise SystemExit("Stage-1 checkpoint has no valid v7 source SHA256")
print(value)
PY
)

# Verify v8 Stage-1 origin and apply the immutable 527/747 metric-safe behavior
# safety floor without blocking the previously demonstrated RL improvement.
"${TRACE_VB_PYTHON}" - \
  "${stage1_checkpoint}" \
  "${stage1_candidate_index}" \
  "${MINIMUM_STAGE1_CORRECT}" \
  "${physical_gpus}" \
  "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}" \
  "${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}" \
  "${TRACE_VB_COT_ENCODER_SHA256}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

from src.utils.safe_checkpoint import safe_load_checkpoint
sys.path.insert(0, str(Path.cwd() / "scripts"))
from trace_vb_checkpoint_contract import checkpoint_provenance

checkpoint_path = Path(sys.argv[1]).resolve()
index_path = Path(sys.argv[2]).resolve()
minimum = int(sys.argv[3])
physical_gpus = sys.argv[4]
registered_sha = sys.argv[5]
payload_sha = sys.argv[6]
cot_sha = sys.argv[7]
checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu")
if checkpoint.get("trace_vb_schema_version") != "trace_vb_v8":
    raise SystemExit("Stage 2 requires a TRACE-VB-v8 checkpoint")
if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
    raise SystemExit("Stage 2 requires a v8 Stage-1 checkpoint")
v7_source_sha = str(
    checkpoint.get("trace_vb_v7_source_checkpoint_sha256", "")
).strip().lower()
try:
    checkpoint_provenance(
        checkpoint,
        expected_registered_checkpoint_sha256=registered_sha,
        expected_registered_payload_sha256=payload_sha,
        expected_v7_source_sha256=v7_source_sha,
        expected_cot_encoder_sha256=cot_sha,
    )
except RuntimeError as error:
    raise SystemExit(f"Stage-1 checkpoint provenance failed: {error}") from error
index = json.loads(index_path.read_text(encoding="utf-8"))
if index.get("schema_version") != "trace_vb_v8_stage1_candidate_index_v1":
    raise SystemExit("Stage-1 candidate index has the wrong schema")
if index.get("physical_gpus") != physical_gpus:
    raise SystemExit("Stage-1 candidate index uses a different physical GPU set")
if Path(index.get("selected_checkpoint", "")).resolve() != checkpoint_path:
    raise SystemExit("Stage-1 checkpoint is not the exact selected candidate")
if int(index.get("candidate_count", -1)) != 5:
    raise SystemExit("Stage-1 index must contain step 0 plus four intervals")
if index.get("expected_steps") != [0, 512, 1024, 1536, 2048]:
    raise SystemExit("Stage-1 index does not cover the five formal boundaries")
candidates = index.get("candidates")
if not isinstance(candidates, list) or len(candidates) != 5:
    raise SystemExit("Stage-1 index candidate payload is incomplete")
step0_matches = [item for item in candidates if item.get("step") == 0]
if len(step0_matches) != 1:
    raise SystemExit("Stage-1 index has no unique step-0 candidate")
if type(step0_matches[0].get("correct_count")) is not int or (
    step0_matches[0]["correct_count"] < minimum
):
    raise SystemExit(
        f"Stage-1 step-0 safety floor failed: "
        f"{step0_matches[0].get('correct_count')}<{minimum}"
    )
selected_step = int(index.get("selected_step", -1))
matches = [item for item in candidates if int(item["step"]) == selected_step]
if len(matches) != 1:
    raise SystemExit("cannot identify the selected Stage-1 candidate")
selected = matches[0]
correct = int(selected.get("correct_count", -1))
accuracy = float(selected.get("accuracy", float("nan")))
if (
    int(index.get("selected_correct_count", -1)) != correct
    or not math.isclose(
        float(index.get("selected_accuracy", float("nan"))),
        accuracy,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    or Path(index.get("selected_checkpoint", "")).resolve()
    != checkpoint_path
):
    raise SystemExit("Stage-1 selected top-level facts differ from its record")
if not math.isfinite(accuracy) or not math.isclose(
    accuracy, correct / 747, rel_tol=0.0, abs_tol=1e-12
):
    raise SystemExit("Stage-1 selected accuracy is inconsistent")
expected_checkpoint_sha = str(selected.get("checkpoint_sha256", ""))
digest = hashlib.sha256()
with checkpoint_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
actual_checkpoint_sha = digest.hexdigest()
if actual_checkpoint_sha != expected_checkpoint_sha:
    raise SystemExit("Stage-1 selected checkpoint SHA256 is inconsistent")
behavior = selected.get("behavior", {})
failures = []
if correct < minimum:
    failures.append(f"correct_count={correct}<{minimum}")
thresholds = {
    "valid_answer_fraction": (0.98, "min"),
    "unique_prediction_ratio": (0.20, "min"),
    "top1_mode_fraction": (0.20, "max"),
    "nonempty_output_fraction": (0.98, "min"),
}
for key, (threshold, direction) in thresholds.items():
    value = float(behavior.get(key, float("nan")))
    if not math.isfinite(value):
        failures.append(f"{key} is non-finite")
    elif direction == "min" and value < threshold:
        failures.append(f"{key}={value}<{threshold}")
    elif direction == "max" and value > threshold:
        failures.append(f"{key}={value}>{threshold}")
if failures:
    raise SystemExit("Stage-1 behavior safety gate failed: " + "; ".join(failures))
PY

# Replay the complete Stage-1 registry contract at the Stage-2 boundary.  A
# selected-path check alone cannot detect an injected/tampered candidate set.
STAGE1_REPLAY_INDEX=$(mktemp /tmp/trace_vb_v8_stage1_replay_index_XXXXXX.json)
STAGE1_REPLAY_BEST=$(mktemp /tmp/trace_vb_v8_stage1_replay_best_XXXXXX.txt)
trap 'rm -f "${STAGE1_REPLAY_INDEX}" "${STAGE1_REPLAY_BEST}"' EXIT
stage1_candidate_dir=$(dirname "${stage1_candidate_index}")/candidates
mapfile -t stage1_candidate_records < <(
  find "${stage1_candidate_dir}" -maxdepth 1 -type f \
    -name 'candidate_step_*.json' -print | sort
)
[[ "${#stage1_candidate_records[@]}" -eq 5 ]] || \
  trace_vb_die "Stage-2 entry requires five formal Stage-1 candidates"
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/trace_vb_candidate_registry.py" select \
  --phase stage1 \
  --physical-gpus "${physical_gpus}" \
  --expected-steps 0,512,1024,1536,2048 \
  --candidates "${stage1_candidate_records[@]}" \
  --registered-checkpoint-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}" \
  --registered-payload-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}" \
  --v7-source-sha256 "${V7_SOURCE_SHA256}" \
  --cot-encoder-sha256 "${TRACE_VB_COT_ENCODER_SHA256}" \
  --output "${STAGE1_REPLAY_INDEX}" \
  --best-checkpoint-record "${STAGE1_REPLAY_BEST}"
cmp -s "${STAGE1_REPLAY_INDEX}" "${stage1_candidate_index}" || \
  trace_vb_die "Stage-1 candidate index differs from its registry replay"
replayed_stage1_checkpoint=$(<"${STAGE1_REPLAY_BEST}")
[[ "$(readlink -f "${replayed_stage1_checkpoint}")" == \
  "$(readlink -f "${stage1_checkpoint}")" ]] || \
  trace_vb_die "Stage-1 checkpoint is not the replayed formal selection"

common_run_args=(
  --model "${TRACE_VB_MODEL_CONFIG}"
  --dataset gsm8k_aug_nl
  --trainer default
  --devices 0,1,2,3
  --workspace_path "${CODE_ROOT}"
  --seed "${TRAIN_SEED}"
  --disable_early_stopping
  data_module.dataset_dir="${DATASET_DIR}"
  data_module.enforce_registered_source=true
  data_module.tiny_dataset=false
  data_module.epoch_scaling=1
  batch_size=4
  val_batch_size=1
  num_workers=4
  pin_memory=true
  persistent_workers=false
  trainer.strategy=ddp_find_unused_parameters_true
  trainer.num_sanity_val_steps=0
  trainer.limit_val_batches=1.0
  trainer.check_val_every_n_epoch=1
  trainer.val_check_interval=1.0
  trainer.gradient_clip_val=0
  save_top_k=0
  save_last=true
  save_weights_only=false
  model.model_kwargs.do_trace_rl=true
  model.model_kwargs.readcot_config.compact_anchor_max_chars=64
  model.model_kwargs.trace_policy_config.answer_context_mode=question_and_commit
  model.model_kwargs.trace_policy_config.validation_path=student_commit
  model.model_kwargs.trace_policy_config.registered_capability_checkpoint_path="${TRACE_VB_REGISTERED_CAPABILITY}"
  model.model_kwargs.trace_policy_config.registered_capability_checkpoint_sha256="${TRACE_VB_REGISTERED_CAPABILITY_SHA256}"
  model.model_kwargs.trace_policy_config.registered_capability_payload_tensors="${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_TENSORS}"
  model.model_kwargs.trace_policy_config.registered_capability_payload_sha256="${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}"
  model.model_kwargs.trace_policy_config.metric_safe_baseline_path="${TRACE_VB_METRIC_SAFE_BASELINE}"
  model.model_kwargs.trace_policy_config.metric_safe_baseline_sha256="${TRACE_VB_METRIC_SAFE_BASELINE_SHA256}"
  model.model_kwargs.trace_policy_config.metric_safe_baseline_correct_count="${TRACE_VB_METRIC_SAFE_BASELINE_CORRECT}"
  model.model_kwargs.trace_policy_config.metric_safe_baseline_questions="${TRACE_VB_METRIC_SAFE_BASELINE_QUESTIONS}"
  model.model_kwargs.trace_policy_config.registered_capability_validation_path="${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION}"
  model.model_kwargs.trace_policy_config.registered_capability_validation_sha256="${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_SHA256}"
  model.model_kwargs.trace_policy_config.registered_capability_validation_correct_count="${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_CORRECT}"
  model.model_kwargs.trace_policy_config.registered_capability_validation_questions="${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_QUESTIONS}"
  model.model_kwargs.trace_policy_config.compact_target_max_equations=2
  model.model_kwargs.trace_policy_config.compact_target_max_new_tokens=48
  model.model_kwargs.trace_policy_config.stage1_freeze_path_lora=true
  model.model_kwargs.trace_policy_config.stage1_freeze_capability_query_prior=true
  model.model_kwargs.trace_policy_config.stage1_require_capability_spine_parity=true
  model.model_kwargs.trace_policy_config.stage2_recovery_checkpoint_interval="${STAGE2_RECOVERY_CHECKPOINT_INTERVAL}"
  model.model_kwargs.trace_policy_config.stage0_cot_encoder_checkpoint_sha256="${TRACE_VB_COT_ENCODER_SHA256}"
  model.model_kwargs.trace_policy_config.sufficiency_teacher_checkpoint_sha256="${TRACE_VB_SUFFICIENCY_TEACHER_SHA256}"
  model.model_kwargs.trace_policy_config.visual_record_limit=0
  model.model_kwargs.answer_generation_config.max_new_tokens=48
  model.model_kwargs.answer_generation_config.do_sample=false
  model.model_kwargs.trace_rl_config.n_train_samples_per_epoch="${QUESTIONS_PER_INTERVAL}"
  model.model_kwargs.trace_rl_config.group_size=8
  model.model_kwargs.trace_rl_config.rollout_micro_batch_size=1
  model.model_kwargs.trace_rl_config.exp_batch_size=1
  model.model_kwargs.trace_rl_config.use_trajectory_policy_loss=true
  model.model_kwargs.trace_rl_config.use_answer_policy_loss=false
  model.model_kwargs.trace_rl_config.use_terminal_exact_reward=true
  model.model_kwargs.trace_rl_config.use_evidence_gated_group_rl=true
  model.model_kwargs.trace_rl_config.use_gae=false
  model.model_kwargs.trace_rl_config.use_head_only_ppo=false
  model.model_kwargs.trace_rl_config.policy_update_epochs="${POLICY_UPDATES_PER_ROLLOUT}"
  model.model_kwargs.trace_rl_config.trajectory_clip_epsilon=0.12
  model.model_kwargs.trace_rl_config.stage1_policy_kl_weight=0.05
  model.model_kwargs.trace_rl_config.stage1_policy_target_kl=0.01
  model.model_kwargs.trace_rl_config.score_calibration_batches=64
  model.model_kwargs.trace_rl_config.use_gold_likelihood_fallback=true
  model.model_kwargs.trace_rl_config.score_proxy_minimum_pairs=64
  model.model_kwargs.trace_rl_config.score_proxy_minimum_auc=0.60
  model.model_kwargs.trace_rl_config.minimum_gold_score_gap=0.002
  model.model_kwargs.trace_rl_config.use_semantic_anchor=false
  model.model_kwargs.trace_rl_config.semantic_anchor_initial_weight=0.0
  model.model_kwargs.trace_rl_config.semantic_anchor_minimum_weight=0.0
  model.model_kwargs.trace_rl_config.role_entropy_weights=[1.0,0.9,0.75,0.6,0.45,0.3,0.15,0.0]
  model.model_kwargs.trace_rl_config.dense_outcome_weight=0.0
  model.model_kwargs.trace_rl_config.step_reward_weight=0.15
  model.model_kwargs.trace_rl_config.step_reward_discount=0.90
  model.model_kwargs.trace_rl_config.trajectory_length_weight=0.0
  model.model_kwargs.trace_rl_config.actor_head_lr=8.0e-7
  model.model_kwargs.trace_rl_config.actor_feature_lr=2.0e-7
  model.training_kwargs.scheduler.warmup_steps=75
  model.training_kwargs.scheduler.num_training_steps="${TOTAL_OPTIMIZER_STEPS}"
)

run_python() {
  env \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
    TRACE_PROJECT_ROOT="${CODE_ROOT}" \
    TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
    TRACE_LOG_ROOT="${LOG_ROOT}" \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${physical_gpus}" \
    "${TRACE_VB_PYTHON}" run.py "$@"
}

register_candidate() {
  local step=$1
  local rollout_batches=$2
  local summary=$3
  local checkpoint=$4
  local phase_input=${5:-false}
  local summary_copy=${CANDIDATE_DIR}/validation_rollout_$(printf '%04d' "${rollout_batches}").json
  local record=${CANDIDATE_DIR}/candidate_rollout_$(printf '%04d' "${rollout_batches}").json
  cp "${summary}" "${summary_copy}"
  register_args=(
    register
    --phase stage2 \
    --physical-gpus "${physical_gpus}" \
    --step "${step}" \
    --rollout-batches "${rollout_batches}" \
    --summary "${summary_copy}" \
    --checkpoint "${checkpoint}" \
    --output "${record}" \
    --validation-schema "${TRACE_VB_VALIDATION_SCHEMA}" \
    --questions "${TRACE_VB_VALIDATION_QUESTIONS}" \
    --world-size 4 \
    --registered-checkpoint-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}" \
    --registered-payload-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}" \
    --v7-source-sha256 "${V7_SOURCE_SHA256}" \
    --cot-encoder-sha256 "${TRACE_VB_COT_ENCODER_SHA256}" \
    --stage1-reference-checkpoint "${stage1_checkpoint}"
  )
  if [[ "${phase_input}" == true ]]; then
    register_args+=(--phase-input)
  fi
  "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/trace_vb_candidate_registry.py" \
    "${register_args[@]}"
}

completed_intervals=0
current_checkpoint=
resume_attempt=
if [[ -n "${STAGE2_RESUME_CKPT}" ]]; then
  [[ -d "${OUT_DIR}" && -s "${OUT_DIR}/manifest.txt" ]] || \
    trace_vb_die "Stage-2 resume requires the existing RUN_TAG output"
  [[ -f "${STAGE2_RESUME_CKPT}" ]] || \
    trace_vb_die "missing Stage-2 resume checkpoint: ${STAGE2_RESUME_CKPT}"
  resume_attempt=${TRACE_VB_RESUME_ATTEMPT:-$(date +%Y%m%d-%H%M%S)}
  trace_vb_require_safe_tag "${resume_attempt}"
  [[ -s "${CANDIDATE_DIR}/candidate_rollout_0000.json" ]] || \
    trace_vb_die "Stage-2 resume is missing its Stage-1 input candidate"
  for interval in $(seq 1 "${FORMAL_INTERVALS}"); do
    rollout_batches=$((interval * ROLLOUT_BATCHES_PER_INTERVAL))
    record=${CANDIDATE_DIR}/candidate_rollout_$(printf '%04d' "${rollout_batches}").json
    if [[ -s "${record}" ]]; then
      completed_intervals=${interval}
    else
      break
    fi
  done
  completed_record=${CANDIDATE_DIR}/candidate_rollout_$(printf '%04d' $((completed_intervals * ROLLOUT_BATCHES_PER_INTERVAL))).json
  "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/audit_resume_checkpoint_v8.py" \
    --checkpoint "${STAGE2_RESUME_CKPT}" \
    --stage 2 \
    --completed-intervals "${completed_intervals}" \
    --completed-candidate "${completed_record}" \
    --physical-gpus "${physical_gpus}" \
    --interval-rollout-batches "${ROLLOUT_BATCHES_PER_INTERVAL}" \
    --registered-checkpoint-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}" \
    --registered-payload-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}" \
    --v7-source-sha256 "${V7_SOURCE_SHA256}" \
    --cot-encoder-sha256 "${TRACE_VB_COT_ENCODER_SHA256}" \
    --stage1-reference-checkpoint "${stage1_checkpoint}" \
    --output "${OUT_DIR}/resume_checkpoint_contract_${resume_attempt}.json"
  current_checkpoint=${STAGE2_RESUME_CKPT}
  cat >> "${OUT_DIR}/manifest.txt" <<EOF
resume_at=$(date --iso-8601=seconds)
resume_checkpoint=${STAGE2_RESUME_CKPT}
resume_checkpoint_sha256=$(sha256sum "${STAGE2_RESUME_CKPT}" | awk '{print $1}')
resume_completed_intervals=${completed_intervals}
resume_attempt=${resume_attempt}
EOF
else
  [[ ! -e "${OUT_DIR}" ]] || trace_vb_die "refusing to reuse output: ${OUT_DIR}"
  mkdir -p "${OUT_DIR}" "${CANDIDATE_DIR}" "${TMP_ROOT}" "${LOG_PARENT}"
  cat > "${OUT_DIR}/manifest.txt" <<EOF
model=${TRACE_VB_VERSION}
phase=short_terminal_plus_step_latent_RL
run_tag=${RUN_TAG}
checkpoint_schema=${TRACE_VB_CHECKPOINT_SCHEMA}
validation_schema=${TRACE_VB_VALIDATION_SCHEMA}
project_root=${CODE_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
stage1_checkpoint=${stage1_checkpoint}
stage1_checkpoint_sha256=$(sha256sum "${stage1_checkpoint}" | awk '{print $1}')
v7_source_checkpoint_sha256=${V7_SOURCE_SHA256}
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
stage1_candidate_index=${stage1_candidate_index}
stage1_candidate_index_sha256=$(sha256sum "${stage1_candidate_index}" | awk '{print $1}')
stage1_candidate_index_registry_replay=exact_byte_match
stage1_safety_floor=${MINIMUM_STAGE1_CORRECT}_of_${TRACE_VB_VALIDATION_QUESTIONS}
terminal_reward=exact_answer_correctness
step_reward=role_local_gold_CoT_semantic_agreement
step_reward_weight=0.15
step_reward_discount=0.90
stage1_policy_KL_weight=0.05
actor_head_lr=8e-7
actor_feature_lr=2e-7
rollout_batches_per_interval=${ROLLOUT_BATCHES_PER_INTERVAL}
recovery_checkpoint_interval_rollout_batches=${STAGE2_RECOVERY_CHECKPOINT_INTERVAL}
recovery_checkpoint_policy=diagnostic_only_not_formal_resume_validated_boundary_rollback
questions_per_interval=${QUESTIONS_PER_INTERVAL}
maximum_policy_updates_per_rollout=${POLICY_UPDATES_PER_ROLLOUT}
actual_optimizer_steps=read_from_each_validation_summary_due_KL_early_stop
formal_intervals=${FORMAL_INTERVALS}
scheduled_optimizer_steps=${TOTAL_OPTIMIZER_STEPS}
full_validation_optimizer_steps=read_from_each_validation_summary
full_validation_rollout_batches=${EXPECTED_ROLLOUT_BATCHES}
final_selection=stage1_input_plus_four_RL_candidates_exact_integer_then_earliest
test_split=false
evidence_pipeline=false
physical_gpus=${physical_gpus}
formal_gpus=${TRACE_VB_FORMAL_GPUS}
fixed_gpus=${TRACE_VB_FIXED_GPUS}
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF
  selected_summary=$("${TRACE_VB_PYTHON}" - "${stage1_candidate_index}" <<'PY'
import json
import sys
from pathlib import Path
index = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(index["selected_step"])
matches = [item for item in index["candidates"] if int(item["step"]) == step]
if len(matches) != 1:
    raise SystemExit("Stage-1 selected summary is ambiguous")
print(matches[0]["summary"])
PY
  )
  register_candidate 0 0 "${selected_summary}" "${stage1_checkpoint}" true
  current_checkpoint=${stage1_checkpoint}
fi

for interval in $(seq 1 "${FORMAL_INTERVALS}"); do
  if (( interval <= completed_intervals )); then
    continue
  fi
  rollout_batches=$((interval * ROLLOUT_BATCHES_PER_INTERVAL))
  interval_tag=${RUN_TAG}_stage2_rollout$(printf '%04d' "${rollout_batches}")
  if [[ -n "${resume_attempt}" ]]; then
    interval_tag=${interval_tag}_resume_${resume_attempt}
  fi
  checkpoint_args=(--resume_ckpt_path "${current_checkpoint}")
  if (( interval == 1 && completed_intervals == 0 )); then
    # The Stage-1 phase input is a weights-only Stage-2 boundary-zero restart.
    checkpoint_args=(--load_ckpt_path "${current_checkpoint}")
  fi
  run_python \
    "${common_run_args[@]}" \
    "${checkpoint_args[@]}" \
    --log_suffix "${interval_tag}" \
    trainer.max_epochs="${interval}" \
    trainer.max_steps=-1 \
    trainer.limit_train_batches="${ROLLOUT_BATCHES_PER_INTERVAL}" \
    trainer.default_root_dir="${OUT_DIR}/trainer_rollout$(printf '%04d' "${rollout_batches}")" \
    2>&1 | tee -a "${OUT_DIR}/train.log"
  interval_logger=$(trace_vb_find_single_logger_dir "${LOG_PARENT}" "${interval_tag}")
  summary=${interval_logger}/validation_epoch_$(printf '%03d' $((interval - 1))).json
  checkpoint=${interval_logger}/checkpoints/last.ckpt
  [[ -f "${checkpoint}" ]] || \
    trace_vb_die "Stage-2 interval ${interval} produced no last checkpoint"
  optimizer_step=$("${TRACE_VB_PYTHON}" - "${summary}" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = int(summary.get("global_step", -1))
if step < 0:
    raise SystemExit("Stage-2 validation summary has no valid global_step")
print(step)
PY
  )
  register_candidate \
    "${optimizer_step}" "${rollout_batches}" "${summary}" "${checkpoint}" false
  # Periodic full-state files are diagnostic snapshots only.  Formal resume
  # always rolls back to the latest registered validation boundary.  Once that
  # boundary is registered, delete the now-redundant diagnostic snapshots.
  find "${interval_logger}/checkpoints" -maxdepth 1 -type f \
    -name 'stage2-recovery-rollout*-globalstep*.ckpt' -delete
  current_checkpoint=${checkpoint}
  cat >> "${OUT_DIR}/manifest.txt" <<EOF
optimizer_step_${optimizer_step}_rollout_batches=${rollout_batches}
optimizer_step_${optimizer_step}_checkpoint=${checkpoint}
optimizer_step_${optimizer_step}_checkpoint_sha256=$(sha256sum "${checkpoint}" | awk '{print $1}')
optimizer_step_${optimizer_step}_validated_at=$(date --iso-8601=seconds)
EOF
done

mapfile -t candidate_records < <(
  find "${CANDIDATE_DIR}" -maxdepth 1 -type f \
    -name 'candidate_rollout_*.json' -print | sort
)
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/trace_vb_candidate_registry.py" finalize-completed \
  --phase stage2 \
  --physical-gpus "${physical_gpus}" \
  --expected-rollout-batches "${EXPECTED_ROLLOUT_BATCHES}" \
  --candidates "${candidate_records[@]}" \
  --registered-checkpoint-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}" \
  --registered-payload-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}" \
  --v7-source-sha256 "${V7_SOURCE_SHA256}" \
  --cot-encoder-sha256 "${TRACE_VB_COT_ENCODER_SHA256}" \
  --stage1-reference-checkpoint "${stage1_checkpoint}" \
  --index "${OUT_DIR}/candidate_index.json" \
  --best-checkpoint-record "${OUT_DIR}/best_checkpoint.txt" \
  --last-checkpoint-record "${OUT_DIR}/last_checkpoint.txt" \
  --manifest "${OUT_DIR}/manifest.txt" \
  --expected-last-checkpoint "${current_checkpoint}"
