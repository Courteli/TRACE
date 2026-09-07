#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage1-best-checkpoint>" >&2
  exit 2
fi

physical_gpus=$1
stage1_checkpoint=$2
TRAIN_SEED=${TRAIN_SEED:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v7_stage2_seed${TRAIN_SEED}}
[[ "${TRAIN_SEED}" =~ ^[0-9]+$ ]] || trace_vb_die "TRAIN_SEED must be non-negative"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  trace_vb_die "RUN_TAG contains unsafe path or glob characters: ${RUN_TAG}"
RUN_ROOT=${TRACE_VB_ARTIFACT_ROOT}/training
LOG_ROOT=${TRACE_VB_ARTIFACT_ROOT}/logs
TMP_ROOT=${TRACE_VB_ARTIFACT_ROOT}/tmp
DATASET_DIR=${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL
out_dir=${RUN_ROOT}/${RUN_TAG}
log_root=${LOG_ROOT}/${TRACE_VB_MODEL_CONFIG}/gsm8k_aug_nl-gsm8k_aug_nl

EXPECTED_VALIDATION_QUESTIONS=747
STAGE1_MINIMUM_CORRECT=537
FORMAL_EPOCHS=10
ROLLOUT_BATCHES_PER_EPOCH=512
POLICY_UPDATES_PER_ROLLOUT=2
SCHEDULED_OPTIMIZER_STEPS=10240

trace_vb_require_four_gpus "${physical_gpus}"
[[ -f "${stage1_checkpoint}" ]] || \
  trace_vb_die "missing TRACE-VB Stage-1 checkpoint: ${stage1_checkpoint}"
[[ -z "${RESUME_CKPT_PATH:-}" ]] || \
  trace_vb_die "formal Stage 2 starts from the validation-selected Stage-1 checkpoint"
[[ ! -e "${out_dir}" ]] || \
  trace_vb_die "refusing to reuse Stage-2 output directory: ${out_dir}"
if [[ -d "${log_root}" ]] && \
  find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${RUN_TAG}" -print -quit | grep -q .; then
  trace_vb_die "refusing to reuse Stage-2 logger tag: ${RUN_TAG}"
fi

stage1_hparams="$(dirname "$(dirname "${stage1_checkpoint}")")/hparams.yaml"
[[ -f "${stage1_hparams}" ]] || trace_vb_die "missing Stage-1 hparams.yaml"
grep -Fq "workspace_path: ${TRACE_VB_STAGE1_CODE_ROOT}" "${stage1_hparams}" || \
  trace_vb_die "Stage 1 was not produced from the registered TRACE-VB-v7 code root"
grep -q "src.models.trace_vb.LitTRACEVB" "${stage1_hparams}" || \
  trace_vb_die "Stage-1 checkpoint is not a TRACE-VB-v7 policy checkpoint"
grep -qi "do_trace_rl: false" "${stage1_hparams}" || \
  trace_vb_die "Stage-1 checkpoint has the wrong training phase"
grep -Fq "answer_context_mode: question_and_commit" "${stage1_hparams}" || \
  trace_vb_die "Stage 1 did not preserve the question+COMMIT information bridge"

mkdir -p "${out_dir}" "${TMP_ROOT}" "${log_root}"
cd "${CODE_ROOT}"

# Checkpoint metadata and tensors are decoded by PyTorch's restricted
# weights-only unpickler. The helper allowlists data-only OmegaConf containers;
# it never permits project classes or arbitrary pickle callables.
"${TRACE_VB_PYTHON}" - "${stage1_checkpoint}" "${DATASET_DIR}" <<'PY'
import sys
from pathlib import Path

from src.utils.safe_checkpoint import safe_load_checkpoint

path = Path(sys.argv[1])
dataset_dir = sys.argv[2]
checkpoint = safe_load_checkpoint(path, map_location="cpu")
if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
    raise SystemExit("Stage 2 requires a Stage-1 TRACE-VB checkpoint")
if checkpoint.get("trace_vb_schema_version") != "trace_vb_v7":
    raise SystemExit("Stage 2 requires a TRACE-VB-v7 checkpoint")
if checkpoint.get("trace_vb_stage2_objective") != (
    "capability_preserving_role_semantic_cot_sft"
):
    raise SystemExit("Stage-1 objective marker is missing or inconsistent")
if "trace_stage1_policy_reference" in checkpoint:
    raise SystemExit("a Stage-1 checkpoint must not contain a Stage-2 reference")

state = checkpoint.get("state_dict", {})
required_prefixes = (
    "trajectory_policy.",
    "trajectory_posterior.",
    "posterior_context_norm.",
    "plan_forecaster.",
    "solve_text_decoder.",
    "sufficiency_head.",
    "capability_state_norm.",
    "capability_latent_bridge.",
    "capability_anchor_gate_predictor.",
)
for prefix in required_prefixes:
    if not any(name.startswith(prefix) for name in state):
        raise SystemExit(f"Stage-1 checkpoint is missing {prefix}")
if "capability_latent_queries" not in state:
    raise SystemExit("Stage-1 checkpoint is missing capability_latent_queries")
for name in ("capability_trace_view", "capability_trace_step_views"):
    if name not in state:
        raise SystemExit(f"Stage-1 checkpoint is missing {name}")
for adapter, expected in (("trace_cot_encoder", 504), ("trace_capability", 504)):
    coverage = sum(
        1 for name in state
        if f".{adapter}." in name
        and (".lora_A." in name or ".lora_B." in name)
        and name.endswith(".weight")
    )
    if coverage != expected:
        raise SystemExit(
            f"Stage-1 {adapter} LoRA coverage is {coverage}, expected {expected}"
        )

config = checkpoint.get("hyper_parameters", {}).get("all_config")
if config is None:
    raise SystemExit("Stage-1 checkpoint has no resolved training config")
if str(config.model.target) != "src.models.trace_vb.LitTRACEVB":
    raise SystemExit("Stage-1 checkpoint has the wrong model target")
if str(config.data_module.dataset_dir) != dataset_dir:
    raise SystemExit("Stage-1 checkpoint used a different dataset directory")
if not bool(config.data_module.enforce_registered_source):
    raise SystemExit("Stage-1 checkpoint bypassed the registered dataset")
if bool(config.data_module.tiny_dataset):
    raise SystemExit("Stage-1 checkpoint used a tiny dataset")
PY

# Require all ten Stage-1 full-validation records, verify the DDP de-padding
# contract, and gate the selected checkpoint at an integer 537/747 threshold.
stage1_run_dir=$(dirname "$(dirname "${stage1_checkpoint}")")
stage1_summary_index=${stage1_run_dir}/validation_summary_index.json
[[ -s "${stage1_summary_index}" ]] || \
  trace_vb_die "missing centralized Stage-1 validation summary index"
"${TRACE_VB_PYTHON}" - \
  "${stage1_checkpoint}" \
  "${stage1_run_dir}" \
  "${out_dir}/stage1_validation_gate.json" \
  "${EXPECTED_VALIDATION_QUESTIONS}" \
  "${STAGE1_MINIMUM_CORRECT}" \
  "${FORMAL_EPOCHS}" \
  "${DATASET_DIR}/gsm8k_val_processed.jsonl" \
  "${stage1_summary_index}" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from src.utils.safe_checkpoint import safe_load_checkpoint

checkpoint_path = Path(sys.argv[1]).resolve()
run_dir = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3])
expected = int(sys.argv[4])
minimum_correct = int(sys.argv[5])
formal_epochs = int(sys.argv[6])
validation_source = Path(sys.argv[7]).resolve()
index_path = Path(sys.argv[8]).resolve()

index = json.loads(index_path.read_text(encoding="utf-8"))
if index.get("schema_version") != "trace_vb_v7_stage1_validation_index_v1":
    raise SystemExit("Stage-1 validation index has the wrong schema")
if int(index.get("formal_epochs", -1)) != formal_epochs:
    raise SystemExit("Stage-1 validation index has the wrong epoch count")
if Path(index.get("published_run_dir", "")).resolve() != run_dir:
    raise SystemExit("Stage-1 validation index belongs to another run directory")
entries = index.get("summaries", [])
if len(entries) != formal_epochs:
    raise SystemExit("Stage-1 validation index is incomplete")
indexed_epochs = [int(entry.get("epoch_index", -1)) for entry in entries]
if sorted(indexed_epochs) != list(range(formal_epochs)):
    raise SystemExit("Stage-1 validation index does not cover each epoch once")
for entry in entries:
    epoch = int(entry["epoch_index"])
    expected_path = (run_dir / f"validation_epoch_{epoch:03d}.json").resolve()
    if Path(entry.get("checkpoint_run_copy", "")).resolve() != expected_path:
        raise SystemExit(f"Stage-1 validation index path mismatch at epoch {epoch}")
    if not expected_path.is_file():
        raise SystemExit(f"missing indexed Stage-1 summary for epoch {epoch}")
    digest = hashlib.sha256(expected_path.read_bytes()).hexdigest()
    if digest != entry.get("sha256"):
        raise SystemExit(f"Stage-1 validation summary hash mismatch at epoch {epoch}")

source_rows = [
    json.loads(line)
    for line in validation_source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
source_ids = [int(row.get("id", index)) for index, row in enumerate(source_rows)]
if len(source_ids) != expected or len(set(source_ids)) != expected:
    raise SystemExit(
        "registered validation source must contain exactly "
        f"{expected} unique source ids"
    )

checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu")
match = re.match(r"^epoch(\d+)__step\d+__monitor[-+0-9.eE]+\.ckpt$", checkpoint_path.name)
selected_epoch = int(match.group(1)) if match else int(checkpoint.get("epoch", -1))
if not 0 <= selected_epoch < formal_epochs:
    raise SystemExit(f"cannot identify selected Stage-1 epoch: {selected_epoch}")

summaries = []
failures = []
for epoch in range(formal_epochs):
    path = run_dir / f"validation_epoch_{epoch:03d}.json"
    if not path.is_file():
        failures.append(f"missing {path.name}")
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "trace_vb_v7_validation_behavior_v1":
        failures.append(f"{path.name}: wrong schema")
    if summary.get("validation_path") != "student_commit":
        failures.append(f"{path.name}: validation did not use q+COMMIT")
    if int(summary.get("epoch_index", -1)) != epoch:
        failures.append(f"{path.name}: wrong epoch_index")
    if int(summary.get("world_size", -1)) != 4:
        failures.append(f"{path.name}: world_size is not 4")
    if int(summary.get("unique_questions", -1)) != expected:
        failures.append(f"{path.name}: unique_questions is not {expected}")
    correct = int(summary.get("correct_count", -1))
    accuracy = float(summary.get("accuracy", float("nan")))
    if not math.isfinite(accuracy) or correct < 0:
        failures.append(f"{path.name}: invalid accuracy/correct_count")
    elif not math.isclose(accuracy, correct / expected, abs_tol=1e-12, rel_tol=0.0):
        failures.append(f"{path.name}: accuracy is inconsistent with exact count")
    for key in (
        "valid_answer_fraction",
        "unique_prediction_ratio",
        "top1_mode_fraction",
        "nonempty_output_fraction",
    ):
        value = float(summary.get(key, float("nan")))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            failures.append(f"{path.name}: invalid {key}")
    summaries.append(summary)

if len(summaries) == formal_epochs:
    selected = summaries[selected_epoch]
    selected_correct = int(selected["correct_count"])
    selected_accuracy = float(selected["accuracy"])
    best_accuracy = max(float(item["accuracy"]) for item in summaries)
    if selected_accuracy + 1e-12 < best_accuracy:
        failures.append(
            f"selected epoch accuracy {selected_accuracy:.12f} is below "
            f"run best {best_accuracy:.12f}"
        )
    if selected_correct < minimum_correct:
        failures.append(
            f"selected checkpoint has {selected_correct}/{expected}, "
            f"requires at least {minimum_correct}/{expected}"
        )
    selected_behavior_thresholds = {
        "valid_answer_fraction": (0.98, "minimum"),
        "unique_prediction_ratio": (0.20, "minimum"),
        "top1_mode_fraction": (0.20, "maximum"),
        "nonempty_output_fraction": (0.98, "minimum"),
    }
    for key, (threshold, direction) in selected_behavior_thresholds.items():
        value = float(selected[key])
        if direction == "minimum" and value < threshold:
            failures.append(f"selected {key}={value:.6f}<{threshold:.6f}")
        if direction == "maximum" and value > threshold:
            failures.append(f"selected {key}={value:.6f}>{threshold:.6f}")
else:
    selected = {}
    selected_correct = -1
    selected_accuracy = float("nan")
    best_accuracy = float("nan")

report = {
    "schema_version": "trace_vb_v7_stage1_to_stage2_gate_v1",
    "status": "FAIL" if failures else "PASS",
    "checkpoint": str(checkpoint_path),
    "validation_summary_index": str(index_path),
    "stage1_tag": index.get("stage1_tag"),
    "selected_epoch": selected_epoch,
    "selected_correct_count": selected_correct,
    "selected_accuracy": selected_accuracy,
    "minimum_correct_count": minimum_correct,
    "selected_behavior_thresholds": {
        "minimum_valid_answer_fraction": 0.98,
        "minimum_unique_prediction_ratio": 0.20,
        "maximum_top1_mode_fraction": 0.20,
        "minimum_nonempty_output_fraction": 0.98,
    },
    "expected_unique_questions": expected,
    "formal_epochs_checked": formal_epochs,
    "best_stage1_accuracy": best_accuracy,
    "deduplication_contract": (
        "four_rank_gather_then_unique_immutable_source_id"
    ),
    "failures": failures,
}
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if failures:
    raise SystemExit("Stage-1 validation gate failed: " + "; ".join(failures))
PY

TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" tools/data_contract_audit.py >/dev/null

# Fail closed if the resolved formal configuration drifts away from the v7
# actor-only, role-local credit contract encoded by this launcher.
"${TRACE_VB_PYTHON}" - <<'PY'
from omegaconf import OmegaConf

config = OmegaConf.load("src/configs/models/trace_vb_policy_qwen3_instruct.yaml")
model = config.model.model_kwargs
rl = model.trace_rl_config
expected = {
    "group_size": 8,
    "n_train_samples_per_epoch": 2048,
    "rollout_micro_batch_size": 1,
    "exp_batch_size": 1,
    "use_trajectory_policy_loss": True,
    "use_answer_policy_loss": False,
    "use_terminal_exact_reward": True,
    "use_evidence_gated_group_rl": True,
    "policy_update_epochs": 2,
    "score_calibration_batches": 64,
    "use_gold_likelihood_fallback": True,
    "use_semantic_anchor": False,
    "dense_outcome_weight": 0.0,
    "step_reward_weight": 0.15,
    "step_reward_discount": 0.90,
    "trajectory_length_weight": 0.0,
    "actor_head_lr": 2.0e-6,
    "actor_feature_lr": 4.0e-7,
}
for key, value in expected.items():
    if rl.get(key) != value:
        raise SystemExit(f"formal v7 config drift: {key}={rl.get(key)!r}, expected {value!r}")
if list(rl.role_entropy_weights) != [1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.0]:
    raise SystemExit("formal v7 role-entropy schedule drifted")
if str(model.trace_policy_config.answer_context_mode) != "question_and_commit":
    raise SystemExit("formal v7 answer context must be question_and_commit")
if str(model.trace_policy_config.get("validation_path", "student_commit")) != "student_commit":
    raise SystemExit("formal v7 validation must use the student COMMIT path")
if bool(model.answer_generation_config.do_sample):
    raise SystemExit("formal validation answer generation must be deterministic")
if int(model.answer_generation_config.max_new_tokens) != 48:
    raise SystemExit("formal validation answer budget must be 48 tokens")
PY

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE-VB-v7
full_name=Capability-Anchored_Role-Structured_Latent_Reasoning_with_Role-Local_Credit
phase=stage2_actor_only_role_local_latent_rl
model_config=${TRACE_VB_MODEL_CONFIG}
project_root=${CODE_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
stage1_checkpoint=${stage1_checkpoint}
stage1_checkpoint_sha256=$(sha256sum "${stage1_checkpoint}" | awk '{print $1}')
stage1_gate=selected_validation_best_at_least_${STAGE1_MINIMUM_CORRECT}_of_${EXPECTED_VALIDATION_QUESTIONS}
stage1_gate_report=${out_dir}/stage1_validation_gate.json
dataset_dir=${DATASET_DIR}
latent_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT
stochastic_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE
commit_is_deterministic=true
answer_context=question_plus_COMMIT
private_latent_answer_access=false
terminal_signal=exact_answer_correctness
terminal_group_signal=question_local_group_relative_exact_outcome
all_correct_terminal_group_signal=zero
all_wrong_terminal_group_signal=calibrated_frozen_gold_answer_likelihood_rank_or_zero
fallback_calibration_batches=64
fallback_minimum_pairs=64
fallback_minimum_auc=0.60
fallback_minimum_within_question_gap=0.002
role_local_signal=existing_gold_CoT_semantic_agreement_at_PLAN_SOLVE1_5_REFINE
role_local_discount=0.90
role_local_weight=0.15
commit_role_reward=zero_schema_masked
credit_combination=terminal_group_advantage_plus_discounted_role_local_advantage
entropy_schedule=1.0_0.9_0.75_0.6_0.45_0.3_0.15_0.0
entropy_activation=failed_paths_only
optimization=actor_only_clipped_latent_policy_updates
trainable_actor_features=policy_step_embedding_plus_shared_policy_trunk
trainable_actor_heads=seven_stochastic_role_mean_and_log_std_heads
frozen_components=backbone_answer_channel_transition_dynamics_COMMIT_stage1_reference
policy_update_epochs_per_rollout=${POLICY_UPDATES_PER_ROLLOUT}
trajectory_clip_epsilon=0.12
stage1_reference_KL_weight=0.02
stage1_reference_KL_stop=0.01
actor_head_lr=2e-6
actor_feature_lr=4e-7
group_size=8_iid_question_conditioned_paths
source_training_questions=6726
unique_training_questions_per_epoch=2048
rollout_batches_per_epoch=${ROLLOUT_BATCHES_PER_EPOCH}
maximum_optimizer_steps_per_epoch=1024
scheduled_optimizer_steps=${SCHEDULED_OPTIMIZER_STEPS}
epochs=${FORMAL_EPOCHS}
validation=every_epoch_four_rank_gather_deduplicate_exactly_${EXPECTED_VALIDATION_QUESTIONS}
checkpoint_selection=exact_unrounded_full_validation_accuracy
answer_generation=deterministic_max_48_tokens
checkpoint_loading=restricted_weights_only_with_data_only_OmegaConf_allowlist
selected_checkpoint_behavior_gate=validity_diversity_mode_fraction_nonempty_output
physical_gpus=${physical_gpus}
minimum_free_memory_gate_mib=${TRACE_VB_MIN_FREE_GPU_MIB}
requested_global_question_batch_size=4
effective_per_device_question_batch_size=1
tiny_dataset=false
test_times=1
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF

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
    --trainer default \
    --devices 0,1,2,3 \
    --workspace_path "${CODE_ROOT}" \
    --load_ckpt_path "${stage1_checkpoint}" \
    --test_times 1 \
    --seed "${TRAIN_SEED}" \
    --disable_early_stopping \
    --log_suffix "${RUN_TAG}" \
    data_module.dataset_dir="${DATASET_DIR}" \
    data_module.enforce_registered_source=true \
    data_module.tiny_dataset=false \
    data_module.epoch_scaling=1 \
    batch_size=4 \
    val_batch_size=1 \
    num_workers=4 \
    pin_memory=true \
    persistent_workers=false \
    trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0 \
    trainer.max_epochs="${FORMAL_EPOCHS}" \
    trainer.max_steps=-1 \
    trainer.limit_train_batches="${ROLLOUT_BATCHES_PER_EPOCH}" \
    trainer.limit_val_batches=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.val_check_interval=1.0 \
    trainer.gradient_clip_val=0 \
    trainer.default_root_dir="${out_dir}/trainer" \
    save_top_k=1 \
    save_last=true \
    save_weights_only=false \
    model.model_kwargs.do_trace_rl=true \
    model.model_kwargs.trace_policy_config.answer_context_mode=question_and_commit \
    model.model_kwargs.trace_policy_config.validation_path=student_commit \
    model.model_kwargs.trace_policy_config.visual_record_limit=0 \
    model.model_kwargs.answer_generation_config.max_new_tokens=48 \
    model.model_kwargs.answer_generation_config.do_sample=false \
    model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048 \
    model.model_kwargs.trace_rl_config.group_size=8 \
    model.model_kwargs.trace_rl_config.rollout_micro_batch_size=1 \
    model.model_kwargs.trace_rl_config.exp_batch_size=1 \
    model.model_kwargs.trace_rl_config.use_trajectory_policy_loss=true \
    model.model_kwargs.trace_rl_config.use_answer_policy_loss=false \
    model.model_kwargs.trace_rl_config.use_terminal_exact_reward=true \
    model.model_kwargs.trace_rl_config.use_evidence_gated_group_rl=true \
    model.model_kwargs.trace_rl_config.use_gae=false \
    model.model_kwargs.trace_rl_config.use_head_only_ppo=false \
    model.model_kwargs.trace_rl_config.policy_update_epochs=2 \
    model.model_kwargs.trace_rl_config.trajectory_clip_epsilon=0.12 \
    model.model_kwargs.trace_rl_config.stage1_policy_kl_weight=0.02 \
    model.model_kwargs.trace_rl_config.stage1_policy_target_kl=0.01 \
    model.model_kwargs.trace_rl_config.score_calibration_batches=64 \
    model.model_kwargs.trace_rl_config.use_gold_likelihood_fallback=true \
    model.model_kwargs.trace_rl_config.score_proxy_minimum_pairs=64 \
    model.model_kwargs.trace_rl_config.score_proxy_minimum_auc=0.60 \
    model.model_kwargs.trace_rl_config.minimum_gold_score_gap=0.002 \
    model.model_kwargs.trace_rl_config.use_semantic_anchor=false \
    model.model_kwargs.trace_rl_config.semantic_anchor_initial_weight=0.0 \
    model.model_kwargs.trace_rl_config.semantic_anchor_minimum_weight=0.0 \
    model.model_kwargs.trace_rl_config.role_entropy_weights=[1.0,0.9,0.75,0.6,0.45,0.3,0.15,0.0] \
    model.model_kwargs.trace_rl_config.dense_outcome_weight=0.0 \
    model.model_kwargs.trace_rl_config.step_reward_weight=0.15 \
    model.model_kwargs.trace_rl_config.step_reward_discount=0.90 \
    model.model_kwargs.trace_rl_config.trajectory_length_weight=0.0 \
    model.model_kwargs.trace_rl_config.actor_head_lr=2.0e-6 \
    model.model_kwargs.trace_rl_config.actor_feature_lr=4.0e-7 \
    model.training_kwargs.scheduler.warmup_steps=150 \
    model.training_kwargs.scheduler.num_training_steps="${SCHEDULED_OPTIMIZER_STEPS}" \
    2>&1 | tee "${out_dir}/train.log"

# Validate every Stage-2 epoch summary before publishing a checkpoint. The
# selected file must correspond to the exact (not filename-rounded) maximum.
mapfile -t checkpoint_paths < <(
  "${TRACE_VB_PYTHON}" - \
    "${log_root}" \
    "${RUN_TAG}" \
    "${out_dir}/validation_contract.json" \
    "${EXPECTED_VALIDATION_QUESTIONS}" \
    "${FORMAL_EPOCHS}" \
    "${DATASET_DIR}/gsm8k_val_processed.jsonl" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
run_tag = sys.argv[2]
output = Path(sys.argv[3])
expected = int(sys.argv[4])
formal_epochs = int(sys.argv[5])
validation_source = Path(sys.argv[6]).resolve()

source_rows = [
    json.loads(line)
    for line in validation_source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
source_ids = [int(row.get("id", index)) for index, row in enumerate(source_rows)]
if len(source_ids) != expected or len(set(source_ids)) != expected:
    raise SystemExit(
        "registered validation source must contain exactly "
        f"{expected} unique source ids"
    )

run_dirs = sorted(path for path in log_root.glob(f"*_{run_tag}") if path.is_dir())
if len(run_dirs) != 1:
    raise SystemExit(f"expected one Stage-2 logger directory, found {len(run_dirs)}")
run_dir = run_dirs[0]
summaries = []
for epoch in range(formal_epochs):
    path = run_dir / f"validation_epoch_{epoch:03d}.json"
    if not path.is_file():
        raise SystemExit(f"missing full-validation summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "trace_vb_v7_validation_behavior_v1":
        raise SystemExit(f"wrong validation schema in {path}")
    if summary.get("validation_path") != "student_commit":
        raise SystemExit(f"validation did not use q+COMMIT in {path}")
    if int(summary.get("epoch_index", -1)) != epoch:
        raise SystemExit(f"wrong epoch index in {path}")
    if int(summary.get("world_size", -1)) != 4:
        raise SystemExit(f"validation was not aggregated from four ranks: {path}")
    if int(summary.get("unique_questions", -1)) != expected:
        raise SystemExit(f"validation did not deduplicate to {expected}: {path}")
    correct = int(summary.get("correct_count", -1))
    accuracy = float(summary.get("accuracy", float("nan")))
    if correct < 0 or not math.isfinite(accuracy):
        raise SystemExit(f"invalid exact outcome metrics in {path}")
    if not math.isclose(accuracy, correct / expected, abs_tol=1e-12, rel_tol=0.0):
        raise SystemExit(f"rounded or inconsistent accuracy in {path}")
    for key in (
        "valid_answer_fraction",
        "unique_prediction_ratio",
        "top1_mode_fraction",
        "nonempty_output_fraction",
    ):
        value = float(summary.get(key, float("nan")))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"invalid {key} in {path}")
    summaries.append(summary)

best_accuracy = max(float(item["accuracy"]) for item in summaries)
best_epochs = {
    epoch
    for epoch, item in enumerate(summaries)
    if math.isclose(float(item["accuracy"]), best_accuracy, abs_tol=1e-12, rel_tol=0.0)
}
checkpoint_dir = run_dir / "checkpoints"
pattern = re.compile(r"^epoch(\d+)__step(\d+)__monitor([-+0-9.eE]+)\.ckpt$")
candidates = []
for path in checkpoint_dir.glob("epoch*__step*__monitor*.ckpt"):
    match = pattern.match(path.name)
    if match:
        candidates.append((int(match.group(1)), float(match.group(3)), path))
if len(candidates) != 1:
    raise SystemExit(f"expected exactly one top-1 Stage-2 checkpoint, found {len(candidates)}")
checkpoint_epoch, filename_score, best_checkpoint = candidates[0]
if checkpoint_epoch not in best_epochs:
    raise SystemExit("published checkpoint does not match exact validation maximum")
selected = summaries[checkpoint_epoch]
selected_thresholds = {
    "valid_answer_fraction": (0.98, "minimum"),
    "unique_prediction_ratio": (0.20, "minimum"),
    "top1_mode_fraction": (0.20, "maximum"),
    "nonempty_output_fraction": (0.98, "minimum"),
}
for key, (threshold, direction) in selected_thresholds.items():
    value = float(selected[key])
    if direction == "minimum" and value < threshold:
        raise SystemExit(f"selected checkpoint {key}={value:.6f}<{threshold:.6f}")
    if direction == "maximum" and value > threshold:
        raise SystemExit(f"selected checkpoint {key}={value:.6f}>{threshold:.6f}")
if not math.isclose(
    filename_score,
    float(summaries[checkpoint_epoch]["accuracy"]),
    abs_tol=5.1e-7,
    rel_tol=0.0,
):
    raise SystemExit("checkpoint filename score disagrees with exact validation JSON")
last_checkpoint = checkpoint_dir / "last.ckpt"
if not last_checkpoint.is_file():
    raise SystemExit("Stage 2 completed without a recoverable last checkpoint")

report = {
    "schema_version": "trace_vb_v7_stage2_validation_contract_v1",
    "status": "PASS",
    "run_directory": str(run_dir),
    "expected_unique_questions": expected,
    "deduplication_contract": (
        "four_rank_gather_then_unique_immutable_source_id"
    ),
    "epochs_checked": formal_epochs,
    "best_epochs": sorted(best_epochs),
    "best_accuracy": best_accuracy,
    "best_correct_count": int(selected["correct_count"]),
    "selected_behavior_thresholds": {
        "minimum_valid_answer_fraction": 0.98,
        "minimum_unique_prediction_ratio": 0.20,
        "maximum_top1_mode_fraction": 0.20,
        "minimum_nonempty_output_fraction": 0.98,
    },
    "best_checkpoint": str(best_checkpoint),
    "last_checkpoint": str(last_checkpoint),
    "epoch_summaries": summaries,
}
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(best_checkpoint)
print(last_checkpoint)
PY
)

[[ "${#checkpoint_paths[@]}" -eq 2 ]] || \
  trace_vb_die "Stage-2 validation contract did not return two checkpoints"
best_checkpoint=${checkpoint_paths[0]}
last_checkpoint=${checkpoint_paths[1]}
[[ -f "${best_checkpoint}" ]] || trace_vb_die "missing Stage-2 best checkpoint"
[[ -f "${last_checkpoint}" ]] || trace_vb_die "missing Stage-2 last checkpoint"

"${TRACE_VB_PYTHON}" - "${best_checkpoint}" <<'PY'
import sys
from src.utils.safe_checkpoint import safe_load_checkpoint

checkpoint = safe_load_checkpoint(sys.argv[1], map_location="cpu")
if int(checkpoint.get("trace_policy_training_stage", -1)) != 2:
    raise SystemExit("published checkpoint is not TRACE-VB Stage 2")
if checkpoint.get("trace_vb_schema_version") != "trace_vb_v7":
    raise SystemExit("published checkpoint has the wrong TRACE-VB schema")
if checkpoint.get("trace_vb_stage2_objective") != (
    "capability_anchored_role_local_latent_rl"
):
    raise SystemExit("published checkpoint has the wrong Stage-2 objective")
reference = checkpoint.get("trace_stage1_policy_reference")
if not isinstance(reference, dict) or not reference:
    raise SystemExit("Stage-2 checkpoint is missing its immutable Stage-1 reference")
PY

cat >> "${out_dir}/manifest.txt" <<EOF
best_checkpoint=${best_checkpoint}
last_checkpoint=${last_checkpoint}
validation_contract=${out_dir}/validation_contract.json
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
printf '%s\n' "${last_checkpoint}" > "${out_dir}/last_checkpoint.txt"
