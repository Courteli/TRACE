#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> [v7-validation-best-checkpoint]" >&2
  exit 2
fi
physical_gpus=$1
v7_best_checkpoint=${2:-${V7_BEST_CKPT:-}}
[[ -n "${v7_best_checkpoint}" ]] || \
  trace_vb_die "V7_BEST_CKPT (or the second argument) is required"

TRAIN_SEED=${TRAIN_SEED:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v8_stage1_seed${TRAIN_SEED}}
STAGE1_RESUME_CKPT=${STAGE1_RESUME_CKPT:-}
V7_STAGE1_DIR=${V7_STAGE1_DIR:-}
trace_vb_require_safe_tag "${RUN_TAG}"
[[ "${TRAIN_SEED}" =~ ^[0-9]+$ ]] || trace_vb_die "TRAIN_SEED must be non-negative"

CODE_ROOT=${TRACE_VB_CODE_ROOT}
DATASET_DIR=${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL
LOG_ROOT=${TRACE_VB_ARTIFACT_ROOT}/logs
TMP_ROOT=${TRACE_VB_ARTIFACT_ROOT}/tmp
OUT_DIR=${TRACE_VB_ARTIFACT_ROOT}/training/${RUN_TAG}
LOG_PARENT=${LOG_ROOT}/${TRACE_VB_MODEL_CONFIG}/gsm8k_aug_nl-gsm8k_aug_nl
CANDIDATE_DIR=${OUT_DIR}/candidates
CHECKPOINT_DIR=${OUT_DIR}/checkpoints
METRIC_SAFE_BASELINE_ARTIFACT=${OUT_DIR}/metric_safe_baseline.json
REGISTERED_CAPABILITY_VALIDATION_ARTIFACT=${OUT_DIR}/registered_capability_validation.json
INTERVAL_BATCHES=512
FORMAL_INTERVALS=4
TOTAL_STAGE1_STEPS=2048
EXPECTED_STEPS=0,512,1024,1536,2048

trace_vb_require_four_gpus "${physical_gpus}"
trace_vb_require_capability
trace_vb_require_metric_artifacts
trace_vb_require_cot_encoder
trace_vb_require_sufficiency_cache
[[ -f "${v7_best_checkpoint}" ]] || \
  trace_vb_die "missing v7 validation-best checkpoint: ${v7_best_checkpoint}"
V7_SOURCE_SHA256=$(sha256sum "${v7_best_checkpoint}" | awk '{print $1}')
[[ -n "${V7_STAGE1_DIR}" ]] || trace_vb_die "V7_STAGE1_DIR is required"
trace_vb_require_no_v7_training
[[ -z "${RESUME_CKPT_PATH:-}" ]] || \
  trace_vb_die "use STAGE1_RESUME_CKPT; generic RESUME_CKPT_PATH is forbidden"

cd "${CODE_ROOT}"
TRAIN_VAL_AUDIT_TMP=/tmp/trace_vb_v8_train_val_audit_$$.json
V7_TRIGGER_AUDIT_TMP=/tmp/trace_vb_v8_v7_trigger_audit_$$.json
trace_vb_audit_v7_trigger \
  "${V7_STAGE1_DIR}" "${v7_best_checkpoint}" "${V7_TRIGGER_AUDIT_TMP}"
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/audit_train_val_only_v8.py" \
  --dataset-dir "${DATASET_DIR}" \
  --output "${TRAIN_VAL_AUDIT_TMP}"
"${TRACE_VB_PYTHON}" tools/build_trace_vb_sufficiency_cache.py \
  --audit_only \
  --output "${TRACE_VB_SUFFICIENCY_CACHE}" \
  --min_valid_row_fraction 0.50 \
  --min_valid_prefix_fraction 0.20 \
  --min_nonzero_gain_fraction 0.30 \
  --min_score_span_mean 0.05 \
  > /tmp/trace_vb_v8_sufficiency_cache_audit_$$.json

# The formal initialization must be the v7 Stage-1 validation best, not the
# old capability checkpoint itself.  The model-side rewind migrates only the
# capability spine while retaining v7's learned role/posterior/semantic heads.
"${TRACE_VB_PYTHON}" - \
  "${v7_best_checkpoint}" "${DATASET_DIR}" <<'PY'
import sys
from src.utils.safe_checkpoint import safe_load_checkpoint

checkpoint = safe_load_checkpoint(sys.argv[1], map_location="cpu")
if checkpoint.get("trace_vb_schema_version") != "trace_vb_v7":
    raise SystemExit("v8 initialization requires a TRACE-VB-v7 checkpoint")
if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
    raise SystemExit("v8 initialization requires a v7 Stage-1 checkpoint")
state = checkpoint.get("state_dict", {})
required_prefixes = (
    "state_norm.",
    "trajectory_policy.",
    "trajectory_posterior.",
    "posterior_context_norm.",
    "plan_forecaster.",
    "solve_text_decoder.",
    "sufficiency_head.",
)
for prefix in required_prefixes:
    if not any(name.startswith(prefix) for name in state):
        raise SystemExit(f"v7 initialization is missing {prefix}")
for adapter in ("default", "trace_capability"):
    coverage = sum(
        1
        for name in state
        if f".{adapter}." in name
        and (".lora_A." in name or ".lora_B." in name)
        and name.endswith(".weight")
    )
    if coverage != 504:
        raise SystemExit(f"v7 {adapter} LoRA coverage is {coverage}, expected 504")
config = checkpoint.get("hyper_parameters", {}).get("all_config")
if config is None or str(config.model.target) != "src.models.trace_vb.LitTRACEVB":
    raise SystemExit("v7 initialization has the wrong model target")
if str(config.data_module.dataset_dir) != sys.argv[2]:
    raise SystemExit("v7 initialization used a different dataset")
policy = config.model.model_kwargs.trace_policy_config
if str(policy.answer_context_mode) != "question_and_commit":
    raise SystemExit("v7 initialization did not use question+COMMIT")
PY

common_run_args=(
  --model "${TRACE_VB_MODEL_CONFIG}"
  --dataset gsm8k_aug_nl
  --trainer trace_vb_stage1_v2
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
  num_workers=0
  pin_memory=false
  persistent_workers=false
  trainer.num_sanity_val_steps=0
  trainer.limit_val_batches=1.0
  trainer.check_val_every_n_epoch=1
  trainer.val_check_interval=1.0
  trainer.gradient_clip_val=0.3
  save_top_k=0
  save_last=true
  save_weights_only=false
  model.model_kwargs.do_trace_rl=false
  model.model_kwargs.readcot_config.use_hybrid=false
  model.model_kwargs.readcot_config.use_anchor_loss=false
  model.model_kwargs.readcot_config.use_anchor_gate=false
  model.model_kwargs.readcot_config.anchor_text_mode=compact_equation
  model.model_kwargs.readcot_config.compact_anchor_max_chars=64
  model.model_kwargs.trace_policy_config.answer_context_mode=question_and_commit
  model.model_kwargs.trace_policy_config.deployment_compact_reasoning=true
  model.model_kwargs.trace_policy_config.commit_causal_summary=true
  model.model_kwargs.trace_policy_config.validation_path=student_commit
  model.model_kwargs.trace_policy_config.use_capability_anchor=true
  model.model_kwargs.trace_policy_config.capability_expected_lora_tensors=504
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
  model.model_kwargs.trace_policy_config.stage0_expected_lora_tensors=504
  model.model_kwargs.trace_policy_config.stage1_freeze_capability_spine=true
  model.model_kwargs.trace_policy_config.stage1_freeze_path_lora=true
  model.model_kwargs.trace_policy_config.stage1_freeze_capability_query_prior=true
  model.model_kwargs.trace_policy_config.stage1_require_capability_spine_parity=true
  model.model_kwargs.trace_policy_config.stage1_stochastic_paths=1
  model.model_kwargs.trace_policy_config.stage1_posterior_samples=1
  model.model_kwargs.trace_policy_config.stage1_posterior_kl_weight=0.05
  model.model_kwargs.trace_policy_config.stage1_sampled_answer_weight=0.0
  model.model_kwargs.trace_policy_config.stage1_map_compact_weight=0.25
  model.model_kwargs.trace_policy_config.stage1_map_answer_suffix_weight=1.0
  model.model_kwargs.trace_policy_config.stage1_capability_kl_weight=1.0
  model.model_kwargs.trace_policy_config.stage1_capability_kl_scope=full_target
  model.model_kwargs.trace_policy_config.stage1_capability_temperature=2.0
  model.model_kwargs.trace_policy_config.stage1_capability_lora_lr=0.0
  model.model_kwargs.trace_policy_config.stage1_role_lr=2.0e-6
  model.model_kwargs.trace_policy_config.compact_target_max_equations=2
  model.model_kwargs.trace_policy_config.compact_target_max_new_tokens=48
  model.model_kwargs.trace_policy_config.stage1_answer_activation_checkpoint=true
  model.model_kwargs.trace_policy_config.stage1_solve_weight=0.02
  model.model_kwargs.trace_policy_config.stage1_solve_text_weight=0.02
  model.model_kwargs.trace_policy_config.stage1_commit_weight=0.20
  model.model_kwargs.trace_policy_config.solve_text_decoder_hidden_size=512
  model.model_kwargs.trace_policy_config.solve_text_decoder_max_tokens=96
  model.model_kwargs.trace_policy_config.solve_text_decoder_fail_on_truncation=true
  model.model_kwargs.trace_policy_config.solve_text_ce_chunk_size=8
  model.model_kwargs.trace_policy_config.stage1_posterior_activation_offload=false
  model.model_kwargs.trace_policy_config.stage1_recovery_checkpoint_interval=128
  model.model_kwargs.trace_policy_config.stage1_host_memory_guard_interval=10
  model.model_kwargs.trace_policy_config.stage1_maximum_rank_rss_gib=20.0
  model.model_kwargs.trace_policy_config.stage1_minimum_host_available_gib=192.0
  model.model_kwargs.trace_policy_config.sufficiency_cache_path="${TRACE_VB_SUFFICIENCY_CACHE}"
  model.model_kwargs.trace_policy_config.sufficiency_teacher_checkpoint_sha256="${TRACE_VB_SUFFICIENCY_TEACHER_SHA256}"
  model.model_kwargs.trace_policy_config.sufficiency_tokenizer_sha256="${TRACE_VB_SUFFICIENCY_TOKENIZER_SHA256}"
  model.model_kwargs.trace_policy_config.sufficiency_prompt_sha256="${TRACE_VB_SUFFICIENCY_PROMPT_SHA256}"
  model.model_kwargs.trace_policy_config.sufficiency_composite_sha256="${TRACE_VB_SUFFICIENCY_COMPOSITE_SHA256}"
  model.model_kwargs.trace_policy_config.stage0_cot_encoder_checkpoint_sha256="${TRACE_VB_COT_ENCODER_SHA256}"
  model.model_kwargs.trace_policy_config.visual_record_limit=0
  model.model_kwargs.answer_generation_config.max_new_tokens=48
  model.model_kwargs.answer_generation_config.do_sample=false
  model.training_kwargs.optimizer.lr=2.0e-6
  model.training_kwargs.scheduler.warmup_steps=75
  model.training_kwargs.scheduler.num_training_steps="${TOTAL_STAGE1_STEPS}"
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
  local summary=$2
  local checkpoint=$3
  local summary_copy=${CANDIDATE_DIR}/validation_step_$(printf '%04d' "${step}").json
  local record=${CANDIDATE_DIR}/candidate_step_$(printf '%04d' "${step}").json
  cp "${summary}" "${summary_copy}"
  "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/trace_vb_candidate_registry.py" register \
    --phase stage1 \
    --physical-gpus "${physical_gpus}" \
    --step "${step}" \
    --summary "${summary_copy}" \
    --checkpoint "${checkpoint}" \
    --output "${record}" \
    --validation-schema "${TRACE_VB_VALIDATION_SCHEMA}" \
    --questions "${TRACE_VB_VALIDATION_QUESTIONS}" \
    --world-size 4 \
    --registered-checkpoint-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}" \
    --registered-payload-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}" \
    --v7-source-sha256 "${V7_SOURCE_SHA256}" \
    --cot-encoder-sha256 "${TRACE_VB_COT_ENCODER_SHA256}"
}

completed_intervals=0
current_checkpoint=
resume_attempt=
if [[ -n "${STAGE1_RESUME_CKPT}" ]]; then
  [[ -d "${OUT_DIR}" && -s "${OUT_DIR}/manifest.txt" ]] || \
    trace_vb_die "Stage-1 resume requires the existing RUN_TAG output"
  [[ -f "${STAGE1_RESUME_CKPT}" ]] || \
    trace_vb_die "missing Stage-1 resume checkpoint: ${STAGE1_RESUME_CKPT}"
  resume_attempt=${TRACE_VB_RESUME_ATTEMPT:-$(date +%Y%m%d-%H%M%S)}
  trace_vb_require_safe_tag "${resume_attempt}"
  [[ -s "${CANDIDATE_DIR}/candidate_step_0000.json" ]] || \
    trace_vb_die "Stage-1 resume is missing the immutable step-0 candidate"
  trace_vb_require_sha256 \
    "run-local metric-safe baseline" \
    "${METRIC_SAFE_BASELINE_ARTIFACT}" \
    "${TRACE_VB_METRIC_SAFE_BASELINE_SHA256}"
  trace_vb_require_sha256 \
    "run-local registered capability validation" \
    "${REGISTERED_CAPABILITY_VALIDATION_ARTIFACT}" \
    "${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_SHA256}"
  "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/audit_rewind_non_degradation_v8.py" \
    --trigger-contract "${OUT_DIR}/v7_trigger_contract.json" \
    --validation-summary "${CANDIDATE_DIR}/validation_step_0000.json" \
    --validation-schema "${TRACE_VB_VALIDATION_SCHEMA}" \
    --step0-checkpoint "${CHECKPOINT_DIR}/stage1_step0000.ckpt" \
    --metric-safe-baseline "${METRIC_SAFE_BASELINE_ARTIFACT}" \
    --registered-capability-validation "${REGISTERED_CAPABILITY_VALIDATION_ARTIFACT}" \
    --output "${OUT_DIR}/rewind_non_degradation_gate.json"
  for interval in $(seq 1 "${FORMAL_INTERVALS}"); do
    step=$((interval * INTERVAL_BATCHES))
    record=${CANDIDATE_DIR}/candidate_step_$(printf '%04d' "${step}").json
    if [[ -s "${record}" ]]; then
      completed_intervals=${interval}
    else
      break
    fi
  done
  completed_record=${CANDIDATE_DIR}/candidate_step_$(printf '%04d' $((completed_intervals * INTERVAL_BATCHES))).json
  "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/audit_resume_checkpoint_v8.py" \
    --checkpoint "${STAGE1_RESUME_CKPT}" \
    --stage 1 \
    --completed-intervals "${completed_intervals}" \
    --completed-candidate "${completed_record}" \
    --physical-gpus "${physical_gpus}" \
    --stage1-interval-steps "${INTERVAL_BATCHES}" \
    --registered-checkpoint-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}" \
    --registered-payload-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}" \
    --v7-source-sha256 "${V7_SOURCE_SHA256}" \
    --cot-encoder-sha256 "${TRACE_VB_COT_ENCODER_SHA256}" \
    --output "${OUT_DIR}/resume_checkpoint_contract_${resume_attempt}.json"
  current_checkpoint=${STAGE1_RESUME_CKPT}
  cat >> "${OUT_DIR}/manifest.txt" <<EOF
resume_at=$(date --iso-8601=seconds)
resume_checkpoint=${STAGE1_RESUME_CKPT}
resume_checkpoint_sha256=$(sha256sum "${STAGE1_RESUME_CKPT}" | awk '{print $1}')
resume_completed_intervals=${completed_intervals}
resume_attempt=${resume_attempt}
resume_rewind_path_to_capability=false
EOF
else
  [[ ! -e "${OUT_DIR}" ]] || trace_vb_die "refusing to reuse output: ${OUT_DIR}"
  mkdir -p "${OUT_DIR}" "${CANDIDATE_DIR}" "${CHECKPOINT_DIR}" \
    "${TMP_ROOT}" "${LOG_PARENT}"
  cp /tmp/trace_vb_v8_sufficiency_cache_audit_$$.json \
    "${OUT_DIR}/sufficiency_cache_audit.json"
  cp "${TRAIN_VAL_AUDIT_TMP}" "${OUT_DIR}/train_val_data_contract.json"
  cp "${V7_TRIGGER_AUDIT_TMP}" "${OUT_DIR}/v7_trigger_contract.json"
  cp "${TRACE_VB_METRIC_SAFE_BASELINE}" \
    "${METRIC_SAFE_BASELINE_ARTIFACT}"
  cp "${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION}" \
    "${REGISTERED_CAPABILITY_VALIDATION_ARTIFACT}"
  trace_vb_require_sha256 \
    "run-local metric-safe baseline" \
    "${METRIC_SAFE_BASELINE_ARTIFACT}" \
    "${TRACE_VB_METRIC_SAFE_BASELINE_SHA256}"
  trace_vb_require_sha256 \
    "run-local registered capability validation" \
    "${REGISTERED_CAPABILITY_VALIDATION_ARTIFACT}" \
    "${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_SHA256}"
  cat > "${OUT_DIR}/manifest.txt" <<EOF
model=${TRACE_VB_VERSION}
phase=stage1_capability_spine_preserving_role_formation
run_tag=${RUN_TAG}
checkpoint_schema=${TRACE_VB_CHECKPOINT_SCHEMA}
validation_schema=${TRACE_VB_VALIDATION_SCHEMA}
project_root=${CODE_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
v7_best_checkpoint=${v7_best_checkpoint}
v7_best_checkpoint_sha256=$(sha256sum "${v7_best_checkpoint}" | awk '{print $1}')
v7_stage1_dir=${V7_STAGE1_DIR}
v7_trigger_contract=${OUT_DIR}/v7_trigger_contract.json
registered_capability=${TRACE_VB_REGISTERED_CAPABILITY}
registered_capability_sha256=${TRACE_VB_REGISTERED_CAPABILITY_SHA256}
registered_capability_payload_tensors=${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_TENSORS}
registered_capability_payload_sha256=${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}
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
cot_encoder_checkpoint=${TRACE_VB_COT_ENCODER}
cot_encoder_checkpoint_sha256=${TRACE_VB_COT_ENCODER_SHA256}
cot_encoder_strict_validation=${TRACE_VB_COT_ENCODER_VALIDATION_CORRECT}_of_${TRACE_VB_VALIDATION_QUESTIONS}
sufficiency_cache=${TRACE_VB_SUFFICIENCY_CACHE}
sufficiency_teacher_provenance=${TRACE_VB_SUFFICIENCY_TEACHER}
sufficiency_teacher_sha256=${TRACE_VB_SUFFICIENCY_TEACHER_SHA256}
sufficiency_composite_sha256=${TRACE_VB_SUFFICIENCY_COMPOSITE_SHA256}
information_bridge=question_plus_COMMIT_only
latent_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT
initialization=retain_v7_posterior_semantic_decoder_policy_features_then_one_time_capability_spine_rewind_and_exact_9_tensor_zero_action_reset
zero_action_reset=mean_heads_plan_solve_check_refine_alias_commit_weight_bias_plus_action_projector_bias
zero_action_reset_operation_count=1
zero_action_reset_tensor_count=9
deployment_lora=frozen
capability_query_prior=frozen
compact_target=last_two_complete_causal_equations_plus_answer
compact_anchor_max_chars=64_complete_equation_no_character_slice
capability_KL=full_target_temperature_2_weight_1.0
answer_suffix_CE_weight=1.0
compact_CE_weight=0.25
solve_text_CE_weight=0.02
commit_alignment_weight=0.20
role_lr=2e-6
interval_batches=${INTERVAL_BATCHES}
formal_intervals=${FORMAL_INTERVALS}
total_stage1_steps=${TOTAL_STAGE1_STEPS}
recovery_checkpoint_interval_batches=128
recovery_checkpoint_policy=diagnostic_only_not_formal_resume_validated_boundary_rollback
full_validation_steps=${EXPECTED_STEPS}
candidate_selection=maximum_exact_integer_correct_then_earliest_step
test_split=false
evidence_pipeline=false
physical_gpus=${physical_gpus}
formal_gpus=${TRACE_VB_FORMAL_GPUS}
fixed_gpus=${TRACE_VB_FIXED_GPUS}
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF

  # This is the one and only rewind call.  It starts from the formal v7 best,
  # replaces only the capability spine, installs the stronger CoT encoder,
  # performs strict 747-question validation, and saves an explicit step-0
  # non-degradation checkpoint.  No optimizer step is taken.
  initial_tag=${RUN_TAG}_stage1_step0000
  initial_checkpoint=${CHECKPOINT_DIR}/stage1_step0000.ckpt
  run_python \
    "${common_run_args[@]}" \
    --load_ckpt_path "${v7_best_checkpoint}" \
    --cot_encoder_ckpt_path "${TRACE_VB_COT_ENCODER}" \
    --rewind_path_to_capability \
    --registered_capability_ckpt_path "${TRACE_VB_REGISTERED_CAPABILITY}" \
    --validate_only \
    --save_validation_checkpoint "${initial_checkpoint}" \
    --log_suffix "${initial_tag}" \
    trainer.max_epochs=1 \
    trainer.max_steps=-1 \
    trainer.limit_train_batches=1 \
    trainer.default_root_dir="${OUT_DIR}/trainer_step0000" \
    2>&1 | tee -a "${OUT_DIR}/train.log"
  [[ -f "${initial_checkpoint}" ]] || \
    trace_vb_die "initial validation did not save its step-0 checkpoint"
  initial_logger=$(trace_vb_find_single_logger_dir "${LOG_PARENT}" "${initial_tag}")
  initial_summary=${initial_logger}/validation_epoch_000.json
  "${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/audit_rewind_non_degradation_v8.py" \
    --trigger-contract "${OUT_DIR}/v7_trigger_contract.json" \
    --validation-summary "${initial_summary}" \
    --validation-schema "${TRACE_VB_VALIDATION_SCHEMA}" \
    --step0-checkpoint "${initial_checkpoint}" \
    --metric-safe-baseline "${METRIC_SAFE_BASELINE_ARTIFACT}" \
    --registered-capability-validation "${REGISTERED_CAPABILITY_VALIDATION_ARTIFACT}" \
    --output "${OUT_DIR}/rewind_non_degradation_gate.json"
  register_candidate 0 "${initial_summary}" "${initial_checkpoint}"
  current_checkpoint=${initial_checkpoint}
  cat >> "${OUT_DIR}/manifest.txt" <<EOF
step_0_checkpoint=${initial_checkpoint}
step_0_checkpoint_sha256=$(sha256sum "${initial_checkpoint}" | awk '{print $1}')
step_0_rewind_path_to_capability=true
step_0_optimizer_updates=0
EOF
fi
rm -f /tmp/trace_vb_v8_sufficiency_cache_audit_$$.json
rm -f "${TRAIN_VAL_AUDIT_TMP}"
rm -f "${V7_TRIGGER_AUDIT_TMP}"

for interval in $(seq 1 "${FORMAL_INTERVALS}"); do
  if (( interval <= completed_intervals )); then
    continue
  fi
  step=$((interval * INTERVAL_BATCHES))
  interval_tag=${RUN_TAG}_stage1_step$(printf '%04d' "${step}")
  if [[ -n "${resume_attempt}" ]]; then
    interval_tag=${interval_tag}_resume_${resume_attempt}
  fi
  checkpoint_args=(--resume_ckpt_path "${current_checkpoint}")
  if (( interval == 1 && completed_intervals == 0 )); then
    # Boundary zero is validation-only and intentionally has no optimizer
    # state.  This remains a weights-only phase restart after interruption.
    checkpoint_args=(--load_ckpt_path "${current_checkpoint}")
  fi
  # Deliberately absent here and on every recovery path:
  # --rewind_path_to_capability
  run_python \
    "${common_run_args[@]}" \
    "${checkpoint_args[@]}" \
    --log_suffix "${interval_tag}" \
    trainer.max_epochs="${interval}" \
    trainer.max_steps=-1 \
    trainer.limit_train_batches="${INTERVAL_BATCHES}" \
    trainer.default_root_dir="${OUT_DIR}/trainer_step$(printf '%04d' "${step}")" \
    2>&1 | tee -a "${OUT_DIR}/train.log"
  interval_logger=$(trace_vb_find_single_logger_dir "${LOG_PARENT}" "${interval_tag}")
  summary=${interval_logger}/validation_epoch_$(printf '%03d' $((interval - 1))).json
  checkpoint=${interval_logger}/checkpoints/last.ckpt
  [[ -f "${checkpoint}" ]] || \
    trace_vb_die "Stage-1 interval ${interval} produced no last checkpoint"
  register_candidate "${step}" "${summary}" "${checkpoint}"
  # The registered last.ckpt is the sole candidate for this interval.  Remove
  # only recovery artifacts created inside this completed v8 logger directory;
  # never touch the candidate, another run, or an external checkpoint.
  find "${interval_logger}/checkpoints" -maxdepth 1 -type f \
    \( -name 'recovery-*.ckpt' -o -name 'host-memory-guard-*.ckpt' \) \
    -delete
  current_checkpoint=${checkpoint}
  cat >> "${OUT_DIR}/manifest.txt" <<EOF
step_${step}_checkpoint=${checkpoint}
step_${step}_checkpoint_sha256=$(sha256sum "${checkpoint}" | awk '{print $1}')
step_${step}_rewind_path_to_capability=false
step_${step}_validated_at=$(date --iso-8601=seconds)
EOF
done

mapfile -t candidate_records < <(
  find "${CANDIDATE_DIR}" -maxdepth 1 -type f \
    -name 'candidate_step_*.json' -print | sort
)
"${TRACE_VB_PYTHON}" "${SCRIPT_DIR}/trace_vb_candidate_registry.py" finalize-completed \
  --phase stage1 \
  --physical-gpus "${physical_gpus}" \
  --expected-steps "${EXPECTED_STEPS}" \
  --candidates "${candidate_records[@]}" \
  --registered-checkpoint-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}" \
  --registered-payload-sha256 "${TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256}" \
  --v7-source-sha256 "${V7_SOURCE_SHA256}" \
  --cot-encoder-sha256 "${TRACE_VB_COT_ENCODER_SHA256}" \
  --index "${OUT_DIR}/candidate_index.json" \
  --best-checkpoint-record "${OUT_DIR}/best_checkpoint.txt" \
  --last-checkpoint-record "${OUT_DIR}/last_checkpoint.txt" \
  --manifest "${OUT_DIR}/manifest.txt" \
  --expected-last-checkpoint "${current_checkpoint}"
