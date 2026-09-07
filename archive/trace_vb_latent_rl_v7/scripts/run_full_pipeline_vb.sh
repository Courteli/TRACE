#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> [cache-and-evidence-physical-gpu]" >&2
  exit 2
fi

physical_gpus=$1
evidence_gpu=${2:-${physical_gpus%%,*}}
[[ "${evidence_gpu}" =~ ^[0-9]+$ ]] || \
  trace_vb_die "invalid cache/evidence GPU: ${evidence_gpu}"
case ",${physical_gpus}," in
  *",${evidence_gpu},"*) ;;
  *) trace_vb_die "cache/evidence GPU must be one of the four training GPUs" ;;
esac

TRAIN_SEED=${TRAIN_SEED:-0}
pipeline_tag=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v7_full_seed${TRAIN_SEED}}
[[ "${TRAIN_SEED}" =~ ^[0-9]+$ ]] || trace_vb_die "TRAIN_SEED must be non-negative"
[[ "${pipeline_tag}" =~ ^[A-Za-z0-9._-]+$ ]] || \
  trace_vb_die "PIPELINE_TAG contains unsafe path or glob characters: ${pipeline_tag}"
pipeline_dir=${TRACE_VB_ARTIFACT_ROOT}/pipelines/${pipeline_tag}
training_root=${TRACE_VB_ARTIFACT_ROOT}/training
evidence_root=${TRACE_VB_ARTIFACT_ROOT}/evidence
tmp_root=${TRACE_VB_ARTIFACT_ROOT}/tmp
stage1_tag=${pipeline_tag}_stage1
stage2_tag=${pipeline_tag}_stage2
evidence_tag=${pipeline_tag}_evidence

trace_vb_require_four_gpus "${physical_gpus}"
trace_vb_require_capability
trace_vb_require_stage0
[[ ! -e "${pipeline_dir}" ]] || \
  trace_vb_die "refusing to reuse pipeline directory: ${pipeline_dir}"
[[ ! -e "${training_root}/${stage1_tag}" ]] || \
  trace_vb_die "refusing to reuse Stage-1 tag: ${stage1_tag}"
[[ ! -e "${training_root}/${stage2_tag}" ]] || \
  trace_vb_die "refusing to reuse Stage-2 tag: ${stage2_tag}"
[[ ! -e "${evidence_root}/${evidence_tag}" ]] || \
  trace_vb_die "refusing to reuse evidence tag: ${evidence_tag}"

mkdir -p "${pipeline_dir}" "${training_root}" "${evidence_root}" "${tmp_root}"
cd "${CODE_ROOT}"

for script in \
  scripts/trace_vb_common.sh \
  scripts/prepare_sufficiency_cache_vb.sh \
  scripts/run_preflight_validation_v7.sh \
  scripts/run_stage1_vb.sh \
  scripts/run_stage2_vb.sh \
  scripts/run_evidence_vb.sh \
  scripts/run_full_pipeline_vb.sh; do
  bash -n "${script}"
done

# Every checkpoint-loading entry point in the formal chain must opt into the
# restricted loader. This prevents a stale downstream script from silently
# weakening the v7 safety contract.
for script in \
  scripts/prepare_sufficiency_cache_vb.sh \
  scripts/run_preflight_validation_v7.sh \
  scripts/run_stage1_vb.sh \
  scripts/run_stage2_vb.sh \
  scripts/run_evidence_vb.sh; do
  grep -Fq "TORCH_FORCE_WEIGHTS_ONLY_LOAD=1" "${script}" || \
    trace_vb_die "formal script lacks restricted checkpoint loading: ${script}"
done
grep -Fq "trace_vb_v7" scripts/run_stage1_vb.sh || \
  trace_vb_die "Stage-1 launcher does not declare the v7 schema"
grep -Fq "trace_vb_v7" scripts/run_stage2_vb.sh || \
  trace_vb_die "Stage-2 launcher does not declare the v7 schema"
grep -Fq "trace_vb_v7" scripts/run_evidence_vb.sh || \
  trace_vb_die "evidence launcher does not declare the v7 schema"

# The formal chain is gated by the complete repository test suite and the
# immutable registered-data audit. No training process starts if either fails.
env \
  TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" -m unittest discover -s tests \
  > "${pipeline_dir}/unit_tests.log" 2>&1
env \
  TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" tools/data_contract_audit.py \
  > "${pipeline_dir}/data_contract_audit.json"

bash "${SCRIPT_DIR}/prepare_sufficiency_cache_vb.sh" "${evidence_gpu}"
trace_vb_require_cache
cp "$(dirname "${TRACE_VB_SUFFICIENCY_CACHE}")/audit.json" \
  "${pipeline_dir}/sufficiency_cache_audit.json"
cache_validation_contract="$(dirname "${TRACE_VB_SUFFICIENCY_CACHE}")/trace_vb_v7_validation_source_contract.json"
[[ -s "${cache_validation_contract}" ]] || \
  trace_vb_die "cache preparation did not publish the 747 source-id contract"
cp "${cache_validation_contract}" \
  "${pipeline_dir}/validation_source_id_contract.json"
if [[ -s "$(dirname "${TRACE_VB_SUFFICIENCY_CACHE}")/pilot256_audit.json" ]]; then
  cp "$(dirname "${TRACE_VB_SUFFICIENCY_CACHE}")/pilot256_audit.json" \
    "${pipeline_dir}/sufficiency_pilot256_audit.json"
fi

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
  tools/build_trace_vb_sufficiency_cache.py \
  tools/trace_policy_task_summary.py \
  tools/trace_policy_geometry_summary.py \
  tools/trace_policy_stage_comparison.py \
  tools/trace_policy_causal_summary.py \
  tools/verify_evidence_complete.py \
  tests/test_trace_policy.py \
  tests/test_trace_vb_model_contract.py \
  > "${pipeline_dir}/source_sha256.txt"

cat > "${pipeline_dir}/manifest.txt" <<EOF
model=TRACE-VB-v7
full_name=Capability-Anchored_Role-Structured_Latent_Reasoning_with_Role-Local_Credit
project_root=${CODE_ROOT}
data_root=${TRACE_VB_DATA_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
model_config=${TRACE_VB_MODEL_CONFIG}
pipeline=registered_capability_to_role_semantic_Stage1_to_actor_only_role_local_Stage2_to_complete_paired_evidence
pipeline_tag=${pipeline_tag}
physical_training_gpus=${physical_gpus}
cache_and_evidence_gpu=${evidence_gpu}
minimum_free_gpu_memory_mib=${TRACE_VB_MIN_FREE_GPU_MIB}
train_seed=${TRAIN_SEED}
capability_checkpoint=${TRACE_VB_REGISTERED_CAPABILITY}
capability_checkpoint_sha256=${TRACE_VB_REGISTERED_CAPABILITY_SHA256}
capability_verified_validation=540_of_747
cot_encoder_checkpoint=${TRACE_VB_REGISTERED_STAGE0}
cot_encoder_checkpoint_sha256=${TRACE_VB_REGISTERED_STAGE0_SHA256}
cot_encoder_verified_validation=0.848
sufficiency_cache=${TRACE_VB_SUFFICIENCY_CACHE}
sufficiency_cache_sha256=$(sha256sum "${TRACE_VB_SUFFICIENCY_CACHE}" | awk '{print $1}')
latent_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT
answer_context=question_plus_COMMIT
private_latent_answer_access=false
stage1=capability_preserving_role_semantic_CoT_SFT
stage1_step_supervision=PLAN_forecast_SOLVE1_5_text_semantics_REFINE_endpoint_COMMIT_alignment
stage1_training=10_full_epochs_6726_questions_each
stage1_validation=every_epoch_four_rank_gather_then_exact_747_unique_questions
stage1_to_stage2_gate=selected_best_at_least_537_of_747
stage1_behavior_gate=validity_diversity_mode_fraction_nonempty_output
stage2=actor_only_group_relative_latent_policy_optimization
stage2_terminal_signal=exact_answer_correctness
stage2_role_local_signal=discounted_existing_gold_CoT_semantic_agreement
stage2_role_local_weight=0.15
stage2_role_local_discount=0.90
stage2_group_size=8
stage2_training=10_epochs_2048_unique_questions_each
stage2_policy_updates_per_rollout=2_maximum_with_KL_stop
stage2_trainable_features=policy_step_embedding_shared_policy_trunk
stage2_trainable_heads=seven_stochastic_role_mean_and_log_std_heads
stage2_validation=every_epoch_four_rank_gather_then_exact_747_unique_questions
stage2_checkpoint_selection=exact_unrounded_full_validation_accuracy
stage2_behavior_gate=validity_diversity_mode_fraction_nonempty_output
answer_generation=deterministic_max_48_tokens
checkpoint_loading=restricted_weights_only_with_data_only_OmegaConf_allowlist
preflight=complete_unit_test_suite_plus_registered_data_audit_plus_cache_audit
capability_parity_gate=frozen_all_role_teacher_must_equal_540_of_747
student_initial_gate=question_plus_COMMIT_must_reach_at_least_449_of_747
evidence=paired_stage1_vs_stage2_full_task_OOD_geometry_and_causal_suite
test_times=1
started_at=$(date --iso-8601=seconds)
EOF

capability_preflight_tag=${pipeline_tag}_capability_preflight
RUN_TAG="${capability_preflight_tag}" \
  bash "${SCRIPT_DIR}/run_preflight_validation_v7.sh" \
    "${physical_gpus}" capability
cp "${TRACE_VB_ARTIFACT_ROOT}/preflight/${capability_preflight_tag}/validation_gate.json" \
  "${pipeline_dir}/capability_parity_gate.json"

student_preflight_tag=${pipeline_tag}_student_preflight
RUN_TAG="${student_preflight_tag}" \
  bash "${SCRIPT_DIR}/run_preflight_validation_v7.sh" \
    "${physical_gpus}" student
cp "${TRACE_VB_ARTIFACT_ROOT}/preflight/${student_preflight_tag}/validation_gate.json" \
  "${pipeline_dir}/student_initial_gate.json"

RUN_TAG="${stage1_tag}" TRAIN_SEED="${TRAIN_SEED}" \
  bash "${SCRIPT_DIR}/run_stage1_vb.sh" \
    "${physical_gpus}" "${TRACE_VB_REGISTERED_CAPABILITY}"
stage1_record=${training_root}/${stage1_tag}/best_checkpoint.txt
[[ -s "${stage1_record}" ]] || \
  trace_vb_die "Stage 1 did not publish a validation-selected checkpoint"
stage1_checkpoint=$(<"${stage1_record}")
[[ -f "${stage1_checkpoint}" ]] || \
  trace_vb_die "Stage-1 best-checkpoint record points to a missing file"

# run_stage2_vb.sh revalidates all ten Stage-1 747-question summaries and the
# integer 537/747 threshold before it can launch a single optimizer step.
RUN_TAG="${stage2_tag}" TRAIN_SEED="${TRAIN_SEED}" \
  bash "${SCRIPT_DIR}/run_stage2_vb.sh" \
    "${physical_gpus}" "${stage1_checkpoint}"
stage2_record=${training_root}/${stage2_tag}/best_checkpoint.txt
[[ -s "${stage2_record}" ]] || \
  trace_vb_die "Stage 2 did not publish a validation-selected checkpoint"
stage2_checkpoint=$(<"${stage2_record}")
[[ -f "${stage2_checkpoint}" ]] || \
  trace_vb_die "Stage-2 best-checkpoint record points to a missing file"

cp "${training_root}/${stage2_tag}/stage1_validation_gate.json" \
  "${pipeline_dir}/stage1_validation_gate.json"
cp "${training_root}/${stage2_tag}/validation_contract.json" \
  "${pipeline_dir}/stage2_validation_contract.json"

RUN_TAG="${evidence_tag}" OUT="${evidence_root}/${evidence_tag}" \
  bash "${SCRIPT_DIR}/run_evidence_vb.sh" \
    "${evidence_gpu}" "${stage1_checkpoint}" "${stage2_checkpoint}"

complete_record=${evidence_root}/${evidence_tag}/COMPLETE.json
[[ -s "${complete_record}" ]] || \
  trace_vb_die "complete evidence gate did not produce COMPLETE.json"
strict_validation_summary=${evidence_root}/${evidence_tag}/validation747_summary.json
[[ -s "${strict_validation_summary}" ]] || \
  trace_vb_die "evidence did not publish paired 747 source-id validation"
env TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
  "${TRACE_VB_PYTHON}" tools/verify_evidence_complete.py \
    --evidence-root "${evidence_root}/${evidence_tag}" \
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
evidence_dir=${evidence_root}/${evidence_tag}
complete_record=${pipeline_dir}/COMPLETE.json
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${TRACE_VB_REGISTERED_CAPABILITY}" \
  > "${pipeline_dir}/capability_checkpoint.txt"
printf '%s\n' "${stage1_checkpoint}" > "${pipeline_dir}/stage1_best.txt"
printf '%s\n' "${stage2_checkpoint}" > "${pipeline_dir}/stage2_best.txt"
