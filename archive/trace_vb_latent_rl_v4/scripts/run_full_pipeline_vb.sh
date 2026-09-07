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
[[ "${evidence_gpu}" =~ ^[0-9]+$ ]] || trace_vb_die "invalid evidence GPU: ${evidence_gpu}"
TRAIN_SEED=${TRAIN_SEED:-0}
pipeline_tag=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_full_seed${TRAIN_SEED}}
pipeline_dir=${TRACE_VB_ARTIFACT_ROOT}/pipelines/${pipeline_tag}
training_root=${TRACE_VB_ARTIFACT_ROOT}/training
evidence_root=${TRACE_VB_ARTIFACT_ROOT}/evidence
tmp_root=${TRACE_VB_ARTIFACT_ROOT}/tmp
stage1_tag=${pipeline_tag}_stage1
stage2_tag=${pipeline_tag}_stage2
evidence_tag=${pipeline_tag}_evidence

trace_vb_require_four_gpus "${physical_gpus}"
trace_vb_require_stage0
mkdir -p "${pipeline_dir}" "${training_root}" "${evidence_root}" "${tmp_root}"
cd "${CODE_ROOT}"

for script in \
  scripts/trace_vb_common.sh \
  scripts/prepare_sufficiency_cache_vb.sh \
  scripts/run_stage1_vb.sh \
  scripts/run_stage2_vb.sh \
  scripts/run_evidence_vb.sh \
  scripts/wait_for_four_gpus_and_run_full_vb.sh \
  scripts/run_full_pipeline_vb.sh; do
  bash -n "${script}"
done

TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" -m unittest discover -s tests \
  > "${pipeline_dir}/unit_tests.log" 2>&1
TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" tools/data_contract_audit.py \
  > "${pipeline_dir}/data_contract_audit.json"
TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" tools/trace_vb_pipeline_contract_audit.py \
  > "${pipeline_dir}/pipeline_contract_audit.json"

bash "${SCRIPT_DIR}/prepare_sufficiency_cache_vb.sh" "${evidence_gpu}"
cp "$(dirname "${TRACE_VB_SUFFICIENCY_CACHE}")/audit.json" \
  "${pipeline_dir}/sufficiency_cache_audit.json"
if [[ -s "$(dirname "${TRACE_VB_SUFFICIENCY_CACHE}")/pilot256_audit.json" ]]; then
  cp "$(dirname "${TRACE_VB_SUFFICIENCY_CACHE}")/pilot256_audit.json" \
    "${pipeline_dir}/sufficiency_pilot256_audit.json"
fi

sha256sum \
  run.py \
  src/models/trace_vb.py \
  src/models/trace_policy.py \
  src/modules/trace_vb.py \
  src/modules/trace_policy.py \
  src/datasets/gsm8k_aug_nl.py \
  src/configs/models/trace_vb_policy_qwen3_instruct.yaml \
  src/configs/datasets/gsm8k_aug_nl.yaml \
  src/configs/trainer/trace_vb_stage1_v2.yaml \
  src/configs/trainer/default.yaml \
  scripts/trace_vb_common.sh \
  scripts/prepare_sufficiency_cache_vb.sh \
  scripts/run_stage1_vb.sh \
  scripts/run_stage2_vb.sh \
  scripts/run_evidence_vb.sh \
  scripts/wait_for_four_gpus_and_run_full_vb.sh \
  scripts/run_full_pipeline_vb.sh \
  tools/build_trace_vb_sufficiency_cache.py \
  tools/trace_vb_pipeline_contract_audit.py \
  tools/trace_vb_stage1_smoke.py \
  tools/trace_vb_stage1_ddp_smoke.py \
  tools/trace_vb_stage2_smoke.py \
  tools/trace_vb_stage2_ddp_smoke.py \
  tools/trace_policy_task_summary.py \
  tools/trace_policy_geometry_summary.py \
  tools/trace_policy_stage_comparison.py \
  tools/trace_policy_causal_summary.py \
  tools/verify_evidence_complete.py \
  > "${pipeline_dir}/source_sha256.txt"

cat > "${pipeline_dir}/manifest.txt" <<EOF
model=TRACE-VB-v4
full_name=Role-Structured_Latent_Reasoning_with_Semantic-to-Outcome_Value_Bridging
project_root=${CODE_ROOT}
data_root=${TRACE_VB_DATA_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
model_config=${TRACE_VB_MODEL_CONFIG}
pipeline=registered_Stage0_to_text_grounded_VB_Stage1_to_terminal_GAE_PPO_with_decayed_semantic_anchor_to_complete_paired_evidence
pipeline_tag=${pipeline_tag}
physical_training_gpus=${physical_gpus}
cache_and_evidence_gpu=${evidence_gpu}
minimum_free_gpu_memory_mib=${TRACE_VB_MIN_FREE_GPU_MIB}
ddp_smoke_minimum_headroom_mib=${TRACE_VB_DDP_MIN_HEADROOM_MIB:-4096}
train_seed=${TRAIN_SEED}
stage0_checkpoint=${TRACE_VB_REGISTERED_STAGE0}
stage0_sha256=${TRACE_VB_REGISTERED_STAGE0_SHA256}
sufficiency_cache=${TRACE_VB_SUFFICIENCY_CACHE}
sufficiency_cache_sha256=$(sha256sum "${TRACE_VB_SUFFICIENCY_CACHE}" | awk '{print $1}')
latent_roles=PLAN,SOLVE1,SOLVE2,SOLVE3,SOLVE4,SOLVE5,REFINE,COMMIT
answer_context=question_plus_COMMIT_latent_only
private_latent_answer_access=false
epoch1_minimum_full_validation_accuracy=${TRACE_VB_EPOCH1_MIN_ACCURACY}
stage1_epochs=10_full_train_and_747_validation
stage2_epochs=10_2048_questions_and_full_747_validation
stage2_group_size=8
rollout_micro_batch_size=1
critic_warmup_rollout_batches=256
policy_update_epochs=4
stage2_scheduler_steps=20480
rl_reward=terminal_exact_correctness_only
stage1_step_supervision=PLAN_forecast_SOLVE1_5_residual_to_text_decode_REFINE_endpoint
stage1_solve_text_partition=complete_sample_local_gold_CoT_tokens_balanced_contiguous_SOLVE1_5
stage1_solve_text_coverage=every_gold_CoT_token_exactly_once
stage1_solve_text_truncation=forbidden_fail_closed
stage2_step_credit=role_critic_plus_terminal_GAE_without_shaped_step_reward
stage2_semantic_anchor=frozen_Stage1_decoder_on_deterministic_mean_path
stage2_semantic_anchor_reward_shaping=false
stage2_single_gpu_smoke=required_before_ddp_smoke
stage2_four_rank_ddp_smoke=one_critic_warmup_plus_two_actor_critic_updates_per_rank
test_times=1
started_at=$(date --iso-8601=seconds)
EOF

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  TMPDIR="${tmp_root}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${evidence_gpu}" \
  "${TRACE_VB_PYTHON}" tools/trace_vb_stage1_smoke.py \
    --stage0-checkpoint "${TRACE_VB_REGISTERED_STAGE0}" \
    --stress-cases 3 \
    --output "${pipeline_dir}/stage1_single_gpu_smoke.json" \
    2>&1 | tee "${pipeline_dir}/stage1_single_gpu_smoke.log"

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  TMPDIR="${tmp_root}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${TRACE_VB_PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    tools/trace_vb_stage1_ddp_smoke.py \
    --stage0-checkpoint "${TRACE_VB_REGISTERED_STAGE0}" \
    --optimizer-steps 2 \
    --output "${pipeline_dir}/stage1_four_rank_smoke.json" \
    2>&1 | tee "${pipeline_dir}/stage1_four_rank_smoke.log"

RUN_TAG="${stage1_tag}" TRAIN_SEED="${TRAIN_SEED}" \
  bash "${SCRIPT_DIR}/run_stage1_vb.sh" \
    "${physical_gpus}" "${TRACE_VB_REGISTERED_STAGE0}"
stage1_checkpoint=$(<"${training_root}/${stage1_tag}/best_checkpoint.txt")

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  TMPDIR="${tmp_root}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${evidence_gpu}" \
  "${TRACE_VB_PYTHON}" tools/trace_vb_stage2_smoke.py \
    --stage1-checkpoint "${stage1_checkpoint}" \
    --output "${pipeline_dir}/stage2_smoke.json" \
    2>&1 | tee "${pipeline_dir}/stage2_smoke.log"

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  TMPDIR="${tmp_root}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${TRACE_VB_PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    tools/trace_vb_stage2_ddp_smoke.py \
    --stage1-checkpoint "${stage1_checkpoint}" \
    --actor-updates 2 \
    --output "${pipeline_dir}/stage2_four_rank_smoke.json" \
    2>&1 | tee "${pipeline_dir}/stage2_four_rank_smoke.log"

RUN_TAG="${stage2_tag}" TRAIN_SEED="${TRAIN_SEED}" \
  bash "${SCRIPT_DIR}/run_stage2_vb.sh" \
    "${physical_gpus}" "${stage1_checkpoint}"
stage2_checkpoint=$(<"${training_root}/${stage2_tag}/best_checkpoint.txt")

RUN_TAG="${evidence_tag}" OUT="${evidence_root}/${evidence_tag}" \
  bash "${SCRIPT_DIR}/run_evidence_vb.sh" \
    "${evidence_gpu}" "${stage1_checkpoint}" "${stage2_checkpoint}"

complete_record=${evidence_root}/${evidence_tag}/COMPLETE.json
[[ -s "${complete_record}" ]] || trace_vb_die "complete evidence gate did not produce COMPLETE.json"
"${TRACE_VB_PYTHON}" tools/verify_evidence_complete.py \
  --evidence-root "${evidence_root}/${evidence_tag}" \
  --stage1-checkpoint "${stage1_checkpoint}" \
  --final-checkpoint "${stage2_checkpoint}" \
  > "${pipeline_dir}/evidence_completeness_recheck.json"
cp "${complete_record}" "${pipeline_dir}/COMPLETE.json"

cat >> "${pipeline_dir}/manifest.txt" <<EOF
stage1_checkpoint=${stage1_checkpoint}
stage2_checkpoint=${stage2_checkpoint}
evidence_dir=${evidence_root}/${evidence_tag}
complete_record=${pipeline_dir}/COMPLETE.json
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${TRACE_VB_REGISTERED_STAGE0}" > "${pipeline_dir}/stage0_best.txt"
printf '%s\n' "${stage1_checkpoint}" > "${pipeline_dir}/stage1_best.txt"
printf '%s\n' "${stage2_checkpoint}" > "${pipeline_dir}/stage2_best.txt"
