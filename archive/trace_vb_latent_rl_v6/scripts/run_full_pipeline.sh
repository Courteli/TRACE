#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_ROOT=${DATA_ROOT:-/disk1/dingxukai/TRACE}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/disk1/dingxukai/TRACE/role_semantic_runs}
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
RUN_ROOT=${RUN_ROOT:-${ARTIFACT_ROOT}/training}
PIPELINE_ROOT=${PIPELINE_ROOT:-${ARTIFACT_ROOT}/pipelines}
EVIDENCE_ROOT=${EVIDENCE_ROOT:-${ARTIFACT_ROOT}/evidence}
TMP_ROOT=${TMP_ROOT:-${ARTIFACT_ROOT}/tmp}
LOG_ROOT=${LOG_ROOT:-${ARTIFACT_ROOT}/logs}
TRAIN_SEED=${TRAIN_SEED:-0}

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage0-checkpoint> [evidence-physical-gpu]" >&2
  exit 2
fi
physical_gpus=$1
STAGE0_CKPT_OVERRIDE=$2
evidence_gpu=${3:-${physical_gpus%%,*}}
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "Formal TRACE requires exactly four training GPUs" >&2
  exit 2
fi

if [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi
if [[ ! -f "${STAGE0_CKPT_OVERRIDE}" ]]; then
  echo "Missing explicit fresh CoT-SFT checkpoint: ${STAGE0_CKPT_OVERRIDE}" >&2
  exit 2
fi
pipeline_tag=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_seed${TRAIN_SEED}}
training_pipeline_tag=${pipeline_tag}
pipeline_dir="${PIPELINE_ROOT}/${pipeline_tag}"
stage1_tag="${training_pipeline_tag}_stage1"
stage2_tag="${training_pipeline_tag}_stage2"
evidence_tag="${training_pipeline_tag}_evidence"
mkdir -p "${pipeline_dir}" "${TMP_ROOT}"
cd "${CODE_ROOT}"

TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" -m unittest discover -s tests \
  > "${pipeline_dir}/unit_tests.log" 2>&1
TRACE_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" tools/data_contract_audit.py \
  > "${pipeline_dir}/data_contract_audit.json"
TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${DATA_ROOT}" \
  "${PYTHON}" tools/role_pipeline_contract_audit.py \
  > "${pipeline_dir}/preflight_contract_audit.json"

sha256sum \
  run.py \
  src/models/read.py \
  src/models/trace_policy.py \
  src/modules/trace_policy.py \
  src/datasets/gsm8k_aug_nl.py \
  src/configs/models/trace_policy_qwen3_instruct.yaml \
  src/configs/datasets/gsm8k_aug_nl.yaml \
  scripts/run_full_pipeline.sh \
  scripts/run_stage1_formation.sh \
  scripts/run_stage2_refinement.sh \
  scripts/run_evidence.sh \
  tools/data_contract_audit.py \
  tools/role_pipeline_contract_audit.py \
  tools/verify_evidence_complete.py \
  tools/trace_policy_mechanism_smoke.py \
  tools/trace_policy_ddp_memory_smoke.py \
  tools/trace_policy_stage2_stability_smoke.py \
  tools/trace_prepare_pca_fit_set.py \
  tools/trace_policy_task_summary.py \
  tools/trace_policy_geometry_summary.py \
  tools/trace_policy_stage_comparison.py \
  tools/trace_policy_causal_summary.py \
  tests/test_chunked_masked_causal_ce.py \
  tests/test_ddp_memory_gate.py \
  > "${pipeline_dir}/source_sha256.txt"

cat > "${pipeline_dir}/manifest.txt" <<EOF
model=TRACE-Role-Latent-RL
full_name=Trajectory-Structured_Latent_Reasoning_with_Outcome-Based_Refinement
project_root=${CODE_ROOT}
data_root=${DATA_ROOT}
artifact_root=${ARTIFACT_ROOT}
pipeline=explicit_stage0_to_new_role_stage1_to_role_RL_stage2_to_complete_paired_evidence
training_pipeline_tag=${training_pipeline_tag}
physical_training_gpus=${physical_gpus}
evidence_gpu=${evidence_gpu}
train_seed=${TRAIN_SEED}
explicit_cots_per_question=1
generated_cots=false
stage0_protocol=explicit_registered_full_data_fresh_CoT-SFT_best
stage0_checkpoint=${STAGE0_CKPT_OVERRIDE}
latent_roles=PLAN,SOLVE1,SOLVE2,SOLVE3,SOLVE4,SOLVE5,CHECK,COMMIT
stage1_epochs=10_full_validation
stage2_epochs=10_full_validation
test_times=1
started_at=$(date --iso-8601=seconds)
EOF

run_stage1_stress_preflight() {
  local stage0_checkpoint=$1
  env \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TRACE_PROJECT_ROOT="${CODE_ROOT}" \
    TRACE_DATA_ROOT="${DATA_ROOT}" \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${evidence_gpu}" \
    "${PYTHON}" tools/trace_policy_mechanism_smoke.py \
      --stage0-checkpoint "${stage0_checkpoint}" \
      --stress-cases 3 \
      --output "${pipeline_dir}/stage1_stress_preflight.json" \
      2>&1 | tee "${pipeline_dir}/stage1_stress_preflight.log"
}

run_stage1_ddp_stress_preflight() {
  local stage0_checkpoint=$1
  env \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TRACE_PROJECT_ROOT="${CODE_ROOT}" \
    TRACE_DATA_ROOT="${DATA_ROOT}" \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${physical_gpus}" \
    "${PYTHON}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node=4 \
      tools/trace_policy_ddp_memory_smoke.py \
      --stage0-checkpoint "${stage0_checkpoint}" \
      --optimizer-steps 3 \
      --output "${pipeline_dir}/stage1_ddp_stress_preflight.json" \
      2>&1 | tee "${pipeline_dir}/stage1_ddp_stress_preflight.log"
}

stage0_checkpoint=${STAGE0_CKPT_OVERRIDE}
run_stage1_stress_preflight "${stage0_checkpoint}"
run_stage1_ddp_stress_preflight "${stage0_checkpoint}"
RUN_TAG="${stage1_tag}" RUN_ROOT="${RUN_ROOT}" LOG_ROOT="${LOG_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  TRAIN_SEED="${TRAIN_SEED}" \
  bash "${SCRIPT_DIR}/run_stage1_formation.sh" \
    "${physical_gpus}" \
    "${stage0_checkpoint}"
stage1_checkpoint=$(<"${RUN_ROOT}/${stage1_tag}/best_checkpoint.txt")

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${DATA_ROOT}" \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${evidence_gpu}" \
  "${PYTHON}" tools/trace_policy_stage2_stability_smoke.py \
    --stage1-checkpoint "${stage1_checkpoint}" \
    --output "${pipeline_dir}/stage2_stability_smoke.json" \
    > "${pipeline_dir}/stage2_stability_smoke.log"

RUN_TAG="${stage2_tag}" RUN_ROOT="${RUN_ROOT}" LOG_ROOT="${LOG_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  TRAIN_SEED="${TRAIN_SEED}" \
  bash "${SCRIPT_DIR}/run_stage2_refinement.sh" \
    "${physical_gpus}" \
    "${stage1_checkpoint}"
stage2_checkpoint=$(<"${RUN_ROOT}/${stage2_tag}/best_checkpoint.txt")

RUN_TAG="${evidence_tag}" OUT="${EVIDENCE_ROOT}/${evidence_tag}" \
  DATA_ROOT="${DATA_ROOT}" ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  bash "${SCRIPT_DIR}/run_evidence.sh" \
    "${evidence_gpu}" \
    "${stage1_checkpoint}" \
    "${stage2_checkpoint}"

evidence_complete="${EVIDENCE_ROOT}/${evidence_tag}/COMPLETE.json"
if [[ ! -s "${evidence_complete}" ]]; then
  echo "Complete evidence gate did not produce ${evidence_complete}" >&2
  exit 1
fi
"${PYTHON}" tools/verify_evidence_complete.py \
  --evidence-root "${EVIDENCE_ROOT}/${evidence_tag}" \
  --stage1-checkpoint "${stage1_checkpoint}" \
  --final-checkpoint "${stage2_checkpoint}" \
  > "${pipeline_dir}/evidence_completeness_recheck.json"
cp "${evidence_complete}" "${pipeline_dir}/COMPLETE.json"

cat >> "${pipeline_dir}/manifest.txt" <<EOF
stage0_checkpoint=${stage0_checkpoint}
stage1_checkpoint=${stage1_checkpoint}
stage2_checkpoint=${stage2_checkpoint}
evidence_dir=${EVIDENCE_ROOT}/${evidence_tag}
complete_record=${pipeline_dir}/COMPLETE.json
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${stage0_checkpoint}" > "${pipeline_dir}/stage0_best.txt"
printf '%s\n' "${stage1_checkpoint}" > "${pipeline_dir}/stage1_best.txt"
printf '%s\n' "${stage2_checkpoint}" > "${pipeline_dir}/stage2_best.txt"
