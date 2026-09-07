#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_ROOT=${DATA_ROOT:-/disk1/dingxukai/TRACE}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/disk1/dingxukai/TRACE/role_semantic_runs}
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
TRAIN_SEED=${TRAIN_SEED:-0}
PIPELINE_ROOT=${PIPELINE_ROOT:-${ARTIFACT_ROOT}/pipelines}
STAGE2_RUN_ROOT=${RUN_ROOT:-${ARTIFACT_ROOT}/training}
EVIDENCE_ROOT=${EVIDENCE_ROOT:-${ARTIFACT_ROOT}/evidence}
TMP_ROOT=${TMP_ROOT:-${ARTIFACT_ROOT}/tmp}
LOG_ROOT=${LOG_ROOT:-${ARTIFACT_ROOT}/logs}

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage1-best-checkpoint> [evidence-gpu]" >&2
  exit 2
fi

physical_gpus=$1
stage1_checkpoint=$2
evidence_gpu=${3:-${physical_gpus%%,*}}
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]] \
  || [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "Formal Stage 2 requires four unique physical GPUs" >&2
  exit 2
fi
if [[ ! -f "${stage1_checkpoint}" ]]; then
  echo "Missing new Stage-1 checkpoint: ${stage1_checkpoint}" >&2
  exit 2
fi
run_tag=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_stage2_jointtrust_seed${TRAIN_SEED}}
stage2_tag=${run_tag}_stage2
evidence_tag=${run_tag}_evidence
smoke_gpu=${physical_gpus%%,*}
pipeline_dir="${PIPELINE_ROOT}/${run_tag}"
mkdir -p "${pipeline_dir}" "${TMP_ROOT}"
cd "${CODE_ROOT}"

"${PYTHON}" -m unittest discover -s tests \
  > "${pipeline_dir}/unit_tests.log" 2>&1
TRACE_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" tools/data_contract_audit.py \
  > "${pipeline_dir}/data_contract_audit.json"
TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${DATA_ROOT}" \
  "${PYTHON}" tools/role_pipeline_contract_audit.py \
  > "${pipeline_dir}/preflight_contract_audit.json"
TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${DATA_ROOT}" \
  TMPDIR="${TMP_ROOT}" CUDA_VISIBLE_DEVICES="${smoke_gpu}" \
  "${PYTHON}" tools/trace_policy_stage2_stability_smoke.py \
    --stage1-checkpoint "${stage1_checkpoint}" \
    --output "${pipeline_dir}/stage2_stability_smoke.json" \
    > "${pipeline_dir}/stage2_stability_smoke.log"

cat > "${pipeline_dir}/manifest.txt" <<EOF
model=TRACE-Role-Latent-RL
phase=role_semantic_stage2_plus_complete_paired_evidence
stage1_checkpoint=${stage1_checkpoint}
physical_training_gpus=${physical_gpus}
evidence_gpu=${evidence_gpu}
stage2_stability_smoke_gpu=${smoke_gpu}
stage2_stability_smoke=passed_before_full_training
train_seed=${TRAIN_SEED}
latent_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_CHECK_COMMIT
stage2_credit=position_local_role_rewards_plus_terminal_answer_reward
training_budget=2048_unique_questions_x_10_epochs
group_size=8
trace_steps=8
semantic_reward_positions=PLAN_through_CHECK
commit_sampling=false
policy_update_epochs=1
epochs=10
source_training_split_questions=6726
unique_training_questions_per_epoch=2048
full_validation_every_epoch=true
test_times=1
started_at=$(date --iso-8601=seconds)
EOF

sha256sum \
  src/models/trace_policy.py \
  src/modules/trace_policy.py \
  src/configs/models/trace_policy_qwen3_instruct.yaml \
  src/configs/datasets/gsm8k_aug_nl.yaml \
  scripts/run_stage2_refinement.sh \
  scripts/run_stage2_and_evidence.sh \
  scripts/run_evidence.sh \
  tools/role_pipeline_contract_audit.py \
  tools/verify_evidence_complete.py \
  tools/trace_policy_stage2_stability_smoke.py \
  > "${pipeline_dir}/source_sha256.txt"

RUN_TAG="${stage2_tag}" TRAIN_SEED="${TRAIN_SEED}" \
  DATA_ROOT="${DATA_ROOT}" ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  RUN_ROOT="${STAGE2_RUN_ROOT}" LOG_ROOT="${LOG_ROOT}" \
  bash "${SCRIPT_DIR}/run_stage2_refinement.sh" \
    "${physical_gpus}" "${stage1_checkpoint}"
stage2_checkpoint=$(
  <"${STAGE2_RUN_ROOT}/${stage2_tag}/best_checkpoint.txt"
)

RUN_TAG="${evidence_tag}" OUT="${EVIDENCE_ROOT}/${evidence_tag}" \
  DATA_ROOT="${DATA_ROOT}" ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  bash "${SCRIPT_DIR}/run_evidence.sh" \
    "${evidence_gpu}" "${stage1_checkpoint}" "${stage2_checkpoint}"

if [[ ! -s "${EVIDENCE_ROOT}/${evidence_tag}/COMPLETE.json" ]]; then
  echo "Stage-2 evidence did not pass its COMPLETE gate" >&2
  exit 1
fi
cp "${EVIDENCE_ROOT}/${evidence_tag}/COMPLETE.json" \
  "${pipeline_dir}/COMPLETE.json"
cat >> "${pipeline_dir}/manifest.txt" <<EOF
stage2_checkpoint=${stage2_checkpoint}
evidence_dir=${EVIDENCE_ROOT}/${evidence_tag}
complete_record=${EVIDENCE_ROOT}/${evidence_tag}/COMPLETE.json
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${stage2_checkpoint}" \
  > "${pipeline_dir}/stage2_best.txt"
