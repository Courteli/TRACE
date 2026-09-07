#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/TRACE
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
TRAIN_SEED=${TRAIN_SEED:-0}
PIPELINE_ROOT=${PIPELINE_ROOT:-${ROOT}/run_outputs/pipelines}
STAGE2_RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/training}
EVIDENCE_ROOT=${EVIDENCE_ROOT:-${ROOT}/run_outputs/evidence}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/tmp}

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage1-best-ckpt> [evidence-gpu]" >&2
  exit 2
fi

physical_gpus=$1
stage1_checkpoint=$2
evidence_gpu=${3:-${physical_gpus%%,*}}
run_tag=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_stage2_jointtrust_seed${TRAIN_SEED}}
stage2_tag=${run_tag}_stage2
evidence_tag=${run_tag}_evidence
smoke_gpu=${physical_gpus%%,*}
pipeline_dir="${PIPELINE_ROOT}/${run_tag}"
mkdir -p "${pipeline_dir}" "${TMP_ROOT}"
cd "${ROOT}"

"${PYTHON}" -m unittest discover -s tests \
  > "${pipeline_dir}/unit_tests.log" 2>&1
"${PYTHON}" tools/data_contract_audit.py \
  > "${pipeline_dir}/data_contract_audit.json"
"${PYTHON}" tools/contract_audit.py \
  --output-dir "${pipeline_dir}/contract" \
  > "${pipeline_dir}/contract_audit.json"
TMPDIR="${TMP_ROOT}" CUDA_VISIBLE_DEVICES="${smoke_gpu}" \
  "${PYTHON}" tools/trace_policy_stage2_stability_smoke.py \
    --stage1-checkpoint "${stage1_checkpoint}" \
    --output "${pipeline_dir}/stage2_stability_smoke.json" \
    > "${pipeline_dir}/stage2_stability_smoke.log"

cat > "${pipeline_dir}/manifest.txt" <<EOF
model=TRACE-Policy-v3
phase=causal_outcome_stage2_plus_complete_evidence
stage1_checkpoint=${stage1_checkpoint}
physical_training_gpus=${physical_gpus}
evidence_gpu=${evidence_gpu}
stage2_stability_smoke_gpu=${smoke_gpu}
stage2_stability_smoke=passed_before_full_training
train_seed=${TRAIN_SEED}
method_objective=full_answer_and_latent_policy_with_target_KL
training_budget=matched_old_answer_only_stage2
group_size=4_epoch0_then_8
trace_steps=8
counterfactual_directions=2
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
  tools/trace_policy_stage2_stability_smoke.py \
  tools/contract_audit.py \
  > "${pipeline_dir}/source_sha256.txt"

RUN_TAG="${stage2_tag}" TRAIN_SEED="${TRAIN_SEED}" \
  bash scripts/run_stage2_refinement.sh \
    "${physical_gpus}" "${stage1_checkpoint}"
stage2_checkpoint=$(
  <"${STAGE2_RUN_ROOT}/${stage2_tag}/best_checkpoint.txt"
)

RUN_TAG="${evidence_tag}" OUT="${EVIDENCE_ROOT}/${evidence_tag}" \
  bash scripts/run_evidence.sh \
    "${evidence_gpu}" "${stage1_checkpoint}" "${stage2_checkpoint}"

cat >> "${pipeline_dir}/manifest.txt" <<EOF
stage2_checkpoint=${stage2_checkpoint}
evidence_dir=${EVIDENCE_ROOT}/${evidence_tag}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${stage2_checkpoint}" \
  > "${pipeline_dir}/stage2_best.txt"
