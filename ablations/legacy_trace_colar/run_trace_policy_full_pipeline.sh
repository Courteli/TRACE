#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/trace_policy/training}
PIPELINE_ROOT=${PIPELINE_ROOT:-${ROOT}/run_outputs/trace_policy/pipelines}
TRAIN_SEED=${TRAIN_SEED:-0}

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> [evidence-physical-gpu]" >&2
  exit 2
fi
physical_gpus=$1
evidence_gpu=${2:-${physical_gpus%%,*}}
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "Formal TRACE requires exactly four training GPUs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "Training GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi
if [[ ! "${evidence_gpu}" =~ ^[0-9]+$ ]]; then
  echo "Evidence GPU must be one physical GPU ID" >&2
  exit 2
fi

pipeline_tag=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_policy_seed${TRAIN_SEED}}
pipeline_dir="${PIPELINE_ROOT}/${pipeline_tag}"
stage1_tag="${pipeline_tag}_stage1"
stage2_tag="${pipeline_tag}_stage2"
evidence_tag="${pipeline_tag}_evidence"
mkdir -p "${pipeline_dir}"

cat > "${pipeline_dir}/manifest.txt" <<EOF
model=TRACE-Policy
full_name=Set-Anchored_Stochastic_Latent_Trajectories_with_Counterfactual_Outcome_Refinement
pipeline=full_stage1_stage2_paired_evidence
physical_training_gpus=${physical_gpus}
evidence_gpu=${evidence_gpu}
train_seed=${TRAIN_SEED}
stage1_epochs=10_with_full_validation
stage1_teacher=frozen_stage0_cot_lora_plus_dependency_trained_monotone_compressor
stage2_epochs=10_full_6726_unique_questions_2_policy_updates_and_full_validation
stage2_group_size=8_iid
validation_monitor=question_id_deduplicated_full_split
test_times=1
started_at=$(date --iso-8601=seconds)
EOF

cd "${ROOT}"
"${PYTHON}" tools/trace_policy_contract_audit.py \
  > "${pipeline_dir}/preflight_contract_audit.json"

RUN_TAG="${stage1_tag}" \
RUN_ROOT="${RUN_ROOT}" \
TRAIN_SEED="${TRAIN_SEED}" \
  bash run_trace_policy_stage1_full.sh "${physical_gpus}"
stage1_best_file="${RUN_ROOT}/${stage1_tag}/best_checkpoint.txt"
if [[ ! -s "${stage1_best_file}" ]]; then
  echo "Stage 1 did not publish a best checkpoint path" >&2
  exit 1
fi
stage1_checkpoint=$(<"${stage1_best_file}")
if [[ ! -f "${stage1_checkpoint}" ]]; then
  echo "Stage 1 best checkpoint is missing: ${stage1_checkpoint}" >&2
  exit 1
fi

RUN_TAG="${stage2_tag}" \
RUN_ROOT="${RUN_ROOT}" \
TRAIN_SEED="${TRAIN_SEED}" \
  bash run_trace_policy_stage2_full.sh \
    "${physical_gpus}" \
    "${stage1_checkpoint}"
stage2_best_file="${RUN_ROOT}/${stage2_tag}/best_checkpoint.txt"
if [[ ! -s "${stage2_best_file}" ]]; then
  echo "Stage 2 did not publish a best checkpoint path" >&2
  exit 1
fi
stage2_checkpoint=$(<"${stage2_best_file}")
if [[ ! -f "${stage2_checkpoint}" ]]; then
  echo "Stage 2 best checkpoint is missing: ${stage2_checkpoint}" >&2
  exit 1
fi

RUN_TAG="${evidence_tag}" \
  bash run_trace_policy_full_evidence.sh \
    "${evidence_gpu}" \
    "${stage1_checkpoint}" \
    "${stage2_checkpoint}"

cat >> "${pipeline_dir}/manifest.txt" <<EOF
stage1_checkpoint=${stage1_checkpoint}
stage2_checkpoint=${stage2_checkpoint}
evidence_dir=${ROOT}/run_outputs/trace_policy/evidence/${evidence_tag}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${stage1_checkpoint}" > "${pipeline_dir}/stage1_best.txt"
printf '%s\n' "${stage2_checkpoint}" > "${pipeline_dir}/stage2_best.txt"
