#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/TRACE
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/training}
PIPELINE_ROOT=${PIPELINE_ROOT:-${ROOT}/run_outputs/pipelines}
TRAIN_SEED=${TRAIN_SEED:-0}
RESUME_PIPELINE_TAG=${RESUME_PIPELINE_TAG:-}
RESUME_STAGE1_CKPT=${RESUME_STAGE1_CKPT:-}
STAGE0_CKPT_OVERRIDE=${STAGE0_CKPT_OVERRIDE:-}

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> [evidence-physical-gpu]" >&2
  exit 2
fi
physical_gpus=$1
evidence_gpu=${2:-${physical_gpus%%,*}}
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "Formal TRACE requires four GPUs" >&2
  exit 2
fi
stage1_gpus="${physical_gpus}"

pipeline_tag=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_seed${TRAIN_SEED}}
training_pipeline_tag=${RESUME_PIPELINE_TAG:-${pipeline_tag}}
pipeline_dir="${PIPELINE_ROOT}/${pipeline_tag}"
stage0_tag="${training_pipeline_tag}_stage0"
stage1_tag="${training_pipeline_tag}_stage1"
stage2_tag="${training_pipeline_tag}_stage2"
evidence_tag="${training_pipeline_tag}_evidence"
mkdir -p "${pipeline_dir}" "${ROOT}/run_roots/tmp"
cd "${ROOT}"

"${PYTHON}" -m unittest discover -s tests \
  > "${pipeline_dir}/unit_tests.log" 2>&1
"${PYTHON}" tools/data_contract_audit.py \
  > "${pipeline_dir}/data_contract_audit.json"
"${PYTHON}" tools/trace_stage1_target_audit.py \
  --output "${pipeline_dir}/stage1_target_audit.json" \
  > "${pipeline_dir}/stage1_target_audit.stdout.json"
"${PYTHON}" tools/contract_audit.py \
  --output-dir "${pipeline_dir}/contract" \
  > "${pipeline_dir}/preflight_contract_audit.json"

if [[ -n "${RESUME_PIPELINE_TAG}" ]]; then
  if [[ -z "${RESUME_STAGE1_CKPT}" ]]; then
    echo "RESUME_STAGE1_CKPT is required with RESUME_PIPELINE_TAG" >&2
    exit 2
  fi
  if [[ ! -f "${RESUME_STAGE1_CKPT}" ]]; then
    echo "Missing Stage-1 resume checkpoint: ${RESUME_STAGE1_CKPT}" >&2
    exit 2
  fi
  if [[ -n "${STAGE0_CKPT_OVERRIDE}" ]]; then
    if [[ ! -f "${STAGE0_CKPT_OVERRIDE}" ]]; then
      echo "Missing completed Stage-0 checkpoint: ${STAGE0_CKPT_OVERRIDE}" >&2
      exit 2
    fi
  elif [[ ! -f "${RUN_ROOT}/${stage0_tag}/best_checkpoint.txt" ]]; then
    echo "Missing completed Stage-0 record for ${RESUME_PIPELINE_TAG}" >&2
    exit 2
  fi
  "${PYTHON}" - \
    "${ROOT}/run_outputs/supervisor/${RESUME_PIPELINE_TAG}/source_hashes.json" \
    "${pipeline_dir}/resume_source_integrity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path("/disk1/dingxukai/TRACE")
source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
expected = json.loads(source_path.read_text(encoding="utf-8"))
protected = (
    "src/models/trace_policy.py",
    "src/modules/trace_policy.py",
    "src/datasets/gsm8k_aug_nl.py",
    "src/configs/models/trace_policy_qwen3_instruct.yaml",
    "src/configs/datasets/gsm8k_aug_nl.yaml",
)
rows = {}
for relative in protected:
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows[relative] = {
        "expected": expected.get(relative),
        "actual": digest,
        "matches": expected.get(relative) == digest,
    }
report = {
    "status": "PASS" if all(row["matches"] for row in rows.values()) else "FAIL",
    "protected_sources": rows,
}
output_path.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if report["status"] != "PASS":
    raise SystemExit("Training-critical source changed since the interrupted run")
PY
fi

cat > "${pipeline_dir}/manifest.txt" <<EOF
model=TRACE-Policy-v3
full_name=Causal_Exchangeable_Trajectory_Formation_with_Outcome-Based_Refinement
project_root=${ROOT}
pipeline=$(
  if [[ -n "${RESUME_PIPELINE_TAG}" ]]; then
    printf '%s' "resume_stage1_plus_stage2_plus_paired_evidence"
  elif [[ -n "${STAGE0_CKPT_OVERRIDE}" ]]; then
    printf '%s' "fresh_stage1_from_completed_stage0_plus_stage2_plus_paired_evidence"
  else
    printf '%s' "fresh_stage0_plus_stage1_plus_stage2_plus_paired_evidence"
  fi
)
training_pipeline_tag=${training_pipeline_tag}
resume_pipeline_tag=${RESUME_PIPELINE_TAG:-none}
resume_stage1_checkpoint=${RESUME_STAGE1_CKPT:-none}
initial_physical_gpus=${physical_gpus}
stage1_physical_gpus=${stage1_gpus}
stage1_execution=four_rank_gpu_only_ddp_effective_batch_4
evidence_gpu=${evidence_gpu}
train_seed=${TRAIN_SEED}
explicit_cots_per_question=1
generated_cots=false
stage0_protocol=$(
  if [[ -n "${STAGE0_CKPT_OVERRIDE}" ]]; then
    printf '%s' "reuse_frozen_registered-data_fresh_CoT-SFT_control"
  else
    printf '%s' "3_epochs_full_validation"
  fi
)
stage1_epochs=10_full_validation
stage2_epochs=10_full_validation
test_times=1
started_at=$(date --iso-8601=seconds)
EOF

run_stage1_stress_preflight() {
  local stage0_checkpoint=$1
  env \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TMPDIR="${ROOT}/run_roots/tmp" \
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
    TMPDIR="${ROOT}/run_roots/tmp" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${stage1_gpus}" \
    "${PYTHON}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node=4 \
      tools/isolated_gpu_ddp_entry.py \
      tools/trace_policy_ddp_memory_smoke.py \
      --stage0-checkpoint "${stage0_checkpoint}" \
      --optimizer-steps 3 \
      --expected-world-size 4 \
      --accumulation-steps 1 \
      --output "${pipeline_dir}/stage1_ddp_stress_preflight.json" \
      2>&1 | tee "${pipeline_dir}/stage1_ddp_stress_preflight.log"
}

wait_for_training_gpus() {
  local threshold_mib=${TRACE_GPU_IDLE_THRESHOLD_MIB:-3072}
  local poll_seconds=${TRACE_GPU_POLL_SECONDS:-30}
  while true; do
    local selected
    selected=$("${PYTHON}" - "${threshold_mib}" <<'PY'
import subprocess
import sys

threshold = int(sys.argv[1])
output = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
used = {}
for line in output.splitlines():
    index, memory = (int(value.strip()) for value in line.split(","))
    used[index] = memory
idle = sorted(index for index, memory in used.items() if memory <= threshold)
if len(idle) < 4:
    raise SystemExit(1)
print(",".join(str(index) for index in idle[:4]))
PY
    ) && {
      printf '%s\n' "${selected}"
      return 0
    }
    sleep "${poll_seconds}"
  done
}

run_stage2_stability_preflight() {
  local stage1_checkpoint=$1
  env \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TMPDIR="${ROOT}/run_roots/tmp" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${evidence_gpu}" \
    "${PYTHON}" tools/trace_policy_stage2_stability_smoke.py \
      --stage1-checkpoint "${stage1_checkpoint}" \
      --output "${pipeline_dir}/stage2_stability_preflight.json" \
      2>&1 | tee "${pipeline_dir}/stage2_stability_preflight.log"
}

if [[ -n "${RESUME_PIPELINE_TAG}" ]]; then
  if [[ -n "${STAGE0_CKPT_OVERRIDE}" ]]; then
    stage0_checkpoint=${STAGE0_CKPT_OVERRIDE}
  else
    stage0_checkpoint=$(<"${RUN_ROOT}/${stage0_tag}/best_checkpoint.txt")
  fi
  RUN_TAG="${stage1_tag}" RUN_ROOT="${RUN_ROOT}" TRAIN_SEED="${TRAIN_SEED}" \
    RESUME_CKPT_PATH="${RESUME_STAGE1_CKPT}" \
    bash scripts/run_stage1_formation.sh \
      "${stage1_gpus}" \
      "${stage0_checkpoint}"
elif [[ -n "${STAGE0_CKPT_OVERRIDE}" ]]; then
  if [[ ! -f "${STAGE0_CKPT_OVERRIDE}" ]]; then
    echo "Missing completed fresh Stage-0 checkpoint: ${STAGE0_CKPT_OVERRIDE}" >&2
    exit 2
  fi
  stage0_checkpoint=${STAGE0_CKPT_OVERRIDE}
  run_stage1_stress_preflight "${stage0_checkpoint}"
  run_stage1_ddp_stress_preflight "${stage0_checkpoint}"
  RUN_TAG="${stage1_tag}" RUN_ROOT="${RUN_ROOT}" TRAIN_SEED="${TRAIN_SEED}" \
    bash scripts/run_stage1_formation.sh \
      "${stage1_gpus}" \
      "${stage0_checkpoint}"
else
  RUN_TAG="${stage0_tag}" RUN_ROOT="${RUN_ROOT}" TRAIN_SEED="${TRAIN_SEED}" \
    bash scripts/run_stage0_cot.sh "${physical_gpus}"
  stage0_checkpoint=$(<"${RUN_ROOT}/${stage0_tag}/best_checkpoint.txt")

  run_stage1_stress_preflight "${stage0_checkpoint}"
  run_stage1_ddp_stress_preflight "${stage0_checkpoint}"
  RUN_TAG="${stage1_tag}" RUN_ROOT="${RUN_ROOT}" TRAIN_SEED="${TRAIN_SEED}" \
    bash scripts/run_stage1_formation.sh \
      "${stage1_gpus}" \
      "${stage0_checkpoint}"
fi
stage1_checkpoint=$(<"${RUN_ROOT}/${stage1_tag}/best_checkpoint.txt")
physical_gpus=$(wait_for_training_gpus)
evidence_gpu=${physical_gpus%%,*}
cat >> "${pipeline_dir}/manifest.txt" <<EOF
stage2_physical_gpus=${physical_gpus}
stage2_resource_acquired_at=$(date --iso-8601=seconds)
EOF
run_stage2_stability_preflight "${stage1_checkpoint}"

RUN_TAG="${stage2_tag}" RUN_ROOT="${RUN_ROOT}" TRAIN_SEED="${TRAIN_SEED}" \
  bash scripts/run_stage2_refinement.sh \
    "${physical_gpus}" \
    "${stage1_checkpoint}"
stage2_checkpoint=$(<"${RUN_ROOT}/${stage2_tag}/best_checkpoint.txt")

RUN_TAG="${evidence_tag}" \
  bash scripts/run_evidence.sh \
    "${evidence_gpu}" \
    "${stage1_checkpoint}" \
    "${stage2_checkpoint}"

cat >> "${pipeline_dir}/manifest.txt" <<EOF
stage0_checkpoint=${stage0_checkpoint}
stage1_checkpoint=${stage1_checkpoint}
stage2_checkpoint=${stage2_checkpoint}
evidence_dir=${ROOT}/run_outputs/evidence/${evidence_tag}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${stage0_checkpoint}" > "${pipeline_dir}/stage0_best.txt"
printf '%s\n' "${stage1_checkpoint}" > "${pipeline_dir}/stage1_best.txt"
printf '%s\n' "${stage2_checkpoint}" > "${pipeline_dir}/stage2_best.txt"
