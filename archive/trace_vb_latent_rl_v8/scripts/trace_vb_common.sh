#!/usr/bin/env bash

# Immutable identifiers shared by the train/validation-only TRACE-VB-v8
# entry points.  The CoT encoder used for role supervision is deliberately
# separate from the historical teacher that produced the offline sufficiency
# cache; conflating the two would make the cache provenance false.
TRACE_VB_VERSION=TRACE-VB-v8
TRACE_VB_CHECKPOINT_SCHEMA=trace_vb_v8
TRACE_VB_VALIDATION_SCHEMA=trace_vb_v8_validation_behavior_v1
TRACE_VB_MODEL_CONFIG=trace_vb_policy_qwen3_instruct
TRACE_VB_ARTIFACT_ROOT=/disk1/dingxukai/TRACE/trace_vb_v8_runs
TRACE_VB_DATA_ROOT=/disk1/dingxukai/TRACE
TRACE_VB_CODE_ROOT=/home/dingxukai/TRACE/trace_vb_latent_rl_v8
TRACE_VB_BASE_MODEL=/home/dingxukai/RoT/ckpt/base/Qwen3-4B-Instruct/qwen3-instruct

TRACE_VB_REGISTERED_CAPABILITY=/disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260711-032736_236970_20260711_trace_bridge_final_guarded_mb4_gpu0123_stage2_trace_guarded/checkpoints/epoch1__step1024__monitor0.723.ckpt
TRACE_VB_REGISTERED_CAPABILITY_SHA256=d27ef63b1d462aa94fbbecc636cec31985a3687e5d2d036de9cfd87b01ec1525
TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_TENSORS=517
TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_SHA256=b1e9a973bbf4f2eeaa18b7d49df16c38db50cea87e998008ba18a46cdff9e049

# Immutable metric anchors.  The student path is the comparable Stage-1/2
# floor; the capability path is separately recorded provenance and is not
# compared numerically across decoding paths.
TRACE_VB_METRIC_SAFE_BASELINE=/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/20260818-213000_trace_vb_v7_full_seed0/student_initial_gate.json
TRACE_VB_METRIC_SAFE_BASELINE_SHA256=9b888b36956f70affae153526b716ad26343164b4aa66d6d3d54f70561144aa2
TRACE_VB_METRIC_SAFE_BASELINE_CORRECT=527
TRACE_VB_METRIC_SAFE_BASELINE_QUESTIONS=747
TRACE_VB_REGISTERED_CAPABILITY_VALIDATION=/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/20260818-213000_trace_vb_v7_full_seed0/capability_parity_gate.json
TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_SHA256=59123d2bfc020335f54d70862a904e1b70e5ea1ce764483692809ae4e6772116
TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_CORRECT=540
TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_QUESTIONS=747

# Strong 651/747 CoT encoder used by v8 for training-only semantic targets.
TRACE_VB_COT_ENCODER=/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/checkpoints/epoch0__step6726__monitor0.871.ckpt
TRACE_VB_COT_ENCODER_SHA256=df90292c2da854852651a9e75b1e8221484c6dc58b8c81a56400897a8dec7f54
TRACE_VB_COT_ENCODER_VALIDATION_CORRECT=651

# This older checkpoint is provenance only.  It produced the immutable
# sufficiency cache and must never be passed as v8's CoT encoder.
TRACE_VB_SUFFICIENCY_TEACHER=/disk1/dingxukai/TRACE/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260720-072412_674141_20260720-041426_trace_full_seed0_stage0/checkpoints/epoch1__step3364__monitor0.848.ckpt
TRACE_VB_SUFFICIENCY_TEACHER_SHA256=1e58984dcae9dfd6885a2d5a58c8948d2832a7e19ec467273f8f74546bc7aeaa
TRACE_VB_SUFFICIENCY_CACHE=/disk1/dingxukai/TRACE/trace_vb_runs/cache/gsm8k_prefix_sufficiency_v1.pt
TRACE_VB_SUFFICIENCY_TOKENIZER_SHA256=b8b23250b705d1778c35090c2ba7056811524756334f207b537263647e60f29a
TRACE_VB_SUFFICIENCY_PROMPT_SHA256=d899631d75286aec8c79ecc8b6bf82bca9a70e11c721d36ea26da399ce977e4c
TRACE_VB_SUFFICIENCY_COMPOSITE_SHA256=f1eb1296b6342e072431e53778e77687aa53016654938695132f328734bdfecb

TRACE_VB_FORMAL_GPUS=${TRACE_VB_FORMAL_GPUS:-2,3,4,5}
TRACE_VB_FIXED_GPUS=${TRACE_VB_FIXED_GPUS:-${TRACE_VB_FORMAL_GPUS}}
TRACE_VB_MIN_FREE_GPU_MIB=21500
TRACE_VB_MAX_IDLE_UTILIZATION=10
TRACE_VB_VALIDATION_QUESTIONS=747
TRACE_VB_PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python

trace_vb_die() {
  echo "TRACE-VB-v8: $*" >&2
  exit 2
}

trace_vb_require_safe_tag() {
  local tag=$1
  [[ "${tag}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    trace_vb_die "unsafe run tag: ${tag}"
}

trace_vb_require_sha256() {
  local label=$1
  local path=$2
  local expected=$3
  [[ -f "${path}" ]] || trace_vb_die "missing ${label}: ${path}"
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || \
    trace_vb_die "${label} SHA256 mismatch: ${actual}"
}

trace_vb_require_capability() {
  trace_vb_require_sha256 \
    "registered capability checkpoint" \
    "${TRACE_VB_REGISTERED_CAPABILITY}" \
    "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}"
}

trace_vb_require_metric_artifacts() {
  trace_vb_require_sha256 \
    "metric-safe student baseline" \
    "${TRACE_VB_METRIC_SAFE_BASELINE}" \
    "${TRACE_VB_METRIC_SAFE_BASELINE_SHA256}"
  trace_vb_require_sha256 \
    "registered capability validation" \
    "${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION}" \
    "${TRACE_VB_REGISTERED_CAPABILITY_VALIDATION_SHA256}"
}

trace_vb_require_cot_encoder() {
  trace_vb_require_sha256 \
    "strong CoT encoder checkpoint" \
    "${TRACE_VB_COT_ENCODER}" \
    "${TRACE_VB_COT_ENCODER_SHA256}"
}

trace_vb_require_sufficiency_cache() {
  [[ -s "${TRACE_VB_SUFFICIENCY_CACHE}" ]] || \
    trace_vb_die "missing immutable sufficiency cache: ${TRACE_VB_SUFFICIENCY_CACHE}"
}

trace_vb_require_formal_gpu_set() {
  local physical_gpus=$1
  local label=${2:-requested}
  [[ "${TRACE_VB_FORMAL_GPUS}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*)){3}$ ]] || \
    trace_vb_die "TRACE_VB_FORMAL_GPUS must contain exactly four numeric IDs"
  [[ "${TRACE_VB_FIXED_GPUS}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*)){3}$ ]] || \
    trace_vb_die "TRACE_VB_FIXED_GPUS must contain exactly four numeric IDs"
  [[ "${physical_gpus}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*)){3}$ ]] || \
    trace_vb_die "${label} GPU set must contain exactly four numeric IDs"
  local formal_array fixed_array gpu_array
  IFS=',' read -r -a formal_array <<< "${TRACE_VB_FORMAL_GPUS}"
  IFS=',' read -r -a fixed_array <<< "${TRACE_VB_FIXED_GPUS}"
  IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
  local -A seen=()
  local gpu
  for gpu in "${formal_array[@]}"; do
    [[ -z "${seen[${gpu}]+x}" ]] || \
      trace_vb_die "TRACE_VB_FORMAL_GPUS contains duplicate GPU ${gpu}"
    seen[${gpu}]=1
  done
  seen=()
  for gpu in "${fixed_array[@]}"; do
    [[ -z "${seen[${gpu}]+x}" ]] || \
      trace_vb_die "TRACE_VB_FIXED_GPUS contains duplicate GPU ${gpu}"
    seen[${gpu}]=1
  done
  [[ "${TRACE_VB_FIXED_GPUS}" == "${TRACE_VB_FORMAL_GPUS}" ]] || \
    trace_vb_die \
      "fixed GPU set ${TRACE_VB_FIXED_GPUS} differs from formal set ${TRACE_VB_FORMAL_GPUS}"
  seen=()
  for gpu in "${gpu_array[@]}"; do
    [[ -z "${seen[${gpu}]+x}" ]] || \
      trace_vb_die "${label} GPU set contains duplicate GPU ${gpu}"
    seen[${gpu}]=1
  done
  [[ "${physical_gpus}" == "${TRACE_VB_FORMAL_GPUS}" ]] || \
    trace_vb_die \
      "${label} GPU set ${physical_gpus} differs from formal set ${TRACE_VB_FORMAL_GPUS}"
}

trace_vb_require_four_gpus() {
  local physical_gpus=$1
  local min_free_mib=${2:-${TRACE_VB_MIN_FREE_GPU_MIB}}
  trace_vb_require_formal_gpu_set "${physical_gpus}" "formal run"
  local gpu_array
  IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
  local gpu free total utilization
  for gpu in "${gpu_array[@]}"; do
    read -r total free utilization < <(
      nvidia-smi -i "${gpu}" \
        --query-gpu=memory.total,memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); print $1, $2, $3}'
    )
    [[ "${total}" =~ ^[0-9]+$ && "${free}" =~ ^[0-9]+$ \
      && "${utilization}" =~ ^[0-9]+$ ]] || \
      trace_vb_die "could not read memory for GPU ${gpu}"
    (( free >= min_free_mib )) || \
      trace_vb_die "GPU ${gpu} has ${free} MiB free; ${min_free_mib} MiB required"
    (( utilization <= TRACE_VB_MAX_IDLE_UTILIZATION )) || \
      trace_vb_die \
        "GPU ${gpu} utilization is ${utilization}%; idle threshold is ${TRACE_VB_MAX_IDLE_UTILIZATION}%"
  done
}

trace_vb_v7_training_is_active() {
  local detector_status=0
  "${TRACE_VB_PYTHON}" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/detect_active_v7_training_v8.py" \
    --quiet || detector_status=$?
  if (( detector_status == 0 )); then
    return 0
  fi
  if (( detector_status == 3 )); then
    return 1
  fi
  trace_vb_die "could not determine whether v7 training is active"
}

trace_vb_require_no_v7_training() {
  if trace_vb_v7_training_is_active; then
    trace_vb_die "v7 training/recovery process is still alive"
  fi
}

trace_vb_audit_v7_trigger() {
  local stage1_dir=$1
  local checkpoint=$2
  local output=$3
  [[ -n "${stage1_dir}" ]] || trace_vb_die "V7_STAGE1_DIR is required"
  local audit_args=(
    --stage1-dir "${stage1_dir}"
    --output "${output}"
  )
  if [[ -n "${checkpoint}" ]]; then
    audit_args+=(--checkpoint "${checkpoint}")
  fi
  "${TRACE_VB_PYTHON}" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/audit_v7_trigger_for_v8.py" \
    "${audit_args[@]}"
}

trace_vb_find_single_logger_dir() {
  local log_parent=$1
  local tag=$2
  local matches=()
  mapfile -t matches < <(
    find "${log_parent}" -mindepth 1 -maxdepth 1 -type d \
      -name "*_${tag}" -print | sort
  )
  [[ "${#matches[@]}" -eq 1 ]] || \
    trace_vb_die "expected one logger directory for ${tag}, found ${#matches[@]}"
  printf '%s\n' "${matches[0]}"
}

trace_vb_assert_no_test_tokens() {
  local script=$1
  if grep -Eq 'run_evidence|trainer\.test|--do_test|--test_ckpt_path' "${script}"; then
    trace_vb_die "train-only launcher contains a forbidden test/evidence token: ${script}"
  fi
}
