#!/usr/bin/env bash

# Shared immutable identifiers for every formal TRACE-VB-v7 entry point.
TRACE_VB_MODEL_CONFIG=trace_vb_policy_qwen3_instruct
TRACE_VB_ARTIFACT_ROOT=/disk1/dingxukai/TRACE/trace_vb_v7_runs
TRACE_VB_DATA_ROOT=/disk1/dingxukai/TRACE
TRACE_VB_STAGE1_CODE_ROOT=/home/dingxukai/TRACE/trace_vb_latent_rl_v7
TRACE_VB_BASE_MODEL=/home/dingxukai/RoT/ckpt/base/Qwen3-4B-Instruct/qwen3-instruct
TRACE_VB_REGISTERED_CAPABILITY=/disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260711-032736_236970_20260711_trace_bridge_final_guarded_mb4_gpu0123_stage2_trace_guarded/checkpoints/epoch1__step1024__monitor0.723.ckpt
TRACE_VB_REGISTERED_CAPABILITY_SHA256=d27ef63b1d462aa94fbbecc636cec31985a3687e5d2d036de9cfd87b01ec1525
TRACE_VB_REGISTERED_STAGE0=/disk1/dingxukai/TRACE/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260720-072412_674141_20260720-041426_trace_full_seed0_stage0/checkpoints/epoch1__step3364__monitor0.848.ckpt
TRACE_VB_REGISTERED_STAGE0_SHA256=1e58984dcae9dfd6885a2d5a58c8948d2832a7e19ec467273f8f74546bc7aeaa
TRACE_VB_SUFFICIENCY_CACHE=/disk1/dingxukai/TRACE/trace_vb_runs/cache/gsm8k_prefix_sufficiency_v1.pt
TRACE_VB_MIN_FREE_GPU_MIB=21500
TRACE_VB_EPOCH1_MIN_ACCURACY=0.60
TRACE_VB_PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python

trace_vb_die() {
  echo "TRACE-VB: $*" >&2
  exit 2
}

trace_vb_require_stage0() {
  local checkpoint=${1:-${TRACE_VB_REGISTERED_STAGE0}}
  [[ -f "${checkpoint}" ]] || trace_vb_die "missing registered Stage-0 checkpoint: ${checkpoint}"
  local actual_sha
  actual_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
  [[ "${actual_sha}" == "${TRACE_VB_REGISTERED_STAGE0_SHA256}" ]] || \
    trace_vb_die "Stage-0 SHA256 mismatch: ${actual_sha}"
}

trace_vb_require_capability() {
  local checkpoint=${1:-${TRACE_VB_REGISTERED_CAPABILITY}}
  [[ -f "${checkpoint}" ]] || \
    trace_vb_die "missing registered capability checkpoint: ${checkpoint}"
  local actual_sha
  actual_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
  [[ "${actual_sha}" == "${TRACE_VB_REGISTERED_CAPABILITY_SHA256}" ]] || \
    trace_vb_die "capability checkpoint SHA256 mismatch: ${actual_sha}"
}

trace_vb_require_four_gpus() {
  local physical_gpus=$1
  local min_free_mib=${2:-${TRACE_VB_MIN_FREE_GPU_MIB}}
  local gpu_array
  IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
  [[ "${#gpu_array[@]}" -eq 4 ]] || \
    trace_vb_die "formal training requires exactly four physical GPUs"
  [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -eq 4 ]] || \
    trace_vb_die "GPU IDs must be unique: ${physical_gpus}"
  local gpu free total
  for gpu in "${gpu_array[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || trace_vb_die "invalid physical GPU ID: ${gpu}"
    read -r total free < <(
      nvidia-smi -i "${gpu}" \
        --query-gpu=memory.total,memory.free \
        --format=csv,noheader,nounits |
        awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
    )
    [[ -n "${total}" && -n "${free}" ]] || \
      trace_vb_die "could not read memory for physical GPU ${gpu}"
    (( free >= min_free_mib )) || \
      trace_vb_die "GPU ${gpu} has ${free} MiB free; ${min_free_mib} MiB is required"
  done
}

trace_vb_require_cache() {
  local cache=${1:-${TRACE_VB_SUFFICIENCY_CACHE}}
  [[ -s "${cache}" ]] || trace_vb_die "missing offline sufficiency cache: ${cache}"
}
