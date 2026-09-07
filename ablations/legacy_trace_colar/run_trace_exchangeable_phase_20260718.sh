#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_exchangeable_qwen3_instruct
DATASET_DIR=${DATASET_DIR:-/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc}
STAGE0_CKPT=${STAGE0_CKPT:-/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/checkpoints/epoch0__step6726__monitor0.871.ckpt}
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/trace_exchangeable/20260718_exchangeable_local_ranking}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_exchangeable/tmp}
MAX_STARTUP_MEMORY_MIB=${MAX_STARTUP_MEMORY_MIB:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}

usage() {
  cat <<'EOF'
Usage:
  run_trace_exchangeable_phase_20260718.sh stage1-formation <physical-gpu>
  run_trace_exchangeable_phase_20260718.sh stage1-plain <physical-gpu>
  run_trace_exchangeable_phase_20260718.sh stage2-answer <physical-gpu-csv> <stage1-ckpt> <formation:true|false>
  run_trace_exchangeable_phase_20260718.sh stage2-full <physical-gpu-csv> <stage1-ckpt> <formation:true|false>

This script starts only the requested phase. It does not wait for GPUs, enqueue
another phase, or launch evaluation automatically.
EOF
}

phase=${1:-}
physical_gpus=${2:-}
initial_ckpt=${3:-}
formation=${4:-true}
if [[ -z "${phase}" || -z "${physical_gpus}" ]]; then
  usage
  exit 2
fi

require_clean_gpu() {
  local gpu=$1
  local used
  used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
  used=${used//[[:space:]]/}
  if [[ ! "${used}" =~ ^[0-9]+$ ]]; then
    echo "Could not read memory usage for physical GPU ${gpu}" >&2
    exit 2
  fi
  if (( used > MAX_STARTUP_MEMORY_MIB )); then
    echo "Physical GPU ${gpu} is not clean: ${used} MiB already used" >&2
    exit 2
  fi
}

mkdir -p "${RUN_ROOT}" "${TMP_ROOT}"
cd "${ROOT}"

case "${phase}" in
  stage1-formation)
    initial_ckpt="${STAGE0_CKPT}"
    formation=true
    do_rl=false
    ranking_weight=0.0
    max_epochs=5
    logical_devices=0
    batch_size=1
    lr=1.0e-5
    warmup_steps=600
    training_steps=33630
    ;;
  stage1-plain)
    initial_ckpt="${STAGE0_CKPT}"
    formation=false
    do_rl=false
    ranking_weight=0.0
    max_epochs=5
    logical_devices=0
    batch_size=1
    lr=1.0e-5
    warmup_steps=600
    training_steps=33630
    ;;
  stage2-answer|stage2-full)
    if [[ -z "${initial_ckpt}" || ! -f "${initial_ckpt}" ]]; then
      echo "A valid Stage 1 checkpoint is required for ${phase}" >&2
      exit 2
    fi
    IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
    if [[ "${#gpu_array[@]}" -ne 4 ]]; then
      echo "${phase} requires exactly four clean 24 GB GPUs; got ${#gpu_array[@]}" >&2
      exit 2
    fi
    logical_devices=$(seq -s, 0 $((${#gpu_array[@]} - 1)))
    batch_size=${#gpu_array[@]}
    do_rl=true
    max_epochs=5
    lr=8.0e-7
    warmup_steps=75
    training_steps=2560
    if [[ "${phase}" == "stage2-full" ]]; then
      ranking_weight=0.10
    else
      ranking_weight=0.0
    fi
    ;;
  *)
    usage
    exit 2
    ;;
esac

IFS=',' read -r -a requested_gpu_array <<< "${physical_gpus}"
unique_gpu_count=$(printf '%s\n' "${requested_gpu_array[@]}" | sort -u | wc -l)
if [[ "${unique_gpu_count}" -ne "${#requested_gpu_array[@]}" ]]; then
  echo "Physical GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi
for gpu in "${requested_gpu_array[@]}"; do
  require_clean_gpu "${gpu}"
done

run_tag="${RUN_TAG:-20260718_${phase}_formation-${formation}}"
out_dir="${RUN_ROOT}/${run_tag}"
mkdir -p "${out_dir}"
{
  printf 'phase=%s\n' "${phase}"
  printf 'physical_gpus=%s\n' "${physical_gpus}"
  printf 'logical_devices=%s\n' "${logical_devices}"
  printf 'initial_ckpt=%s\n' "${initial_ckpt}"
  printf 'formation=%s\n' "${formation}"
  printf 'ranking_weight=%s\n' "${ranking_weight}"
  printf 'train_seed=%s\n' "${TRAIN_SEED}"
  printf 'tiny_dataset=false\n'
  printf 'epoch_scaling=1\n'
  printf 'limit_train_batches=1.0\n'
  printf 'limit_val_batches=1.0\n'
  printf 'test_times=1\n'
  printf 'started_at=%s\n' "$(date '+%F %T')"
} > "${out_dir}/manifest.txt"

strategy=auto
if [[ "${batch_size}" -gt 1 ]]; then
  strategy=ddp_find_unused_parameters_true
fi

env \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${PYTHON}" run.py \
    --model "${MODEL}" \
    --dataset qsa \
    --trainer default \
    --devices "${logical_devices}" \
    --workspace_path /disk1/dingxukai \
    --load_ckpt_path "${initial_ckpt}" \
    --test_times 1 \
    --seed "${TRAIN_SEED}" \
    --disable_early_stopping \
    --log_suffix "${run_tag}" \
    dataset_dir="${DATASET_DIR}" \
    data_module.tiny_dataset=false \
    data_module.epoch_scaling=1 \
    batch_size="${batch_size}" \
    val_batch_size=1 \
    num_workers=4 \
    persistent_workers=false \
    trainer.strategy="${strategy}" \
    trainer.num_sanity_val_steps=0 \
    trainer.max_epochs="${max_epochs}" \
    trainer.max_steps=-1 \
    trainer.limit_train_batches=1.0 \
    trainer.limit_val_batches=1.0 \
    trainer.check_val_every_n_epoch=5 \
    trainer.val_check_interval=1.0 \
    trainer.gradient_clip_val=0 \
    trainer.default_root_dir="${out_dir}/trainer" \
    model.model_kwargs.do_trace_rl="${do_rl}" \
    model.model_kwargs.trace_bridge_config.enable_trajectory_formation="${formation}" \
    model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048 \
    model.model_kwargs.trace_rl_config.group_size=8 \
    model.model_kwargs.trace_rl_config.exp_batch_size=1 \
    model.model_kwargs.trace_rl_config.stage2_local_ranking_weight="${ranking_weight}" \
    model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.05 \
    model.model_kwargs.trace_rl_config.stage2_accuracy_gradient_guard=true \
    model.model_kwargs.trace_rl_config.stage2_ranking_grad_ratio=0.25 \
    model.model_kwargs.trace_rl_config.stage2_ranking_micro_batch_size=1 \
    model.training_kwargs.optimizer.lr="${lr}" \
    model.training_kwargs.scheduler.warmup_steps="${warmup_steps}" \
    model.training_kwargs.scheduler.num_training_steps="${training_steps}" \
    2>&1 | tee "${out_dir}/train.log"

log_root="${ROOT}/logs/${MODEL}/qsa-gsm"
mapfile -t matched_log_dirs < <(
  find "${log_root}" -mindepth 1 -maxdepth 1 -type d -name "*_${run_tag}" -print | sort
)
if [[ "${#matched_log_dirs[@]}" -ne 1 ]]; then
  echo "Expected exactly one logger directory for ${run_tag}; found ${#matched_log_dirs[@]}" >&2
  exit 1
fi
log_dir="${matched_log_dirs[0]}"
mapfile -t final_checkpoints < <(
  find "${log_dir}/checkpoints" -maxdepth 1 -type f \
    -name "epoch$((max_epochs - 1))__step${training_steps}__monitor*.ckpt" \
    ! -name "last.ckpt" -print | sort
)
if [[ "${#final_checkpoints[@]}" -ne 1 ]]; then
  echo "Expected one final monitored checkpoint at epoch $((max_epochs - 1)), step ${training_steps}; found ${#final_checkpoints[@]}" >&2
  exit 1
fi
if [[ ! -f "${log_dir}/checkpoints/last.ckpt" || ! -f "${log_dir}/hparams.yaml" ]]; then
  echo "Final checkpoint package is incomplete in ${log_dir}" >&2
  exit 1
fi

printf 'logger_dir=%s\n' "${log_dir}" >> "${out_dir}/manifest.txt"
printf 'checkpoint=%s\n' "${final_checkpoints[0]}" >> "${out_dir}/manifest.txt"
printf 'last_checkpoint=%s\n' "${log_dir}/checkpoints/last.ckpt" >> "${out_dir}/manifest.txt"
printf 'finished_at=%s\n' "$(date '+%F %T')" >> "${out_dir}/manifest.txt"
