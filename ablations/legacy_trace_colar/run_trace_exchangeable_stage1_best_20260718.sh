#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_exchangeable_qwen3_instruct
DATASET_DIR=${DATASET_DIR:-/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc}
STAGE0_CKPT=${STAGE0_CKPT:-/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/checkpoints/epoch0__step6726__monitor0.871.ckpt}
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/trace_exchangeable/mainline_training}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_exchangeable/tmp}
MAX_STARTUP_MEMORY_MIB=${MAX_STARTUP_MEMORY_MIB:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}

usage() {
  cat <<'EOF'
Usage:
  run_trace_exchangeable_stage1_best_20260718.sh <physical-gpu>

Runs the formal TRACE Stage 1 budget: at most 10 complete 6,726-question
epochs, complete 747-question validation after every epoch, and patience-4
early stopping. The validation-best checkpoint is written to the manifest.
EOF
}

physical_gpu=${1:-}
if [[ -z "${physical_gpu}" || ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  usage
  exit 2
fi
if [[ ! -f "${STAGE0_CKPT}" ]]; then
  echo "Missing Stage 0 checkpoint: ${STAGE0_CKPT}" >&2
  exit 2
fi

used=$(nvidia-smi -i "${physical_gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
used=${used//[[:space:]]/}
if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > MAX_STARTUP_MEMORY_MIB )); then
  echo "Physical GPU ${physical_gpu} is not clean: ${used:-unknown} MiB already used" >&2
  exit 2
fi

RUN_TAG=${RUN_TAG:-20260718_trace_exchangeable_stage1_seed${TRAIN_SEED}}
out_dir="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "${out_dir}" "${TMP_ROOT}"
cd "${ROOT}"

cat > "${out_dir}/manifest.txt" <<EOF
phase=stage1-formation
physical_gpu=${physical_gpu}
initial_ckpt=${STAGE0_CKPT}
train_seed=${TRAIN_SEED}
max_epochs=10
questions_per_epoch=6726
validation_questions=747
validation_every_epoch=true
early_stopping_patience=4
tiny_dataset=false
epoch_scaling=1
test_times=1
started_at=$(date '+%F %T')
EOF

env \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  "${PYTHON}" run.py \
    --model "${MODEL}" \
    --dataset qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /disk1/dingxukai \
    --load_ckpt_path "${STAGE0_CKPT}" \
    --test_times 1 \
    --seed "${TRAIN_SEED}" \
    --log_suffix "${RUN_TAG}" \
    dataset_dir="${DATASET_DIR}" \
    data_module.tiny_dataset=false \
    data_module.epoch_scaling=1 \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    persistent_workers=false \
    trainer.strategy=auto \
    trainer.num_sanity_val_steps=0 \
    trainer.max_epochs=10 \
    trainer.max_steps=-1 \
    trainer.limit_train_batches=1.0 \
    trainer.limit_val_batches=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.val_check_interval=1.0 \
    trainer.gradient_clip_val=0.3 \
    trainer.default_root_dir="${out_dir}/trainer" \
    save_top_k=1 \
    save_last=true \
    model.model_kwargs.do_trace_rl=false \
    model.model_kwargs.trace_bridge_config.enable_trajectory_formation=true \
    model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048 \
    model.model_kwargs.trace_rl_config.group_size=8 \
    model.model_kwargs.trace_rl_config.exp_batch_size=1 \
    model.model_kwargs.trace_rl_config.stage2_local_ranking_weight=0.0 \
    model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.05 \
    model.model_kwargs.trace_rl_config.stage2_accuracy_gradient_guard=true \
    model.model_kwargs.trace_rl_config.stage2_ranking_grad_ratio=0.25 \
    model.model_kwargs.trace_rl_config.stage2_ranking_micro_batch_size=1 \
    model.training_kwargs.optimizer.lr=1.0e-5 \
    model.training_kwargs.scheduler.warmup_steps=600 \
    model.training_kwargs.scheduler.num_training_steps=67260 \
    2>&1 | tee "${out_dir}/train.log"

log_root="${ROOT}/logs/${MODEL}/qsa-gsm"
mapfile -t matched_log_dirs < <(
  find "${log_root}" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_TAG}" -print | sort
)
if [[ "${#matched_log_dirs[@]}" -ne 1 ]]; then
  echo "Expected exactly one logger directory for ${RUN_TAG}; found ${#matched_log_dirs[@]}" >&2
  exit 1
fi
log_dir="${matched_log_dirs[0]}"
mapfile -t monitored_checkpoints < <(
  find "${log_dir}/checkpoints" -maxdepth 1 -type f \
    -name "epoch*__step*__monitor*.ckpt" ! -name "last.ckpt" -print | sort
)
if [[ "${#monitored_checkpoints[@]}" -ne 1 ]]; then
  echo "Expected exactly one top-1 Stage 1 checkpoint; found ${#monitored_checkpoints[@]}" >&2
  exit 1
fi

best_checkpoint="${monitored_checkpoints[0]}"
best_name=$(basename "${best_checkpoint}")
if [[ ! "${best_name}" =~ ^epoch([0-9])__step([0-9]+)__monitor([-+0-9.eE]+)\.ckpt$ ]]; then
  echo "Unexpected Stage 1 best checkpoint name: ${best_name}" >&2
  exit 1
fi
best_epoch="${BASH_REMATCH[1]}"
best_step="${BASH_REMATCH[2]}"
best_monitor="${BASH_REMATCH[3]}"
if (( best_step != (best_epoch + 1) * 6726 )); then
  echo "Stage 1 best checkpoint epoch/step mismatch: ${best_name}" >&2
  exit 1
fi
if [[ ! -f "${log_dir}/checkpoints/last.ckpt" || ! -f "${log_dir}/hparams.yaml" ]]; then
  echo "Stage 1 checkpoint package is incomplete in ${log_dir}" >&2
  exit 1
fi

cat >> "${out_dir}/manifest.txt" <<EOF
logger_dir=${log_dir}
checkpoint=${best_checkpoint}
best_checkpoint=${best_checkpoint}
best_epoch=${best_epoch}
best_step=${best_step}
best_monitor=${best_monitor}
last_checkpoint=${log_dir}/checkpoints/last.ckpt
finished_at=$(date '+%F %T')
EOF
