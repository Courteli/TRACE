#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/TRACE
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
DATASET_DIR=${DATASET_DIR:-${ROOT}/data/raw/GSM8k-Aug-NL}
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/training}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/tmp}
TRAIN_SEED=${TRAIN_SEED:-0}

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv>" >&2
  exit 2
fi
physical_gpus=$1
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "Fresh CoT warm start requires exactly four GPUs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi

cd "${ROOT}"
"${PYTHON}" tools/data_contract_audit.py >/dev/null
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_stage0_seed${TRAIN_SEED}}
out_dir="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "${out_dir}" "${TMP_ROOT}"

resume_args=()
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_CKPT_PATH}" ]]; then
    echo "Missing Stage-0 resume checkpoint: ${RESUME_CKPT_PATH}" >&2
    exit 2
  fi
  resume_args=(--resume_ckpt_path "${RESUME_CKPT_PATH}")
fi

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE
phase=generic_cot_sft_warm_start
trace_method_stage=false
initialization=fresh_Qwen3-4B-Instruct
old_checkpoint_reused=false
dataset_dir=${DATASET_DIR}
train_file=gsm8k_train_processed.jsonl
train_questions=6726
validation_questions=747
physical_gpus=${physical_gpus}
requested_global_batch_size=4
effective_per_device_batch_size=1
max_epochs=3
full_validation_every_epoch=true
test_times=1
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF

env \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${PYTHON}" run.py \
    --model cot_qwen3_instruct \
    --dataset gsm8k_aug_nl \
    --trainer default \
    --devices 0,1,2,3 \
    --workspace_path /disk1/dingxukai \
    "${resume_args[@]}" \
    --test_times 1 \
    --seed "${TRAIN_SEED}" \
    --log_suffix "${RUN_TAG}" \
    data_module.dataset_dir="${DATASET_DIR}" \
    data_module.enforce_registered_source=true \
    data_module.tiny_dataset=false \
    data_module.epoch_scaling=1 \
    batch_size=4 \
    val_batch_size=1 \
    num_workers=4 \
    persistent_workers=false \
    trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0 \
    trainer.max_epochs=3 \
    trainer.limit_train_batches=1.0 \
    trainer.limit_val_batches=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.val_check_interval=1.0 \
    trainer.gradient_clip_val=1.0 \
    trainer.default_root_dir="${out_dir}/trainer" \
    save_top_k=1 \
    save_last=true \
    save_weights_only=false \
    2>&1 | tee "${out_dir}/train.log"

log_root="${ROOT}/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl"
best_checkpoint=$(
  find "${log_root}" -type f \
    -path "*_${RUN_TAG}/checkpoints/epoch*__step*__monitor*.ckpt" \
    ! -name "last.ckpt" -print |
    sort |
    tail -n 1
)
last_checkpoint=$(
  find "${log_root}" -type f \
    -path "*_${RUN_TAG}/checkpoints/last.ckpt" -print |
    sort |
    tail -n 1
)
if [[ -z "${best_checkpoint}" || ! -f "${best_checkpoint}" ]]; then
  echo "Fresh CoT SFT completed without a validation-best checkpoint" >&2
  exit 1
fi
if [[ -z "${last_checkpoint}" || ! -f "${last_checkpoint}" ]]; then
  echo "Fresh CoT SFT completed without last.ckpt" >&2
  exit 1
fi
cat >> "${out_dir}/manifest.txt" <<EOF
best_checkpoint=${best_checkpoint}
last_checkpoint=${last_checkpoint}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
