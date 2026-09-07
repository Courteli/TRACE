#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_final_qwen3_instruct
DATASET_DIR=${DATASET_DIR:-${ROOT}/run_outputs/trace_final/data/gsm8k_multirationale_v1}
STAGE0_CKPT=${STAGE0_CKPT:-/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/checkpoints/epoch0__step6726__monitor0.871.ckpt}
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/trace_final/training}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_final/tmp}
MAX_STARTUP_MEMORY_MIB=${MAX_STARTUP_MEMORY_MIB:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}

physical_gpu=${1:-}
if [[ -z "${physical_gpu}" || ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 <physical-gpu>" >&2
  exit 2
fi
for required in \
  "${STAGE0_CKPT}" \
  "${DATASET_DIR}/train.json" \
  "${DATASET_DIR}/val.json" \
  "${DATASET_DIR}/rationale_set_audit.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing formal Stage 1 input: ${required}" >&2
    exit 2
  fi
done

used=$(nvidia-smi -i "${physical_gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
used=${used//[[:space:]]/}
if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > MAX_STARTUP_MEMORY_MIB )); then
  echo "Physical GPU ${physical_gpu} is not clean: ${used:-unknown} MiB" >&2
  exit 2
fi

RUN_TAG=${RUN_TAG:-20260718_trace_final_stage1_seed${TRAIN_SEED}}
out_dir="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "${out_dir}" "${TMP_ROOT}"
cd "${ROOT}"

resume_args=()
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_CKPT_PATH}" ]]; then
    echo "Missing Stage 1 resume checkpoint: ${RESUME_CKPT_PATH}" >&2
    exit 2
  fi
  resume_args=(--resume_ckpt_path "${RESUME_CKPT_PATH}")
fi

cat > "${out_dir}/manifest.txt" <<EOF
phase=stage1_set_anchored_formation
physical_gpu=${physical_gpu}
initial_ckpt=${STAGE0_CKPT}
resume_checkpoint=${RESUME_CKPT_PATH:-none}
dataset_dir=${DATASET_DIR}
train_seed=${TRAIN_SEED}
max_epochs=10
questions_per_epoch=6726
validation_questions=747
validation_every_epoch=true
early_stopping_patience=4
checkpoint_state=full_model_optimizer_scheduler
checkpoint_retention=validation_best_plus_rolling_last
answer_path_bottleneck=true
tiny_dataset=false
epoch_scaling=1
test_times=1
started_at=$(date --iso-8601=seconds)
EOF

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  "${PYTHON}" run.py \
    --model "${MODEL}" \
    --dataset trace_qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /disk1/dingxukai \
    --load_ckpt_path "${STAGE0_CKPT}" \
    "${resume_args[@]}" \
    --test_times 1 \
    --seed "${TRAIN_SEED}" \
    --log_suffix "${RUN_TAG}" \
    data_module.dataset_dir="${DATASET_DIR}" \
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
    model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
    model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true \
    model.training_kwargs.optimizer.lr=1.0e-5 \
    model.training_kwargs.scheduler.warmup_steps=600 \
    model.training_kwargs.scheduler.num_training_steps=67260 \
    2>&1 | tee "${out_dir}/train.log"

log_root="${ROOT}/logs/${MODEL}/trace_qsa-gsm"
mapfile -t log_dirs < <(
  find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${RUN_TAG}" -print | sort
)
if [[ "${#log_dirs[@]}" -eq 0 ]]; then
  echo "No logger directory found for ${RUN_TAG}" >&2
  exit 1
fi
latest_log_dir="${log_dirs[$((${#log_dirs[@]} - 1))]}"
mapfile -t best_files < <(
  find "${log_dirs[@]/%//checkpoints}" -maxdepth 1 -type f \
    -name "epoch*__step*__monitor*.ckpt" ! -name "last.ckpt" -print
)
if [[ "${#best_files[@]}" -eq 0 ]]; then
  echo "No validation-best Stage 1 checkpoint was found" >&2
  exit 1
fi
best_checkpoint=$(
  printf '%s\n' "${best_files[@]}" |
    "${PYTHON}" -c '
import re
import sys

pattern = re.compile(r"epoch(\d+)__step(\d+)__monitor([-+0-9.eE]+)\.ckpt$")
items = []
for raw in sys.stdin:
    path = raw.strip()
    match = pattern.search(path)
    if match:
        items.append((float(match.group(3)), int(match.group(2)), path))
if not items:
    raise SystemExit("No parseable validation-best Stage 1 checkpoint")
print(max(items)[2])
'
)
best_name=$(basename "${best_checkpoint}")
if [[ ! "${best_name}" =~ ^epoch([0-9])__step([0-9]+)__monitor([-+0-9.eE]+)\.ckpt$ ]]; then
  echo "Unexpected Stage 1 checkpoint name: ${best_name}" >&2
  exit 1
fi
best_epoch="${BASH_REMATCH[1]}"
best_step="${BASH_REMATCH[2]}"
best_monitor="${BASH_REMATCH[3]}"
if (( best_step != (best_epoch + 1) * 6726 )); then
  echo "Stage 1 epoch/step mismatch: ${best_name}" >&2
  exit 1
fi
if [[ ! -f "${latest_log_dir}/checkpoints/last.ckpt" ]]; then
  echo "Missing Stage 1 last.ckpt" >&2
  exit 1
fi

cat >> "${out_dir}/manifest.txt" <<EOF
logger_dir=${latest_log_dir}
best_logger_dir=$(dirname "$(dirname "${best_checkpoint}")")
best_checkpoint=${best_checkpoint}
best_epoch=${best_epoch}
best_step=${best_step}
best_monitor=${best_monitor}
last_checkpoint=${latest_log_dir}/checkpoints/last.ckpt
finished_at=$(date --iso-8601=seconds)
EOF
