#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_final_qwen3_instruct
DATASET_DIR=${DATASET_DIR:-${ROOT}/run_outputs/trace_final/data/gsm8k_multirationale_v1}
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/trace_final/training}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_final/tmp}
MAX_STARTUP_MEMORY_MIB=${MAX_STARTUP_MEMORY_MIB:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
STAGE2_VARIANT=${STAGE2_VARIANT:-outcome_local}
LOCAL_RANKING_WEIGHT=${LOCAL_RANKING_WEIGHT:-0.10}
SFT_REPLAY_WEIGHT=${SFT_REPLAY_WEIGHT:-0.05}
ACCURACY_GRADIENT_GUARD=${ACCURACY_GRADIENT_GUARD:-true}
RANKING_GRAD_RATIO=${RANKING_GRAD_RATIO:-0.25}

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage1-checkpoint>" >&2
  exit 2
fi
physical_gpus=$1
stage1_checkpoint=$2
if [[ ! -f "${stage1_checkpoint}" ]]; then
  echo "Missing Stage 1 checkpoint: ${stage1_checkpoint}" >&2
  exit 2
fi
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "Stage 2 requires exactly four GPUs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi
for gpu in "${gpu_array[@]}"; do
  used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
  used=${used//[[:space:]]/}
  if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > MAX_STARTUP_MEMORY_MIB )); then
    echo "Physical GPU ${gpu} is not clean: ${used:-unknown} MiB" >&2
    exit 2
  fi
done

RUN_TAG=${RUN_TAG:-20260718_trace_final_stage2_seed${TRAIN_SEED}}
out_dir="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "${out_dir}" "${TMP_ROOT}"
cd "${ROOT}"

resume_args=()
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_CKPT_PATH}" ]]; then
    echo "Missing Stage 2 resume checkpoint: ${RESUME_CKPT_PATH}" >&2
    exit 2
  fi
  resume_args=(--resume_ckpt_path "${RESUME_CKPT_PATH}")
fi

cat > "${out_dir}/manifest.txt" <<EOF
phase=stage2_${STAGE2_VARIANT}
stage2_variant=${STAGE2_VARIANT}
physical_gpus=${physical_gpus}
initial_ckpt=${stage1_checkpoint}
resume_checkpoint=${RESUME_CKPT_PATH:-none}
dataset_dir=${DATASET_DIR}
train_seed=${TRAIN_SEED}
epochs=10
n_train_samples_per_epoch=2048
group_size=8
center_path_in_each_group=true
full_validation_every_epoch=true
validation_questions=747
checkpoint_selection=best_full_validation_accuracy
checkpoint_state=full_model_optimizer_scheduler
checkpoint_retention=validation_best_plus_rolling_last
local_ranking_weight=${LOCAL_RANKING_WEIGHT}
sft_replay_weight=${SFT_REPLAY_WEIGHT}
accuracy_gradient_guard=${ACCURACY_GRADIENT_GUARD}
ranking_grad_ratio=${RANKING_GRAD_RATIO}
test_times=1
started_at=$(date --iso-8601=seconds)
EOF

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${PYTHON}" run.py \
    --model "${MODEL}" \
    --dataset trace_qsa \
    --trainer default \
    --devices 0,1,2,3 \
    --workspace_path /disk1/dingxukai \
    --load_ckpt_path "${stage1_checkpoint}" \
    "${resume_args[@]}" \
    --test_times 1 \
    --seed "${TRAIN_SEED}" \
    --disable_early_stopping \
    --log_suffix "${RUN_TAG}" \
    data_module.dataset_dir="${DATASET_DIR}" \
    data_module.tiny_dataset=false \
    data_module.epoch_scaling=1 \
    batch_size=4 \
    val_batch_size=1 \
    num_workers=4 \
    persistent_workers=false \
    trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0 \
    trainer.max_epochs=10 \
    trainer.max_steps=-1 \
    trainer.limit_train_batches=1.0 \
    trainer.limit_val_batches=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.val_check_interval=1.0 \
    trainer.gradient_clip_val=0 \
    trainer.default_root_dir="${out_dir}/trainer" \
    save_top_k=1 \
    save_last=true \
    model.model_kwargs.do_trace_rl=true \
    model.model_kwargs.trace_bridge_config.enable_trajectory_formation=true \
    model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
    model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true \
    model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048 \
    model.model_kwargs.trace_rl_config.group_size=8 \
    model.model_kwargs.trace_rl_config.exp_batch_size=1 \
    model.model_kwargs.trace_rl_config.stage2_local_ranking_weight="${LOCAL_RANKING_WEIGHT}" \
    model.model_kwargs.trace_rl_config.stage2_sft_replay_weight="${SFT_REPLAY_WEIGHT}" \
    model.model_kwargs.trace_rl_config.stage2_accuracy_gradient_guard="${ACCURACY_GRADIENT_GUARD}" \
    model.model_kwargs.trace_rl_config.stage2_ranking_grad_ratio="${RANKING_GRAD_RATIO}" \
    model.model_kwargs.trace_rl_config.stage2_ranking_micro_batch_size=1 \
    model.training_kwargs.optimizer.lr=8.0e-7 \
    model.training_kwargs.scheduler.warmup_steps=75 \
    model.training_kwargs.scheduler.num_training_steps=5120 \
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
  echo "No validation-best Stage 2 checkpoint was found" >&2
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
    raise SystemExit("No parseable validation-best Stage 2 checkpoint")
print(max(items)[2])
'
)
best_name=$(basename "${best_checkpoint}")
if [[ ! "${best_name}" =~ ^epoch([0-9])__step([0-9]+)__monitor([-+0-9.eE]+)\.ckpt$ ]]; then
  echo "Unexpected Stage 2 checkpoint name: ${best_name}" >&2
  exit 1
fi
best_epoch="${BASH_REMATCH[1]}"
best_step="${BASH_REMATCH[2]}"
best_monitor="${BASH_REMATCH[3]}"
if (( best_step != (best_epoch + 1) * 512 )); then
  echo "Stage 2 epoch/step mismatch: ${best_name}" >&2
  exit 1
fi
if [[ ! -f "${latest_log_dir}/checkpoints/last.ckpt" ]]; then
  echo "Missing Stage 2 last.ckpt" >&2
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
