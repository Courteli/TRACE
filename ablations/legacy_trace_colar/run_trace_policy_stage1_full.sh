#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_policy_qwen3_instruct
DATASET_DIR=${DATASET_DIR:-${ROOT}/run_outputs/trace_final/data/gsm8k_multirationale_v1}
STAGE0_CKPT=${STAGE0_CKPT:-/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/checkpoints/epoch0__step6726__monitor0.871.ckpt}
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/trace_policy/training}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_policy/tmp}
TRAIN_SEED=${TRAIN_SEED:-0}

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv>" >&2
  exit 2
fi
physical_gpus=$1
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "Formal Stage 1 requires exactly four GPUs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi
for required in \
  "${STAGE0_CKPT}" \
  "${DATASET_DIR}/train.json" \
  "${DATASET_DIR}/val.json" \
  "${DATASET_DIR}/test.json" \
  "${DATASET_DIR}/rationale_set_audit.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing formal input: ${required}" >&2
    exit 2
  fi
done
stage0_hparams="$(dirname "$(dirname "${STAGE0_CKPT}")")/hparams.yaml"
if [[ ! -f "${stage0_hparams}" ]]; then
  echo "Missing Stage-0 hparams: ${stage0_hparams}" >&2
  exit 2
fi
if ! grep -q "target: src.models.cot.LitCot" "${stage0_hparams}" ||
   ! grep -q "sft_method: cot" "${stage0_hparams}"; then
  echo "Stage 0 must be a plain CoT-SFT checkpoint" >&2
  exit 2
fi
"${PYTHON}" - "${STAGE0_CKPT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
keys = list(checkpoint.get("state_dict", {}))
prohibited = (
    "trajectory_policy",
    "trace_",
    "latent_bridge",
    "step_compressor",
    "latent_relation",
)
found = {token: [key for key in keys if token in key] for token in prohibited}
found = {token: values for token, values in found.items() if values}
if found:
    raise SystemExit(
        "Stage-0 checkpoint contains latent-policy state: "
        + ", ".join(f"{token}={len(values)}" for token, values in found.items())
    )
if not keys:
    raise SystemExit("Stage-0 checkpoint has no model state")
PY

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_policy_stage1_seed${TRAIN_SEED}}
out_dir="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "${out_dir}" "${TMP_ROOT}"

resume_args=()
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_CKPT_PATH}" ]]; then
    echo "Missing Stage 1 resume checkpoint: ${RESUME_CKPT_PATH}" >&2
    exit 2
  fi
  resume_args=(--resume_ckpt_path "${RESUME_CKPT_PATH}")
fi

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE-Policy
full_name=Set-Anchored_Stochastic_Latent_Trajectories_with_Counterfactual_Outcome_Refinement
phase=stage1_set_anchored_formation
physical_gpus=${physical_gpus}
initial_checkpoint=${STAGE0_CKPT}
initial_checkpoint_sha256=$(sha256sum "${STAGE0_CKPT}" | awk '{print $1}')
initial_checkpoint_contract=plain_CoT_SFT_without_latent_policy_state
resume_checkpoint=${RESUME_CKPT_PATH:-none}
dataset_dir=${DATASET_DIR}
train_questions=6726
validation_questions=747
test_questions=1319
teacher_set_size=4
max_semantic_modes=2
student_paths_per_question=4_iid
explicit_teacher_adapter=frozen_stage0_cot_lora
teacher_adapter_checkpointed=true
center_or_primary_path=false
path_bottleneck=true
requested_global_batch_size=4
effective_per_device_batch_size=1
optimizer_steps_per_epoch=1682
scheduled_optimizer_steps=16820
max_epochs=10
full_validation_every_epoch=true
early_stopping_patience=4
tiny_dataset=false
epoch_scaling=1
test_times=1
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF

cd "${ROOT}"
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
    --load_ckpt_path "${STAGE0_CKPT}" \
    "${resume_args[@]}" \
    --test_times 1 \
    --seed "${TRAIN_SEED}" \
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
    trainer.gradient_clip_val=0.3 \
    trainer.default_root_dir="${out_dir}/trainer" \
    save_top_k=1 \
    save_last=true \
    save_weights_only=false \
    model.model_kwargs.do_trace_rl=false \
    model.model_kwargs.trace_policy_config.stage1_teacher_set_size=4 \
    model.model_kwargs.trace_policy_config.stage1_max_semantic_modes=2 \
    model.model_kwargs.trace_policy_config.visual_record_limit=0 \
    model.training_kwargs.optimizer.lr=1.0e-5 \
    model.training_kwargs.scheduler.warmup_steps=150 \
    model.training_kwargs.scheduler.num_training_steps=16820 \
    2>&1 | tee "${out_dir}/train.log"

log_root="${ROOT}/logs/${MODEL}/trace_qsa-gsm"
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
  echo "Stage 1 completed without a validation-best checkpoint" >&2
  exit 1
fi
if [[ -z "${last_checkpoint}" || ! -f "${last_checkpoint}" ]]; then
  echo "Stage 1 completed without last.ckpt" >&2
  exit 1
fi
cat >> "${out_dir}/manifest.txt" <<EOF
best_checkpoint=${best_checkpoint}
last_checkpoint=${last_checkpoint}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
