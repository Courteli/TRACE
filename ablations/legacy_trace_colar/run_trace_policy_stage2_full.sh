#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_policy_qwen3_instruct
DATASET_DIR=${DATASET_DIR:-${ROOT}/run_outputs/trace_final/data/gsm8k_multirationale_v1}
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/trace_policy/training}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_policy/tmp}
TRAIN_SEED=${TRAIN_SEED:-0}

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage1-best-checkpoint>" >&2
  exit 2
fi
physical_gpus=$1
stage1_checkpoint=$2
if [[ ! -f "${stage1_checkpoint}" ]]; then
  echo "Missing Stage 1 checkpoint: ${stage1_checkpoint}" >&2
  exit 2
fi
"${PYTHON}" - "${stage1_checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
    raise SystemExit("Stage 2 must initialize from a Stage-1 TRACE checkpoint")
keys = checkpoint.get("state_dict", {})
if not any(".trace_teacher." in key for key in keys):
    raise SystemExit(
        "Stage-1 checkpoint is missing the frozen CoT teacher adapter"
    )
if any(".trace_answer." in key for key in keys):
    raise SystemExit("Stage-1 checkpoint unexpectedly contains Stage-2 state")
PY
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "Formal Stage 2 requires exactly four GPUs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_policy_stage2_seed${TRAIN_SEED}}
out_dir="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "${out_dir}" "${TMP_ROOT}"

resume_args=()
load_args=(--load_ckpt_path "${stage1_checkpoint}")
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_CKPT_PATH}" ]]; then
    echo "Missing Stage 2 resume checkpoint: ${RESUME_CKPT_PATH}" >&2
    exit 2
  fi
  resume_args=(--resume_ckpt_path "${RESUME_CKPT_PATH}")
  load_args=()
fi

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE-Policy
full_name=Set-Anchored_Stochastic_Latent_Trajectories_with_Counterfactual_Outcome_Refinement
phase=stage2_counterfactual_outcome_refinement
physical_gpus=${physical_gpus}
stage1_checkpoint=${stage1_checkpoint}
resume_checkpoint=${RESUME_CKPT_PATH:-none}
dataset_dir=${DATASET_DIR}
epochs=10
unique_training_questions_per_epoch=6726
dataloader_rows_per_epoch=6728
ddp_padding_duplicates_per_epoch=2
group_size=8
requested_global_question_batch_size=4
effective_per_device_question_batch_size=1
rollout_batches_per_epoch=1682
policy_updates_per_rollout=2
optimizer_steps_per_epoch=3364
scheduled_optimizer_steps=33640
epoch_question_budget=full_6726_unique_questions_plus_2_standard_DDP_padding_rows
dataset_subset_mutation=false
rollout_schema=iid_conditional_gaussian
center_or_primary_path=false
path_labels=greedy_bottleneck_answer
answer_token_labels=sampled_answer
hard_pairs_per_group=1
counterfactual_directions=correct_to_wrong_and_wrong_to_correct
counterfactual_suffix=recomputed_with_recipient_innovations
transition_credit=all_8_positions_for_each_eligible_pair
stage1_prior=immutable_policy_head_KL_with_frozen_path_dynamics
explicit_teacher_adapter=frozen_stage0_cot_lora_from_stage1_checkpoint
answer_adapter=phase_isolated_LoRA_initialized_from_stage1
full_validation_every_epoch=true
validation_questions=747
checkpoint_selection=best_full_validation_accuracy
checkpoint_state=full_model_optimizer_scheduler_and_stage1_reference
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
    "${load_args[@]}" \
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
    trainer.limit_train_batches=1682 \
    trainer.limit_val_batches=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.val_check_interval=1.0 \
    trainer.gradient_clip_val=0 \
    trainer.default_root_dir="${out_dir}/trainer" \
    save_top_k=1 \
    save_last=true \
    save_weights_only=false \
    model.model_kwargs.do_trace_rl=true \
    model.model_kwargs.trace_policy_config.visual_record_limit=0 \
    model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=6726 \
    model.model_kwargs.trace_rl_config.group_size=8 \
    model.model_kwargs.trace_rl_config.rollout_micro_batch_size=1 \
    model.model_kwargs.trace_rl_config.exp_batch_size=1 \
    model.model_kwargs.trace_rl_config.use_trajectory_policy_loss=true \
    model.model_kwargs.trace_rl_config.use_answer_policy_loss=true \
    model.model_kwargs.trace_rl_config.policy_update_epochs=2 \
    model.training_kwargs.optimizer.lr=8.0e-7 \
    model.training_kwargs.scheduler.warmup_steps=150 \
    model.training_kwargs.scheduler.num_training_steps=33640 \
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
  echo "Stage 2 completed without a validation-best checkpoint" >&2
  exit 1
fi
if [[ -z "${last_checkpoint}" || ! -f "${last_checkpoint}" ]]; then
  echo "Stage 2 completed without last.ckpt" >&2
  exit 1
fi
cat >> "${out_dir}/manifest.txt" <<EOF
best_checkpoint=${best_checkpoint}
last_checkpoint=${last_checkpoint}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
