#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_ROOT=${DATA_ROOT:-/disk1/dingxukai/TRACE}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/disk1/dingxukai/TRACE/role_semantic_runs}
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
DATASET_DIR=${DATASET_DIR:-${DATA_ROOT}/data/raw/GSM8k-Aug-NL}
RUN_ROOT=${RUN_ROOT:-${ARTIFACT_ROOT}/training}
TMP_ROOT=${TMP_ROOT:-${ARTIFACT_ROOT}/tmp}
LOG_ROOT=${LOG_ROOT:-${ARTIFACT_ROOT}/logs}
TRAIN_SEED=${TRAIN_SEED:-0}

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage1-best-checkpoint>" >&2
  exit 2
fi
physical_gpus=$1
stage1_checkpoint=$2
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "TRACE Stage 2 requires exactly four GPUs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi
if [[ ! -f "${stage1_checkpoint}" ]]; then
  echo "Missing Stage-1 checkpoint: ${stage1_checkpoint}" >&2
  exit 2
fi
stage1_hparams="$(dirname "$(dirname "${stage1_checkpoint}")")/hparams.yaml"
if [[ ! -f "${stage1_hparams}" ]] \
  || ! grep -Fq "workspace_path: ${CODE_ROOT}" "${stage1_hparams}" \
  || ! grep -qi "do_trace_rl: false" "${stage1_hparams}"; then
  echo "Stage 2 requires a new-code Stage-1 checkpoint with do_trace_rl=false" >&2
  exit 2
fi

cd "${CODE_ROOT}"
TRACE_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" tools/data_contract_audit.py >/dev/null
"${PYTHON}" - "${stage1_checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
    raise SystemExit("Stage 2 must initialize from a Stage-1 TRACE checkpoint")
keys = checkpoint.get("state_dict", {})
required = (
    "trajectory_policy.",
    "trajectory_posterior.",
)
for prefix in required:
    if not any(key.startswith(prefix) for key in keys):
        raise SystemExit(f"Stage-1 checkpoint is missing {prefix}")
if not any(".trace_cot_encoder." in key for key in keys):
    raise SystemExit("Stage-1 checkpoint lacks the frozen single-CoT encoder")
if any(".trace_answer." in key for key in keys):
    raise SystemExit("Stage-1 checkpoint unexpectedly contains Stage-2 state")
PY

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_stage2_seed${TRAIN_SEED}}
out_dir="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "${out_dir}" "${TMP_ROOT}"
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
  echo "Formal role-semantic Stage 2 must start from this run's Stage-1 checkpoint; external RESUME_CKPT_PATH is forbidden" >&2
  exit 2
fi
tee_args=(-a)

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE-Role-Latent-RL
phase=stage2_role_semantic_latent_policy_refinement
stage1_checkpoint=${stage1_checkpoint}
dataset_dir=${DATASET_DIR}
train_reward_teacher=frozen_Stage1_CoT_role_semantic_targets
train_CoT_usage=reward_scoring_only_no_policy_conditioning
deployment_policy_cot_conditioning=false
validation_policy_cot_conditioning=false
test_policy_cot_conditioning=false
posthoc_role_evidence_CoT_usage=scoring_only_never_answer_conditioning
group_size=8_iid_question_only_prior_paths
latent_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_CHECK_COMMIT
role_specific_policy_heads=true
answer_context=question_plus_COMMIT_latent_only
deployment_output=compact_equations_plus_answer_with_48_token_budget
step_credit=frozen_stage1_role_semantic_scores_plus_terminal_answer_reward
return_definition=position_local_discounted_role_rewards_plus_terminal_reward
role_entropy=PLAN_high_SOLVE_decay_CHECK_low_COMMIT_deterministic
working_memory=within_question_recurrent_with_CHECK_correction
transition_advantage_normalization=within_question_per_role_position
terminal_path_credit=exact_answer_plus_frozen_gold_answer_score_minus_length_penalty
stage1_prior=role_policy_KL
answer_decoder_trainable_in_stage2=false
dense_outcome_weight=0.25
rollout_micro_batch_size=2
optimization_micro_batch_size=2
ddp_gradient_sync=standard_lightning_ddp_each_micro_batch
epochs=10
source_training_split_questions=6726
unique_training_questions_per_epoch=2048
epoch_sampling=distributed_shuffle_prefix_without_dataset_mutation
dataloader_rows_per_epoch=2048
ddp_padding_duplicates_per_epoch=0
dataset_subset_mutation=false
requested_global_question_batch_size=4
effective_per_device_question_batch_size=1
rollout_batches_per_epoch=512
policy_updates_per_rollout=1
optimizer_steps_per_epoch=512
scheduled_optimizer_steps=5120
training_budget=2048_unique_questions_x_8_paths_x_10_epochs
full_validation_every_epoch=true
checkpoint_selection=best_full_validation_accuracy
tiny_dataset=false
test_times=1
train_seed=${TRAIN_SEED}
external_resume=false
started_at=$(date --iso-8601=seconds)
EOF

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${DATA_ROOT}" \
  TMPDIR="${TMP_ROOT}" \
  TRACE_LOG_ROOT="${LOG_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${PYTHON}" run.py \
    --model trace_policy_qwen3_instruct \
    --dataset gsm8k_aug_nl \
    --trainer default \
    --devices 0,1,2,3 \
    --workspace_path "${CODE_ROOT}" \
    --load_ckpt_path "${stage1_checkpoint}" \
    --test_times 1 \
    --seed "${TRAIN_SEED}" \
    --disable_early_stopping \
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
    trainer.max_epochs=10 \
    trainer.max_steps=-1 \
    trainer.limit_train_batches=512 \
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
    model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048 \
    model.model_kwargs.trace_rl_config.group_size=8 \
    model.model_kwargs.trace_rl_config.rollout_micro_batch_size=2 \
    model.model_kwargs.trace_rl_config.exp_batch_size=2 \
    model.model_kwargs.trace_rl_config.use_trajectory_policy_loss=true \
    model.model_kwargs.trace_rl_config.use_answer_policy_loss=false \
    model.model_kwargs.trace_rl_config.policy_update_epochs=1 \
    model.model_kwargs.trace_rl_config.dense_outcome_weight=0.25 \
    model.model_kwargs.trace_rl_config.step_reward_weight=0.30 \
    model.model_kwargs.trace_rl_config.step_reward_discount=0.90 \
    model.model_kwargs.trace_rl_config.role_entropy_coefficient=0.001 \
    model.training_kwargs.optimizer.lr=8.0e-7 \
    model.training_kwargs.scheduler.warmup_steps=75 \
    model.training_kwargs.scheduler.num_training_steps=5120 \
    2>&1 | tee "${tee_args[@]}" "${out_dir}/train.log"

log_root="${LOG_ROOT}/trace_policy_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl"
best_checkpoint=$("${PYTHON}" - "${log_root}" "${RUN_TAG}" <<'PY'
import re
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
run_tag = sys.argv[2]
pattern = re.compile(r"__monitor([-+0-9.eE]+)\.ckpt$")
candidates = []
for path in log_root.glob(f"*_{run_tag}/checkpoints/epoch*__step*__monitor*.ckpt"):
    match = pattern.search(path.name)
    if match:
        candidates.append((float(match.group(1)), path.stat().st_mtime, path))
if candidates:
    print(max(candidates, key=lambda item: (item[0], item[1]))[2])
PY
)
last_checkpoint=$(
  find "${log_root}" -type f \
    -path "*_${RUN_TAG}/checkpoints/last.ckpt" \
    -printf '%T@ %p\n' |
    sort -n |
    tail -n 1 |
    cut -d' ' -f2-
)
if [[ -z "${best_checkpoint}" || ! -f "${best_checkpoint}" ]]; then
  echo "Stage 2 completed without a validation-best checkpoint" >&2
  exit 1
fi
if [[ -z "${last_checkpoint}" || ! -f "${last_checkpoint}" ]]; then
  echo "Stage 2 completed without a recoverable full-state checkpoint" >&2
  exit 1
fi
cat >> "${out_dir}/manifest.txt" <<EOF
best_checkpoint=${best_checkpoint}
last_checkpoint=${last_checkpoint}
external_resume=false
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
