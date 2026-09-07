#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage1-best-checkpoint>" >&2
  exit 2
fi
physical_gpus=$1
stage1_checkpoint=$2
TRAIN_SEED=${TRAIN_SEED:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_stage2_seed${TRAIN_SEED}}
RUN_ROOT=${TRACE_VB_ARTIFACT_ROOT}/training
LOG_ROOT=${TRACE_VB_ARTIFACT_ROOT}/logs
TMP_ROOT=${TRACE_VB_ARTIFACT_ROOT}/tmp
DATASET_DIR=${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL
out_dir=${RUN_ROOT}/${RUN_TAG}

trace_vb_require_four_gpus "${physical_gpus}"
[[ -f "${stage1_checkpoint}" ]] || \
  trace_vb_die "missing TRACE-VB Stage-1 checkpoint: ${stage1_checkpoint}"
[[ -z "${RESUME_CKPT_PATH:-}" ]] || \
  trace_vb_die "formal Stage 2 always starts from this pipeline's Stage-1 best checkpoint"

stage1_hparams="$(dirname "$(dirname "${stage1_checkpoint}")")/hparams.yaml"
[[ -f "${stage1_hparams}" ]] || trace_vb_die "missing Stage-1 hparams.yaml"
grep -Fq "workspace_path: ${CODE_ROOT}" "${stage1_hparams}" || \
  trace_vb_die "Stage 1 was not produced from the isolated TRACE-VB code root"
grep -qi "do_trace_rl: false" "${stage1_hparams}" || \
  trace_vb_die "Stage-1 checkpoint has the wrong phase"
grep -Fq "answer_context_mode: path_only" "${stage1_hparams}" || \
  trace_vb_die "Stage-1 checkpoint did not enforce the COMMIT-only answer bottleneck"

"${TRACE_VB_PYTHON}" - "${stage1_checkpoint}" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
    raise SystemExit("Stage 2 requires a Stage-1 TRACE-VB checkpoint")
keys = tuple(checkpoint.get("state_dict", {}))
required_fragments = ("trajectory_policy.", "plan_forecaster", "sufficiency_head")
for fragment in required_fragments:
    if not any(fragment in key for key in keys):
        raise SystemExit(f"Stage-1 checkpoint is missing {fragment}")
if any("trajectory_posterior" in key for key in keys):
    raise SystemExit("formal TRACE-VB Stage 1 must not contain a posterior module")
PY

mkdir -p "${out_dir}" "${TMP_ROOT}"
cd "${CODE_ROOT}"
TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" tools/data_contract_audit.py >/dev/null

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE-VB
full_name=Role-Structured_Latent_Reasoning_with_Semantic-to-Outcome_Value_Bridging
phase=stage2_terminal_outcome_latent_actor_critic
model_config=${TRACE_VB_MODEL_CONFIG}
project_root=${CODE_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
stage1_checkpoint=${stage1_checkpoint}
stage1_checkpoint_sha256=$(sha256sum "${stage1_checkpoint}" | awk '{print $1}')
dataset_dir=${DATASET_DIR}
latent_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT
commit_is_deterministic=true
answer_context=path_only_COMMIT
reward=terminal_exact_correctness_only
dense_outcome_reward=false
cot_semantic_RL_reward=false
length_reward=false
credit_assignment=role_conditioned_critic_plus_GAE
critic_initialization=copy_from_stage1_sufficiency_head
critic_warmup_rollout_batches=256
critic_warmup_actor_frozen=true
gamma=1.0
gae_lambda=0.95
group_size=8_iid_question_only_paths
rollout_micro_batch_size=1
optimization_micro_batch_size=1
ppo_mode=head_only_saved_pre_action_states
policy_update_epochs_per_rollout=4
trajectory_clip_epsilon=0.12
stage1_reference_KL_weight=0.02
actor_lr=8e-7
critic_lr=1e-4
source_training_questions=6726
unique_training_questions_per_epoch=2048
rollout_batches_per_epoch=512
ppo_optimizer_steps_per_epoch=2048
scheduled_optimizer_steps=20480
epochs=10
full_747_validation_every_epoch=true
checkpoint_selection=best_full_validation_accuracy
answer_decoder_trainable=false
backbone_trainable=false
transition_dynamics_trainable=false
trainable_modules=stochastic_role_actor_heads_plus_role_critic
physical_gpus=${physical_gpus}
minimum_free_memory_gate_mib=${TRACE_VB_MIN_FREE_GPU_MIB}
requested_global_question_batch_size=4
effective_per_device_question_batch_size=1
tiny_dataset=false
test_times=1
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  TRACE_LOG_ROOT="${LOG_ROOT}" \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${TRACE_VB_PYTHON}" run.py \
    --model "${TRACE_VB_MODEL_CONFIG}" \
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
    pin_memory=true \
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
    model.model_kwargs.trace_policy_config.answer_context_mode=path_only \
    model.model_kwargs.trace_policy_config.visual_record_limit=0 \
    model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048 \
    model.model_kwargs.trace_rl_config.group_size=8 \
    model.model_kwargs.trace_rl_config.rollout_micro_batch_size=1 \
    model.model_kwargs.trace_rl_config.exp_batch_size=1 \
    model.model_kwargs.trace_rl_config.use_trajectory_policy_loss=true \
    model.model_kwargs.trace_rl_config.use_answer_policy_loss=false \
    model.model_kwargs.trace_rl_config.use_terminal_exact_reward=true \
    model.model_kwargs.trace_rl_config.use_gae=true \
    model.model_kwargs.trace_rl_config.use_head_only_ppo=true \
    model.model_kwargs.trace_rl_config.policy_update_epochs=4 \
    model.model_kwargs.trace_rl_config.critic_warmup_batches=256 \
    model.model_kwargs.trace_rl_config.gamma=1.0 \
    model.model_kwargs.trace_rl_config.gae_lambda=0.95 \
    model.model_kwargs.trace_rl_config.dense_outcome_weight=0.0 \
    model.model_kwargs.trace_rl_config.step_reward_weight=0.0 \
    model.model_kwargs.trace_rl_config.trajectory_length_weight=0.0 \
    model.model_kwargs.trace_rl_config.actor_lr=8.0e-7 \
    model.model_kwargs.trace_rl_config.critic_lr=1.0e-4 \
    model.training_kwargs.scheduler.warmup_steps=300 \
    model.training_kwargs.scheduler.num_training_steps=20480 \
    2>&1 | tee "${out_dir}/train.log"

log_root="${LOG_ROOT}/${TRACE_VB_MODEL_CONFIG}/gsm8k_aug_nl-gsm8k_aug_nl"
best_checkpoint=$("${TRACE_VB_PYTHON}" - "${log_root}" "${RUN_TAG}" <<'PY'
import re
import sys
from pathlib import Path
root, tag = Path(sys.argv[1]), sys.argv[2]
pattern = re.compile(r"__monitor([-+0-9.eE]+)\.ckpt$")
candidates = []
for path in root.glob(f"*_{tag}/checkpoints/epoch*__step*__monitor*.ckpt"):
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
[[ -n "${best_checkpoint}" && -f "${best_checkpoint}" ]] || \
  trace_vb_die "Stage 2 completed without a validation-best checkpoint"
[[ -n "${last_checkpoint}" && -f "${last_checkpoint}" ]] || \
  trace_vb_die "Stage 2 completed without a recoverable last checkpoint"
cat >> "${out_dir}/manifest.txt" <<EOF
best_checkpoint=${best_checkpoint}
last_checkpoint=${last_checkpoint}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
printf '%s\n' "${last_checkpoint}" > "${out_dir}/last_checkpoint.txt"
