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
grep -Fq "workspace_path: ${TRACE_VB_STAGE1_CODE_ROOT}" "${stage1_hparams}" || \
  trace_vb_die "Stage 1 was not produced from the registered TRACE-VB-v5 code root"
grep -qi "do_trace_rl: false" "${stage1_hparams}" || \
  trace_vb_die "Stage-1 checkpoint has the wrong phase"
grep -Fq "answer_context_mode: question_and_commit" "${stage1_hparams}" || \
  trace_vb_die "Stage-1 checkpoint did not preserve the question+COMMIT bridge"

"${TRACE_VB_PYTHON}" - "${stage1_checkpoint}" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
    raise SystemExit("Stage 2 requires a Stage-1 TRACE-VB checkpoint")
if checkpoint.get("trace_vb_schema_version") != "trace_vb_v5":
    raise SystemExit("Stage 2 requires a question+COMMIT TRACE-VB-v5 checkpoint")
keys = tuple(checkpoint.get("state_dict", {}))
required_fragments = (
    "trajectory_policy.",
    "trajectory_posterior.",
    "posterior_context_norm.",
    "plan_forecaster",
    "solve_text_decoder.",
    "sufficiency_head",
)
for fragment in required_fragments:
    if not any(fragment in key for key in keys):
        raise SystemExit(f"Stage-1 checkpoint is missing {fragment}")
PY

mkdir -p "${out_dir}" "${TMP_ROOT}"
cd "${CODE_ROOT}"
TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" tools/data_contract_audit.py >/dev/null
PYTHONDONTWRITEBYTECODE=1 "${TRACE_VB_PYTHON}" \
  tools/trace_vb_v6_stage2_contract_audit.py >/dev/null

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE-VB-v6
full_name=Role-Structured_Latent_Reasoning_with_Evidence-Gated_Group_Refinement
phase=stage2_calibrated_evidence_gated_group_latent_rl
model_config=${TRACE_VB_MODEL_CONFIG}
project_root=${CODE_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
stage1_checkpoint=${stage1_checkpoint}
stage1_checkpoint_sha256=$(sha256sum "${stage1_checkpoint}" | awk '{print $1}')
dataset_dir=${DATASET_DIR}
latent_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT
commit_is_deterministic=true
answer_context=question_plus_COMMIT_latent_only
private_latent_answer_access=false
reward=terminal_exact_correctness_only
dense_outcome_reward=false
cot_semantic_RL_reward=false
semantic_anchor_auxiliary_loss=true
semantic_anchor_path=deterministic_mean_only
semantic_anchor_decoder=frozen_stage1_SOLVE_text_decoder
semantic_anchor_initial_weight=0.02
semantic_anchor_minimum_weight=0.01
semantic_anchor_decay_batches=4096
semantic_anchor_active_during_score_calibration=false
length_reward=false
semantic_anchor_changes_terminal_reward=false
credit_assignment=question_local_observed_preference_without_critic
mixed_group_signal=group_relative_exact_correctness
all_correct_group_signal=zero
all_wrong_group_signal=calibrated_frozen_gold_answer_likelihood_rank_or_zero
score_proxy_calibration=mixed_group_correct_vs_wrong_pairwise_auc
score_proxy_calibration_rollout_batches=128
score_proxy_minimum_pairs=64
score_proxy_minimum_auc=0.60
score_proxy_minimum_within_question_gap=0.002
score_proxy_fail_closed=true
step_level_RL_signal=shared_question_local_path_advantage_on_seven_stochastic_roles
independent_step_reward=false
critic=false
group_size=8_iid_question_conditioned_latent_paths
rollout_micro_batch_size=1
optimization_micro_batch_size=1
ppo_mode=head_only_evidence_gated_group_ppo_plus_one_mean_path_semantic_anchor
policy_update_epochs_per_rollout=2_maximum
trajectory_clip_epsilon=0.12
stage1_reference_KL_weight=0.02
stage1_reference_KL_early_stop=0.01
actor_lr=8e-7
source_training_questions=6726
unique_training_questions_per_epoch=2048
rollout_batches_per_epoch=512
calibration_batches_without_optimizer_update=128
maximum_optimizer_steps_per_epoch=1024
scheduled_optimizer_steps=10240
epochs=10
full_747_validation_every_epoch=true
checkpoint_selection=best_full_validation_accuracy
answer_decoder_trainable=false
backbone_trainable=false
transition_dynamics_trainable=false
trainable_modules=stochastic_role_actor_heads_only
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
    model.model_kwargs.trace_policy_config.answer_context_mode=question_and_commit \
    model.model_kwargs.trace_policy_config.visual_record_limit=0 \
    model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048 \
    model.model_kwargs.trace_rl_config.group_size=8 \
    model.model_kwargs.trace_rl_config.rollout_micro_batch_size=1 \
    model.model_kwargs.trace_rl_config.exp_batch_size=1 \
    model.model_kwargs.trace_rl_config.use_trajectory_policy_loss=true \
    model.model_kwargs.trace_rl_config.use_answer_policy_loss=false \
    model.model_kwargs.trace_rl_config.use_terminal_exact_reward=true \
    model.model_kwargs.trace_rl_config.use_evidence_gated_group_rl=true \
    model.model_kwargs.trace_rl_config.use_gae=false \
    model.model_kwargs.trace_rl_config.use_head_only_ppo=true \
    model.model_kwargs.trace_rl_config.policy_update_epochs=2 \
    model.model_kwargs.trace_rl_config.score_calibration_batches=128 \
    model.model_kwargs.trace_rl_config.use_gold_likelihood_fallback=true \
    model.model_kwargs.trace_rl_config.score_proxy_minimum_pairs=64 \
    model.model_kwargs.trace_rl_config.score_proxy_minimum_auc=0.60 \
    model.model_kwargs.trace_rl_config.minimum_gold_score_gap=0.002 \
    model.model_kwargs.trace_rl_config.use_semantic_anchor=true \
    model.model_kwargs.trace_rl_config.semantic_anchor_initial_weight=0.02 \
    model.model_kwargs.trace_rl_config.semantic_anchor_minimum_weight=0.01 \
    model.model_kwargs.trace_rl_config.semantic_anchor_decay_batches=4096 \
    model.model_kwargs.trace_rl_config.semantic_anchor_plan_weight=0.25 \
    model.model_kwargs.trace_rl_config.semantic_anchor_solve_text_weight=0.50 \
    model.model_kwargs.trace_rl_config.semantic_anchor_refine_weight=0.25 \
    model.model_kwargs.trace_rl_config.dense_outcome_weight=0.0 \
    model.model_kwargs.trace_rl_config.step_reward_weight=0.0 \
    model.model_kwargs.trace_rl_config.trajectory_length_weight=0.0 \
    model.model_kwargs.trace_rl_config.actor_lr=8.0e-7 \
    model.model_kwargs.trace_rl_config.stage1_policy_target_kl=0.01 \
    model.training_kwargs.scheduler.warmup_steps=150 \
    model.training_kwargs.scheduler.num_training_steps=10240 \
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
