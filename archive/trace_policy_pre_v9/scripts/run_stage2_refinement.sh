#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/TRACE
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
DATASET_DIR=${DATASET_DIR:-${ROOT}/data/raw/GSM8k-Aug-NL}
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/training}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/tmp}
LOG_ROOT=${LOG_ROOT:-${ROOT}/logs}
WORKSPACE_PATH=${WORKSPACE_PATH:-/disk1/dingxukai}
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

cd "${ROOT}"
"${PYTHON}" tools/data_contract_audit.py >/dev/null
"${PYTHON}" - "${stage1_checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
    raise SystemExit("Stage 2 must initialize from a Stage-1 TRACE checkpoint")
if checkpoint.get("trace_policy_version") != "TRACE-Policy-v3":
    raise SystemExit("Stage 2 requires a TRACE-Policy-v3 Stage-1 checkpoint")
keys = checkpoint.get("state_dict", {})
required = (
    "trajectory_policy.",
    "trajectory_posterior.",
    "transition_action_decoder.",
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
resume_args=()
load_args=(--load_ckpt_path "${stage1_checkpoint}")
is_resume=false
tee_args=()
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_CKPT_PATH}" ]]; then
    echo "Missing Stage-2 resume checkpoint: ${RESUME_CKPT_PATH}" >&2
    exit 2
  fi
  resume_args=(--resume_ckpt_path "${RESUME_CKPT_PATH}")
  load_args=()
  is_resume=true
  tee_args=(-a)
fi

if [[ "${is_resume}" == true && -f "${out_dir}/manifest.txt" ]]; then
  cat >> "${out_dir}/manifest.txt" <<EOF
resume_checkpoint=${RESUME_CKPT_PATH}
resume_checkpoint_sha256=$(sha256sum "${RESUME_CKPT_PATH}" | awk '{print $1}')
resumed_at=$(date --iso-8601=seconds)
EOF
  printf '\n===== Stage 2 resumed at %s from %s =====\n' \
    "$(date --iso-8601=seconds)" "${RESUME_CKPT_PATH}" \
    >> "${out_dir}/train.log"
else
  cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE
phase=stage2_causal_outcome_trajectory_refinement_v3
stage1_checkpoint=${stage1_checkpoint}
dataset_dir=${DATASET_DIR}
cot_visible_in_stage2=false
group_size=4_iid_paths_epoch0_then_8_iid_paths_epochs1_to_9
center_or_primary_path=false
path_labels=greedy_complete_path_only_answer
answer_token_labels=sampled_answer
answer_context=complete_8_state_latent_path_only_question_KV_masked
deployment_output=compact_equations_plus_answer_with_48_token_budget
positive_positive_relation=nearest_correct_peer_local_radius
positive_negative_relation=nearest_exact_wrong_or_score_ranked_low_path
single_positive_groups=counterfactual_credit_without_positive_peer_radius
wrong_wrong_dispersion=false
counterfactual_directions=correct_to_wrong_and_wrong_to_correct
transition_credit=4_rotating_positions_per_update_covering_all_8_with_causal_suffix_recomputation
transition_credit_scorer=frozen_stage1_path_only_direct_answer_scorer
transition_credit_comparison=question_fixed_path_intervened
transition_advantage_normalization=within_question_per_transition
counterfactual_coverage=logged_question_and_path_slot_fractions
terminal_path_credit=exact_answer_plus_noise_floored_within_question_frozen_gold_likelihood
homogeneous_outcome_groups=score_ranked_pair_only_above_registered_numerical_floor
stage1_prior=target_KL_for_latent_policy_and_answer_behavior
answer_policy_relative_weight=1.0
stage1_answer_kl_weight=0.05_above_target_0.02
dense_outcome_weight=0.25
execution_acceleration=saved_rollout_policy_states_plus_shared_path_cache_and_batched_microsteps
execution_acceleration_changes_objective=false
rollout_path_cache_reuse=greedy_sampled_current_reference_and_gold_scores_share_one_sampled_path
hard_pair_base_scoring=reused_frozen_gold_scores_for_all_sampled_paths
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
matched_budget_reference=old_answer_only_stage2
full_validation_every_epoch=true
checkpoint_selection=best_full_validation_accuracy
tiny_dataset=false
test_times=1
train_seed=${TRAIN_SEED}
resume_checkpoint=${RESUME_CKPT_PATH:-}
started_at=$(date --iso-8601=seconds)
EOF
fi

env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
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
    --workspace_path "${WORKSPACE_PATH}" \
    "${load_args[@]}" \
    "${resume_args[@]}" \
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
    model.model_kwargs.trace_rl_config.minimum_group_size=4 \
    model.model_kwargs.trace_rl_config.maximum_group_size=8 \
    model.model_kwargs.trace_rl_config.small_group_warmup_epochs=1 \
    model.model_kwargs.trace_rl_config.rollout_micro_batch_size=2 \
    model.model_kwargs.trace_rl_config.exp_batch_size=2 \
    model.model_kwargs.trace_rl_config.use_trajectory_policy_loss=true \
    model.model_kwargs.trace_rl_config.use_answer_policy_loss=true \
    model.model_kwargs.trace_rl_config.policy_update_epochs=1 \
    model.model_kwargs.trace_rl_config.stage1_answer_kl_weight=0.05 \
    model.model_kwargs.trace_rl_config.stage1_answer_kl_target=0.02 \
    model.model_kwargs.trace_rl_config.stage1_policy_kl_weight=0.05 \
    model.model_kwargs.trace_rl_config.stage1_policy_kl_target=0.02 \
    model.model_kwargs.trace_rl_config.answer_policy_weight=1.0 \
    model.model_kwargs.trace_rl_config.dense_outcome_weight=0.25 \
    model.model_kwargs.trace_rl_config.minimum_gold_score_std=0.001 \
    model.model_kwargs.trace_rl_config.minimum_gold_score_gap=0.002 \
    model.model_kwargs.trace_rl_config.counterfactual_steps_per_pair=4 \
    model.training_kwargs.optimizer.lr=8.0e-7 \
    model.training_kwargs.scheduler.warmup_steps=75 \
    model.training_kwargs.scheduler.num_training_steps=5120 \
    2>&1 | tee "${tee_args[@]}" "${out_dir}/train.log"

log_root="${LOG_ROOT}/trace_policy_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl"
best_checkpoint=$("${PYTHON}" - "${log_root}" "${RUN_TAG}" "${RESUME_CKPT_PATH:-}" <<'PY'
import re
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
run_tag = sys.argv[2]
resume_checkpoint = Path(sys.argv[3]) if sys.argv[3] else None
pattern = re.compile(r"__monitor([-+0-9.eE]+)\.ckpt$")
candidates = []
for path in log_root.glob(f"*_{run_tag}/checkpoints/epoch*__step*__monitor*.ckpt"):
    match = pattern.search(path.name)
    if match:
        candidates.append((float(match.group(1)), path.stat().st_mtime, path))
if resume_checkpoint is not None and resume_checkpoint.is_file():
    match = pattern.search(resume_checkpoint.name)
    if match:
        candidates.append(
            (
                float(match.group(1)),
                resume_checkpoint.stat().st_mtime,
                resume_checkpoint,
            )
        )
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
  last_checkpoint="${RESUME_CKPT_PATH:-}"
fi
if [[ -z "${last_checkpoint}" || ! -f "${last_checkpoint}" ]]; then
  echo "Stage 2 completed without a recoverable full-state checkpoint" >&2
  exit 1
fi
cat >> "${out_dir}/manifest.txt" <<EOF
best_checkpoint=${best_checkpoint}
last_checkpoint=${last_checkpoint}
resumed=${is_resume}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
