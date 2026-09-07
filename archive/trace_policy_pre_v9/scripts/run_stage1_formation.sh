#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/TRACE
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
DATASET_DIR=${DATASET_DIR:-${ROOT}/data/raw/GSM8k-Aug-NL}
RUN_ROOT=${RUN_ROOT:-${ROOT}/run_outputs/training}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/tmp}
TRAIN_SEED=${TRAIN_SEED:-0}
STAGE1_NUM_WORKERS=${STAGE1_NUM_WORKERS:-0}
STAGE1_PIN_MEMORY=${STAGE1_PIN_MEMORY:-false}
STAGE1_MAX_EPOCHS=${STAGE1_MAX_EPOCHS:-10}
STAGE1_FIRST_EPOCH_MIN_MONITOR=${STAGE1_FIRST_EPOCH_MIN_MONITOR:-0.50}
STAGE1_WORLD_SIZE=${STAGE1_WORLD_SIZE:-4}

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <stage1-physical-gpu-csv> <fresh-cot-sft-checkpoint>" >&2
  exit 2
fi
physical_gpus=$1
stage0_checkpoint=$2
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne "${STAGE1_WORLD_SIZE}" ]]; then
  echo "TRACE Stage 1 expected ${STAGE1_WORLD_SIZE} GPUs, got ${#gpu_array[@]}" >&2
  exit 2
fi
if [[ "${STAGE1_WORLD_SIZE}" -ne 4 ]]; then
  echo "Formal Stage 1 requires four GPU-only DDP ranks" >&2
  exit 2
fi
stage1_global_micro_batch=${STAGE1_WORLD_SIZE}
stage1_accumulate_grad_batches=$((4 / STAGE1_WORLD_SIZE))
if [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne "${STAGE1_WORLD_SIZE}" ]]; then
  echo "GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi
if [[ ! -f "${stage0_checkpoint}" ]]; then
  echo "Missing fresh CoT-SFT checkpoint: ${stage0_checkpoint}" >&2
  exit 2
fi

cd "${ROOT}"
"${PYTHON}" tools/data_contract_audit.py >/dev/null
"${PYTHON}" - "${stage0_checkpoint}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
keys = list(checkpoint.get("state_dict", {}))
prohibited = ("trajectory_policy", "trajectory_posterior", "trace_")
found = [key for key in keys if any(token in key for token in prohibited)]
if found:
    raise SystemExit("Stage 0 contains TRACE state and is not a fresh CoT SFT")
if not keys:
    raise SystemExit("Stage 0 checkpoint has no model state")
hparams = checkpoint.get("hyper_parameters", {})
config = hparams.get("all_config")
if config is None:
    raise SystemExit("Stage 0 checkpoint has no saved experiment config")
if str(config.model.target) != "src.models.cot.LitCot":
    raise SystemExit("Stage 0 is not the registered generic CoT warm start")
data = config.data_module
expected_dir = "/disk1/dingxukai/TRACE/data/raw/GSM8k-Aug-NL"
if str(data.dataset_dir) != expected_dir:
    raise SystemExit(f"Stage 0 used an unregistered dataset: {data.dataset_dir}")
if not bool(data.enforce_registered_source) or bool(data.tiny_dataset):
    raise SystemExit("Stage 0 bypassed the registered full-data contract")
expected_files = {
    "train_file": "gsm8k_train_processed.jsonl",
    "val_file": "gsm8k_val_processed.jsonl",
    "test_file": "gsm8k_test_processed.jsonl",
}
for field, expected in expected_files.items():
    if str(data.get(field)) != expected:
        raise SystemExit(
            f"Stage 0 has invalid {field}: {data.get(field)!r}"
        )
args = config.args
if args.get("load_ckpt_path") or args.get("resume_ckpt_path"):
    raise SystemExit("Stage 0 was not trained from the fresh base model")
PY

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_stage1_seed${TRAIN_SEED}}
out_dir="${RUN_ROOT}/${RUN_TAG}"
mkdir -p "${out_dir}" "${TMP_ROOT}"
is_resume=false
tee_args=()
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_CKPT_PATH}" ]]; then
    echo "Missing Stage-1 resume checkpoint: ${RESUME_CKPT_PATH}" >&2
    exit 2
  fi
  is_resume=true
  tee_args=(-a)
fi

if [[ "${is_resume}" == true && -f "${out_dir}/manifest.txt" ]]; then
  cat >> "${out_dir}/manifest.txt" <<EOF
resume_checkpoint=${RESUME_CKPT_PATH}
resume_checkpoint_sha256=$(sha256sum "${RESUME_CKPT_PATH}" | awk '{print $1}')
resumed_at=$(date --iso-8601=seconds)
dataloader_workers=${STAGE1_NUM_WORKERS}
dataloader_pin_memory=${STAGE1_PIN_MEMORY}
process_recycling=one_full_epoch_per_process
EOF
  printf '\n===== Stage 1 resumed at %s from %s =====\n' \
    "$(date --iso-8601=seconds)" "${RESUME_CKPT_PATH}" \
    >> "${out_dir}/train.log"
else
  cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE
phase=stage1_deployment_consistent_exchangeable_formation_v3
initial_checkpoint=${stage0_checkpoint}
initial_checkpoint_sha256=$(sha256sum "${stage0_checkpoint}" | awk '{print $1}')
dataset_dir=${DATASET_DIR}
explicit_cots_per_question=1_original_only
generated_cots=false
posterior_samples_per_question=4_iid_exchangeable
privileged_sampled_paths_per_question=0
canonical_deployment_paths_per_question=1_question_only_conditional_mean
canonical_deployment_path_is_rollout_member=false
canonical_deployment_path_is_reasoning_mode=false
canonical_deployment_path_task_supervision=true
canonical_deployment_path_structure_supervision=true
stage1_training_risk=0.50_exchangeable_posterior_expectation_plus_0.50_canonical_deployment
fixed_view_ids=false
center_or_primary_path=false
single_cot_target_set=false
compression_schedule=sampled_action_conditioned_strictly_monotone
alignment=action_and_path_conditioned_single_CoT_corridor
action_transition_identifiability=within_question_symmetric_InfoNCE
action_injection=rms_normalized_bounded_learned_gate
minimum_action_entropy=true
posterior_saved_activation_offload=disabled
stage1_memory_placement=gpu_only
trajectory_activation_checkpoint=gpu_recompute_preserve_rng
answer_decoder_activation_checkpoint=gpu_recompute
sampled_answer_decoder_micro_batch_size=1_exact_token_weighted_CE
dataloader_workers=${STAGE1_NUM_WORKERS}
dataloader_pin_memory=${STAGE1_PIN_MEMORY}
process_recycling=one_full_epoch_per_process
ddp_find_unused_parameters=true_dynamic_checkpoint_compatible
ddp_strategy=standard_dynamic_graph_gradient_bucket_views
ddp_launcher=torchrun_four_rank
ddp_gpu_visibility=one_physical_gpu_per_rank
cuda_context_placement=rank_local_only
trajectory_recurrence=stage0_preserving_identity_centered_residual
posterior_conditioning=question_plus_original_CoT_without_separate_answer_field
original_CoT_may_state_final_answer=true
dependency_supervision=runtime_rule_induced_weak_graph_not_human_gold
deployment_prior_conditioning=question_only
answer_context=complete_8_state_latent_path_only_question_KV_masked
stage1_decoder_supervision=all_4_exchangeable_paths_complete_trace_plus_canonical_deployment_complete_trace_plus_uniformly_resampled_direct_value_calibration
visible_targets_define_route_identity=false
compact_target_source=numerically_validated_answer_causal_arithmetic_trace_compiled_only_from_the_single_original_CoT
compact_target_budget=128_tokens_hard_cap_matching_deployment_average_target_remains_near_33
deployment_output=compact_equations_plus_answer
strict_answer_bottleneck=true
physical_gpus=${physical_gpus}
requested_global_batch_size=4
effective_per_device_batch_size=1
stage1_world_size=${STAGE1_WORLD_SIZE}
global_micro_batch_size=${stage1_global_micro_batch}
gradient_accumulation_steps=${stage1_accumulate_grad_batches}
effective_optimizer_batch_size=4
stage1_execution=four_rank_gpu_only_ddp
optimizer_steps_per_epoch=1682
scheduled_optimizer_steps=16820
max_epochs=${STAGE1_MAX_EPOCHS}
full_validation_every_epoch=true
early_stopping_patience=4
first_epoch_accuracy_redline=${STAGE1_FIRST_EPOCH_MIN_MONITOR}
tiny_dataset=false
test_times=1
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF
fi

log_root="${ROOT}/logs/trace_policy_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl"
current_checkpoint=${RESUME_CKPT_PATH:-}
recycle_index=0

checkpoint_progress() {
  "${PYTHON}" - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
completed_epochs = int(checkpoint.get("epoch", -1)) + 1
wait_count = 0
patience = 4
stopped_epoch = 0
for name, state in checkpoint.get("callbacks", {}).items():
    if str(name).startswith("EarlyStopping"):
        wait_count = int(state.get("wait_count", 0))
        patience = int(state.get("patience", patience))
        stopped_epoch = int(state.get("stopped_epoch", 0))
        break
early_stopped = stopped_epoch > 0 or wait_count >= patience
print(completed_epochs, int(early_stopped), wait_count, patience)
PY
}

checkpoint_monitor() {
  "${PYTHON}" - "$1" <<'PY'
import math
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
score = None
for name, state in checkpoint.get("callbacks", {}).items():
    if str(name).startswith("ModelCheckpoint"):
        score = state.get("current_score", state.get("best_model_score"))
        break
if score is None:
    raise SystemExit("Stage-1 checkpoint has no validation monitor")
value = float(score)
if not math.isfinite(value):
    raise SystemExit("Stage-1 checkpoint validation monitor is non-finite")
print(f"{value:.12f}")
PY
}

newest_last_checkpoint() {
  find "${log_root}" -type f \
    -path "*/${RUN_TAG}_epoch_process_*/checkpoints/last.ckpt" \
    -printf '%T@ %p\n' |
    sort -n |
    tail -n 1 |
    cut -d' ' -f2-
}

while true; do
  completed_epochs=0
  early_stopped=0
  wait_count=0
  patience=4
  if [[ -n "${current_checkpoint}" ]]; then
    read -r completed_epochs early_stopped wait_count patience < <(
      checkpoint_progress "${current_checkpoint}"
    )
  fi
  if (( early_stopped == 1 || completed_epochs >= STAGE1_MAX_EPOCHS )); then
    break
  fi

  target_max_epochs=$((completed_epochs + 1))
  run_checkpoint_args=(--load_ckpt_path "${stage0_checkpoint}")
  if [[ -n "${current_checkpoint}" ]]; then
    run_checkpoint_args=(--resume_ckpt_path "${current_checkpoint}")
  fi
  printf '\n===== Stage 1 epoch process %d: completed=%d target=%d at %s =====\n' \
    "${recycle_index}" "${completed_epochs}" "${target_max_epochs}" \
    "$(date --iso-8601=seconds)" >> "${out_dir}/train.log"

  env \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRACE_LOGGER_VERSION="${RUN_TAG}_epoch_process_${recycle_index}" \
    CUDA_VISIBLE_DEVICES="${physical_gpus}" \
    "${PYTHON}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${STAGE1_WORLD_SIZE}" \
      tools/isolated_gpu_ddp_entry.py \
      run.py \
      --model trace_policy_qwen3_instruct \
      --dataset gsm8k_aug_nl \
      --trainer trace_stage1_gpu4_dynamic \
      --devices 0 \
      --workspace_path /disk1/dingxukai \
      "${run_checkpoint_args[@]}" \
      --test_times 1 \
      --seed "${TRAIN_SEED}" \
      --log_suffix "${RUN_TAG}" \
      data_module.dataset_dir="${DATASET_DIR}" \
      data_module.enforce_registered_source=true \
      data_module.tiny_dataset=false \
      data_module.epoch_scaling=1 \
      batch_size=1 \
      val_batch_size=1 \
      num_workers="${STAGE1_NUM_WORKERS}" \
      pin_memory="${STAGE1_PIN_MEMORY}" \
      persistent_workers=false \
      trainer.accumulate_grad_batches="${stage1_accumulate_grad_batches}" \
      trainer.num_sanity_val_steps=0 \
      trainer.max_epochs="${target_max_epochs}" \
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
      model.model_kwargs.trace_policy_config.stage1_posterior_samples=4 \
      model.model_kwargs.trace_policy_config.stage1_deployment_risk_mix=0.50 \
      model.model_kwargs.trace_policy_config.stage1_posterior_activation_offload=false \
      model.model_kwargs.trace_policy_config.stage1_offload_cache_release_interval=0 \
      model.model_kwargs.trace_policy_config.stage1_trajectory_activation_checkpoint=true \
      model.model_kwargs.trace_policy_config.stage1_answer_activation_checkpoint=true \
      model.model_kwargs.trace_policy_config.stage1_sampled_answer_micro_batch_size=1 \
      model.model_kwargs.trace_policy_config.visual_record_limit=0 \
      model.training_kwargs.optimizer.lr=1.0e-5 \
      model.training_kwargs.scheduler.warmup_steps=150 \
      model.training_kwargs.scheduler.num_training_steps=16820 \
      2>&1 | tee "${tee_args[@]}" "${out_dir}/train.log"

  next_checkpoint=$(newest_last_checkpoint)
  if [[ -z "${next_checkpoint}" || ! -f "${next_checkpoint}" ]]; then
    echo "Stage 1 epoch process completed without last.ckpt" >&2
    exit 1
  fi
  read -r next_completed next_stopped next_wait next_patience < <(
    checkpoint_progress "${next_checkpoint}"
  )
  next_monitor=$(checkpoint_monitor "${next_checkpoint}")
  if (( next_completed != target_max_epochs )); then
    echo "Stage 1 epoch process did not reach target epoch: " \
      "expected=${target_max_epochs}, actual=${next_completed}" >&2
    exit 1
  fi
  current_checkpoint=${next_checkpoint}
  tee_args=(-a)
  cat >> "${out_dir}/manifest.txt" <<EOF
epoch_process_${recycle_index}_completed_epochs=${next_completed}
epoch_process_${recycle_index}_last_checkpoint=${current_checkpoint}
epoch_process_${recycle_index}_early_stopping_wait=${next_wait}/${next_patience}
epoch_process_${recycle_index}_monitor=${next_monitor}
epoch_process_${recycle_index}_finished_at=$(date --iso-8601=seconds)
EOF
  if (( next_completed == 1 )) && ! "${PYTHON}" - \
      "${next_monitor}" "${STAGE1_FIRST_EPOCH_MIN_MONITOR}" <<'PY'
import sys

observed = float(sys.argv[1])
minimum = float(sys.argv[2])
raise SystemExit(0 if observed >= minimum else 1)
PY
  then
    cat >> "${out_dir}/manifest.txt" <<EOF
aborted_after_epoch1=true
abort_reason=validation_below_registered_redline
observed_epoch1_monitor=${next_monitor}
required_epoch1_monitor=${STAGE1_FIRST_EPOCH_MIN_MONITOR}
aborted_at=$(date --iso-8601=seconds)
EOF
    echo "Stage 1 epoch-1 validation ${next_monitor} is below the " \
      "registered redline ${STAGE1_FIRST_EPOCH_MIN_MONITOR}; stopping." >&2
    exit 3
  fi
  recycle_index=$((recycle_index + 1))
done

best_checkpoint=$("${PYTHON}" - "${log_root}" "${RUN_TAG}" <<'PY'
import re
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
run_tag = sys.argv[2]
pattern = re.compile(r"__monitor([-+0-9.eE]+)\.ckpt$")
candidates = []
for path in log_root.glob(
    f"{run_tag}_epoch_process_*/checkpoints/"
    "epoch*__step*__monitor*.ckpt"
):
    match = pattern.search(path.name)
    if match:
        candidates.append((float(match.group(1)), path.stat().st_mtime, path))
if candidates:
    print(max(candidates, key=lambda item: (item[0], item[1]))[2])
PY
)
last_checkpoint=${current_checkpoint:-$(newest_last_checkpoint)}
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
resumed=${is_resume}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
