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
STAGE1_NUM_WORKERS=${STAGE1_NUM_WORKERS:-0}
STAGE1_PIN_MEMORY=${STAGE1_PIN_MEMORY:-false}
STAGE1_MAX_EPOCHS=${STAGE1_MAX_EPOCHS:-10}

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <fresh-cot-sft-checkpoint>" >&2
  exit 2
fi
physical_gpus=$1
stage0_checkpoint=$2
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "TRACE Stage 1 requires exactly four GPUs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi
if [[ ! -f "${stage0_checkpoint}" ]]; then
  echo "Missing fresh CoT-SFT checkpoint: ${stage0_checkpoint}" >&2
  exit 2
fi

cd "${CODE_ROOT}"
TRACE_DATA_ROOT="${DATA_ROOT}" "${PYTHON}" tools/data_contract_audit.py >/dev/null
"${PYTHON}" - "${stage0_checkpoint}" "${DATASET_DIR}" <<'PY'
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
expected_dir = sys.argv[2]
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
if [[ -n "${RESUME_CKPT_PATH:-}" ]]; then
  echo "Formal role-semantic Stage 1 must start from the explicit Stage-0 checkpoint; external RESUME_CKPT_PATH is forbidden" >&2
  exit 2
fi
tee_args=(-a)

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE-Role-Latent-RL
phase=stage1_role_semantic_latent_formation
initial_checkpoint=${stage0_checkpoint}
initial_checkpoint_sha256=$(sha256sum "${stage0_checkpoint}" | awk '{print $1}')
dataset_dir=${DATASET_DIR}
explicit_cots_per_question=1_original_only
generated_cots=false
latent_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_CHECK_COMMIT
role_supervision=PLAN_summary_SOLVE_monotone_CoT_chunks_CHECK_endpoint_correction_summary_residual
working_memory=within_question_recurrent_PLAN_SOLVE_CHECK_COMMIT
role_specific_policy_heads=true
commit_is_answer_readout=true
posterior_saved_activation_offload=cpu_exact_autograd
offload_cache_release_interval=10_batches_after_optimizer_step
dataloader_workers=${STAGE1_NUM_WORKERS}_in_process_prevents_forking_offload_resident_state
dataloader_pin_memory=${STAGE1_PIN_MEMORY}
process_recycling=one_full_epoch_per_process
trajectory_recurrence=stage0_preserving_identity_centered_residual
posterior_conditioning=question_plus_original_CoT_without_separate_answer_field
original_CoT_may_state_final_answer=true
role_target_construction=PLAN_teacher_summary_SOLVE_five_monotone_CoT_chunks_CHECK_endpoint_error
short_CoT_handling=empty_SOLVE_slots_masked_final_step_anchored
generated_role_annotations=false
deployment_prior_conditioning=question_only
answer_context=question_plus_COMMIT_latent_only
stage1_decoder_supervision=sampled_answer_plus_MAP_role_compact_target
compact_target_source=ordered_original_CoT_SOLVE_spans_plus_gold_answer
compact_target_budget=48_tokens_matching_deployment
deployment_output=compact_equations_plus_answer
latent_read_bottleneck=COMMIT_only_with_question_context
physical_gpus=${physical_gpus}
requested_global_batch_size=4
effective_per_device_batch_size=1
optimizer_steps_per_epoch=1682
scheduled_optimizer_steps=16820
max_epochs=${STAGE1_MAX_EPOCHS}
full_validation_every_epoch=true
early_stopping_patience=4
tiny_dataset=false
test_times=1
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF

log_root="${LOG_ROOT}/trace_policy_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl"
current_checkpoint=
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

newest_last_checkpoint() {
  find "${log_root}" -type f \
    -path "*_${RUN_TAG}/checkpoints/last.ckpt" \
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
    TRACE_PROJECT_ROOT="${CODE_ROOT}" \
    TRACE_DATA_ROOT="${DATA_ROOT}" \
    TRACE_LOG_ROOT="${LOG_ROOT}" \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${physical_gpus}" \
    "${PYTHON}" run.py \
      --model trace_policy_qwen3_instruct \
      --dataset gsm8k_aug_nl \
      --trainer default \
      --devices 0,1,2,3 \
      --workspace_path "${CODE_ROOT}" \
      "${run_checkpoint_args[@]}" \
      --test_times 1 \
      --seed "${TRAIN_SEED}" \
      --log_suffix "${RUN_TAG}" \
      data_module.dataset_dir="${DATASET_DIR}" \
      data_module.enforce_registered_source=true \
      data_module.tiny_dataset=false \
      data_module.epoch_scaling=1 \
      batch_size=4 \
      val_batch_size=1 \
      num_workers="${STAGE1_NUM_WORKERS}" \
      pin_memory="${STAGE1_PIN_MEMORY}" \
      persistent_workers=false \
      trainer.strategy=ddp_find_unused_parameters_true \
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
      model.model_kwargs.trace_policy_config.stage1_posterior_samples=3 \
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
epoch_process_${recycle_index}_finished_at=$(date --iso-8601=seconds)
EOF
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
for path in log_root.glob(f"*_{run_tag}/checkpoints/epoch*__step*__monitor*.ckpt"):
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
external_resume=false
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
