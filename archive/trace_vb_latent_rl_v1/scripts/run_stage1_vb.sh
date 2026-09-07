#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> [registered-stage0-checkpoint]" >&2
  exit 2
fi
physical_gpus=$1
stage0_checkpoint=${2:-${TRACE_VB_REGISTERED_STAGE0}}
TRAIN_SEED=${TRAIN_SEED:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_stage1_seed${TRAIN_SEED}}
RUN_ROOT=${TRACE_VB_ARTIFACT_ROOT}/training
LOG_ROOT=${TRACE_VB_ARTIFACT_ROOT}/logs
TMP_ROOT=${TRACE_VB_ARTIFACT_ROOT}/tmp
DATASET_DIR=${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL
out_dir=${RUN_ROOT}/${RUN_TAG}

trace_vb_require_four_gpus "${physical_gpus}"
trace_vb_require_stage0 "${stage0_checkpoint}"
trace_vb_require_cache
[[ -z "${RESUME_CKPT_PATH:-}" ]] || \
  trace_vb_die "formal Stage 1 always starts from the registered Stage-0 checkpoint"

mkdir -p "${out_dir}" "${TMP_ROOT}"
cd "${CODE_ROOT}"
TRACE_PROJECT_ROOT="${CODE_ROOT}" TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  "${TRACE_VB_PYTHON}" tools/data_contract_audit.py >/dev/null
"${TRACE_VB_PYTHON}" tools/build_trace_vb_sufficiency_cache.py \
  --audit_only \
  --output "${TRACE_VB_SUFFICIENCY_CACHE}" \
  --min_valid_row_fraction 0.50 \
  --min_valid_prefix_fraction 0.20 \
  --min_nonzero_gain_fraction 0.30 \
  --min_score_span_mean 0.05 \
  > "${out_dir}/sufficiency_cache_audit.json"

"${TRACE_VB_PYTHON}" - "${stage0_checkpoint}" "${DATASET_DIR}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
keys = list(checkpoint.get("state_dict", {}))
if not keys:
    raise SystemExit("Stage 0 checkpoint has no model state")
prohibited = ("trajectory_policy", "trajectory_posterior", "trace_vb", "trace_")
found = [key for key in keys if any(token in key for token in prohibited)]
if found:
    raise SystemExit("registered Stage 0 unexpectedly contains TRACE state")
config = checkpoint.get("hyper_parameters", {}).get("all_config")
if config is None or str(config.model.target) != "src.models.cot.LitCot":
    raise SystemExit("Stage 0 is not the registered generic CoT warm start")
data = config.data_module
if str(data.dataset_dir) != sys.argv[2]:
    raise SystemExit("Stage 0 used a different dataset directory")
if not bool(data.enforce_registered_source) or bool(data.tiny_dataset):
    raise SystemExit("Stage 0 bypassed the registered full-data contract")
expected = {
    "train_file": "gsm8k_train_processed.jsonl",
    "val_file": "gsm8k_val_processed.jsonl",
    "test_file": "gsm8k_test_processed.jsonl",
}
for field, value in expected.items():
    if str(data.get(field)) != value:
        raise SystemExit(f"invalid Stage-0 {field}: {data.get(field)!r}")
if config.args.get("load_ckpt_path") or config.args.get("resume_ckpt_path"):
    raise SystemExit("Stage 0 was not trained from the fresh base model")
PY

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE-VB
full_name=Role-Structured_Latent_Reasoning_with_Semantic-to-Outcome_Value_Bridging
phase=stage1_latent_semantic_formation
model_config=${TRACE_VB_MODEL_CONFIG}
project_root=${CODE_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
initial_checkpoint=${stage0_checkpoint}
initial_checkpoint_sha256=${TRACE_VB_REGISTERED_STAGE0_SHA256}
sufficiency_cache=${TRACE_VB_SUFFICIENCY_CACHE}
sufficiency_cache_sha256=$(sha256sum "${TRACE_VB_SUFFICIENCY_CACHE}" | awk '{print $1}')
dataset_dir=${DATASET_DIR}
source_training_questions=6726
validation_questions=747
generated_cots=false
latent_roles=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT
physical_latent_steps=8
stochastic_action_steps=7
commit_is_deterministic=true
answer_context=path_only_COMMIT
stage1_paths=one_deterministic_mean_plus_one_stochastic_prior
cot_conditioned_posterior=false
hybrid_or_anchor_branch=false
plan_supervision=five_target_forecast
solve_supervision=five_monotone_contiguous_CoT_chunks
refine_supervision=answer_ready_endpoint
sufficiency_supervision=offline_prefix_answer_sufficiency_with_leakage_mask
working_memory=within_question_causal_latent_KV
physical_gpus=${physical_gpus}
minimum_free_memory_gate_mib=${TRACE_VB_MIN_FREE_GPU_MIB}
requested_global_batch_size=4
effective_per_device_batch_size=1
optimizer_steps_per_epoch=1682
scheduled_optimizer_steps=16820
epochs=10
full_training_split_every_epoch=true
full_747_validation_every_epoch=true
early_stopping=false
tiny_dataset=false
test_times=1
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF

log_root="${LOG_ROOT}/${TRACE_VB_MODEL_CONFIG}/gsm8k_aug_nl-gsm8k_aug_nl"
current_checkpoint=

checkpoint_completed_epochs() {
  "${TRACE_VB_PYTHON}" - "$1" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint.get("epoch", -1)) + 1)
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

# Recycle the Python/DDP processes after every complete epoch.  This preserves
# the exact 10-epoch/full-validation protocol while avoiding the allocator
# growth that caused the preceding role-latent run to fail mid-epoch two.
for target_max_epochs in $(seq 1 10); do
  run_checkpoint_args=(--load_ckpt_path "${stage0_checkpoint}")
  if [[ -n "${current_checkpoint}" ]]; then
    run_checkpoint_args=(--resume_ckpt_path "${current_checkpoint}")
  fi
  printf '\n===== TRACE-VB Stage 1 target epoch %d at %s =====\n' \
    "${target_max_epochs}" "$(date --iso-8601=seconds)" >> "${out_dir}/train.log"

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
      "${run_checkpoint_args[@]}" \
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
      num_workers=0 \
      pin_memory=false \
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
      model.model_kwargs.readcot_config.use_hybrid=false \
      model.model_kwargs.readcot_config.use_anchor_loss=false \
      model.model_kwargs.trace_policy_config.answer_context_mode=path_only \
      model.model_kwargs.trace_policy_config.stage1_stochastic_paths=1 \
      model.model_kwargs.trace_policy_config.stage1_posterior_samples=0 \
      model.model_kwargs.trace_policy_config.sufficiency_cache_path="${TRACE_VB_SUFFICIENCY_CACHE}" \
      model.model_kwargs.trace_policy_config.visual_record_limit=0 \
      model.training_kwargs.optimizer.lr=1.0e-5 \
      model.training_kwargs.scheduler.warmup_steps=150 \
      model.training_kwargs.scheduler.num_training_steps=16820 \
      2>&1 | tee -a "${out_dir}/train.log"

  next_checkpoint=$(newest_last_checkpoint)
  [[ -n "${next_checkpoint}" && -f "${next_checkpoint}" ]] || \
    trace_vb_die "Stage 1 epoch ${target_max_epochs} produced no last.ckpt"
  completed_epochs=$(checkpoint_completed_epochs "${next_checkpoint}")
  [[ "${completed_epochs}" -eq "${target_max_epochs}" ]] || \
    trace_vb_die "Stage 1 expected ${target_max_epochs} completed epochs, found ${completed_epochs}"
  current_checkpoint=${next_checkpoint}
  cat >> "${out_dir}/manifest.txt" <<EOF
epoch_${target_max_epochs}_last_checkpoint=${current_checkpoint}
epoch_${target_max_epochs}_finished_at=$(date --iso-8601=seconds)
EOF
done

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
[[ -n "${best_checkpoint}" && -f "${best_checkpoint}" ]] || \
  trace_vb_die "Stage 1 completed without a validation-best checkpoint"
cat >> "${out_dir}/manifest.txt" <<EOF
best_checkpoint=${best_checkpoint}
last_checkpoint=${current_checkpoint}
finished_at=$(date --iso-8601=seconds)
EOF
printf '%s\n' "${best_checkpoint}" > "${out_dir}/best_checkpoint.txt"
printf '%s\n' "${current_checkpoint}" > "${out_dir}/last_checkpoint.txt"
