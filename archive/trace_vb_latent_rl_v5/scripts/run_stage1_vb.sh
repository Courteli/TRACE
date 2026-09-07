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
STAGE1_RESUME_CKPT=${STAGE1_RESUME_CKPT:-}
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
  trace_vb_die "use STAGE1_RESUME_CKPT for guarded same-run recovery"
if [[ -n "${STAGE1_RESUME_CKPT}" && ! -f "${STAGE1_RESUME_CKPT}" ]]; then
  trace_vb_die "missing Stage-1 recovery checkpoint: ${STAGE1_RESUME_CKPT}"
fi

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

if [[ -n "${STAGE1_RESUME_CKPT}" ]]; then
  "${TRACE_VB_PYTHON}" - "${STAGE1_RESUME_CKPT}" "${DATASET_DIR}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if checkpoint.get("trace_policy_training_stage") != 1:
    raise SystemExit("recovery checkpoint is not TRACE-VB Stage 1")
if checkpoint.get("trace_vb_schema_version") != "trace_vb_v5":
    raise SystemExit("recovery checkpoint has the wrong TRACE-VB schema")
state = checkpoint.get("state_dict", {})
required_prefixes = (
    "trajectory_policy.",
    "trajectory_posterior.",
    "posterior_context_norm.",
    "solve_text_decoder.",
    "sufficiency_head.",
)
for prefix in required_prefixes:
    if not any(name.startswith(prefix) for name in state):
        raise SystemExit(f"recovery checkpoint is missing {prefix}")
config = checkpoint.get("hyper_parameters", {}).get("all_config")
if config is None or str(config.model.target) != "src.models.trace_vb.LitTRACEVB":
    raise SystemExit("recovery checkpoint does not instantiate LitTRACEVB")
data = config.data_module
if str(data.dataset_dir) != sys.argv[2]:
    raise SystemExit("recovery checkpoint used a different dataset directory")
if not bool(data.enforce_registered_source) or bool(data.tiny_dataset):
    raise SystemExit("recovery checkpoint bypassed the full-data contract")
PY
fi

cat > "${out_dir}/manifest.txt" <<EOF
model=TRACE-VB-v5
full_name=Role-Structured_Latent_Reasoning_with_Semantic-to-Outcome_Value_Bridging
phase=stage1_latent_semantic_formation
model_config=${TRACE_VB_MODEL_CONFIG}
project_root=${CODE_ROOT}
artifact_root=${TRACE_VB_ARTIFACT_ROOT}
initial_checkpoint=${stage0_checkpoint}
initial_checkpoint_sha256=${TRACE_VB_REGISTERED_STAGE0_SHA256}
initial_resume_checkpoint=${STAGE1_RESUME_CKPT:-none}
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
answer_context=question_plus_COMMIT_latent_only
private_latent_answer_access=false
stage1_paths=one_training_only_CoT_teacher_plus_one_question_only_deployment_mean
cot_conditioned_posterior=one_training_only_sample_local_teacher_path
posterior_to_prior_alignment=masked_role_KL_weight_0.05
hybrid_or_anchor_branch=false
deployment_supervision=compact_reasoning_plus_answer_from_same_sample_gold_CoT
deployment_generation=compact_reasoning_plus_answer_48_token_budget
compact_target_rendering=equation_only_max_32_chars_per_observed_role_endpoint
compact_budget_policy=protected_final_observed_equation_then_earlier_equations_in_causal_order
protected_map_answer_suffix=same_compact_forward_weight_0.50
answer_activation_memory=GPU_recomputation_exact_objective_no_CPU_offload
teacher_answer_supervision=answer_only_weight_0.35
plan_supervision=five_target_forecast
solve_supervision=shared_autoregressive_decoder_from_five_SOLVE_residuals
solve_text_partition=complete_sample_local_gold_CoT_tokens_balanced_contiguous_SOLVE1_5
solve_text_coverage=every_gold_CoT_token_exactly_once
solve_text_truncation=forbidden_fail_closed
solve_decoder_sample_conditioning=one_action_induced_SOLVE_residual_only
solve_decoder_question_access=false
solve_decoder_answer_field_access=false
solve_decoder_inference_presence=false
refine_supervision=answer_ready_endpoint
commit_bridge=deterministic_REFINE_residual_plus_final_text_CoT_state_alignment
commit_supervision=final_text_CoT_teacher_state_weight_0.10
sufficiency_supervision=offline_prefix_answer_sufficiency_with_leakage_mask
step_level_supervision=PLAN_forecast_plus_SOLVE1_5_text_decode_plus_REFINE_endpoint_plus_COMMIT_alignment
working_memory=within_question_causal_latent_KV
physical_gpus=${physical_gpus}
minimum_free_memory_gate_mib=${TRACE_VB_MIN_FREE_GPU_MIB}
requested_global_batch_size=4
effective_per_device_batch_size=1
ddp_find_unused_parameters=false
ddp_gradient_as_bucket_view=true
ddp_static_graph=true
ddp_broadcast_buffers=false
ddp_bucket_cap_mib=4
saved_activation_offload=forbidden_GPU_resident_activations_only
solve_text_memory=GPU_token_chunk_projection_with_backward_recomputation
recovery_checkpoint_interval=200_optimizer_steps
host_memory_guard_interval=10_optimizer_steps
maximum_rank_rss_gib=20
minimum_host_available_gib=192
optimizer_steps_per_epoch=1682
scheduled_optimizer_steps=16820
epochs=10
full_training_split_every_epoch=true
full_747_validation_every_epoch=true
epoch1_behavior_gate=accuracy_plus_validity_plus_diversity_plus_mode_fraction
early_stopping=false
tiny_dataset=false
test_times=1
train_seed=${TRAIN_SEED}
started_at=$(date --iso-8601=seconds)
EOF

log_root="${LOG_ROOT}/${TRACE_VB_MODEL_CONFIG}/gsm8k_aug_nl-gsm8k_aug_nl"
mkdir -p "${log_root}"
current_checkpoint=${STAGE1_RESUME_CKPT}

checkpoint_completed_epochs() {
  "${TRACE_VB_PYTHON}" - "$1" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
fit_loop = checkpoint.get("loops", {}).get("fit_loop", {})
epoch_progress = fit_loop.get("epoch_progress", {}).get("total", {})
processed = epoch_progress.get("processed")
if processed is None:
    raise SystemExit("checkpoint is missing Lightning epoch progress")
processed = int(processed)
if processed < 0:
    raise SystemExit(f"invalid processed epoch count: {processed}")
print(processed)
PY
}

checkpoint_current_monitor() {
  "${TRACE_VB_PYTHON}" - "$1" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
scores = []
for state in checkpoint.get("callbacks", {}).values():
    if state.get("monitor") != "monitor":
        continue
    score = state.get("current_score")
    if score is not None:
        scores.append(float(score))
if len(scores) != 1:
    raise SystemExit(
        f"expected exactly one current full-validation monitor, found {len(scores)}"
    )
print(scores[0])
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
newest_host_guard_checkpoint() {
  find "${log_root}" -type f \
    -path "*_${RUN_TAG}/checkpoints/host-memory-guard-*.ckpt" \
    -printf '%T@ %p\n' |
    sort -n |
    tail -n 1 |
    cut -d' ' -f2-
}

wait_for_host_memory_headroom() {
  local resume_gib=${TRACE_VB_HOST_RESUME_AVAILABLE_GIB:-224}
  local available_kib
  local required_kib=$((resume_gib * 1024 * 1024))
  while true; do
    available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if (( available_kib >= required_kib )); then
      return
    fi
    printf 'TRACE-VB host guard waiting: %.1f GiB available, %d GiB required\n' \
      "$(awk -v kib="${available_kib}" 'BEGIN {print kib / 1048576.0}')" \
      "${resume_gib}" >> "${out_dir}/train.log"
    sleep 30
  done
}

maximum_guard_restarts=${TRACE_VB_MAX_HOST_GUARD_RESTARTS:-20}
memory_guard_restarts=0
initial_completed_epochs=0
initial_resume_monitor=
if [[ -n "${STAGE1_RESUME_CKPT}" ]]; then
  initial_completed_epochs=$(checkpoint_completed_epochs "${STAGE1_RESUME_CKPT}")
  if (( initial_completed_epochs < 0 || initial_completed_epochs > 10 )); then
    trace_vb_die \
      "Stage 1 resume checkpoint has invalid completed epoch count: ${initial_completed_epochs}"
  fi
  if (( initial_completed_epochs > 0 )); then
    initial_resume_monitor=$(checkpoint_current_monitor "${STAGE1_RESUME_CKPT}")
  fi
fi
cat >> "${out_dir}/manifest.txt" <<EOF
initial_completed_epochs=${initial_completed_epochs}
initial_resume_full_validation_accuracy=${initial_resume_monitor:-none}
EOF



# Recycle the Python/DDP processes after every complete epoch.  This preserves
# the exact 10-epoch/full-validation protocol while avoiding the allocator
# growth that caused the preceding role-latent run to fail mid-epoch two.
for target_max_epochs in $(seq 1 10); do
  if (( target_max_epochs <= initial_completed_epochs )); then
    printf 'Skipping already completed TRACE-VB Stage 1 epoch %d\n' \
      "${target_max_epochs}" >> "${out_dir}/train.log"
    continue
  fi
  run_checkpoint_args=(--load_ckpt_path "${stage0_checkpoint}")
  if [[ -n "${current_checkpoint}" ]]; then
    run_checkpoint_args=(--resume_ckpt_path "${current_checkpoint}")
  fi
  printf '\n===== TRACE-VB Stage 1 target epoch %d at %s =====\n' \
    "${target_max_epochs}" "$(date --iso-8601=seconds)" >> "${out_dir}/train.log"

  while true; do
    guard_before=$(newest_host_guard_checkpoint)
    set +e
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
      --trainer trace_vb_stage1_v2 \
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
      model.model_kwargs.readcot_config.use_anchor_gate=false \
      model.model_kwargs.readcot_config.anchor_text_mode=compact_equation \
      model.model_kwargs.readcot_config.compact_anchor_max_chars=32 \
      model.model_kwargs.trace_policy_config.answer_context_mode=question_and_commit \
      model.model_kwargs.trace_policy_config.deployment_compact_reasoning=true \
      model.model_kwargs.trace_policy_config.commit_residual_bridge=true \
      model.model_kwargs.trace_policy_config.stage1_stochastic_paths=1 \
      model.model_kwargs.trace_policy_config.stage1_posterior_samples=1 \
      model.model_kwargs.trace_policy_config.stage1_posterior_kl_weight=0.05 \
      model.model_kwargs.trace_policy_config.stage1_sampled_answer_weight=0.35 \
      model.model_kwargs.trace_policy_config.stage1_map_compact_weight=1.0 \
      model.model_kwargs.trace_policy_config.stage1_map_answer_suffix_weight=0.50 \
      model.model_kwargs.trace_policy_config.compact_target_max_new_tokens=48 \
      model.model_kwargs.trace_policy_config.stage1_answer_activation_checkpoint=true \
      model.model_kwargs.trace_policy_config.stage1_solve_weight=0.02 \
      model.model_kwargs.trace_policy_config.stage1_solve_text_weight=0.05 \
      model.model_kwargs.trace_policy_config.stage1_commit_weight=0.10 \
      model.model_kwargs.trace_policy_config.solve_text_decoder_hidden_size=512 \
      model.model_kwargs.trace_policy_config.solve_text_decoder_max_tokens=96 \
      model.model_kwargs.trace_policy_config.solve_text_decoder_fail_on_truncation=true \
      model.model_kwargs.trace_policy_config.solve_text_ce_chunk_size=8 \
      model.model_kwargs.trace_policy_config.stage1_posterior_activation_offload=false \
      model.model_kwargs.trace_policy_config.stage1_recovery_checkpoint_interval=200 \
      model.model_kwargs.trace_policy_config.stage1_offload_cache_release_interval=0 \
      model.model_kwargs.trace_policy_config.stage1_host_memory_guard_interval=10 \
      model.model_kwargs.trace_policy_config.stage1_maximum_rank_rss_gib=20.0 \
      model.model_kwargs.trace_policy_config.stage1_minimum_host_available_gib=192.0 \
      model.model_kwargs.trace_policy_config.sufficiency_cache_path="${TRACE_VB_SUFFICIENCY_CACHE}" \
      model.model_kwargs.trace_policy_config.visual_record_limit=0 \
      model.model_kwargs.answer_generation_config.max_new_tokens=48 \
      model.training_kwargs.optimizer.lr=1.0e-5 \
      model.training_kwargs.scheduler.warmup_steps=150 \
      model.training_kwargs.scheduler.num_training_steps=16820 \
      2>&1 | tee -a "${out_dir}/train.log"
    run_status=${PIPESTATUS[0]}
    set -e
    if [[ "${run_status}" -eq 0 ]]; then
      break
    fi

    guard_after=$(newest_host_guard_checkpoint)
    if [[
      -n "${guard_after}"
      && "${guard_after}" != "${guard_before}"
      && -f "${guard_after}"
    ]]; then
      memory_guard_restarts=$((memory_guard_restarts + 1))
      if (( memory_guard_restarts > maximum_guard_restarts )); then
        trace_vb_die           "host-memory guard exceeded ${maximum_guard_restarts} restarts"
      fi
      current_checkpoint=${guard_after}
      run_checkpoint_args=(--resume_ckpt_path "${current_checkpoint}")
      cat >> "${out_dir}/manifest.txt" <<EOF
memory_guard_restart_${memory_guard_restarts}_checkpoint=${current_checkpoint}
memory_guard_restart_${memory_guard_restarts}_at=$(date --iso-8601=seconds)
EOF
      wait_for_host_memory_headroom
      continue
    fi
    exit "${run_status}"
  done


  next_checkpoint=$(newest_last_checkpoint)
  [[ -n "${next_checkpoint}" && -f "${next_checkpoint}" ]] || \
    trace_vb_die "Stage 1 epoch ${target_max_epochs} produced no last.ckpt"
  completed_epochs=$(checkpoint_completed_epochs "${next_checkpoint}")
  [[ "${completed_epochs}" -eq "${target_max_epochs}" ]] || \
    trace_vb_die "Stage 1 expected ${target_max_epochs} completed epochs, found ${completed_epochs}"
  current_checkpoint=${next_checkpoint}
  current_monitor=$(checkpoint_current_monitor "${current_checkpoint}")
  cat >> "${out_dir}/manifest.txt" <<EOF
epoch_${target_max_epochs}_last_checkpoint=${current_checkpoint}
epoch_${target_max_epochs}_full_validation_accuracy=${current_monitor}
epoch_${target_max_epochs}_finished_at=$(date --iso-8601=seconds)
EOF
  if [[ "${target_max_epochs}" -eq 1 ]]; then
    validation_summary="$(dirname "$(dirname "${current_checkpoint}")")/validation_epoch_000.json"
    "${TRACE_VB_PYTHON}" - "${validation_summary}" \
      "${out_dir}/epoch1_behavior_gate.json" \
      "${TRACE_VB_EPOCH1_MIN_ACCURACY}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
minimum_accuracy = float(sys.argv[3])
if not source.is_file():
    raise SystemExit(f"missing full-validation behavior summary: {source}")
summary = json.loads(source.read_text(encoding="utf-8"))
report = dict(summary)
report.update({
    "status": "PASS",
    "thresholds": {
        "expected_questions": 747,
        "expected_world_size": 4,
        "minimum_accuracy": minimum_accuracy,
        "minimum_valid_answer_fraction": 0.98,
        "minimum_unique_prediction_ratio": 0.20,
        "maximum_top1_mode_fraction": 0.20,
        "minimum_nonempty_output_fraction": 0.98,
    },
})
failures = []
if int(summary.get("unique_questions", -1)) != 747:
    failures.append(f"unique_questions={summary.get('unique_questions')}!=747")
if int(summary.get("world_size", -1)) != 4:
    failures.append(f"world_size={summary.get('world_size')}!=4")
if float(summary.get("accuracy", 0.0)) < minimum_accuracy:
    failures.append(
        f"accuracy={float(summary.get('accuracy', 0.0)):.4f}"
        f"<{minimum_accuracy:.4f}"
    )
if float(summary.get("valid_answer_fraction", 0.0)) < 0.98:
    failures.append(
        "valid_answer_fraction="
        f"{float(summary.get('valid_answer_fraction', 0.0)):.4f}<0.98"
    )
if float(summary.get("unique_prediction_ratio", 0.0)) < 0.20:
    failures.append(
        "unique_prediction_ratio="
        f"{float(summary.get('unique_prediction_ratio', 0.0)):.4f}<0.20"
    )
if float(summary.get("top1_mode_fraction", 1.0)) > 0.20:
    failures.append(
        "top1_mode_fraction="
        f"{float(summary.get('top1_mode_fraction', 1.0)):.4f}>0.20"
    )
if float(summary.get("nonempty_output_fraction", 0.0)) < 0.98:
    failures.append(
        "nonempty_output_fraction="
        f"{float(summary.get('nonempty_output_fraction', 0.0)):.4f}<0.98"
    )
if failures:
    report["status"] = "FAIL"
    report["failures"] = failures
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if failures:
    raise SystemExit("TRACE-VB-v5 Epoch-1 behavior gate failed: " + "; ".join(failures))
PY
  fi
done

best_checkpoint=$("${TRACE_VB_PYTHON}" - \
  "${log_root}" "${RUN_TAG}" \
  "${STAGE1_RESUME_CKPT}" "${initial_resume_monitor}" <<'PY'
import re
import sys
from pathlib import Path
root, tag = Path(sys.argv[1]), sys.argv[2]
initial_checkpoint = Path(sys.argv[3]) if sys.argv[3] else None
initial_score = sys.argv[4]
pattern = re.compile(r"__monitor([-+0-9.eE]+)\.ckpt$")
candidates = []
for path in root.glob(f"*_{tag}/checkpoints/epoch*__step*__monitor*.ckpt"):
    match = pattern.search(path.name)
    if match:
        candidates.append((float(match.group(1)), path.stat().st_mtime, path))
if initial_checkpoint is not None:
    if not initial_checkpoint.is_file() or not initial_score:
        raise SystemExit("invalid initial Stage-1 best-checkpoint candidate")
    candidates.append(
        (float(initial_score), initial_checkpoint.stat().st_mtime, initial_checkpoint)
    )
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
