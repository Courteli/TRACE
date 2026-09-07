#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"
EVIDENCE_ROOT=${TRACE_VB_ARTIFACT_ROOT}/evidence
TMP_ROOT=${TRACE_VB_ARTIFACT_ROOT}/tmp

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <physical-gpu> <stage1-best-checkpoint> <stage2-best-checkpoint>" >&2
  exit 2
fi
physical_gpu=$1
stage1_checkpoint=$2
stage2_checkpoint=$3
[[ "${physical_gpu}" =~ ^[0-9]+$ ]] || trace_vb_die "invalid evidence GPU: ${physical_gpu}"

validate_checkpoint() {
  local phase=$1 checkpoint=$2 expected_rl=$3
  [[ -f "${checkpoint}" ]] || trace_vb_die "missing ${phase} checkpoint: ${checkpoint}"
  local hparams
  hparams="$(dirname "$(dirname "${checkpoint}")")/hparams.yaml"
  [[ -f "${hparams}" ]] || trace_vb_die "missing ${phase} hparams: ${hparams}"
  grep -q "src.models.trace_vb.LitTRACEVB" "${hparams}" || \
    trace_vb_die "${phase} is not a TRACE-VB policy checkpoint"
  grep -Fq "workspace_path: ${CODE_ROOT}" "${hparams}" || \
    trace_vb_die "${phase} was not produced by the isolated code root"
  grep -qi "do_trace_rl: ${expected_rl}" "${hparams}" || \
    trace_vb_die "${phase} has the wrong training phase"
  grep -Fq "answer_context_mode: question_and_commit" "${hparams}" || \
    trace_vb_die "${phase} does not preserve the question+COMMIT bridge"
  "${TRACE_VB_PYTHON}" - "${checkpoint}" "${expected_rl}" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected_stage = 2 if sys.argv[2].lower() == "true" else 1
if int(checkpoint.get("trace_policy_training_stage", -1)) != expected_stage:
    raise SystemExit(f"checkpoint phase marker is not Stage {expected_stage}")
if checkpoint.get("trace_vb_schema_version") != "trace_vb_v5":
    raise SystemExit("evidence requires a question+COMMIT TRACE-VB-v5 checkpoint")
keys = tuple(checkpoint.get("state_dict", {}))
for fragment in (
    "trajectory_policy.",
    "trajectory_posterior.",
    "posterior_context_norm.",
    "plan_forecaster",
    "solve_text_decoder.",
    "sufficiency_head",
):
    if not any(fragment in key for key in keys):
        raise SystemExit(f"checkpoint is missing {fragment}")
if expected_stage == 2:
    if not any("value_critic" in key for key in keys):
        raise SystemExit("Stage-2 checkpoint is missing the outcome critic")
    if "trace_stage1_policy_reference" not in checkpoint:
        raise SystemExit("Stage-2 checkpoint is missing its immutable Stage-1 prior")
PY
}
validate_checkpoint "Stage 1" "${stage1_checkpoint}" false
validate_checkpoint "Stage 2" "${stage2_checkpoint}" true

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_complete_evidence}
OUT=${OUT:-${EVIDENCE_ROOT}/${RUN_TAG}}
mkdir -p "${OUT}" "${TMP_ROOT}"
cd "${CODE_ROOT}"

run_test() {
  local name=$1 checkpoint=$2
  shift 2
  mkdir -p "${OUT}/${name}"
  env \
    TRACE_PROJECT_ROOT="${CODE_ROOT}" \
    TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    "${TRACE_VB_PYTHON}" run.py \
      --model "${TRACE_VB_MODEL_CONFIG}" \
      --dataset gsm8k_aug_nl \
      --trainer default \
      --devices 0 \
      --workspace_path "${CODE_ROOT}" \
      --test_ckpt_path "${checkpoint}" \
      --test_times 1 \
      --seed 0 \
      trainer.logger.save_dir="${OUT}" \
      trainer.logger.name="${name}" \
      trainer.logger.version=run \
      val_batch_size=1 \
      num_workers=4 \
      persistent_workers=false \
      data_module.dataset_dir="${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL" \
      data_module.enforce_registered_source=true \
      "$@" \
      2>&1 | tee "${OUT}/${name}/test.log"
}

cat > "${OUT}/manifest.txt" <<EOF
model=TRACE-VB-v5
model_config=${TRACE_VB_MODEL_CONFIG}
stage1_checkpoint=${stage1_checkpoint}
stage2_checkpoint=${stage2_checkpoint}
physical_gpu=${physical_gpu}
test_times=1
gsm8k_questions=1319
geometry_questions=200
geometry_rollouts_per_question=8
geometry_rollout_schema=iid_role_conditioned_gaussian_with_deterministic_COMMIT
projection_contract=global_unlabeled_PCA_no_manual_offsets_no_path_rescaling
causal_bootstrap_draws=10000
all_transition_interval=familywise_95_percent_Bonferroni_bootstrap
causal_primary=equal_norm_single_transition_replacement_with_suffix_and_COMMIT_recomputation
causal_co_primary=terminal_outcome_value_calibration
prefix_curve_interpretation=COMMIT_bottleneck_sanity_only_not_stepwise_contribution
ood_datasets=GSMHard,SVAMP,MultiArith
comparison=paired_stage1_vs_stage2
total_L_definition=8_latent_states_plus_all_generated_answer_tokens
gsm8k_source=${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL
generated_cots=false
latent_roles=PLAN,SOLVE1,SOLVE2,SOLVE3,SOLVE4,SOLVE5,REFINE,COMMIT
record_role_schema=PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT
answer_context=question_plus_COMMIT_latent_only
answer_decoder_question_attention=true
private_latent_answer_access=false
deployment_policy_CoT_conditioning=false
posthoc_text_CoT_usage=semantic_alignment_and_frozen_decoder_audit_only
deployment_solve_text_decoder=false
started_at=$(date --iso-8601=seconds)
EOF

"${TRACE_VB_PYTHON}" tools/trace_prepare_pca_fit_set.py \
  --source-file "${TRACE_VB_DATA_ROOT}/data/raw/GSM8k-Aug-NL/gsm8k_train_processed.jsonl" \
  --output-dir "${OUT}/pca_fit_dataset" \
  --count 200

run_evidence_suite() {
  local prefix=$1 checkpoint=$2 rl_mode=$3
  local phase_override="model.model_kwargs.do_trace_rl=${rl_mode}"

  run_test "${prefix}_pca_fit_train200" "${checkpoint}" \
    "${phase_override}" \
    data_module.dataset_dir="${OUT}/pca_fit_dataset" \
    data_module.enforce_registered_source=false \
    data_module.test_file=gsm8k_test_processed.jsonl \
    model.model_kwargs.trace_policy_config.visual_record_limit=200 \
    model.model_kwargs.trace_policy_config.visual_group_size=8 \
    model.model_kwargs.trace_policy_config.visual_seed=314159

  # Full 1,319-question IID test; visual_record_limit limits only the paired
  # geometry cache, not evaluation rows.
  run_test "${prefix}_gsm8k_geometry200" "${checkpoint}" \
    "${phase_override}" \
    model.model_kwargs.trace_policy_config.visual_record_limit=200 \
    model.model_kwargs.trace_policy_config.visual_group_size=8 \
    model.model_kwargs.trace_policy_config.visual_seed=271828

  run_test "${prefix}_gsmhard" "${checkpoint}" \
    "${phase_override}" \
    data_module.target=src.datasets.gsm8k_aug_nl.GSM8KAugNLDataModule \
    data_module.dataset_name=gsmhard \
    data_module.dataset_dir="${TRACE_VB_DATA_ROOT}/data/raw/GSM8K-Hard" \
    data_module.train_file=gsmhard_test_processed.jsonl \
    data_module.val_file=gsmhard_test_processed.jsonl \
    data_module.test_file=gsmhard_test_processed.jsonl \
    data_module.enforce_registered_source=false \
    model.model_kwargs.trace_policy_config.visual_record_limit=0

  run_test "${prefix}_svamp" "${checkpoint}" \
    "${phase_override}" \
    data_module.target=src.datasets.gsm8k_aug_nl.GSM8KAugNLDataModule \
    data_module.dataset_name=svamp \
    data_module.dataset_dir="${TRACE_VB_DATA_ROOT}/data/raw/SVAMP" \
    data_module.train_file=svamp_test_processed.jsonl \
    data_module.val_file=svamp_test_processed.jsonl \
    data_module.test_file=svamp_test_processed.jsonl \
    data_module.enforce_registered_source=false \
    model.model_kwargs.trace_policy_config.visual_record_limit=0

  run_test "${prefix}_multiarith" "${checkpoint}" \
    "${phase_override}" \
    data_module.target=src.datasets.gsm8k_aug_nl.GSM8KAugNLDataModule \
    data_module.dataset_name=multiarith \
    data_module.dataset_dir="${TRACE_VB_DATA_ROOT}/data/raw/MultiArith" \
    data_module.train_file=multiarith_test_processed.jsonl \
    data_module.val_file=multiarith_test_processed.jsonl \
    data_module.test_file=multiarith_test_processed.jsonl \
    data_module.enforce_registered_source=false \
    model.model_kwargs.trace_policy_config.visual_record_limit=0
}

run_evidence_suite stage1 "${stage1_checkpoint}" false
run_evidence_suite final "${stage2_checkpoint}" true

"${TRACE_VB_PYTHON}" tools/trace_policy_task_summary.py \
  --evidence-root "${OUT}" \
  --stage1-checkpoint "${stage1_checkpoint}" \
  --final-checkpoint "${stage2_checkpoint}" \
  --output-dir "${OUT}/task_summary"

stage1_fit_records="${OUT}/stage1_pca_fit_train200/run/trace_policy_visual_test.pt"
final_fit_records="${OUT}/final_pca_fit_train200/run/trace_policy_visual_test.pt"
[[ -f "${stage1_fit_records}" && -f "${final_fit_records}" ]] || \
  trace_vb_die "missing one or both shared-PCA fit caches"
for prefix in stage1 final; do
  records="${OUT}/${prefix}_gsm8k_geometry200/run/trace_policy_visual_test.pt"
  [[ -f "${records}" ]] || trace_vb_die "missing ${prefix} geometry cache"
  "${TRACE_VB_PYTHON}" tools/trace_policy_geometry_summary.py \
    --fit-records "${stage1_fit_records}" \
    --additional-fit-records "${final_fit_records}" \
    --records "${records}" \
    --output-dir "${OUT}/${prefix}_geometry_summary"
done

stage1_pca="${OUT}/stage1_geometry_summary/global_train_fit_pca.pt"
final_pca="${OUT}/final_geometry_summary/global_train_fit_pca.pt"
"${TRACE_VB_PYTHON}" - "${stage1_pca}" "${final_pca}" <<'PY'
import sys
import torch
left = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
right = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
for key in ("mean", "components", "explained_ratio"):
    if not torch.allclose(left[key], right[key], atol=1e-7, rtol=1e-6):
        raise SystemExit("Stage-1 and Stage-2 geometry used different PCA bases")
PY

stage1_records="${OUT}/stage1_gsm8k_geometry200/run/trace_policy_visual_test.pt"
final_records="${OUT}/final_gsm8k_geometry200/run/trace_policy_visual_test.pt"
"${TRACE_VB_PYTHON}" tools/trace_policy_stage_comparison.py \
  --stage1-records "${stage1_records}" \
  --final-records "${final_records}" \
  --stage1-geometry "${OUT}/stage1_geometry_summary/question_geometry.csv" \
  --final-geometry "${OUT}/final_geometry_summary/question_geometry.csv" \
  --shared-pca "${stage1_pca}" \
  --output-dir "${OUT}/stage_comparison"

mkdir -p "${OUT}/causal_summary"
env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TRACE_PROJECT_ROOT="${CODE_ROOT}" \
  TRACE_DATA_ROOT="${TRACE_VB_DATA_ROOT}" \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  "${TRACE_VB_PYTHON}" tools/trace_policy_causal_summary.py \
    --checkpoint "${stage2_checkpoint}" \
    --records "${final_records}" \
    --output-dir "${OUT}/causal_summary" \
    --device cuda:0 \
    --count 200 \
    --bootstrap 10000 \
    2>&1 | tee "${OUT}/causal_summary/run.log"

cat >> "${OUT}/manifest.txt" <<EOF
stage1_records=${stage1_records}
final_records=${final_records}
shared_pca_stage1_fit_records=${stage1_fit_records}
shared_pca_final_fit_records=${final_fit_records}
shared_pca_state=${stage1_pca}
stage1_geometry_summary=${OUT}/stage1_geometry_summary
final_geometry_summary=${OUT}/final_geometry_summary
paired_stage_comparison=${OUT}/stage_comparison
task_summary=${OUT}/task_summary
causal_summary=${OUT}/causal_summary
finished_at=$(date --iso-8601=seconds)
EOF

"${TRACE_VB_PYTHON}" tools/verify_evidence_complete.py \
  --evidence-root "${OUT}" \
  --stage1-checkpoint "${stage1_checkpoint}" \
  --final-checkpoint "${stage2_checkpoint}" \
  --write-complete \
  > "${OUT}/completeness_gate.json"
