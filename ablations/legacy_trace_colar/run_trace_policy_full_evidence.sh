#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_policy/tmp}

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <physical-gpu> <stage1-best-checkpoint> <stage2-best-checkpoint>" >&2
  exit 2
fi
physical_gpu=$1
stage1_checkpoint=$2
stage2_checkpoint=$3

validate_checkpoint() {
  local phase=$1
  local checkpoint=$2
  local expected_rl=$3
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing ${phase} checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  local hparams
  hparams="$(dirname "$(dirname "${checkpoint}")")/hparams.yaml"
  if [[ ! -f "${hparams}" ]]; then
    echo "Missing ${phase} checkpoint hparams: ${hparams}" >&2
    exit 2
  fi
  if ! grep -q "src.models.trace_policy.LitTRACEPolicy" "${hparams}"; then
    echo "${phase} checkpoint is not a stochastic TRACE policy model" >&2
    exit 2
  fi
  if ! grep -qi "do_trace_rl: ${expected_rl}" "${hparams}"; then
    echo "${phase} checkpoint has the wrong training phase" >&2
    exit 2
  fi
  "${PYTHON}" - "${checkpoint}" "${expected_rl}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected_stage = 2 if sys.argv[2].lower() == "true" else 1
actual_stage = int(checkpoint.get("trace_policy_training_stage", -1))
if actual_stage != expected_stage:
    raise SystemExit(
        f"checkpoint phase marker is {actual_stage}, expected {expected_stage}"
    )
keys = checkpoint.get("state_dict", {})
if not any(".trace_teacher." in key for key in keys):
    raise SystemExit("checkpoint is missing the frozen CoT teacher adapter")
if expected_stage == 2 and "trace_stage1_policy_reference" not in checkpoint:
    raise SystemExit(
        "Stage-2 checkpoint is missing its immutable Stage-1 policy prior"
    )
PY
}
validate_checkpoint "Stage 1" "${stage1_checkpoint}" "false"
validate_checkpoint "Stage 2" "${stage2_checkpoint}" "true"

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)_trace_policy_full_evidence}
OUT=${OUT:-${ROOT}/run_outputs/trace_policy/evidence/${RUN_TAG}}
mkdir -p "${OUT}" "${TMP_ROOT}"

run_test() {
  local name=$1
  local checkpoint=$2
  shift 2
  mkdir -p "${OUT}/${name}"
  env \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    "${PYTHON}" run.py \
      --model trace_policy_qwen3_instruct \
      --dataset trace_qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /disk1/dingxukai \
      --test_ckpt_path "${checkpoint}" \
      --test_times 1 \
      --seed 0 \
      trainer.logger.save_dir="${OUT}" \
      trainer.logger.name="${name}" \
      trainer.logger.version=run \
      val_batch_size=1 \
      num_workers=4 \
      persistent_workers=false \
      "$@" \
      2>&1 | tee "${OUT}/${name}/test.log"
}

cat > "${OUT}/manifest.txt" <<EOF
stage1_checkpoint=${stage1_checkpoint}
stage2_checkpoint=${stage2_checkpoint}
physical_gpu=${physical_gpu}
test_times=1
gsm8k_questions=1319
geometry_questions=200
geometry_rollouts_per_question=8
geometry_rollout_schema=iid_conditional_gaussian
projection_contract=global_unlabeled_PCA_no_manual_offsets_no_path_rescaling
ood_datasets=GSMHard,SVAMP,MultiArith
comparison=paired_stage1_vs_stage2
started_at=$(date --iso-8601=seconds)
EOF

# Fit the global PCA basis on 200 training questions selected without outcomes.
# Their task accuracy is not reported; this split exists only to freeze an
# unlabeled projection before any held-out test trajectory is displayed.
"${PYTHON}" tools/trace_prepare_pca_fit_set.py \
  --source-dir "${ROOT}/run_outputs/trace_final/data/gsm8k_multirationale_v1" \
  --output-dir "${OUT}/pca_fit_dataset" \
  --count 200
run_evidence_suite() {
  local prefix=$1
  local checkpoint=$2
  local rl_mode=$3
  local phase_override="model.model_kwargs.do_trace_rl=${rl_mode}"

  run_test "${prefix}_pca_fit_train200" "${checkpoint}" \
    "${phase_override}" \
    data_module.dataset_dir="${OUT}/pca_fit_dataset" \
    model.model_kwargs.trace_policy_config.visual_record_limit=200 \
    model.model_kwargs.trace_policy_config.visual_group_size=8 \
    model.model_kwargs.trace_policy_config.visual_seed=314159

  # This run is both the deterministic GSM8K main-table evaluation and the
  # source of the pre-registered first 200 full-path evidence records.
  run_test "${prefix}_gsm8k_geometry200" "${checkpoint}" \
    "${phase_override}" \
    model.model_kwargs.trace_policy_config.visual_record_limit=200 \
    model.model_kwargs.trace_policy_config.visual_group_size=8 \
    model.model_kwargs.trace_policy_config.visual_seed=271828

  run_test "${prefix}_gsmhard" "${checkpoint}" \
    "${phase_override}" \
    data_module.target=src.datasets.gsm8k_aug_nl.GSM8KAugNLDataModule \
    data_module.dataset_name=gsmhard \
    data_module.dataset_dir=/home/dingxukai/RoT/data/GSM8k-Hard \
    data_module.train_file=gsmhard_test_processed.jsonl \
    data_module.val_file=gsmhard_test_processed.jsonl \
    data_module.test_file=gsmhard_test_processed.jsonl \
    model.model_kwargs.trace_policy_config.visual_record_limit=0

  run_test "${prefix}_svamp" "${checkpoint}" \
    "${phase_override}" \
    data_module.target=src.datasets.gsm8k_aug_nl.GSM8KAugNLDataModule \
    data_module.dataset_name=svamp \
    data_module.dataset_dir=/home/dingxukai/RoT/data/SVAMP \
    data_module.train_file=svamp_test_processed.jsonl \
    data_module.val_file=svamp_test_processed.jsonl \
    data_module.test_file=svamp_test_processed.jsonl \
    model.model_kwargs.trace_policy_config.visual_record_limit=0

  run_test "${prefix}_multiarith" "${checkpoint}" \
    "${phase_override}" \
    data_module.target=src.datasets.gsm8k_aug_nl.GSM8KAugNLDataModule \
    data_module.dataset_name=multiarith \
    data_module.dataset_dir=/home/dingxukai/RoT/data/Multiarith \
    data_module.train_file=multiarith_test_processed.jsonl \
    data_module.val_file=multiarith_test_processed.jsonl \
    data_module.test_file=multiarith_test_processed.jsonl \
    model.model_kwargs.trace_policy_config.visual_record_limit=0
}

run_evidence_suite stage1 "${stage1_checkpoint}" false
run_evidence_suite final "${stage2_checkpoint}" true

"${PYTHON}" tools/trace_policy_task_summary.py \
  --evidence-root "${OUT}" \
  --stage1-checkpoint "${stage1_checkpoint}" \
  --final-checkpoint "${stage2_checkpoint}" \
  --output-dir "${OUT}/task_summary"

stage1_fit_records="${OUT}/stage1_pca_fit_train200/run/trace_policy_visual_test.pt"
final_fit_records="${OUT}/final_pca_fit_train200/run/trace_policy_visual_test.pt"
if [[ ! -f "${stage1_fit_records}" || ! -f "${final_fit_records}" ]]; then
  echo "Missing one or both shared-PCA fit caches" >&2
  exit 1
fi
for prefix in stage1 final; do
  records="${OUT}/${prefix}_gsm8k_geometry200/run/trace_policy_visual_test.pt"
  if [[ ! -f "${records}" ]]; then
    echo "Missing ${prefix} 200-question cache: ${records}" >&2
    exit 1
  fi
  "${PYTHON}" tools/trace_policy_geometry_summary.py \
    --fit-records "${stage1_fit_records}" \
    --additional-fit-records "${final_fit_records}" \
    --records "${records}" \
    --output-dir "${OUT}/${prefix}_geometry_summary"
done

stage1_pca="${OUT}/stage1_geometry_summary/global_train_fit_pca.pt"
final_pca="${OUT}/final_geometry_summary/global_train_fit_pca.pt"
"${PYTHON}" - "${stage1_pca}" "${final_pca}" <<'PY'
import sys
import torch

left = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
right = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
for key in ("mean", "components", "explained_ratio"):
    if not torch.allclose(left[key], right[key], atol=1e-7, rtol=1e-6):
        raise SystemExit(
            "Stage-1 and Final geometry did not use the same shared PCA"
        )
PY
stage1_records="${OUT}/stage1_gsm8k_geometry200/run/trace_policy_visual_test.pt"
final_records="${OUT}/final_gsm8k_geometry200/run/trace_policy_visual_test.pt"
"${PYTHON}" tools/trace_policy_stage_comparison.py \
  --stage1-records "${stage1_records}" \
  --final-records "${final_records}" \
  --stage1-geometry "${OUT}/stage1_geometry_summary/question_geometry.csv" \
  --final-geometry "${OUT}/final_geometry_summary/question_geometry.csv" \
  --shared-pca "${stage1_pca}" \
  --output-dir "${OUT}/stage_comparison"

mkdir -p "${OUT}/causal_summary"
env \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  "${PYTHON}" tools/trace_policy_causal_summary.py \
    --checkpoint "${stage2_checkpoint}" \
    --records "${final_records}" \
    --output-dir "${OUT}/causal_summary" \
    --device cuda:0 \
    --count 200 \
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
