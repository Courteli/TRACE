#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
PY=/home/dingxukai/miniconda3/envs/ROT/bin/python
GPU=2
CKPT=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260714-003100_749268_20260713_trace_bridge_answeronly_fullbudget_control_stage2_answer_only/checkpoints/epoch4__step2560__monitor0.725936.ckpt
DATASET=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc
OUT=${ROOT}/run_outputs/trace/20260716_answeronly_hidden_prefix_gpu2/smoke
RUN_ROOT=${ROOT}/run_roots/trace/20260716_answeronly_hidden_prefix_gpu2/smoke
BATCH_SIZE=${BATCH_SIZE:-8}

mkdir -p "${OUT}" "${RUN_ROOT}"
cd "${ROOT}"

run_one() {
  local label="$1"
  local prefix_k="$2"
  local base="${OUT}/${label}_b${BATCH_SIZE}"
  mkdir -p "${base}/logs"
  if find "${base}/logs" -type f -name 'test_*_gsm_pid*.json' -print -quit | grep -q .; then
    printf '[skip] completed smoke label=%s batch=%s\n' "${label}" "${BATCH_SIZE}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" run.py \
      --model trace_bridge_qwen3_instruct_vizstrong \
      --dataset qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /home/dingxukai \
      --test_ckpt_path "${CKPT}" \
      --test_times 1 \
      --seed 0 \
      dataset_dir="${DATASET}" \
      tiny_dataset=true \
      batch_size="${BATCH_SIZE}" \
      val_batch_size="${BATCH_SIZE}" \
      num_workers=2 \
      persistent_workers=false \
      trainer.num_sanity_val_steps=0 \
      trainer.strategy=auto \
      trainer.default_root_dir="${RUN_ROOT}/${label}_b${BATCH_SIZE}" \
      trainer.logger.save_dir="${base}/logs" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
      model.model_kwargs.trace_bridge_config.trace_eval_hidden_prefix_k="${prefix_k}" \
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=false \
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
      model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true \
      2>&1 | tee "${base}/eval.log"
}

run_one baseline -1
run_one prefix8 8
run_one prefix0 0

"${PY}" - "${OUT}" "${BATCH_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
batch_size = int(sys.argv[2])

def load(label):
    paths = sorted((out / f"{label}_b{batch_size}" / "logs" / "tb" / "run").glob("test_*_gsm_pid*.json"))
    if not paths:
        raise SystemExit(f"missing result for {label}")
    return paths[-1], json.loads(paths[-1].read_text())

baseline_path, baseline = load("baseline")
prefix8_path, prefix8 = load("prefix8")
sample_ids = sorted((key for key in baseline if key.isdigit()), key=int)
fields = ("pred_answer", "output_string", "output_length", "acc")
mismatches = []
for sample_id in sample_ids:
    if any(baseline[sample_id].get(field) != prefix8[sample_id].get(field) for field in fields):
        mismatches.append(sample_id)

report = {
    "batch_size": batch_size,
    "n_samples": len(sample_ids),
    "baseline_json": str(baseline_path),
    "prefix8_json": str(prefix8_path),
    "compared_fields": list(fields),
    "n_mismatches": len(mismatches),
    "mismatch_ids": mismatches,
    "parity": not mismatches,
}
(out / f"prefix8_parity_b{batch_size}.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if mismatches:
    raise SystemExit("k=8 did not reproduce the unmodified generation path")
PY
