#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
PY=/home/dingxukai/miniconda3/envs/ROT/bin/python
GPU_PRIMARY="${GPU_PRIMARY:-7}"
GPU_DUAL="${GPU_DUAL:-1,7}"
OUT="${OUT:-${ROOT}/run_outputs/trace_bridge/20260716_trace_bridge_answeronly_epoch4_final_evidence_gpu7}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/run_roots/trace_bridge/20260716_trace_bridge_answeronly_epoch4_final_evidence_gpu7}"
CKPT="${CKPT:-${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260714-003100_749268_20260713_trace_bridge_answeronly_fullbudget_control_stage2_answer_only/checkpoints/epoch4__step2560__monitor0.725936.ckpt}"
MODEL=trace_bridge_qwen3_instruct_vizstrong
POLL_SECONDS="${POLL_SECONDS:-120}"
DUAL_GPU1_MAX_USED_MIB="${DUAL_GPU1_MAX_USED_MIB:-9000}"
DUAL_GPU7_MAX_USED_MIB="${DUAL_GPU7_MAX_USED_MIB:-7000}"
PRIMARY_EXCLUSIVE_MAX_USED_MIB="${PRIMARY_EXCLUSIVE_MAX_USED_MIB:-1000}"

BRIDGE_RECORD=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/matched_bridge_rollouts/logs/tb/run/trace_bridge_visual_test.pt
STAGE1_RECORD=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/matched_stage1_rollouts/logs/tb/run/trace_bridge_visual_test.pt
BRIDGE_ROWS=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/matched_bridge_rollouts/geometry_raw/trace_bridge_geometry_rows.csv
STAGE1_ROWS=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/matched_stage1_rollouts/geometry_raw/trace_bridge_geometry_rows.csv
STAGE1_RAW=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/matched_stage1_rollouts/geometry_raw/trace_bridge_geometry_summary.json
STAGE1_CENTERED=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/matched_stage1_rollouts/geometry_stage2_metric/trace_bridge_geometry_summary.json

declare -A DATASETS=(
  [gsm8k]=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc
  [gsmhard]=/home/dingxukai/RoT/data/GSM8k-Hard/readcot_qsa_test
  [svamp]=/home/dingxukai/RoT/data/SVAMP/readcot_qsa_test
  [multiarith]=/home/dingxukai/RoT/data/Multiarith/readcot_qsa_test
)

mkdir -p "${OUT}" "${RUN_ROOT}" "${OUT}/benchmarks" "${OUT}/behavior_interventions"
cd "${ROOT}"

exec 9>"${OUT}/pipeline.lock"
if ! flock -n 9; then
  printf '[skip] another epoch4 evidence pipeline already holds %s\n' "${OUT}/pipeline.lock"
  exit 0
fi

timestamp() {
  date '+%F %T'
}

gpu_used_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' '
}

wait_for_dual_capacity() {
  local used1 used7
  while true; do
    used1="$(gpu_used_mib 1)"
    used7="$(gpu_used_mib 7)"
    printf '%s dual_wait gpu1_used=%sMiB gpu7_used=%sMiB\n' "$(timestamp)" "${used1}" "${used7}" | tee -a "${OUT}/resource_queue.log"
    if (( used1 <= DUAL_GPU1_MAX_USED_MIB && used7 <= DUAL_GPU7_MAX_USED_MIB )); then
      return
    fi
    sleep "${POLL_SECONDS}"
  done
}

wait_for_primary_capacity() {
  local used
  while true; do
    used="$(gpu_used_mib "${GPU_PRIMARY}")"
    printf '%s primary_wait gpu%s_used=%sMiB\n' "$(timestamp)" "${GPU_PRIMARY}" "${used}" | tee -a "${OUT}/resource_queue.log"
    if (( used <= PRIMARY_EXCLUSIVE_MAX_USED_MIB )); then
      return
    fi
    sleep "${POLL_SECONDS}"
  done
}

result_json() {
  local log_dir="$1"
  find "${log_dir}/tb/run" -maxdepth 1 -type f -name 'test_*_gsm_pid*.json' -print 2>/dev/null | sort | tail -n 1
}

has_result() {
  [[ -n "$(result_json "$1")" ]]
}

wait_for_auxiliary_ood() {
  local key="$1"
  local log_dir="$2"
  local pid_file="${OUT}/ood_gpu1_launcher.pid"
  local worker_pid worker_cmd
  case "${key}" in
    gsmhard|svamp|multiarith) ;;
    *) return ;;
  esac
  [[ -f "${pid_file}" ]] || return
  worker_pid="$(cat "${pid_file}")"
  while kill -0 "${worker_pid}" 2>/dev/null && ! has_result "${log_dir}"; do
    worker_cmd="$(ps -p "${worker_pid}" -o args= 2>/dev/null || true)"
    [[ "${worker_cmd}" == *run_trace_answeronly_epoch4_ood_gpu1_20260716.sh* ]] || return
    printf '%s [wait] auxiliary GPU1 worker owns key=%s\n' "$(timestamp)" "${key}" | tee -a "${OUT}/pipeline_status.log"
    sleep "${POLL_SECONDS}"
  done
}

run_eval_attempt() {
  local key="$1"
  local dataset_dir="$2"
  local base="$3"
  local intervention="$4"
  local capture_visuals="$5"
  local physical_gpus="$6"
  local logical_devices="$7"
  local strategy="$8"
  local attempt="$9"
  local log_dir="${base}/logs"
  local log_file="${base}/eval_${attempt}.log"
  local -a trace_args

  mkdir -p "${log_dir}"
  if [[ "${capture_visuals}" == 1 ]]; then
    trace_args=(
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=true
      model.model_kwargs.trace_bridge_config.trace_visual_group_views=8
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=200
      model.model_kwargs.trace_bridge_config.trace_visual_latent_noise_scale=0.015
      model.model_kwargs.trace_bridge_config.trace_visual_noise_seed=0
      model.model_kwargs.trace_bridge_config.trace_visual_do_sample=true
      model.model_kwargs.trace_bridge_config.trace_visual_temperature=0.95
      model.model_kwargs.trace_bridge_config.trace_visual_top_p=0.97
      model.model_kwargs.trace_bridge_config.trace_visual_micro_batch_size=4
      model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=false
      model.model_kwargs.trace_rl_config.stage2_latent_noise_scale=0.015
      model.model_kwargs.trace_rl_config.exp_batch_size=4
    )
  else
    trace_args=(
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=false
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0
      model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true
    )
  fi

  printf '%s [run] key=%s intervention=%s physical_gpus=%s attempt=%s\n' \
    "$(timestamp)" "${key}" "${intervention}" "${physical_gpus}" "${attempt}" | tee -a "${OUT}/pipeline_status.log"
  set +e
  TMPDIR=/tmp \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
    "${PY}" run.py \
      --model "${MODEL}" \
      --dataset qsa \
      --trainer default \
      --devices "${logical_devices}" \
      --workspace_path /home/dingxukai \
      --test_ckpt_path "${CKPT}" \
      --test_times 1 \
      dataset_dir="${dataset_dir}" \
      batch_size=1 \
      val_batch_size=1 \
      num_workers=2 \
      persistent_workers=false \
      do_trace_rl=true \
      trainer.num_sanity_val_steps=0 \
      trainer.strategy="${strategy}" \
      trainer.default_root_dir="${RUN_ROOT}/${key}_${intervention}_${attempt}" \
      trainer.logger.save_dir="${log_dir}" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
      model.model_kwargs.trace_bridge_config.trace_eval_intervention="${intervention}" \
      model.model_kwargs.trace_bridge_config.trace_eval_intervention_seed=0 \
      "${trace_args[@]}" \
      2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

run_eval() {
  local key="$1"
  local dataset_dir="$2"
  local base="$3"
  local intervention="$4"
  local capture_visuals="$5"
  local allow_dual="$6"
  local primary_log="${base}/eval_gpu${GPU_PRIMARY}.log"

  mkdir -p "${base}"
  wait_for_auxiliary_ood "${key}" "${base}/logs"
  if has_result "${base}/logs"; then
    printf '%s [skip] completed key=%s intervention=%s\n' "$(timestamp)" "${key}" "${intervention}" | tee -a "${OUT}/pipeline_status.log"
    return
  fi

  if run_eval_attempt "${key}" "${dataset_dir}" "${base}" "${intervention}" "${capture_visuals}" "${GPU_PRIMARY}" 0 auto "gpu${GPU_PRIMARY}"; then
    if has_result "${base}/logs"; then
      printf 'gpu%s\n' "${GPU_PRIMARY}" > "${base}/completed_attempt.txt"
      return
    fi
  fi

  if ! grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "${primary_log}"; then
    printf '%s [error] non-OOM failure key=%s; see %s\n' "$(timestamp)" "${key}" "${primary_log}" | tee -a "${OUT}/pipeline_status.log"
    return 1
  fi

  if [[ "${allow_dual}" == 1 ]]; then
    wait_for_dual_capacity
    if run_eval_attempt "${key}" "${dataset_dir}" "${base}" "${intervention}" "${capture_visuals}" "${GPU_DUAL}" 0,1 ddp_find_unused_parameters_true gpu1_7; then
      if has_result "${base}/logs"; then
        printf 'gpu1_7\n' > "${base}/completed_attempt.txt"
        return
      fi
    fi
  fi

  # Per-question records and visual caches are rank-zero-local under DDP.
  # Wait for a clean GPU7 rather than accepting a truncated evidence file.
  wait_for_primary_capacity
  run_eval_attempt "${key}" "${dataset_dir}" "${base}" "${intervention}" "${capture_visuals}" "${GPU_PRIMARY}" 0 auto gpu7_exclusive_retry
  if ! has_result "${base}/logs"; then
    printf '%s [error] retry produced no result for key=%s\n' "$(timestamp)" "${key}" | tee -a "${OUT}/pipeline_status.log"
    return 1
  fi
  printf 'gpu7_exclusive_retry\n' > "${base}/completed_attempt.txt"
}

run_stage() {
  local name="$1"
  local marker="$2"
  local log_file="$3"
  shift 3
  if [[ -f "${marker}" ]]; then
    printf '%s [skip] stage=%s\n' "$(timestamp)" "${name}" | tee -a "${OUT}/pipeline_status.log"
    return
  fi
  printf '%s [run] stage=%s\n' "$(timestamp)" "${name}" | tee -a "${OUT}/pipeline_status.log"
  "$@" 2>&1 | tee "${log_file}"
  touch "${marker}"
}

{
  printf 'checkpoint=%s\n' "${CKPT}"
  printf 'checkpoint_sha256=%s\n' "$(sha256sum "${CKPT}" | awk '{print $1}')"
  printf 'model=%s\n' "${MODEL}"
  printf 'primary_gpu=%s\n' "${GPU_PRIMARY}"
  printf 'conditional_dual_gpus=%s\n' "${GPU_DUAL}"
  printf 'test_times=1\n'
  printf 'visual_questions=200\n'
  printf 'rollouts_per_visual_question=8\n'
  printf 'created_at=%s\n' "$(timestamp)"
} > "${OUT}/manifest.txt"

# The deterministic first prediction is the benchmark result. Eight sampled
# audit views are generated only while recording the first 200 GSM8K items.
run_eval gsm8k "${DATASETS[gsm8k]}" "${OUT}/benchmarks/gsm8k" none 1 0
run_eval gsmhard "${DATASETS[gsmhard]}" "${OUT}/benchmarks/gsmhard" none 0 1
run_eval svamp "${DATASETS[svamp]}" "${OUT}/benchmarks/svamp" none 0 1
run_eval multiarith "${DATASETS[multiarith]}" "${OUT}/benchmarks/multiarith" none 0 1

"${PY}" - "${OUT}" <<'PY'
import csv
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
names = {
    "gsm8k": "GSM8K",
    "gsmhard": "GSMHard",
    "svamp": "SVAMP",
    "multiarith": "MultiArith",
}
rows = []
for key, label in names.items():
    base = out / "benchmarks" / key
    attempt_file = base / "completed_attempt.txt"
    attempt = attempt_file.read_text().strip() if attempt_file.exists() else f"gpu7"
    log = base / f"eval_{attempt}.log"
    text = log.read_text(errors="ignore")
    values = {}
    for match in re.finditer(r"│\s*(test/[^│]+?)\s*│\s*([-0-9.]+)\s*│", text):
        values[match.group(1).strip()] = float(match.group(2))
    if "test/acc" not in values:
        raise RuntimeError(f"No final test metrics found in {log}")
    latent = values.get("test/n_latent_forward", 0.0)
    output_len = values.get("test/output_length", 0.0)
    rows.append({
        "dataset": label,
        "accuracy_percent": 100.0 * values["test/acc"],
        "n_latent": latent,
        "output_length": output_len,
        "total_L": latent + output_len,
        "attempt": attempt,
        "log": str(log),
    })

with (out / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "| Dataset | Accuracy | #Latent | Output length | #L | Device attempt |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
]
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['accuracy_percent']:.2f}% | {row['n_latent']:.2f} | "
        f"{row['output_length']:.2f} | {row['total_L']:.2f} | {row['attempt']} |"
    )
(out / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

TRACE_RECORD=${OUT}/benchmarks/gsm8k/logs/tb/run/trace_bridge_visual_test.pt
TRACE_VISUAL=${OUT}/visual_epoch4_gsm8k
TRACE_RAW=${TRACE_VISUAL}/geometry_200_raw/trace_bridge_geometry_summary.json
TRACE_CENTERED=${TRACE_VISUAL}/geometry_200_stage2_centered/trace_bridge_geometry_summary.json
TRACE_ROWS=${TRACE_VISUAL}/geometry_200_raw/trace_bridge_geometry_rows.csv

if [[ ! -f "${TRACE_RECORD}" ]]; then
  printf '[error] expected 200-question visual record not found: %s\n' "${TRACE_RECORD}" >&2
  exit 1
fi

mkdir -p "${TRACE_VISUAL}"
run_stage per_model_visuals "${TRACE_VISUAL}/.visuals.done" "${TRACE_VISUAL}/visualize.log" \
  "${PY}" tools/trace_bridge_visualize.py \
    --records "${TRACE_RECORD}" --out_dir "${TRACE_VISUAL}" --max_records 9 --pca_fit_records 200
run_stage geometry_raw "${TRACE_VISUAL}/geometry_200_raw/.done" "${TRACE_VISUAL}/geometry_200_raw.log" \
  "${PY}" tools/trace_bridge_geometry_summary.py \
    --records "${TRACE_RECORD}" --out_dir "${TRACE_VISUAL}/geometry_200_raw" --max_records 200
run_stage geometry_centered "${TRACE_VISUAL}/geometry_200_stage2_centered/.done" "${TRACE_VISUAL}/geometry_200_stage2_centered.log" \
  "${PY}" tools/trace_bridge_geometry_summary.py \
    --records "${TRACE_RECORD}" --out_dir "${TRACE_VISUAL}/geometry_200_stage2_centered" --max_records 200 \
    --signature_representation stage2_centered --signature_raw_mix 0.25

GLOBAL_RAW=${OUT}/global_pca_heatmaps_raw
GLOBAL_CENTERED=${OUT}/global_pca_heatmaps_stage2_centered
mkdir -p "${GLOBAL_RAW}" "${GLOBAL_CENTERED}"
run_stage global_pca_raw "${GLOBAL_RAW}/.done" "${GLOBAL_RAW}/run.log" \
  "${PY}" tools/trace_bridge_compare_rollouts.py \
    --record "BRIDGE=${BRIDGE_RECORD}" --record "Stage1=${STAGE1_RECORD}" --record "TRACE-answeronly-e4=${TRACE_RECORD}" \
    --out_dir "${GLOBAL_RAW}" --max_records 200 --pca_fit_records 200 \
    --signature_representation raw --selection_names fixed outcome_balanced \
    --normalizations normalized raw --require_common_questions 200 --force
run_stage global_pca_centered "${GLOBAL_CENTERED}/.done" "${GLOBAL_CENTERED}/run.log" \
  "${PY}" tools/trace_bridge_compare_rollouts.py \
    --record "BRIDGE=${BRIDGE_RECORD}" --record "Stage1=${STAGE1_RECORD}" --record "TRACE-answeronly-e4=${TRACE_RECORD}" \
    --out_dir "${GLOBAL_CENTERED}" --max_records 200 --pca_fit_records 200 \
    --signature_representation stage2_centered --signature_raw_mix 0.25 \
    --selection_names fixed outcome_balanced --normalizations normalized raw \
    --require_common_questions 200 --force

ASSIGNMENT_OUT=${OUT}/assignment_heatmaps_fixed
mkdir -p "${ASSIGNMENT_OUT}"
run_stage assignment_heatmaps "${ASSIGNMENT_OUT}/.done" "${ASSIGNMENT_OUT}/run.log" \
  "${PY}" tools/trace_bridge_compare_assignments.py \
    --record "BRIDGE=${BRIDGE_RECORD}" --record "Stage1=${STAGE1_RECORD}" --record "TRACE-answeronly-e4=${TRACE_RECORD}" \
    --indices 0 1 2 --out_dir "${ASSIGNMENT_OUT}" --max_records 200 \
    --selection_rule 'predeclared first three common GSM8K indices; no outcome, assignment, or geometry used'

STRUCTURE_OUT=${OUT}/trajectory_structure_200
mkdir -p "${STRUCTURE_OUT}"
run_stage trajectory_structure_200 "${STRUCTURE_OUT}/.done" "${STRUCTURE_OUT}/run.log" \
  "${PY}" tools/trace_bridge_trajectory_structure.py \
    --rows "BRIDGE=${BRIDGE_ROWS}" --rows "Stage1=${STAGE1_ROWS}" --rows "TRACE-answeronly-e4=${TRACE_ROWS}" \
    --comparison 'Stage1-formation=BRIDGE,Stage1' \
    --comparison 'Outcome-refinement=Stage1,TRACE-answeronly-e4' \
    --out_dir "${STRUCTURE_OUT}" --bootstrap_trials 10000 --seed 0

GEOMETRY_DELTA_OUT=${OUT}/stage1_to_answeronly_epoch4_geometry_delta
mkdir -p "${GEOMETRY_DELTA_OUT}"
run_stage stage_geometry_delta "${GEOMETRY_DELTA_OUT}/.done" "${GEOMETRY_DELTA_OUT}/run.log" \
  "${PY}" tools/trace_bridge_stage_geometry_delta.py \
    --stage1_raw "${STAGE1_RAW}" --trace_raw "${TRACE_RAW}" \
    --stage1_centered "${STAGE1_CENTERED}" --trace_centered "${TRACE_CENTERED}" \
    --out_dir "${GEOMETRY_DELTA_OUT}" --bootstrap_trials 10000 --seed 0

PREFIX_OUT=${OUT}/prefix_probe
mkdir -p "${PREFIX_OUT}"
if [[ ! -f "${PREFIX_OUT}/.done" && ! -f "${PREFIX_OUT}/failed.txt" ]]; then
  printf '%s [run] stage=prefix_probe\n' "$(timestamp)" | tee -a "${OUT}/pipeline_status.log"
  if "${PY}" tools/trace_bridge_prefix_probe.py \
      --records "${TRACE_RECORD}" --out_dir "${PREFIX_OUT}" --max_records 200 \
      --repeats 40 --projection_dim 128 --ridge 1.0 --raw_mix 0.25 --seed 0 \
      2>&1 | tee "${PREFIX_OUT}/run.log"; then
    touch "${PREFIX_OUT}/.done"
  else
    printf 'Prefix probe was not estimable; see run.log (usually fewer than 20 mixed-outcome questions).\n' > "${PREFIX_OUT}/failed.txt"
  fi
fi

for intervention in reverse shuffle mean_repeat random_direction; do
  run_eval "gsm8k_${intervention}" "${DATASETS[gsm8k]}" \
    "${OUT}/behavior_interventions/${intervention}" "${intervention}" 0 0
done

NORMAL_JSON="$(result_json "${OUT}/benchmarks/gsm8k/logs")"
CAUSAL_ARGS=()
for intervention in reverse shuffle mean_repeat random_direction; do
  INTERVENTION_JSON="$(result_json "${OUT}/behavior_interventions/${intervention}/logs")"
  CAUSAL_ARGS+=(--intervention "${intervention}=${INTERVENTION_JSON}")
done
CAUSAL_OUT=${OUT}/behavior_interventions/summary
mkdir -p "${CAUSAL_OUT}"
run_stage behavior_intervention_summary "${CAUSAL_OUT}/.done" "${CAUSAL_OUT}/run.log" \
  "${PY}" tools/trace_bridge_causal_evidence.py \
    --reference "${NORMAL_JSON}" "${CAUSAL_ARGS[@]}" --out_dir "${CAUSAL_OUT}" \
    --bootstrap_trials 10000 --seed 0

printf '%s\n' "$(timestamp)" > "${OUT}/pipeline_done.txt"
printf '%s [done] epoch4 final evidence pipeline\n' "$(timestamp)" | tee -a "${OUT}/pipeline_status.log"
