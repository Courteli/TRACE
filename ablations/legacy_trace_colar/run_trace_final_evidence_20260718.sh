#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 <four-gpu-csv> <stage1-ckpt> <answer-only-ckpt> <final-ckpt> <output-root>" >&2
  exit 2
fi
physical_gpus=$1
stage1_checkpoint=$2
answeronly_checkpoint=$3
final_checkpoint=$4
out_root=$5
for checkpoint in \
  "${stage1_checkpoint}" \
  "${answeronly_checkpoint}" \
  "${final_checkpoint}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
  fi
done
IFS=',' read -r -a gpus <<< "${physical_gpus}"
if [[ "${#gpus[@]}" -ne 4 ]] || \
   [[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "Evidence runner requires exactly four unique GPUs" >&2
  exit 2
fi

datasets=(gsm8k gsmhard svamp multiarith)
mkdir -p "${out_root}"

active_pids=()
cleanup_children() {
  local pid
  for pid in "${active_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${active_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup_children EXIT INT TERM

run_checkpoint_wave() {
  local checkpoint=$1
  local label=$2
  local pids=()
  active_pids=()
  for dataset_index in "${!datasets[@]}"; do
    local dataset=${datasets[${dataset_index}]}
    local output="${out_root}/${label}/${dataset}"
    if [[ -f "${output}/manifest.txt" ]] && \
       rg -q '^finished_at=' "${output}/manifest.txt"; then
      echo "SKIP completed ${label}/${dataset}"
      continue
    fi
    setsid env \
      LIMIT_TEST_BATCHES=1.0 \
      VISUAL_RECORD_LIMIT=200 \
      VISUAL_GROUP_VIEWS=8 \
      EVAL_SEED=271828 \
      bash "${ROOT}/run_trace_final_eval_20260718.sh" \
        "${gpus[${dataset_index}]}" \
        "${checkpoint}" \
        "${dataset}" \
        "${output}" \
        > "${out_root}/${label}_${dataset}.log" 2>&1 &
    pids+=("$!")
    active_pids+=("${pids[-1]}")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  active_pids=()
}

run_checkpoint_wave "${stage1_checkpoint}" stage1
run_checkpoint_wave "${answeronly_checkpoint}" answeronly
run_checkpoint_wave "${final_checkpoint}" final

cache() {
  local label=$1
  local dataset=$2
  printf '%s/%s/%s/logs/tb/run/trace_final_visual_test.pt' \
    "${out_root}" "${label}" "${dataset}"
}
for label in stage1 answeronly final; do
  for dataset in "${datasets[@]}"; do
    if [[ ! -f "$(cache "${label}" "${dataset}")" ]]; then
      echo "Missing TRACE final cache: $(cache "${label}" "${dataset}")" >&2
      exit 1
    fi
  done
done

"${PYTHON}" "${ROOT}/tools/trace_final_task_summary.py" \
  --evidence-root "${out_root}" \
  --out-dir "${out_root}/task_summary" \
  --bootstrap-trials 10000 \
  --seed 0 \
  2>&1 | tee "${out_root}/task_summary.log"

"${PYTHON}" "${ROOT}/tools/trace_exchangeable_geometry_summary.py" \
  --record "Stage1_GSM8K=$(cache stage1 gsm8k)" \
  --record "AnswerOnly_GSM8K=$(cache answeronly gsm8k)" \
  --record "Final_GSM8K=$(cache final gsm8k)" \
  --record "Final_GSMHard=$(cache final gsmhard)" \
  --record "Final_SVAMP=$(cache final svamp)" \
  --record "Final_MultiArith=$(cache final multiarith)" \
  --out_dir "${out_root}/complete_path_geometry" \
  --max_records 200 \
  --label_permutations 128 \
  --bootstrap_trials 10000 \
  --ranking_margin 0.08 \
  --baseline_label Stage1_GSM8K \
  --target_label Final_GSM8K \
  --comparison Stage1_GSM8K:Final_GSM8K \
  --comparison AnswerOnly_GSM8K:Final_GSM8K \
  --seed 0 \
  2>&1 | tee "${out_root}/complete_path_geometry.log"

"${PYTHON}" "${ROOT}/tools/trace_final_outcome_probe.py" \
  --record "Stage1_GSM8K=$(cache stage1 gsm8k)" \
  --record "AnswerOnly_GSM8K=$(cache answeronly gsm8k)" \
  --record "Final_GSM8K=$(cache final gsm8k)" \
  --record "Final_GSMHard=$(cache final gsmhard)" \
  --record "Final_SVAMP=$(cache final svamp)" \
  --record "Final_MultiArith=$(cache final multiarith)" \
  --crossfit-label Stage1_GSM8K \
  --crossfit-label AnswerOnly_GSM8K \
  --crossfit-label Final_GSM8K \
  --transfer Final_GSM8K:Final_GSMHard \
  --transfer Final_GSM8K:Final_SVAMP \
  --transfer Final_GSM8K:Final_MultiArith \
  --out-dir "${out_root}/outcome_probe" \
  --max-records 200 \
  --pca-components 8 \
  --folds 5 \
  --bootstrap-trials 10000 \
  --permutation-trials 128 \
  --seed 0 \
  2>&1 | tee "${out_root}/outcome_probe.log"

"${PYTHON}" "${ROOT}/tools/trace_exchangeable_visualize.py" \
  --records "$(cache final gsm8k)" \
  --out_dir "${out_root}/publication_visuals" \
  --max_records 200 \
  --n_representative 3 \
  2>&1 | tee "${out_root}/publication_visuals.log"

"${PYTHON}" "${ROOT}/tools/trace_final_stage_comparison_visualize.py" \
  --record "Stage1=$(cache stage1 gsm8k)" \
  --record "AnswerOnly=$(cache answeronly gsm8k)" \
  --record "Final=$(cache final gsm8k)" \
  --out-dir "${out_root}/matched_stage_visuals" \
  --max-records 200 \
  --n-representative 3 \
  --bootstrap-trials 10000 \
  2>&1 | tee "${out_root}/matched_stage_visuals.log"

cat > "${out_root}/evidence_done.txt" <<EOF
stage1_checkpoint=${stage1_checkpoint}
answeronly_checkpoint=${answeronly_checkpoint}
final_checkpoint=${final_checkpoint}
test_times=1
full_task_datasets=GSM8K,GSMHard,SVAMP,MultiArith
geometry_questions=200
paths_per_question=8
completed_at=$(date --iso-8601=seconds)
EOF
