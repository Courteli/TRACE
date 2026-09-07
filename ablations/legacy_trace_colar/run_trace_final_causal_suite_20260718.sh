#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <checkpoint> <output-root>" >&2
  exit 2
fi
physical_gpus=$1
checkpoint=$2
out_root=$3
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing checkpoint: ${checkpoint}" >&2
  exit 2
fi
IFS=',' read -r -a gpus <<< "${physical_gpus}"
if [[ "${#gpus[@]}" -ne 4 ]] || \
   [[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "Causal suite requires exactly four unique GPUs" >&2
  exit 2
fi

conditions=(
  "full|-1|-1|none"
  "no_path|0|-1|none"
  "prefix_1|1|-1|none"
  "prefix_2|2|-1|none"
  "prefix_3|3|-1|none"
  "prefix_4|4|-1|none"
  "prefix_5|5|-1|none"
  "prefix_6|6|-1|none"
  "prefix_7|7|-1|none"
  "reverse|-1|-1|reverse"
  "shuffle|-1|-1|shuffle"
  "mean_repeat|-1|-1|mean_repeat"
  "random_direction|-1|-1|random_direction"
  "same_norm_random_path|-1|-1|same_norm_random_path"
  "same_question_swap|-1|-1|same_question_swap"
  "cross_question_swap|-1|-1|cross_question_swap"
  "drop_z0|-1|0|none"
  "drop_z1|-1|1|none"
  "drop_z2|-1|2|none"
  "drop_z3|-1|3|none"
  "drop_z4|-1|4|none"
  "drop_z5|-1|5|none"
  "drop_z6|-1|6|none"
  "drop_z7|-1|7|none"
  "replace_t0|-1|-1|replace_transition_0"
  "replace_t1|-1|-1|replace_transition_1"
  "replace_t2|-1|-1|replace_transition_2"
  "replace_t3|-1|-1|replace_transition_3"
  "replace_t4|-1|-1|replace_transition_4"
  "replace_t5|-1|-1|replace_transition_5"
  "replace_t6|-1|-1|replace_transition_6"
  "replace_t7|-1|-1|replace_transition_7"
)

mkdir -p "${out_root}"
pids=()
names=()
cleanup_children() {
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup_children EXIT INT TERM

for condition_index in "${!conditions[@]}"; do
  IFS='|' read -r name prefix drop intervention <<< \
    "${conditions[${condition_index}]}"
  if [[ -f "${out_root}/${name}/manifest.txt" ]] && \
     rg -q '^finished_at=' "${out_root}/${name}/manifest.txt"; then
    echo "SKIP completed causal condition ${name}"
    continue
  fi
  if [[ "${#pids[@]}" -eq 4 ]]; then
    for process_index in "${!pids[@]}"; do
      if ! wait "${pids[${process_index}]}"; then
        echo "Causal condition failed: ${names[${process_index}]}" >&2
        exit 1
      fi
    done
    pids=()
    names=()
  fi
  slot=${#pids[@]}
  gpu=${gpus[${slot}]}
  data_batch_size=1
  limit_test_batches=200
  if [[ "${name}" == "cross_question_swap" ]]; then
    data_batch_size=4
    limit_test_batches=50
  fi
  setsid env \
    LIMIT_TEST_BATCHES="${limit_test_batches}" \
    EXPECTED_EVAL_QUESTIONS=200 \
    EVAL_DATA_BATCH_SIZE="${data_batch_size}" \
    VISUAL_RECORD_LIMIT=0 \
    TRACE_PREFIX_K="${prefix}" \
    TRACE_HIDDEN_DROP_INDEX="${drop}" \
    TRACE_INTERVENTION="${intervention}" \
    EVAL_SEED=271828 \
    bash "${ROOT}/run_trace_final_eval_20260718.sh" \
      "${gpu}" \
      "${checkpoint}" \
      gsm8k \
      "${out_root}/${name}" \
      > "${out_root}/${name}.log" 2>&1 &
  pids+=("$!")
  names+=("${name}")
done
for process_index in "${!pids[@]}"; do
  if ! wait "${pids[${process_index}]}"; then
    echo "Causal condition failed: ${names[${process_index}]}" >&2
    exit 1
  fi
done

"${PYTHON}" "${ROOT}/tools/trace_final_causal_summary.py" \
  --suite-root "${out_root}" \
  --out-dir "${out_root}/summary" \
  --bootstrap-trials 10000 \
  --seed 0 \
  2>&1 | tee "${out_root}/causal_summary.log"
