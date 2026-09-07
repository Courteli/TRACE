#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <stage1-evidence-root> <final-evidence-root> <output-root>" >&2
  exit 2
fi

STAGE1=$1
FINAL=$2
OUT_ROOT=$3
for root in "${STAGE1}" "${FINAL}"; do
  if [[ ! -f "${root}/evidence_done.txt" ]]; then
    echo "Incomplete evidence root: ${root}" >&2
    exit 2
  fi
done

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

"${PYTHON}" tools/trace_exchangeable_mainline_summary.py \
  --stage1_root "${STAGE1}" \
  --final_root "${FINAL}" \
  --out_dir "${OUT_ROOT}/task" \
  --bootstrap_trials 10000 \
  --length_tolerance 0.5 \
  --seed 0 \
  2>&1 | tee "${OUT_ROOT}/task.log"

cache_path() {
  local root=$1
  local dataset=$2
  printf '%s/%s_seed271828/logs/tb/run/trace_exchangeable_visual_test.pt' \
    "${root}" "${dataset}"
}

for dataset in gsm8k gsmhard svamp multiarith; do
  stage1_cache=$(cache_path "${STAGE1}" "${dataset}")
  final_cache=$(cache_path "${FINAL}" "${dataset}")
  if [[ ! -f "${stage1_cache}" || ! -f "${final_cache}" ]]; then
    echo "Missing matched geometry cache for ${dataset}" >&2
    exit 1
  fi
  "${PYTHON}" tools/trace_exchangeable_geometry_summary.py \
    --record "Stage1=${stage1_cache}" \
    --record "TRACE=${final_cache}" \
    --out_dir "${OUT_ROOT}/geometry/${dataset}" \
    --max_records 200 \
    --label_permutations 128 \
    --bootstrap_trials 10000 \
    --ranking_margin 0.08 \
    --baseline_label Stage1 \
    --target_label TRACE \
    --seed 0 \
    2>&1 | tee "${OUT_ROOT}/geometry_${dataset}.log"
done

cat > "${OUT_ROOT}/analysis_done.txt" <<EOF
training_seeds=1
test_times=1
task_eval=full
geometry_questions=200
geometry_rollout_seed=271828
completed_at=$(date '+%F %T')
EOF
