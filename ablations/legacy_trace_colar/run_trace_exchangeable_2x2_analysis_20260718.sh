#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python

usage() {
  cat <<'EOF'
Usage:
  run_trace_exchangeable_2x2_analysis_20260718.sh \
    <S1P-evidence-root> <S1F-evidence-root> \
    <M00-evidence-root> <M01-evidence-root> \
    <M10-evidence-root> <M11-evidence-root> <output-root>

All six roots must have been produced by
run_trace_exchangeable_evidence_20260718.sh. This command performs analysis
only; it starts no GPU job and creates no waiter.
EOF
}

if [[ "$#" -ne 7 ]]; then
  usage
  exit 2
fi

S1P=$1
S1F=$2
M00=$3
M01=$4
M10=$5
M11=$6
OUT_ROOT=$7

for root in "${S1P}" "${S1F}" "${M00}" "${M01}" "${M10}" "${M11}"; do
  if [[ ! -d "${root}" ]]; then
    echo "Missing evidence root: ${root}" >&2
    exit 2
  fi
done

mkdir -p "${OUT_ROOT}"
cd "${ROOT}"

"${PYTHON}" tools/trace_exchangeable_2x2_task_summary.py \
  --cell "S1P=${S1P}" \
  --cell "S1F=${S1F}" \
  --cell "M00=${M00}" \
  --cell "M01=${M01}" \
  --cell "M10=${M10}" \
  --cell "M11=${M11}" \
  --out_dir "${OUT_ROOT}/task" \
  --bootstrap_trials 10000 \
  --length_tolerance 0.5 \
  --seed 0 \
  2>&1 | tee "${OUT_ROOT}/task.log"

cache_path() {
  local root=$1
  local dataset=$2
  local rollout_seed=$3
  printf '%s/%s_seed%s/logs/tb/run/trace_exchangeable_visual_test.pt' \
    "${root}" "${dataset}" "${rollout_seed}"
}

run_geometry() {
  local dataset=$1
  local rollout_seed=$2
  local output_name=$3
  local output_dir="${OUT_ROOT}/geometry/${output_name}"
  local args=()
  local label
  local root
  for entry in \
    "S1P=${S1P}" \
    "S1F=${S1F}" \
    "M00=${M00}" \
    "M01=${M01}" \
    "M10=${M10}" \
    "M11=${M11}"; do
    label=${entry%%=*}
    root=${entry#*=}
    cache=$(cache_path "${root}" "${dataset}" "${rollout_seed}")
    if [[ ! -f "${cache}" ]]; then
      echo "Missing cache: ${cache}" >&2
      exit 1
    fi
    args+=(--record "${label}=${cache}")
  done
  "${PYTHON}" tools/trace_exchangeable_geometry_summary.py \
    "${args[@]}" \
    --out_dir "${output_dir}" \
    --max_records 200 \
    --label_permutations 128 \
    --bootstrap_trials 10000 \
    --ranking_margin 0.08 \
    --baseline_label M10 \
    --target_label M11 \
    --seed 0 \
    2>&1 | tee "${OUT_ROOT}/geometry_${output_name}.log"
}

run_geometry gsm8k 271828 gsm8k_seed0
run_geometry gsm8k 314159 gsm8k_seed1
run_geometry gsmhard 271828 gsmhard
run_geometry svamp 271828 svamp
run_geometry multiarith 271828 multiarith

cat > "${OUT_ROOT}/analysis_done.txt" <<EOF
test_times=1
cells=S1P,S1F,M00,M01,M10,M11
geometry_questions=200
rollout_seeds=271828,314159
completed_at=$(date '+%F %T')
EOF
