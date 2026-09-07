#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
EVIDENCE_SCRIPT=${ROOT}/run_trace_exchangeable_evidence_20260718.sh

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 <gpu> <stage1-ckpt> <final-evidence-root> <out-root> <label>" >&2
  exit 2
fi

gpu=$1
stage1_checkpoint=$2
final_root=$3
out_root=$4
label=$5

if [[ ! -f "${stage1_checkpoint}" ]]; then
  echo "Missing Stage1 checkpoint: ${stage1_checkpoint}" >&2
  exit 2
fi
if [[ ! -f "${final_root}/evidence_done.txt" ]]; then
  echo "Final evidence is incomplete: ${final_root}" >&2
  exit 2
fi

mkdir -p "${out_root}"
stage1_root="${out_root}/stage1_evidence"
comparison_root="${out_root}/paired_comparison"

OUT_ROOT="${stage1_root}" \
RUN_SECOND_GSM_GEOMETRY_SEED=false \
"${EVIDENCE_SCRIPT}" \
  "${gpu}" \
  "${stage1_checkpoint}" \
  "${label}_best_Stage1"

"${PYTHON}" "${ROOT}/tools/trace_exchangeable_mainline_summary.py" \
  --stage1_root "${stage1_root}" \
  --final_root "${final_root}" \
  --out_dir "${comparison_root}/task" \
  --bootstrap_trials 10000 \
  --length_tolerance 0.5 \
  --seed 0 \
  2>&1 | tee "${comparison_root}/task_summary.log"

declare -A DATASET_LABELS=(
  [gsm8k]=GSM8K
  [gsmhard]=GSMHard
  [svamp]=SVAMP
  [multiarith]=MultiArith
)

for dataset in gsm8k gsmhard svamp multiarith; do
  stage1_cache="${stage1_root}/${dataset}_seed271828/logs/tb/run/trace_exchangeable_visual_test.pt"
  final_cache="${final_root}/${dataset}_seed271828/logs/tb/run/trace_exchangeable_visual_test.pt"
  if [[ ! -f "${stage1_cache}" || ! -f "${final_cache}" ]]; then
    echo "Missing paired geometry cache for ${dataset}" >&2
    exit 1
  fi
  "${PYTHON}" "${ROOT}/tools/trace_exchangeable_geometry_summary.py" \
    --record "Stage1=${stage1_cache}" \
    --record "Final=${final_cache}" \
    --baseline_label Stage1 \
    --target_label Final \
    --out_dir "${comparison_root}/geometry/${dataset}" \
    --max_records 200 \
    --label_permutations 128 \
    --bootstrap_trials 10000 \
    --ranking_margin 0.08 \
    --seed 0 \
    2>&1 | tee "${comparison_root}/geometry_${dataset}.log"
done

"${PYTHON}" "${ROOT}/tools/trace_exchangeable_paired_figures.py" \
  --task_summary "${comparison_root}/task/trace_mainline_task_summary.json" \
  --geometry_summary "GSM8K=${comparison_root}/geometry/gsm8k/trace_exchangeable_geometry_summary.json" \
  --geometry_summary "GSMHard=${comparison_root}/geometry/gsmhard/trace_exchangeable_geometry_summary.json" \
  --geometry_summary "SVAMP=${comparison_root}/geometry/svamp/trace_exchangeable_geometry_summary.json" \
  --geometry_summary "MultiArith=${comparison_root}/geometry/multiarith/trace_exchangeable_geometry_summary.json" \
  --stage1_records "${stage1_root}/gsm8k_seed271828/logs/tb/run/trace_exchangeable_visual_test.pt" \
  --final_records "${final_root}/gsm8k_seed271828/logs/tb/run/trace_exchangeable_visual_test.pt" \
  --out_dir "${comparison_root}/publication_figures" \
  --max_records 200 \
  --case_count 2 \
  2>&1 | tee "${comparison_root}/publication_figures.log"

cat > "${out_root}/PAIRED_PROOF_COMPLETE" <<EOF
stage1_checkpoint=${stage1_checkpoint}
final_evidence_root=${final_root}
training_seeds=1
rollout_seeds=1
test_times=1
paired_questions_per_geometry_dataset=200
completed_at=$(date '+%F %T')
EOF
