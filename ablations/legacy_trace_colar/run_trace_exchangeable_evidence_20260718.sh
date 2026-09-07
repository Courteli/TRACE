#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_exchangeable_qwen3_instruct
RUN_SECOND_GSM_GEOMETRY_SEED=${RUN_SECOND_GSM_GEOMETRY_SEED:-true}

usage() {
  cat <<'EOF'
Usage:
  run_trace_exchangeable_evidence_20260718.sh <physical-gpu> <checkpoint> <label>

Runs test_times=1 on complete GSM8K, GSMHard, SVAMP, and MultiArith test sets.
Each task receives one matched 200-question eight-path geometry cache. Set
RUN_SECOND_GSM_GEOMETRY_SEED=true for an optional second GSM8K geometry pass.
The script does not wait for a GPU or enqueue any later job.
EOF
}

gpu=${1:-}
checkpoint=${2:-}
label=${3:-}
if [[ -z "${gpu}" || -z "${checkpoint}" || -z "${label}" || ! -f "${checkpoint}" ]]; then
  usage
  exit 2
fi

declare -A DATASETS=(
  [gsm8k]=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc
  [gsmhard]=/home/dingxukai/RoT/data/GSM8k-Hard/readcot_qsa_test
  [svamp]=/home/dingxukai/RoT/data/SVAMP/readcot_qsa_test
  [multiarith]=/home/dingxukai/RoT/data/Multiarith/readcot_qsa_test
)
declare -A EXPECTED_TEST_COUNTS=(
  [gsm8k]=1319
  [gsmhard]=1319
  [svamp]=1000
  [multiarith]=180
)

OUT_ROOT=${OUT_ROOT:-${ROOT}/run_outputs/trace_exchangeable/evidence/${label}}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_exchangeable/tmp}
MAX_STARTUP_MEMORY_MIB=${MAX_STARTUP_MEMORY_MIB:-1024}
mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
cd "${ROOT}"

used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
used=${used//[[:space:]]/}
if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > MAX_STARTUP_MEMORY_MIB )); then
  echo "Physical GPU ${gpu} is not clean: ${used:-unknown} MiB already used" >&2
  exit 2
fi

run_eval() {
  local dataset_key=$1
  local rollout_seed=$2
  local test_limit=${3:-1.0}
  local run_name="${dataset_key}_seed${rollout_seed}"
  local out_dir="${OUT_ROOT}/${run_name}"
  mkdir -p "${out_dir}/logs"
  {
    printf 'dataset=%s\n' "${dataset_key}"
    printf 'dataset_dir=%s\n' "${DATASETS[${dataset_key}]}"
    printf 'checkpoint=%s\n' "${checkpoint}"
    printf 'rollout_seed=%s\n' "${rollout_seed}"
    printf 'test_limit=%s\n' "${test_limit}"
    printf 'test_times=1\n'
    printf 'started_at=%s\n' "$(date '+%F %T')"
  } > "${out_dir}/manifest.txt"

  env \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" run.py \
      --model "${MODEL}" \
      --dataset qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /disk1/dingxukai \
      --test_ckpt_path "${checkpoint}" \
      --test_times 1 \
      --seed "${rollout_seed}" \
      dataset_dir="${DATASETS[${dataset_key}]}" \
      batch_size=1 \
      val_batch_size=1 \
      num_workers=2 \
      persistent_workers=false \
      trainer.num_sanity_val_steps=0 \
      trainer.limit_test_batches="${test_limit}" \
      trainer.strategy=auto \
      trainer.default_root_dir="${out_dir}/trainer" \
      trainer.logger.save_dir="${out_dir}/logs" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=true \
      model.model_kwargs.trace_bridge_config.trace_visual_group_views=8 \
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=200 \
      model.model_kwargs.trace_bridge_config.trace_visual_seed="${rollout_seed}" \
      model.model_kwargs.trace_bridge_config.trace_visual_do_sample=true \
      model.model_kwargs.trace_bridge_config.trace_visual_temperature=0.95 \
      model.model_kwargs.trace_bridge_config.trace_visual_top_p=0.97 \
      2>&1 | tee "${out_dir}/eval.log"

  local expected_count="${EXPECTED_TEST_COUNTS[${dataset_key}]}"
  if [[ "${test_limit}" != "1.0" ]]; then
    expected_count="${test_limit}"
  fi
  "${PYTHON}" - "${out_dir}/logs/tb/run" "${expected_count}" "${dataset_key}" <<'PY'
import json
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
dataset_key = sys.argv[3]
files = sorted(log_dir.glob("test_*.json"))
if not files:
    raise SystemExit(
        f"{dataset_key}: no test JSON found in {log_dir}"
    )
result_path = max(files, key=lambda path: (path.stat().st_mtime_ns, str(path)))
payload = json.loads(result_path.read_text(encoding="utf-8"))
numeric_records = [key for key in payload if str(key).isdigit()]
if len(numeric_records) != expected:
    raise SystemExit(
        f"{dataset_key}: expected {expected} evaluated questions, "
        f"found {len(numeric_records)} in {result_path}"
    )
metadata = payload.get("test_metadata", {})
if int(metadata.get("test_times", -1)) != 1:
    raise SystemExit(f"{dataset_key}: test_times is not exactly 1")
print(
    f"VERIFIED_FULL_EVAL dataset={dataset_key} "
    f"questions={len(numeric_records)} test_times=1"
)
PY
  printf 'finished_at=%s\n' "$(date '+%F %T')" >> "${out_dir}/manifest.txt"
}

run_eval gsm8k 271828 1.0
if [[ "${RUN_SECOND_GSM_GEOMETRY_SEED}" == "true" ]]; then
  # This optional pass changes only the sampled visual rollout seed. It never
  # replaces or contributes to the full deterministic GSM8K task benchmark.
  run_eval gsm8k 314159 200
fi
run_eval gsmhard 271828
run_eval svamp 271828
run_eval multiarith 271828

cache_path() {
  local dataset_key=$1
  local rollout_seed=$2
  printf '%s/%s_seed%s/logs/tb/run/trace_exchangeable_visual_test.pt' \
    "${OUT_ROOT}" "${dataset_key}" "${rollout_seed}"
}

expected_caches=(
  "$(cache_path gsm8k 271828)"
  "$(cache_path gsmhard 271828)"
  "$(cache_path svamp 271828)"
  "$(cache_path multiarith 271828)"
)
if [[ "${RUN_SECOND_GSM_GEOMETRY_SEED}" == "true" ]]; then
  expected_caches+=("$(cache_path gsm8k 314159)")
fi
for cache in "${expected_caches[@]}"; do
  if [[ ! -f "${cache}" ]]; then
    echo "Missing expected visual cache: ${cache}" >&2
    exit 1
  fi
done

geometry_records=(
  --record "GSM8K=$(cache_path gsm8k 271828)"
  --record "GSMHard=$(cache_path gsmhard 271828)"
  --record "SVAMP=$(cache_path svamp 271828)"
  --record "MultiArith=$(cache_path multiarith 271828)"
)
if [[ "${RUN_SECOND_GSM_GEOMETRY_SEED}" == "true" ]]; then
  geometry_records+=(--record "GSM8K_seed1=$(cache_path gsm8k 314159)")
fi

"${PYTHON}" tools/trace_exchangeable_geometry_summary.py \
  "${geometry_records[@]}" \
  --out_dir "${OUT_ROOT}/complete_path_geometry" \
  --max_records 200 \
  --label_permutations 128 \
  --bootstrap_trials 10000 \
  --ranking_margin 0.08 \
  --seed 0 \
  2>&1 | tee "${OUT_ROOT}/complete_path_geometry.log"

"${PYTHON}" tools/trace_exchangeable_visualize.py \
  --records "$(cache_path gsm8k 271828)" \
  --out_dir "${OUT_ROOT}/publication_visuals" \
  --max_records 200 \
  --n_representative 3 \
  2>&1 | tee "${OUT_ROOT}/publication_visuals.log"

cat > "${OUT_ROOT}/evidence_done.txt" <<EOF
checkpoint=${checkpoint}
label=${label}
test_times=1
gsm8k_primary_rollout_seed=271828
second_gsm_geometry_seed_enabled=${RUN_SECOND_GSM_GEOMETRY_SEED}
completed_at=$(date '+%F %T')
EOF
