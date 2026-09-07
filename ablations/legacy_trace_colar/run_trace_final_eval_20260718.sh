#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_final_qwen3_instruct
GSM8K_DIR=${GSM8K_DIR:-${ROOT}/run_outputs/trace_final/data/gsm8k_multirationale_v1}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_final/tmp}
MAX_STARTUP_MEMORY_MIB=${MAX_STARTUP_MEMORY_MIB:-1024}

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 <physical-gpu> <checkpoint> <dataset-key> <output-dir>" >&2
  exit 2
fi

gpu=$1
checkpoint=$2
dataset_key=$3
out_dir=$4
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing checkpoint: ${checkpoint}" >&2
  exit 2
fi

declare -A DATASETS=(
  [gsm8k]="${GSM8K_DIR}"
  [gsmhard]=/home/dingxukai/RoT/data/GSM8k-Hard/readcot_qsa_test
  [svamp]=/home/dingxukai/RoT/data/SVAMP/readcot_qsa_test
  [multiarith]=/home/dingxukai/RoT/data/Multiarith/readcot_qsa_test
)
declare -A EXPECTED_COUNTS=(
  [gsm8k]=1319
  [gsmhard]=1319
  [svamp]=1000
  [multiarith]=180
)
if [[ -z "${DATASETS[${dataset_key}]+x}" ]]; then
  echo "Unknown dataset key: ${dataset_key}" >&2
  exit 2
fi

LIMIT_TEST_BATCHES=${LIMIT_TEST_BATCHES:-1.0}
VISUAL_RECORD_LIMIT=${VISUAL_RECORD_LIMIT:-0}
VISUAL_GROUP_VIEWS=${VISUAL_GROUP_VIEWS:-8}
EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE:-4}
EVAL_DATA_BATCH_SIZE=${EVAL_DATA_BATCH_SIZE:-1}
EXPECTED_EVAL_QUESTIONS=${EXPECTED_EVAL_QUESTIONS:-}
TRACE_PREFIX_K=${TRACE_PREFIX_K:--1}
TRACE_HIDDEN_DROP_INDEX=${TRACE_HIDDEN_DROP_INDEX:--1}
TRACE_INTERVENTION=${TRACE_INTERVENTION:-none}
EVAL_SEED=${EVAL_SEED:-271828}

used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
used=${used//[[:space:]]/}
if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > MAX_STARTUP_MEMORY_MIB )); then
  echo "Physical GPU ${gpu} is not clean: ${used:-unknown} MiB" >&2
  exit 2
fi

mkdir -p "${out_dir}/logs" "${TMP_ROOT}"
cat > "${out_dir}/manifest.txt" <<EOF
dataset=${dataset_key}
dataset_dir=${DATASETS[${dataset_key}]}
checkpoint=${checkpoint}
test_times=1
seed=${EVAL_SEED}
limit_test_batches=${LIMIT_TEST_BATCHES}
visual_record_limit=${VISUAL_RECORD_LIMIT}
visual_group_views=${VISUAL_GROUP_VIEWS}
eval_micro_batch_size=${EVAL_MICRO_BATCH_SIZE}
eval_data_batch_size=${EVAL_DATA_BATCH_SIZE}
trace_prefix_k=${TRACE_PREFIX_K}
trace_hidden_drop_index=${TRACE_HIDDEN_DROP_INDEX}
trace_intervention=${TRACE_INTERVENTION}
started_at=$(date --iso-8601=seconds)
EOF

cd "${ROOT}"
env \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PYTHON}" run.py \
    --model "${MODEL}" \
    --dataset trace_qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /disk1/dingxukai \
    --test_ckpt_path "${checkpoint}" \
    --test_times 1 \
    --seed "${EVAL_SEED}" \
    data_module.dataset_dir="${DATASETS[${dataset_key}]}" \
    data_module.tiny_dataset=false \
    batch_size=1 \
    val_batch_size="${EVAL_DATA_BATCH_SIZE}" \
    num_workers=2 \
    persistent_workers=false \
    trainer.num_sanity_val_steps=0 \
    trainer.limit_test_batches="${LIMIT_TEST_BATCHES}" \
    trainer.strategy=auto \
    trainer.default_root_dir="${out_dir}/trainer" \
    trainer.logger.save_dir="${out_dir}/logs" \
    trainer.logger.name=tb \
    trainer.logger.version=run \
    model.model_kwargs.trace_bridge_config.save_trace_visual_info=true \
    model.model_kwargs.trace_bridge_config.trace_visual_group_views="${VISUAL_GROUP_VIEWS}" \
    model.model_kwargs.trace_bridge_config.trace_visual_record_limit="${VISUAL_RECORD_LIMIT}" \
    model.model_kwargs.trace_bridge_config.trace_visual_seed="${EVAL_SEED}" \
    model.model_kwargs.trace_bridge_config.trace_visual_do_sample=true \
    model.model_kwargs.trace_bridge_config.trace_visual_temperature=0.95 \
    model.model_kwargs.trace_bridge_config.trace_visual_top_p=0.97 \
    model.model_kwargs.trace_bridge_config.trace_eval_hidden_prefix_k="${TRACE_PREFIX_K}" \
    model.model_kwargs.trace_bridge_config.trace_eval_hidden_drop_index="${TRACE_HIDDEN_DROP_INDEX}" \
    model.model_kwargs.trace_bridge_config.trace_eval_intervention="${TRACE_INTERVENTION}" \
    model.model_kwargs.trace_bridge_config.trace_eval_intervention_seed="${EVAL_SEED}" \
    model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true \
    model.model_kwargs.trace_rl_config.exp_batch_size="${EVAL_MICRO_BATCH_SIZE}" \
    2>&1 | tee "${out_dir}/eval.log"

expected=${EXPECTED_COUNTS[${dataset_key}]}
if [[ -n "${EXPECTED_EVAL_QUESTIONS}" ]]; then
  expected=${EXPECTED_EVAL_QUESTIONS}
elif [[ "${LIMIT_TEST_BATCHES}" != "1.0" ]]; then
  expected=$((LIMIT_TEST_BATCHES * EVAL_DATA_BATCH_SIZE))
  if (( expected > EXPECTED_COUNTS[${dataset_key}] )); then
    expected=${EXPECTED_COUNTS[${dataset_key}]}
  fi
fi

"${PYTHON}" - "${out_dir}/logs/tb/run" "${expected}" "${dataset_key}" \
  "${VISUAL_RECORD_LIMIT}" <<'PY'
import json
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
dataset_key = sys.argv[3]
visual_limit = int(sys.argv[4])
files = sorted(log_dir.glob("test_*.json"))
if not files:
    raise SystemExit(f"{dataset_key}: no test JSON under {log_dir}")
result_path = max(
    files,
    key=lambda path: (path.stat().st_mtime_ns, str(path)),
)
payload = json.loads(result_path.read_text(encoding="utf-8"))
numeric = [key for key in payload if str(key).isdigit()]
if len(numeric) != expected:
    raise SystemExit(
        f"{dataset_key}: expected {expected} records, found {len(numeric)}"
    )
metadata = payload.get("test_metadata", {})
if int(metadata.get("test_times", -1)) != 1:
    raise SystemExit(f"{dataset_key}: test_times must be exactly 1")
cache = log_dir / "trace_final_visual_test.pt"
if visual_limit > 0 and not cache.is_file():
    raise SystemExit(f"{dataset_key}: missing visual cache {cache}")
print(
    f"VERIFIED_TRACE_FINAL_EVAL dataset={dataset_key} "
    f"questions={len(numeric)} test_times=1 visual={visual_limit}"
)
print(f"RESULT_JSON={result_path}")
PY

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "${out_dir}/manifest.txt"
