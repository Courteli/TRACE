#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_exchangeable_qwen3_instruct
DATASET_DIR=${DATASET_DIR:-/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_exchangeable/tmp}
MAX_STARTUP_MEMORY_MIB=${MAX_STARTUP_MEMORY_MIB:-1024}

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <physical-gpu> <stage1-formation-checkpoint> <label>" >&2
  exit 2
fi

gpu=$1
checkpoint=$2
label=$3
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing Stage 1 checkpoint: ${checkpoint}" >&2
  exit 2
fi

used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
used=${used//[[:space:]]/}
if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > MAX_STARTUP_MEMORY_MIB )); then
  echo "Physical GPU ${gpu} is not clean: ${used:-unknown} MiB already used" >&2
  exit 2
fi

OUT_ROOT=${OUT_ROOT:-${ROOT}/run_outputs/trace_exchangeable/eligibility/${label}}
mkdir -p "${OUT_ROOT}/logs" "${TMP_ROOT}"
cd "${ROOT}"

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
    --seed 271828 \
    dataset_dir="${DATASET_DIR}" \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=2 \
    persistent_workers=false \
    trainer.num_sanity_val_steps=0 \
    trainer.limit_test_batches=128 \
    trainer.strategy=auto \
    trainer.default_root_dir="${OUT_ROOT}/trainer" \
    trainer.logger.save_dir="${OUT_ROOT}/logs" \
    trainer.logger.name=tb \
    trainer.logger.version=run \
    model.model_kwargs.trace_bridge_config.save_trace_visual_info=true \
    model.model_kwargs.trace_bridge_config.trace_visual_group_views=8 \
    model.model_kwargs.trace_bridge_config.trace_visual_record_limit=128 \
    model.model_kwargs.trace_bridge_config.trace_visual_seed=271828 \
    model.model_kwargs.trace_bridge_config.trace_visual_do_sample=true \
    model.model_kwargs.trace_bridge_config.trace_visual_temperature=0.95 \
    model.model_kwargs.trace_bridge_config.trace_visual_top_p=0.97 \
    2>&1 | tee "${OUT_ROOT}/pilot.log"

cache="${OUT_ROOT}/logs/tb/run/trace_exchangeable_visual_test.pt"
if [[ ! -f "${cache}" ]]; then
  echo "Missing pilot cache: ${cache}" >&2
  exit 1
fi

"${PYTHON}" tools/trace_exchangeable_geometry_summary.py \
  --record "PILOT=${cache}" \
  --out_dir "${OUT_ROOT}/geometry" \
  --max_records 128 \
  --label_permutations 64 \
  --bootstrap_trials 2000 \
  --ranking_margin 0.08 \
  --seed 0 \
  > "${OUT_ROOT}/geometry.log"

"${PYTHON}" tools/trace_exchangeable_eligibility_gate.py \
  --summary "${OUT_ROOT}/geometry/trace_exchangeable_geometry_summary.json" \
  --label PILOT \
  --out "${OUT_ROOT}/eligibility_gate.json" \
  2>&1 | tee "${OUT_ROOT}/eligibility_gate.log"
