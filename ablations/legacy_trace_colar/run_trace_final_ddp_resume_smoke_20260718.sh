#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_final_qwen3_instruct
DATASET_DIR=${DATASET_DIR:-${ROOT}/run_outputs/trace_final/data/gsm8k_gold_smoke}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_final/tmp}
MAX_STARTUP_MEMORY_MIB=${MAX_STARTUP_MEMORY_MIB:-1024}

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <initial-checkpoint>" >&2
  exit 2
fi
physical_gpus=$1
initial_checkpoint=$2
if [[ ! -f "${initial_checkpoint}" ]]; then
  echo "Missing initial checkpoint: ${initial_checkpoint}" >&2
  exit 2
fi
for required in \
  "${DATASET_DIR}/train.json" \
  "${DATASET_DIR}/val.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing smoke input: ${required}" >&2
    exit 2
  fi
done
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]] || \
   [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "The resume smoke requires exactly four unique GPUs" >&2
  exit 2
fi
for gpu in "${gpu_array[@]}"; do
  used=$(nvidia-smi -i "${gpu}" \
    --query-gpu=memory.used \
    --format=csv,noheader,nounits)
  used=${used//[[:space:]]/}
  if [[ ! "${used}" =~ ^[0-9]+$ ]] || \
     (( used > MAX_STARTUP_MEMORY_MIB )); then
    echo "Physical GPU ${gpu} is not clean: ${used:-unknown} MiB" >&2
    exit 2
  fi
done

SMOKE_ID=${SMOKE_ID:-20260718_trace_final_ddp_fullstate_resume4}
OUT_ROOT=${OUT_ROOT:-${ROOT}/run_outputs/trace_final/smoke/${SMOKE_ID}}
SAVE_TAG=${SMOKE_ID}_save
RESUME_TAG=${SMOKE_ID}_resume
mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
cd "${ROOT}"

common_args=(
  --model "${MODEL}"
  --dataset trace_qsa
  --trainer default
  --devices 0,1,2,3
  --workspace_path /disk1/dingxukai
  --load_ckpt_path "${initial_checkpoint}"
  --test_times 1
  --seed 0
  --disable_early_stopping
  data_module.dataset_dir="${DATASET_DIR}"
  data_module.tiny_dataset=true
  data_module.epoch_scaling=1
  batch_size=4
  val_batch_size=1
  num_workers=0
  persistent_workers=false
  trainer.strategy=ddp_find_unused_parameters_true
  trainer.num_sanity_val_steps=0
  trainer.log_every_n_steps=1
  trainer.max_epochs=1
  trainer.limit_train_batches=1
  trainer.limit_val_batches=0
  trainer.enable_progress_bar=false
  trainer.gradient_clip_val=0
  save_top_k=0
  save_last=true
  save_weights_only=false
  model.model_kwargs.do_trace_rl=true
  model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0
  model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048
  model.model_kwargs.trace_rl_config.group_size=8
  model.model_kwargs.trace_rl_config.exp_batch_size=1
  model.model_kwargs.trace_rl_config.stage2_local_ranking_weight=0.10
  model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.05
  model.model_kwargs.trace_rl_config.stage2_accuracy_gradient_guard=true
  model.model_kwargs.trace_rl_config.stage2_ranking_grad_ratio=0.25
  model.model_kwargs.trace_rl_config.stage2_ranking_micro_batch_size=1
  model.training_kwargs.optimizer.lr=8e-7
  model.training_kwargs.scheduler.warmup_steps=0
  model.training_kwargs.scheduler.num_training_steps=2
)

run_smoke() {
  local tag=$1
  local max_steps=$2
  local root=$3
  shift 3
  env \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TMPDIR="${TMP_ROOT}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${physical_gpus}" \
    "${PYTHON}" run.py \
      "${common_args[@]}" \
      --log_suffix "${tag}" \
      trainer.max_steps="${max_steps}" \
      trainer.default_root_dir="${root}/trainer" \
      "$@"
}

run_smoke \
  "${SAVE_TAG}" \
  1 \
  "${OUT_ROOT}/save" \
  > "${OUT_ROOT}/save.log" 2>&1

find_last_checkpoint() {
  local tag=$1
  find "${ROOT}/logs/${MODEL}/trace_qsa-gsm" \
    -path "*_${tag}/checkpoints/last.ckpt" \
    -type f \
    -printf '%T@ %p\n' |
    sort -n |
    tail -n 1 |
    cut -d' ' -f2-
}

saved_checkpoint=$(find_last_checkpoint "${SAVE_TAG}")
if [[ -z "${saved_checkpoint}" || ! -f "${saved_checkpoint}" ]]; then
  echo "Four-GPU save smoke did not create last.ckpt" >&2
  exit 1
fi

run_smoke \
  "${RESUME_TAG}" \
  2 \
  "${OUT_ROOT}/resume" \
  trainer.max_epochs=2 \
  --resume_ckpt_path "${saved_checkpoint}" \
  > "${OUT_ROOT}/resume.log" 2>&1

resumed_checkpoint=$(find_last_checkpoint "${RESUME_TAG}")
if [[ -z "${resumed_checkpoint}" || ! -f "${resumed_checkpoint}" ]]; then
  echo "Four-GPU resume smoke did not create last.ckpt" >&2
  exit 1
fi

"${PYTHON}" - \
  "${saved_checkpoint}" \
  "${resumed_checkpoint}" \
  "${OUT_ROOT}/smoke_audit.json" <<'PY'
import json
import sys
from pathlib import Path

import torch

save_path = Path(sys.argv[1])
resume_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])


def inspect(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "global_step": int(checkpoint.get("global_step", -1)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "optimizer_states": len(checkpoint.get("optimizer_states", [])),
        "lr_schedulers": len(checkpoint.get("lr_schedulers", [])),
        "has_loops": bool(checkpoint.get("loops")),
        "state_dict_keys": len(checkpoint.get("state_dict", {})),
    }


saved = inspect(save_path)
resumed = inspect(resume_path)
checks = {
    "saved_after_step_1": saved["global_step"] == 1,
    "resumed_after_step_2": resumed["global_step"] == 2,
    "optimizer_state_present": (
        saved["optimizer_states"] == 1
        and resumed["optimizer_states"] == 1
    ),
    "scheduler_state_present": (
        saved["lr_schedulers"] == 1
        and resumed["lr_schedulers"] == 1
    ),
    "loop_state_present": saved["has_loops"] and resumed["has_loops"],
}
payload = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "saved": saved,
    "resumed": resumed,
    "scope": (
        "Four-rank Stage 2 PPO + Stage 1 replay, full-state save, and "
        "full-state continuation from global_step 1 to 2."
    ),
}
output_path.write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
if payload["status"] != "PASS":
    raise SystemExit(1)
PY
