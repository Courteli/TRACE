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
checkpoint=$2
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing checkpoint: ${checkpoint}" >&2
  exit 2
fi
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]] || \
   [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" -ne 4 ]]; then
  echo "The Stage 2 smoke requires exactly four unique GPUs" >&2
  exit 2
fi
for gpu in "${gpu_array[@]}"; do
  used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
  used=${used//[[:space:]]/}
  if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > MAX_STARTUP_MEMORY_MIB )); then
    echo "Physical GPU ${gpu} is not clean: ${used:-unknown} MiB" >&2
    exit 2
  fi
done

RUN_TAG=${RUN_TAG:-20260718_trace_final_stage2_ddp_smoke4}
OUT_ROOT=${OUT_ROOT:-${ROOT}/run_outputs/trace_final/smoke/${RUN_TAG}}
mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
cd "${ROOT}"

env \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${PYTHON}" run.py \
    --model "${MODEL}" \
    --dataset trace_qsa \
    --trainer default \
    --devices 0,1,2,3 \
    --workspace_path /disk1/dingxukai \
    --load_ckpt_path "${checkpoint}" \
    --test_times 1 \
    --seed 0 \
    --disable_early_stopping \
    --log_suffix "${RUN_TAG}" \
    data_module.dataset_dir="${DATASET_DIR}" \
    data_module.tiny_dataset=true \
    batch_size=4 \
    val_batch_size=1 \
    num_workers=0 \
    persistent_workers=false \
    trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0 \
    trainer.log_every_n_steps=1 \
    trainer.max_epochs=1 \
    trainer.max_steps=1 \
    trainer.limit_train_batches=1 \
    trainer.limit_val_batches=0 \
    trainer.enable_progress_bar=false \
    trainer.gradient_clip_val=0 \
    trainer.default_root_dir="${OUT_ROOT}/trainer" \
    save_top_k=0 \
    save_last=false \
    model.model_kwargs.do_trace_rl=true \
    model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
    model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048 \
    model.model_kwargs.trace_rl_config.group_size=8 \
    model.model_kwargs.trace_rl_config.exp_batch_size=1 \
    model.model_kwargs.trace_rl_config.stage2_local_ranking_weight=0.10 \
    model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.05 \
    model.model_kwargs.trace_rl_config.stage2_accuracy_gradient_guard=true \
    model.model_kwargs.trace_rl_config.stage2_ranking_grad_ratio=0.25 \
    model.model_kwargs.trace_rl_config.stage2_ranking_micro_batch_size=1 \
    model.training_kwargs.optimizer.lr=8e-7 \
    model.training_kwargs.scheduler.warmup_steps=0 \
    model.training_kwargs.scheduler.num_training_steps=1 \
    2>&1 | tee "${OUT_ROOT}/smoke.log"

"${PYTHON}" - \
  "${ROOT}/logs/${MODEL}/trace_qsa-gsm" \
  "${RUN_TAG}" \
  "${OUT_ROOT}/smoke_audit.json" <<'PY'
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_root = Path(sys.argv[1])
run_tag = sys.argv[2]
output_path = Path(sys.argv[3])
events = [
    path
    for path in event_root.rglob("events.out.tfevents.*")
    if path.parent.name.endswith(f"_{run_tag}")
]
if not events:
    raise SystemExit(f"No TensorBoard event file found for {run_tag}")
loaded = []
for event in events:
    accumulator = EventAccumulator(
        str(event),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    loaded.append(
        (len(accumulator.Tags().get("scalars", [])), event, accumulator)
    )
_, event, accumulator = max(
    loaded,
    key=lambda item: (item[0], item[1].stat().st_size),
)
tags = set(accumulator.Tags().get("scalars", []))


def values(tag):
    if tag not in tags:
        return []
    return [item.value for item in accumulator.Scalars(tag)]


steps = values("train/optimizer_did_step")
peaks = values("train/cuda_peak_reserved_max_gib")
replay = values("train/stage2_sft_replay_loss")
payload = {
    "event_file": str(event),
    "checks": {
        "optimizer_step_completed": len(steps) == 1 and steps[0] == 1.0,
        "stage1_replay_executed": len(replay) == 1,
        "peak_reserved_below_23_gib": bool(peaks) and max(peaks) < 23.0,
    },
    "observed": {
        "optimizer_steps": steps,
        "stage1_replay_loss": replay,
        "peak_reserved_max_gib": peaks,
    },
    "note": (
        "The clean Stage 0 smoke validates four-rank rollout, PPO, dual-path "
        "replay, synchronization, and memory. Ranking activation is covered by "
        "unit tests because an untrained bottleneck need not produce mixed outcomes."
    ),
}
payload["status"] = (
    "PASS" if all(payload["checks"].values()) else "FAIL"
)
output_path.write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))
if payload["status"] != "PASS":
    raise SystemExit(1)
PY
