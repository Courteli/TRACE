#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_exchangeable_qwen3_instruct
DATASET_DIR=${DATASET_DIR:-/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc}
TMP_ROOT=${TMP_ROOT:-${ROOT}/run_roots/trace_exchangeable/tmp}
MAX_STARTUP_MEMORY_MIB=${MAX_STARTUP_MEMORY_MIB:-1024}

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <four-physical-gpu-csv> <stage1-formation-checkpoint>" >&2
  exit 2
fi

physical_gpus=$1
checkpoint=$2
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing Stage 1 checkpoint: ${checkpoint}" >&2
  exit 2
fi
IFS=',' read -r -a gpu_array <<< "${physical_gpus}"
if [[ "${#gpu_array[@]}" -ne 4 ]]; then
  echo "The DDP smoke requires exactly four GPUs" >&2
  exit 2
fi
unique_gpu_count=$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)
if [[ "${unique_gpu_count}" -ne 4 ]]; then
  echo "Physical GPU IDs must be unique: ${physical_gpus}" >&2
  exit 2
fi
for gpu in "${gpu_array[@]}"; do
  used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
  used=${used//[[:space:]]/}
  if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > MAX_STARTUP_MEMORY_MIB )); then
    echo "Physical GPU ${gpu} is not clean: ${used:-unknown} MiB already used" >&2
    exit 2
  fi
done

RUN_TAG=${RUN_TAG:-20260718_trace_exchangeable_ddp_smoke4}
OUT_ROOT=${OUT_ROOT:-${ROOT}/run_outputs/trace_exchangeable/${RUN_TAG}}
mkdir -p "${OUT_ROOT}" "${TMP_ROOT}"
cd "${ROOT}"

env \
  TMPDIR="${TMP_ROOT}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${physical_gpus}" \
  "${PYTHON}" run.py \
    --model "${MODEL}" \
    --dataset qsa \
    --trainer default \
    --devices 0,1,2,3 \
    --workspace_path /disk1/dingxukai \
    --load_ckpt_path "${checkpoint}" \
    --test_times 1 \
    --disable_early_stopping \
    --log_suffix "${RUN_TAG}" \
    dataset_dir="${DATASET_DIR}" \
    batch_size=4 \
    val_batch_size=1 \
    num_workers=0 \
    persistent_workers=false \
    trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0 \
    trainer.log_every_n_steps=1 \
    trainer.max_epochs=1 \
    trainer.max_steps=4 \
    trainer.limit_train_batches=4 \
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
    model.training_kwargs.scheduler.num_training_steps=4 \
    2>&1 | tee "${OUT_ROOT}/smoke.log"

"${PYTHON}" - \
  "logs/trace_exchangeable_qwen3_instruct/qsa-gsm" \
  "${RUN_TAG}" \
  "${OUT_ROOT}/smoke_audit.json" <<'PY'
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_root = Path(sys.argv[1])
run_tag = sys.argv[2]
output_path = Path(sys.argv[3])
candidates = [
    path
    for path in event_root.rglob("events.out.tfevents.*")
    if path.parent.name.endswith(f"_{run_tag}")
]
if not candidates:
    raise SystemExit(f"No TensorBoard event file found for {run_tag}")
loaded = []
for event_file in candidates:
    candidate = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    candidate.Reload()
    loaded.append((len(candidate.Tags().get("scalars", [])), event_file, candidate))
_, event_file, accumulator = max(loaded, key=lambda item: (item[0], item[1].stat().st_size))
tags = set(accumulator.Tags().get("scalars", []))


def values(tag):
    return [item.value for item in accumulator.Scalars(tag)] if tag in tags else []


eligible = values("train/stage2_rank_eligible_groups")
active = values("train/stage2_rank_active_triplet_count")
ranking_grad = values("train/stage2_guard_ranking_grad_norm")
optimizer_steps = values("train/optimizer_did_step")
peak_reserved = values("train/cuda_peak_reserved_max_gib")
payload = {
    "event_file": str(event_file),
    "checks": {
        "four_optimizer_steps": len(optimizer_steps) == 4 and min(optimizer_steps) == 1.0,
        "mixed_group_observed": bool(eligible) and max(eligible) > 0.0,
        "active_hinge_observed": bool(active) and max(active) > 0.0,
        "ranking_gradient_observed": bool(ranking_grad) and max(ranking_grad) > 0.0,
        "peak_reserved_below_23_gib": bool(peak_reserved) and max(peak_reserved) < 23.0,
    },
    "observed": {
        "eligible_groups": eligible,
        "active_triplets": active,
        "ranking_grad_norm": ranking_grad,
        "optimizer_steps": optimizer_steps,
        "peak_reserved_max_gib": peak_reserved,
    },
}
payload["status"] = "PASS" if all(payload["checks"].values()) else "FAIL"
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
if payload["status"] != "PASS":
    raise SystemExit(1)
PY
