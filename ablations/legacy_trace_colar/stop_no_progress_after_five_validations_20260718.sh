#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
PY=/home/dingxukai/miniconda3/envs/ROT/bin/python
OUT=${ROOT}/run_outputs/trace/20260716_stage1_component_ablations_gpu0
VARIANT_OUT=${OUT}/no_progress_anchor
LOGGER=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260717-111715_987629
PIPELINE=${ROOT}/run_trace_stage1_component_ablations_gpu0_20260716.sh
TARGET_VALIDATIONS=5
POLL_SECONDS=60

mkdir -p "${VARIANT_OUT}"
exec 9>"${VARIANT_OUT}/budget5_watcher.lock"
if ! flock -n 9; then
  printf '[skip] budget-five watcher is already active\n'
  exit 0
fi

timestamp() {
  date '+%F %T'
}

validation_count() {
  "${PY}" - "${LOGGER}" <<'PY'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
ea.Reload()
print(len(ea.Scalars("val/acc")))
PY
}

find_best_checkpoint() {
  "${PY}" - "${LOGGER}/checkpoints" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in root.glob("epoch*.ckpt"):
    match = re.search(r"epoch(\d+).*monitor(-?\d+(?:\.\d+)?)\.ckpt$", path.name)
    if match:
        rows.append((float(match.group(2)), int(match.group(1)), path))
if not rows:
    raise SystemExit("no monitored checkpoint found")
print(max(rows)[2])
PY
}

while true; do
  count="$(validation_count 2>/dev/null || printf '0')"
  printf '%s [watch] completed_validations=%s target=%s\n' \
    "$(timestamp)" "${count}" "${TARGET_VALIDATIONS}" \
    >> "${VARIANT_OUT}/budget5_watcher.log"
  if (( count >= TARGET_VALIDATIONS )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

# Give ModelCheckpoint time to finish after the validation scalar is flushed.
sleep 90

train_pid="$(
  pgrep -fo 'python run.py .*20260716_stage1_component_ablations_gpu0/no_progress_anchor/train' \
    || true
)"
parent_pid=""
if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" 2>/dev/null; then
  parent_pid="$(ps -o ppid= -p "${train_pid}" | tr -d ' ')"
fi

# Stop the old queue first so it cannot launch no_multiview with the former
# ten-epoch budget, then ask Lightning to exit at its next safe boundary.
if [[ -n "${parent_pid}" ]] && kill -0 "${parent_pid}" 2>/dev/null; then
  kill -TERM "${parent_pid}"
fi
if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" 2>/dev/null; then
  kill -TERM "${train_pid}"
fi

for _ in $(seq 1 120); do
  if [[ -z "${train_pid}" ]] || ! kill -0 "${train_pid}" 2>/dev/null; then
    break
  fi
  sleep 5
done
if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" 2>/dev/null; then
  printf '%s [error] training process did not exit after SIGTERM pid=%s\n' \
    "$(timestamp)" "${train_pid}" >> "${VARIANT_OUT}/budget5_watcher.log"
  exit 1
fi

best="$(find_best_checkpoint)"
printf '%s\n' "${best}" > "${VARIANT_OUT}/best_checkpoint.txt"
{
  printf '%s [capped] completed_validations=%s best=%s\n' \
    "$(timestamp)" "${TARGET_VALIDATIONS}" "$(basename "${best}")"
  printf '%s [relaunch] stage1_max_epochs=5\n' "$(timestamp)"
} >> "${VARIANT_OUT}/budget5_watcher.log"

while pgrep -f 'bash run_trace_stage1_component_ablations_gpu0_20260716.sh' \
  >/dev/null 2>&1; do
  sleep 5
done

nohup env STAGE1_MAX_EPOCHS=5 GPU=0 \
  bash "${PIPELINE}" \
  > "${OUT}/relaunch_budget5.log" 2>&1 &
printf '%s\n' "$!" > "${OUT}/relaunch_budget5.pid"
printf '%s [relaunched] pid=%s\n' "$(timestamp)" "$!" \
  >> "${VARIANT_OUT}/budget5_watcher.log"
