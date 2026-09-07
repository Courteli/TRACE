#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
SESSION="${SESSION:?Set SESSION to the tmux session to stop.}"
DATASET="${DATASET:?Set DATASET to the completed dataset name to wait for.}"
RUN_DIR="${RUN_DIR:-${ROOT}/logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260704-044959_549439_trace_v2_rl_qwen3_c5_g8_gpu7}"
MIN_ITEMS="${MIN_ITEMS:-1}"
MIN_TIMES="${MIN_TIMES:-5}"
CKPT_CONTAINS="${CKPT_CONTAINS:-}"
INTERVAL="${INTERVAL:-300}"
LOG="${LOG:-${ROOT}/run_outputs/trace/trace_stop_${SESSION}_after_${DATASET}.log}"

cd "${ROOT}"
mkdir -p "$(dirname "${LOG}")"

echo "[TRACE stop-after-dataset] watching session=${SESSION} dataset=${DATASET}" | tee -a "${LOG}"

while tmux has-session -t "${SESSION}" 2>/dev/null; do
  if python - "${RUN_DIR}" "${DATASET}" "${MIN_ITEMS}" "${MIN_TIMES}" "${CKPT_CONTAINS}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "tools")
from trace_summarize_runs import summarize_json

run_dir = Path(sys.argv[1])
dataset = sys.argv[2]
min_items = int(sys.argv[3])
min_times = int(sys.argv[4])
ckpt_contains = sys.argv[5]

for path in sorted(run_dir.glob("test_*.json"), key=lambda p: p.stat().st_mtime):
    try:
        row = summarize_json(path)
    except Exception:
        continue
    times = row.get("effective_test_times") or row.get("test_times") or 0
    ckpt_path = row.get("ckpt_path") or ""
    if ckpt_contains and ckpt_contains not in ckpt_path:
        continue
    if row.get("dataset") == dataset and times >= min_times and (row.get("n_items") or 0) >= min_items:
        print(f"{path} dataset={dataset} times={times} n_items={row.get('n_items')} acc={row.get('acc')} ckpt={ckpt_path}")
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    echo "[TRACE stop-after-dataset] completed ${DATASET}; stopping ${SESSION}" | tee -a "${LOG}"
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    exit 0
  fi
  sleep "${INTERVAL}"
done

echo "[TRACE stop-after-dataset] ${SESSION} is no longer active" | tee -a "${LOG}"
