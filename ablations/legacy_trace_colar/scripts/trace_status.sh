#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
RUN_OUTPUTS="${ROOT}/run_outputs/trace"

echo "== tmux TRACE sessions =="
tmux ls 2>/dev/null | grep -E '^trace_' || true

echo
echo "== recent TRACE logs =="
if [[ -d "${RUN_OUTPUTS}" ]]; then
  find "${RUN_OUTPUTS}" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' \
    | sort -nr \
    | head -5 \
    | while read -r _ path; do
        echo "--- ${path}"
        tail -20 "${path}" || true
      done
else
  echo "No TRACE run_outputs directory yet: ${RUN_OUTPUTS}"
fi

echo
echo "== newest TRACE checkpoints =="
if [[ -d "${ROOT}/logs/trace_colar_qwen3_instruct" ]]; then
  find "${ROOT}/logs/trace_colar_qwen3_instruct" -path '*checkpoints/*.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -10 \
    | cut -d' ' -f2-
else
  echo "No TRACE checkpoint directory yet."
fi

echo
echo "== TRACE gate report =="
TRACE_PY="${TRACE_PY:-/home/dingxukai/miniconda3/envs/ROT/bin/python}"
if [[ -f "${ROOT}/tools/trace_gate_report.py" && -x "${TRACE_PY}" ]]; then
  "${TRACE_PY}" "${ROOT}/tools/trace_gate_report.py" \
    --latest "${TRACE_STATUS_LATEST:-8}" \
    --json_out "${RUN_OUTPUTS}/gate_report_latest.json" || true
else
  echo "No TRACE gate report tool or Python runtime yet."
fi

echo
echo "== TRACE evidence gaps =="
if [[ -f "${ROOT}/tools/trace_evidence_snapshot.py" && -x "${TRACE_PY}" ]]; then
  "${TRACE_PY}" "${ROOT}/tools/trace_evidence_snapshot.py" --include_train --latest_per_root 30 >/tmp/trace_status_snapshot.out || true
  cat /tmp/trace_status_snapshot.out || true
  if [[ -f "${RUN_OUTPUTS}/evidence_snapshot.md" ]]; then
    awk '/^## Missing Evidence/{flag=1; next} /^## /{flag=0} flag {print}' "${RUN_OUTPUTS}/evidence_snapshot.md"
  fi
else
  echo "No TRACE evidence snapshot tool or Python runtime yet."
fi
