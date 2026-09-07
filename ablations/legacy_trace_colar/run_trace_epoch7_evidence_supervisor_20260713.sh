#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
SESSION=trace_epoch7_solid_evidence_gpu5_0713
DONE=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/solid_evidence_queue_done.txt

while [[ ! -f "${DONE}" ]]; do
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    while true; do
      used="$(nvidia-smi --id=5 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      if (( used < 500 )); then
        break
      fi
      sleep 60
    done
    tmux new-session -d -s "${SESSION}" \
      "cd ${ROOT} && GPU=5 bash run_trace_epoch7_solid_evidence_20260713.sh"
    printf '%s restarted %s\n' "$(date '+%F %T')" "${SESSION}" \
      >> "${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/supervisor.log"
  fi
  sleep 60
done
