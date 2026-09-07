#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-trace_bridge_recovery_waiter_0707}"
LOG="${LOG:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/recovery_waiter_20260707.log}"
SNAPSHOT="${SNAPSHOT:-run_trace_bridge_pipeline_snapshot_20260707.sh}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
set -euo pipefail
cd /disk1/dingxukai/trace_colar

log() { echo \"[\$(date '+%F %T')] \$*\" | tee -a ${LOG}; }

wait_session() {
  local session=\"\$1\"
  while tmux has-session -t \"\${session}\" 2>/dev/null; do sleep 300; done
}

latest_ckpt() {
  local model_name=\"\$1\"
  local suffix=\"\$2\"
  find \"/disk1/dingxukai/trace_colar/logs/\${model_name}/qsa-gsm\" -path \"*\${suffix}*/checkpoints/*.ckpt\" ! -name 'last.ckpt' -print 2>/dev/null | sort | tail -n 1
}

ensure_file_from_ckpt() {
  local out_file=\"\$1\"
  local model_name=\"\$2\"
  local suffix=\"\$3\"
  if [[ -s \"\${out_file}\" ]]; then return 0; fi
  local ckpt
  ckpt=\$(latest_ckpt \"\${model_name}\" \"\${suffix}\")
  if [[ -n \"\${ckpt}\" ]]; then
    mkdir -p \"\$(dirname \"\${out_file}\")\"
    echo \"\${ckpt}\" > \"\${out_file}\"
    log \"recovered ckpt file \${out_file}: \${ckpt}\"
    return 0
  fi
  log \"missing ckpt for suffix=\${suffix} model=\${model_name}\"
  return 1
}

recover_baseline() {
  local run_tag=20260707_trace_bridge_baseline_gpu7
  local out=/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/\${run_tag}
  local best=\${out}/bridge_baseline_best_ckpt.txt
  wait_session trace_bridge_baseline_gpu7_0707
  ensure_file_from_ckpt \"\${best}\" bridge_qwen3_instruct_hybrid_compact_anchor_gate \${run_tag}_bridge_baseline || return 0
  if [[ ! -s \${out}/summary_bridge_baseline.md ]]; then
    log \"recover baseline eval\"
    CKPT=\$(cat \"\${best}\") MODE=eval_bridge_from_ckpt GPU=7 RUN_TAG=\${run_tag} TEST_TIMES=1 EVAL_LABEL=bridge_baseline bash ${SNAPSHOT} 2>&1 | tee -a ${LOG}
  fi
}

recover_stage1_trace() {
  local session=\"\$1\"
  local model=\"\$2\"
  local run_tag=\"\$3\"
  local gpu=\"\$4\"
  local label=\"\$5\"
  local out=/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/\${run_tag}
  local suffix=\${run_tag}_stage1
  local best=\${out}/\${suffix}_best_ckpt.txt
  wait_session \"\${session}\"
  ensure_file_from_ckpt \"\${best}\" \"\${model}\" \"\${suffix}\" || return 0
  if [[ ! -s \${out}/summary_\${label}.md ]]; then
    log \"recover trace eval run_tag=\${run_tag} label=\${label}\"
    CKPT=\$(cat \"\${best}\") TRACE_MODEL=\${model} MODE=eval_trace_from_ckpt GPU=\${gpu} RUN_TAG=\${run_tag} TEST_TIMES=1 EVAL_LABEL=\${label} bash ${SNAPSHOT} 2>&1 | tee -a ${LOG}
  fi
}

recover_stage2_trace() {
  local session=\"\$1\"
  local model=\"\$2\"
  local run_tag=\"\$3\"
  local gpu=\"\$4\"
  local variant=\"\$5\"
  local out=/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/\${run_tag}
  wait_session \"\${session}\"
  if [[ -s \${out}/summary_\${variant}.md ]]; then return 0; fi

  local stage2_best=\${out}/\${variant}_best_ckpt.txt
  if ensure_file_from_ckpt \"\${stage2_best}\" \"\${model}\" \"\${run_tag}_\${variant}\"; then
    log \"recover stage2 eval run_tag=\${run_tag} variant=\${variant}\"
    CKPT=\$(cat \"\${stage2_best}\") TRACE_MODEL=\${model} MODE=eval_trace_from_ckpt GPU=\${gpu} RUN_TAG=\${run_tag} TEST_TIMES=1 EVAL_LABEL=\${variant} bash ${SNAPSHOT} 2>&1 | tee -a ${LOG}
    return 0
  fi

  local stage1_best=\${out}/\${run_tag}_stage1_best_ckpt.txt
  ensure_file_from_ckpt \"\${stage1_best}\" \"\${model}\" \"\${run_tag}_stage1\" || return 0
  log \"recover full stage2 run_tag=\${run_tag} variant=\${variant}\"
  STAGE1_CKPT=\$(cat \"\${stage1_best}\") TRACE_MODEL=\${model} MODE=\${variant}_from_ckpt GPU=\${gpu} RUN_TAG=\${run_tag} TEST_TIMES=1 bash ${SNAPSHOT} 2>&1 | tee -a ${LOG}
}

log 'recovery waiter started'
recover_baseline
recover_stage1_trace trace_bridge_stage1_v2_gpu2_0707 trace_bridge_qwen3_instruct 20260707_trace_bridge_stage1_v2_gpu2 2 trace_stage1
recover_stage2_trace trace_bridge_stage2_conservative_v2_gpu4_0707 trace_bridge_qwen3_instruct 20260707_trace_bridge_stage2_conservative_v2_gpu4 4 stage2_conservative
recover_stage2_trace trace_bridge_stage2_strong_v2_gpu6_0707 trace_bridge_qwen3_instruct 20260707_trace_bridge_stage2_strong_v2_gpu6 6 stage2_strong
wait_session trace_bridge_structmv_stage2_from_stage1_gpu1_0707
recover_stage1_trace trace_bridge_structmv_stage1_gpu1_0707 trace_bridge_qwen3_instruct_structmv 20260707_trace_bridge_structmv_stage1_gpu1 1 trace_stage1
log 'recovery waiter finished'
"

echo "[launched-waiter] ${SESSION}"
tmux ls | grep trace_bridge || true
