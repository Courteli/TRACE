#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

SESSION="${SESSION:-trace_bridge_final_full_snapshot_eval_after_0707}"
ROOT="${ROOT:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge}"
AUDIT_OUT="${AUDIT_OUT:-${ROOT}/result_audit_final_full_20260707}"
LOG="${LOG:-${AUDIT_OUT}/snapshot_eval_after.log}"
RUN_REGEX="${RUN_REGEX:-20260707_trace_bridge_final_full_(bridge_baseline_gpu7|(structmv|viewteacher|vizstrong)_(stage1|stage2_conservative|stage2_strong)_gpu(1|2|4|5|6|7))}"
SNAPSHOT_EVAL_MAX_PER_RUN="${SNAPSHOT_EVAL_MAX_PER_RUN:-12}"
WATCH_SESSIONS="${WATCH_SESSIONS:-trace_bridge_final_full_structmv_stage1_gpu2_0707 trace_bridge_final_full_structmv_stage2_conservative_gpu4_0707 trace_bridge_final_full_structmv_stage2_strong_gpu6_0707 trace_bridge_final_full_viewteacher_stage1_gpu7_0707 trace_bridge_final_full_viewteacher_stage2_conservative_gpu5_0707 trace_bridge_final_full_viewteacher_stage2_strong_gpu1_0707 trace_bridge_final_full_bridge_baseline_gpu7_0707 trace_bridge_final_full_vizstrong_stage1_gpu2_0707 trace_bridge_final_full_ckpt_snapshot_watch_0707}"
MANIFEST_GLOB="${MANIFEST_GLOB:-${ROOT}/20260707_trace_bridge_final_full_*/manifest.txt}"

mkdir -p "${AUDIT_OUT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  set -euo pipefail
  cd /disk1/dingxukai/trace_colar
  echo '[snapshot-eval] waiting for training/snapshot sessions at '\$(date '+%F %T') | tee -a ${LOG}
  while true; do
    alive=0
    for s in ${WATCH_SESSIONS}; do
      if tmux has-session -t \${s} 2>/dev/null; then alive=1; fi
    done
    if [[ \${alive} -eq 0 ]]; then break; fi
    sleep 600
  done

  echo '[snapshot-eval] starting snapshot evaluations at '\$(date '+%F %T') | tee -a ${LOG}
  for manifest in ${MANIFEST_GLOB}; do
    [[ -e \"\${manifest}\" ]] || continue
    run_dir=\$(dirname \"\${manifest}\")
    run_tag=\$(awk -F= '\$1==\"run_tag\"{print \$2}' \"\${manifest}\")
    gpu=\$(awk -F= '\$1==\"gpu\"{print \$2}' \"\${manifest}\")
    trace_model=\$(awk -F= '\$1==\"trace_model\"{print \$2}' \"\${manifest}\")
    if [[ -z \"\${run_tag}\" || -z \"\${gpu}\" ]]; then continue; fi
    if [[ ! \"\${run_tag}\" =~ ${RUN_REGEX} ]]; then continue; fi
    if [[ ! -d \"\${run_dir}/ckpt_snapshots\" ]]; then continue; fi
    if [[ \"\${trace_model}\" == bridge_* ]]; then
      eval_mode=eval_bridge_from_ckpt
    else
      eval_mode=eval_trace_from_ckpt
    fi
    mapfile -t ckpts < <(find \"\${run_dir}/ckpt_snapshots\" -path '*/checkpoints/*.ckpt' -printf '%T@ %p\n' | sort -n | tail -n ${SNAPSHOT_EVAL_MAX_PER_RUN} | cut -d' ' -f2-)
    for ckpt in \"\${ckpts[@]}\"; do
      base=\$(basename \"\${ckpt}\" .ckpt | tr -c '[:alnum:]_' '_')
      label=\"snapshot_\${base}\"
      if [[ -s \"\${run_dir}/summary_\${label}.md\" ]]; then
        echo \"[snapshot-eval] skip existing \${run_tag} \${label}\" | tee -a ${LOG}
        continue
      fi
      echo \"[snapshot-eval] run=\${run_tag} gpu=\${gpu} label=\${label} ckpt=\${ckpt}\" | tee -a ${LOG}
      CKPT=\"\${ckpt}\" MODE=\"\${eval_mode}\" GPU=\"\${gpu}\" RUN_TAG=\"\${run_tag}\" TRACE_MODEL=\"\${trace_model}\" EVAL_LABEL=\"\${label}\" TEST_TIMES=1 bash run_trace_bridge_pipeline_snapshot_20260707.sh 2>&1 | tee -a ${LOG}
      /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_bridge_collect_results.py --run-regex '${RUN_REGEX}' --out ${AUDIT_OUT} >/dev/null 2>&1 || true
    done
  done
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_bridge_collect_results.py --run-regex '${RUN_REGEX}' --out ${AUDIT_OUT} 2>&1 | tee -a ${LOG}
  echo '[snapshot-eval] finished at '\$(date '+%F %T') | tee -a ${LOG}
"

echo "[launched] ${SESSION}"
tmux ls | grep trace_bridge_final || true
