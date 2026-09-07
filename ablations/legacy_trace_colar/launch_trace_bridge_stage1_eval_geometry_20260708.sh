#!/usr/bin/env bash
set -euo pipefail

cd /disk1/dingxukai/trace_colar

SESSION="${SESSION:-trace_bridge_bridgefull_stage1_eval_geometry_gpu1_0708}"
GPU="${GPU:-1}"
ROOT="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge"
LOG="${ROOT}/20260708_trace_bridge_bridgefull_stage1_eval_geometry_gpu${GPU}.outer.log"

mkdir -p "${ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "
  set -euo pipefail
  cd /disk1/dingxukai/trace_colar
  echo '[stage1-eval] started at '\$(date '+%F %T')' on GPU=${GPU}' | tee -a ${LOG}

  run_eval() {
    local model=\"\$1\"
    local label=\"\$2\"
    local ckpt=\"\$3\"
    local tag=\"20260708_trace_bridge_bridgefull_stage1eval_\${label}_gpu${GPU}\"
    echo '[stage1-eval] '\${label}' ckpt='\${ckpt}' at '\$(date '+%F %T') | tee -a ${LOG}
    CKPT=\"\${ckpt}\" \
    EVAL_LABEL=\"\${label}\" \
    TRACE_MODEL=\"\${model}\" \
    MODE=eval_trace_from_ckpt \
    GPU=${GPU} \
    RUN_TAG=\"\${tag}\" \
    TEST_TIMES=1 \
    bash run_trace_bridge_pipeline_snapshot_20260707.sh 2>&1 | tee -a ${LOG}
  }

  run_eval trace_bridge_qwen3_instruct_viewteacher viewteacher_stage1_epoch0_m653 /disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_viewteacher/qsa-gsm/20260708-003948_666327_20260708_trace_bridge_bridgefull_viewteacher_stage1_gpu2_stage1/checkpoints/epoch0__step6726__monitor0.653.ckpt
  run_eval trace_bridge_qwen3_instruct_viewteacher viewteacher_cons_stage1_epoch0_m648 /disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_viewteacher/qsa-gsm/20260708-003948_116504_20260708_trace_bridge_bridgefull_viewteacher_stage2_conservative_gpu4_stage1/checkpoints/epoch0__step6726__monitor0.648.ckpt
  run_eval trace_bridge_qwen3_instruct_viewteacher viewteacher_strong_stage1_epoch1_m652 /disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_viewteacher/qsa-gsm/20260708-003948_539428_20260708_trace_bridge_bridgefull_viewteacher_stage2_strong_gpu6_stage1/checkpoints/epoch1__step13452__monitor0.652.ckpt
  run_eval trace_bridge_qwen3_instruct_vizstrong vizstrong_stage1_epoch1_m661 /disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_284260_20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1_stage1/checkpoints/epoch1__step13452__monitor0.661.ckpt
  run_eval trace_bridge_qwen3_instruct_vizstrong vizstrong_cons_stage1_epoch0_m644 /disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_183637_20260708_trace_bridge_bridgefull_vizstrong_stage2_conservative_gpu5_stage1/checkpoints/epoch0__step6726__monitor0.644.ckpt
  run_eval trace_bridge_qwen3_instruct_vizstrong vizstrong_strong_stage1_epoch1_m655 /disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_632116_20260708_trace_bridge_bridgefull_vizstrong_stage2_strong_gpu7_stage1/checkpoints/epoch1__step13452__monitor0.655.ckpt

  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_bridge_collect_results.py \
    --run-regex '20260708_trace_bridge_bridgefull_stage1eval_.*_gpu${GPU}' \
    --out /disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_stage1eval_20260708 \
    2>&1 | tee -a ${LOG}
  echo '[stage1-eval] finished at '\$(date '+%F %T') | tee -a ${LOG}
"

echo "[launched] ${SESSION}: GPU=${GPU}"
