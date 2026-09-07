#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
MODEL=trace_bridge_qwen3_instruct_vizstrong
DATASET=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc
STAGE1_CKPT=/disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_284260_20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1_stage1/checkpoints/epoch1__step13452__monitor0.661.ckpt
BRIDGE_CKPT=/disk1/dingxukai/trace_colar/logs/bridge_qwen3_instruct_hybrid_compact_anchor_gate/qsa-gsm/20260708-153949_757553_20260708_trace_bridge_bridgefull_bridge_baseline_gpu2_bridge_baseline/checkpoints/epoch0__step6726__monitor0.655.ckpt
BASELINE_SUMMARY=${ROOT}/run_outputs/trace_bridge/20260708_trace_bridge_bridgefull_bridge_baseline_gpu2/summary_snapshot_epoch0__step6726__monitor0_655__mtime1783503244088392417__size674673616_.md

TRACE_FINAL_SESSION=${TRACE_FINAL_SESSION:-trace_final_guarded_gpu0123_0711}
ANSWER_FINAL_SESSION=${ANSWER_FINAL_SESSION:-trace_final_vizstrong_answer_gpu457_0710}
TRACE_EVENT_PATTERN=${TRACE_EVENT_PATTERN:-'*20260711_trace_bridge_final_guarded_gpu0123_stage2_trace_guarded'}
ANSWER_EVENT_PATTERN=${ANSWER_EVENT_PATTERN:-'*20260710_trace_bridge_final_vizstrong_answeronly_gpu457_stage2_answer_only'}
TRACE_SNAPSHOTS=${TRACE_SNAPSHOTS:-${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_guarded_epoch_snapshots/trace}
ANSWER_SNAPSHOTS=${ANSWER_SNAPSHOTS:-${ROOT}/run_outputs/trace_bridge/20260710_trace_bridge_epoch_snapshots/answer_only}
TRACE_VAL_OUT=${TRACE_VAL_OUT:-${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_final_guarded_gpu0123}
ANSWER_VAL_OUT=${ANSWER_VAL_OUT:-${ROOT}/run_outputs/trace_bridge/20260710_trace_bridge_final_vizstrong_answeronly_gpu457}

SWEEP_OUT=${SWEEP_OUT:-${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_epoch_test_sweep}
TRACE_TEST_OUT=${TRACE_TEST_OUT:-${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_guarded_testbest_eval_gpu0}
ANSWER_TEST_OUT=${ANSWER_TEST_OUT:-${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_answer_testbest_eval_gpu4}
STAGE1_OUT=${STAGE1_OUT:-${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_stage1_rollout_audit_gpu1}
BRIDGE_OUT=${BRIDGE_OUT:-${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_bridge_rollout_audit_gpu2}
AUDIT_OUT=${AUDIT_OUT:-${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_final_goal_audit}
AUDIT_GPUS=${AUDIT_GPUS:-0,1,2,3,4,5,7}
TRACE_EVAL_GPU=${TRACE_EVAL_GPU:-0}
ANSWER_EVAL_GPU=${ANSWER_EVAL_GPU:-4}
STAGE1_EVAL_GPU=${STAGE1_EVAL_GPU:-1}
BRIDGE_EVAL_GPU=${BRIDGE_EVAL_GPU:-2}

write_failure_report() {
  local status=$?
  mkdir -p "${AUDIT_OUT}"
  printf 'Guarded TRACE candidate audit failed with exit status %s at %s\n' \
    "${status}" "$(date '+%F %T')" > "${AUDIT_OUT}/watcher_failed.txt"
  exit "${status}"
}
trap write_failure_report ERR

mkdir -p "${SWEEP_OUT}" "${AUDIT_OUT}"
cd "${ROOT}"

while tmux has-session -t "${TRACE_FINAL_SESSION}" 2>/dev/null \
  || tmux has-session -t "${ANSWER_FINAL_SESSION}" 2>/dev/null; do
  sleep 300
done

mapfile -t trace_ckpts < <(find "${TRACE_SNAPSHOTS}" -maxdepth 1 -type f -name 'trace_epoch*_step*.ckpt' -print | sort -V)
mapfile -t answer_ckpts < <(find "${ANSWER_SNAPSHOTS}" -maxdepth 1 -type f -name 'answer_only_epoch*_step*.ckpt' -print | sort -V)

if [[ "${#trace_ckpts[@]}" -ne 10 || "${#answer_ckpts[@]}" -ne 10 ]]; then
  printf 'Incomplete epoch snapshots; expected 10 each: trace=%s answer=%s\n' "${#trace_ckpts[@]}" "${#answer_ckpts[@]}" \
    > "${AUDIT_OUT}/watcher_failed.txt"
  exit 1
fi

printf 'trace_snapshots=%s\nanswer_snapshots=%s\n' "${#trace_ckpts[@]}" "${#answer_ckpts[@]}" \
  > "${SWEEP_OUT}/snapshot_counts.txt"

run_gsm8k_candidate() {
  local variant="$1"
  local ckpt="$2"
  local gpu="$3"
  local stem
  stem="$(basename "${ckpt}" .ckpt)"
  local out_dir="${SWEEP_OUT}/${variant}/${stem}"
  local log_file="${out_dir}/eval.log"
  mkdir -p "${out_dir}"
  printf '%s\n' "${ckpt}" > "${out_dir}/ckpt.txt"

  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" run.py \
    --model "${MODEL}" \
    --dataset qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /home/dingxukai \
    --test_ckpt_path "${ckpt}" \
    --test_times 1 \
    dataset_dir="${DATASET}" \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=2 \
    persistent_workers=false \
    trainer.num_sanity_val_steps=0 \
    trainer.default_root_dir="${out_dir}/root" \
    trainer.logger.save_dir="${out_dir}/logs" \
    trainer.logger.name=tb \
    trainer.logger.version=run \
    model.model_kwargs.trace_bridge_config.save_trace_visual_info=false \
    model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
    > "${log_file}" 2>&1
}

labels=()
ckpts=()
for ckpt in "${trace_ckpts[@]}"; do
  labels+=(trace)
  ckpts+=("${ckpt}")
done
for ckpt in "${answer_ckpts[@]}"; do
  labels+=(answer_only)
  ckpts+=("${ckpt}")
done

IFS=',' read -r -a gpus <<< "${AUDIT_GPUS}"
pids=()
for i in "${!ckpts[@]}"; do
  slot=$((i % ${#gpus[@]}))
  if [[ -n "${pids[slot]:-}" ]]; then
    wait "${pids[slot]}"
  fi
  run_gsm8k_candidate "${labels[i]}" "${ckpts[i]}" "${gpus[slot]}" &
  pids[slot]=$!
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON}" - "${SWEEP_OUT}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for variant in ("trace", "answer_only"):
    for log in sorted((root / variant).glob("*/eval.log")):
        text = log.read_text(errors="ignore")
        values = {}
        for match in re.finditer(r"│\s*(test/[^│]+?)\s*│\s*([-0-9.]+)\s*│", text):
            values[match.group(1).strip()] = float(match.group(2))
        if "test/acc" not in values:
            raise RuntimeError(f"No test metrics in {log}")
        checkpoint = (log.parent / "ckpt.txt").read_text().strip()
        epoch_match = re.search(r"epoch(\d+)", Path(checkpoint).name)
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        latent = values.get("test/n_latent_forward", 0.0)
        output_length = values.get("test/output_length", 0.0)
        rows.append({
            "variant": variant,
            "epoch": epoch,
            "checkpoint": checkpoint,
            "accuracy": values["test/acc"] * 100.0,
            "n_latent": latent,
            "output_length": output_length,
            "L": latent + output_length,
        })

rows.sort(key=lambda row: (row["variant"], row["epoch"]))
with (root / "results.tsv").open("w", encoding="utf-8") as handle:
    handle.write("variant\tepoch\taccuracy\tn_latent\toutput_length\tL\tcheckpoint\n")
    for row in rows:
        handle.write(
            f'{row["variant"]}\t{row["epoch"]}\t{row["accuracy"]:.6f}\t'
            f'{row["n_latent"]:.6f}\t{row["output_length"]:.6f}\t{row["L"]:.6f}\t'
            f'{row["checkpoint"]}\n'
        )

best = {}
for variant in ("trace", "answer_only"):
    candidates = [row for row in rows if row["variant"] == variant]
    best[variant] = max(candidates, key=lambda row: (row["accuracy"], -row["L"], -row["epoch"]))
    (root / f"best_{variant}_ckpt.txt").write_text(best[variant]["checkpoint"] + "\n")
(root / "best_candidates.json").write_text(json.dumps(best, indent=2) + "\n")
PY

TRACE_EVENT="$(find "${ROOT}/logs/${MODEL}/qsa-gsm" -maxdepth 1 -type d \
  -name "${TRACE_EVENT_PATTERN}" -print | sort | tail -n 1)"
ANSWER_EVENT="$(find "${ROOT}/logs/${MODEL}/qsa-gsm" -maxdepth 1 -type d \
  -name "${ANSWER_EVENT_PATTERN}" -print | sort | tail -n 1)"
"${PYTHON}" tools/trace_bridge_training_dynamics.py \
  --trace_event "${TRACE_EVENT}" \
  --answer_event "${ANSWER_EVENT}" \
  --sweep_results "${SWEEP_OUT}/results.tsv" \
  --out_dir "${AUDIT_OUT}/training_dynamics" \
  --stage1_monitor 0.661

TRACE_TEST_CKPT="$(cat "${SWEEP_OUT}/best_trace_ckpt.txt")"
ANSWER_TEST_CKPT="$(cat "${SWEEP_OUT}/best_answer_only_ckpt.txt")"
TRACE_VAL_CKPT="$(cat "${TRACE_VAL_OUT}/stage2_trace_guarded_best_ckpt.txt")"
ANSWER_VAL_CKPT="$(cat "${ANSWER_VAL_OUT}/stage2_answer_only_best_ckpt.txt")"

"${PYTHON}" - "${AUDIT_OUT}/checkpoint_provenance.json" <<PY
import json
import sys

payload = {
    "primary_protocol": "validation-best monitored checkpoint; one deterministic test per dataset",
    "diagnostic_protocol": "every epoch snapshot tested once on GSM8K; test-best reported separately",
    "stage1_checkpoint": "${STAGE1_CKPT}",
    "bridge_checkpoint": "${BRIDGE_CKPT}",
    "validation_best": {
        "trace": "${TRACE_VAL_CKPT}",
        "answer_only": "${ANSWER_VAL_CKPT}",
    },
    "test_best": {
        "trace": "${TRACE_TEST_CKPT}",
        "answer_only": "${ANSWER_TEST_CKPT}",
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\\n")
PY

env MODE=eval_trace_from_ckpt \
  RUN_TAG=20260711_trace_bridge_guarded_testbest_eval_gpu0 \
  OUT_DIR="${TRACE_TEST_OUT}" ROOT_DIR="${TRACE_TEST_OUT}/roots" \
  TRACE_MODEL="${MODEL}" GPU="${TRACE_EVAL_GPU}" CKPT="${TRACE_TEST_CKPT}" \
  EVAL_LABEL=stage2_trace_guarded_testbest TEST_TIMES=1 \
  bash run_trace_bridge_pipeline_snapshot_20260707.sh \
  > "${SWEEP_OUT}/trace_testbest_full_eval.log" 2>&1 &
trace_eval_pid=$!

env MODE=eval_trace_from_ckpt \
  RUN_TAG=20260711_trace_bridge_answer_testbest_eval_gpu4 \
  OUT_DIR="${ANSWER_TEST_OUT}" ROOT_DIR="${ANSWER_TEST_OUT}/roots" \
  TRACE_MODEL="${MODEL}" GPU="${ANSWER_EVAL_GPU}" CKPT="${ANSWER_TEST_CKPT}" \
  EVAL_LABEL=stage2_answer_only_testbest TEST_TIMES=1 \
  bash run_trace_bridge_pipeline_snapshot_20260707.sh \
  > "${SWEEP_OUT}/answer_testbest_full_eval.log" 2>&1 &
answer_eval_pid=$!

env MODE=eval_trace_from_ckpt \
  RUN_TAG=20260711_trace_bridge_stage1_rollout_audit_gpu1 \
  OUT_DIR="${STAGE1_OUT}" ROOT_DIR="${STAGE1_OUT}/roots" \
  TRACE_MODEL="${MODEL}" GPU="${STAGE1_EVAL_GPU}" CKPT="${STAGE1_CKPT}" \
  EVAL_LABEL=stage1_rollout_baseline TEST_TIMES=1 \
  bash run_trace_bridge_pipeline_snapshot_20260707.sh \
  > "${SWEEP_OUT}/stage1_full_eval.log" 2>&1 &
stage1_eval_pid=$!

wait "${trace_eval_pid}"
wait "${answer_eval_pid}"
wait "${stage1_eval_pid}"

env MODE=eval_trace_from_ckpt \
  RUN_TAG=20260711_trace_bridge_bridge_rollout_audit_gpu2 \
  OUT_DIR="${BRIDGE_OUT}" ROOT_DIR="${BRIDGE_OUT}/roots" \
  TRACE_MODEL="${MODEL}" GPU="${BRIDGE_EVAL_GPU}" CKPT="${BRIDGE_CKPT}" \
  EVAL_LABEL=bridge_rollout_baseline TEST_TIMES=1 \
  TRACE_EVAL_DISABLE_VIEW_EMBEDDINGS=true \
  bash run_trace_bridge_pipeline_snapshot_20260707.sh \
  > "${SWEEP_OUT}/bridge_full_eval.log" 2>&1

BRIDGE_RECORD=${BRIDGE_OUT}/eval_bridge_rollout_baseline_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
BRIDGE_GEOMETRY=${BRIDGE_OUT}/visual_bridge_rollout_baseline_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json
STAGE1_RECORD=${STAGE1_OUT}/eval_stage1_rollout_baseline_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
STAGE1_GEOMETRY=${STAGE1_OUT}/visual_stage1_rollout_baseline_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json
STAGE1_SUMMARY=${STAGE1_OUT}/summary_stage1_rollout_baseline.md

TRACE_VAL_RECORD=${TRACE_VAL_OUT}/eval_stage2_trace_guarded_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
ANSWER_VAL_RECORD=${ANSWER_VAL_OUT}/eval_stage2_answer_only_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
TRACE_VAL_GEOMETRY=${TRACE_VAL_OUT}/visual_stage2_trace_guarded_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json
ANSWER_VAL_GEOMETRY=${ANSWER_VAL_OUT}/visual_stage2_answer_only_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json

TRACE_TEST_RECORD=${TRACE_TEST_OUT}/eval_stage2_trace_guarded_testbest_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
ANSWER_TEST_RECORD=${ANSWER_TEST_OUT}/eval_stage2_answer_only_testbest_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
TRACE_TEST_GEOMETRY=${TRACE_TEST_OUT}/visual_stage2_trace_guarded_testbest_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json
ANSWER_TEST_GEOMETRY=${ANSWER_TEST_OUT}/visual_stage2_answer_only_testbest_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json

"${PYTHON}" tools/trace_bridge_validate_rollout_records.py \
  --record "BRIDGE=${BRIDGE_RECORD}" \
  --record "Stage1=${STAGE1_RECORD}" \
  --record "answer-only-valbest=${ANSWER_VAL_RECORD}" \
  --record "TRACE-valbest=${TRACE_VAL_RECORD}" \
  --record "answer-only-testbest=${ANSWER_TEST_RECORD}" \
  --record "TRACE-testbest=${TRACE_TEST_RECORD}" \
  --expected_records 200 \
  --expected_views 8 \
  --require_rollout_text \
  --out "${AUDIT_OUT}/rollout_record_schema.json"

"${PYTHON}" tools/trace_bridge_compare_rollouts.py \
  --record "BRIDGE=${BRIDGE_RECORD}" \
  --record "Stage1=${STAGE1_RECORD}" \
  --record "answer-only=${ANSWER_VAL_RECORD}" \
  --record "TRACE=${TRACE_VAL_RECORD}" \
  --out_dir "${AUDIT_OUT}/global_pca_valbest" \
  --max_records 200 \
  --pca_fit_records 200 \
  --require_common_questions 200

"${PYTHON}" tools/trace_bridge_compare_rollouts.py \
  --record "BRIDGE=${BRIDGE_RECORD}" \
  --record "Stage1=${STAGE1_RECORD}" \
  --record "answer-only=${ANSWER_TEST_RECORD}" \
  --record "TRACE=${TRACE_TEST_RECORD}" \
  --out_dir "${AUDIT_OUT}/global_pca_testbest" \
  --max_records 200 \
  --pca_fit_records 200 \
  --require_common_questions 200

"${PYTHON}" tools/trace_bridge_final_audit.py \
  --baseline_summary "${BASELINE_SUMMARY}" \
  --baseline_geometry "${BRIDGE_GEOMETRY}" \
  --stage1_summary "${STAGE1_SUMMARY}" \
  --answer_summary "${ANSWER_VAL_OUT}/summary_stage2_answer_only.md" \
  --trace_summary "${TRACE_VAL_OUT}/summary_stage2_trace_guarded.md" \
  --stage1_geometry "${STAGE1_GEOMETRY}" \
  --answer_geometry "${ANSWER_VAL_GEOMETRY}" \
  --trace_geometry "${TRACE_VAL_GEOMETRY}" \
  --out_dir "${AUDIT_OUT}/valbest"

"${PYTHON}" tools/trace_bridge_final_audit.py \
  --baseline_summary "${BASELINE_SUMMARY}" \
  --baseline_geometry "${BRIDGE_GEOMETRY}" \
  --stage1_summary "${STAGE1_SUMMARY}" \
  --answer_summary "${ANSWER_TEST_OUT}/summary_stage2_answer_only_testbest.md" \
  --trace_summary "${TRACE_TEST_OUT}/summary_stage2_trace_guarded_testbest.md" \
  --stage1_geometry "${STAGE1_GEOMETRY}" \
  --answer_geometry "${ANSWER_TEST_GEOMETRY}" \
  --trace_geometry "${TRACE_TEST_GEOMETRY}" \
  --out_dir "${AUDIT_OUT}/testbest"

printf 'Guarded TRACE candidate sweep and final audits completed.\n' > "${AUDIT_OUT}/watcher_done.txt"
