#!/usr/bin/env bash
set -euo pipefail

: "${PS1:=}"
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT

cd /disk1/dingxukai/trace_colar

MODE="${MODE:-stage1}"
GPU="${GPU:-2}"
RUN_TAG="${RUN_TAG:-20260707_trace_bridge_${MODE}_gpu${GPU}}"
TEST_TIMES="${TEST_TIMES:-1}"
TRACE_MODEL="${TRACE_MODEL:-trace_bridge_qwen3_instruct}"
DATASET_DIR="${DATASET_DIR:-/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc}"
COT_CKPT="${COT_CKPT:-/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/checkpoints/epoch0__step6726__monitor0.871.ckpt}"
OUT_DIR="${OUT_DIR:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/${RUN_TAG}}"
ROOT_DIR="${ROOT_DIR:-/disk1/dingxukai/trace_colar/run_roots/trace_bridge/${RUN_TAG}}"

mkdir -p "${OUT_DIR}" "${ROOT_DIR}"

declare -A EVAL_DATASETS=(
  ["gsm8k_aug"]="/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc"
  ["gsmhard"]="/home/dingxukai/RoT/data/GSM8k-Hard/readcot_qsa_test"
  ["svamp"]="/home/dingxukai/RoT/data/SVAMP/readcot_qsa_test"
  ["multiarith"]="/home/dingxukai/RoT/data/Multiarith/readcot_qsa_test"
)

write_manifest() {
  {
    printf 'mode=%s\n' "${MODE}"
    printf 'gpu=%s\n' "${GPU}"
    printf 'run_tag=%s\n' "${RUN_TAG}"
    printf 'trace_model=%s\n' "${TRACE_MODEL}"
    printf 'cot_ckpt=%s\n' "${COT_CKPT}"
    printf 'dataset_dir=%s\n' "${DATASET_DIR}"
    printf 'test_times=%s\n' "${TEST_TIMES}"
    printf 'created_at=%s\n' "$(date '+%F %T')"
  } > "${OUT_DIR}/manifest.txt"
}

latest_ckpt_for_suffix() {
  local model_name="$1"
  local suffix="$2"
  find "/disk1/dingxukai/trace_colar/logs/${model_name}/qsa-gsm" \
    -path "*${suffix}*/checkpoints/*.ckpt" ! -name "last.ckpt" -print 2>/dev/null | sort | tail -n 1
}

train_bridge_baseline() {
  local suffix="${RUN_TAG}_bridge_baseline"
  echo "[train] BRIDGE baseline ${suffix}" | tee -a "${OUT_DIR}/train_bridge_baseline.log"
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model bridge_qwen3_instruct_hybrid_compact_anchor_gate \
    --dataset qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /home/dingxukai \
    --load_ckpt_path "${COT_CKPT}" \
    --do_test \
    --test_times "${TEST_TIMES}" \
    --log_suffix "${suffix}" \
    dataset_dir="${DATASET_DIR}" \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    trainer.num_sanity_val_steps=0 \
    trainer.val_check_interval=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.default_root_dir="${ROOT_DIR}/bridge_baseline_train" \
    2>&1 | tee -a "${OUT_DIR}/train_bridge_baseline.log"
  local ckpt
  ckpt="$(latest_ckpt_for_suffix bridge_qwen3_instruct_hybrid_compact_anchor_gate "${suffix}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[error] no BRIDGE baseline ckpt found" | tee -a "${OUT_DIR}/manifest.txt"
    exit 1
  fi
  echo "${ckpt}" | tee "${OUT_DIR}/bridge_baseline_best_ckpt.txt"
}

train_trace_stage1() {
  local suffix="$1"
  local log_file="$2"
  echo "[train] TRACE-BRIDGE Stage1 ${suffix}" | tee -a "${log_file}"
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model "${TRACE_MODEL}" \
    --dataset qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /home/dingxukai \
    --load_ckpt_path "${COT_CKPT}" \
    --do_test \
    --test_times "${TEST_TIMES}" \
    --log_suffix "${suffix}" \
    dataset_dir="${DATASET_DIR}" \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    trainer.num_sanity_val_steps=0 \
    trainer.val_check_interval=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.default_root_dir="${ROOT_DIR}/${suffix}_train" \
    2>&1 | tee -a "${log_file}"
  local ckpt
  ckpt="$(latest_ckpt_for_suffix "${TRACE_MODEL}" "${suffix}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[error] no TRACE-BRIDGE Stage1 ckpt found for ${suffix}" | tee -a "${OUT_DIR}/manifest.txt"
    exit 1
  fi
  echo "${ckpt}" | tee "${OUT_DIR}/${suffix}_best_ckpt.txt"
}

train_trace_stage2() {
  local stage1_ckpt="$1"
  local variant="$2"
  local suffix="${RUN_TAG}_${variant}"
  local log_file="${OUT_DIR}/train_${variant}.log"
  local rl_args=()
  if [[ "${variant}" == "stage2_strong" ]]; then
    rl_args=(
      n_train_samples_per_epoch=2048
      group_size=8
      model.model_kwargs.trace_rl_config.temperature=0.9
      model.model_kwargs.trace_rl_config.trace_reward_weight=0.18
      model.model_kwargs.trace_rl_config.pos_mode_fit_weight=0.16
      model.model_kwargs.trace_rl_config.mode_diversity_weight=0.08
      model.model_kwargs.trace_rl_config.neg_repulsion_weight=0.14
      model.model_kwargs.trace_rl_config.output_length_penalty_weight=0.015
      model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.08
      lr=7e-7
    )
  else
    rl_args=(
      n_train_samples_per_epoch=1024
      group_size=8
      model.model_kwargs.trace_rl_config.temperature=0.8
      model.model_kwargs.trace_rl_config.trace_reward_weight=0.10
      model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.12
      lr=5e-7
    )
  fi

  echo "[train] TRACE-BRIDGE ${variant} from ${stage1_ckpt}" | tee -a "${log_file}"
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model "${TRACE_MODEL}" \
    --dataset qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /home/dingxukai \
    --load_ckpt_path "${stage1_ckpt}" \
    --do_test \
    --test_times "${TEST_TIMES}" \
    --log_suffix "${suffix}" \
    dataset_dir="${DATASET_DIR}" \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    do_trace_rl=true \
    trainer.num_sanity_val_steps=0 \
    trainer.max_epochs=10 \
    trainer.max_steps=-1 \
    trainer.val_check_interval=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.gradient_clip_val=0 \
    trainer.default_root_dir="${ROOT_DIR}/${variant}_train" \
    model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
    "${rl_args[@]}" \
    2>&1 | tee -a "${log_file}"

  local ckpt
  ckpt="$(latest_ckpt_for_suffix "${TRACE_MODEL}" "${suffix}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[error] no TRACE-BRIDGE ${variant} ckpt found" | tee -a "${OUT_DIR}/manifest.txt"
    exit 1
  fi
  echo "${ckpt}" | tee "${OUT_DIR}/${variant}_best_ckpt.txt"
}

eval_ckpt() {
  local model_name="$1"
  local ckpt="$2"
  local label="$3"
  local is_trace="$4"
  for key in gsm8k_aug gsmhard svamp multiarith; do
    local data_dir="${EVAL_DATASETS[$key]}"
    local log_file="${OUT_DIR}/eval_${label}_${key}.log"
    local log_dir="${OUT_DIR}/eval_${label}_${key}_logs"
    local extra_args=()
    if [[ "${is_trace}" == "1" ]]; then
      extra_args=(
        model.model_kwargs.trace_bridge_config.trace_visual_group_views=8
        model.model_kwargs.trace_bridge_config.trace_visual_record_limit=200
      )
    fi
    echo "[eval] ${label} ${key}" | tee "${log_file}"
    TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
      --model "${model_name}" \
      --dataset qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /home/dingxukai \
      --test_ckpt_path "${ckpt}" \
      --test_times "${TEST_TIMES}" \
      dataset_dir="${data_dir}" \
      batch_size=1 \
      val_batch_size=1 \
      num_workers=2 \
      persistent_workers=false \
      trainer.num_sanity_val_steps=0 \
      trainer.default_root_dir="${ROOT_DIR}/eval_${label}_${key}" \
      trainer.logger.save_dir="${log_dir}" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      "${extra_args[@]}" \
      2>&1 | tee -a "${log_file}"
  done
}

summarize_label() {
  local label="$1"
  python - "${OUT_DIR}" "${label}" <<'PY'
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
label = sys.argv[2]
names = {
    "gsm8k_aug": "GSM8K-Aug",
    "gsmhard": "GSM-Hard",
    "svamp": "SVAMP",
    "multiarith": "MultiArith",
}
rows = []
for key, name in names.items():
    text = (out / f"eval_{label}_{key}.log").read_text(errors="ignore")
    vals = {}
    for m in re.finditer(r"│\s*(test/[^│]+?)\s*│\s*([-0-9.]+)\s*│", text):
        vals[m.group(1).strip()] = float(m.group(2))
    if not vals:
        continue
    acc = vals.get("test/acc", 0.0) * 100.0
    lat = vals.get("test/n_latent_forward", 0.0)
    out_len = vals.get("test/output_length", 0.0)
    total = lat + out_len
    dep = vals.get("test/dep_f1", 0.0)
    res = vals.get("test/residual_similarity", 0.0)
    rows.append((name, acc, lat, out_len, total, dep, res))

md = out / f"summary_{label}.md"
with md.open("w", encoding="utf-8") as f:
    f.write("| Label | Dataset | Acc | #Latent | OutputLen | #L | Dep-F1 | Res-Sim |\\n")
    f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\\n")
    for row in rows:
        f.write(f"| {label} | {row[0]} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.2f} | {row[4]:.2f} | {row[5]:.3f} | {row[6]:.3f} |\\n")
    if rows:
        f.write(f"\\nAverage Acc: {sum(r[1] for r in rows)/len(rows):.2f}\\n")
        f.write(f"Average #L: {sum(r[4] for r in rows)/len(rows):.2f}\\n")
print(md.read_text() if md.exists() else "")
PY
}

run_trace_visuals() {
  local label="$1"
  local record="${OUT_DIR}/eval_${label}_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt"
  if [[ ! -f "${record}" ]]; then
    echo "[warn] visual record not found: ${record}" | tee -a "${OUT_DIR}/manifest.txt"
    return
  fi
  local visual_dir="${OUT_DIR}/visual_${label}_gsm8k_aug"
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_bridge_visualize.py \
    --records "${record}" \
    --out_dir "${visual_dir}" \
    --max_records 9 \
    2>&1 | tee "${OUT_DIR}/visual_${label}.log"
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_bridge_geometry_summary.py \
    --records "${record}" \
    --out_dir "${visual_dir}/geometry_200" \
    --max_records 200 \
    2>&1 | tee "${OUT_DIR}/geometry_${label}.log"
}

write_manifest

case "${MODE}" in
  baseline)
    train_bridge_baseline
    CKPT="$(cat "${OUT_DIR}/bridge_baseline_best_ckpt.txt")"
    eval_ckpt bridge_qwen3_instruct_hybrid_compact_anchor_gate "${CKPT}" bridge_baseline 0
    summarize_label bridge_baseline
    ;;
  stage1)
    STAGE1_SUFFIX="${RUN_TAG}_stage1"
    train_trace_stage1 "${STAGE1_SUFFIX}" "${OUT_DIR}/train_stage1.log"
    CKPT="$(cat "${OUT_DIR}/${STAGE1_SUFFIX}_best_ckpt.txt")"
    eval_ckpt "${TRACE_MODEL}" "${CKPT}" trace_stage1 1
    summarize_label trace_stage1
    run_trace_visuals trace_stage1
    ;;
  stage2_conservative|stage2_strong)
    STAGE1_SUFFIX="${RUN_TAG}_stage1"
    train_trace_stage1 "${STAGE1_SUFFIX}" "${OUT_DIR}/train_stage1_for_${MODE}.log"
    STAGE1_CKPT="$(cat "${OUT_DIR}/${STAGE1_SUFFIX}_best_ckpt.txt")"
    train_trace_stage2 "${STAGE1_CKPT}" "${MODE}"
    CKPT="$(cat "${OUT_DIR}/${MODE}_best_ckpt.txt")"
    eval_ckpt "${TRACE_MODEL}" "${CKPT}" "${MODE}" 1
    summarize_label "${MODE}"
    run_trace_visuals "${MODE}"
    ;;
  stage2_conservative_from_ckpt|stage2_strong_from_ckpt)
    : "${STAGE1_CKPT:?Set STAGE1_CKPT to a completed TRACE-BRIDGE Stage1 checkpoint.}"
    VARIANT="${MODE%_from_ckpt}"
    train_trace_stage2 "${STAGE1_CKPT}" "${VARIANT}"
    CKPT="$(cat "${OUT_DIR}/${VARIANT}_best_ckpt.txt")"
    eval_ckpt "${TRACE_MODEL}" "${CKPT}" "${VARIANT}" 1
    summarize_label "${VARIANT}"
    run_trace_visuals "${VARIANT}"
    ;;
  stage2_both_from_ckpt)
    : "${STAGE1_CKPT:?Set STAGE1_CKPT to a completed TRACE-BRIDGE Stage1 checkpoint.}"
    for VARIANT in stage2_conservative stage2_strong; do
      train_trace_stage2 "${STAGE1_CKPT}" "${VARIANT}"
      CKPT="$(cat "${OUT_DIR}/${VARIANT}_best_ckpt.txt")"
      eval_ckpt "${TRACE_MODEL}" "${CKPT}" "${VARIANT}" 1
      summarize_label "${VARIANT}"
      run_trace_visuals "${VARIANT}"
    done
    ;;
  eval_bridge_from_ckpt)
    : "${CKPT:?Set CKPT to a completed BRIDGE checkpoint.}"
    LABEL="${EVAL_LABEL:-bridge_baseline}"
    eval_ckpt bridge_qwen3_instruct_hybrid_compact_anchor_gate "${CKPT}" "${LABEL}" 0
    summarize_label "${LABEL}"
    ;;
  eval_trace_from_ckpt)
    : "${CKPT:?Set CKPT to a completed TRACE-BRIDGE checkpoint.}"
    LABEL="${EVAL_LABEL:-trace_eval}"
    eval_ckpt "${TRACE_MODEL}" "${CKPT}" "${LABEL}" 1
    summarize_label "${LABEL}"
    run_trace_visuals "${LABEL}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use baseline, stage1, stage2_conservative, stage2_strong, *_from_ckpt, stage2_both_from_ckpt, or eval_*_from_ckpt." >&2
    exit 2
    ;;
esac

echo "[done] ${MODE} outputs: ${OUT_DIR}" | tee -a "${OUT_DIR}/manifest.txt"
