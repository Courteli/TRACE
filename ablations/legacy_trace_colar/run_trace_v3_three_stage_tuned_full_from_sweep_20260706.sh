#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

CKPT="${CKPT:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260706-032913_586957_trace_v3_three_stage_full_20260705_stage1solid_stage2_trace_multipath_rl/checkpoints/epoch4__step2560__monitor0.347.ckpt}"
GPU="${GPU:-2}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706}"
SUMMARY_CSV="${SUMMARY_CSV:-${ARTIFACT_DIR}/sweep_summary.csv}"
WAIT_SESSION="${WAIT_SESSION:-}"
TOP_K="${TOP_K:-3}"
TARGET_ACC="${TARGET_ACC:-0.30}"
TEST_TIMES="${TEST_TIMES:-1}"

mkdir -p "${ARTIFACT_DIR}/full_test"

if [[ -n "${WAIT_SESSION}" ]]; then
  echo "[full-from-sweep] waiting for tmux session ${WAIT_SESSION}"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

if [[ ! -s "${SUMMARY_CSV}" ]]; then
  echo "[full-from-sweep] ERROR: missing summary CSV ${SUMMARY_CSV}" >&2
  exit 2
fi

FULL_SUMMARY="${ARTIFACT_DIR}/full_test_summary.csv"
echo "source_tag,val_acc,json_path,dataset,test_file,n_items,n_predictions,acc,avg_L,avg_output_len,max_L,min_L,latent_temperature,eol_temperature,compression_factor,max_new_tokens,ckpt" > "${FULL_SUMMARY}"

TOP_FILE="${ARTIFACT_DIR}/full_test/top_candidates.tsv"
/home/dingxukai/miniconda3/envs/ROT/bin/python - <<PY > "${TOP_FILE}"
import csv
from pathlib import Path

summary = Path("${SUMMARY_CSV}")
rows = []
with summary.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            row["_acc"] = float(row["acc"])
        except Exception:
            continue
        rows.append(row)
rows.sort(key=lambda r: r["_acc"], reverse=True)
for row in rows[: int("${TOP_K}")]:
    fields = [
        row["tag"],
        row["acc"],
        row["max_L"],
        row["min_L"],
        row["latent_temperature"],
        row["eol_temperature"],
        row["compression_factor"],
        row["max_new_tokens"],
    ]
    print("\t".join(fields))
PY

if [[ ! -s "${TOP_FILE}" ]]; then
  echo "[full-from-sweep] ERROR: no candidates parsed from ${SUMMARY_CSV}" >&2
  exit 3
fi

run_full_eval() {
  local source_tag="$1"
  local val_acc="$2"
  local max_l="$3"
  local min_l="$4"
  local latent_temp="$5"
  local eol_temp="$6"
  local compression_factor="$7"
  local max_new_tokens="$8"
  local tag="full_${source_tag}"
  local log_dir="${ARTIFACT_DIR}/full_test/${tag}"
  mkdir -p "${log_dir}"
  echo "[full-from-sweep] start ${tag}: val_acc=${val_acc} max_l=${max_l} min_l=${min_l} latent_temp=${latent_temp} eol_temp=${eol_temp} cf=${compression_factor}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
    --model=trace_multipath_qwen3_instruct \
    --dataset=gsm8k_aug_nl \
    --devices=0 \
    --test_ckpt_path="${CKPT}" \
    --test_times="${TEST_TIMES}" \
    --seed=0 \
    --workspace_path=/home/dingxukai \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=0 \
    persistent_workers=False \
    num_sanity_val_steps=0 \
    trainer.logger.save_dir="${log_dir}" \
    trainer.logger.name="tb" \
    trainer.logger.version="run" \
    trainer.default_root_dir="${ARTIFACT_DIR}/full_test/roots/${tag}" \
    dataset_name="gsm8k_aug_nl" \
    dataset_dir="/home/dingxukai/RoT/data/GSM8k-Aug-NL" \
    test_file="gsm8k_test_processed.jsonl" \
    max_n_latent_forward="${max_l}" \
    min_n_latent_forward="${min_l}" \
    latent_temperature="${latent_temp}" \
    eol_temperature="${eol_temp}" \
    compression_factor="${compression_factor}" \
    max_new_tokens="${max_new_tokens}" \
    2>&1 | tee "${ARTIFACT_DIR}/full_test/${tag}.log"

  local json_path
  json_path="$(find "${log_dir}/tb/run" -maxdepth 1 -type f -name 'test_*.json' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "${json_path}" ]]; then
    echo "[full-from-sweep] ERROR: no test json for ${tag}" >&2
    exit 4
  fi
  local copied="${ARTIFACT_DIR}/full_test/${tag}.json"
  cp -f "${json_path}" "${copied}"
  local row
  row="$(/home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_eval_json_summary.py --csv "${copied}" | tail -n +2)"
  echo "${source_tag},${val_acc},${row}" >> "${FULL_SUMMARY}"
  tail -1 "${FULL_SUMMARY}"

  local full_acc
  full_acc="$(echo "${row}" | /home/dingxukai/miniconda3/envs/ROT/bin/python -c 'import csv,sys; print(next(csv.DictReader(sys.stdin, fieldnames=["path","dataset","test_file","n_items","n_predictions","acc","avg_L","avg_output_len","max_L","min_L","latent_temperature","eol_temperature","compression_factor","max_new_tokens","ckpt"]))["acc"])')"
  /home/dingxukai/miniconda3/envs/ROT/bin/python - <<PY
import sys
sys.exit(0 if float("${full_acc}") >= float("${TARGET_ACC}") else 1)
PY
}

while IFS=$'\t' read -r source_tag val_acc max_l min_l latent_temp eol_temp compression_factor max_new_tokens; do
  if run_full_eval "${source_tag}" "${val_acc}" "${max_l}" "${min_l}" "${latent_temp}" "${eol_temp}" "${compression_factor}" "${max_new_tokens}"; then
    echo "[full-from-sweep] target reached; stopping after ${source_tag}"
    exit 0
  fi
done < "${TOP_FILE}"

echo "[full-from-sweep] target not reached by top ${TOP_K}. Summary: ${FULL_SUMMARY}"
