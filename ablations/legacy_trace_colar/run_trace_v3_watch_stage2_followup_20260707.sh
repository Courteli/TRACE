#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR is required}"
TAG="${TAG:?TAG is required}"
GPU="${GPU:-6}"
MIN_L="${MIN_L:-12}"
MAX_L="${MAX_L:-40}"
LATENT_TEMP="${LATENT_TEMP:-0.7}"
EOL_TEMP="${EOL_TEMP:-1.0}"
COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-5}"
TEST_TIMES="${TEST_TIMES:-1}"
THRESHOLD="${THRESHOLD:-0.29}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"
FOLLOWUP_DIR="${FOLLOWUP_DIR:-${ARTIFACT_DIR}/followup_${TAG}}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-1024}"
GPU_FREE_POLL_SECONDS="${GPU_FREE_POLL_SECONDS:-60}"

SUMMARY="${ARTIFACT_DIR}/gsm8k_min${MIN_L}_summary.csv"
BEST_CKPT_FILE="${ARTIFACT_DIR}/best_ckpt.txt"
WATCH_LOG="${ARTIFACT_DIR}/watch_followup.log"

mkdir -p "${ARTIFACT_DIR}"
exec > >(tee -a "${WATCH_LOG}") 2>&1

current_gpu_used_mb() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v gpu="${GPU}" '$1 + 0 == gpu { gsub(/ /, "", $2); print $2; found = 1 } END { if (!found) print "" }'
}

wait_for_gpu_free() {
  local used_mb
  while true; do
    used_mb="$(current_gpu_used_mb)"
    if [[ -n "${used_mb}" && "${used_mb}" -le "${GPU_MAX_USED_MB}" ]]; then
      return 0
    fi
    echo "[watch-followup] waiting for gpu=${GPU} free used_mb=${used_mb:-unknown} max_used_mb=${GPU_MAX_USED_MB}"
    sleep "${GPU_FREE_POLL_SECONDS}"
  done
}

echo "[watch-followup] start $(date '+%F %T')"
echo "[watch-followup] artifact_dir=${ARTIFACT_DIR}"
echo "[watch-followup] summary=${SUMMARY}"
echo "[watch-followup] threshold=${THRESHOLD}"

start_epoch="$(date +%s)"
while [[ ! -s "${SUMMARY}" || ! -s "${BEST_CKPT_FILE}" ]]; do
  now="$(date +%s)"
  elapsed=$((now - start_epoch))
  if [[ "${MAX_WAIT_SECONDS}" != "0" && "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]]; then
    echo "[watch-followup] timeout after ${elapsed}s"
    exit 4
  fi
  echo "[watch-followup] waiting elapsed=${elapsed}s"
  sleep "${POLL_SECONDS}"
done

ACC="$(/home/dingxukai/miniconda3/envs/ROT/bin/python - "${SUMMARY}" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("no rows")
row = rows[-1]
print(row["acc"])
PY
)"
CKPT="$(cat "${BEST_CKPT_FILE}")"
echo "[watch-followup] acc=${ACC}"
echo "[watch-followup] ckpt=${CKPT}"

if ! /home/dingxukai/miniconda3/envs/ROT/bin/python - "${ACC}" "${THRESHOLD}" <<'PY'
import sys
acc = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if acc >= threshold else 1)
PY
then
  echo "[watch-followup] acc below threshold; skip followup"
  exit 0
fi

echo "[watch-followup] acc reached threshold; running OOD/visual/geometry"
wait_for_gpu_free
CKPT="${CKPT}" \
TAG="${TAG}" \
GPU="${GPU}" \
ARTIFACT_DIR="${FOLLOWUP_DIR}" \
MAX_L="${MAX_L}" \
MIN_L="${MIN_L}" \
LATENT_TEMP="${LATENT_TEMP}" \
EOL_TEMP="${EOL_TEMP}" \
COMPRESSION_FACTOR="${COMPRESSION_FACTOR}" \
TEST_TIMES="${TEST_TIMES}" \
RUN_OOD=1 \
RUN_VIS=1 \
RUN_GEOMETRY=1 \
bash run_trace_v3_candidate_followup_20260707.sh

echo "[watch-followup] done $(date '+%F %T')"
