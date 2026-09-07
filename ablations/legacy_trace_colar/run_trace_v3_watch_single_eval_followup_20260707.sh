#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

SUMMARY="${SUMMARY:?SUMMARY is required}"
TAG="${TAG:?TAG is required}"
CKPT="${CKPT:?CKPT is required}"
GPU="${GPU:-2}"
THRESHOLD="${THRESHOLD:-0.29}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"
FOLLOWUP_DIR="${FOLLOWUP_DIR:?FOLLOWUP_DIR is required}"
MAX_L="${MAX_L:-40}"
MIN_L="${MIN_L:-12}"
LATENT_TEMP="${LATENT_TEMP:-0.7}"
EOL_TEMP="${EOL_TEMP:-1.0}"
COMPRESSION_FACTOR="${COMPRESSION_FACTOR:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEST_TIMES="${TEST_TIMES:-1}"

mkdir -p "${FOLLOWUP_DIR}"
WATCH_LOG="${FOLLOWUP_DIR}/watch_single_eval_followup.log"
exec > >(tee -a "${WATCH_LOG}") 2>&1

echo "[watch-single] start $(date '+%F %T')"
echo "[watch-single] summary=${SUMMARY}"
echo "[watch-single] tag=${TAG}"
echo "[watch-single] threshold=${THRESHOLD}"

start_epoch="$(date +%s)"
while true; do
  if [[ -s "${SUMMARY}" ]]; then
    ROW_ACC="$(/home/dingxukai/miniconda3/envs/ROT/bin/python - "${SUMMARY}" "${TAG}" <<'PY'
import csv, sys
summary, tag = sys.argv[1], sys.argv[2]
with open(summary, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("tag") == tag:
            print(row.get("acc", ""))
            raise SystemExit(0)
raise SystemExit(1)
PY
)" || ROW_ACC=""
    if [[ -n "${ROW_ACC}" ]]; then
      break
    fi
  fi
  now="$(date +%s)"
  elapsed=$((now - start_epoch))
  if [[ "${MAX_WAIT_SECONDS}" != "0" && "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]]; then
    echo "[watch-single] timeout after ${elapsed}s"
    exit 4
  fi
  echo "[watch-single] waiting elapsed=${elapsed}s"
  sleep "${POLL_SECONDS}"
done

echo "[watch-single] acc=${ROW_ACC}"
if ! /home/dingxukai/miniconda3/envs/ROT/bin/python - "${ROW_ACC}" "${THRESHOLD}" <<'PY'
import sys
acc = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if acc >= threshold else 1)
PY
then
  echo "[watch-single] acc below threshold; skip followup"
  exit 0
fi

echo "[watch-single] acc reached threshold; running OOD/visual/geometry"
CKPT="${CKPT}" \
TAG="${TAG}" \
GPU="${GPU}" \
ARTIFACT_DIR="${FOLLOWUP_DIR}" \
MAX_L="${MAX_L}" \
MIN_L="${MIN_L}" \
LATENT_TEMP="${LATENT_TEMP}" \
EOL_TEMP="${EOL_TEMP}" \
COMPRESSION_FACTOR="${COMPRESSION_FACTOR}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
TEST_TIMES="${TEST_TIMES}" \
RUN_OOD=1 \
RUN_VIS=1 \
RUN_GEOMETRY=1 \
bash run_trace_v3_candidate_followup_20260707.sh

echo "[watch-single] done $(date '+%F %T')"
