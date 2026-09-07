#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

CKPT="/disk1/dingxukai/trace_colar/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260706-032913_586957_trace_v3_three_stage_full_20260705_stage1solid_stage2_trace_multipath_rl/checkpoints/epoch4__step2560__monitor0.347.ckpt"
ARTIFACT_DIR="${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706/origin_epoch4_exactL_more_20260707"
THRESHOLD="${THRESHOLD:-0.29}"
GPU="${GPU:-4}"

for L in 18 20; do
  TAG="origin_epoch4_exactL${L}_t07_cf5"
  echo "[exactL-sweep] eval ${TAG}"
  CKPT="${CKPT}" \
    TAG="${TAG}" \
    GPU="${GPU}" \
    ARTIFACT_DIR="${ARTIFACT_DIR}" \
    MAX_L="${L}" \
    MIN_L="${L}" \
    LATENT_TEMP=0.7 \
    EOL_TEMP=1.0 \
    COMPRESSION_FACTOR=5 \
    TEST_TIMES=1 \
    bash run_trace_v3_direct_single_eval_20260706.sh

  ACC="$(TAG="${TAG}" ARTIFACT_DIR="${ARTIFACT_DIR}" /home/dingxukai/miniconda3/envs/ROT/bin/python - <<'PY'
import csv, os
tag = os.environ["TAG"]
summary = os.path.join(os.environ["ARTIFACT_DIR"], "single_eval_summary.csv")
acc = ""
with open(summary, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("tag") == tag:
            acc = row.get("acc", "")
print(acc)
PY
)"
  echo "[exactL-sweep] ${TAG} acc=${ACC}"

  if ACC="${ACC}" THRESHOLD="${THRESHOLD}" /home/dingxukai/miniconda3/envs/ROT/bin/python - <<'PY'
import os, sys
acc = float(os.environ["ACC"])
threshold = float(os.environ["THRESHOLD"])
sys.exit(0 if acc >= threshold else 1)
PY
  then
    echo "[exactL-sweep] ${TAG} reached threshold; running followup"
    CKPT="${CKPT}" \
      TAG="${TAG}_followup" \
      GPU="${GPU}" \
      ARTIFACT_DIR="${ARTIFACT_DIR}/${TAG}_followup" \
      MAX_L="${L}" \
      MIN_L="${L}" \
      LATENT_TEMP=0.7 \
      EOL_TEMP=1.0 \
      COMPRESSION_FACTOR=5 \
      TEST_TIMES=1 \
      RUN_OOD=1 \
      RUN_VIS=1 \
      RUN_GEOMETRY=1 \
      bash run_trace_v3_candidate_followup_20260707.sh
  fi
done

echo "[exactL-sweep] done"
