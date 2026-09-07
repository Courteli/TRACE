#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

GPU="${GPU:-6}"
THRESHOLD="${THRESHOLD:-0.29}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"
CKPT="${CKPT:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260706-032913_586957_trace_v3_three_stage_full_20260705_stage1solid_stage2_trace_multipath_rl/checkpoints/epoch4__step2560__monitor0.347.ckpt}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706/origin_epoch4_maxcap_candidates}"
SUMMARY="${ARTIFACT_DIR}/single_eval_summary.csv"
CAP32_TAG="${CAP32_TAG:-origin_epoch4_max32_m12_t07_cf5}"
DET_TAG="${DET_TAG:-origin_epoch4_detmean_m12_t07_cf5}"

get_acc() {
  local tag="$1"
  [[ -f "${SUMMARY}" ]] || return 1
  awk -F, -v needle="${tag}" '$1 == needle { print $7 }' "${SUMMARY}" | tail -1
}

wait_for_gpu_free() {
  while true; do
    local mem
    mem="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    echo "[watch] $(date '+%F %T') gpu${GPU}_mem=${mem}"
    if [[ "${mem}" -lt 2000 ]]; then
      return 0
    fi
    sleep "${SLEEP_SECONDS}"
  done
}

run_followup() {
  local tag="$1"
  local max_l="$2"
  local lp_det="$3"
  local out_dir="${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706/followup_${tag}"
  echo "[watch] followup tag=${tag} max_l=${max_l} lp_det=${lp_det}"
  CKPT="${CKPT}" \
    TAG="${tag}" \
    GPU="${GPU}" \
    ARTIFACT_DIR="${out_dir}" \
    MAX_L="${max_l}" \
    MIN_L=12 \
    LATENT_TEMP=0.7 \
    EOL_TEMP=1.0 \
    COMPRESSION_FACTOR=5 \
    TEST_TIMES=1 \
    LP_DETERMINISTIC="${lp_det}" \
    bash run_trace_v3_candidate_followup_20260707.sh
}

echo "[watch] start $(date '+%F %T')"
while true; do
  cap32_acc="$(get_acc "${CAP32_TAG}" || true)"
  if [[ -n "${cap32_acc}" ]]; then
    echo "[watch] cap32 acc=${cap32_acc}"
    break
  fi
  sleep "${SLEEP_SECONDS}"
done

if awk -v acc="${cap32_acc}" -v threshold="${THRESHOLD}" 'BEGIN { exit(acc >= threshold ? 0 : 1) }'; then
  wait_for_gpu_free
  run_followup "${CAP32_TAG}" 32 0
  exit 0
fi

det_acc="$(get_acc "${DET_TAG}" || true)"
if [[ -z "${det_acc}" ]]; then
  wait_for_gpu_free
  echo "[watch] running deterministic mean-path eval"
  CKPT="${CKPT}" \
    TAG="${DET_TAG}" \
    GPU="${GPU}" \
    ARTIFACT_DIR="${ARTIFACT_DIR}" \
    MAX_L=40 \
    MIN_L=12 \
    LATENT_TEMP=0.7 \
    EOL_TEMP=1.0 \
    COMPRESSION_FACTOR=5 \
    MAX_NEW_TOKENS=16 \
    TEST_TIMES=1 \
    EXTRA_RUN_ARGS="lp_determinisitc=true" \
    bash run_trace_v3_direct_single_eval_20260706.sh
  det_acc="$(get_acc "${DET_TAG}" || true)"
fi

echo "[watch] det acc=${det_acc}"
if [[ -n "${det_acc}" ]] && awk -v acc="${det_acc}" -v threshold="${THRESHOLD}" 'BEGIN { exit(acc >= threshold ? 0 : 1) }'; then
  wait_for_gpu_free
  run_followup "${DET_TAG}" 40 1
else
  echo "[watch] no candidate reached threshold=${THRESHOLD}"
fi

echo "[watch] done $(date '+%F %T')"
