#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

GPU="${GPU:-6}"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
MAX_ROUNDS="${MAX_ROUNDS:-120}"
BASE_OUT="${BASE_OUT:-${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706}"

LATENT_DIR="${LATENT_DIR:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260706-235206_153162_trace_v3_three_stage_stage2_conservative_latent_20260706/checkpoints}"
BOTH_DIR="${BOTH_DIR:-${ROOT}/logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260707-003809_671515_trace_v3_three_stage_stage2_conservative_both_20260707/checkpoints}"
FREEZE_DIR="${FREEZE_DIR:-}"

echo "[high-epoch-test-queue] start $(date '+%F %T') gpu=${GPU} sleep=${SLEEP_SECONDS}s max_rounds=${MAX_ROUNDS}"
echo "[high-epoch-test-queue] latent_dir=${LATENT_DIR}"
echo "[high-epoch-test-queue] both_dir=${BOTH_DIR}"
echo "[high-epoch-test-queue] freeze_dir=${FREEZE_DIR}"

tag_done() {
  local summary="$1"
  local tag="$2"
  [[ -f "${summary}" ]] || return 1
  awk -F, -v needle="${tag}" '$1 == needle { found = 1 } END { exit(found ? 0 : 1) }' "${summary}"
}

eval_one() {
  local family="$1"
  local ckpt="$2"
  local min_epoch="$3"
  local name
  local epoch
  name="$(basename "${ckpt}")"
  epoch="$(sed -n 's/^epoch\([0-9][0-9]*\)__.*$/\1/p' <<< "${name}")"
  [[ -n "${epoch}" ]] || return 0
  (( epoch >= min_epoch )) || return 0

  local artifact_dir="${BASE_OUT}/stage2_conservative_${family}_epoch${epoch}_full"
  local summary="${artifact_dir}/single_eval_summary.csv"
  mkdir -p "${artifact_dir}"

  local specs=(
    "m8_t1_e1_cf5 8 1.0 1.0 5"
    "m12_t07_e1_cf5 12 0.7 1.0 5"
  )

  local spec
  for spec in "${specs[@]}"; do
    read -r suffix min_l latent_temp eol_temp cf <<< "${spec}"
    local tag="cons_${family}_epoch${epoch}_full_${suffix}"
    local lock_dir="${artifact_dir}/${tag}.lock"

    if tag_done "${summary}" "${tag}"; then
      echo "[high-epoch-test-queue] skip done tag=${tag}"
      continue
    fi
    if ! mkdir "${lock_dir}" 2>/dev/null; then
      echo "[high-epoch-test-queue] skip locked tag=${tag}"
      continue
    fi

    echo "[high-epoch-test-queue] eval tag=${tag} ckpt=${ckpt}"
    CKPT="${ckpt}" \
      TAG="${tag}" \
      GPU="${GPU}" \
      ARTIFACT_DIR="${artifact_dir}" \
      MAX_L=40 \
      MIN_L="${min_l}" \
      LATENT_TEMP="${latent_temp}" \
      EOL_TEMP="${eol_temp}" \
      COMPRESSION_FACTOR="${cf}" \
      MAX_NEW_TOKENS=16 \
      TEST_TIMES=1 \
      bash run_trace_v3_direct_single_eval_20260706.sh
    rmdir "${lock_dir}"
  done
}

round=0
while (( round < MAX_ROUNDS )); do
  round=$((round + 1))
  echo "[high-epoch-test-queue] scan round=${round} $(date '+%F %T')"

  if [[ -d "${LATENT_DIR}" ]]; then
    while IFS= read -r ckpt; do
      eval_one "latent" "${ckpt}" 1
    done < <(find "${LATENT_DIR}" -maxdepth 1 -type f -name 'epoch*.ckpt' | sort)
  fi

  if [[ -d "${BOTH_DIR}" ]]; then
    while IFS= read -r ckpt; do
      eval_one "both" "${ckpt}" 1
    done < <(find "${BOTH_DIR}" -maxdepth 1 -type f -name 'epoch*.ckpt' | sort)
  fi

  if [[ -n "${FREEZE_DIR}" && -d "${FREEZE_DIR}" ]]; then
    while IFS= read -r ckpt; do
      eval_one "freeze" "${ckpt}" 0
    done < <(find "${FREEZE_DIR}" -maxdepth 1 -type f -name 'epoch*.ckpt' | sort)
  fi

  sleep "${SLEEP_SECONDS}"
done

echo "[high-epoch-test-queue] done $(date '+%F %T')"
