#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
cd "${ROOT}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
FAMILY="${FAMILY:?FAMILY is required}"
GPU="${GPU:-0}"
BASE_OUT="${BASE_OUT:-${ROOT}/run_outputs/trace_multipath/tune_three_stage_20260706/epoch_ckpt_full_eval}"
MIN_EPOCH="${MIN_EPOCH:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
MAX_ROUNDS="${MAX_ROUNDS:-120}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEST_TIMES="${TEST_TIMES:-1}"
SPECS="${SPECS:-m12_t07_e1_cf5:12:40:0.7:1.0:5}"
RUN_FOLLOWUP="${RUN_FOLLOWUP:-1}"
FOLLOWUP_THRESHOLD="${FOLLOWUP_THRESHOLD:-0.29}"
FOLLOWUP_BASE_OUT="${FOLLOWUP_BASE_OUT:-}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-${ROOT}/run_outputs/trace_multipath/gpu_locks}"
GPU_LOCK_POLL_SECONDS="${GPU_LOCK_POLL_SECONDS:-60}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-1024}"
GPU_FREE_POLL_SECONDS="${GPU_FREE_POLL_SECONDS:-60}"

mkdir -p "${BASE_OUT}"
mkdir -p "${GPU_LOCK_ROOT}"

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
    echo "[ckpt-full-eval] waiting for gpu=${GPU} free used_mb=${used_mb:-unknown} max_used_mb=${GPU_MAX_USED_MB}"
    sleep "${GPU_FREE_POLL_SECONDS}"
  done
}

acquire_gpu_lock() {
  local lock_dir="${GPU_LOCK_ROOT}/gpu${GPU}.lock"
  while ! mkdir "${lock_dir}" 2>/dev/null; do
    echo "[ckpt-full-eval] waiting for gpu lock ${lock_dir}"
    sleep "${GPU_LOCK_POLL_SECONDS}"
  done
  wait_for_gpu_free
  echo "${lock_dir}"
}

tag_done() {
  local summary="$1"
  local tag="$2"
  [[ -f "${summary}" ]] || return 1
  awk -F, -v needle="${tag}" '$1 == needle { found = 1 } END { exit(found ? 0 : 1) }' "${summary}"
}

eval_ckpt() {
  local ckpt="$1"
  local name epoch
  name="$(basename "${ckpt}")"
  epoch="$(sed -n 's/^epoch\([0-9][0-9]*\)__.*$/\1/p' <<< "${name}")"
  [[ -n "${epoch}" ]] || return 0
  (( epoch >= MIN_EPOCH )) || return 0

  local artifact_dir="${BASE_OUT}/${FAMILY}_epoch${epoch}"
  local summary="${artifact_dir}/single_eval_summary.csv"
  mkdir -p "${artifact_dir}"

  local spec suffix min_l max_l latent_temp eol_temp cf tag lock_dir
  for spec in ${SPECS}; do
    IFS=: read -r suffix min_l max_l latent_temp eol_temp cf <<< "${spec}"
    tag="${FAMILY}_epoch${epoch}_${suffix}"
    lock_dir="${artifact_dir}/${tag}.lock"

    if tag_done "${summary}" "${tag}"; then
      echo "[ckpt-full-eval] skip done tag=${tag}"
      continue
    fi
    if ! mkdir "${lock_dir}" 2>/dev/null; then
      echo "[ckpt-full-eval] skip locked tag=${tag}"
      continue
    fi

    local gpu_lock_dir
    gpu_lock_dir="$(acquire_gpu_lock)"
    echo "[ckpt-full-eval] eval tag=${tag} ckpt=${ckpt}"
    set +e
    CKPT="${ckpt}" \
      TAG="${tag}" \
      GPU="${GPU}" \
      ARTIFACT_DIR="${artifact_dir}" \
      MAX_L="${max_l}" \
      MIN_L="${min_l}" \
      LATENT_TEMP="${latent_temp}" \
      EOL_TEMP="${eol_temp}" \
      COMPRESSION_FACTOR="${cf}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      TEST_TIMES="${TEST_TIMES}" \
      bash run_trace_v3_direct_single_eval_20260706.sh
    local eval_status=$?
    set -e
    if (( eval_status != 0 )); then
      rmdir "${gpu_lock_dir}"
      rmdir "${lock_dir}"
      return "${eval_status}"
    fi

    if [[ "${RUN_FOLLOWUP}" == "1" ]]; then
      local acc
      acc="$(/home/dingxukai/miniconda3/envs/ROT/bin/python - "${summary}" "${tag}" <<'PY'
import csv, sys
summary, tag = sys.argv[1], sys.argv[2]
with open(summary, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("tag") == tag:
            print(row.get("acc", ""))
            raise SystemExit(0)
raise SystemExit(1)
PY
)" || acc=""
      if [[ -n "${acc}" ]] && /home/dingxukai/miniconda3/envs/ROT/bin/python - "${acc}" "${FOLLOWUP_THRESHOLD}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
      then
        local followup_dir
        followup_dir="${FOLLOWUP_BASE_OUT:-${artifact_dir}/followup_${tag}}"
        echo "[ckpt-full-eval] acc=${acc} reached threshold=${FOLLOWUP_THRESHOLD}; followup -> ${followup_dir}"
        set +e
        CKPT="${ckpt}" \
          TAG="${tag}" \
          GPU="${GPU}" \
          ARTIFACT_DIR="${followup_dir}" \
          MAX_L="${max_l}" \
          MIN_L="${min_l}" \
          LATENT_TEMP="${latent_temp}" \
          EOL_TEMP="${eol_temp}" \
          COMPRESSION_FACTOR="${cf}" \
          MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
          TEST_TIMES="${TEST_TIMES}" \
          RUN_OOD=1 \
          RUN_VIS=1 \
          RUN_GEOMETRY=1 \
          bash run_trace_v3_candidate_followup_20260707.sh
        local followup_status=$?
        set -e
        if (( followup_status != 0 )); then
          rmdir "${gpu_lock_dir}"
          rmdir "${lock_dir}"
          return "${followup_status}"
        fi
      else
        echo "[ckpt-full-eval] acc=${acc:-NA} below threshold=${FOLLOWUP_THRESHOLD}; skip followup"
      fi
    fi
    rmdir "${gpu_lock_dir}"
    rmdir "${lock_dir}"
  done
}

echo "[ckpt-full-eval] start $(date '+%F %T') family=${FAMILY} dir=${CHECKPOINT_DIR} gpu=${GPU}"
round=0
while (( round < MAX_ROUNDS )); do
  round=$((round + 1))
  echo "[ckpt-full-eval] scan round=${round} $(date '+%F %T')"
  if [[ -d "${CHECKPOINT_DIR}" ]]; then
    while IFS= read -r ckpt; do
      eval_ckpt "${ckpt}"
    done < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'epoch*.ckpt' | sort)
  else
    echo "[ckpt-full-eval] waiting for missing dir=${CHECKPOINT_DIR}"
  fi
  sleep "${SLEEP_SECONDS}"
done

echo "[ckpt-full-eval] done $(date '+%F %T')"
