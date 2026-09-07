#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
SESSION=${TRACE_SESSION:-trace_final_guarded_gpu0123_0711}
EVENT_PATTERN=${TRACE_EVENT_PATTERN:-'*20260711_trace_bridge_final_guarded_gpu0123_stage2_trace_guarded'}
SNAPSHOT_ROOT=${TRACE_SNAPSHOT_ROOT:-${ROOT}/run_outputs/trace_bridge/20260711_trace_bridge_guarded_epoch_snapshots}
LABEL=${TRACE_SNAPSHOT_LABEL:-trace}
EVENT_ROOT=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm

mkdir -p "${SNAPSHOT_ROOT}/${LABEL}"

find_event_dir() {
  find "${EVENT_ROOT}" -maxdepth 1 -type d -name "${EVENT_PATTERN}" -print 2>/dev/null \
    | sort \
    | tail -n 1
}

capture_named_checkpoints() {
  local event_dir="$1"
  local checkpoint name epoch step target before after

  while IFS= read -r checkpoint; do
    name="$(basename "${checkpoint}")"
    if [[ ! "${name}" =~ ^epoch([0-9]+)__step([0-9]+)__monitor.*\.ckpt$ ]]; then
      continue
    fi
    epoch="${BASH_REMATCH[1]}"
    step="${BASH_REMATCH[2]}"
    target="${SNAPSHOT_ROOT}/${LABEL}/${LABEL}_epoch${epoch}_step${step}.ckpt"
    [[ ! -f "${target}" ]] || continue

    before="$(stat -c '%Y:%s' "${checkpoint}")"
    sleep 5
    after="$(stat -c '%Y:%s' "${checkpoint}")"
    [[ "${before}" == "${after}" ]] || continue

    ln "${checkpoint}" "${target}"
    printf '%s epoch=%s step=%s source=%s target=%s kind=named-hardlink\n' \
      "$(date '+%F %T')" "${epoch}" "${step}" "${checkpoint}" "${target}" \
      >> "${SNAPSHOT_ROOT}/snapshot_manifest.log"
  done < <(
    find "${event_dir}/checkpoints" -maxdepth 1 -type f \
      -name 'epoch*__step*__monitor*.ckpt' -print 2>/dev/null | sort -V
  )
}

capture_last_checkpoint() {
  local event_dir="$1"
  local checkpoint="${event_dir}/checkpoints/last.ckpt"
  local state_file="${SNAPSHOT_ROOT}/${LABEL}/last_seen.state"
  [[ -f "${checkpoint}" ]] || return 0

  local fingerprint
  fingerprint="$(stat -c '%Y:%s' "${checkpoint}")"
  local previous=""
  if [[ -f "${state_file}" ]]; then
    IFS= read -r previous < "${state_file}"
  fi
  [[ "${fingerprint}" != "${previous}" ]] || return 0

  sleep 20
  [[ "${fingerprint}" == "$(stat -c '%Y:%s' "${checkpoint}")" ]] || return 0

  local metadata
  metadata="$(${PYTHON} -c "import torch; x=torch.load('${checkpoint}', map_location='cpu', weights_only=False); print(f\"{int(x.get('epoch', -1))}:{int(x.get('global_step', -1))}\")")"
  local epoch="${metadata%%:*}"
  local step="${metadata##*:}"
  local target="${SNAPSHOT_ROOT}/${LABEL}/${LABEL}_epoch${epoch}_step${step}.ckpt"
  if [[ ! -f "${target}" ]]; then
    ln "${checkpoint}" "${target}"
    printf '%s epoch=%s step=%s source=%s target=%s kind=last-hardlink\n' \
      "$(date '+%F %T')" "${epoch}" "${step}" "${checkpoint}" "${target}" \
      >> "${SNAPSHOT_ROOT}/snapshot_manifest.log"
  fi
  printf '%s\n' "${fingerprint}" > "${state_file}"
}

event_dir=""
while true; do
  if [[ -z "${event_dir}" ]]; then
    event_dir="$(find_event_dir)"
  fi
  if [[ -n "${event_dir}" ]]; then
    capture_named_checkpoints "${event_dir}"
    capture_last_checkpoint "${event_dir}"
  fi
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    break
  fi
  sleep 120
done

if [[ -z "${event_dir}" ]]; then
  printf '%s event directory was never created.\n' "${LABEL}" > "${SNAPSHOT_ROOT}/${LABEL}_watcher_failed.txt"
  exit 1
fi

capture_named_checkpoints "${event_dir}"
capture_last_checkpoint "${event_dir}"
printf '%s\n' "${event_dir}" > "${SNAPSHOT_ROOT}/event_dir.txt"
printf '%s epoch snapshot watcher completed.\n' "${LABEL}" > "${SNAPSHOT_ROOT}/${LABEL}_watcher_done.txt"
