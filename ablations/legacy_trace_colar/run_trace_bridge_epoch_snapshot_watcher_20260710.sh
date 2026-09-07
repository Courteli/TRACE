#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
SNAPSHOT_ROOT=${ROOT}/run_outputs/trace_bridge/20260710_trace_bridge_epoch_snapshots
TRACE_EVENT=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260710-180018_287694_20260710_trace_bridge_final_streaming_modecontrast_gpu0123_stage2_trace_modecontrast
ANSWER_EVENT=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260710-174013_167099_20260710_trace_bridge_final_vizstrong_answeronly_gpu457_stage2_answer_only

mkdir -p "${SNAPSHOT_ROOT}/trace" "${SNAPSHOT_ROOT}/answer_only"

capture_last_checkpoint() {
  local label="$1"
  local event_dir="$2"
  local out_dir="$3"
  local checkpoint="${event_dir}/checkpoints/last.ckpt"
  local state_file="${out_dir}/last_seen.state"
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
  local target="${out_dir}/${label}_epoch${epoch}_step${step}.ckpt"
  if [[ ! -f "${target}" ]]; then
    cp --reflink=auto --preserve=timestamps "${checkpoint}" "${target}"
    printf '%s epoch=%s step=%s source=%s target=%s\n' \
      "$(date '+%F %T')" "${epoch}" "${step}" "${checkpoint}" "${target}" \
      >> "${SNAPSHOT_ROOT}/snapshot_manifest.log"
  fi
  printf '%s\n' "${fingerprint}" > "${state_file}"
}

while true; do
  capture_last_checkpoint trace "${TRACE_EVENT}" "${SNAPSHOT_ROOT}/trace"
  capture_last_checkpoint answer_only "${ANSWER_EVENT}" "${SNAPSHOT_ROOT}/answer_only"

  if ! tmux has-session -t trace_final_streaming_mode_gpu0123_0710 2>/dev/null \
    && ! tmux has-session -t trace_final_vizstrong_answer_gpu457_0710 2>/dev/null; then
    break
  fi
  sleep 300
done

capture_last_checkpoint trace "${TRACE_EVENT}" "${SNAPSHOT_ROOT}/trace"
capture_last_checkpoint answer_only "${ANSWER_EVENT}" "${SNAPSHOT_ROOT}/answer_only"
printf 'Epoch snapshot watcher completed.\n' > "${SNAPSHOT_ROOT}/watcher_done.txt"
