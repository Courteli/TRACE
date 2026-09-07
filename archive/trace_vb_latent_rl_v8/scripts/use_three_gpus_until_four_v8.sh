#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=trace_vb_common.sh
source "${SCRIPT_DIR}/trace_vb_common.sh"

HELD_GPUS=${TRACE_VB_HELD_GPUS-4,5,7}
SCAN_GPUS=${TRACE_VB_SCAN_GPUS:-0,1,2,3,4,5,6,7}
POLL_SECONDS=${POLL_SECONDS:-10}
STABLE_POLLS=${TRACE_VB_STABLE_POLLS:-2}
TRAIN_SEED=${TRAIN_SEED:-0}
PIPELINE_TAG=${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)_trace_vb_v8_train_seed${TRAIN_SEED}}
CONTROL_TAG=${CONTROL_TAG:-$(date +%Y%m%d-%H%M%S)_v7_validation_curve_until_v8}
V7_STAGE1_DIR=${V7_STAGE1_DIR:-/disk1/dingxukai/TRACE/trace_vb_v7_runs/training/20260818-213000_trace_vb_v7_full_seed0_stage1}
RESUME_PHASE=${TRACE_VB_RESUME_PHASE:-}
RECOVERY_CHECKPOINT=${TRACE_VB_RECOVERY_CHECKPOINT:-}

trace_vb_require_safe_tag "${PIPELINE_TAG}"
trace_vb_require_safe_tag "${CONTROL_TAG}"
[[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] && (( POLL_SECONDS >= 10 )) || trace_vb_die "invalid poll interval"
[[ "${STABLE_POLLS}" =~ ^[1-9][0-9]*$ ]] || trace_vb_die "invalid stable poll count"
[[ -d "${V7_STAGE1_DIR}" ]] || trace_vb_die "missing formal v7 Stage-1 directory"
if [[ -n "${RESUME_PHASE}" ]]; then
  [[ "${RESUME_PHASE}" == stage1 || "${RESUME_PHASE}" == stage2 ]] || \
    trace_vb_die "TRACE_VB_RESUME_PHASE must be stage1 or stage2"
  [[ -f "${RECOVERY_CHECKPOINT}" ]] || \
    trace_vb_die "TRACE_VB_RECOVERY_CHECKPOINT is required for resume handoff"
  [[ -n "${V7_BEST_CKPT:-}" && -f "${V7_BEST_CKPT}" ]] || \
    trace_vb_die "V7_BEST_CKPT is required for resume handoff"
fi

held_gpus=()
if [[ -n "${HELD_GPUS}" ]]; then
  IFS=',' read -r -a held_gpus <<< "${HELD_GPUS}"
fi
IFS=',' read -r -a scan_gpus <<< "${SCAN_GPUS}"
(( ${#held_gpus[@]} <= 3 )) || trace_vb_die "at most three initially held GPUs are allowed"
if (( ${#held_gpus[@]} > 0 )); then
  [[ "$(printf '%s\n' "${held_gpus[@]}" | sort -u | wc -l)" -eq "${#held_gpus[@]}" ]] || \
    trace_vb_die "held GPUs must be unique"
fi

declare -A is_held=()
for gpu in "${held_gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || trace_vb_die "invalid held GPU ${gpu}"
  is_held[${gpu}]=1
done
for gpu in "${scan_gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || trace_vb_die "invalid scanned GPU ${gpu}"
done

control_root=${TRACE_VB_ARTIFACT_ROOT}/useful_controls/${CONTROL_TAG}
mkdir -p "${control_root}"
exec > >(tee -a "${control_root}/controller.log") 2>&1
echo "$(date --iso-8601=seconds) starting useful controls on ${HELD_GPUS}; scan set=${SCAN_GPUS}"

# Include this controller PID so cleanup from a replaced controller can never
# signal a newly launched worker that happens to use the same GPU and tag.
session_prefix=trace_v8_ctl_${CONTROL_TAG}_$$
worker_sessions=()
worker_gpus=()
queues=(early middle late)
next_queue_index=0
controller_handoff=false

launch_worker() {
  local gpu=$1
  local queue=${queues[$((next_queue_index % ${#queues[@]}))]}
  local free util session pane_pid
  read -r free util < <(
    nvidia-smi -i "${gpu}" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits |
      awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
  )
  if (( free < TRACE_VB_MIN_FREE_GPU_MIB )); then
    echo "$(date --iso-8601=seconds) skipped raced GPU ${gpu}: only ${free} MiB free"
    return 1
  fi
  if (( util > TRACE_VB_MAX_IDLE_UTILIZATION )); then
    echo "$(date --iso-8601=seconds) skipped raced GPU ${gpu}: utilization ${util}%"
    return 1
  fi
  session=${session_prefix}_gpu${gpu}
  trace_vb_require_safe_tag "${session}"
  tmux has-session -t "${session}" 2>/dev/null && trace_vb_die "worker session already exists: ${session}"
  tmux new-session -d -s "${session}" \
    env TRACE_VB_CONTROL_WORKER=1 \
    bash "${SCRIPT_DIR}/run_v7_validation_queue_v8.sh" \
      "${gpu}" "${queue}" "${CONTROL_TAG}" "${control_root}"
  pane_pid=$(tmux display-message -p -t "${session}" '#{pane_pid}')
  printf '%s\n' "${pane_pid}" > "${control_root}/gpu${gpu}_${queue}.pane_pid"
  worker_sessions+=("${session}")
  worker_gpus+=("${gpu}")
  next_queue_index=$((next_queue_index + 1))
  echo "$(date --iso-8601=seconds) launched ${queue} queue session=${session} pane_pid=${pane_pid}"
}

stop_workers() {
  local session
  for session in "${worker_sessions[@]}"; do
    tmux has-session -t "${session}" 2>/dev/null && tmux send-keys -t "${session}" C-c || true
  done
  for _ in $(seq 1 25); do
    local any=false
    for session in "${worker_sessions[@]}"; do
      tmux has-session -t "${session}" 2>/dev/null && any=true
    done
    [[ "${any}" == false ]] && break
    sleep 1
  done
  for session in "${worker_sessions[@]}"; do
    tmux has-session -t "${session}" 2>/dev/null && tmux kill-session -t "${session}" || true
  done
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ "${controller_handoff}" != true ]]; then
    stop_workers
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM HUP

for gpu in "${held_gpus[@]}"; do
  if ! launch_worker "${gpu}"; then
    unset 'is_held['"${gpu}"']'
  fi
done
held_gpus=("${worker_gpus[@]}")
HELD_GPUS=$(IFS=,; printf '%s' "${held_gpus[*]}")

eligible_signature=
stable_count=0
last_signature=
while true; do
  active_sessions=()
  active_gpus=()
  for index in "${!worker_sessions[@]}"; do
    session=${worker_sessions[${index}]}
    gpu=${worker_gpus[${index}]}
    if tmux has-session -t "${session}" 2>/dev/null; then
      active_sessions+=("${session}")
      active_gpus+=("${gpu}")
    else
      echo "$(date --iso-8601=seconds) releasing finished worker session=${session} gpu=${gpu}"
      unset 'is_held['"${gpu}"']'
    fi
  done
  worker_sessions=("${active_sessions[@]}")
  worker_gpus=("${active_gpus[@]}")
  held_gpus=("${active_gpus[@]}")
  HELD_GPUS=$(IFS=,; printf '%s' "${held_gpus[*]}")

  eligible=()
  status_parts=()
  for gpu in "${scan_gpus[@]}"; do
    [[ -n "${is_held[${gpu}]+x}" ]] && continue
    read -r free util < <(
      nvidia-smi -i "${gpu}" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits |
        awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
    )
    status_parts+=("${gpu}:${free}MiB:${util}%")
    if (( free >= TRACE_VB_MIN_FREE_GPU_MIB )) && (( util <= TRACE_VB_MAX_IDLE_UTILIZATION )); then
      eligible+=("${gpu}")
    fi
  done
  worker_parts=()
  for session in "${worker_sessions[@]}"; do
    if tmux has-session -t "${session}" 2>/dev/null; then worker_parts+=("${session}:running"); else worker_parts+=("${session}:finished"); fi
  done
  signature="external=${status_parts[*]} workers=${worker_parts[*]}"
  if [[ "${signature}" != "${last_signature}" ]]; then
    echo "$(date --iso-8601=seconds) ${signature}"
    last_signature=${signature}
  fi
  next_signature=$(IFS=,; printf '%s' "${eligible[*]:-}")
  if [[ -n "${next_signature}" && "${next_signature}" == "${eligible_signature}" ]]; then
    stable_count=$((stable_count + 1))
  elif [[ -n "${next_signature}" ]]; then
    eligible_signature=${next_signature}
    stable_count=1
  else
    eligible_signature=
    stable_count=0
  fi

  if [[ -n "${eligible_signature}" ]] && (( stable_count >= STABLE_POLLS )); then
    needed=$((4 - ${#held_gpus[@]}))
    if (( ${#eligible[@]} >= needed )); then
      if [[ -n "${RESUME_PHASE}" ]]; then
        # A formal resume is cryptographically bound to the original ordered
        # physical-GPU string.  Reconstruct that canonical order instead of
        # using the order in which individual cards happened to become free.
        IFS=',' read -r -a selected <<< "${TRACE_VB_FORMAL_GPUS}"
      else
        selected=("${held_gpus[@]}")
        for ((index=0; index<needed; index++)); do selected+=("${eligible[${index}]}"); done
      fi
      selected_csv=$(IFS=,; printf '%s' "${selected[*]}")
      echo "$(date --iso-8601=seconds) enough external GPUs stable for ${stable_count} polls; stopping only managed controls"
      stop_workers
      echo "$(date --iso-8601=seconds) managed controls stopped; handing off formal GPUs ${selected_csv}"
      printf '%s\n' "${selected_csv}" > "${control_root}/formal_gpu_handoff.txt"
      controller_handoff=true
      trap - EXIT INT TERM HUP
      export TRACE_VB_FORMAL_GPUS=${selected_csv}
      export TRACE_VB_FIXED_GPUS=${selected_csv}
      export TRACE_VB_SUPERVISOR_SESSION=${TMUX:+$(tmux display-message -p '#S')}
      export PIPELINE_TAG TRAIN_SEED POLL_SECONDS V7_STAGE1_DIR V7_BEST_CKPT
      if [[ -n "${RESUME_PHASE}" ]]; then
        exec bash "${SCRIPT_DIR}/wait_for_four_gpus_and_resume_train_only_v8.sh" \
          "${selected_csv}" "${PIPELINE_TAG}" \
          "${RESUME_PHASE}" "${RECOVERY_CHECKPOINT}"
      fi
      exec bash "${SCRIPT_DIR}/wait_for_four_gpus_and_run_train_only_v8.sh"
    fi
    if (( ${#held_gpus[@]} < 3 )); then
      available_slots=$((3 - ${#held_gpus[@]}))
      launch_count=${#eligible[@]}
      (( launch_count <= available_slots )) || launch_count=${available_slots}
      for ((index=0; index<launch_count; index++)); do
        new_gpu=${eligible[${index}]}
        echo "$(date --iso-8601=seconds) adding newly free GPU ${new_gpu} to useful controls"
        if launch_worker "${new_gpu}"; then
          held_gpus+=("${new_gpu}")
          is_held[${new_gpu}]=1
        fi
      done
      HELD_GPUS=$(IFS=,; printf '%s' "${held_gpus[*]}")
      eligible_signature=
      stable_count=0
    fi
  fi
  sleep "${POLL_SECONDS}"
done
