#!/usr/bin/env bash
set -euo pipefail

# Keep one newly idle card occupied with a real, isolated native-v9 Stage-2
# seed replicate until the four-card supervisor asks us to hand the card over.
# Nothing here writes into the formal run's stage directories.

SOURCE_ROOT=/home/dingxukai/TRACE/trace_role_bridge_native_v9
ROT_PYTHON=/home/dingxukai/miniconda3/envs/ROT/bin/python
RUN_ROOT=${1:?usage: run_gpu_reservation_experiment.sh RUN_ROOT PHYSICAL_GPU}
PHYSICAL_GPU=${2:?usage: run_gpu_reservation_experiment.sh RUN_ROOT PHYSICAL_GPU}
RESERVE_ROOT=${RUN_ROOT}.gpu_reservations/gpu${PHYSICAL_GPU}
LOG_FILE=${RESERVE_ROOT}/reservation.log
TRIAL_COUNTER=${RESERVE_ROOT}/next_trial.txt

[[ "${PHYSICAL_GPU}" =~ ^[0-7]$ ]] || {
  echo "invalid physical GPU: ${PHYSICAL_GPU}" >&2
  exit 2
}

mkdir -p "${RESERVE_ROOT}"
printf '%s\n' "$$" > "${RESERVE_ROOT}/runner.pid"
ps -o lstart= -p "$$" 2>/dev/null | \
  sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
  > "${RESERVE_ROOT}/runner.identity"
ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' \
  > "${RESERVE_ROOT}/runner.pgid"

CHILD_PID=
stop_child() {
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
    wait "${CHILD_PID}" 2>/dev/null || true
  fi
  echo "$(date '+%F %T') action=reservation_stopped gpu=${PHYSICAL_GPU}" \
    >> "${LOG_FILE}"
  exit 0
}
trap stop_child TERM INT HUP

if [[ -s "${TRIAL_COUNTER}" ]]; then
  TRIAL=$(<"${TRIAL_COUNTER}")
else
  TRIAL=1
fi

MAIN_STAGE1=$(<"${RUN_ROOT}/state/stage1_best.txt")
[[ -f "${MAIN_STAGE1}" ]]

while true; do
  printf '%s\n' "$((TRIAL + 1))" > "${TRIAL_COUNTER}"
  TRIAL_ROOT=${RESERVE_ROOT}/trial$(printf '%03d' "${TRIAL}")/lineage
  LOCAL_STAGE1=${TRIAL_ROOT}/stage1/checkpoints/formal-stage1-parent.ckpt
  mkdir -p "${TRIAL_ROOT}/stage1/checkpoints" \
    "${TRIAL_ROOT}/stage2/recovery"

  # A hard link gives the isolated lineage guard a local immutable parent
  # without duplicating a multi-gigabyte checkpoint. Fall back to a reflink or
  # ordinary copy when the filesystem does not support hard links.
  if [[ ! -f "${LOCAL_STAGE1}" ]]; then
    if ! ln "${MAIN_STAGE1}" "${LOCAL_STAGE1}" 2>/dev/null; then
      cp --reflink=auto "${MAIN_STAGE1}" "${LOCAL_STAGE1}"
    fi
  fi

  RESUME_ARGS=()
  if LOCAL_LAST=$(
    "${ROT_PYTHON}" "${SOURCE_ROOT}/scripts/select_last_checkpoint.py" \
      "${TRIAL_ROOT}/stage2" \
      --target src.models.trace_role_native.LitTRACERoleNative \
      --do-trace-rl true 2>/dev/null
  ); then
    RESUME_ARGS=(--resume_ckpt_path "${LOCAL_LAST}")
  fi

  SEED=$((1701 + PHYSICAL_GPU * 100 + TRIAL))
  {
    echo "$(date '+%F %T') action=reservation_trial_start gpu=${PHYSICAL_GPU} trial=${TRIAL} seed=${SEED}"
    echo "purpose=single_gpu_stage2_small_batch_stability_replicate batch_size=1"
    if [[ ${#RESUME_ARGS[@]} -gt 0 ]]; then
      echo "resume=${RESUME_ARGS[1]}"
    fi
  } >> "${LOG_FILE}"

  export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}
  export TRACE_NATIVE_RUN_ROOT=${TRIAL_ROOT}
  export TRACE_NATIVE_LOG_ROOT=${TRIAL_ROOT}/stage2
  # Reservation experiments are disposable auxiliary runs. They must never
  # consume checkpoint quota needed by the formal run.
  unset TRACE_NATIVE_RECOVERY_DIR
  unset TRACE_NATIVE_RECOVERY_EVERY_N_STEPS
  export TOKENIZERS_PARALLELISM=false
  export PYTHONDONTWRITEBYTECODE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  cd "${SOURCE_ROOT}"
  set +e
  "${ROT_PYTHON}" run.py \
    --model trace_role_native_qwen3_instruct \
    --dataset qsa \
    --devices 0 \
    --seed "${SEED}" \
    --load_ckpt_path "${LOCAL_STAGE1}" \
    "${RESUME_ARGS[@]}" \
    --disable_early_stopping \
    --log_suffix "native_v9_reserve_gpu${PHYSICAL_GPU}_trial${TRIAL}" \
    model.model_kwargs.do_trace_rl=true \
    model.training_kwargs.optimizer.lr=8e-7 \
    model.training_kwargs.scheduler.warmup_steps=75 \
    model.training_kwargs.scheduler.num_training_steps=5120 \
    trainer.max_epochs=10 \
    trainer.num_sanity_val_steps=0 \
    trainer.gradient_clip_val=0 \
    trainer.strategy=auto \
    dataloader.batch_size=1 \
    dataloader.val_batch_size=1 \
    save_top_k=0 \
    save_last=false \
    save_weights_only=true \
    >> "${LOG_FILE}" 2>&1 &
  CHILD_PID=$!
  printf '%s\n' "${CHILD_PID}" > "${RESERVE_ROOT}/child.pid"
  wait "${CHILD_PID}"
  STATUS=$?
  CHILD_PID=
  set -e

  echo "$(date '+%F %T') action=reservation_trial_exit gpu=${PHYSICAL_GPU} trial=${TRIAL} status=${STATUS}" \
    >> "${LOG_FILE}"
  if [[ ${STATUS} -eq 0 ]]; then
    TRIAL=$((TRIAL + 1))
  fi
  sleep 5
done
