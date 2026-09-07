#!/usr/bin/env bash
set -euo pipefail

# Packaging-only change: resolve local paths without changing experiment settings.
SOURCE_ROOT=${TRACE_SOURCE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
REPO_ROOT=$(cd -- "${SOURCE_ROOT}/../.." && pwd)
ROT_PYTHON=${TRACE_PYTHON:-python}
export TRACE_DATA_ROOT=${TRACE_DATA_ROOT:-${REPO_ROOT}/data}
: "${TRACE_MODEL_PATH:?Set TRACE_MODEL_PATH to the complete original base-model directory}"
PHYSICAL_GPUS=${PHYSICAL_GPUS:-0,1,2,3}
RUN_ROOT=${1:-${REPO_ROOT}/runs/role_native_v9_$(date +%Y%m%d-%H%M%S)}
STATE_ROOT=${RUN_ROOT}/state

IFS=',' read -r -a GPU_ARRAY <<< "${PHYSICAL_GPUS}"
if [[ ${#GPU_ARRAY[@]} -ne 4 ]]; then
  echo "Formal native v9 training requires exactly four physical GPUs." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${STATE_ROOT}"
if [[ ! -d "${RUN_ROOT}/code_snapshot" ]]; then
  cp -a "${SOURCE_ROOT}" "${RUN_ROOT}/code_snapshot"
else
  # A repaired source tree is picked up on the next supervised retry while
  # checkpoints and the fresh-run lineage remain untouched.
  cp -a "${SOURCE_ROOT}/." "${RUN_ROOT}/code_snapshot/"
fi
TRAIN_ROOT=${RUN_ROOT}/code_snapshot
export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPUS}
export TRACE_NATIVE_RUN_ROOT=${RUN_ROOT}
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${TRAIN_ROOT}"

echo "run_root=${RUN_ROOT}"
echo "physical_gpus=${PHYSICAL_GPUS}"
echo "lineage=base Qwen3-4B -> fresh Stage0 -> fresh Stage1 -> fresh Stage2"

latest_last() {
  local stage_root=$1
  local target=$2
  local do_rl=$3
  "${ROT_PYTHON}" scripts/select_last_checkpoint.py \
    "${stage_root}" --target "${target}" --do-trace-rl "${do_rl}" 2>/dev/null
}

if [[ -f "${STATE_ROOT}/stage0.done" ]]; then
  STAGE0_CKPT=$(<"${STATE_ROOT}/stage0_best.txt")
  [[ -f "${STAGE0_CKPT}" ]]
else
  RESUME_ARGS=()
  if STAGE0_LAST=$(latest_last "${RUN_ROOT}/stage0" src.models.cot.LitCot false); then
    RESUME_ARGS=(--resume_ckpt_path "${STAGE0_LAST}")
    echo "stage0_resume=${STAGE0_LAST}"
  fi
  export TRACE_NATIVE_LOG_ROOT=${RUN_ROOT}/stage0
  export TRACE_NATIVE_RECOVERY_DIR=${RUN_ROOT}/stage0/recovery
  export TRACE_NATIVE_RECOVERY_EVERY_N_STEPS=100
  "${ROT_PYTHON}" run.py \
    --model cot_qwen3_instruct \
    --dataset qsa \
    --devices 0,1,2,3 \
    --seed 1701 \
    "${RESUME_ARGS[@]}" \
    --log_suffix native_v9_stage0_cot \
    trainer.max_epochs=3 \
    trainer.num_sanity_val_steps=0 \
    trainer.strategy=ddp_find_unused_parameters_true \
    dataloader.batch_size=4 \
    dataloader.val_batch_size=1 \
    save_weights_only=false \
    2>&1 | tee -a "${RUN_ROOT}/stage0.log"
  STAGE0_CKPT=$("${ROT_PYTHON}" scripts/select_best_checkpoint.py \
    "${RUN_ROOT}/stage0" --target src.models.cot.LitCot --do-trace-rl false)
  printf '%s\n' "${STAGE0_CKPT}" > "${STATE_ROOT}/stage0_best.txt"
  touch "${STATE_ROOT}/stage0.done"
fi
echo "stage0_best=${STAGE0_CKPT}"

if [[ -f "${STATE_ROOT}/stage1.done" ]]; then
  STAGE1_CKPT=$(<"${STATE_ROOT}/stage1_best.txt")
  [[ -f "${STAGE1_CKPT}" ]]
else
  RESUME_ARGS=()
  if STAGE1_LAST=$(latest_last "${RUN_ROOT}/stage1" src.models.trace_role_native.LitTRACERoleNative false); then
    RESUME_ARGS=(--resume_ckpt_path "${STAGE1_LAST}")
    echo "stage1_resume=${STAGE1_LAST}"
  fi
  export TRACE_NATIVE_LOG_ROOT=${RUN_ROOT}/stage1
  export TRACE_NATIVE_RECOVERY_DIR=${RUN_ROOT}/stage1/recovery
  export TRACE_NATIVE_RECOVERY_EVERY_N_STEPS=100
  "${ROT_PYTHON}" run.py \
    --model trace_role_native_qwen3_instruct \
    --dataset qsa \
    --devices 0,1,2,3 \
    --seed 1701 \
    --load_ckpt_path "${STAGE0_CKPT}" \
    "${RESUME_ARGS[@]}" \
    --disable_early_stopping \
    --log_suffix native_v9_stage1_roles \
    trainer.max_epochs=10 \
    trainer.num_sanity_val_steps=0 \
    trainer.strategy=ddp_find_unused_parameters_true \
    dataloader.batch_size=4 \
    dataloader.val_batch_size=1 \
    save_weights_only=false \
    2>&1 | tee -a "${RUN_ROOT}/stage1.log"
  STAGE1_CKPT=$("${ROT_PYTHON}" scripts/select_best_checkpoint.py \
    "${RUN_ROOT}/stage1" \
    --target src.models.trace_role_native.LitTRACERoleNative \
    --do-trace-rl false)
  printf '%s\n' "${STAGE1_CKPT}" > "${STATE_ROOT}/stage1_best.txt"
  touch "${STATE_ROOT}/stage1.done"
fi
echo "stage1_best=${STAGE1_CKPT}"

if [[ -f "${STATE_ROOT}/stage2.done" ]]; then
  STAGE2_CKPT=$(<"${STATE_ROOT}/stage2_best.txt")
  [[ -f "${STAGE2_CKPT}" ]]
else
  RESUME_ARGS=()
  if STAGE2_LAST=$(latest_last "${RUN_ROOT}/stage2" src.models.trace_role_native.LitTRACERoleNative true); then
    RESUME_ARGS=(--resume_ckpt_path "${STAGE2_LAST}")
    echo "stage2_resume=${STAGE2_LAST}"
  elif [[ -f "${RUN_ROOT}/stage2/recovery/last.ckpt" ]] || [[ "${TRACE_REQUIRE_STAGE2_RESUME:-0}" == 1 ]]; then
    echo "Stage2 recovery validation failed; refusing to restart Stage2 from initialization." >&2
    exit 1
  fi
  export TRACE_NATIVE_LOG_ROOT=${RUN_ROOT}/stage2
  export TRACE_NATIVE_RECOVERY_DIR=${RUN_ROOT}/stage2/recovery
  export TRACE_NATIVE_RECOVERY_EVERY_N_STEPS=100
  "${ROT_PYTHON}" run.py \
    --model trace_role_native_qwen3_instruct \
    --dataset qsa \
    --devices 0,1,2,3 \
    --seed 1701 \
    --load_ckpt_path "${STAGE1_CKPT}" \
    "${RESUME_ARGS[@]}" \
    --disable_early_stopping \
    --log_suffix native_v9_stage2_joint_rl \
    model.model_kwargs.do_trace_rl=true \
    model.training_kwargs.optimizer.lr=8e-7 \
    model.training_kwargs.scheduler.warmup_steps=75 \
    model.training_kwargs.scheduler.num_training_steps=5120 \
    trainer.max_epochs=10 \
    trainer.num_sanity_val_steps=0 \
    trainer.gradient_clip_val=0 \
    trainer.strategy=ddp_find_unused_parameters_true \
    dataloader.batch_size=4 \
    dataloader.val_batch_size=1 \
    save_weights_only=false \
    2>&1 | tee -a "${RUN_ROOT}/stage2.log"
  STAGE2_CKPT=$("${ROT_PYTHON}" scripts/select_best_checkpoint.py \
    "${RUN_ROOT}/stage2" \
    --target src.models.trace_role_native.LitTRACERoleNative \
    --do-trace-rl true)
  printf '%s\n' "${STAGE2_CKPT}" > "${STATE_ROOT}/stage2_best.txt"
  touch "${STATE_ROOT}/stage2.done"
fi
echo "stage2_best=${STAGE2_CKPT}"

if [[ ! -f "${STATE_ROOT}/evaluation.done" ]]; then
  unset TRACE_NATIVE_RECOVERY_DIR
  unset TRACE_NATIVE_RECOVERY_EVERY_N_STEPS
  # Training-time four-rank validation pads 747 questions to 748.  The final
  # protocol instead runs five audited one-rank passes over the validation
  # split, so every reported result is exactly correct_count / 747.
  CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} "${ROT_PYTHON}" \
    "${SOURCE_ROOT}/tools/run_native_v9_strict_validation.py" \
    --model-root "${TRAIN_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --checkpoint "${STAGE2_CKPT}" \
    --output-dir "${RUN_ROOT}/strict_validation" \
    --replications 5 \
    --seed 1701 \
    --expected-questions 747 \
    2>&1 | tee -a "${RUN_ROOT}/stage2_evaluation.log"
  [[ -s "${RUN_ROOT}/strict_validation/strict_validation_summary.json" ]]
  touch "${STATE_ROOT}/evaluation.done"
fi

"${ROT_PYTHON}" scripts/verify_lineage.py \
  --run-root "${RUN_ROOT}" \
  --stage0 "${STAGE0_CKPT}" \
  --stage1 "${STAGE1_CKPT}" \
  --stage2 "${STAGE2_CKPT}" \
  --output "${RUN_ROOT}/lineage_manifest.json" \
  2>&1 | tee -a "${RUN_ROOT}/lineage_verification.log"

touch "${STATE_ROOT}/complete.done"
echo "native_v9_complete=${RUN_ROOT}"
