#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
ENV=/home/dingxukai/miniconda3/envs/ROT
PYTHON=${ENV}/bin/python
STAGE1_CKPT=/disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_284260_20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1_stage1/checkpoints/epoch1__step13452__monitor0.661.ckpt
RUN_TAG=${RUN_TAG:-20260710_trace_bridge_modecontrast_smoke3_gpu0}
GPU_VISIBLE=${GPU_VISIBLE:-0}
TRAIN_DEVICES=${TRAIN_DEVICES:-0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
TRAIN_STRATEGY=${TRAIN_STRATEGY:-auto}
SMOKE_STEPS=${SMOKE_STEPS:-3}
SMOKE_SAMPLES=${SMOKE_SAMPLES:-${SMOKE_STEPS}}
SMOKE_VAL_BATCHES=${SMOKE_VAL_BATCHES:-2}
GRADIENT_GUARD=${GRADIENT_GUARD:-false}
GEOMETRY_GRAD_RATIO=${GEOMETRY_GRAD_RATIO:-0.25}
GEOMETRY_MICRO_BATCH_SIZE=${GEOMETRY_MICRO_BATCH_SIZE:-1}
OUT_DIR=${ROOT}/run_outputs/trace_bridge/${RUN_TAG}
ROOT_DIR=${ROOT}/run_roots/trace_bridge/${RUN_TAG}
LOG_SUFFIX=${RUN_TAG}_stage2_trace_modecontrast

mkdir -p "${OUT_DIR}" "${ROOT_DIR}"
cd "${ROOT}"

TMPDIR=/tmp TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}" \
"${PYTHON}" run.py \
  --model trace_bridge_qwen3_instruct_vizstrong \
  --dataset qsa \
  --trainer default \
  --devices "${TRAIN_DEVICES}" \
  --workspace_path /home/dingxukai \
  --load_ckpt_path "${STAGE1_CKPT}" \
  --log_suffix "${LOG_SUFFIX}" \
  dataset_dir=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc \
  batch_size="${TRAIN_BATCH_SIZE}" \
  val_batch_size=1 \
  num_workers=2 \
  persistent_workers=false \
  do_trace_rl=true \
  trainer.num_sanity_val_steps=0 \
  trainer.max_epochs=1 \
  trainer.max_steps="${SMOKE_STEPS}" \
  trainer.limit_val_batches="${SMOKE_VAL_BATCHES}" \
  trainer.log_every_n_steps=1 \
  trainer.val_check_interval=1.0 \
  trainer.check_val_every_n_epoch=1 \
  trainer.gradient_clip_val=0 \
  trainer.strategy="${TRAIN_STRATEGY}" \
  trainer.default_root_dir="${ROOT_DIR}" \
  model.model_kwargs.trace_bridge_config.save_trace_visual_info=false \
  model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
  model.model_kwargs.trace_rl_config.n_train_samples_per_epoch="${SMOKE_SAMPLES}" \
  model.model_kwargs.trace_rl_config.group_size=8 \
  model.model_kwargs.trace_rl_config.exp_batch_size=1 \
  model.model_kwargs.trace_rl_config.temperature=0.95 \
  model.model_kwargs.trace_rl_config.top_p=0.97 \
  model.model_kwargs.trace_rl_config.clip_eps=0.12 \
  model.model_kwargs.trace_rl_config.stage2_latent_noise_scale=0.015 \
  model.model_kwargs.trace_rl_config.stage2_center_signatures=true \
  model.model_kwargs.trace_rl_config.stage2_signature_raw_mix=0.25 \
  model.model_kwargs.trace_rl_config.trace_reward_weight=0.0 \
  model.model_kwargs.trace_rl_config.max_modes=3 \
  model.model_kwargs.trace_rl_config.mode_merge_threshold=0.65 \
  model.model_kwargs.trace_rl_config.output_length_penalty_weight=0.03 \
  model.model_kwargs.trace_rl_config.target_output_length=33.5 \
  model.model_kwargs.trace_rl_config.stage2_direct_signature_weight=0.12 \
  model.model_kwargs.trace_rl_config.stage2_direct_pos_weight=1.0 \
  model.model_kwargs.trace_rl_config.stage2_direct_neg_weight=1.0 \
  model.model_kwargs.trace_rl_config.stage2_direct_wrong_div_weight=0.05 \
  model.model_kwargs.trace_rl_config.stage2_direct_neg_margin=0.15 \
  model.model_kwargs.trace_rl_config.stage2_direct_wrong_div_margin=0.65 \
  model.model_kwargs.trace_rl_config.stage2_direct_inter_mode_weight=0.15 \
  model.model_kwargs.trace_rl_config.stage2_direct_inter_mode_margin=0.55 \
  model.model_kwargs.trace_rl_config.stage2_accuracy_gradient_guard="${GRADIENT_GUARD}" \
  model.model_kwargs.trace_rl_config.stage2_geometry_grad_ratio="${GEOMETRY_GRAD_RATIO}" \
  model.model_kwargs.trace_rl_config.stage2_geometry_micro_batch_size="${GEOMETRY_MICRO_BATCH_SIZE}" \
  model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.05 \
  model.training_kwargs.optimizer.lr=8e-7 \
  model.training_kwargs.scheduler.warmup_steps=1 \
  model.training_kwargs.scheduler.num_training_steps="${SMOKE_STEPS}" \
  2>&1 | tee "${OUT_DIR}/train.log"

EVENT_DIR=$(find "${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm" -maxdepth 1 -type d -name "*${LOG_SUFFIX}" | sort | tail -n 1)
TRAINED_CKPT=$(find "${EVENT_DIR}/checkpoints" -maxdepth 1 -type f -name 'last.ckpt' | head -n 1)

"${ENV}/bin/python" tools/trace_bridge_training_gate.py \
  --base_ckpt "${STAGE1_CKPT}" \
  --trained_ckpt "${TRAINED_CKPT}" \
  --event_dir "${EVENT_DIR}" \
  --out "${OUT_DIR}/training_gate.json"
