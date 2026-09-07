#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
ENV=/home/dingxukai/miniconda3/envs/ROT
GPU="${GPU:-2}"
OUT=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/geometry_replication_seed1
LOG_DIR=${OUT}/logs
RECORD=${LOG_DIR}/tb/run/trace_bridge_visual_test.pt
CKPT=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260711-174000_788028_20260711_trace_bridge_final_guarded_saveall_v6_matched_gpu3457_stage2_trace_guarded/checkpoints/epoch7__step4096__monitor0.719251.ckpt
DATASET_DIR=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

if [[ ! -s "${RECORD}" ]]; then
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" \
    "${ENV}/bin/python" run.py \
      --model trace_bridge_qwen3_instruct_vizstrong \
      --dataset qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /home/dingxukai \
      --test_ckpt_path "${CKPT}" \
      --test_times 1 \
      --seed 1 \
      dataset_dir="${DATASET_DIR}" \
      batch_size=1 \
      val_batch_size=1 \
      num_workers=2 \
      persistent_workers=false \
      do_trace_rl=true \
      trainer.num_sanity_val_steps=0 \
      trainer.strategy=auto \
      trainer.limit_test_batches=200 \
      trainer.default_root_dir="${OUT}/root" \
      trainer.logger.save_dir="${LOG_DIR}" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
      model.model_kwargs.trace_rl_config.exp_batch_size=4 \
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=true \
      model.model_kwargs.trace_bridge_config.trace_visual_group_views=8 \
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=200 \
      model.model_kwargs.trace_bridge_config.trace_visual_latent_noise_scale=0.015 \
      model.model_kwargs.trace_bridge_config.trace_visual_noise_seed=1 \
      model.model_kwargs.trace_bridge_config.trace_visual_do_sample=true \
      model.model_kwargs.trace_bridge_config.trace_visual_temperature=0.95 \
      model.model_kwargs.trace_bridge_config.trace_visual_top_p=0.97 \
      2>&1 | tee "${OUT}/eval.log"
fi

test -s "${RECORD}"
"${ENV}/bin/python" tools/trace_bridge_geometry_summary.py \
  --records "${RECORD}" \
  --out_dir "${OUT}/geometry_raw" \
  --max_records 200 \
  > "${OUT}/geometry_raw.log"
"${ENV}/bin/python" tools/trace_bridge_geometry_summary.py \
  --records "${RECORD}" \
  --out_dir "${OUT}/geometry_stage2_metric_null1024" \
  --max_records 200 \
  --signature_representation stage2_centered \
  --signature_raw_mix 0.25 \
  --permutation_null_trials 1024 \
  > "${OUT}/geometry_stage2_metric_null1024.log"

printf 'Epoch7 200-question geometry replication seed1 completed on GPU%s.\n' "${GPU}" > "${OUT}/done.txt"
