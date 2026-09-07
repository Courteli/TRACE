#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
ENV=/home/dingxukai/miniconda3/envs/ROT
GPU="${GPU:-0}"
OUT=${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence/matched_bridge_rollouts
LOG_DIR=${OUT}/logs
RECORD=${LOG_DIR}/tb/run/trace_bridge_visual_test.pt
CKPT=${ROOT}/logs/bridge_qwen3_instruct_hybrid_compact_anchor_gate/qsa-gsm/20260708-153949_757553_20260708_trace_bridge_bridgefull_bridge_baseline_gpu2_bridge_baseline/checkpoints/epoch0__step6726__monitor0.655.ckpt
DATASET_DIR=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc

mkdir -p "${LOG_DIR}"
if [[ -s "${RECORD}" ]]; then
  exit 0
fi

cd "${ROOT}"
TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" \
  "${ENV}/bin/python" run.py \
    --model bridge_qwen3_instruct_hybrid_compact_anchor_gate \
    --dataset qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /home/dingxukai \
    --test_ckpt_path "${CKPT}" \
    --test_times 1 \
    dataset_dir="${DATASET_DIR}" \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=2 \
    persistent_workers=false \
    trainer.num_sanity_val_steps=0 \
    trainer.strategy=auto \
    trainer.limit_test_batches=200 \
    trainer.default_root_dir="${OUT}/root" \
    trainer.logger.save_dir="${LOG_DIR}" \
    trainer.logger.name=tb \
    trainer.logger.version=run \
    model.target=src.models.trace_bridge.LitTRACEBridge \
    model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
    'model.model_kwargs.trace_bridge_config={"use_trace_view_embeddings":false,"use_trace_step_view_embeddings":false,"save_trace_visual_info":true,"trace_visual_group_views":8,"trace_visual_record_limit":200,"trace_visual_micro_batch_size":4,"trace_visual_latent_noise_scale":0.015,"trace_visual_noise_seed":0,"trace_visual_do_sample":true,"trace_visual_temperature":0.95,"trace_visual_top_p":0.97,"rl_signature_source":"residuals"}' \
    2>&1 | tee "${OUT}/eval_matched200.log"

test -s "${RECORD}"
printf 'BRIDGE matched 200-question recorder completed on GPU%s.\n' "${GPU}" > "${OUT}/matched200_done.txt"
