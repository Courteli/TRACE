#!/usr/bin/env bash
set -euo pipefail

ROOT=/disk1/dingxukai/trace_colar
ENV=/home/dingxukai/miniconda3/envs/ROT
GPU="${GPU:-5}"
OUT="${OUT:-${ROOT}/run_outputs/trace_bridge/20260713_trace_bridge_epoch7_final_evidence}"
CAUSAL_OUT="${OUT}/causal_interventions"
STAGE1_OUT="${OUT}/matched_stage1_rollouts"
BRIDGE_OUT="${OUT}/matched_bridge_rollouts"
FINAL_CKPT="$(readlink -f "${OUT}/TRACE_FINAL_EPOCH7.ckpt")"
STAGE1_CKPT=/disk1/dingxukai/trace_colar/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm/20260708-003948_284260_20260708_trace_bridge_bridgefull_vizstrong_stage1_gpu1_stage1/checkpoints/epoch1__step13452__monitor0.661.ckpt
BRIDGE_CKPT=/disk1/dingxukai/trace_colar/logs/bridge_qwen3_instruct_hybrid_compact_anchor_gate/qsa-gsm/20260708-153949_757553_20260708_trace_bridge_bridgefull_bridge_baseline_gpu2_bridge_baseline/checkpoints/epoch0__step6726__monitor0.655.ckpt
TRACE_RECORD=/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/20260711_trace_bridge_final_guarded_saveall_v6_matched_gpu3457/eval_stage2_trace_guarded_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt
DATASET_DIR=/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc

mkdir -p "${CAUSAL_OUT}" "${STAGE1_OUT}" "${BRIDGE_OUT}"
cd "${ROOT}"

run_intervention() {
  local mode="$1"
  local mode_out="${CAUSAL_OUT}/${mode}"
  local log_dir="${mode_out}/logs"
  local log_file="${mode_out}/eval.log"
  mkdir -p "${mode_out}" "${log_dir}"
  if compgen -G "${log_dir}/tb/run/test_*_gsm_pid*.json" >/dev/null; then
    printf '[skip] completed causal intervention: %s\n' "${mode}" | tee -a "${CAUSAL_OUT}/queue.log"
    return
  fi
  printf '[run] causal intervention: %s\n' "${mode}" | tee -a "${CAUSAL_OUT}/queue.log"
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" \
    "${ENV}/bin/python" run.py \
      --model trace_bridge_qwen3_instruct_vizstrong \
      --dataset qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /home/dingxukai \
      --test_ckpt_path "${FINAL_CKPT}" \
      --test_times 1 \
      dataset_dir="${DATASET_DIR}" \
      batch_size=1 \
      val_batch_size=4 \
      num_workers=2 \
      persistent_workers=false \
      do_trace_rl=true \
      trainer.num_sanity_val_steps=0 \
      trainer.strategy=auto \
      trainer.default_root_dir="${mode_out}/root" \
      trainer.logger.save_dir="${log_dir}" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
      model.model_kwargs.trace_bridge_config.save_trace_visual_info=false \
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0 \
      model.model_kwargs.trace_bridge_config.trace_eval_skip_structure_metrics=true \
      model.model_kwargs.trace_bridge_config.trace_eval_intervention="${mode}" \
      model.model_kwargs.trace_bridge_config.trace_eval_intervention_seed=0 \
      2>&1 | tee "${log_file}"
}

for intervention in normal reverse shuffle mean_repeat random_direction; do
  run_intervention "${intervention}"
done

causal_args=()
for intervention in reverse shuffle mean_repeat random_direction; do
  result_json="$(find "${CAUSAL_OUT}/${intervention}/logs/tb/run" -maxdepth 1 -name 'test_*_gsm_pid*.json' -print -quit)"
  causal_args+=(--intervention "${intervention}=${result_json}")
done
NORMAL_JSON="$(find "${CAUSAL_OUT}/normal/logs/tb/run" -maxdepth 1 -name 'test_*_gsm_pid*.json' -print -quit)"
"${ENV}/bin/python" tools/trace_bridge_causal_evidence.py \
  --reference "${NORMAL_JSON}" \
  "${causal_args[@]}" \
  --out_dir "${CAUSAL_OUT}/summary" \
  --bootstrap_trials 10000 \
  --seed 0 \
  2>&1 | tee "${CAUSAL_OUT}/causal_summary.log"

STAGE1_LOG_DIR="${STAGE1_OUT}/logs"
STAGE1_RECORD="${STAGE1_LOG_DIR}/tb/run/trace_bridge_visual_test.pt"
mkdir -p "${STAGE1_LOG_DIR}"
if [[ ! -f "${STAGE1_RECORD}" ]]; then
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" \
    "${ENV}/bin/python" run.py \
      --model trace_bridge_qwen3_instruct_vizstrong \
      --dataset qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /home/dingxukai \
      --test_ckpt_path "${STAGE1_CKPT}" \
      --test_times 1 \
      dataset_dir="${DATASET_DIR}" \
      batch_size=1 \
      val_batch_size=1 \
      num_workers=2 \
      persistent_workers=false \
      trainer.num_sanity_val_steps=0 \
      trainer.strategy=auto \
      trainer.default_root_dir="${STAGE1_OUT}/root" \
      trainer.logger.save_dir="${STAGE1_LOG_DIR}" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
      model.model_kwargs.trace_bridge_config.trace_visual_group_views=8 \
      model.model_kwargs.trace_bridge_config.trace_visual_record_limit=200 \
      model.model_kwargs.trace_bridge_config.trace_visual_latent_noise_scale=0.015 \
      model.model_kwargs.trace_bridge_config.trace_visual_noise_seed=0 \
      model.model_kwargs.trace_bridge_config.trace_visual_do_sample=true \
      model.model_kwargs.trace_bridge_config.trace_visual_temperature=0.95 \
      model.model_kwargs.trace_bridge_config.trace_visual_top_p=0.97 \
      2>&1 | tee "${STAGE1_OUT}/eval.log"
fi

"${ENV}/bin/python" tools/trace_bridge_visualize.py \
  --records "${STAGE1_RECORD}" \
  --out_dir "${STAGE1_OUT}/visual" \
  --max_records 9 \
  --pca_fit_records 200 \
  2>&1 | tee "${STAGE1_OUT}/visual.log"
"${ENV}/bin/python" tools/trace_bridge_geometry_summary.py \
  --records "${STAGE1_RECORD}" \
  --out_dir "${STAGE1_OUT}/geometry_raw" \
  --max_records 200 \
  2>&1 | tee "${STAGE1_OUT}/geometry_raw.log"
"${ENV}/bin/python" tools/trace_bridge_geometry_summary.py \
  --records "${STAGE1_RECORD}" \
  --out_dir "${STAGE1_OUT}/geometry_stage2_metric" \
  --max_records 200 \
  --signature_representation stage2_centered \
  --signature_raw_mix 0.25 \
  2>&1 | tee "${STAGE1_OUT}/geometry_stage2_metric.log"

# BRIDGE is loaded through the parameter-free TRACE recorder. Both learned
# TRACE view embeddings are disabled, so the primary prediction remains the
# exact BRIDGE model; matched Gaussian perturbations create the audit views.
BRIDGE_LOG_DIR="${BRIDGE_OUT}/logs"
BRIDGE_RECORD="${BRIDGE_LOG_DIR}/tb/run/trace_bridge_visual_test.pt"
mkdir -p "${BRIDGE_LOG_DIR}"
if [[ ! -f "${BRIDGE_RECORD}" ]]; then
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" \
    "${ENV}/bin/python" run.py \
      --model bridge_qwen3_instruct_hybrid_compact_anchor_gate \
      --dataset qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /home/dingxukai \
      --test_ckpt_path "${BRIDGE_CKPT}" \
      --test_times 1 \
      dataset_dir="${DATASET_DIR}" \
      batch_size=1 \
      val_batch_size=1 \
      num_workers=2 \
      persistent_workers=false \
      trainer.num_sanity_val_steps=0 \
      trainer.strategy=auto \
      trainer.default_root_dir="${BRIDGE_OUT}/root" \
      trainer.logger.save_dir="${BRIDGE_LOG_DIR}" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      model.target=src.models.trace_bridge.LitTRACEBridge \
      model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
      'model.model_kwargs.trace_bridge_config={"use_trace_view_embeddings":false,"use_trace_step_view_embeddings":false,"save_trace_visual_info":true,"trace_visual_group_views":8,"trace_visual_record_limit":200,"trace_visual_latent_noise_scale":0.015,"trace_visual_noise_seed":0,"trace_visual_do_sample":true,"trace_visual_temperature":0.95,"trace_visual_top_p":0.97,"rl_signature_source":"residuals"}' \
      2>&1 | tee "${BRIDGE_OUT}/eval.log"
fi

BRIDGE_NATIVE_JSON=/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/20260708_trace_bridge_bridgefull_bridge_baseline_gpu2/eval_snapshot_epoch0__step6726__monitor0_655__mtime1783503244088392417__size674673616__gsm8k_aug_logs/tb/run/test_20260708-215007_gsm_pid45195.json
BRIDGE_RECORDED_JSON="$(find "${BRIDGE_LOG_DIR}/tb/run" -maxdepth 1 -name 'test_*_gsm_pid*.json' -print -quit)"
"${ENV}/bin/python" tools/trace_bridge_prediction_parity.py \
  --reference "${BRIDGE_NATIVE_JSON}" \
  --candidate "${BRIDGE_RECORDED_JSON}" \
  --out_dir "${BRIDGE_OUT}/recorder_parity" \
  --reference_name 'native BRIDGE evaluation' \
  --candidate_name 'BRIDGE through parameter-free TRACE recorder' \
  2>&1 | tee "${BRIDGE_OUT}/recorder_parity.log"

"${ENV}/bin/python" tools/trace_bridge_visualize.py \
  --records "${BRIDGE_RECORD}" \
  --out_dir "${BRIDGE_OUT}/visual" \
  --max_records 9 \
  --pca_fit_records 200 \
  2>&1 | tee "${BRIDGE_OUT}/visual.log"
"${ENV}/bin/python" tools/trace_bridge_geometry_summary.py \
  --records "${BRIDGE_RECORD}" \
  --out_dir "${BRIDGE_OUT}/geometry_raw" \
  --max_records 200 \
  2>&1 | tee "${BRIDGE_OUT}/geometry_raw.log"
"${ENV}/bin/python" tools/trace_bridge_geometry_summary.py \
  --records "${BRIDGE_RECORD}" \
  --out_dir "${BRIDGE_OUT}/geometry_stage2_metric" \
  --max_records 200 \
  --signature_representation stage2_centered \
  --signature_raw_mix 0.25 \
  2>&1 | tee "${BRIDGE_OUT}/geometry_stage2_metric.log"
"${ENV}/bin/python" tools/trace_bridge_compare_rollouts.py \
  --record "BRIDGE=${BRIDGE_RECORD}" \
  --record "Stage1=${STAGE1_RECORD}" \
  --record "TRACE-epoch7=${TRACE_RECORD}" \
  --out_dir "${OUT}/global_pca_bridge_stage1_epoch7" \
  --max_records 200 \
  --pca_fit_records 200 \
  --signature_representation stage2_centered \
  --signature_raw_mix 0.25 \
  --show_indices 189 6 0 \
  --show_selection_rule 'one GSM8K case per paired transition class: rescue q189, regression q6, both-correct q0; closest to class-median delta-L among the first 200; no geometry used' \
  2>&1 | tee "${OUT}/global_pca_bridge_stage1_epoch7.log"
"${ENV}/bin/python" tools/trace_bridge_compare_assignments.py \
  --record "BRIDGE=${BRIDGE_RECORD}" \
  --record "Stage1=${STAGE1_RECORD}" \
  --record "TRACE-epoch7=${TRACE_RECORD}" \
  --indices 189 6 0 \
  --out_dir "${OUT}/assignment_bridge_stage1_epoch7" \
  --max_records 200 \
  --selection_rule 'one GSM8K case per paired transition class: rescue q189, regression q6, both-correct q0; closest to class-median delta-L among the first 200; no assignment or geometry used' \
  2>&1 | tee "${OUT}/assignment_bridge_stage1_epoch7.log"
"${ENV}/bin/python" tools/trace_bridge_stage_geometry_delta.py \
  --stage1_raw "${STAGE1_OUT}/geometry_raw/trace_bridge_geometry_summary.json" \
  --trace_raw /disk1/dingxukai/trace_colar/run_outputs/trace_bridge/20260711_trace_bridge_final_guarded_saveall_v6_matched_gpu3457/visual_stage2_trace_guarded_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json \
  --stage1_centered "${STAGE1_OUT}/geometry_stage2_metric/trace_bridge_geometry_summary.json" \
  --trace_centered "${OUT}/geometry_epoch7_stage2_metric/trace_bridge_geometry_summary.json" \
  --out_dir "${OUT}/stage1_to_epoch7_geometry_delta" \
  --bootstrap_trials 10000 \
  --seed 0 \
  2>&1 | tee "${OUT}/stage1_to_epoch7_geometry_delta.log"

printf 'TRACE epoch7 causal and Stage1 matched-rollout evidence completed.\n' > "${OUT}/solid_evidence_queue_done.txt"
