#!/usr/bin/env bash
set -euo pipefail

: "${PS1:=}"
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT

cd /disk1/dingxukai/trace_colar

MODE="${MODE:-stage1}"
GPU="${GPU:-2}"
RUN_TAG="${RUN_TAG:-20260707_trace_bridge_${MODE}_gpu${GPU}}"
TEST_TIMES="${TEST_TIMES:-1}"
TRACE_MODEL="${TRACE_MODEL:-trace_bridge_qwen3_instruct}"
DATASET_DIR="${DATASET_DIR:-/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc}"
COT_CKPT="${COT_CKPT:-/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/checkpoints/epoch0__step6726__monitor0.871.ckpt}"
OUT_DIR="${OUT_DIR:-/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/${RUN_TAG}}"
ROOT_DIR="${ROOT_DIR:-/disk1/dingxukai/trace_colar/run_roots/trace_bridge/${RUN_TAG}}"

mkdir -p "${OUT_DIR}" "${ROOT_DIR}"

declare -A EVAL_DATASETS=(
  ["gsm8k_aug"]="/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc"
  ["gsmhard"]="/home/dingxukai/RoT/data/GSM8k-Hard/readcot_qsa_test"
  ["svamp"]="/home/dingxukai/RoT/data/SVAMP/readcot_qsa_test"
  ["multiarith"]="/home/dingxukai/RoT/data/Multiarith/readcot_qsa_test"
)

write_manifest() {
  if [[ "${MODE}" == eval_* ]]; then
    {
      printf 'mode=%s\n' "${MODE}"
      printf 'gpu=%s\n' "${GPU}"
      printf 'run_tag=%s\n' "${RUN_TAG}"
      printf 'trace_model=%s\n' "${TRACE_MODEL}"
      printf 'ckpt=%s\n' "${CKPT:-}"
      printf 'eval_label=%s\n' "${EVAL_LABEL:-}"
      printf 'test_times=%s\n' "${TEST_TIMES}"
      printf 'created_at=%s\n' "$(date '+%F %T')"
      printf '\n'
    } >> "${OUT_DIR}/eval_invocations.log"
    if [[ ! -s "${OUT_DIR}/manifest.txt" ]]; then
      {
        printf 'mode=%s\n' "${MODE}"
        printf 'gpu=%s\n' "${GPU}"
        printf 'run_tag=%s\n' "${RUN_TAG}"
        printf 'trace_model=%s\n' "${TRACE_MODEL}"
        printf 'test_times=%s\n' "${TEST_TIMES}"
        printf 'created_at=%s\n' "$(date '+%F %T')"
      } > "${OUT_DIR}/manifest.txt"
    fi
    return
  fi
  {
    printf 'mode=%s\n' "${MODE}"
    printf 'gpu=%s\n' "${GPU}"
    printf 'run_tag=%s\n' "${RUN_TAG}"
    printf 'trace_model=%s\n' "${TRACE_MODEL}"
    printf 'cot_ckpt=%s\n' "${COT_CKPT}"
    printf 'dataset_dir=%s\n' "${DATASET_DIR}"
    printf 'test_times=%s\n' "${TEST_TIMES}"
    printf 'created_at=%s\n' "$(date '+%F %T')"
  } > "${OUT_DIR}/manifest.txt"
}

latest_ckpt_for_suffix() {
  local model_name="$1"
  local suffix="$2"
  python - "/disk1/dingxukai/trace_colar/logs/${model_name}/qsa-gsm" "${suffix}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
suffix = sys.argv[2]
candidates = []
for path in root.glob(f"*{suffix}*/checkpoints/*.ckpt"):
    if path.name == "last.ckpt":
        continue
    match = re.search(r"monitor(-?[0-9]+(?:\.[0-9]+)?)", path.name)
    if match:
        candidates.append((float(match.group(1)), path.stat().st_mtime_ns, str(path)))
if candidates:
    print(max(candidates)[2])
PY
}

train_bridge_baseline() {
  local suffix="${RUN_TAG}_bridge_baseline"
  echo "[train] BRIDGE baseline ${suffix}" | tee -a "${OUT_DIR}/train_bridge_baseline.log"
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model bridge_qwen3_instruct_hybrid_compact_anchor_gate \
    --dataset qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /home/dingxukai \
    --load_ckpt_path "${COT_CKPT}" \
    --do_test \
    --test_times "${TEST_TIMES}" \
    --log_suffix "${suffix}" \
    dataset_dir="${DATASET_DIR}" \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    trainer.num_sanity_val_steps=0 \
    trainer.max_epochs=50 \
    trainer.max_steps=-1 \
    trainer.val_check_interval=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.default_root_dir="${ROOT_DIR}/bridge_baseline_train" \
    model.training_kwargs.scheduler.num_training_steps=336300 \
    2>&1 | tee -a "${OUT_DIR}/train_bridge_baseline.log"
  local ckpt
  ckpt="$(latest_ckpt_for_suffix bridge_qwen3_instruct_hybrid_compact_anchor_gate "${suffix}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[error] no BRIDGE baseline ckpt found" | tee -a "${OUT_DIR}/manifest.txt"
    exit 1
  fi
  echo "${ckpt}" | tee "${OUT_DIR}/bridge_baseline_best_ckpt.txt"
}

train_trace_stage1() {
  local suffix="$1"
  local log_file="$2"
  echo "[train] TRACE-BRIDGE Stage1 ${suffix}" | tee -a "${log_file}"
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model "${TRACE_MODEL}" \
    --dataset qsa \
    --trainer default \
    --devices 0 \
    --workspace_path /home/dingxukai \
    --load_ckpt_path "${COT_CKPT}" \
    --do_test \
    --test_times "${TEST_TIMES}" \
    --log_suffix "${suffix}" \
    dataset_dir="${DATASET_DIR}" \
    batch_size=1 \
    val_batch_size=1 \
    num_workers=4 \
    trainer.num_sanity_val_steps=0 \
    trainer.max_epochs=50 \
    trainer.max_steps=-1 \
    trainer.val_check_interval=1.0 \
    trainer.check_val_every_n_epoch=1 \
    trainer.default_root_dir="${ROOT_DIR}/${suffix}_train" \
    model.training_kwargs.scheduler.num_training_steps=336300 \
    2>&1 | tee -a "${log_file}"
  local ckpt
  ckpt="$(latest_ckpt_for_suffix "${TRACE_MODEL}" "${suffix}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[error] no TRACE-BRIDGE Stage1 ckpt found for ${suffix}" | tee -a "${OUT_DIR}/manifest.txt"
    exit 1
  fi
  echo "${ckpt}" | tee "${OUT_DIR}/${suffix}_best_ckpt.txt"
}

train_trace_stage2() {
  local stage1_ckpt="$1"
  local variant="$2"
  local suffix="${RUN_TAG}_${variant}"
  local log_file="${OUT_DIR}/train_${variant}.log"
  local checkpoint_filename="${TRACE_CHECKPOINT_FILENAME:-}"
  if [[ -z "${checkpoint_filename}" ]]; then
    checkpoint_filename='epoch{epoch}__step{step}__monitor{monitor:.6f}'
  fi
  local rl_args=()
  if [[ "${variant}" == "stage2_trace_modecontrast" || "${variant}" == "stage2_trace_guarded" ]]; then
    rl_args=(
      model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048
      model.model_kwargs.trace_rl_config.group_size=8
      model.model_kwargs.trace_rl_config.exp_batch_size="${TRACE_RL_EXP_BATCH_SIZE:-1}"
      model.model_kwargs.trace_rl_config.temperature=0.95
      model.model_kwargs.trace_rl_config.top_p=0.97
      model.model_kwargs.trace_rl_config.clip_eps=0.12
      model.model_kwargs.trace_rl_config.stage2_latent_noise_scale=0.015
      model.model_kwargs.trace_rl_config.stage2_center_signatures=true
      model.model_kwargs.trace_rl_config.stage2_signature_raw_mix=0.25
      model.model_kwargs.trace_rl_config.trace_reward_weight=0.0
      model.model_kwargs.trace_rl_config.max_modes=3
      model.model_kwargs.trace_rl_config.mode_merge_threshold=0.65
      model.model_kwargs.trace_rl_config.output_length_penalty_weight=0.03
      model.model_kwargs.trace_rl_config.target_output_length=33.5
      model.model_kwargs.trace_rl_config.stage2_direct_signature_weight=0.12
      model.model_kwargs.trace_rl_config.stage2_direct_pos_weight=1.0
      model.model_kwargs.trace_rl_config.stage2_direct_neg_weight=1.0
      model.model_kwargs.trace_rl_config.stage2_direct_wrong_div_weight=0.05
      model.model_kwargs.trace_rl_config.stage2_direct_neg_margin=0.15
      model.model_kwargs.trace_rl_config.stage2_direct_wrong_div_margin=0.65
      model.model_kwargs.trace_rl_config.stage2_direct_inter_mode_weight=0.15
      model.model_kwargs.trace_rl_config.stage2_direct_inter_mode_margin=0.55
      model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.05
      model.training_kwargs.optimizer.lr=8e-7
      model.training_kwargs.scheduler.warmup_steps="${TRACE_RL_WARMUP_STEPS:-300}"
      model.training_kwargs.scheduler.num_training_steps="${TRACE_RL_TRAINING_STEPS:-20480}"
    )
    if [[ "${variant}" == "stage2_trace_guarded" ]]; then
      rl_args+=(
        model.model_kwargs.trace_rl_config.stage2_accuracy_gradient_guard=true
        model.model_kwargs.trace_rl_config.stage2_geometry_grad_ratio="${TRACE_GEOMETRY_GRAD_RATIO:-0.25}"
        model.model_kwargs.trace_rl_config.stage2_geometry_micro_batch_size="${TRACE_GEOMETRY_MICRO_BATCH_SIZE:-1}"
      )
    fi
  elif [[ "${variant}" == "stage2_answer_only" ]]; then
    rl_args=(
      model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048
      model.model_kwargs.trace_rl_config.group_size=8
      model.model_kwargs.trace_rl_config.exp_batch_size="${TRACE_RL_EXP_BATCH_SIZE:-1}"
      model.model_kwargs.trace_rl_config.temperature=0.95
      model.model_kwargs.trace_rl_config.top_p=0.97
      model.model_kwargs.trace_rl_config.clip_eps=0.12
      model.model_kwargs.trace_rl_config.stage2_latent_noise_scale=0.015
      model.model_kwargs.trace_rl_config.stage2_center_signatures=true
      model.model_kwargs.trace_rl_config.stage2_signature_raw_mix=0.25
      model.model_kwargs.trace_rl_config.trace_reward_weight=0.0
      model.model_kwargs.trace_rl_config.output_length_penalty_weight=0.03
      model.model_kwargs.trace_rl_config.target_output_length=33.5
      model.model_kwargs.trace_rl_config.stage2_direct_signature_weight=0.0
      model.model_kwargs.trace_rl_config.stage2_sft_replay_weight="${TRACE_STAGE2_SFT_REPLAY_WEIGHT:-0.05}"
      model.training_kwargs.optimizer.lr=8e-7
      model.training_kwargs.scheduler.warmup_steps="${TRACE_RL_WARMUP_STEPS:-300}"
      model.training_kwargs.scheduler.num_training_steps="${TRACE_RL_TRAINING_STEPS:-20480}"
    )
  elif [[ "${variant}" == "stage2_trace_direct" ]]; then
    rl_args=(
      model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048
      model.model_kwargs.trace_rl_config.group_size=8
      model.model_kwargs.trace_rl_config.exp_batch_size="${TRACE_RL_EXP_BATCH_SIZE:-1}"
      model.model_kwargs.trace_rl_config.temperature=0.95
      model.model_kwargs.trace_rl_config.top_p=0.97
      model.model_kwargs.trace_rl_config.clip_eps=0.12
      model.model_kwargs.trace_rl_config.trace_reward_weight=0.25
      model.model_kwargs.trace_rl_config.trace_bonus_clip=0.35
      model.model_kwargs.trace_rl_config.max_modes=4
      model.model_kwargs.trace_rl_config.mode_merge_threshold=0.88
      model.model_kwargs.trace_rl_config.pos_mode_fit_weight=0.22
      model.model_kwargs.trace_rl_config.mode_diversity_weight=0.12
      model.model_kwargs.trace_rl_config.neg_repulsion_weight=0.26
      model.model_kwargs.trace_rl_config.neg_margin=0.25
      model.model_kwargs.trace_rl_config.step_coherence_weight=0.04
      model.model_kwargs.trace_rl_config.noncollapse_weight=0.04
      model.model_kwargs.trace_rl_config.output_length_penalty_weight=0.04
      model.model_kwargs.trace_rl_config.target_output_length=33.5
      model.model_kwargs.trace_rl_config.stage2_direct_signature_weight=0.18
      model.model_kwargs.trace_rl_config.stage2_direct_pos_weight=1.0
      model.model_kwargs.trace_rl_config.stage2_direct_neg_weight=1.2
      model.model_kwargs.trace_rl_config.stage2_direct_wrong_div_weight=0.25
      model.model_kwargs.trace_rl_config.stage2_direct_neg_margin=0.25
      model.model_kwargs.trace_rl_config.stage2_direct_wrong_div_margin=0.72
      model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.035
      model.training_kwargs.optimizer.lr=8e-7
      model.training_kwargs.scheduler.warmup_steps=300
      model.training_kwargs.scheduler.num_training_steps=20480
    )
  elif [[ "${variant}" == "stage2_trace_hardneg" ]]; then
    rl_args=(
      model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048
      model.model_kwargs.trace_rl_config.group_size=8
      model.model_kwargs.trace_rl_config.exp_batch_size="${TRACE_RL_EXP_BATCH_SIZE:-1}"
      model.model_kwargs.trace_rl_config.temperature=1.05
      model.model_kwargs.trace_rl_config.top_p=0.98
      model.model_kwargs.trace_rl_config.clip_eps=0.14
      model.model_kwargs.trace_rl_config.trace_reward_weight=0.30
      model.model_kwargs.trace_rl_config.trace_bonus_clip=0.45
      model.model_kwargs.trace_rl_config.max_modes=4
      model.model_kwargs.trace_rl_config.mode_merge_threshold=0.92
      model.model_kwargs.trace_rl_config.pos_mode_fit_weight=0.25
      model.model_kwargs.trace_rl_config.mode_diversity_weight=0.18
      model.model_kwargs.trace_rl_config.neg_repulsion_weight=0.36
      model.model_kwargs.trace_rl_config.neg_margin=0.18
      model.model_kwargs.trace_rl_config.step_coherence_weight=0.035
      model.model_kwargs.trace_rl_config.noncollapse_weight=0.05
      model.model_kwargs.trace_rl_config.output_length_penalty_weight=0.05
      model.model_kwargs.trace_rl_config.target_output_length=33.0
      model.model_kwargs.trace_rl_config.stage2_direct_signature_weight=0.28
      model.model_kwargs.trace_rl_config.stage2_direct_pos_weight=0.8
      model.model_kwargs.trace_rl_config.stage2_direct_neg_weight=1.8
      model.model_kwargs.trace_rl_config.stage2_direct_wrong_div_weight=0.55
      model.model_kwargs.trace_rl_config.stage2_direct_neg_margin=0.18
      model.model_kwargs.trace_rl_config.stage2_direct_wrong_div_margin=0.62
      model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.02
      model.training_kwargs.optimizer.lr=7e-7
      model.training_kwargs.scheduler.warmup_steps=300
      model.training_kwargs.scheduler.num_training_steps=20480
    )
  elif [[ "${variant}" == "stage2_tracegain" ]]; then
    rl_args=(
      model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048
      model.model_kwargs.trace_rl_config.group_size=8
      model.model_kwargs.trace_rl_config.exp_batch_size="${TRACE_RL_EXP_BATCH_SIZE:-1}"
      model.model_kwargs.trace_rl_config.temperature=1.0
      model.model_kwargs.trace_rl_config.top_p=0.97
      model.model_kwargs.trace_rl_config.clip_eps=0.12
      model.model_kwargs.trace_rl_config.trace_reward_weight=0.35
      model.model_kwargs.trace_rl_config.trace_bonus_clip=0.45
      model.model_kwargs.trace_rl_config.max_modes=4
      model.model_kwargs.trace_rl_config.mode_merge_threshold=0.90
      model.model_kwargs.trace_rl_config.pos_mode_fit_weight=0.25
      model.model_kwargs.trace_rl_config.mode_diversity_weight=0.16
      model.model_kwargs.trace_rl_config.neg_repulsion_weight=0.35
      model.model_kwargs.trace_rl_config.neg_margin=0.20
      model.model_kwargs.trace_rl_config.step_coherence_weight=0.04
      model.model_kwargs.trace_rl_config.noncollapse_weight=0.04
      model.model_kwargs.trace_rl_config.output_length_penalty_weight=0.04
      model.model_kwargs.trace_rl_config.target_output_length=33.5
      model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.025
      model.training_kwargs.optimizer.lr=1.2e-6
      model.training_kwargs.scheduler.warmup_steps=300
      model.training_kwargs.scheduler.num_training_steps=20480
    )
  elif [[ "${variant}" == "stage2_strong" ]]; then
    rl_args=(
      model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=2048
      model.model_kwargs.trace_rl_config.group_size=8
      model.model_kwargs.trace_rl_config.exp_batch_size="${TRACE_RL_EXP_BATCH_SIZE:-1}"
      model.model_kwargs.trace_rl_config.temperature=0.9
      model.model_kwargs.trace_rl_config.trace_reward_weight=0.18
      model.model_kwargs.trace_rl_config.pos_mode_fit_weight=0.16
      model.model_kwargs.trace_rl_config.mode_diversity_weight=0.08
      model.model_kwargs.trace_rl_config.neg_repulsion_weight=0.14
      model.model_kwargs.trace_rl_config.output_length_penalty_weight=0.015
      model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.08
      model.training_kwargs.optimizer.lr=7e-7
      model.training_kwargs.scheduler.num_training_steps=20480
    )
  else
    rl_args=(
      model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=1024
      model.model_kwargs.trace_rl_config.group_size=8
      model.model_kwargs.trace_rl_config.exp_batch_size="${TRACE_RL_EXP_BATCH_SIZE:-1}"
      model.model_kwargs.trace_rl_config.temperature=0.8
      model.model_kwargs.trace_rl_config.trace_reward_weight=0.10
      model.model_kwargs.trace_rl_config.stage2_sft_replay_weight=0.12
      model.training_kwargs.optimizer.lr=5e-7
      model.training_kwargs.scheduler.num_training_steps=10240
    )
  fi

  echo "[train] TRACE-BRIDGE ${variant} from ${stage1_ckpt}" | tee -a "${log_file}"
  TMPDIR=/tmp TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
    --model "${TRACE_MODEL}" \
    --dataset qsa \
    --trainer default \
    --devices "${TRACE_TRAIN_DEVICES:-0}" \
    --workspace_path /home/dingxukai \
    --load_ckpt_path "${stage1_ckpt}" \
    --disable_early_stopping \
    --do_test \
    --test_times "${TEST_TIMES}" \
    --log_suffix "${suffix}" \
    dataset_dir="${DATASET_DIR}" \
    batch_size="${TRACE_TRAIN_BATCH_SIZE:-1}" \
    val_batch_size=1 \
    num_workers=4 \
    do_trace_rl=true \
    trainer.num_sanity_val_steps=0 \
    trainer.max_epochs="${TRACE_MAX_EPOCHS:-10}" \
    trainer.max_steps=-1 \
    trainer.val_check_interval=1.0 \
    trainer.check_val_every_n_epoch="${TRACE_VAL_EVERY_N_EPOCH:-1}" \
    trainer.gradient_clip_val=0 \
    trainer.strategy="${TRACE_TRAIN_STRATEGY:-auto}" \
    trainer.default_root_dir="${ROOT_DIR}/${variant}_train" \
    save_top_k="${TRACE_SAVE_TOP_K:--1}" \
    filename="${checkpoint_filename}" \
    model.model_kwargs.hybrid_generation_config.max_new_tokens=48 \
    "${rl_args[@]}" \
    2>&1 | tee -a "${log_file}"

  local ckpt
  ckpt="$(latest_ckpt_for_suffix "${TRACE_MODEL}" "${suffix}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[error] no TRACE-BRIDGE ${variant} ckpt found" | tee -a "${OUT_DIR}/manifest.txt"
    exit 1
  fi
  echo "${ckpt}" | tee "${OUT_DIR}/${variant}_best_ckpt.txt"
}

eval_ckpt() {
  local model_name="$1"
  local ckpt="$2"
  local label="$3"
  local is_trace="$4"
  for key in gsm8k_aug gsmhard svamp multiarith; do
    local data_dir="${EVAL_DATASETS[$key]}"
    local log_file="${OUT_DIR}/eval_${label}_${key}.log"
    local log_dir="${OUT_DIR}/eval_${label}_${key}_logs"
    local extra_args=()
    if [[ "${is_trace}" == "1" && "${key}" == "gsm8k_aug" ]]; then
      extra_args=(
        model.model_kwargs.trace_bridge_config.trace_visual_group_views=8
        model.model_kwargs.trace_bridge_config.trace_visual_record_limit=200
        model.model_kwargs.trace_bridge_config.trace_visual_latent_noise_scale=0.015
        model.model_kwargs.trace_bridge_config.trace_visual_noise_seed=0
        model.model_kwargs.trace_bridge_config.trace_visual_do_sample=true
        model.model_kwargs.trace_bridge_config.trace_visual_temperature=0.95
        model.model_kwargs.trace_bridge_config.trace_visual_top_p=0.97
        model.model_kwargs.trace_rl_config.stage2_latent_noise_scale=0.015
      )
    elif [[ "${is_trace}" == "1" ]]; then
      extra_args=(
        model.model_kwargs.trace_bridge_config.save_trace_visual_info=false
        model.model_kwargs.trace_bridge_config.trace_visual_record_limit=0
      )
    fi
    if [[ "${is_trace}" == "1" && "${TRACE_EVAL_DISABLE_VIEW_EMBEDDINGS:-false}" == "true" ]]; then
      extra_args+=(
        model.model_kwargs.trace_bridge_config.use_trace_view_embeddings=false
        model.model_kwargs.trace_bridge_config.use_trace_step_view_embeddings=false
      )
    fi
    echo "[eval] ${label} ${key}" | tee "${log_file}"
    TMPDIR=/tmp TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU}" python run.py \
      --model "${model_name}" \
      --dataset qsa \
      --trainer default \
      --devices 0 \
      --workspace_path /home/dingxukai \
      --test_ckpt_path "${ckpt}" \
      --test_times "${TEST_TIMES}" \
      dataset_dir="${data_dir}" \
      batch_size=1 \
      val_batch_size=1 \
      num_workers=2 \
      persistent_workers=false \
      trainer.num_sanity_val_steps=0 \
      trainer.default_root_dir="${ROOT_DIR}/eval_${label}_${key}" \
      trainer.logger.save_dir="${log_dir}" \
      trainer.logger.name=tb \
      trainer.logger.version=run \
      "${extra_args[@]}" \
      2>&1 | tee -a "${log_file}"
  done
}

summarize_label() {
  local label="$1"
  python - "${OUT_DIR}" "${label}" <<'PY'
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
label = sys.argv[2]
names = {
    "gsm8k_aug": "GSM8K-Aug",
    "gsmhard": "GSM-Hard",
    "svamp": "SVAMP",
    "multiarith": "MultiArith",
}
rows = []
for key, name in names.items():
    text = (out / f"eval_{label}_{key}.log").read_text(errors="ignore")
    vals = {}
    for m in re.finditer(r"│\s*(test/[^│]+?)\s*│\s*([-0-9.]+)\s*│", text):
        vals[m.group(1).strip()] = float(m.group(2))
    if not vals:
        continue
    acc = vals.get("test/acc", 0.0) * 100.0
    lat = vals.get("test/n_latent_forward", 0.0)
    out_len = vals.get("test/output_length", 0.0)
    total = lat + out_len
    dep = vals.get("test/dep_f1", 0.0)
    res = vals.get("test/residual_similarity", 0.0)
    rows.append((name, acc, lat, out_len, total, dep, res))

md = out / f"summary_{label}.md"
with md.open("w", encoding="utf-8") as f:
    f.write("| Label | Dataset | Acc | #Latent | OutputLen | #L | Dep-F1 | Res-Sim |\\n")
    f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\\n")
    for row in rows:
        f.write(f"| {label} | {row[0]} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.2f} | {row[4]:.2f} | {row[5]:.3f} | {row[6]:.3f} |\\n")
    if rows:
        f.write(f"\\nAverage Acc: {sum(r[1] for r in rows)/len(rows):.2f}\\n")
        f.write(f"Average #L: {sum(r[4] for r in rows)/len(rows):.2f}\\n")
print(md.read_text() if md.exists() else "")
PY
}

run_trace_visuals() {
  local label="$1"
  local record="${OUT_DIR}/eval_${label}_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt"
  if [[ ! -f "${record}" ]]; then
    echo "[warn] visual record not found: ${record}" | tee -a "${OUT_DIR}/manifest.txt"
    return
  fi
  local visual_dir="${OUT_DIR}/visual_${label}_gsm8k_aug"
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_bridge_visualize.py \
    --records "${record}" \
    --out_dir "${visual_dir}" \
    --max_records 9 \
    --pca_fit_records 200 \
    2>&1 | tee "${OUT_DIR}/visual_${label}.log"
  /home/dingxukai/miniconda3/envs/ROT/bin/python tools/trace_bridge_geometry_summary.py \
    --records "${record}" \
    --out_dir "${visual_dir}/geometry_200" \
    --max_records 200 \
    2>&1 | tee "${OUT_DIR}/geometry_${label}.log"
}

write_manifest

case "${MODE}" in
  baseline)
    train_bridge_baseline
    CKPT="$(cat "${OUT_DIR}/bridge_baseline_best_ckpt.txt")"
    eval_ckpt bridge_qwen3_instruct_hybrid_compact_anchor_gate "${CKPT}" bridge_baseline 0
    summarize_label bridge_baseline
    ;;
  stage1)
    STAGE1_SUFFIX="${RUN_TAG}_stage1"
    train_trace_stage1 "${STAGE1_SUFFIX}" "${OUT_DIR}/train_stage1.log"
    CKPT="$(cat "${OUT_DIR}/${STAGE1_SUFFIX}_best_ckpt.txt")"
    eval_ckpt "${TRACE_MODEL}" "${CKPT}" trace_stage1 1
    summarize_label trace_stage1
    run_trace_visuals trace_stage1
    ;;
  stage2_conservative|stage2_strong|stage2_tracegain|stage2_trace_direct|stage2_trace_hardneg|stage2_trace_modecontrast|stage2_trace_guarded|stage2_answer_only)
    STAGE1_SUFFIX="${RUN_TAG}_stage1"
    train_trace_stage1 "${STAGE1_SUFFIX}" "${OUT_DIR}/train_stage1_for_${MODE}.log"
    STAGE1_CKPT="$(cat "${OUT_DIR}/${STAGE1_SUFFIX}_best_ckpt.txt")"
    train_trace_stage2 "${STAGE1_CKPT}" "${MODE}"
    CKPT="$(cat "${OUT_DIR}/${MODE}_best_ckpt.txt")"
    eval_ckpt "${TRACE_MODEL}" "${CKPT}" "${MODE}" 1
    summarize_label "${MODE}"
    run_trace_visuals "${MODE}"
    ;;
  stage2_conservative_from_ckpt|stage2_strong_from_ckpt|stage2_tracegain_from_ckpt|stage2_trace_direct_from_ckpt|stage2_trace_hardneg_from_ckpt|stage2_trace_modecontrast_from_ckpt|stage2_trace_guarded_from_ckpt|stage2_answer_only_from_ckpt)
    : "${STAGE1_CKPT:?Set STAGE1_CKPT to a completed TRACE-BRIDGE Stage1 checkpoint.}"
    VARIANT="${MODE%_from_ckpt}"
    train_trace_stage2 "${STAGE1_CKPT}" "${VARIANT}"
    CKPT="$(cat "${OUT_DIR}/${VARIANT}_best_ckpt.txt")"
    eval_ckpt "${TRACE_MODEL}" "${CKPT}" "${VARIANT}" 1
    summarize_label "${VARIANT}"
    run_trace_visuals "${VARIANT}"
    ;;
  stage2_both_from_ckpt)
    : "${STAGE1_CKPT:?Set STAGE1_CKPT to a completed TRACE-BRIDGE Stage1 checkpoint.}"
    for VARIANT in stage2_conservative stage2_strong; do
      train_trace_stage2 "${STAGE1_CKPT}" "${VARIANT}"
      CKPT="$(cat "${OUT_DIR}/${VARIANT}_best_ckpt.txt")"
      eval_ckpt "${TRACE_MODEL}" "${CKPT}" "${VARIANT}" 1
      summarize_label "${VARIANT}"
      run_trace_visuals "${VARIANT}"
    done
    ;;
  eval_bridge_from_ckpt)
    : "${CKPT:?Set CKPT to a completed BRIDGE checkpoint.}"
    LABEL="${EVAL_LABEL:-bridge_baseline}"
    eval_ckpt bridge_qwen3_instruct_hybrid_compact_anchor_gate "${CKPT}" "${LABEL}" 0
    summarize_label "${LABEL}"
    ;;
  eval_trace_from_ckpt)
    : "${CKPT:?Set CKPT to a completed TRACE-BRIDGE checkpoint.}"
    LABEL="${EVAL_LABEL:-trace_eval}"
    eval_ckpt "${TRACE_MODEL}" "${CKPT}" "${LABEL}" 1
    summarize_label "${LABEL}"
    run_trace_visuals "${LABEL}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use baseline, stage1, stage2_conservative, stage2_strong, stage2_tracegain, stage2_trace_direct, stage2_trace_hardneg, stage2_trace_modecontrast, stage2_trace_guarded, stage2_answer_only, *_from_ckpt, stage2_both_from_ckpt, or eval_*_from_ckpt." >&2
    exit 2
    ;;
esac

echo "[done] ${MODE} outputs: ${OUT_DIR}" | tee -a "${OUT_DIR}/manifest.txt"
