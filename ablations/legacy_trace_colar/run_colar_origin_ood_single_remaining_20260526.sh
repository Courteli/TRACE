#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dingxukai/colar origin"
OUT_DIR="$ROOT/run_outputs/origin_colar_ood_20260526_single"
mkdir -p "$OUT_DIR"

set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

R2_CKPT="$ROOT/logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260427-002950_142853_origin_colar_qwen3_instruct_r2_cotinit_rawgsm_lr3e-5_50epoch_gpu2/checkpoints/epoch42__step289218__monitor0.475.ckpt"
R5_CKPT="$ROOT/logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260427-002951_286878_origin_colar_qwen3_instruct_r5_cotinit_rawgsm_lr3e-5_50epoch_gpu3/checkpoints/epoch49__step336300__monitor0.256.ckpt"

run_eval() {
  local factor="$1"
  local ckpt="$2"
  local dataset="$3"
  local data_dir="$4"
  local test_file="$5"
  local gpu="$6"
  local version="20260526_origin_colar_r${factor}_${dataset}_ood_single"
  local log_file="$OUT_DIR/${version}.log"

  echo "[$(date '+%F %T')] START r=${factor} dataset=${dataset} gpu=${gpu}" | tee -a "$log_file"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" python run.py \
    --model=colar_qwen3_instruct \
    --dataset=gsm8k_aug_nl \
    --devices=0 \
    --workspace_path=/home/dingxukai \
    --test_ckpt_path="$ckpt" \
    --test_times=1 \
    dataset_name="$dataset" \
    dataset_dir="$data_dir" \
    test_file="$test_file" \
    max_compression_factor="$factor" \
    compression_factor="$factor" \
    max_new_tokens=16 \
    batch_size=1 \
    val_batch_size=1 \
    trainer.logger.name="ood_colar_qwen3_instruct" \
    trainer.logger.version="$version" \
    trainer.default_root_dir="$ROOT/run_roots/${version}" \
    2>&1 | tee -a "$log_file"
  echo "[$(date '+%F %T')] DONE r=${factor} dataset=${dataset}" | tee -a "$log_file"
}

run_group() {
  local factor="$1"
  local ckpt="$2"
  local gpu="$3"
  run_eval "$factor" "$ckpt" svamp "/home/dingxukai/RoT/data/SVAMP" "svamp_test_processed.jsonl" "$gpu"
  run_eval "$factor" "$ckpt" multiarith "/home/dingxukai/RoT/data/Multiarith" "multiarith_test_processed.jsonl" "$gpu"
}

run_group 2 "$R2_CKPT" 4 &
PID_R2=$!
run_group 5 "$R5_CKPT" 5 &
PID_R5=$!

wait "$PID_R2"
wait "$PID_R5"

echo "[$(date '+%F %T')] REMAINING SINGLE OOD EVALS DONE" | tee -a "$OUT_DIR/summary.log"
