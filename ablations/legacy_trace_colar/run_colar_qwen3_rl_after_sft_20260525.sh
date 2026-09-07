#!/usr/bin/env bash
set -euo pipefail

# Launch the official CoLaR RL stage after the current official SFT-compression
# runs finish. This keeps the public two-stage CoLaR recipe:
#   Stage 1: SFT compression from a CoT checkpoint.
#   Stage 2: RL/GRPO from the trained CoLaR-SFT checkpoint.

ROOT="/home/dingxukai/colar origin"
OUT_DIR="${ROOT}/run_outputs/origin_colar_full_20260525"
mkdir -p "${OUT_DIR}"

C2_SESSION="origin_colar_qwen3_c2_rl_from_sft_3epoch_gpu4_20260525"
C5_SESSION="origin_colar_qwen3_c5_rl_from_sft_3epoch_gpu5_20260525"

C2_SFT_SESSION="origin_colar_qwen3_c2_cotinit_50epoch_gpu4_20260525"
C5_SFT_SESSION="origin_colar_qwen3_c5_cotinit_50epoch_gpu5_20260525"

C2_SFT_DIR="${ROOT}/logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260525-172642_802951_origin_colar_qwen3_c2_cotinit_rawgsm_lr3e-5_50epoch_gpu4_20260525"
C5_SFT_DIR="${ROOT}/logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260525-172642_884973_origin_colar_qwen3_c5_cotinit_rawgsm_lr3e-5_50epoch_gpu5_20260525"

best_ckpt() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
ckpt_dir = run_dir / "checkpoints"
if not ckpt_dir.exists():
    sys.exit(1)

best = None
for path in ckpt_dir.glob("*.ckpt"):
    if path.name == "last.ckpt":
        continue
    m = re.search(r"monitor(-?\d+(?:\.\d+)?)", path.name)
    if not m:
        continue
    score = float(m.group(1))
    if best is None or score > best[0]:
        best = (score, path)

if best is not None:
    print(best[1])
elif (ckpt_dir / "last.ckpt").exists():
    print(ckpt_dir / "last.ckpt")
else:
    sys.exit(1)
PY
}

wait_for_sft() {
  local session="$1"
  local run_dir="$2"
  echo "[$(date '+%F %T')] Waiting for ${session} to finish and produce a checkpoint..."
  while tmux has-session -t "${session}" 2>/dev/null; do
    sleep 300
  done
  until best_ckpt "${run_dir}" >/dev/null 2>&1; do
    echo "[$(date '+%F %T')] ${run_dir} has no usable checkpoint yet; waiting..."
    sleep 120
  done
}

launch_rl() {
  local name="$1"
  local gpu="$2"
  local factor="$3"
  local ckpt="$4"
  local log_file="${OUT_DIR}/${name}.log"

  if tmux has-session -t "${name}" 2>/dev/null; then
    echo "[$(date '+%F %T')] ${name} already exists; skip launching."
    return 0
  fi

  echo "[$(date '+%F %T')] Launching ${name} from ${ckpt}"
  tmux new-session -d -s "${name}" \
    "cd '${ROOT}' && \
     source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && \
     conda activate ROT && \
     CUDA_VISIBLE_DEVICES=${gpu} python run.py \
       --model=colar_qwen3_instruct \
       --dataset=gsm8k_aug_nl \
       --devices=0 \
       --load_ckpt_path='${ckpt}' \
       --do_test \
       --test_times=5 \
       --workspace_path=/home/dingxukai \
       --log_suffix=${name} \
       batch_size=1 \
       val_batch_size=1 \
       max_epochs=3 \
       num_sanity_val_steps=0 \
       max_compression_factor=${factor} \
       compression_factor=${factor} \
       max_new_tokens=16 \
       do_rl=True \
       group_size=8 \
       exp_batch_size=8 \
       n_train_samples_per_epoch=512 \
       lr=3e-5 \
       trainer.default_root_dir='${ROOT}/run_roots/${name}' \
       2>&1 | tee -a '${log_file}'"
}

wait_for_sft "${C2_SFT_SESSION}" "${C2_SFT_DIR}"
wait_for_sft "${C5_SFT_SESSION}" "${C5_SFT_DIR}"

C2_CKPT="$(best_ckpt "${C2_SFT_DIR}")"
C5_CKPT="$(best_ckpt "${C5_SFT_DIR}")"

launch_rl "${C2_SESSION}" 4 2 "${C2_CKPT}"
launch_rl "${C5_SESSION}" 5 5 "${C5_CKPT}"

echo "[$(date '+%F %T')] RL launch script finished."
