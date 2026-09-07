#!/usr/bin/env bash
set -euo pipefail

cd "/home/dingxukai/colar origin"
set +u
source /home/dingxukai/miniconda3/etc/profile.d/conda.sh
conda activate ROT
set -u

COT_SUFFIX="origin_cot_qwen3_instruct_rawgsm_lr3e-5_3epoch_gpus1_2"
COT_ROOT="/home/dingxukai/colar origin/run_roots/origin_cot_qwen3_instruct_gpus1_2"

echo "[stage1] Training origin Qwen3-Instruct CoT-SFT checkpoint on GPUs 1,2"
CUDA_VISIBLE_DEVICES=1,2 python run.py \
  --model=cot_qwen3_instruct \
  --dataset=gsm8k_aug_nl \
  --devices=0,1 \
  --do_test \
  --test_times=1 \
  --workspace_path=/home/dingxukai \
  --log_suffix="${COT_SUFFIX}" \
  batch_size=2 \
  val_batch_size=1 \
  max_epochs=3 \
  num_sanity_val_steps=0 \
  max_new_tokens=256 \
  lr=3e-5 \
  trainer.default_root_dir="${COT_ROOT}"

COT_RUN_DIR="$(python - <<'PY'
from pathlib import Path

base = Path("/home/dingxukai/colar origin/logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl")
runs = sorted(
    [p for p in base.glob("*origin_cot_qwen3_instruct_rawgsm_lr3e-5_3epoch_gpus1_2") if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
)
if not runs:
    raise SystemExit("No origin CoT-SFT run directory found.")
print(runs[-1])
PY
)"

COT_CKPT="$(python - "${COT_RUN_DIR}" <<'PY'
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
ckpt_dir = run_dir / "checkpoints"
epoch_ckpts = list(ckpt_dir.glob("epoch*monitor*.ckpt"))
if epoch_ckpts:
    def score(path: Path):
        match = re.search(r"monitor(-?\d+(?:\.\d+)?)", path.name)
        return float(match.group(1)) if match else float("-inf")
    print(max(epoch_ckpts, key=score))
elif (ckpt_dir / "last.ckpt").exists():
    print(ckpt_dir / "last.ckpt")
else:
    raise SystemExit(f"No checkpoint found under {ckpt_dir}")
PY
)"

echo "[stage1] Best origin CoT-SFT checkpoint: ${COT_CKPT}"

R2_SESSION="origin_colar_qwen3_instruct_r2_cotinit_gpu1"
R5_SESSION="origin_colar_qwen3_instruct_r5_cotinit_gpu2"
tmux kill-session -t "${R2_SESSION}" 2>/dev/null || true
tmux kill-session -t "${R5_SESSION}" 2>/dev/null || true

echo "[stage2] Launching origin CoLaR r=2 on GPU 1 from CoT-SFT checkpoint"
tmux new-session -d -s "${R2_SESSION}" "cd '/home/dingxukai/colar origin' && source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && conda activate ROT && CUDA_VISIBLE_DEVICES=1 python run.py --model=colar_qwen3_instruct --dataset=gsm8k_aug_nl --devices=0 --load_ckpt_path='${COT_CKPT}' --do_test --test_times=5 --workspace_path=/home/dingxukai --log_suffix=origin_colar_qwen3_instruct_r2_cotinit_rawgsm_lr3e-5_3epoch_gpu1 batch_size=1 val_batch_size=1 max_epochs=3 num_sanity_val_steps=0 max_compression_factor=2 compression_factor=2 max_new_tokens=16 lr=3e-5 trainer.default_root_dir='/home/dingxukai/colar origin/run_roots/origin_colar_qwen3_instruct_r2_cotinit_gpu1' 2>&1 | tee -a '/home/dingxukai/colar origin/run_outputs/origin_colar_qwen3_instruct_r2_cotinit_gpu1.log'"

echo "[stage2] Launching origin CoLaR r=5 on GPU 2 from CoT-SFT checkpoint"
tmux new-session -d -s "${R5_SESSION}" "cd '/home/dingxukai/colar origin' && source /home/dingxukai/miniconda3/etc/profile.d/conda.sh && conda activate ROT && CUDA_VISIBLE_DEVICES=2 python run.py --model=colar_qwen3_instruct --dataset=gsm8k_aug_nl --devices=0 --load_ckpt_path='${COT_CKPT}' --do_test --test_times=5 --workspace_path=/home/dingxukai --log_suffix=origin_colar_qwen3_instruct_r5_cotinit_rawgsm_lr3e-5_3epoch_gpu2 batch_size=1 val_batch_size=1 max_epochs=3 num_sanity_val_steps=0 max_compression_factor=5 compression_factor=5 max_new_tokens=16 lr=3e-5 trainer.default_root_dir='/home/dingxukai/colar origin/run_roots/origin_colar_qwen3_instruct_r5_cotinit_gpu2' 2>&1 | tee -a '/home/dingxukai/colar origin/run_outputs/origin_colar_qwen3_instruct_r5_cotinit_gpu2.log'"

echo "[stage2] Started ${R2_SESSION} and ${R5_SESSION}"
