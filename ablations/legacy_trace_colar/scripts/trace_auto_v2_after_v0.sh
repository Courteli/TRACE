#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
V0_SESSION="trace_v0_sft_qwen3_c5_gpu0"
WATCH_SESSION="trace_auto_v2_after_v0"
RUN_OUTPUTS="${ROOT}/run_outputs/trace"
mkdir -p "${RUN_OUTPUTS}"

best_ckpt() {
  python - "${ROOT}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_glob = "*trace_v0_sft_qwen3_c5_gpu0*"
base = root / "logs" / "trace_colar_qwen3_instruct" / "gsm8k_aug_nl-gsm8k_aug_nl"
if not base.exists():
    raise SystemExit(1)

best = None
for run_dir in base.glob(run_glob):
    if not run_dir.is_dir():
        continue
    for path in run_dir.glob("checkpoints/*.ckpt"):
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
    raise SystemExit(0)

lasts = []
for run_dir in base.glob(run_glob):
    if not run_dir.is_dir():
        continue
    lasts.extend(run_dir.glob("checkpoints/last.ckpt"))
lasts = sorted(lasts, key=lambda p: p.stat().st_mtime, reverse=True)
if lasts:
    print(lasts[0])
    raise SystemExit(0)

raise SystemExit(1)
PY
}

if [[ "${1:-}" == "--print-best" ]]; then
  best_ckpt
  exit $?
fi

V2_GPU="${1:-0}"
ANSWER_ONLY_GPU="${2:-4}"
POST_GPU="${3:-7}"
POST_SAMPLES="${4:-64}"
GROUP_SIZE="${GROUP_SIZE:-8}"
EXP_BATCH_SIZE="${EXP_BATCH_SIZE:-1}"
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-512}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
MAX_N_LATENT_FORWARD="${MAX_N_LATENT_FORWARD:-64}"
LR="${LR:-1e-6}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-1.0}"
TRACE_REWARD_WEIGHT="${TRACE_REWARD_WEIGHT:-0.1}"
TRACE_CENTER_TRAJECTORIES="${TRACE_CENTER_TRAJECTORIES:-True}"
TRACE_FILTER_MIXED="${TRACE_FILTER_MIXED:-False}"
TRACE_FILTER_CANDIDATE_FACTOR="${TRACE_FILTER_CANDIDATE_FACTOR:-1.0}"
TRACE_FILTER_CANDIDATE_COUNT="${TRACE_FILTER_CANDIDATE_COUNT:-0}"
TRACE_FILTER_BATCH_SIZE="${TRACE_FILTER_BATCH_SIZE:-1}"
TRACE_FILTER_MIXED_FILL_FRACTION="${TRACE_FILTER_MIXED_FILL_FRACTION:-0.5}"
ANSWER_TRACE_FILTER_MIXED="${ANSWER_TRACE_FILTER_MIXED:-False}"
ANSWER_TRACE_RESAMPLE_MIXED="${ANSWER_TRACE_RESAMPLE_MIXED:-False}"
ANSWER_TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS="${ANSWER_TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS:-4}"
ANSWER_TRACE_RESAMPLE_MIXED_TARGET_FRAC="${ANSWER_TRACE_RESAMPLE_MIXED_TARGET_FRAC:-1.0}"
TEST_TIMES="${TEST_TIMES:-5}"
TRACE_OOD_MAX_SAMPLES="${TRACE_OOD_MAX_SAMPLES:-0}"

if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  echo "${WATCH_SESSION} already exists."
  exit 0
fi

tmux new-session -d -s "${WATCH_SESSION}" \
  "set -euo pipefail; \
   cd '${ROOT}'; \
   echo '[TRACE auto] waiting for ${V0_SESSION}' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'; \
   while tmux has-session -t '${V0_SESSION}' 2>/dev/null; do sleep 300; done; \
   until CKPT=\$(bash -lc 'cd \"${ROOT}\" && scripts/trace_auto_v2_after_v0.sh --print-best' 2>/dev/null); do \
     echo '[TRACE auto] no v0 checkpoint yet; waiting' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'; \
     sleep 120; \
   done; \
	   echo \"[TRACE auto] using checkpoint: \${CKPT}\" | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'; \
	   GROUP_SIZE='${GROUP_SIZE}' EXP_BATCH_SIZE='${EXP_BATCH_SIZE}' N_TRAIN_SAMPLES='${N_TRAIN_SAMPLES}' MAX_EPOCHS='${MAX_EPOCHS}' MAX_N_LATENT_FORWARD='${MAX_N_LATENT_FORWARD}' TRACE_REWARD_WEIGHT='${TRACE_REWARD_WEIGHT}' LR='${LR}' LATENT_TEMPERATURE='${LATENT_TEMPERATURE}' TRACE_CENTER_TRAJECTORIES='${TRACE_CENTER_TRAJECTORIES}' TRACE_FILTER_MIXED='${TRACE_FILTER_MIXED}' TRACE_FILTER_CANDIDATE_FACTOR='${TRACE_FILTER_CANDIDATE_FACTOR}' TRACE_FILTER_CANDIDATE_COUNT='${TRACE_FILTER_CANDIDATE_COUNT}' TRACE_FILTER_BATCH_SIZE='${TRACE_FILTER_BATCH_SIZE}' TRACE_FILTER_MIXED_FILL_FRACTION='${TRACE_FILTER_MIXED_FILL_FRACTION}' bash scripts/trace_train_v2_rl.sh '${V2_GPU}' \"\${CKPT}\"; \
	   GROUP_SIZE='${GROUP_SIZE}' EXP_BATCH_SIZE='${EXP_BATCH_SIZE}' N_TRAIN_SAMPLES='${N_TRAIN_SAMPLES}' MAX_EPOCHS='${MAX_EPOCHS}' MAX_N_LATENT_FORWARD='${MAX_N_LATENT_FORWARD}' LR='${LR}' LATENT_TEMPERATURE='${LATENT_TEMPERATURE}' TRACE_FILTER_MIXED='${ANSWER_TRACE_FILTER_MIXED}' TRACE_FILTER_CANDIDATE_FACTOR='${TRACE_FILTER_CANDIDATE_FACTOR}' TRACE_FILTER_CANDIDATE_COUNT='${TRACE_FILTER_CANDIDATE_COUNT}' TRACE_FILTER_BATCH_SIZE='${TRACE_FILTER_BATCH_SIZE}' TRACE_FILTER_MIXED_FILL_FRACTION='${TRACE_FILTER_MIXED_FILL_FRACTION}' TRACE_RESAMPLE_MIXED='${ANSWER_TRACE_RESAMPLE_MIXED}' TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS='${ANSWER_TRACE_RESAMPLE_MIXED_MAX_ATTEMPTS}' TRACE_RESAMPLE_MIXED_TARGET_FRAC='${ANSWER_TRACE_RESAMPLE_MIXED_TARGET_FRAC}' bash scripts/trace_train_answer_only_rl.sh '${ANSWER_ONLY_GPU}' \"\${CKPT}\"; \
	   DIAG_GROUP_SIZE='${GROUP_SIZE}' TEST_TIMES='${TEST_TIMES}' TRACE_OOD_MAX_SAMPLES='${TRACE_OOD_MAX_SAMPLES}' bash scripts/trace_auto_post_after_rl.sh '${POST_GPU}' '${POST_SAMPLES}'; \
	   echo '[TRACE auto] launched v2, answer-only RL, and post-validation watcher' | tee -a '${RUN_OUTPUTS}/${WATCH_SESSION}.log'"

echo "Started ${WATCH_SESSION}"
