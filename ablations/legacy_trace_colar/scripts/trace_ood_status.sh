#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/dingxukai/trace_colar"
RUN_DIR="${ROOT}/logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260704-044959_549439_trace_v2_rl_qwen3_c5_g8_gpu7"
EVIDENCE="${ROOT}/run_outputs/trace/evidence_snapshot.json"

cd "${ROOT}"

echo "== Selected-best OOD sessions =="
tmux ls 2>/dev/null | rg 'trace_test_ood_v2_g8_best0341|trace_refresh_after_v2_g8_best0341|trace_stop_gsmhard_.*_0341' || true

echo
echo "== GPU memory =="
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

echo
echo "== Latest selected-best test files =="
find "${RUN_DIR}" -maxdepth 1 -name 'test_*.json' \
  -printf '%T@ %TY-%Tm-%Td %TH:%TM:%TS %s %f\n' 2>/dev/null \
  | sort -nr | head -20 || true

echo
echo "== Selected-best result summaries =="
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, "tools")
from trace_summarize_runs import summarize_json

run_dir = Path("logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/20260704-044959_549439_trace_v2_rl_qwen3_c5_g8_gpu7")
rows = []
for path in sorted(run_dir.glob("test_*.json"), key=lambda p: p.stat().st_mtime)[-30:]:
    try:
        row = summarize_json(path)
    except Exception:
        continue
    ckpt = row.get("ckpt_path") or "-"
    if ckpt != "-":
        ckpt = Path(ckpt).name
    rows.append(
        (
            path.name,
            row.get("dataset", "-"),
            row.get("effective_test_times", "-"),
            row.get("n_items", "-"),
            row.get("acc"),
            row.get("n_latent_forward"),
            row.get("eval_signature") or "legacy_default",
            ckpt,
        )
    )

print("| file | dataset | times | n | acc | #L | eval | ckpt |")
print("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
for file, dataset, times, n_items, acc, n_latent, eval_signature, ckpt in rows:
    acc_text = "-" if acc is None else f"{acc:.4f}"
    latent_text = "-" if n_latent is None else f"{n_latent:.2f}"
    eval_text = "default"
    if "min_n_latent_forward=40" in eval_signature:
        eval_text = "minL40"
    elif "min_n_latent_forward=32" in eval_signature:
        eval_text = "minL32"
    elif "min_n_latent_forward=24" in eval_signature:
        eval_text = "minL24"
    elif eval_signature not in ("legacy_default", "unknown"):
        eval_text = eval_signature
    print(f"| {file} | {dataset} | {times} | {n_items} | {acc_text} | {latent_text} | {eval_text} | {ckpt} |")
PY

echo
echo "== OOD log tails =="
for log in "${ROOT}"/run_outputs/trace/trace_test_ood_v2_g8_best0341_*.log; do
  echo "-- ${log##*/} --"
  if [ -f "${log}" ]; then
    rg '\[TRACE OOD\]|Test progress|Testing DataLoader|test_acc|Test results|done|Traceback|CUDA out of memory|Error|error' "${log}" | tail -30 || true
  else
    echo "missing"
  fi
done

echo "-- selected-best log.txt --"
if [ -f "${RUN_DIR}/log.txt" ]; then
  rg 'Test progress|test_acc|Test results|Traceback|CUDA out of memory|Error|error' "${RUN_DIR}/log.txt" | tail -40 || true
else
  echo "missing"
fi

echo
echo "== Evidence status =="
if [ -f "${EVIDENCE}" ]; then
  python - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("run_outputs/trace/evidence_snapshot.json").read_text())
print(json.dumps(data.get("status", {}), indent=2))
PY
else
  echo "missing ${EVIDENCE}"
fi
