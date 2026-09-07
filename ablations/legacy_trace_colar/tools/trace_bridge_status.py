#!/usr/bin/env python
import argparse
import json
import subprocess
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:  # pragma: no cover
    EventAccumulator = None


RUNS = [
    ("baseline", "bridge_qwen3_instruct_hybrid_compact_anchor_gate", "20260707-180307_890146_20260707_trace_bridge_baseline_gpu7_bridge_baseline"),
    ("stage1_v2", "trace_bridge_qwen3_instruct", "20260707-181012_487435_20260707_trace_bridge_stage1_v2_gpu2_stage1"),
    ("stage2_conservative_v2_stage1", "trace_bridge_qwen3_instruct", "20260707-181012_622417_20260707_trace_bridge_stage2_conservative_v2_gpu4_stage1"),
    ("stage2_strong_v2_stage1", "trace_bridge_qwen3_instruct", "20260707-181012_646020_20260707_trace_bridge_stage2_strong_v2_gpu6_stage1"),
    ("structmv_stage1", "trace_bridge_qwen3_instruct_structmv", "20260707-182403_912579_20260707_trace_bridge_structmv_stage1_gpu1_stage1"),
]


def discover_runs(log_root, active_only=False):
    active_tokens = set()
    if active_only:
        for session in tmux_sessions():
            if session.startswith("trace_bridge_"):
                active_tokens.add(session.removeprefix("trace_bridge_").removesuffix("_0707"))
    discovered = []
    for model_name in (
        "bridge_qwen3_instruct_hybrid_compact_anchor_gate",
        "trace_bridge_qwen3_instruct",
        "trace_bridge_qwen3_instruct_structmv",
    ):
        root = log_root / model_name / "qsa-gsm"
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("*20260707*")):
            label = run_dir.name.split("_20260707_", 1)[-1] if "_20260707_" in run_dir.name else run_dir.name
            if active_only and not any(token in run_dir.name for token in active_tokens):
                continue
            discovered.append((label, model_name, run_dir.name))
    seen = set()
    merged = []
    for run in RUNS + discovered:
        key = (run[1], run[2])
        if key in seen:
            continue
        seen.add(key)
        merged.append(run)
    return merged


def tmux_sessions():
    try:
        out = subprocess.check_output(["tmux", "ls"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line.split(":", 1)[0] for line in out.splitlines()]


def latest_scalar(event_file, tag):
    if EventAccumulator is None or event_file is None:
        return None
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    values = acc.Scalars(tag)
    if not values:
        return None
    last = values[-1]
    return {"step": last.step, "value": last.value, "count": len(values)}


def summarize_run(log_root, label, model_name, version):
    run_dir = log_root / model_name / "qsa-gsm" / version
    events = sorted(run_dir.glob("events.out.tfevents*"))
    event_file = events[-1] if events else None
    ckpts = sorted(run_dir.glob("checkpoints/*.ckpt"))
    scalars = {}
    for tag in (
        "train/total_loss",
        "train/dep_f1",
        "train/trace_stage1_path_loss",
        "train/trace_stage1_multiview_distance",
        "val/acc",
        "test/acc",
        "train/trace_rl/mode_count",
        "train/trace_rl/effective_modes",
    ):
        value = latest_scalar(event_file, tag)
        if value is not None:
            scalars[tag] = value
    return {
        "label": label,
        "run_dir": str(run_dir),
        "event_file": str(event_file) if event_file else None,
        "ckpts": [str(p) for p in ckpts],
        "scalars": scalars,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="/disk1/dingxukai/trace_colar/logs")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--active-only", action="store_true")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    sessions = tmux_sessions()
    result = {
        "tmux_trace_bridge": [s for s in sessions if "trace_bridge" in s],
        "runs": [summarize_run(log_root, *run) for run in discover_runs(log_root, active_only=args.active_only)],
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("tmux:", ", ".join(result["tmux_trace_bridge"]) or "none")
    for run in result["runs"]:
        print(f"\n[{run['label']}]")
        print("run_dir:", run["run_dir"])
        print("ckpts:", len(run["ckpts"]))
        for tag, value in run["scalars"].items():
            print(f"{tag}: step={value['step']} value={value['value']:.6f} n={value['count']}")


if __name__ == "__main__":
    main()
