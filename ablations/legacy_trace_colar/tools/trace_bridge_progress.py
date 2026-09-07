#!/usr/bin/env python
import argparse
import json
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = [
    {
        "label": "baseline",
        "model": "bridge_qwen3_instruct_hybrid_compact_anchor_gate",
        "version": "20260707-180307_890146_20260707_trace_bridge_baseline_gpu7_bridge_baseline",
        "out": "20260707_trace_bridge_baseline_gpu7",
        "epoch_steps": 6726,
    },
    {
        "label": "stage1_v2",
        "model": "trace_bridge_qwen3_instruct",
        "version": "20260707-181012_487435_20260707_trace_bridge_stage1_v2_gpu2_stage1",
        "out": "20260707_trace_bridge_stage1_v2_gpu2",
        "epoch_steps": 6726,
    },
    {
        "label": "stage2_cons_v2_stage1",
        "model": "trace_bridge_qwen3_instruct",
        "version": "20260707-181012_622417_20260707_trace_bridge_stage2_conservative_v2_gpu4_stage1",
        "out": "20260707_trace_bridge_stage2_conservative_v2_gpu4",
        "epoch_steps": 6726,
    },
    {
        "label": "stage2_strong_v2_stage1",
        "model": "trace_bridge_qwen3_instruct",
        "version": "20260707-181012_646020_20260707_trace_bridge_stage2_strong_v2_gpu6_stage1",
        "out": "20260707_trace_bridge_stage2_strong_v2_gpu6",
        "epoch_steps": 6726,
    },
    {
        "label": "structmv_stage1",
        "model": "trace_bridge_qwen3_instruct_structmv",
        "version": "20260707-182403_912579_20260707_trace_bridge_structmv_stage1_gpu1_stage1",
        "out": "20260707_trace_bridge_structmv_stage1_gpu1",
        "epoch_steps": 6726,
    },
]


def load_scalars(event_file):
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        vals = acc.Scalars(tag)
        if vals:
            out[tag] = vals[-1]
    return out


def summarize_run(run, log_root, out_root):
    run_dir = log_root / run["model"] / "qsa-gsm" / run["version"]
    events = sorted(run_dir.glob("events.out.tfevents*"))
    scalars = load_scalars(events[-1]) if events else {}
    step = max((v.step for v in scalars.values()), default=0)
    epoch_steps = run["epoch_steps"]
    progress = min(100.0, 100.0 * step / max(epoch_steps, 1))

    first_wall = None
    last_wall = None
    if events and scalars:
        wall_times = [v.wall_time for v in scalars.values()]
        first_wall = min(wall_times)
        last_wall = max(wall_times)
    eta_minutes = None
    if first_wall and last_wall and step > 0 and last_wall > first_wall:
        rate = step / (last_wall - first_wall)
        eta_minutes = max(0.0, (epoch_steps - step) / max(rate, 1e-6) / 60.0)

    out_dir = out_root / run["out"]
    return {
        "label": run["label"],
        "step": step,
        "epoch_steps": epoch_steps,
        "first_epoch_progress": progress,
        "eta_minutes_to_first_val": eta_minutes,
        "train_loss": scalars.get("train/total_loss").value if "train/total_loss" in scalars else None,
        "dep_f1": scalars.get("train/dep_f1").value if "train/dep_f1" in scalars else None,
        "path_loss": scalars.get("train/trace_stage1_path_loss").value if "train/trace_stage1_path_loss" in scalars else None,
        "multiview_distance": scalars.get("train/trace_stage1_multiview_distance").value
        if "train/trace_stage1_multiview_distance" in scalars
        else None,
        "ckpt_count": len(list((run_dir / "checkpoints").glob("*.ckpt"))),
        "has_summary": any(out_dir.glob("summary_*.md")),
        "has_visual": any(out_dir.glob("visual_*_gsm8k_aug/*.png")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="/disk1/dingxukai/trace_colar/logs")
    parser.add_argument("--out-root", default="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = [summarize_run(run, Path(args.log_root), Path(args.out_root)) for run in RUNS]
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print("| Run | Step | First-val | ETA min | Loss | Path | MultiView | ckpt | summary | visual |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        def fmt(value, digits=3):
            return "-" if value is None else f"{value:.{digits}f}"

        print(
            "| {label} | {step}/{epoch_steps} | {progress:.1f}% | {eta} | {loss} | {path} | {mv} | {ckpt} | {summary} | {visual} |".format(
                label=row["label"],
                step=row["step"],
                epoch_steps=row["epoch_steps"],
                progress=row["first_epoch_progress"],
                eta=fmt(row["eta_minutes_to_first_val"], 1),
                loss=fmt(row["train_loss"]),
                path=fmt(row["path_loss"]),
                mv=fmt(row["multiview_distance"], 6),
                ckpt=row["ckpt_count"],
                summary="yes" if row["has_summary"] else "no",
                visual="yes" if row["has_visual"] else "no",
            )
        )


if __name__ == "__main__":
    main()
