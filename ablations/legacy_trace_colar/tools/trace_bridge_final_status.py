#!/usr/bin/env python
import argparse
import json
import re
import subprocess
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PHASE_RE = re.compile(
    r"(?P<tag>20\d{6}_trace_bridge_.*_gpu(?P<gpu>\d+))_(?P<phase>stage1|stage2_conservative|stage2_strong|bridge_baseline)$"
)
TRACE_SESSION_RE = re.compile(r"^trace_bridge_(?P<inner>.+)_(?P<mmdd>\d{4})$")
AUX_SESSION_PARTS = {
    "audit_waiter",
    "status_watch",
    "ckpt_snapshot_watch",
    "snapshot_eval_after",
}


def tmux_sessions():
    try:
        proc = subprocess.run(["tmux", "ls"], check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return set()
    sessions = set()
    for line in proc.stdout.splitlines():
        if ":" in line:
            sessions.add(line.split(":", 1)[0])
    return sessions


def load_scalars(event_file: Path):
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    values = {}
    history = {}
    for tag in acc.Tags().get("scalars", []):
        scalars = acc.Scalars(tag)
        if scalars:
            values[tag] = scalars[-1]
            history[tag] = scalars
    return values, history


def latest_value(values, tag):
    item = values.get(tag)
    return None if item is None else float(item.value)


def max_value(history, tag):
    vals = history.get(tag) or []
    return None if not vals else max(float(x.value) for x in vals)


def estimate_eta_minutes(history, current_step, target_step):
    if current_step <= 0 or target_step is None or current_step >= target_step:
        return None
    scalar_points = [point for points in history.values() for point in points if point.step <= current_step]
    if not scalar_points:
        return None
    first = min(scalar_points, key=lambda p: p.wall_time)
    last = max(scalar_points, key=lambda p: p.wall_time)
    if last.wall_time <= first.wall_time or last.step <= first.step:
        return None
    steps_per_second = (last.step - first.step) / (last.wall_time - first.wall_time)
    if steps_per_second <= 0:
        return None
    return (target_step - current_step) / steps_per_second / 60.0


def fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def find_event_runs(log_root: Path):
    runs = []
    model_roots = list(log_root.glob("trace_bridge_qwen3_instruct*/qsa-gsm"))
    model_roots.extend(log_root.glob("bridge_qwen3_instruct_hybrid_compact_anchor_gate/qsa-gsm"))
    for model_dir in sorted(model_roots):
        for run_dir in sorted(model_dir.glob("*trace_bridge*")):
            match = PHASE_RE.search(run_dir.name)
            if not match:
                continue
            events = sorted(run_dir.glob("events.out.tfevents*"))
            runs.append(
                {
                    "model": model_dir.parent.name,
                    "run_dir": run_dir,
                    "run_tag": match.group("tag"),
                    "phase": match.group("phase"),
                    "gpu": match.group("gpu"),
                    "events": events,
                }
            )
    return runs


def summarize_run(run, out_root: Path, sessions):
    values = {}
    history = {}
    if run["events"]:
        values, history = load_scalars(run["events"][-1])
    step = max((int(v.step) for v in values.values()), default=0)
    epoch_steps = 6726 if run["phase"] == "stage1" else None
    first_val_progress = None
    if epoch_steps:
        first_val_progress = min(100.0, 100.0 * step / epoch_steps)
    eta_minutes = estimate_eta_minutes(history, step, epoch_steps)

    out_dir = out_root / run["run_tag"]
    ckpts = sorted(out_dir.glob("*best_ckpt.txt"))
    snapshot_ckpts = sorted(out_dir.glob("ckpt_snapshots/*/checkpoints/*.ckpt"))
    summaries = sorted(out_dir.glob("summary_*.md"))
    visual_records = sorted(out_dir.glob("eval_*_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt"))
    visual_pngs = sorted(out_dir.glob("visual_*_gsm8k_aug/*.png"))
    geometry_jsons = sorted(out_dir.glob("visual_*_gsm8k_aug/geometry_200/*.json"))
    session = session_from_run_tag(run["run_tag"])

    return {
        "model": run["model"],
        "tag": run["run_tag"],
        "phase": run["phase"],
        "gpu": run["gpu"],
        "session": session,
        "alive": session in sessions,
        "step": step,
        "first_val_progress": first_val_progress,
        "eta_minutes_to_first_val": eta_minutes,
        "loss": latest_value(values, "train/total_loss"),
        "path": latest_value(values, "train/trace_stage1_path_loss"),
        "anchor_span": latest_value(values, "train/trace_stage1_progress_anchor_span"),
        "mv_distance": latest_value(values, "train/trace_stage1_multiview_distance"),
        "teacher_distance": latest_value(values, "train/trace_stage1_multiview_teacher_distance"),
        "teacher_distance_max": max_value(history, "train/trace_stage1_multiview_teacher_distance"),
        "monitor": latest_value(values, "monitor"),
        "val_acc": latest_value(values, "val/acc"),
        "test_acc": latest_value(values, "test/acc"),
        "ckpt_count": len(ckpts) + len(snapshot_ckpts),
        "summary_count": len(summaries),
        "visual_record_count": len(visual_records),
        "visual_png_count": len(visual_pngs),
        "geometry_json_count": len(geometry_jsons),
        "out_dir": str(out_dir),
        "event_dir": str(run["run_dir"]),
    }


def phase_from_inner(inner):
    if "_bridge_baseline_" in inner or inner.startswith("full_bridge_baseline_"):
        return "queued:bridge_baseline"
    if "_stage2_conservative_" in inner:
        return "queued:stage2_conservative"
    if "_stage2_strong_" in inner:
        return "queued:stage2_strong"
    if "_stage1_" in inner:
        return "queued:stage1"
    return "queued"


def session_from_run_tag(run_tag):
    match = re.match(r"^(20\d{2})(\d{4})_trace_bridge_(.+)$", run_tag)
    if not match:
        return run_tag
    return f"trace_bridge_{match.group(3)}_{match.group(2)}"


def run_tag_from_session_inner(inner, mmdd):
    return f"2026{mmdd}_trace_bridge_{inner}"


def display_run_tag(run_tag):
    return re.sub(r"^20\d{6}_trace_bridge_", "", run_tag)


def gpu_from_inner(inner):
    match = re.search(r"gpu(\d+)$", inner)
    return match.group(1) if match else "-"


def queued_session_rows(sessions, represented_sessions, out_root: Path):
    rows = []
    for session in sorted(sessions):
        match = TRACE_SESSION_RE.match(session)
        if not match:
            continue
        inner = match.group("inner")
        if any(part in inner for part in AUX_SESSION_PARTS) or session in represented_sessions:
            continue
        run_tag = run_tag_from_session_inner(inner, match.group("mmdd"))
        out_dir = out_root / run_tag
        snapshot_ckpts = list(out_dir.glob("ckpt_snapshots/*/checkpoints/*.ckpt"))
        rows.append(
            {
                "model": "queued/no-events",
                "tag": run_tag,
                "phase": phase_from_inner(inner),
                "gpu": gpu_from_inner(inner),
                "session": session,
                "alive": True,
                "step": 0,
                "first_val_progress": None,
                "eta_minutes_to_first_val": None,
                "loss": None,
                "path": None,
                "anchor_span": None,
                "mv_distance": None,
                "teacher_distance": None,
                "teacher_distance_max": None,
                "monitor": None,
                "val_acc": None,
                "test_acc": None,
                "ckpt_count": len(list(out_dir.glob("*best_ckpt.txt"))) + len(snapshot_ckpts),
                "summary_count": len(list(out_dir.glob("summary_*.md"))),
                "visual_record_count": len(list(out_dir.glob("eval_*_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt"))),
                "visual_png_count": len(list(out_dir.glob("visual_*_gsm8k_aug/*.png"))),
                "geometry_json_count": len(list(out_dir.glob("visual_*_gsm8k_aug/geometry_200/*.json"))),
                "out_dir": str(out_dir),
                "event_dir": None,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="/disk1/dingxukai/trace_colar/logs")
    parser.add_argument("--out-root", default="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge")
    parser.add_argument("--run-regex", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sessions = tmux_sessions()
    runs = find_event_runs(Path(args.log_root))
    if args.run_regex:
        pattern = re.compile(args.run_regex)
        runs = [run for run in runs if pattern.search(run["run_tag"])]
    rows = [summarize_run(run, Path(args.out_root), sessions) for run in runs]
    represented_sessions = {row["session"] for row in rows}
    queued_rows = queued_session_rows(sessions, represented_sessions, Path(args.out_root))
    if args.run_regex:
        pattern = re.compile(args.run_regex)
        queued_rows = [row for row in queued_rows if pattern.search(row["tag"])]
    rows.extend(queued_rows)
    rows.sort(key=lambda r: (r["model"], r["tag"], r["phase"]))

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print(
        "| Run | Phase | GPU | Alive | Step | First-val | ETA min | Loss | Path | Anchor span | MV dist | Teacher latest/max | monitor | val acc | ckpt | summary | visual | geom |"
    )
    print(
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        progress = "-" if row["first_val_progress"] is None else f"{row['first_val_progress']:.1f}%"
        teacher = f"{fmt(row['teacher_distance'], 4)}/{fmt(row['teacher_distance_max'], 4)}"
        print(
            "| {run} | {phase} | {gpu} | {alive} | {step} | {progress} | {eta} | {loss} | {path} | {anchor_span} | {mv} | {teacher} | {monitor} | {val_acc} | {ckpt} | {summary} | {visual} | {geom} |".format(
                run=display_run_tag(row["tag"]),
                phase=row["phase"],
                gpu=row["gpu"],
                alive="yes" if row["alive"] else "no",
                step=row["step"],
                progress=progress,
                eta=fmt(row["eta_minutes_to_first_val"], 1),
                loss=fmt(row["loss"]),
                path=fmt(row["path"]),
                anchor_span=fmt(row["anchor_span"], 4),
                mv=fmt(row["mv_distance"], 5),
                teacher=teacher,
                monitor=fmt(row["monitor"]),
                val_acc=fmt(row["val_acc"]),
                ckpt=row["ckpt_count"],
                summary=row["summary_count"],
                visual=row["visual_record_count"] + row["visual_png_count"],
                geom=row["geometry_json_count"],
            )
        )


if __name__ == "__main__":
    main()
