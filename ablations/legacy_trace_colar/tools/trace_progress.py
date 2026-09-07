#!/usr/bin/env python3
import argparse
from pathlib import Path
from statistics import mean
from datetime import timedelta


DEFAULT_TAGS = [
    "train/total_loss",
    "train/answer_loss",
    "train/state_loss",
    "train/transition_loss",
    "train/trace/state_cos",
    "train/trace/transition_cos",
    "train/trace/student_step_cos",
    "train/trace_stage1_path_loss",
    "train/trace_stage1_anchor_loss",
    "train/trace_stage1_direction_loss",
    "train/trace_stage1_step_loss",
    "train/trace_stage1_bootstrap_loss",
    "train/trace_stage1_compression_view",
    "train/trace_stage1_anchor_count",
    "train/trace_stage1_bootstrap_anchor_count",
    "train/rewards",
    "train/accuracies",
    "train/n_latent_forward",
    "train/trace/mode_count",
    "train/trace/effective_modes",
    "train/trace/hard_count",
    "train/trace/mixed_frac",
    "val/acc",
    "val/monitor",
    "val/n_latent_forward",
    "test/acc",
    "test/n_latent_forward",
    "monitor",
    "epoch",
]


def latest_run(log_root: Path, run_contains: str):
    runs = [p for p in log_root.glob("*") if p.is_dir() and run_contains in p.name]
    if not runs:
        raise SystemExit(f"no run matching {run_contains!r} under {log_root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_root",
        default="logs/trace_trajectory_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl",
    )
    parser.add_argument("--run_contains", required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=6726)
    args = parser.parse_args()

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        raise SystemExit(f"tensorboard is required: {exc}")

    run_dir = latest_run(Path(args.log_root), args.run_contains)
    events = sorted(run_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not events:
        raise SystemExit(f"no tensorboard event file under {run_dir}")
    event_path = events[-1]
    ea = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()

    print(f"run: {run_dir.name}")
    print(f"event: {event_path.name}")
    tags = ea.Tags().get("scalars", [])
    train_vals = ea.Scalars("train/total_loss") if "train/total_loss" in tags else []
    if train_vals:
        last_step = train_vals[-1].step
        progress = 100.0 * (last_step % args.steps_per_epoch) / args.steps_per_epoch
        epoch_idx = last_step // args.steps_per_epoch
        print(f"progress: epoch={epoch_idx}, step={last_step}, epoch_progress={progress:.1f}%")
        if len(train_vals) >= 2:
            dt = train_vals[-1].wall_time - train_vals[0].wall_time
            ds = train_vals[-1].step - train_vals[0].step
            if ds > 0:
                sec_per_step = dt / ds
                remaining = args.steps_per_epoch - (last_step % args.steps_per_epoch)
                print(f"speed: {sec_per_step:.3f} sec/step")
                print(f"eta_epoch_end: {timedelta(seconds=int(remaining * sec_per_step))}")

    for tag in DEFAULT_TAGS:
        if tag not in tags:
            continue
        vals = ea.Scalars(tag)
        if not vals:
            continue
        window_vals = [v.value for v in vals[-args.window :]]
        print(
            f"{tag}: step={vals[-1].step}, last={vals[-1].value:.6g}, "
            f"window{min(args.window, len(window_vals))}={mean(window_vals):.6g}, n={len(vals)}"
        )


if __name__ == "__main__":
    main()
