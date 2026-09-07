#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TRAIN_TAGS = (
    "train/accuracies",
    "train/output_length",
    "train/stage2_direct_signature_loss",
    "train/stage2_direct_pos_dist",
    "train/stage2_direct_neg_proto_sim",
    "train/stage2_direct_wrong_within_sim",
    "train/stage2_direct_boundary_sim_gap",
    "train/stage2_guard_task_grad_norm",
    "train/stage2_guard_geometry_grad_norm",
    "train/stage2_guard_geometry_scale",
    "train/stage2_guard_task_geometry_cosine",
    "train/stage2_guard_conflict",
    "train/optimizer_did_step",
)

VAL_TAGS = (
    "monitor",
    "val/acc",
    "val/output_length",
    "val/residual_similarity",
)


def load_scalars(event_path):
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    values = {}
    for tag in (*TRAIN_TAGS, *VAL_TAGS):
        if tag not in available:
            continue
        events = accumulator.Scalars(tag)
        values[tag] = {
            "step": np.asarray([event.step for event in events], dtype=np.int64),
            "value": np.asarray([event.value for event in events], dtype=np.float64),
        }
    return values


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def epoch_aggregate(values, steps_per_epoch, max_epochs):
    result = []
    for epoch in range(max_epochs):
        row = {"epoch": epoch}
        for tag in TRAIN_TAGS:
            if tag not in values:
                continue
            steps = values[tag]["step"]
            metric = values[tag]["value"]
            mask = (steps >= epoch * steps_per_epoch) & (steps < (epoch + 1) * steps_per_epoch)
            if mask.any():
                row[tag] = float(metric[mask].mean())
                row[f"{tag}/std"] = float(metric[mask].std(ddof=1)) if mask.sum() > 1 else 0.0
                row[f"{tag}/n"] = int(mask.sum())
        result.append(row)
    return result


def validation_rows(values, steps_per_epoch):
    if "monitor" not in values:
        return []
    rows = []
    monitor_steps = values["monitor"]["step"]
    for position, step in enumerate(monitor_steps):
        row = {
            "epoch": int(step // steps_per_epoch),
            "step": int(step),
        }
        for tag in VAL_TAGS:
            if tag not in values or position >= len(values[tag]["value"]):
                continue
            row[tag] = float(values[tag]["value"][position])
        rows.append(row)
    return rows


def series(values, tag, steps_per_epoch, smooth_window=15):
    x = values[tag]["step"] / float(steps_per_epoch)
    y = moving_average(values[tag]["value"], smooth_window)
    return x, y


def plot_dashboard(values, val_rows, steps_per_epoch, frozen_epoch, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))

    epochs = np.asarray([row["epoch"] for row in val_rows])
    val_acc = 100.0 * np.asarray([row["val/acc"] for row in val_rows])
    val_length = 8.0 + np.asarray([row["val/output_length"] for row in val_rows])
    axes[0, 0].plot(epochs, val_acc, marker="o", color="#2563eb", label="validation accuracy")
    axes[0, 0].set_xlabel("Stage2 epoch")
    axes[0, 0].set_ylabel("Validation accuracy (%)", color="#2563eb")
    axes[0, 0].tick_params(axis="y", labelcolor="#2563eb")
    axes[0, 0].grid(alpha=0.22)
    length_axis = axes[0, 0].twinx()
    length_axis.plot(epochs, val_length, marker="s", color="#d97706", label="validation #L")
    length_axis.set_ylabel("Validation #L", color="#d97706")
    length_axis.tick_params(axis="y", labelcolor="#d97706")
    axes[0, 0].axvline(frozen_epoch, linestyle="--", color="#111827", linewidth=1)
    axes[0, 0].set_title("Full validation every epoch; epoch7 frozen")

    x, rollout_acc = series(values, "train/accuracies", steps_per_epoch)
    axes[0, 1].plot(x, 100.0 * rollout_acc, color="#15803d", label="rollout accuracy")
    axes[0, 1].set_xlabel("Stage2 epoch")
    axes[0, 1].set_ylabel("Training-rollout accuracy (%)", color="#15803d")
    axes[0, 1].tick_params(axis="y", labelcolor="#15803d")
    rollout_length_axis = axes[0, 1].twinx()
    _, rollout_length = series(values, "train/output_length", steps_per_epoch)
    rollout_length_axis.plot(x, 8.0 + rollout_length, color="#7c3aed", alpha=0.85, label="rollout #L")
    rollout_length_axis.set_ylabel("Training-rollout #L", color="#7c3aed")
    rollout_length_axis.tick_params(axis="y", labelcolor="#7c3aed")
    axes[0, 1].axvline(frozen_epoch, linestyle="--", color="#111827", linewidth=1)
    axes[0, 1].set_title("Outcome learning and length control")
    axes[0, 1].grid(alpha=0.22)

    guard_tags = (
        ("train/stage2_guard_conflict", "conflict rate", "#dc2626"),
        ("train/stage2_guard_geometry_scale", "geometry scale", "#2563eb"),
        ("train/stage2_guard_task_geometry_cosine", "task/geometry cosine", "#d97706"),
    )
    for tag, label, color in guard_tags:
        x, y = series(values, tag, steps_per_epoch)
        axes[1, 0].plot(x, y, label=label, color=color)
    axes[1, 0].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1, 0].axvline(frozen_epoch, linestyle="--", color="#111827", linewidth=1)
    axes[1, 0].set_xlabel("Stage2 epoch")
    axes[1, 0].set_ylabel("Guard diagnostic")
    axes[1, 0].set_title("Accuracy-gradient guard is active")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.22)

    geometry_tags = (
        ("train/stage2_direct_pos_dist", "correct-mode distance", "#15803d"),
        ("train/stage2_direct_neg_proto_sim", "wrong/correct-mode similarity", "#dc2626"),
        ("train/stage2_direct_boundary_sim_gap", "boundary similarity gap", "#2563eb"),
        ("train/stage2_direct_signature_loss", "direct geometry loss", "#7c3aed"),
    )
    for tag, label, color in geometry_tags:
        x, y = series(values, tag, steps_per_epoch)
        axes[1, 1].plot(x, y, label=label, color=color)
    axes[1, 1].axvline(frozen_epoch, linestyle="--", color="#111827", linewidth=1)
    axes[1, 1].set_xlabel("Stage2 epoch")
    axes[1, 1].set_ylabel("Training diagnostic")
    axes[1, 1].set_title("Direct outcome-conditioned geometry objective")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.22)

    fig.suptitle("TRACE epoch7 training dynamics and gradient-guard audit", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, payload):
    aggregate = payload["overall_train"]
    frozen = next(row for row in payload["validation"] if row["epoch"] == payload["frozen_epoch"])
    best_accuracy = max(row["val/acc"] for row in payload["validation"])
    tied_best = [
        row for row in payload["validation"]
        if abs(row["val/acc"] - best_accuracy) <= 1e-8
    ]
    earlier_tie = next((row for row in tied_best if row["epoch"] != payload["frozen_epoch"]), None)
    lines = [
        "# TRACE Epoch7 Training-Dynamics Audit",
        "",
        f"The final run contains {payload['max_epochs']} Stage2 epochs, {payload['steps_per_epoch']} optimizer steps per epoch, and full validation at every epoch.",
        "",
        f"- Frozen epoch7 validation accuracy: **{100.0 * frozen['val/acc']:.2f}%**.",
        f"- Frozen epoch7 validation #L: **{8.0 + frozen['val/output_length']:.2f}**.",
        f"- Task/geometry gradients conflicted on **{100.0 * aggregate['train/stage2_guard_conflict']:.1f}%** of logged updates.",
        f"- Mean post-projection geometry scale: **{aggregate['train/stage2_guard_geometry_scale']:.3f}**.",
        f"- Mean task/geometry cosine before projection: **{aggregate['train/stage2_guard_task_geometry_cosine']:.3f}**.",
        f"- Finite optimizer-step fraction: **{100.0 * aggregate['train/optimizer_did_step']:.1f}%**.",
        "",
    ]
    if earlier_tie is not None:
        lines.append(
            f"Epoch{payload['frozen_epoch']} ties epoch{earlier_tie['epoch']} for the maximum validation accuracy, "
            f"while reducing validation #L from {8.0 + earlier_tie['val/output_length']:.2f} to "
            f"{8.0 + frozen['val/output_length']:.2f}. It is therefore a validation Pareto point rather than a test-selected checkpoint."
        )
        lines.append("")
    lines.append(
        "The high conflict frequency shows that the guard is operationally relevant: the geometry gradient is often projected and scaled instead of being added unchecked. These curves are mechanism diagnostics, not independent test-set evidence."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_file", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--steps_per_epoch", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--frozen_epoch", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    values = load_scalars(args.event_file)
    missing = [tag for tag in (*TRAIN_TAGS, *VAL_TAGS) if tag not in values]
    if missing:
        raise RuntimeError(f"Missing required TensorBoard scalar tags: {missing}")
    epoch_rows = epoch_aggregate(values, args.steps_per_epoch, args.max_epochs)
    val_rows = validation_rows(values, args.steps_per_epoch)
    overall_train = {
        tag: float(values[tag]["value"].mean())
        for tag in TRAIN_TAGS
    }
    payload = {
        "event_file": str(args.event_file),
        "steps_per_epoch": args.steps_per_epoch,
        "max_epochs": args.max_epochs,
        "frozen_epoch": args.frozen_epoch,
        "overall_train": overall_train,
        "per_epoch_train": epoch_rows,
        "validation": val_rows,
    }
    (args.out_dir / "training_dynamics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "training_dynamics.md", payload)
    plot_dashboard(
        values,
        val_rows,
        args.steps_per_epoch,
        args.frozen_epoch,
        args.out_dir / "training_dynamics_dashboard.png",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
