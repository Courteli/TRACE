#!/usr/bin/env python3
"""Render submission-ready TRACE stagewise training dynamics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


COLORS = {
    "blue": "#6687B8",
    "green": "#69B17D",
    "gold": "#E6A314",
    "pink": "#E5A6C4",
    "text": "#27313B",
    "muted": "#66717E",
    "grid": "#D9DEE5",
}

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Liberation Serif",
            "Nimbus Roman No9 L",
            "DejaVu Serif",
        ],
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-log-dir", type=Path, required=True)
    parser.add_argument("--stage2-log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=101)
    return parser.parse_args()


def load_events(path: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def scalar_series(accumulator: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    points = accumulator.Scalars(tag)
    if not points:
        raise ValueError(f"Missing scalar tag: {tag}")
    return (
        np.asarray([point.step for point in points], dtype=np.float64),
        np.asarray([point.value for point in points], dtype=np.float64),
    )


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    window = max(window, 3)
    if window % 2 == 0:
        window -= 1
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], width=0.7, length=2.5)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
        color=COLORS["text"],
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    stage1 = load_events(args.stage1_log_dir)
    stage2 = load_events(args.stage2_log_dir)

    path_steps, path_values = scalar_series(stage1, "train/trace_stage1_path_loss")
    _, position_values = scalar_series(stage1, "train/trace_stage1_position_loss")
    _, direction_values = scalar_series(stage1, "train/trace_stage1_direction_loss")
    view_steps, view_distance = scalar_series(stage1, "train/trace_stage1_multiview_distance")
    _, diversity_hinge = scalar_series(stage1, "train/trace_stage1_multiview_diversity_hinge")
    val_steps, val_accuracy = scalar_series(stage2, "val/acc")
    _, val_output_length = scalar_series(stage2, "val/output_length")

    rows = []
    for idx in range(len(path_steps)):
        rows.append(
            {
                "stage": "TRACE Stage 1",
                "step": int(path_steps[idx]),
                "trajectory_progress_percent": 100.0 * path_steps[idx] / path_steps.max(),
                "path_loss": path_values[idx],
                "position_loss": position_values[idx],
                "direction_loss": direction_values[idx],
                "multiview_distance": view_distance[idx],
                "multiview_diversity_hinge": diversity_hinge[idx],
            }
        )
    for idx in range(len(val_steps)):
        rows.append(
            {
                "stage": "TRACE Stage 2",
                "epoch": idx + 1,
                "step": int(val_steps[idx]),
                "validation_accuracy_percent": 100.0 * val_accuracy[idx],
                "validation_output_length": val_output_length[idx],
            }
        )
    write_csv(args.output_dir / "source_data" / "training_dynamics.csv", rows)

    x = 100.0 * path_steps / path_steps.max()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))

    ax = axes[0]
    style_axis(ax)
    for values, color, label in (
        (path_values, COLORS["blue"], "Combined path"),
        (position_values, COLORS["green"], "Position"),
        (direction_values, COLORS["pink"], "Direction"),
    ):
        ax.plot(x, values, color=color, linewidth=0.45, alpha=0.15)
        ax.plot(x, smooth(values, args.smooth_window), color=color, linewidth=1.35, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Stage 1 training progress (%)")
    ax.set_ylabel("Trajectory loss")
    ax.set_title("Trajectory formation", loc="left", fontweight="bold")
    ax.legend(loc="upper right", handlelength=1.5)
    panel_label(ax, "a")

    ax = axes[1]
    style_axis(ax)
    distance_smooth = smooth(view_distance, args.smooth_window)
    hinge_smooth = smooth(diversity_hinge, args.smooth_window)
    ax.plot(100.0 * view_steps / view_steps.max(), view_distance, color=COLORS["pink"], linewidth=0.45, alpha=0.14)
    ax.plot(100.0 * view_steps / view_steps.max(), distance_smooth, color=COLORS["pink"], linewidth=1.45, label="View distance")
    ax.axhline(0.16, color=COLORS["gold"], linestyle="--", linewidth=1.0, label="Diversity margin")
    ax.fill_between(
        100.0 * view_steps / view_steps.max(),
        0,
        hinge_smooth,
        color=COLORS["blue"],
        alpha=0.14,
        label="Margin violation",
    )
    ax.set_xlabel("Stage 1 training progress (%)")
    ax.set_ylabel("Signature distance / violation")
    ax.set_title("Multi-view separation", loc="left", fontweight="bold")
    ax.legend(loc="upper left", handlelength=1.5)
    panel_label(ax, "b")

    ax = axes[2]
    style_axis(ax)
    epochs = np.arange(1, len(val_accuracy) + 1)
    accuracy_line = ax.plot(
        epochs,
        100.0 * val_accuracy,
        "o-",
        color=COLORS["pink"],
        markeredgecolor=COLORS["text"],
        markeredgewidth=0.35,
        markersize=4.4,
        linewidth=1.45,
        label="Validation accuracy",
    )[0]
    best_idx = int(np.argmax(val_accuracy))
    ax.scatter(
        [epochs[best_idx]],
        [100.0 * val_accuracy[best_idx]],
        marker="*",
        s=66,
        color=COLORS["gold"],
        edgecolor=COLORS["text"],
        linewidth=0.4,
        zorder=4,
    )
    ax.text(
        epochs[best_idx] + 0.12,
        100.0 * val_accuracy[best_idx] + 0.18,
        f"best: {100.0 * val_accuracy[best_idx]:.2f}%",
        ha="left",
        fontsize=5.8,
        color=COLORS["pink"],
        fontweight="bold",
    )
    ax.set_ylim(100.0 * val_accuracy.min() - 0.25, 100.0 * val_accuracy.max() + 0.65)
    ax.set_xlabel("Stage 2 epoch")
    ax.set_ylabel("Validation accuracy (%)", color=COLORS["pink"])
    ax.tick_params(axis="y", colors=COLORS["pink"])
    ax.set_xticks(epochs)
    length_axis = ax.twinx()
    length_axis.spines["top"].set_visible(False)
    length_axis.spines["right"].set_color(COLORS["muted"])
    length_axis.tick_params(axis="y", colors=COLORS["gold"], width=0.7, length=2.5)
    length_line = length_axis.plot(
        epochs,
        val_output_length,
        "s-",
        color=COLORS["gold"],
        markersize=3.7,
        linewidth=1.2,
        label="Output length",
    )[0]
    length_axis.set_ylabel("Validation output length", color=COLORS["gold"])
    ax.set_title("Outcome refinement", loc="left", fontweight="bold")
    panel_label(ax, "c")

    fig.text(
        0.5,
        0.012,
        "Training diagnostics only: panels a-b use a 101-point moving mean; panel c shows full-epoch validation and the selected checkpoint.",
        ha="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.075, right=0.93, top=0.84, bottom=0.24, wspace=0.39)
    save_figure(fig, args.output_dir / "fig_training_dynamics")
    plt.close(fig)


if __name__ == "__main__":
    main()
