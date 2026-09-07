#!/usr/bin/env python3
"""Audit outcome-blind camera angles for the TRACE global-PCA path figure."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#6687B8",
    "orange": "#E6A314",
    "pink": "#E5A6C4",
    "text": "#27313B",
    "muted": "#66717E",
    "grid": "#D9DEE5",
    "white": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_paths(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    outcomes: dict[tuple[str, int], bool] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["stage"], int(row["view"]))
            grouped[key].append(
                (
                    int(row["latent_step"]),
                    np.asarray(
                        [
                            float(row["global_pc1"]),
                            float(row["global_pc2"]),
                            float(row["global_pc3"]),
                        ]
                    ),
                )
            )
            outcomes[key] = bool(int(row["correct"]))
    stage1 = np.stack(
        [np.stack([point for _, point in sorted(grouped[("Stage 1", view)])]) for view in range(1, 9)]
    )
    final = np.stack(
        [np.stack([point for _, point in sorted(grouped[("Final", view)])]) for view in range(1, 9)]
    )
    final_outcomes = np.asarray([outcomes[("Final", view)] for view in range(1, 9)])
    return stage1, final, final_outcomes


def camera_projection(points: np.ndarray, elev: float, azim: float) -> np.ndarray:
    elevation = math.radians(elev)
    azimuth = math.radians(azim)
    right = np.asarray([-math.sin(azimuth), math.cos(azimuth), 0.0])
    up = np.asarray(
        [
            -math.sin(elevation) * math.cos(azimuth),
            -math.sin(elevation) * math.sin(azimuth),
            math.cos(elevation),
        ]
    )
    return np.stack([points @ right, points @ up], axis=-1)


def pairwise_mean(points: np.ndarray) -> float:
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(distances[np.triu_indices(len(points), 1)].mean())


def camera_score(paths: np.ndarray, elev: float, azim: float) -> tuple[float, dict[str, float]]:
    minimum = paths.min(axis=(0, 1))
    maximum = paths.max(axis=(0, 1))
    spans = np.maximum(maximum - minimum, 1e-6)
    box_spans = np.clip(spans, spans.max() * 0.55, None)
    display_points = (paths - (minimum + maximum) / 2.0) * (box_spans / spans)
    projected = camera_projection(display_points, elev, azim)
    diagonal = max(float(np.linalg.norm(np.ptp(projected.reshape(-1, 2), axis=0))), 1e-6)

    # z0 is shared and z1 contains a large common excursion. Score the sustained
    # trajectory fan from z2 to z8 without using correctness labels.
    path_separation = np.mean(
        [pairwise_mean(projected[:, step]) for step in range(2, projected.shape[1])]
    ) / diagonal
    projected_length = np.linalg.norm(np.diff(projected, axis=1), axis=-1).sum(axis=1).mean()
    spatial_length = np.linalg.norm(np.diff(display_points, axis=1), axis=-1).sum(axis=1).mean()
    length_retention = float(projected_length / max(spatial_length, 1e-6))
    terminal_separation = pairwise_mean(projected[:, -1]) / diagonal
    score = 0.52 * path_separation + 0.38 * length_retention + 0.10 * terminal_separation
    return score, {
        "separation": float(path_separation),
        "length": length_retention,
        "terminal": float(terminal_separation),
    }


def draw_panel(
    ax: plt.Axes,
    stage1: np.ndarray,
    final: np.ndarray,
    outcomes: np.ndarray,
    elev: float,
    azim: float,
    title: str,
) -> None:
    all_points = np.concatenate([stage1, final], axis=0)
    score, pieces = camera_score(all_points, elev, azim)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    for path in stage1:
        ax.plot(*path.T, color=COLORS["blue"], linestyle=(0, (2.8, 2.0)), linewidth=0.65, alpha=0.24)
    for view, (path, correct) in enumerate(zip(final, outcomes), start=1):
        color = COLORS["pink"] if correct else COLORS["orange"]
        ax.plot(*path.T, color=color, linewidth=1.2, alpha=0.78)
        ax.scatter(*path[-1], color=color, marker="*" if correct else "X", s=28, edgecolor=COLORS["text"], linewidth=0.3)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(COLORS["grid"])
        axis._axinfo["grid"].update(color=COLORS["grid"], linewidth=0.3)
    spans = np.ptp(all_points, axis=(0, 1))
    ax.set_box_aspect(tuple(np.clip(spans, spans.max() * 0.55, None)), zoom=1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(
        f"{title}\nelev={elev:.0f}, azim={azim:.0f} | score={score:.3f}\n"
        f"sep={pieces['separation']:.3f}, length={pieces['length']:.3f}",
        fontsize=7.2,
        color=COLORS["text"],
        pad=1,
    )


def main() -> None:
    args = parse_args()
    stage1, final, outcomes = load_paths(args.source_csv)
    all_paths = np.concatenate([stage1, final], axis=0)
    candidates = []
    for elev in np.arange(8.0, 52.0, 2.0):
        for azim in np.arange(-180.0, 181.0, 3.0):
            score, _ = camera_score(all_paths, float(elev), float(azim))
            candidates.append((score, float(elev), float(azim)))
    candidates.sort(reverse=True)
    best = candidates[0]
    views = [
        (24.0, -58.0, "Previous"),
        (best[1], best[2], "Objective optimum"),
        (25.0, -20.0, "High separation"),
        (20.0, 60.0, "Oblique"),
        (30.0, 110.0, "High path retention"),
        (0.0, 0.0, "PC2-PC3 facing"),
    ]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "Times", "Liberation Serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(10.2, 6.2))
    for position, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 3, position, projection="3d")
        draw_panel(ax, stage1, final, outcomes, elev, azim, title)
    fig.suptitle(
        "Outcome-blind camera audit: the coordinates and scale are identical in all panels",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"],
        y=0.985,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.00, hspace=0.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"best_elev={best[1]:.0f} best_azim={best[2]:.0f} score={best[0]:.6f}")


if __name__ == "__main__":
    main()
