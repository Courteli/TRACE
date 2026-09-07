#!/usr/bin/env python3
"""Build the compact, evidence-led TRACE submission figure suite.

The suite deliberately separates four questions:

1. What does a real, unwarped latent path look like under one global PCA?
2. Does refinement improve accuracy, length, and fixed-view reliability?
3. Is trajectory structure above matched permutation controls at population scale?
4. Is outcome geometry genuinely separable within the same question?

All quantitative figures emit source CSV files. Unsupported causal and paraphrase
claims remain explicit pending rows in the claim-audit table; no placeholder
values are synthesized.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from trace_bridge_geometry_summary import signature_separation


COLORS = {
    "blue": "#6687B8",
    "green": "#69B17D",
    "orange": "#E6A314",
    "pink": "#E5A6C4",
    "text": "#27313B",
    "muted": "#66717E",
    "grid": "#D9DEE5",
    "light": "#F5F7F9",
    "white": "#FFFFFF",
    "stage1": "#6687B8",
    "final": "#E5A6C4",
}


def register_times_compatible_font() -> str:
    candidates = sorted(
        Path.home().glob(
            ".cache/Tectonic/bundles/data/**/TeXGyreTermesX-*.otf"
        )
    )
    family = "Times New Roman"
    for path in candidates:
        font_manager.fontManager.addfont(str(path))
        if path.name == "TeXGyreTermesX-Regular.otf":
            family = font_manager.FontProperties(fname=str(path)).get_name()
    return family


FIGURE_SERIF = register_times_compatible_font()


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        # Nimbus Roman and Liberation Serif are metric-compatible Times faces.
        "font.serif": [
            FIGURE_SERIF,
            "Times New Roman",
            "Nimbus Roman No9 L",
            "Times",
            "Liberation Serif",
        ],
        "font.size": 7.4,
        "axes.titlesize": 9.1,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-records", type=Path, required=True)
    parser.add_argument("--final-records", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--bootstrap-trials", type=int, default=5000)
    parser.add_argument("--geometry-permutations", type=int, default=128)
    parser.add_argument("--probe-epochs", type=int, default=650)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(base.with_suffix(".png"), dpi=420, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.035)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
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


def load_aligned_records(
    stage1_path: Path, final_path: Path, max_records: int
) -> tuple[list[dict], list[dict]]:
    stage1_raw = torch.load(stage1_path, map_location="cpu", weights_only=False)
    final_raw = torch.load(final_path, map_location="cpu", weights_only=False)
    stage1 = {int(record["idx"]): record for record in stage1_raw[:max_records]}
    final = {int(record["idx"]): record for record in final_raw[:max_records]}
    indices = sorted(set(stage1) & set(final))
    if len(indices) != max_records:
        raise ValueError(f"Expected {max_records} aligned records, found {len(indices)}")
    return [stage1[idx] for idx in indices], [final[idx] for idx in indices]


def style_axis(ax: plt.Axes, grid: str = "both") -> None:
    ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.48, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], width=0.7, length=2.4)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9.4,
        fontweight="bold",
        va="top",
        color=COLORS["text"],
    )


def bootstrap_mean_ci(
    values: np.ndarray, trials: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(trials, dtype=np.float64)
    chunk = 500
    for start in range(0, trials, chunk):
        end = min(start + chunk, trials)
        sample = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[sample].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def multiview_counts(records: list[dict]) -> np.ndarray:
    return np.asarray(
        [int(record["multiview_acc"].float().sum().item()) for record in records],
        dtype=np.int64,
    )


def cumulative_paths(record: dict) -> np.ndarray:
    residuals = record["multiview_implicit_residuals"].float().cpu().numpy()
    cumulative = np.cumsum(residuals, axis=1, dtype=np.float32)
    origin = np.zeros((cumulative.shape[0], 1, cumulative.shape[-1]), dtype=np.float32)
    return np.concatenate([origin, cumulative], axis=1)


def select_mixed_rescue_case(stage1: list[dict], final: list[dict]) -> tuple[int, int, int]:
    candidates = []
    for position, (stage1_record, final_record) in enumerate(zip(stage1, final)):
        stage1_correct = int(stage1_record["multiview_acc"].float().sum().item())
        final_correct = int(final_record["multiview_acc"].float().sum().item())
        if 0 < final_correct < len(final_record["multiview_acc"]) and final_correct > stage1_correct:
            candidates.append(
                (
                    final_correct - stage1_correct,
                    final_correct,
                    -int(final_record["idx"]),
                    position,
                    stage1_correct,
                    final_correct,
                )
            )
    if not candidates:
        raise ValueError("No mixed-outcome rescue case is available")
    _, _, _, position, stage1_correct, final_correct = max(candidates)
    return position, stage1_correct, final_correct


def fit_global_pca(
    stage1: list[dict], final: list[dict], excluded_position: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for position, records in enumerate((stage1, final)):
        del position  # stage loop label is intentionally unused
        for record_position, record in enumerate(records):
            if record_position == excluded_position:
                continue
            path = cumulative_paths(record)[:, 1:, :]
            arrays.append(torch.from_numpy(path.reshape(-1, path.shape[-1])))
    points = torch.cat(arrays, dim=0).float()
    torch.manual_seed(0)
    mean = points.mean(dim=0)
    _, singular_values, components = torch.pca_lowrank(
        points, q=3, center=True, niter=5
    )
    centered_ss = ((points - mean) ** 2).sum().item()
    explained = (singular_values**2 / max(centered_ss, 1e-12)).cpu().numpy()
    return mean.cpu().numpy(), components.cpu().numpy(), explained


def project_paths(paths: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    flat = paths.reshape(-1, paths.shape[-1])
    projected = (flat - mean) @ components
    return projected.reshape(*paths.shape[:-1], 3)


def padded_limits(values: np.ndarray, padding: float = 0.08) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1e-6)
    return low - padding * span, high + padding * span


def camera_projection(points: np.ndarray, elev: float, azim: float) -> np.ndarray:
    """Project points onto an orthographic camera plane for view selection."""
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


def pairwise_projected_distance(points: np.ndarray) -> float:
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(distances[np.triu_indices(len(points), 1)].mean())


def select_outcome_blind_camera(paths: np.ndarray) -> tuple[float, float, float, dict[str, float]]:
    """Choose a readable camera without using path correctness labels.

    The score rewards sustained screen-space separation, retention of the true
    polyline length, and terminal visibility. Coordinates are transformed only
    by the same axis box aspect used by the rendered 3D plot.
    """
    minimum = paths.min(axis=(0, 1))
    maximum = paths.max(axis=(0, 1))
    spans = np.maximum(maximum - minimum, 1e-6)
    box_spans = np.clip(spans, spans.max() * 0.55, None)
    display_points = (paths - (minimum + maximum) / 2.0) * (box_spans / spans)

    best: tuple[float, float, float, dict[str, float]] | None = None
    for elev in np.arange(8.0, 52.0, 2.0):
        for azim in np.arange(-180.0, 181.0, 3.0):
            projected = camera_projection(display_points, float(elev), float(azim))
            diagonal = max(
                float(np.linalg.norm(np.ptp(projected.reshape(-1, 2), axis=0))),
                1e-6,
            )
            # z0 is shared and z1 contains a large common excursion. Score the
            # sustained trajectory fan from z2 onward without outcome labels.
            separation = float(
                np.mean(
                    [
                        pairwise_projected_distance(projected[:, step])
                        for step in range(2, projected.shape[1])
                    ]
                )
                / diagonal
            )
            projected_length = float(
                np.linalg.norm(np.diff(projected, axis=1), axis=-1)
                .sum(axis=1)
                .mean()
            )
            spatial_length = float(
                np.linalg.norm(np.diff(display_points, axis=1), axis=-1)
                .sum(axis=1)
                .mean()
            )
            length_retention = projected_length / max(spatial_length, 1e-6)
            terminal = pairwise_projected_distance(projected[:, -1]) / diagonal
            score = 0.52 * separation + 0.38 * length_retention + 0.10 * terminal
            pieces = {
                "screen_separation": separation,
                "path_length_retention": length_retention,
                "terminal_separation": terminal,
            }
            candidate = (score, float(elev), float(azim), pieces)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return best[1], best[2], best[0], best[3]


def make_global_pca_figure(
    stage1: list[dict], final: list[dict], output_dir: Path
) -> dict:
    hero_position, stage1_correct, final_correct = select_mixed_rescue_case(stage1, final)
    hero_idx = int(final[hero_position]["idx"])
    mean, components, explained = fit_global_pca(stage1, final, hero_position)
    stage1_paths = cumulative_paths(stage1[hero_position])
    final_paths = cumulative_paths(final[hero_position])
    stage1_projection = project_paths(stage1_paths, mean, components)
    final_projection = project_paths(final_paths, mean, components)
    outcomes = final[hero_position]["multiview_acc"].float().cpu().numpy() > 0.5

    # PCA signs are arbitrary. Orient PC1 so the mean Final path reads left-to-right.
    orientation_sign = 1.0
    if outcomes.any():
        orientation_path = final_projection[outcomes].mean(axis=0)
        if orientation_path[-1, 0] < orientation_path[0, 0]:
            orientation_sign = -1.0
            stage1_projection[..., 0] *= -1.0
            final_projection[..., 0] *= -1.0

    all_points = np.concatenate([stage1_projection, final_projection], axis=0)
    camera_elev, camera_azim, camera_score, camera_components = (
        select_outcome_blind_camera(all_points)
    )

    fig = plt.figure(figsize=(7.2, 3.18))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=camera_elev, azim=camera_azim)

    for view in range(stage1_projection.shape[0]):
        path = stage1_projection[view]
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=COLORS["blue"],
            linestyle=(0, (3.0, 2.2)),
            linewidth=0.95,
            alpha=0.34,
            zorder=1,
        )

    for view, correct in enumerate(outcomes):
        path = final_projection[view]
        color = COLORS["pink"] if correct else COLORS["orange"]
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            linewidth=1.5 if correct else 1.7,
            alpha=0.76 if correct else 0.98,
            zorder=4,
        )
        ax.scatter(
            path[1:-1, 0],
            path[1:-1, 1],
            path[1:-1, 2],
            color=color,
            s=11,
            alpha=0.74 if correct else 0.98,
            edgecolor=COLORS["white"],
            linewidth=0.25,
            zorder=5,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            color=color,
            marker="*" if correct else "X",
            s=54 if correct else 46,
            edgecolor=COLORS["text"],
            linewidth=0.45,
            zorder=8,
        )

    correct_paths = final_projection[outcomes]
    if len(correct_paths):
        correct_mean = correct_paths.mean(axis=0)
        # A narrow centerline indicates direction without covering the six
        # observed correct paths, which remain the visual foreground.
        ax.plot(
            correct_mean[:, 0],
            correct_mean[:, 1],
            correct_mean[:, 2],
            color=COLORS["text"],
            linewidth=0.75,
            linestyle=(0, (1.2, 1.4)),
            alpha=0.52,
            zorder=6,
        )
        for step in (2, 5):
            movement = correct_mean[step + 1] - correct_mean[step]
            ax.quiver(
                correct_mean[step, 0],
                correct_mean[step, 1],
                correct_mean[step, 2],
                movement[0],
                movement[1],
                movement[2],
                color=COLORS["pink"],
                linewidth=1.05,
                arrow_length_ratio=0.28,
                normalize=False,
                zorder=7,
            )

    origin = final_projection[0, 0]
    ax.scatter(
        origin[0],
        origin[1],
        origin[2],
        marker="s",
        s=36,
        color=COLORS["text"],
        edgecolor=COLORS["white"],
        linewidth=0.6,
        zorder=10,
    )

    ax.set_xlim(*padded_limits(all_points[..., 0]))
    ax.set_ylim(*padded_limits(all_points[..., 1]))
    ax.set_zlim(*padded_limits(all_points[..., 2]))
    ranges = np.ptp(all_points, axis=(0, 1))
    ranges = np.clip(ranges, np.max(ranges) * 0.55, None)
    ax.set_box_aspect(tuple(ranges), zoom=1.30)
    ax.xaxis.set_major_locator(MaxNLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(2))
    ax.zaxis.set_major_locator(MaxNLocator(2))
    ax.set_xlabel("Global PC1", labelpad=-10, fontsize=7.0)
    ax.set_ylabel("Global PC2", labelpad=-4, fontsize=7.0)
    ax.set_zlabel("Global PC3", labelpad=-1, fontsize=7.0)
    ax.tick_params(colors=COLORS["muted"], labelsize=6.2, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(COLORS["grid"])
        axis._axinfo["grid"].update(color=COLORS["grid"], linewidth=0.35)

    fig.text(
        0.035,
        0.965,
        "Matched eight-view latent paths across refinement",
        ha="left",
        va="top",
        fontsize=9.6,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.035,
        0.895,
        f"Held-out q{hero_idx}: {stage1_correct}/8 to {final_correct}/8 correct paths",
        ha="left",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )
    fig.text(
        0.965,
        0.895,
        f"PC1-3: {100 * explained.sum():.1f}% variance",
        ha="right",
        va="top",
        fontsize=6.2,
        color=COLORS["muted"],
    )

    if len(correct_paths):
        coordinate_span = np.ptp(all_points, axis=(0, 1))
        coordinate_span = np.maximum(coordinate_span, 1e-6)
        label_offsets = {
            0: np.asarray([0.0, -0.025 * coordinate_span[1], 0.035 * coordinate_span[2]]),
            4: np.asarray([0.0, 0.0, 0.055 * coordinate_span[2]]),
            8: np.asarray([0.0, 0.030 * coordinate_span[1], 0.040 * coordinate_span[2]]),
        }
        for label, step in (("z0", 0), ("z4", 4), ("z8", 8)):
            point = correct_mean[step] + label_offsets[step]
            ax.text(
                point[0],
                point[1],
                point[2],
                f"  {label}",
                color=COLORS["text"] if step == 0 else COLORS["pink"],
                fontsize=6.4,
                fontweight="bold",
                zorder=11,
            )

    handles = [
        Line2D([0], [0], color=COLORS["blue"], linestyle="--", linewidth=1.8, label="Stage 1 paths"),
        Line2D([0], [0], color=COLORS["pink"], linewidth=2.2, label="Final correct"),
        Line2D([0], [0], color=COLORS["orange"], linewidth=2.2, label="Final wrong"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.845),
        ncol=3,
        handlelength=2.2,
        columnspacing=1.05,
    )
    ax.set_position([-0.025, 0.025, 1.05, 0.87])
    save_figure(fig, output_dir / "fig_global_pca_full_path")
    plt.close(fig)

    source_rows = []
    for stage, projection, stage_outcomes in (
        ("Stage 1", stage1_projection, stage1[hero_position]["multiview_acc"].float().cpu().numpy() > 0.5),
        ("Final", final_projection, outcomes),
    ):
        for view in range(projection.shape[0]):
            for step in range(projection.shape[1]):
                source_rows.append(
                    {
                        "question_idx": hero_idx,
                        "stage": stage,
                        "view": view + 1,
                        "correct": int(stage_outcomes[view]),
                        "latent_step": step,
                        "global_pc1": float(projection[view, step, 0]),
                        "global_pc2": float(projection[view, step, 1]),
                        "global_pc3": float(projection[view, step, 2]),
                    }
                )
    write_csv(output_dir / "source_data" / "global_pca_full_path.csv", source_rows)
    return {
        "hero_question_idx": hero_idx,
        "selection_rule": "largest correctness gain among Final mixed-outcome questions; ties favor more correct Final paths then lower index",
        "stage1_correct_paths": stage1_correct,
        "final_correct_paths": final_correct,
        "pca_fit_questions": len(stage1) - 1,
        "pca_joint_stages": True,
        "pca_uses_outcome_labels": False,
        "pca_explained_variance_ratio": [float(value) for value in explained],
        "pc1_orientation_sign": orientation_sign,
        "camera_selection": "outcome-blind grid search maximizing sustained screen-space path separation, projected path-length retention, and terminal visibility",
        "camera_elev": camera_elev,
        "camera_azim": camera_azim,
        "camera_score": camera_score,
        "camera_score_components": camera_components,
        "display_offsets": False,
        "per_path_scaling": False,
    }


def reliability_curves(
    stage1: list[dict],
    final: list[dict],
    trials: int,
    seed: int,
) -> list[dict]:
    stage1_counts = multiview_counts(stage1)
    final_counts = multiview_counts(final)
    rng = np.random.default_rng(seed)
    rows = []
    n = len(stage1_counts)
    bootstrap_indices = rng.integers(0, n, size=(trials, n))
    for threshold in range(1, 9):
        stage1_values = (stage1_counts >= threshold).astype(np.float64) * 100.0
        final_values = (final_counts >= threshold).astype(np.float64) * 100.0
        stage1_boot = stage1_values[bootstrap_indices].mean(axis=1)
        final_boot = final_values[bootstrap_indices].mean(axis=1)
        delta_boot = final_boot - stage1_boot
        rows.append(
            {
                "minimum_correct_paths": threshold,
                "stage1_percent": float(stage1_values.mean()),
                "stage1_ci95_low": float(np.quantile(stage1_boot, 0.025)),
                "stage1_ci95_high": float(np.quantile(stage1_boot, 0.975)),
                "final_percent": float(final_values.mean()),
                "final_ci95_low": float(np.quantile(final_boot, 0.025)),
                "final_ci95_high": float(np.quantile(final_boot, 0.975)),
                "delta_pp": float((final_values - stage1_values).mean()),
                "delta_ci95_low": float(np.quantile(delta_boot, 0.025)),
                "delta_ci95_high": float(np.quantile(delta_boot, 0.975)),
            }
        )
    return rows


def make_refinement_figure(
    stage1: list[dict],
    final: list[dict],
    stagewise_csv: Path,
    output_dir: Path,
    bootstrap_trials: int,
    seed: int,
) -> list[dict]:
    benchmark_rows = read_csv(stagewise_csv)
    curve_rows = reliability_curves(stage1, final, bootstrap_trials, seed)
    stage1_counts = multiview_counts(stage1).astype(np.int64)
    final_counts = multiview_counts(final).astype(np.int64)
    transition = np.zeros((9, 9), dtype=np.int64)
    for stage1_count, final_count in zip(stage1_counts, final_counts):
        transition[stage1_count, final_count] += 1

    fig = plt.figure(figsize=(7.2, 2.92))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.23, 0.92],
        left=0.08,
        right=0.985,
        top=0.78,
        bottom=0.21,
        wspace=0.30,
    )

    forest_grid = grid[0, 0].subgridspec(1, 2, wspace=0.20)
    accuracy_ax = fig.add_subplot(forest_grid[0, 0])
    length_ax = fig.add_subplot(forest_grid[0, 1], sharey=accuracy_ax)
    dataset_colors = {
        "GSM8K": COLORS["blue"],
        "GSMHard": COLORS["green"],
        "SVAMP": COLORS["orange"],
        "MultiArith": COLORS["pink"],
    }
    dataset_order = ["GSM8K", "GSMHard", "SVAMP", "MultiArith"]
    benchmark_by_dataset = {row["dataset"]: row for row in benchmark_rows}
    y_positions = np.arange(len(dataset_order))[::-1]
    for axis in (accuracy_ax, length_ax):
        style_axis(axis, grid="x")
        for stripe_position in y_positions[::2]:
            axis.axhspan(
                stripe_position - 0.46,
                stripe_position + 0.46,
                color=COLORS["light"],
                alpha=0.88,
                linewidth=0,
                zorder=0,
            )
        axis.axvline(0, color=COLORS["muted"], linewidth=0.8, linestyle="--")
        axis.set_ylim(-0.55, 3.55)

    for position, dataset in zip(y_positions, dataset_order):
        row = benchmark_by_dataset[dataset]
        color = dataset_colors[dataset]
        accuracy = float(row["accuracy_gain_pp"])
        accuracy_low = float(row["accuracy_gain_ci95_low"])
        accuracy_high = float(row["accuracy_gain_ci95_high"])
        accuracy_ax.errorbar(
            accuracy,
            position,
            xerr=[[accuracy - accuracy_low], [accuracy_high - accuracy]],
            fmt="o",
            ms=5.6,
            color=color,
            ecolor=color,
            capsize=2.0,
            elinewidth=1.15,
            markeredgecolor=COLORS["text"],
            markeredgewidth=0.4,
            zorder=4,
        )
        accuracy_ax.text(
            accuracy_high + 0.12,
            position,
            f"+{accuracy:.2f}",
            va="center",
            fontsize=6.0,
            color=color,
            fontweight="bold",
        )

        saved = float(row["length_saved"])
        saved_low = float(row["length_saved_ci95_low"])
        saved_high = float(row["length_saved_ci95_high"])
        length_ax.errorbar(
            saved,
            position,
            xerr=[[saved - saved_low], [saved_high - saved]],
            fmt="o",
            ms=5.6,
            color=color,
            ecolor=color,
            capsize=2.0,
            elinewidth=1.15,
            markeredgecolor=COLORS["text"],
            markeredgewidth=0.4,
            zorder=4,
        )
        length_ax.text(
            saved_high + 0.10,
            position,
            f"{saved:.2f}",
            va="center",
            fontsize=6.0,
            color=color,
            fontweight="bold",
        )

    accuracy_ax.set_yticks(y_positions, dataset_order)
    accuracy_ax.tick_params(axis="y", labelcolor=COLORS["text"], labelsize=6.7)
    length_ax.tick_params(axis="y", left=False, labelleft=False)
    accuracy_ax.set_xlim(-0.22, 6.72)
    length_ax.set_xlim(-0.12, 4.92)
    accuracy_ax.set_xlabel("Accuracy gain (percentage points)")
    length_ax.set_xlabel("Total reasoning length saved")
    accuracy_ax.set_title("Accuracy", loc="left", fontsize=7.5, fontweight="bold", pad=4)
    length_ax.set_title("Shorter #L", loc="left", fontsize=7.5, fontweight="bold", pad=4)
    panel_label(accuracy_ax, "a", x=-0.31, y=1.31)
    fig.text(
        0.08,
        0.925,
        "Every benchmark gains accuracy while using fewer reasoning tokens",
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["text"],
    )

    ax = fig.add_subplot(grid[0, 1])
    improved = int((final_counts > stage1_counts).sum())
    unchanged = int((final_counts == stage1_counts).sum())
    regressed = int((final_counts < stage1_counts).sum())
    max_count = int(transition.max())
    relation_colors = {
        "improved": COLORS["pink"],
        "unchanged": COLORS["blue"],
        "regressed": COLORS["orange"],
    }
    for stage1_count in range(9):
        for final_count in range(9):
            count = int(transition[stage1_count, final_count])
            relation = (
                "improved"
                if final_count > stage1_count
                else "regressed"
                if final_count < stage1_count
                else "unchanged"
            )
            alpha = 0.025 if count == 0 else 0.15 + 0.78 * math.sqrt(count / max_count)
            ax.add_patch(
                Rectangle(
                    (final_count - 0.5, stage1_count - 0.5),
                    1,
                    1,
                    facecolor=relation_colors[relation],
                    edgecolor=COLORS["white"],
                    linewidth=0.7,
                    alpha=alpha,
                )
            )
            if count:
                ax.text(
                    final_count,
                    stage1_count,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=5.6 if count < 10 else 6.0,
                    color=COLORS["text"],
                    fontweight="bold" if count >= 10 else "normal",
                )
    ax.plot([-0.5, 8.5], [-0.5, 8.5], color=COLORS["muted"], linewidth=0.7, linestyle="--")
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.tick_params(colors=COLORS["muted"], width=0.6, length=2.0, labelsize=6.1)
    ax.set_xlabel("Final: correct paths among 8")
    ax.set_ylabel("Stage 1: correct paths among 8")
    ax.set_title("Reliability shifts toward 8/8 correct", loc="left", fontsize=8.4, fontweight="bold", pad=24)
    panel_label(ax, "b", x=-0.18, y=1.31)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=COLORS["pink"], markeredgecolor="none", label=f"{improved} improve"),
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=COLORS["blue"], markeredgecolor="none", label=f"{unchanged} unchanged"),
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=COLORS["orange"], markeredgecolor="none", label=f"{regressed} regress"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
        fontsize=5.6,
        handletextpad=0.25,
        columnspacing=0.62,
    )
    majority_stage1 = float((stage1_counts >= 5).mean() * 100.0)
    majority_final = float((final_counts >= 5).mean() * 100.0)
    all_stage1 = float((stage1_counts == 8).mean() * 100.0)
    all_final = float((final_counts == 8).mean() * 100.0)
    ax.text(
        0.5,
        -0.18,
        f"mean {stage1_counts.mean():.2f} to {final_counts.mean():.2f}   |   "
        f"at least 5: {majority_stage1:.0f}% to {majority_final:.0f}%   |   "
        f"8/8: {all_stage1:.0f}% to {all_final:.1f}%",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
        color=COLORS["text"],
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.016,
        "Forest intervals are paired question-bootstrap 95% CIs; matrix cells are counts over the same 200 matched questions.",
        ha="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    save_figure(fig, output_dir / "fig_outcome_refinement_compact")
    plt.close(fig)

    transition_rows = []
    for stage1_count in range(9):
        for final_count in range(9):
            transition_rows.append(
                {
                    "stage1_correct_paths": stage1_count,
                    "final_correct_paths": final_count,
                    "question_count": int(transition[stage1_count, final_count]),
                    "relation": (
                        "improved"
                        if final_count > stage1_count
                        else "regressed"
                        if final_count < stage1_count
                        else "unchanged"
                    ),
                }
            )
    write_csv(output_dir / "source_data" / "reliability_transition_matrix.csv", transition_rows)
    write_csv(output_dir / "source_data" / "reliability_survival_curve.csv", curve_rows)
    return curve_rows


def make_population_structure_figure(
    trajectory_csv: Path,
    identity_csv: Path,
    output_dir: Path,
) -> dict:
    trajectory_rows = read_csv(trajectory_csv)
    identity_rows = read_csv(identity_csv)
    per_question_rows = read_csv(identity_csv.with_name("path_identity_per_question.csv"))
    by_metric: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in trajectory_rows:
        by_metric[row["metric"]][row["stage"]] = row

    metric_order = ["Step correspondence", "Path direction", "Position order"]
    metric_labels = ["Step correspondence", "Question-specific direction", "Ordered progress"]
    fig = plt.figure(figsize=(7.2, 2.62))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.20, 0.88],
        left=0.105,
        right=0.985,
        top=0.80,
        bottom=0.22,
        wspace=0.34,
    )

    ax = fig.add_subplot(grid[0, 0])
    style_axis(ax, grid="x")
    y = np.arange(len(metric_order))[::-1]
    for stripe_position in y[::2]:
        ax.axhspan(
            stripe_position - 0.43,
            stripe_position + 0.43,
            color=COLORS["light"],
            alpha=0.9,
            linewidth=0,
        )
    source_rows = []
    for stage, offset, color, label in (
        ("TRACE Stage 1", 0.16, COLORS["green"], "Stage 1"),
        ("TRACE Final", -0.16, COLORS["pink"], "Final"),
    ):
        for position, metric in zip(y, metric_order):
            row = by_metric[metric][stage]
            null = float(row["permuted_null"])
            observed = float(row["observed"])
            gap = float(row["observed_minus_null"])
            gap_low = float(row["difference_ci95_low"])
            gap_high = float(row["difference_ci95_high"])
            gap_low_on_score = null + gap_low
            gap_high_on_score = null + gap_high
            y_position = position + offset
            ax.plot(
                [null, observed],
                [y_position, y_position],
                color=color,
                linewidth=2.1,
                solid_capstyle="round",
                zorder=2,
            )
            ax.scatter(
                null,
                y_position,
                marker="s",
                s=24,
                facecolor=COLORS["white"],
                edgecolor=COLORS["orange"],
                linewidth=1.0,
                zorder=4,
            )
            ax.errorbar(
                observed,
                y_position,
                xerr=[
                    [observed - gap_low_on_score],
                    [gap_high_on_score - observed],
                ],
                fmt="o",
                ms=5.2,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=1.9,
                markeredgecolor=COLORS["text"],
                markeredgewidth=0.35,
                zorder=5,
            )
            ax.text(
                min(gap_high_on_score + 0.010, 0.985),
                y_position,
                f"+{gap:.3f}",
                va="center",
                fontsize=5.9,
                color=color,
                fontweight="bold",
            )
            source_rows.append(
                {
                    "stage": label,
                    "metric": metric,
                    "observed": observed,
                    "permuted_null": null,
                    "observed_minus_null": gap,
                    "gap_ci95_low": gap_low,
                    "gap_ci95_high": gap_high,
                    "permutation_p": float(row["permutation_p"]),
                }
            )
    ax.set_yticks(y, metric_labels)
    ax.set_xlim(0.45, 1.015)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Matched null  to  observed score")
    ax.set_title(
        "Every trajectory test beats its matched null",
        loc="left",
        fontweight="bold",
        pad=11,
    )
    panel_label(ax, "a", x=-0.20, y=1.18)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor=COLORS["white"],
                markeredgecolor=COLORS["orange"],
                label="matched null",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                color=COLORS["green"],
                markerfacecolor=COLORS["green"],
                label="Stage 1",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                color=COLORS["pink"],
                markerfacecolor=COLORS["pink"],
                label="Final",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.08),
        ncol=3,
        fontsize=5.8,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    identity_map = {row["metric"]: row for row in identity_rows}
    view_row = identity_map["View identity"]
    question_row = identity_map["Question identity"]
    diversity_row = identity_map["Cross-view diversity"]
    same_question_same_view = float(view_row["observed_or_final"])
    same_question_other_view = float(view_row["control_or_stage1"])
    other_question_same_view = float(question_row["control_or_stage1"])
    other_question_other_view = float(
        np.mean(
            [
                float(row["other_question_other_view_similarity"])
                for row in per_question_rows
            ]
        )
    )
    identity_matrix = np.asarray(
        [
            [same_question_other_view, same_question_same_view],
            [other_question_other_view, other_question_same_view],
        ]
    )

    identity_grid = grid[0, 1].subgridspec(
        2, 1, height_ratios=[1.0, 0.26], hspace=0.52
    )
    matrix_ax = fig.add_subplot(identity_grid[0, 0])
    identity_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "trace_identity",
        [COLORS["light"], "#D9E4F1", "#E9C5D8", COLORS["pink"]],
    )
    matrix_ax.imshow(
        identity_matrix,
        vmin=0.78,
        vmax=0.97,
        cmap=identity_cmap,
        aspect="auto",
    )
    for row_position in range(2):
        for column_position in range(2):
            value = identity_matrix[row_position, column_position]
            matrix_ax.text(
                column_position,
                row_position,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=COLORS["text"],
                fontweight="bold",
            )
    matrix_ax.add_patch(
        Rectangle(
            (0.5, -0.5),
            1,
            1,
            fill=False,
            edgecolor=COLORS["orange"],
            linewidth=1.6,
        )
    )
    matrix_ax.text(
        1,
        -0.34,
        "matched",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=COLORS["orange"],
        fontweight="bold",
    )
    matrix_ax.set_xticks([0, 1], ["Different view", "Same view"])
    matrix_ax.set_yticks([0, 1], ["Same question", "Different question"])
    matrix_ax.tick_params(length=0, labelsize=6.3, colors=COLORS["text"])
    matrix_ax.set_title(
        "Identity follows both question and view",
        loc="left",
        fontweight="bold",
        pad=23,
    )
    matrix_ax.text(
        0.5,
        1.04,
        f"question gap +{float(question_row['gap_or_change']):.3f}   |   "
        f"view gap +{float(view_row['gap_or_change']):.3f}",
        transform=matrix_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.9,
        color=COLORS["text"],
        fontweight="bold",
    )
    panel_label(matrix_ax, "b", x=-0.23, y=1.22)
    for spine in matrix_ax.spines.values():
        spine.set_visible(False)

    diversity_ax = fig.add_subplot(identity_grid[1, 0])
    stage1_diversity = float(diversity_row["control_or_stage1"])
    final_diversity = float(diversity_row["observed_or_final"])
    diversity_change = float(diversity_row["gap_or_change"])
    diversity_positive = float(
        np.mean([float(row["diversity_change"]) > 0 for row in per_question_rows])
        * 100.0
    )
    diversity_ax.plot(
        [stage1_diversity, final_diversity],
        [0, 0],
        color=COLORS["muted"],
        linewidth=1.2,
        zorder=1,
    )
    diversity_ax.scatter(
        stage1_diversity,
        0,
        s=30,
        color=COLORS["green"],
        edgecolor=COLORS["text"],
        linewidth=0.35,
        zorder=3,
    )
    diversity_ax.scatter(
        final_diversity,
        0,
        s=34,
        color=COLORS["pink"],
        edgecolor=COLORS["text"],
        linewidth=0.35,
        zorder=3,
    )
    diversity_ax.annotate(
        "",
        xy=(final_diversity, 0),
        xytext=(stage1_diversity, 0),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["pink"],
            "linewidth": 1.0,
        },
    )
    diversity_ax.set_xlim(0.0715, 0.0825)
    diversity_ax.set_ylim(-0.6, 0.6)
    diversity_ax.set_yticks([])
    diversity_ax.set_xticks(
        [stage1_diversity, final_diversity],
        [
            f"Stage 1\n{stage1_diversity:.3f}",
            f"Final\n{final_diversity:.3f}",
        ],
    )
    diversity_ax.tick_params(
        axis="x", length=0, labelsize=5.7, colors=COLORS["text"], pad=1
    )
    diversity_ax.set_title(
        f"Cross-view diversity: +{diversity_change:.3f}  |  "
        f"{diversity_positive:.0f}% of questions increase",
        loc="left",
        fontsize=6.1,
        color=COLORS["text"],
        pad=1,
        fontweight="bold",
    )
    for spine in diversity_ax.spines.values():
        spine.set_visible(False)

    identity_source = []
    for metric in ("Question identity", "View identity", "Cross-view diversity"):
        row = identity_map[metric]
        identity_source.append(
            {
                "metric": metric,
                "observed_or_final": float(row["observed_or_final"]),
                "control_or_stage1": float(row["control_or_stage1"]),
                "gap_or_change": float(row["gap_or_change"]),
                "ci95_low": float(row["gap_ci95_low"]),
                "ci95_high": float(row["gap_ci95_high"]),
                "permutation_p": row.get("permutation_p", ""),
            }
        )

    fig.text(
        0.5,
        0.04,
        "200 matched questions; paired gap intervals use question bootstrap; all six trajectory gaps have matched-permutation p < .001.",
        ha="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    save_figure(fig, output_dir / "fig_trajectory_population_compact")
    plt.close(fig)
    write_csv(output_dir / "source_data" / "trajectory_population_compact.csv", source_rows)
    write_csv(output_dir / "source_data" / "identity_population_compact.csv", identity_source)
    identity_matrix_rows = [
        {
            "question_relation": "same",
            "view_relation": "different",
            "mean_cosine_similarity": same_question_other_view,
        },
        {
            "question_relation": "same",
            "view_relation": "same",
            "mean_cosine_similarity": same_question_same_view,
        },
        {
            "question_relation": "different",
            "view_relation": "different",
            "mean_cosine_similarity": other_question_other_view,
        },
        {
            "question_relation": "different",
            "view_relation": "same",
            "mean_cosine_similarity": other_question_same_view,
        },
    ]
    write_csv(
        output_dir / "source_data" / "identity_control_matrix.csv",
        identity_matrix_rows,
    )
    return {"trajectory": source_rows, "identity": identity_source}


def path_geometry_features(records: list[dict], mixed_only: bool) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    features = []
    outcomes = []
    groups = []
    for question_position, record in enumerate(records):
        labels = record["multiview_acc"].float().cpu()
        if mixed_only and labels.min().item() == labels.max().item():
            continue
        residuals = record["multiview_implicit_residuals"].float().cpu()
        unit = residuals / residuals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gram = unit @ unit.transpose(-1, -2)
        norms = residuals.norm(dim=-1)
        log_norm_ratio = torch.log(norms.clamp_min(1e-8)) - torch.log(
            norms.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        net = residuals.sum(dim=1)
        net_unit = net / net.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        goal_alignment = (unit * net_unit[:, None, :]).sum(dim=-1)
        efficiency = net.norm(dim=-1) / norms.sum(dim=-1).clamp_min(1e-8)
        feature = torch.cat(
            [gram.flatten(start_dim=1), log_norm_ratio, goal_alignment, efficiency[:, None]],
            dim=1,
        )
        features.append(feature)
        outcomes.append(labels)
        groups.extend([question_position] * len(labels))
    return torch.cat(features), torch.cat(outcomes), np.asarray(groups, dtype=np.int64)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (np.abs(comparisons) <= 1e-12).mean())


def grouped_probe_predictions(
    features: torch.Tensor,
    outcomes: torch.Tensor,
    groups: np.ndarray,
    epochs: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    folds = np.array_split(unique_groups, 5)
    predictions = np.zeros(len(outcomes), dtype=np.float64)
    for fold_idx, test_groups in enumerate(folds):
        test_mask_np = np.isin(groups, test_groups)
        train_mask_np = ~test_mask_np
        train_mask = torch.from_numpy(train_mask_np)
        test_mask = torch.from_numpy(test_mask_np)
        mean = features[train_mask].mean(dim=0)
        std = features[train_mask].std(dim=0).clamp_min(1e-4)
        train_x = (features[train_mask] - mean) / std
        test_x = (features[test_mask] - mean) / std
        train_y = outcomes[train_mask]
        torch.manual_seed(seed + fold_idx)
        weight = torch.zeros(features.shape[1], requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=0.03, weight_decay=1e-3)
        positive_weight = (train_y == 0).sum() / (train_y == 1).sum().clamp_min(1)
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = train_x @ weight + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, train_y, pos_weight=positive_weight
            )
            loss.backward()
            optimizer.step()
        predictions[test_mask_np] = (test_x @ weight + bias).detach().cpu().numpy()
    return predictions


def grouped_auc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    trials: int,
    seed: int,
) -> tuple[float, float, float]:
    estimate = binary_auc(labels, scores)
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    bootstrap = []
    for _ in range(trials):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        value = binary_auc(labels[indices], scores[indices])
        if np.isfinite(value):
            bootstrap.append(value)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return estimate, float(low), float(high)


def outcome_probe_summary(
    stage1: list[dict],
    final: list[dict],
    epochs: int,
    trials: int,
    seed: int,
) -> list[dict]:
    rows = []
    for stage_idx, (stage, records) in enumerate((("Stage 1", stage1), ("Final", final))):
        for mixed_only in (False, True):
            features, labels_tensor, groups = path_geometry_features(records, mixed_only)
            scores = grouped_probe_predictions(
                features,
                labels_tensor,
                groups,
                epochs=epochs,
                seed=seed + 100 * stage_idx + int(mixed_only),
            )
            labels = labels_tensor.numpy().astype(np.int64)
            estimate, low, high = grouped_auc_ci(
                labels,
                scores,
                groups,
                trials=trials,
                seed=seed + 500 + 100 * stage_idx + int(mixed_only),
            )
            rows.append(
                {
                    "stage": stage,
                    "population": "Mixed questions only" if mixed_only else "All questions",
                    "n_questions": int(len(np.unique(groups))),
                    "n_paths": int(len(labels)),
                    "n_correct_paths": int(labels.sum()),
                    "heldout_auc": estimate,
                    "auc_ci95_low": low,
                    "auc_ci95_high": high,
                    "split": "five-fold question-held-out",
                    "signature": "8x8 step-direction Gram + relative step norms + goal alignment + path efficiency",
                }
            )
    return rows


def make_outcome_audit_figure(
    stage1: list[dict],
    final: list[dict],
    output_dir: Path,
    probe_epochs: int,
    bootstrap_trials: int,
    geometry_permutations: int,
    seed: int,
) -> tuple[list[dict], dict]:
    probe_rows = outcome_probe_summary(
        stage1,
        final,
        epochs=probe_epochs,
        trials=bootstrap_trials,
        seed=seed,
    )
    stage1_geometry = signature_separation(
        stage1,
        permutation_null_trials=geometry_permutations,
        signature_representation="raw",
    )
    final_geometry = signature_separation(
        final,
        permutation_null_trials=geometry_permutations,
        signature_representation="raw",
    )

    fig = plt.figure(figsize=(7.2, 2.55))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.22, 0.88],
        left=0.105,
        right=0.985,
        top=0.80,
        bottom=0.27,
        wspace=0.42,
    )
    ax = fig.add_subplot(grid[0, 0])
    style_axis(ax, grid="x")
    populations = ["All questions", "Mixed questions only"]
    y = np.arange(2)[::-1]
    for stage, offset, color in (
        ("Stage 1", 0.13, COLORS["green"]),
        ("Final", -0.13, COLORS["pink"]),
    ):
        for position, population in zip(y, populations):
            row = next(
                item for item in probe_rows if item["stage"] == stage and item["population"] == population
            )
            value = float(row["heldout_auc"])
            low = float(row["auc_ci95_low"])
            high = float(row["auc_ci95_high"])
            ax.errorbar(
                value,
                position + offset,
                xerr=[[value - low], [high - value]],
                fmt="o",
                ms=5.4,
                color=color,
                ecolor=color,
                capsize=2.1,
                elinewidth=1.05,
                markeredgecolor=COLORS["text"],
                markeredgewidth=0.4,
                zorder=4,
            )
            ax.text(high + 0.008, position + offset, f"{value:.3f}", color=color, va="center", fontsize=6.2, fontweight="bold")
    ax.axvline(0.5, color=COLORS["muted"], linestyle="--", linewidth=0.85)
    ax.set_yticks(y, ["All questions", "Same-question\ncorrect + wrong"])
    ax.set_xlim(0.38, 0.82)
    ax.set_xlabel("Question-held-out full-path probe AUROC")
    ax.set_title("Outcome signal must survive the within-question control", loc="left", fontweight="bold")
    panel_label(ax, "a", x=-0.19)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markeredgecolor=COLORS["text"], label="Stage 1"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["pink"], markeredgecolor=COLORS["text"], label="Final"),
        ],
        loc="lower right",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    ax = fig.add_subplot(grid[0, 1])
    style_axis(ax, grid="x")
    geometry_rows = []
    for position, (stage, geometry, color) in enumerate(
        (("Stage 1", stage1_geometry, COLORS["green"]), ("Final", final_geometry, COLORS["pink"]))
    ):
        metric = geometry["metrics"]["wrong_rejection_auc_excess_over_null"]
        value = float(metric["mean"])
        low = float(metric["bootstrap_ci95_low"])
        high = float(metric["bootstrap_ci95_high"])
        y_position = 1 - position
        ax.errorbar(
            value,
            y_position,
            xerr=[[value - low], [high - value]],
            fmt="o",
            ms=5.7,
            color=color,
            ecolor=color,
            capsize=2.2,
            elinewidth=1.05,
            markeredgecolor=COLORS["text"],
            markeredgewidth=0.4,
            zorder=4,
        )
        ax.text(high + 0.009, y_position, f"{value:+.3f}", color=color, va="center", fontsize=6.3, fontweight="bold")
        geometry_rows.append(
            {
                "stage": stage,
                "mixed_questions": int(geometry["mixed_count"]),
                "wrong_rejection_auc_excess_over_null": value,
                "ci95_low": low,
                "ci95_high": high,
                "per_question_permutation_trials": geometry_permutations,
            }
        )
    ax.axvline(0, color=COLORS["muted"], linestyle="--", linewidth=0.85)
    ax.set_yticks([1, 0], ["Stage 1", "Final"])
    ax.set_xlim(-0.13, 0.13)
    ax.set_xlabel("Wrong-rejection AUROC above label-permutation null")
    ax.set_title("Mode rejection is not yet established", loc="left", fontweight="bold")
    panel_label(ax, "b", x=-0.19)

    fig.text(
        0.5,
        0.055,
        "Internal claim audit, not a positive main-paper result. All splits are by question; intervals use question bootstrap.",
        ha="center",
        fontsize=6.0,
        color=COLORS["muted"],
    )
    save_figure(fig, output_dir / "fig_outcome_geometry_audit")
    plt.close(fig)
    write_csv(output_dir / "source_data" / "outcome_probe_audit.csv", probe_rows)
    write_csv(output_dir / "source_data" / "outcome_mode_rejection_audit.csv", geometry_rows)
    return probe_rows, {"stage1": stage1_geometry, "final": final_geometry, "rows": geometry_rows}


def p_string(value: float) -> str:
    if value < 0.001:
        return "$<.001$"
    return f"{value:.3f}"


def make_evidence_tables(
    stagewise_csv: Path,
    trajectory_csv: Path,
    identity_csv: Path,
    reliability_rows: list[dict],
    probe_rows: list[dict],
    geometry_audit: dict,
    output_dir: Path,
) -> dict:
    stagewise = {row["dataset"]: row for row in read_csv(stagewise_csv)}
    trajectory = {
        (row["stage"], row["metric"]): row for row in read_csv(trajectory_csv)
    }
    identity = {row["metric"]: row for row in read_csv(identity_csv)}
    gsm8k = stagewise["GSM8K"]
    all_correct = next(row for row in reliability_rows if row["minimum_correct_paths"] == 8)
    final_step = trajectory[("TRACE Final", "Step correspondence")]
    final_direction = trajectory[("TRACE Final", "Path direction")]
    final_order = trajectory[("TRACE Final", "Position order")]
    question_identity = identity["Question identity"]
    view_identity = identity["View identity"]
    mixed_final_probe = next(
        row
        for row in probe_rows
        if row["stage"] == "Final" and row["population"] == "Mixed questions only"
    )
    rejection = geometry_audit["final"]["metrics"]["wrong_rejection_auc_excess_over_null"]

    supported_rows = [
        {
            "claim": "Higher task utility",
            "measure": "GSM8K accuracy gain",
            "estimate": f'{float(gsm8k["accuracy_gain_pp"]):+.2f} pp [{float(gsm8k["accuracy_gain_ci95_low"]):+.2f}, {float(gsm8k["accuracy_gain_ci95_high"]):+.2f}]',
            "test": f'McNemar {p_string(float(gsm8k["mcnemar_exact_p"]))}',
        },
        {
            "claim": "Shorter reasoning",
            "measure": "GSM8K total length saved",
            "estimate": f'{float(gsm8k["length_saved"]):+.2f} [{float(gsm8k["length_saved_ci95_low"]):+.2f}, {float(gsm8k["length_saved_ci95_high"]):+.2f}]',
            "test": "paired question bootstrap",
        },
        {
            "claim": "More repeatable outcomes",
            "measure": "All 8/8 paths correct",
            "estimate": f'{float(all_correct["delta_pp"]):+.1f} pp [{float(all_correct["delta_ci95_low"]):+.1f}, {float(all_correct["delta_ci95_high"]):+.1f}]',
            "test": "paired question bootstrap",
        },
        {
            "claim": "Step-specific path",
            "measure": "Final step correspondence above null",
            "estimate": f'{float(final_step["observed_minus_null"]):+.3f} [{float(final_step["difference_ci95_low"]):+.3f}, {float(final_step["difference_ci95_high"]):+.3f}]',
            "test": f'permutation {p_string(float(final_step["permutation_p"]))}',
        },
        {
            "claim": "Question-specific direction",
            "measure": "Final direction above null",
            "estimate": f'{float(final_direction["observed_minus_null"]):+.3f} [{float(final_direction["difference_ci95_low"]):+.3f}, {float(final_direction["difference_ci95_high"]):+.3f}]',
            "test": f'permutation {p_string(float(final_direction["permutation_p"]))}',
        },
        {
            "claim": "Ordered progress",
            "measure": "Final position order above null",
            "estimate": f'{float(final_order["observed_minus_null"]):+.3f} [{float(final_order["difference_ci95_low"]):+.3f}, {float(final_order["difference_ci95_high"]):+.3f}]',
            "test": f'permutation {p_string(float(final_order["permutation_p"]))}',
        },
        {
            "claim": "Identity without collapse",
            "measure": "Question / view identity gaps",
            "estimate": f'{float(question_identity["gap_or_change"]):+.3f} / {float(view_identity["gap_or_change"]):+.3f}',
            "test": "both permutation $<.001$",
        },
    ]

    audit_rows = supported_rows + [
        {
            "claim": "Paraphrase-stable complete paths",
            "measure": "Global held-out invariance ratio",
            "estimate": "TBD: paraphrase paths not collected",
            "test": "Pending",
        },
        {
            "claim": "Full-path causal mediation",
            "measure": "Question-blocked prefix / transition interventions",
            "estimate": "TBD: valid bottleneck audit not completed",
            "test": "Pending",
        },
        {
            "claim": "Universal correct--wrong separability",
            "measure": "Mixed-question held-out path probe AUROC",
            "estimate": f'{float(mixed_final_probe["heldout_auc"]):.3f} [{float(mixed_final_probe["auc_ci95_low"]):.3f}, {float(mixed_final_probe["auc_ci95_high"]):.3f}]',
            "test": "Not established",
        },
        {
            "claim": "Wrong-path mode rejection",
            "measure": "AUROC excess over permutation null",
            "estimate": f'{float(rejection["mean"]):+.3f} [{float(rejection["bootstrap_ci95_low"]):+.3f}, {float(rejection["bootstrap_ci95_high"]):+.3f}]',
            "test": "Not established",
        },
    ]

    write_csv(output_dir / "source_data" / "trace_supported_evidence.csv", supported_rows)
    write_csv(output_dir / "source_data" / "trace_claim_audit.csv", audit_rows)

    def markdown(rows: list[dict]) -> str:
        lines = [
            "| Claim | Measure | Estimate [95% CI] | Test / status |",
            "|---|---|---:|---|",
        ]
        for row in rows:
            lines.append(
                f'| {row["claim"]} | {row["measure"]} | {row["estimate"]} | {row["test"]} |'
            )
        return "\n".join(lines) + "\n"

    (output_dir / "table_trace_supported_evidence.md").write_text(
        markdown(supported_rows), encoding="utf-8"
    )
    (output_dir / "table_trace_claim_audit.md").write_text(
        markdown(audit_rows), encoding="utf-8"
    )

    def escape_latex(value: str) -> str:
        return value.replace("%", r"\%").replace("--", r"--")

    supported_tex_rows = [
        f'{escape_latex(row["claim"])} & {escape_latex(row["measure"])} & {row["estimate"]} & {row["test"]} ' + r"\\"
        for row in supported_rows
    ]
    supported_tex = "\n".join(
        [
            r"\begin{tabular}{p{0.19\textwidth}p{0.29\textwidth}p{0.24\textwidth}p{0.19\textwidth}}",
            r"\toprule",
            r"Claim & Measure & Estimate [95\% CI] & Test \\",
            r"\midrule",
            *supported_tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output_dir / "table_trace_supported_evidence.tex").write_text(
        supported_tex + "\n", encoding="utf-8"
    )

    audit_tex_rows = []
    for row in audit_rows:
        status = row["test"]
        if status == "Pending":
            status = r"\textit{Pending}"
        elif status == "Not established":
            status = r"\textit{Not established}"
        audit_tex_rows.append(
            f'{escape_latex(row["claim"])} & {escape_latex(row["measure"])} & {row["estimate"]} & {status} ' + r"\\"
        )
    audit_tex = "\n".join(
        [
            r"\begin{tabular}{p{0.20\textwidth}p{0.31\textwidth}p{0.25\textwidth}p{0.15\textwidth}}",
            r"\toprule",
            r"Claim & Required evidence & Current estimate [95\% CI] & Status \\",
            r"\midrule",
            *audit_tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output_dir / "table_trace_claim_audit.tex").write_text(
        audit_tex + "\n", encoding="utf-8"
    )
    return {"supported": supported_rows, "audit": audit_rows}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage1, final = load_aligned_records(
        args.stage1_records, args.final_records, args.max_records
    )
    source_dir = args.evidence_dir / "source_data"
    identity_csv = (
        args.evidence_dir
        / "path_identity_retention"
        / "source_data"
        / "path_identity_summary.csv"
    )

    global_pca = make_global_pca_figure(stage1, final, args.output_dir)
    reliability = make_refinement_figure(
        stage1,
        final,
        source_dir / "stagewise_benchmarks.csv",
        args.output_dir,
        args.bootstrap_trials,
        args.seed,
    )
    population = make_population_structure_figure(
        source_dir / "trajectory_permutation_null.csv",
        identity_csv,
        args.output_dir,
    )
    probe_rows, geometry_audit = make_outcome_audit_figure(
        stage1,
        final,
        args.output_dir,
        args.probe_epochs,
        args.bootstrap_trials,
        args.geometry_permutations,
        args.seed,
    )
    tables = make_evidence_tables(
        source_dir / "stagewise_benchmarks.csv",
        source_dir / "trajectory_permutation_null.csv",
        identity_csv,
        reliability,
        probe_rows,
        geometry_audit,
        args.output_dir,
    )

    manifest = {
        "suite": "TRACE submission figure redesign",
        "n_questions": len(stage1),
        "n_views_per_question": 8,
        "test_times": 1,
        "palette": COLORS,
        "font_request": "Times New Roman",
        "font_embedded": FIGURE_SERIF,
        "global_pca": global_pca,
        "reliability": reliability,
        "population_structure": population,
        "outcome_probe": probe_rows,
        "outcome_geometry_audit": geometry_audit["rows"],
        "tables": tables,
        "claim_boundary": {
            "supported": "utility, fixed-view reliability, permutation-controlled ordered trajectory, identity retention",
            "pending": "paraphrase invariance and valid question-blocked full-path causal mediation",
            "not_established": "universal same-question correct-versus-wrong path separation",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "global_pca": global_pca}, indent=2))


if __name__ == "__main__":
    main()
