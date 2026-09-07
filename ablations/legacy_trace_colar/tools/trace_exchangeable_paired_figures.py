#!/usr/bin/env python3
"""Publication evidence for the single-seed TRACE Stage1-to-Stage2 mainline.

The script deliberately avoids training-factor ablations. It visualizes the
paired task effect, paired geometry effect, and unedited complete paths from
the validation-selected Stage1 and Stage2 checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_exchangeable_visualize import (  # noqa: E402
    _camera_basis,
    _path_distance_matrix,
    fit_global_increment_pca,
    local_outcome_summary,
    project_complete_paths,
    record_outcomes,
    record_residuals,
    select_unlabeled_camera,
)


COLORS = {
    "blue": "#6789B7",
    "green": "#69B482",
    "orange": "#E6A019",
    "pink": "#E7A1BF",
    "ink": "#303640",
    "muted": "#687383",
    "grid": "#DCE2EA",
    "paper": "#FFFFFF",
}
DATASET_COLORS = {
    "GSM8K": COLORS["blue"],
    "GSMHard": COLORS["green"],
    "SVAMP": COLORS["orange"],
    "MultiArith": COLORS["pink"],
}
DATASET_ORDER = ["GSM8K", "GSMHard", "SVAMP", "MultiArith"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.size": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def parse_labeled_path(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected LABEL=/path/to/summary.json")
    label, path = raw.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Both label and path are required")
    return label, Path(path)


def finite_bounds(values: Iterable[float], *, include_zero: bool = True) -> Tuple[float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if include_zero:
        array = np.concatenate([array, np.asarray([0.0])])
    if not array.size:
        return -1.0, 1.0
    low = float(array.min())
    high = float(array.max())
    span = max(high - low, max(abs(low), abs(high)) * 0.15, 1e-3)
    return low - 0.14 * span, high + 0.27 * span


def style_effect_axis(ax: plt.Axes) -> None:
    ax.axvline(0.0, color=COLORS["muted"], linewidth=0.8, linestyle="--", zorder=0)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_forest(
    ax: plt.Axes,
    rows: Sequence[dict],
    *,
    title: str,
    xlabel: str,
    value_format: str,
) -> None:
    bounds = []
    for row_index, row in enumerate(rows):
        y = len(rows) - 1 - row_index
        color = DATASET_COLORS[row["dataset"]]
        mean = row["mean"]
        low = row["low"]
        high = row["high"]
        bounds.extend([low, high])
        ax.plot([low, high], [y, y], color=color, linewidth=1.6, solid_capstyle="round")
        ax.scatter(
            [mean],
            [y],
            s=34,
            color=color,
            edgecolor=COLORS["ink"],
            linewidth=0.45,
            zorder=3,
        )
        ax.text(
            high,
            y,
            "  " + value_format.format(mean),
            va="center",
            ha="left",
            fontsize=7,
            color=color,
            fontweight="bold",
        )
    ax.set_yticks(
        np.arange(len(rows)),
        labels=[row["dataset"] for row in reversed(rows)],
    )
    ax.set_xlim(*finite_bounds(bounds))
    ax.set_title(title, loc="left", pad=5, fontweight="bold")
    ax.set_xlabel(xlabel)
    style_effect_axis(ax)


def plot_task_effects(task_summary_path: Path, out_dir: Path) -> dict:
    payload = json.loads(task_summary_path.read_text(encoding="utf-8"))
    accuracy_rows = []
    length_rows = []
    source_rows = []
    key_map = {
        "gsm8k": "GSM8K",
        "gsmhard": "GSMHard",
        "svamp": "SVAMP",
        "multiarith": "MultiArith",
    }
    for raw_label, values in payload["datasets"].items():
        label = key_map[raw_label]
        accuracy = values["final_minus_stage1"]["metrics"]["accuracy"]
        length = values["final_minus_stage1"]["metrics"]["L"]
        accuracy_rows.append(
            {
                "dataset": label,
                "mean": 100.0 * accuracy["mean"],
                "low": 100.0 * accuracy["ci95_low"],
                "high": 100.0 * accuracy["ci95_high"],
            }
        )
        length_rows.append(
            {
                "dataset": label,
                "mean": -length["mean"],
                "low": -length["ci95_high"],
                "high": -length["ci95_low"],
            }
        )
        source_rows.append(
            {
                "dataset": label,
                "stage1_accuracy": values["stage1"]["aggregate"]["accuracy"]["mean"],
                "final_accuracy": values["final"]["aggregate"]["accuracy"]["mean"],
                "accuracy_delta": accuracy["mean"],
                "accuracy_delta_ci95_low": accuracy["ci95_low"],
                "accuracy_delta_ci95_high": accuracy["ci95_high"],
                "stage1_L": values["stage1"]["aggregate"]["L"]["mean"],
                "final_L": values["final"]["aggregate"]["L"]["mean"],
                "L_delta": length["mean"],
                "L_delta_ci95_low": length["ci95_low"],
                "L_delta_ci95_high": length["ci95_high"],
            }
        )
    accuracy_rows.sort(key=lambda row: DATASET_ORDER.index(row["dataset"]))
    length_rows.sort(key=lambda row: DATASET_ORDER.index(row["dataset"]))
    accuracy_title = (
        "Accuracy rises after outcome refinement"
        if all(row["mean"] > 0.0 for row in accuracy_rows)
        else "Accuracy change after outcome refinement"
    )
    length_title = (
        "Reasoning length does not inflate"
        if all(row["mean"] >= 0.0 for row in length_rows)
        else "Reasoning-length change after refinement"
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.45))
    draw_forest(
        axes[0],
        accuracy_rows,
        title=accuracy_title,
        xlabel="Final minus Stage 1 accuracy (percentage points)",
        value_format="{:+.2f}",
    )
    draw_forest(
        axes[1],
        length_rows,
        title=length_title,
        xlabel="Tokens saved by Final relative to Stage 1",
        value_format="{:+.2f}",
    )
    axes[0].text(
        -0.18,
        1.06,
        "a",
        transform=axes[0].transAxes,
        fontsize=10,
        fontweight="bold",
    )
    axes[1].text(
        -0.18,
        1.06,
        "b",
        transform=axes[1].transAxes,
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Validation-selected checkpoints; all test questions; paired question-bootstrap 95% CIs; test_times=1.",
        ha="center",
        color=COLORS["muted"],
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.27, top=0.84, wspace=0.44)
    base = out_dir / "trace_task_effects"
    save_figure(fig, base)
    plt.close(fig)

    csv_path = out_dir / "trace_task_effects_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    return {"figure": str(base.with_suffix(".pdf")), "source_data": str(csv_path)}


def oriented_metric(metric: dict, orientation: float) -> dict:
    if metric["mean"] is None:
        return {"n": 0, "mean": None, "low": None, "high": None}
    if orientation > 0:
        return {
            "n": metric["n"],
            "mean": metric["mean"],
            "low": metric["ci95_low"],
            "high": metric["ci95_high"],
        }
    return {
        "n": metric["n"],
        "mean": -metric["mean"],
        "low": -metric["ci95_high"],
        "high": -metric["ci95_low"],
    }


def plot_geometry_effects(
    geometry_paths: Sequence[Tuple[str, Path]],
    out_dir: Path,
) -> dict:
    outcome_spec = [
        ("outcome_margin", "Outcome margin", 1.0),
        ("wrong_to_local_correct_distance", "Wrong-path rejection", 1.0),
        ("correct_local_radius", "Correct-path compactness", -1.0),
    ]
    structure_spec = [
        ("position_order_excess_over_null", "Position order", 1.0),
        ("step_alignment_excess_over_null", "Step alignment", 1.0),
        ("final_path_alignment_cos", "Final-path alignment", 1.0),
    ]
    outcome_rows = []
    structure_rows = []
    source_rows = []
    for dataset, path in geometry_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        paired = payload.get("paired_delta")
        if not paired:
            raise ValueError(f"{dataset}: geometry summary has no paired delta")
        metrics = paired["metrics"]
        for family, specification, target in (
            ("outcome", outcome_spec, outcome_rows),
            ("structure", structure_spec, structure_rows),
        ):
            for metric_key, metric_label, orientation in specification:
                oriented = oriented_metric(metrics[metric_key], orientation)
                if oriented["mean"] is None:
                    continue
                row = {
                    "dataset": dataset,
                    "metric": metric_label,
                    **oriented,
                }
                target.append(row)
                source_rows.append(
                    {
                        "family": family,
                        "dataset": dataset,
                        "metric": metric_key,
                        "orientation": orientation,
                        "n": oriented["n"],
                        "oriented_delta": oriented["mean"],
                        "oriented_delta_ci95_low": oriented["low"],
                        "oriented_delta_ci95_high": oriented["high"],
                    }
                )

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 4.25))
    outcome_supported = bool(outcome_rows) and all(
        row["mean"] > 0.0 for row in outcome_rows
    )
    structure_retained = bool(structure_rows) and all(
        row["low"] >= -0.05 for row in structure_rows
    )
    for ax, rows, title, xlabel in (
        (
            axes[0],
            outcome_rows,
            (
                "Outcome feedback improves local path geometry"
                if outcome_supported
                else "Outcome-conditioned local path change"
            ),
            r"Oriented paired change in $D_{\mathrm{path}}$ (right is better)",
        ),
        (
            axes[1],
            structure_rows,
            (
                "The ordered scaffold is retained"
                if structure_retained
                else "Ordered-scaffold change after refinement"
            ),
            "Paired change in structure score (right is better)",
        ),
    ):
        grouped = []
        y_positions = []
        y_labels = []
        y = 0
        metric_order = (
            [item[1] for item in outcome_spec]
            if rows is outcome_rows
            else [item[1] for item in structure_spec]
        )
        bounds = []
        for metric_label in metric_order:
            metric_rows = [
                row
                for row in rows
                if row["metric"] == metric_label
            ]
            metric_rows.sort(key=lambda row: DATASET_ORDER.index(row["dataset"]))
            for row in metric_rows:
                grouped.append((y, row))
                y_positions.append(y)
                y_labels.append(row["dataset"])
                y += 1
            y += 0.65
        for y_position, row in grouped:
            color = DATASET_COLORS[row["dataset"]]
            bounds.extend([row["low"], row["high"]])
            ax.plot(
                [row["low"], row["high"]],
                [y_position, y_position],
                color=color,
                linewidth=1.45,
                solid_capstyle="round",
            )
            ax.scatter(
                row["mean"],
                y_position,
                s=28,
                color=color,
                edgecolor=COLORS["ink"],
                linewidth=0.4,
                zorder=3,
            )
            ax.text(
                row["high"],
                y_position,
                f"  {row['mean']:+.3f}",
                va="center",
                ha="left",
                fontsize=6.2,
                color=color,
                fontweight="bold",
            )
        ax.set_yticks(y_positions, labels=y_labels)
        ax.invert_yaxis()
        ax.set_xlim(*finite_bounds(bounds))
        ax.set_title(title, loc="left", pad=6, fontweight="bold")
        ax.set_xlabel(xlabel)
        style_effect_axis(ax)
        start = 0
        for metric_label in metric_order:
            metric_count = sum(row["metric"] == metric_label for row in rows)
            if metric_count == 0:
                continue
            metric_center = start + 0.5 * (metric_count - 1)
            ax.text(
                -0.34,
                metric_center,
                metric_label,
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right",
                fontsize=7,
                fontweight="bold",
                color=COLORS["ink"],
            )
            start += metric_count + 0.65
    axes[0].text(
        -0.31,
        1.035,
        "a",
        transform=axes[0].transAxes,
        fontsize=10,
        fontweight="bold",
    )
    axes[1].text(
        -0.31,
        1.035,
        "b",
        transform=axes[1].transAxes,
        fontsize=10,
        fontweight="bold",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor=COLORS["ink"],
            linewidth=1.4,
            label=dataset,
        )
        for dataset, color in DATASET_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995))
    fig.text(
        0.5,
        0.012,
        "Same 200 questions and eight exchangeable rollouts per question under one fixed sampling seed; paired 95% CIs.",
        ha="center",
        color=COLORS["muted"],
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.16, top=0.88, wspace=0.70)
    base = out_dir / "trace_geometry_effects"
    save_figure(fig, base)
    plt.close(fig)

    csv_path = out_dir / "trace_geometry_effects_source.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    return {"figure": str(base.with_suffix(".pdf")), "source_data": str(csv_path)}


def load_aligned_records(
    stage1_path: Path,
    final_path: Path,
    max_records: int,
) -> Tuple[List[dict], List[dict]]:
    stage1_raw = torch.load(stage1_path, map_location="cpu", weights_only=False)
    final_raw = torch.load(final_path, map_location="cpu", weights_only=False)
    stage1_map = {int(record["idx"]): record for record in stage1_raw[:max_records]}
    final_map = {int(record["idx"]): record for record in final_raw[:max_records]}
    common = sorted(set(stage1_map) & set(final_map))
    if not common:
        raise ValueError("Stage1 and Final caches have no matched question IDs")
    return [stage1_map[idx] for idx in common], [final_map[idx] for idx in common]


def select_cases(
    stage1_records: Sequence[dict],
    final_records: Sequence[dict],
    count: int,
) -> List[int]:
    candidates = []
    for position, (stage1, final) in enumerate(zip(stage1_records, final_records)):
        stage1_outcomes = record_outcomes(stage1)
        final_outcomes = record_outcomes(final)
        if int(final_outcomes.sum()) < 2 or int((~final_outcomes).sum()) < 1:
            continue
        stage1_local = local_outcome_summary(
            _path_distance_matrix(record_residuals(stage1)),
            stage1_outcomes,
        )
        final_local = local_outcome_summary(
            _path_distance_matrix(record_residuals(final)),
            final_outcomes,
        )
        stage1_margin = (
            stage1_local["outcome_margin"] if stage1_local is not None else -math.inf
        )
        final_margin = final_local["outcome_margin"]
        margin_gain = (
            final_margin - stage1_margin if np.isfinite(stage1_margin) else final_margin
        )
        correctness_gain = int(final_outcomes.sum()) - int(stage1_outcomes.sum())
        balance = min(int(final_outcomes.sum()), int((~final_outcomes).sum()))
        candidates.append(
            (
                (
                    correctness_gain,
                    margin_gain,
                    balance,
                    -int(final["idx"]),
                ),
                position,
            )
        )
    candidates.sort(reverse=True)
    return [position for _, position in candidates[:count]]


def shared_3d_limits(paths: np.ndarray) -> Tuple[np.ndarray, float]:
    flat = paths.reshape(-1, 3)
    minima = flat.min(axis=0)
    maxima = flat.max(axis=0)
    center = 0.5 * (minima + maxima)
    radius = 0.55 * max(float((maxima - minima).max()), 1e-6)
    return center, radius


def set_shared_3d_limits(ax: plt.Axes, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 0.86))


def draw_paths_3d(
    ax: plt.Axes,
    paths: np.ndarray,
    outcomes: np.ndarray,
    *,
    elevation: float,
    azimuth: float,
    limits: Tuple[np.ndarray, float],
    title: str,
) -> None:
    for path, correct in zip(paths, outcomes):
        color = COLORS["pink"] if correct else COLORS["orange"]
        linestyle = "-" if correct else "--"
        endpoint = "*" if correct else "X"
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            linestyle=linestyle,
            linewidth=1.35,
            alpha=0.9,
        )
        ax.scatter(
            path[1:-1, 0],
            path[1:-1, 1],
            path[1:-1, 2],
            color=color,
            s=7,
            alpha=0.72,
            depthshade=False,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            color=color,
            marker=endpoint,
            s=40,
            edgecolor=COLORS["ink"],
            linewidth=0.35,
            depthshade=False,
        )
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        color=COLORS["blue"],
        marker="o",
        s=24,
        edgecolor=COLORS["ink"],
        linewidth=0.4,
        depthshade=False,
    )
    ax.view_init(elev=elevation, azim=azimuth)
    set_shared_3d_limits(ax, *limits)
    ax.set_xlabel("Global PC1", labelpad=1)
    ax.set_ylabel("Global PC2", labelpad=1)
    ax.set_zlabel("Global PC3", labelpad=1)
    ax.tick_params(pad=0.2, length=2)
    ax.set_title(title, loc="left", pad=3, fontweight="bold")
    ax.grid(True, linewidth=0.35, alpha=0.45)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(COLORS["grid"])


def projected_terminal_paths(
    paths: np.ndarray,
    elevation: float,
    azimuth: float,
) -> np.ndarray:
    horizontal, vertical = _camera_basis(elevation, azimuth)
    return np.stack(
        [
            np.tensordot(paths, horizontal, axes=([-1], [0])),
            np.tensordot(paths, vertical, axes=([-1], [0])),
        ],
        axis=-1,
    )


def draw_terminal_paths(
    ax: plt.Axes,
    paths: np.ndarray,
    outcomes: np.ndarray,
    *,
    title: str,
    limits: Tuple[float, float, float, float],
) -> None:
    for path, correct in zip(paths, outcomes):
        color = COLORS["pink"] if correct else COLORS["orange"]
        linestyle = "-" if correct else "--"
        marker = "*" if correct else "X"
        ax.plot(
            path[-4:, 0],
            path[-4:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=1.35,
            alpha=0.9,
        )
        ax.scatter(
            path[-4:-1, 0],
            path[-4:-1, 1],
            color=color,
            s=10,
            alpha=0.72,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            color=color,
            marker=marker,
            s=42,
            edgecolor=COLORS["ink"],
            linewidth=0.35,
            zorder=4,
        )
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", pad=5, fontweight="bold")
    ax.set_xlabel("Camera-plane axis 1")
    ax.set_ylabel("Camera-plane axis 2")
    ax.grid(True, color=COLORS["grid"], linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def plot_paired_case(
    stage1: dict,
    final: dict,
    *,
    mean: np.ndarray,
    components: np.ndarray,
    explained: np.ndarray,
    pca_fit_question_count: int,
    out_dir: Path,
) -> dict:
    question_id = int(final["idx"])
    stage1_outcomes = record_outcomes(stage1)
    final_outcomes = record_outcomes(final)
    stage1_paths = project_complete_paths(record_residuals(stage1), mean, components)
    final_paths = project_complete_paths(record_residuals(final), mean, components)
    combined_paths = np.concatenate([stage1_paths, final_paths], axis=0)
    elevation, azimuth, visibility = select_unlabeled_camera(combined_paths)
    limits = shared_3d_limits(combined_paths)
    stage1_local = local_outcome_summary(
        _path_distance_matrix(record_residuals(stage1)),
        stage1_outcomes,
    )
    final_local = local_outcome_summary(
        _path_distance_matrix(record_residuals(final)),
        final_outcomes,
    )

    fig = plt.figure(figsize=(7.08, 3.15))
    left = fig.add_subplot(1, 2, 1, projection="3d")
    right = fig.add_subplot(1, 2, 2, projection="3d")
    draw_paths_3d(
        left,
        stage1_paths,
        stage1_outcomes,
        elevation=elevation,
        azimuth=azimuth,
        limits=limits,
        title=f"Stage 1: {int(stage1_outcomes.sum())}/8 correct paths",
    )
    draw_paths_3d(
        right,
        final_paths,
        final_outcomes,
        elevation=elevation,
        azimuth=azimuth,
        limits=limits,
        title=f"Final TRACE: {int(final_outcomes.sum())}/8 correct paths",
    )
    handles = [
        Line2D([0], [0], color=COLORS["pink"], marker="*", linewidth=1.5, label="Correct"),
        Line2D(
            [0],
            [0],
            color=COLORS["orange"],
            marker="X",
            linestyle="--",
            linewidth=1.5,
            label="Wrong",
        ),
        Line2D([0], [0], color=COLORS["blue"], marker="o", linewidth=0, label="Origin"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985))
    stage1_margin = None if stage1_local is None else stage1_local["outcome_margin"]
    final_margin = None if final_local is None else final_local["outcome_margin"]
    margin_text = (
        "NA"
        if stage1_margin is None or final_margin is None
        else f"{stage1_margin:+.3f} -> {final_margin:+.3f}"
    )
    fig.text(
        0.5,
        0.012,
        f"q{question_id} | outcome margin {margin_text} | held-out global PCA "
        f"PC1-3={100.0 * float(explained.sum()):.1f}% | no path scaling, offset, or lane.",
        ha="center",
        color=COLORS["muted"],
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.14, top=0.87, wspace=0.02)
    path_base = out_dir / f"trace_paired_complete_paths_q{question_id}"
    save_figure(fig, path_base)
    plt.close(fig)

    stage1_terminal = projected_terminal_paths(stage1_paths, elevation, azimuth)
    final_terminal = projected_terminal_paths(final_paths, elevation, azimuth)
    combined_terminal = np.concatenate(
        [stage1_terminal[:, -4:], final_terminal[:, -4:]],
        axis=0,
    ).reshape(-1, 2)
    minima = combined_terminal.min(axis=0)
    maxima = combined_terminal.max(axis=0)
    padding = np.maximum(0.08 * (maxima - minima), 1e-6)
    terminal_limits = (
        float(minima[0] - padding[0]),
        float(maxima[0] + padding[0]),
        float(minima[1] - padding[1]),
        float(maxima[1] + padding[1]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.65))
    draw_terminal_paths(
        axes[0],
        stage1_terminal,
        stage1_outcomes,
        title="Stage 1 terminal transitions",
        limits=terminal_limits,
    )
    draw_terminal_paths(
        axes[1],
        final_terminal,
        final_outcomes,
        title="Final TRACE terminal transitions",
        limits=terminal_limits,
    )
    fig.legend(handles=handles[:2], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985))
    fig.text(
        0.5,
        0.012,
        "Same PCA, camera, and limits as the complete-path figure; crop shows only the final three transitions.",
        ha="center",
        color=COLORS["muted"],
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.84, wspace=0.28)
    terminal_base = out_dir / f"trace_paired_terminal_paths_q{question_id}"
    save_figure(fig, terminal_base)
    plt.close(fig)

    heatmap_base = plot_paired_heatmaps(stage1, final, out_dir)
    return {
        "question_id": question_id,
        "stage1_correct": int(stage1_outcomes.sum()),
        "final_correct": int(final_outcomes.sum()),
        "stage1_local_geometry": stage1_local,
        "final_local_geometry": final_local,
        "pca_fit_question_count": pca_fit_question_count,
        "explained_variance_ratio": [float(value) for value in explained],
        "camera_protocol": "outcome-blind visibility grid search over both checkpoints",
        "camera_elevation": elevation,
        "camera_azimuth": azimuth,
        "camera_visibility_score": visibility,
        "complete_path_figure": str(path_base.with_suffix(".pdf")),
        "terminal_path_figure": str(terminal_base.with_suffix(".pdf")),
        "heatmap_figure": str(heatmap_base.with_suffix(".pdf")),
    }


def ordered_distance(record: dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    residuals = record_residuals(record)
    outcomes = record_outcomes(record)
    order = np.concatenate([np.flatnonzero(outcomes), np.flatnonzero(~outcomes)])
    distance = _path_distance_matrix(residuals)
    ordered = distance[np.ix_(order, order)]
    ordered_outcomes = outcomes[order]
    labels = []
    correct_index = 0
    wrong_index = 0
    for correct in ordered_outcomes:
        if correct:
            correct_index += 1
            labels.append(f"C{correct_index}")
        else:
            wrong_index += 1
            labels.append(f"W{wrong_index}")
    return ordered, ordered_outcomes, labels


def plot_paired_heatmaps(stage1: dict, final: dict, out_dir: Path) -> Path:
    stage1_distance, stage1_outcomes, stage1_labels = ordered_distance(stage1)
    final_distance, final_outcomes, final_labels = ordered_distance(final)
    maximum = max(float(stage1_distance.max()), float(final_distance.max()), 1e-6)
    cmap = LinearSegmentedColormap.from_list(
        "trace_distance",
        ["#FFF9FC", COLORS["pink"], COLORS["blue"]],
    )
    fig = plt.figure(figsize=(7.08, 3.15))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.035),
        left=0.07,
        right=0.93,
        bottom=0.15,
        top=0.87,
        wspace=0.32,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    colorbar_ax = fig.add_subplot(grid[0, 2])
    image = None
    for ax, matrix, outcomes, labels, title in (
        (
            axes[0],
            stage1_distance,
            stage1_outcomes,
            stage1_labels,
            "Stage 1 complete-path distance",
        ),
        (
            axes[1],
            final_distance,
            final_outcomes,
            final_labels,
            "Final TRACE complete-path distance",
        ),
    ):
        image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=maximum, aspect="equal")
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", pad=6, fontweight="bold")
        for index, correct in enumerate(outcomes):
            color = COLORS["pink"] if correct else COLORS["orange"]
            ax.add_patch(
                plt.Rectangle(
                    (index - 0.5, -0.82),
                    1.0,
                    0.22,
                    color=color,
                    clip_on=False,
                    linewidth=0,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (-0.82, index - 0.5),
                    0.22,
                    1.0,
                    color=color,
                    clip_on=False,
                    linewidth=0,
                )
            )
        threshold = 0.62 * maximum
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if matrix[row, column] > threshold else COLORS["ink"],
                )
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(r"$D_{\mathrm{path}}$")
    colorbar.outline.set_linewidth(0.6)
    fig.text(
        0.5,
        0.018,
        "Rows and columns are sorted within checkpoint: correct rollouts first (pink), then wrong rollouts (orange).",
        ha="center",
        color=COLORS["muted"],
        fontsize=6.5,
    )
    base = out_dir / f"trace_paired_path_heatmaps_q{int(final['idx'])}"
    save_figure(fig, base)
    plt.close(fig)
    return base


def plot_paired_paths(
    stage1_path: Path,
    final_path: Path,
    out_dir: Path,
    *,
    max_records: int,
    case_count: int,
) -> dict:
    stage1_records, final_records = load_aligned_records(
        stage1_path,
        final_path,
        max_records,
    )
    positions = select_cases(stage1_records, final_records, case_count)
    if not positions:
        raise ValueError("No mixed-outcome records are eligible for qualitative paths")
    selected_ids = {int(final_records[position]["idx"]) for position in positions}
    joint_records = [
        record
        for pair in zip(stage1_records, final_records)
        for record in pair
    ]
    mean, components, explained, fit_increment_count = fit_global_increment_pca(
        joint_records,
        excluded_ids=selected_ids,
    )
    cases = []
    for position in positions:
        cases.append(
            plot_paired_case(
                stage1_records[position],
                final_records[position],
                mean=mean,
                components=components,
                explained=explained,
                pca_fit_question_count=len(stage1_records) - len(positions),
                out_dir=out_dir,
            )
        )
    return {
        "selection_protocol": (
            "Predeclared ranking: gain in correct rollouts, then gain in local "
            "outcome margin, then final mixed-group balance, then question ID."
        ),
        "selected_question_ids": sorted(selected_ids),
        "matched_question_count": len(stage1_records),
        "pca_protocol": {
            "fit_unit": "latent residual increments from both checkpoints",
            "displayed_questions_excluded": True,
            "outcome_labels_used_for_pca": False,
            "fit_increment_count": fit_increment_count,
            "explained_variance_ratio": [float(value) for value in explained],
            "per_path_normalization": False,
            "manual_offset_or_lane": False,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_summary", type=Path, required=True)
    parser.add_argument(
        "--geometry_summary",
        action="append",
        type=parse_labeled_path,
        required=True,
    )
    parser.add_argument("--stage1_records", type=Path, required=True)
    parser.add_argument("--final_records", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument("--case_count", type=int, default=2)
    args = parser.parse_args()

    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task = plot_task_effects(args.task_summary, args.out_dir)
    geometry = plot_geometry_effects(args.geometry_summary, args.out_dir)
    paths = plot_paired_paths(
        args.stage1_records,
        args.final_records,
        args.out_dir,
        max_records=args.max_records,
        case_count=args.case_count,
    )
    manifest = {
        "figure_contract": {
            "core_conclusion": (
                "Outcome refinement improves task utility while preserving the "
                "ordered Stage1 scaffold and increasing question-local "
                "correct-versus-wrong path separation."
            ),
            "archetype": "quantitative validation plus separate qualitative paths",
            "backend": "Python/matplotlib",
            "font": "Times New Roman with metric-compatible serif fallbacks",
            "statistics": "paired question-bootstrap 95% confidence intervals",
            "training_seeds": 1,
            "rollout_seeds": 1,
            "test_times": 1,
        },
        "integrity": {
            "checkpoint_selection": "validation accuracy only",
            "test_checkpoint_selection": False,
            "outcome_labels_used_for_pca": False,
            "manual_path_displacement": False,
            "lane_offset": False,
            "per_path_normalization": False,
            "qualitative_selection_is_disclosed": True,
            "aggregate_statistics_cover_all_matched_questions": True,
        },
        "task": task,
        "geometry": geometry,
        "paths": paths,
    }
    manifest_path = args.out_dir / "trace_paired_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
