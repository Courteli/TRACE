#!/usr/bin/env python3
"""Build an honest paper-ready summary of the frozen TRACE epoch7 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#5B7DB1",
    "green": "#66B07A",
    "gold": "#E6A516",
    "pink": "#E5A6C4",
    "bridge": "#5B7DB1",
    "stage1": "#66B07A",
    "trace": "#E6A516",
    "good": "#66B07A",
    "bad": "#E5A6C4",
    "neutral": "#8A9099",
    "all_wrong": "#E5A6C4",
    "mixed": "#E6A516",
    "all_correct": "#66B07A",
    "grid": "#DDE1E6",
    "text": "#29313A",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def style_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AEB5BF")
    ax.tick_params(colors="#4B5563", labelsize=9)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def panel_title(ax, letter: str, title: str, subtitle: str | None = None) -> None:
    ax.text(
        0.0,
        1.13,
        f"{letter}  {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=COLORS["text"],
    )
    if subtitle:
        ax.text(
            0.0,
            1.075,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#59616C",
        )


def plot_accuracy_length(ax, paired: dict) -> None:
    summaries = paired["summaries"]
    dataset_colors = [COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["pink"]]
    for row, color in zip(summaries, dataset_colors):
        x0, y0 = row["reference_L"], row["reference_acc"]
        x1, y1 = row["candidate_L"], row["candidate_acc"]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.3, mutation_scale=13),
        )
        ax.scatter(x0, y0, s=52, facecolor="white", edgecolor=color, linewidth=1.8, zorder=3)
        ax.scatter(x1, y1, s=58, facecolor=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(
            row["dataset"],
            (x1, y1),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            fontweight="bold",
        )
        ax.annotate(
            f"{row['accuracy_delta_pp']:+.2f} pp, {row['length_delta']:+.2f} L",
            ((x0 + x1) / 2, (y0 + y1) / 2),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=7.6,
            color="#59616C",
        )
    ax.set_xlabel("Total reasoning length #L  (left is better)", fontsize=9.5)
    ax.set_ylabel("Accuracy (%)  (up is better)", fontsize=9.5)
    ax.invert_xaxis()
    style_axis(ax)
    panel_title(
        ax,
        "A",
        "Stage2 moves the accuracy-length frontier",
        "Open marker: identical Stage1 checkpoint; filled marker: frozen epoch7",
    )


def plot_structure(ax, structure: dict) -> None:
    stats = structure["method_statistics"]
    methods = ["BRIDGE", "Stage1", "TRACE-epoch7"]
    labels = ["Progress\nspan", "Monotonicity", "Step\nalignment", "Final-path\nalignment"]
    keys = [
        "assignment_progress_span",
        "assignment_progress_inversion_frac",
        "diag_residual_cos",
        "final_path_cos",
    ]
    values = []
    low = []
    high = []
    for method in methods:
        method_values = []
        method_low = []
        method_high = []
        for key in keys:
            item = stats[method][key]
            if key == "assignment_progress_inversion_frac":
                method_values.append(1.0 - item["mean"])
                method_low.append(1.0 - item["bootstrap_ci95_high"])
                method_high.append(1.0 - item["bootstrap_ci95_low"])
            else:
                method_values.append(item["mean"])
                method_low.append(item["bootstrap_ci95_low"])
                method_high.append(item["bootstrap_ci95_high"])
        values.append(method_values)
        low.append(method_low)
        high.append(method_high)

    x = np.arange(len(labels))
    width = 0.24
    method_colors = [COLORS["bridge"], COLORS["stage1"], COLORS["trace"]]
    for index, (method, color) in enumerate(zip(methods, method_colors)):
        y = np.asarray(values[index])
        yerr = np.vstack([y - np.asarray(low[index]), np.asarray(high[index]) - y])
        ax.bar(
            x + (index - 1) * width,
            y,
            width,
            color=color,
            label=method,
            yerr=yerr,
            capsize=2,
            error_kw={"elinewidth": 0.8, "ecolor": color},
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Score (higher is better)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower right")
    style_axis(ax)
    panel_title(
        ax,
        "B",
        "Stage1 creates ordered trajectory structure",
        "Paired 200-question audit; monotonicity = 1 - progress inversion fraction",
    )


def plot_rollout_composition(ax, geometry: dict) -> None:
    centered = geometry["centered"]
    compositions = [centered["stage1_composition"], centered["trace_composition"]]
    labels = ["Stage1", "TRACE epoch7"]
    categories = ["all_wrong", "mixed", "all_correct"]
    category_labels = ["All wrong", "Mixed", "All correct"]
    bottoms = np.zeros(2)
    for category, label in zip(categories, category_labels):
        vals = np.asarray([item[category] for item in compositions], dtype=float)
        bars = ax.bar(
            labels,
            vals,
            bottom=bottoms,
            color=COLORS[category],
            width=0.56,
            label=label,
        )
        for bar, value, bottom in zip(bars, vals, bottoms):
            if value >= 20:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    color="white" if category != "mixed" else "#3B3020",
                    fontsize=9,
                    fontweight="bold",
                )
        bottoms += vals
    paired = centered["paired_rollout"]
    ax.text(
        0.5,
        0.965,
        f"8-path rollout accuracy: {paired['stage1_rollout_accuracy']:.2f}%  ->  {paired['trace_rollout_accuracy']:.2f}%"
        f"  ({paired['rollout_accuracy_delta_pp']:+.2f} pp)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=COLORS["good"],
        fontweight="bold",
    )
    ax.set_ylim(0, 210)
    ax.set_ylabel("Questions (n=200)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.91))
    style_axis(ax)
    panel_title(
        ax,
        "C",
        "Stage2 converts mixed groups into all-correct groups",
        "All-wrong count stays 38; all-correct rises from 58 to 88",
    )


def oriented_interval(item: dict) -> tuple[float, float, float]:
    value = item["oriented_improvement"]
    if item["direction"] == "lower_is_better":
        low = -item["raw_delta_bootstrap_ci95_high"]
        high = -item["raw_delta_bootstrap_ci95_low"]
    else:
        low = item["raw_delta_bootstrap_ci95_low"]
        high = item["raw_delta_bootstrap_ci95_high"]
    return value, low, high


def plot_stage2_geometry_delta(ax, geometry: dict) -> None:
    items = geometry["centered"]["metrics"][:4]
    short_labels = ["Correct compactness", "Wrong-to-correct distance", "Correct/wrong margin", "Wrong dispersion"]
    y = np.arange(len(items))[::-1]
    for yi, item, label in zip(y, items, short_labels):
        value, low, high = oriented_interval(item)
        color = COLORS["good"] if low > 0 else COLORS["bad"] if high < 0 else COLORS["neutral"]
        ax.errorbar(
            value,
            yi,
            xerr=np.asarray([[value - low], [high - value]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=3,
            markersize=6,
        )
        ax.text(high + 0.018, yi, f"{value:+.3f}", va="center", fontsize=8, color=color, fontweight="bold")
    ax.axvline(0, color="#20242A", linewidth=1)
    ax.set_yticks(y, short_labels)
    ax.set_xlabel("Oriented Stage1 -> epoch7 improvement", fontsize=9.5)
    ax.set_xlim(-0.66, 0.16)
    style_axis(ax, grid_axis="x")
    panel_title(
        ax,
        "D",
        "Stage2 geometry is mixed",
        "Correct paths compact strongly; wrong-path margin and dispersion decrease",
    )


def plot_global_pca(ax, image_path: Path) -> None:
    image = mpimg.imread(image_path)
    image = image[int(0.045 * image.shape[0]) :, ...]
    rgb = image[..., :3]
    nonwhite = np.min(rgb, axis=-1) < 0.985
    rows, cols = np.where(nonwhite)
    if rows.size and cols.size:
        margin = 24
        y0 = max(0, int(rows.min()) - margin)
        y1 = min(image.shape[0], int(rows.max()) + margin + 1)
        x0 = max(0, int(cols.min()) - margin)
        x1 = min(image.shape[1], int(cols.max()) + margin + 1)
        image = image[y0:y1, x0:x1]
    ax.imshow(image)
    ax.axis("off")
    panel_title(
        ax,
        "E",
        "Same-question latent trajectories under one global PCA",
        "Joint fit over 48,600 points; examples selected by outcome transition, never by geometry",
    )


def plot_outcome_null(ax, null_data: dict) -> None:
    methods = ["TRACE-epoch7", "TRACE-seed1"]
    labels = ["Compactness", "Wrong distance", "C/W margin", "Wrong dispersion"]
    offsets = [0.12, -0.12]
    colors = [COLORS["gold"], COLORS["pink"]]
    base_y = np.arange(len(labels))[::-1]
    for method, offset, color in zip(methods, offsets, colors):
        items = null_data["methods"][method]["outcome_geometry_null"]
        for yi, item in zip(base_y + offset, items):
            value = item["oriented_excess_over_null"]
            low = item["oriented_excess_bootstrap_ci95_low"]
            high = item["oriented_excess_bootstrap_ci95_high"]
            ax.errorbar(
                value,
                yi,
                xerr=np.asarray([[value - low], [high - value]]),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=3,
                markersize=5,
                label=method if yi == base_y[0] + offset else None,
            )
    ax.axvline(0, color="#20242A", linewidth=1)
    ax.set_yticks(base_y, labels)
    ax.set_xlabel("Excess over within-question outcome-label null", fontsize=9.5)
    ax.set_xlim(-0.13, 0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style_axis(ax, grid_axis="x")
    panel_title(
        ax,
        "F",
        "Outcome-specific geometry is not established",
        "1,024 label permutations per question; every 95% CI crosses zero on both seeds",
    )


def plot_view_identity(ax, null_data: dict) -> None:
    methods = ["BRIDGE", "Stage1", "TRACE-epoch7", "TRACE-seed1"]
    labels = ["BRIDGE", "Stage1", "Epoch7\nseed0", "Epoch7\nseed1"]
    values = [null_data["methods"][method]["view_structure"]["crossfit_view_id_accuracy"] for method in methods]
    colors = [COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["pink"]]
    bars = ax.bar(labels, values, color=colors, width=0.64)
    chance = null_data["methods"]["BRIDGE"]["view_structure"]["chance_view_id_accuracy"]
    ax.axhline(chance, color="#20242A", linestyle="--", linewidth=1.1, label=f"Chance = {chance:.3f}")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.022,
            f"{100 * value:.1f}%",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=bar.get_facecolor(),
        )
    ax.text(
        0.98,
        0.91,
        "All-correct epoch7 mode template\nseed0: 88/88  |  seed1: 84/84\nfixed split: v0  |  v1-v7",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=COLORS["bad"],
        bbox={"facecolor": "white", "edgecolor": "#D7DCE2", "boxstyle": "round,pad=0.35"},
    )
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Cross-fitted view-ID accuracy", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left")
    style_axis(ax)
    panel_title(
        ax,
        "G",
        "The visible branches encode persistent view identity",
        "This control changes the interpretation of the 3D paths, not the task accuracy result",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.evidence_root.resolve()
    output_dir = (args.output_dir or root / "paper_evidence_summary").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paired = load_json(root / "paired_stage1_vs_epoch7" / "paired_evidence.json")
    structure = load_json(root / "trajectory_structure_200" / "trajectory_structure.json")
    geometry = load_json(root / "stage1_to_epoch7_geometry_delta" / "stage_geometry_delta.json")
    null_data = load_json(root / "outcome_geometry_view_null1024" / "outcome_geometry_null.json")
    pca_image = root / "global_pca_bridge_stage1_epoch7" / "trace_bridge_same_question_global_pca_3d_normalized_objective.png"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(18, 28), constrained_layout=False)
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 1.0, 2.05, 1.05],
        left=0.065,
        right=0.985,
        top=0.915,
        bottom=0.035,
        hspace=0.48,
        wspace=0.36,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, :]),
        figure.add_subplot(grid[3, 0]),
        figure.add_subplot(grid[3, 1]),
    ]

    plot_accuracy_length(axes[0], paired)
    plot_structure(axes[1], structure)
    plot_rollout_composition(axes[2], geometry)
    plot_stage2_geometry_delta(axes[3], geometry)
    plot_global_pca(axes[4], pca_image)
    plot_outcome_null(axes[5], null_data)
    plot_view_identity(axes[6], null_data)

    figure.suptitle(
        "TRACE epoch7: supported gains and the outcome-geometry boundary",
        y=0.985,
        fontsize=20,
        fontweight="bold",
        color=COLORS["text"],
    )
    figure.text(
        0.5,
        0.965,
        "Frozen checkpoint | paired question-level uncertainty | test_times=1 | 200-question, 8-path geometry audit",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#59616C",
    )

    png_path = output_dir / "trace_epoch7_paper_evidence_summary.png"
    pdf_path = output_dir / "trace_epoch7_paper_evidence_summary.pdf"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "title": "TRACE epoch7: supported gains and the outcome-geometry boundary",
        "frozen_checkpoint_sha256": "9c166dd2069753f891446851b9b0eb829136e4ad2803cf1daabce57df5f74f3f",
        "sources": [
            "paired_stage1_vs_epoch7/paired_evidence.json",
            "trajectory_structure_200/trajectory_structure.json",
            "stage1_to_epoch7_geometry_delta/stage_geometry_delta.json",
            "outcome_geometry_view_null1024/outcome_geometry_null.json",
            "global_pca_bridge_stage1_epoch7/trace_bridge_same_question_global_pca_3d_normalized_objective.png",
        ],
        "outputs": [str(png_path), str(pdf_path)],
        "interpretation": {
            "supported": [
                "accuracy improves while total reasoning length falls",
                "Stage1 constructs ordered latent-progress structure",
                "Stage2 consolidates correct paths and raises rollout accuracy",
            ],
            "not_established": [
                "geometry tracks correctness beyond fixed view identity",
                "Stage2 improves wrong-path separation over Stage1",
            ],
        },
    }
    with (output_dir / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
