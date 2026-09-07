#!/usr/bin/env python3
"""Build a defensible main-paper figure and table package for TRACE epoch7."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch


COLORS = {
    "blue": "#5B7DB1",
    "green": "#66B07A",
    "gold": "#E6A516",
    "pink": "#E5A6C4",
    "text": "#29313A",
    "muted": "#69727D",
    "grid": "#DDE1E6",
    "light": "#F7F8F9",
    "white": "#FFFFFF",
}
FONT_STACK = ["Times New Roman", "Liberation Serif", "Nimbus Roman No9 L", "DejaVu Serif"]
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": FONT_STACK,
    "font.size": 7.2,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.4,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


PUBLISHED_RESULTS = [
    {"method": "SFT-CoT", "family": "explicit", "GSM8K": (80.77, 127.00), "GSMHard": (54.73, 192.67), "SVAMP": (85.29, 54.48), "MultiArith": (100.00, 66.64)},
    {"method": "iCoT", "family": "answer-only", "GSM8K": (14.21, 0.00), "GSMHard": (3.17, 0.00), "SVAMP": (35.64, 0.00), "MultiArith": (50.83, 0.00)},
    {"method": "Coconut", "family": "latent", "GSM8K": (17.68, 6.00), "GSMHard": (4.35, 6.00), "SVAMP": (44.92, 6.00), "MultiArith": (59.16, 6.00)},
    {"method": "CODI", "family": "latent", "GSM8K": (6.13, 6.00), "GSMHard": (3.51, 6.00), "SVAMP": (12.47, 6.00), "MultiArith": (17.04, 6.00)},
    {"method": "CoLaR (r=5)", "family": "compressed", "GSM8K": (23.94, 21.71), "GSMHard": (6.52, 26.94), "SVAMP": (40.80, 13.12), "MultiArith": (66.11, 13.36)},
    {"method": "CoLaR (r=2)", "family": "compressed", "GSM8K": (40.05, 45.67), "GSMHard": (10.69, 56.16), "SVAMP": (59.50, 29.51), "MultiArith": (84.44, 30.83)},
    {"method": "BRIDGE", "family": "compressed", "GSM8K": (61.33, 41.53), "GSMHard": (18.50, 51.00), "SVAMP": (77.90, 30.45), "MultiArith": (100.00, 32.62)},
    {"method": "TRACE epoch7", "family": "compressed", "GSM8K": (64.59, 37.08), "GSMHard": (21.38, 47.03), "SVAMP": (83.50, 29.39), "MultiArith": (99.44, 31.01)},
]
DATASETS = ["GSM8K", "GSMHard", "SVAMP", "MultiArith"]
DATASET_COLORS = dict(zip(DATASETS, (COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["pink"])))


def pale(color: str, amount: float = 0.86) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color))
    return tuple(rgb * (1.0 - amount) + amount)


def save_figure(fig: plt.Figure, path: Path, tiff: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    if tiff:
        fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="top", color=COLORS["text"])


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def add_box(ax: plt.Axes, xy, width, height, color, title, lines=(), title_size=7.2) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=color,
        facecolor=pale(color, 0.90),
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(xy[0] + 0.018, xy[1] + height - 0.035, title, transform=ax.transAxes, fontsize=title_size, fontweight="bold", color=COLORS["text"], va="top")
    for idx, line in enumerate(lines):
        ax.text(xy[0] + 0.018, xy[1] + height - 0.080 - idx * 0.034, line, transform=ax.transAxes, fontsize=5.8, color=COLORS["muted"], va="top")
    return box


def axes_arrow(ax: plt.Axes, start, end, color=COLORS["muted"], width=1.0, style="-|>") -> None:
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle=style, mutation_scale=8, linewidth=width, color=color))


def draw_method_overview(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.01, 0.975, "TRACE: structured latent trajectories with guarded outcome adaptation", fontsize=10.0, fontweight="bold", color=COLORS["text"], va="top")
    ax.text(0.01, 0.925, "Training constructs and adapts multiple paths; inference keeps one unchanged eight-slot reasoning graph.", fontsize=6.4, color=COLORS["muted"], va="top")

    stage0 = add_box(ax, (0.015, 0.63), 0.145, 0.205, COLORS["blue"], "Stage 0 | CoT SFT", ("Shared warm start", "Question + human CoT"))
    stage1 = add_box(ax, (0.205, 0.47), 0.285, 0.365, COLORS["green"], "Stage 1 | Structured latent SFT", ("Weak CoT-step anchors", "Multi-view teacher compression"), title_size=6.8)
    stage2 = add_box(ax, (0.535, 0.40), 0.285, 0.435, COLORS["gold"], "Stage 2 | Guarded multi-path RL", ("8 paths per question", "Task + direct signature update"), title_size=6.8)
    inference = add_box(ax, (0.855, 0.47), 0.13, 0.365, COLORS["pink"], "Inference", ("Question only", "8 latent slots", "Compact answer"), title_size=7.0)
    del stage0, stage1, stage2, inference
    axes_arrow(ax, (0.163, 0.73), (0.201, 0.73), COLORS["muted"], 1.1)
    axes_arrow(ax, (0.493, 0.68), (0.531, 0.68), COLORS["muted"], 1.1)
    axes_arrow(ax, (0.823, 0.68), (0.851, 0.68), COLORS["muted"], 1.1)

    # Stage1 trajectory and weak anchors.
    slot_x = np.linspace(0.23, 0.465, 8)
    slot_y = 0.55 + 0.040 * np.sin(np.linspace(-0.8, 2.2, 8)) + np.linspace(-0.01, 0.09, 8)
    ax.plot(slot_x, slot_y, transform=ax.transAxes, color=COLORS["green"], linewidth=2.0, solid_capstyle="round")
    ax.scatter(slot_x, slot_y, transform=ax.transAxes, s=np.linspace(12, 25, 8), color=COLORS["green"], edgecolor="white", linewidth=0.45, zorder=3)
    for idx in (0, 3, 7):
        dy = 0.028 if idx == 0 else -0.038
        ax.text(slot_x[idx], slot_y[idx] + dy, f"z{idx + 1}", transform=ax.transAxes, ha="center", fontsize=5.2, color=COLORS["muted"])
    teacher_x = np.linspace(0.245, 0.45, 4)
    for idx, x in enumerate(teacher_x):
        y = 0.675
        ax.scatter([x], [y], transform=ax.transAxes, s=11, marker="s", color=COLORS["blue"], zorder=3)
        target = min(7, idx * 2 + 1)
        ax.plot([x, slot_x[target]], [y - 0.008, slot_y[target] + 0.012], transform=ax.transAxes, color=COLORS["blue"], alpha=0.40, linewidth=0.65)
    terms = [("position", 0.233), ("direction", 0.304), ("step", 0.381), ("non-collapse", 0.447)]
    for label, x in terms:
        ax.text(x, 0.505, label, transform=ax.transAxes, fontsize=5.1, color=COLORS["green"], ha="center")

    # Stage2 fan of outcome-labelled paths.
    start = np.array([0.565, 0.58])
    endpoints = [(0.76, 0.735), (0.77, 0.665), (0.755, 0.58), (0.775, 0.49)]
    path_colors = [COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["pink"]]
    for idx, (end, color) in enumerate(zip(endpoints, path_colors)):
        xs = np.linspace(start[0], end[0], 7)
        ys = np.linspace(start[1], end[1], 7) + 0.018 * np.sin(np.linspace(0, 2.5 * np.pi, 7) + idx)
        ax.plot(xs, ys, transform=ax.transAxes, color=color, linewidth=1.55, alpha=0.95)
        ax.scatter(xs[1:-1], ys[1:-1], transform=ax.transAxes, s=5, color=color, alpha=0.75)
        marker = "X" if color == COLORS["pink"] else "o"
        ax.scatter([xs[-1]], [ys[-1]], transform=ax.transAxes, s=22, marker=marker, color=color, edgecolor="white", linewidth=0.45, zorder=4)
    ax.scatter([start[0]], [start[1]], transform=ax.transAxes, s=18, marker="s", color=COLORS["text"], zorder=4)
    ax.text(0.665, 0.445, "correct modes", transform=ax.transAxes, fontsize=5.4, color=COLORS["green"], ha="center")
    ax.text(0.765, 0.445, "wrong path", transform=ax.transAxes, fontsize=5.4, color=COLORS["pink"], ha="center")

    # Inference path.
    infer_x = np.linspace(0.875, 0.963, 8)
    infer_y = 0.62 + 0.018 * np.sin(np.linspace(0, 2 * np.pi, 8))
    ax.plot(infer_x, infer_y, transform=ax.transAxes, color=COLORS["pink"], linewidth=1.7)
    ax.scatter(infer_x, infer_y, transform=ax.transAxes, s=8, color=COLORS["pink"], edgecolor="white", linewidth=0.3)
    ax.text(0.92, 0.565, "no CoT teacher\nno clustering\nno guard", transform=ax.transAxes, ha="center", va="top", fontsize=5.2, color=COLORS["muted"])

    ax.text(0.018, 0.285, "One coherent trajectory objective", transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=COLORS["text"])
    ax.text(0.018, 0.245, "Position, direction, step magnitude and non-collapse\nmeasure one cumulative residual trajectory.", transform=ax.transAxes, fontsize=5.8, color=COLORS["muted"], va="top")
    ax.text(0.535, 0.285, "One rollout distribution, two guarded updates", transform=ax.transAxes, fontsize=7.0, fontweight="bold", color=COLORS["text"])
    ax.text(0.535, 0.245, "Answer/length GRPO and outcome-conditioned signatures\nshare a rollout distribution and guarded optimizer step.", transform=ax.transAxes, fontsize=5.8, color=COLORS["muted"], va="top")

    ax.plot([0.015, 0.985], [0.17, 0.17], transform=ax.transAxes, color=COLORS["grid"], linewidth=0.8)
    stats = [
        ("8", "latent slots", COLORS["blue"]),
        ("8", "rollouts / question", COLORS["green"]),
        ("0", "Stage2 inference parameters", COLORS["gold"]),
        ("62.9%", "gradient-conflict updates", COLORS["pink"]),
    ]
    for idx, (value, label, color) in enumerate(stats):
        x = 0.13 + idx * 0.245
        ax.text(x, 0.115, value, transform=ax.transAxes, ha="center", fontsize=9.0, fontweight="bold", color=color)
        ax.text(x, 0.078, label, transform=ax.transAxes, ha="center", fontsize=5.7, color=COLORS["muted"])

    save_figure(fig, out_dir / "fig1_trace_method_overview")
    plt.close(fig)


def published_rows() -> list[dict]:
    rows = []
    for row in PUBLISHED_RESULTS:
        item = {"method": row["method"], "family": row["family"]}
        for dataset in DATASETS:
            item[f"{dataset}_acc"], item[f"{dataset}_L"] = row[dataset]
        item["macro_acc"] = float(np.mean([row[d][0] for d in DATASETS]))
        item["macro_L"] = float(np.mean([row[d][1] for d in DATASETS]))
        rows.append(item)
    return rows


def draw_main_performance(paired: dict, out_dir: Path) -> None:
    rows = published_rows()
    compressed = [row for row in rows if row["method"] != "SFT-CoT"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.08, 0.92]})
    ax = axes[0]
    style_axis(ax)
    method_style = {
        "iCoT": (COLORS["muted"], "s"),
        "Coconut": (COLORS["muted"], "s"),
        "CODI": (COLORS["muted"], "s"),
        "CoLaR (r=5)": (COLORS["green"], "^"),
        "CoLaR (r=2)": (COLORS["blue"], "^"),
        "BRIDGE": (COLORS["gold"], "o"),
        "TRACE epoch7": (COLORS["pink"], "o"),
    }
    label_offsets = {
        "iCoT": (1.0, 1.5), "Coconut": (1.0, 1.0), "CODI": (1.0, -2.5),
        "CoLaR (r=5)": (1.0, 1.2), "CoLaR (r=2)": (-10.2, -3.5),
        "BRIDGE": (0.8, -4.0), "TRACE epoch7": (-13.3, 1.4),
    }
    for row in compressed:
        color, marker = method_style[row["method"]]
        size = 52 if row["method"].startswith("TRACE") else 38 if row["method"] == "BRIDGE" else 25
        ax.scatter(row["macro_L"], row["macro_acc"], s=size, marker=marker, color=color, edgecolor=COLORS["text"], linewidth=0.45, zorder=3)
        dx, dy = label_offsets[row["method"]]
        weight = "bold" if row["method"] in ("BRIDGE", "TRACE epoch7") else "normal"
        ax.text(row["macro_L"] + dx, row["macro_acc"] + dy, row["method"], fontsize=6.1, color=color, fontweight=weight)
    bridge = next(row for row in rows if row["method"] == "BRIDGE")
    trace = next(row for row in rows if row["method"] == "TRACE epoch7")
    ax.annotate("", xy=(trace["macro_L"], trace["macro_acc"]), xytext=(bridge["macro_L"], bridge["macro_acc"]), arrowprops={"arrowstyle": "-|>", "color": COLORS["pink"], "lw": 1.8, "mutation_scale": 10})
    ax.text(0.02, 0.965, "SFT-CoT reference: 80.20% / #L 110.20", transform=ax.transAxes, va="top", fontsize=5.8, color=COLORS["muted"])
    ax.set_xlim(-1.2, 45.5)
    ax.set_ylim(4, 72)
    ax.set_xlabel("Macro reasoning length #L (lower is better)")
    ax.set_ylabel("Macro accuracy (%)")
    ax.set_title("Literature accuracy-efficiency frontier", loc="left", fontweight="bold")
    panel_label(ax, "a")

    ax = axes[1]
    style_axis(ax)
    ax.axhline(0, color=COLORS["muted"], linewidth=0.8, linestyle="--")
    ax.axvline(0, color=COLORS["muted"], linewidth=0.8, linestyle="--")
    summaries = paired["summaries"]
    for row in summaries:
        dataset = row["dataset"]
        saved = -row["length_delta"]
        saved_low = -row["length_delta_bootstrap_ci95_high"]
        saved_high = -row["length_delta_bootstrap_ci95_low"]
        gain = row["accuracy_delta_pp"]
        gain_low = row["accuracy_delta_bootstrap_ci95_low"]
        gain_high = row["accuracy_delta_bootstrap_ci95_high"]
        ax.errorbar(
            saved, gain,
            xerr=[[saved - saved_low], [saved_high - saved]],
            yerr=[[gain - gain_low], [gain_high - gain]],
            fmt="o", ms=6.0, color=DATASET_COLORS[dataset], ecolor=DATASET_COLORS[dataset],
            elinewidth=1.1, capsize=2.2, markeredgecolor=COLORS["text"], markeredgewidth=0.45,
        )
        offsets = {"GSM8K": (0.10, 0.34), "GSMHard": (0.08, -0.56), "SVAMP": (-0.98, 0.33), "MultiArith": (0.10, 0.30)}
        dx, dy = offsets[dataset]
        ax.text(saved + dx, gain + dy, dataset, fontsize=6.2, color=DATASET_COLORS[dataset], fontweight="bold")
    ax.text(0.97, 0.04, "all datasets\nin the win-win region", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.1, color=COLORS["green"], fontweight="bold")
    ax.set_xlim(-0.15, 5.05)
    ax.set_ylim(-0.2, 7.15)
    ax.set_xlabel("Reasoning length saved, -Δ#L")
    ax.set_ylabel("Accuracy gain (percentage points)")
    ax.set_title("Controlled Stage1 → TRACE movement", loc="left", fontweight="bold")
    panel_label(ax, "b")
    fig.text(0.5, 0.015, "Points are paired dataset means; error bars are question-bootstrap 95% confidence intervals. MultiArith is ceiling limited.", ha="center", fontsize=5.8, color=COLORS["muted"])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.23, wspace=0.30)
    save_figure(fig, out_dir / "fig2_accuracy_efficiency_frontier")
    plt.close(fig)
    write_csv(out_dir / "source_data" / "fig2_published_results.csv", rows)
    write_csv(out_dir / "source_data" / "fig2_stage1_trace_paired.csv", summaries)


def transformed_metric(stat: dict, metric: str) -> tuple[float, float, float]:
    item = stat[metric]
    if metric == "assignment_progress_inversion_frac":
        return 1.0 - item["mean"], 1.0 - item["bootstrap_ci95_high"], 1.0 - item["bootstrap_ci95_low"]
    return item["mean"], item["bootstrap_ci95_low"], item["bootstrap_ci95_high"]


def draw_structure_to_outcome(trajectory: dict, story: dict, out_dir: Path, paired_transition: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.02, 1.04, 0.94]})
    ax = axes[0]
    style_axis(ax)
    profile_labels = ["BRIDGE", "Stage1", "TRACE epoch7", "TRACE seed1"]
    profile_colors = [COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["pink"]]
    slots = np.arange(1, 9)
    for label, color in zip(profile_labels, profile_colors):
        values = story["population_progress"][label]["mean_progress_profile"]
        ax.plot(slots, values, marker="o", markersize=3.5, linewidth=1.55, color=color, label=label)
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(0.16, 0.50)
    ax.set_xticks(slots)
    ax.set_xlabel("Latent slot")
    ax.set_ylabel("Expected CoT progress")
    ax.set_title("Ordered early-to-late progress", loc="left", fontweight="bold")
    ax.legend(loc="upper left", ncol=1, handlelength=1.4, borderaxespad=0.2)
    panel_label(ax, "a")

    ax = axes[1]
    style_axis(ax)
    metric_specs = [
        ("assignment_progress_span", "Progress span"),
        ("assignment_progress_inversion_frac", "Monotonicity"),
        ("diag_residual_cos", "Step alignment"),
        ("final_path_cos", "Final alignment"),
    ]
    methods = ["BRIDGE", "Stage1", "TRACE-epoch7"]
    colors = [COLORS["blue"], COLORS["green"], COLORS["gold"]]
    y = np.arange(len(metric_specs))[::-1]
    for metric_idx, (metric, _) in enumerate(metric_specs):
        yy = y[metric_idx]
        vals = []
        for method, color in zip(methods, colors):
            value, low, high = transformed_metric(trajectory["method_statistics"][method], metric)
            vals.append(value)
            ax.errorbar(value, yy, xerr=[[value - low], [high - value]], fmt="o", ms=4.5, color=color, ecolor=color, capsize=1.8, linewidth=0.9, markeredgecolor=COLORS["text"], markeredgewidth=0.35)
        ax.plot(vals, [yy] * len(vals), color=COLORS["grid"], linewidth=1.0, zorder=0)
    ax.set_yticks(y, [label for _, label in metric_specs])
    ax.set_xlim(-0.035, 1.02)
    ax.set_xlabel("Score (higher is better)")
    ax.set_title("Stage1 creates structure; Stage2 retains it", loc="left", fontweight="bold")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=4.2) for label, color in zip(("BRIDGE", "Stage1", "TRACE"), colors)]
    ax.legend(handles=handles, loc="lower right", ncol=1, handletextpad=0.3)
    panel_label(ax, "b")

    transition = story["transition"]
    category_colors = [COLORS["pink"], COLORS["gold"], COLORS["green"]]
    category_labels = ["All wrong", "Mixed", "All correct"]
    ax = axes[2]
    if paired_transition:
        matrix = np.asarray(transition["matrix"])
        backgrounds = np.empty((3, 3, 4))
        for row in range(3):
            for col in range(3):
                color = COLORS["blue"] if row == col else COLORS["green"] if col > row else COLORS["pink"]
                backgrounds[row, col] = matplotlib.colors.to_rgba(pale(color, 0.72))
        ax.imshow(backgrounds, interpolation="none", aspect="equal")
        for row in range(3):
            for col in range(3):
                direction = "stable" if row == col else "improve" if col > row else "regress"
                ax.text(col, row, f"{matrix[row, col]}\n{direction}", ha="center", va="center", fontsize=5.4, color=COLORS["text"], fontweight="bold")
        short_labels = ["All\nwrong", "Mixed", "All\ncorrect"]
        ax.set_xticks(range(3), short_labels)
        ax.set_yticks(range(3), short_labels)
        ax.tick_params(length=0, pad=2)
        ax.set_xlabel("TRACE rollout state")
        ax.set_ylabel("Stage1 rollout state")
        ax.set_title("43 improve vs 13 regress", loc="left", fontweight="bold")
        ax.text(1.0, -0.86, "8-path accuracy: 54.44% → 63.38%", ha="center", fontsize=5.8, color=COLORS["green"], fontweight="bold", clip_on=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        caption = "All panels use the same 200 questions and eight rollout views. Panel c is exact-question paired; upward state movement is improvement."
        out_name = "fig3_structure_to_outcome_paired"
    else:
        style_axis(ax)
        stage1_counts = np.asarray(transition["matrix"]).sum(axis=1)
        trace_counts = np.asarray(transition["matrix"]).sum(axis=0)
        bottoms = np.zeros(2)
        for idx, (label, color) in enumerate(zip(category_labels, category_colors)):
            values = np.asarray([stage1_counts[idx], trace_counts[idx]]) / 2.0
            ax.bar([0, 1], values, bottom=bottoms, width=0.55, color=color, edgecolor="white", linewidth=0.6, label=label)
            for x, value, bottom, count in zip([0, 1], values, bottoms, [stage1_counts[idx], trace_counts[idx]]):
                if value > 8:
                    ax.text(x, bottom + value / 2, f"{int(count)}", ha="center", va="center", fontsize=6.1, fontweight="bold", color=COLORS["text"])
            bottoms += values
        ax.set_xticks([0, 1], ["Stage1", "TRACE"])
        ax.set_xlim(-0.30, 1.72)
        ax.set_ylim(0, 116)
        ax.set_ylabel("Questions (%)")
        ax.set_title("More rollout sets become all-correct", loc="left", fontweight="bold")
        ax.text(0.5, 111.5, "8-path accuracy: 54.44% → 63.38%", ha="center", fontsize=6.2, color=COLORS["text"], fontweight="bold")
        ax.text(0.5, 106.0, "all-correct sets: 29% → 44%", ha="center", fontsize=6.1, color=COLORS["green"], fontweight="bold")
        category_midpoints = [stage1_counts[0] / 4, stage1_counts[0] / 2 + stage1_counts[1] / 4, 100 - stage1_counts[2] / 4]
        for label, color, yy in zip(category_labels, category_colors, category_midpoints):
            ax.scatter([1.34], [yy], s=20, marker="s", color=color, edgecolor="white", linewidth=0.35, clip_on=False)
            ax.text(1.43, yy, label, va="center", fontsize=5.7, color=COLORS["text"])
        caption = "All panels use the same 200 questions. Pink denotes the independent rollout seed in a and all-wrong groups in c."
        out_name = "fig3_structure_to_outcome"
    panel_label(ax, "c")
    fig.text(0.5, 0.018, caption, ha="center", fontsize=5.8, color=COLORS["muted"])
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.25, wspace=0.43)
    save_figure(fig, out_dir / out_name)
    plt.close(fig)

    rows = []
    for method in methods:
        for metric, label in metric_specs:
            value, low, high = transformed_metric(trajectory["method_statistics"][method], metric)
            rows.append({"method": method, "metric": metric, "label": label, "mean": value, "ci95_low": low, "ci95_high": high})
    write_csv(out_dir / "source_data" / "fig3_structure_metrics.csv", rows)
    write_csv(out_dir / "source_data" / "fig3_rollout_transition_matrix.csv", [
        {"stage1_state": r, "trace_all_wrong": int(transition["matrix"][i][0]), "trace_mixed": int(transition["matrix"][i][1]), "trace_all_correct": int(transition["matrix"][i][2])}
        for i, r in enumerate(category_labels)
    ])


def draw_optimization_fairness(training: dict, parity: dict, capacity: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.05, 1.0, 0.95]})
    validation = training["validation"]
    ax = axes[0]
    style_axis(ax)
    x = np.asarray([row["val/output_length"] + 8.0 for row in validation])
    y = 100 * np.asarray([row["val/acc"] for row in validation])
    ax.plot(x, y, color=COLORS["green"], linewidth=1.25, alpha=0.8)
    ax.scatter(x, y, c=[COLORS["blue"]] + [COLORS["green"]] * 6 + [COLORS["pink"]] + [COLORS["gold"]] * 2, s=27, edgecolor=COLORS["text"], linewidth=0.35, zorder=3)
    for epoch in (0, 1, 7, 9):
        ax.text(x[epoch] + 0.035, y[epoch] + (0.12 if epoch != 1 else -0.28), f"e{epoch}", fontsize=5.7, color=COLORS["text"])
    ax.scatter([x[7]], [y[7]], s=75, marker="*", color=COLORS["pink"], edgecolor=COLORS["text"], linewidth=0.5, zorder=4)
    ax.annotate("", xy=(x[7], y[7]), xytext=(x[0], y[0]), arrowprops={"arrowstyle": "-|>", "color": COLORS["pink"], "lw": 1.3, "mutation_scale": 9})
    ax.set_xlim(36.55, 39.65)
    ax.set_ylim(68.55, 72.15)
    ax.set_xlabel("Validation total reasoning length #L")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Epoch7 is validation-frozen", loc="left", fontweight="bold")
    panel_label(ax, "a")

    ax = axes[1]
    style_axis(ax)
    per_epoch = training["per_epoch_train"]
    epochs = np.asarray([row["epoch"] for row in per_epoch])
    conflict = np.asarray([row["train/stage2_guard_conflict"] for row in per_epoch])
    scale = np.asarray([row["train/stage2_guard_geometry_scale"] for row in per_epoch])
    ax.plot(epochs, conflict, marker="o", markersize=3.5, color=COLORS["pink"], linewidth=1.4, label="Conflict rate")
    ax.plot(epochs, scale, marker="s", markersize=3.2, color=COLORS["blue"], linewidth=1.4, label="Retained geometry scale")
    ax.axvline(7, color=COLORS["gold"], linestyle="--", linewidth=0.9)
    ax.set_ylim(0, 0.86)
    ax.set_xlim(-0.3, 9.3)
    ax.set_xticks([0, 2, 4, 6, 7, 8, 9])
    ax.set_xlabel("Stage2 epoch")
    ax.set_ylabel("Fraction / scale")
    ax.set_title("The accuracy-gradient guard is active", loc="left", fontweight="bold")
    ax.legend(loc="lower right")
    ax.text(0.02, 0.97, "full run: 62.9% conflict\nmean retained scale: 0.414", transform=ax.transAxes, va="top", fontsize=5.8, color=COLORS["text"])
    panel_label(ax, "b")

    ax = axes[2]
    ax.axis("off")
    ax.set_title("No Stage2 inference expansion", loc="left", fontweight="bold", pad=7)
    p1 = parity["stage1"]
    p2 = parity["stage2"]
    rows = [
        ("Saved parameters", f"{p1['state_dict_numel'] / 1e6:.3f}M", f"{p2['state_dict_numel'] / 1e6:.3f}M"),
        ("State-dict tensors", str(p1["state_dict_key_count"]), str(p2["state_dict_key_count"])),
        ("Latent slots", str(p1["n_latents"]), str(p2["n_latents"])),
        ("Answer-token budget", str(p1["hybrid_max_new_tokens"]), str(p2["hybrid_max_new_tokens"])),
        ("Added inference params", "-", "0"),
    ]
    ax.text(0.52, 0.88, "Stage1", ha="center", fontsize=6.4, fontweight="bold", color=COLORS["green"], transform=ax.transAxes)
    ax.text(0.82, 0.88, "TRACE", ha="center", fontsize=6.4, fontweight="bold", color=COLORS["pink"], transform=ax.transAxes)
    ax.plot([0.02, 0.98], [0.83, 0.83], transform=ax.transAxes, color=COLORS["text"], linewidth=0.8)
    for idx, (label, left, right) in enumerate(rows):
        yy = 0.74 - idx * 0.12
        if idx % 2 == 0:
            ax.add_patch(FancyBboxPatch((0.015, yy - 0.045), 0.965, 0.09, boxstyle="square,pad=0", transform=ax.transAxes, facecolor=COLORS["light"], edgecolor="none"))
        ax.text(0.03, yy, label, transform=ax.transAxes, va="center", fontsize=5.9, color=COLORS["text"])
        ax.text(0.52, yy, left, transform=ax.transAxes, ha="center", va="center", fontsize=6.1, color=COLORS["green"])
        ax.text(0.82, yy, right, transform=ax.transAxes, ha="center", va="center", fontsize=6.1, fontweight="bold", color=COLORS["pink"])
    bridge_numel = capacity["stage1"]["state_dict_numel"]
    stage1_numel = capacity["stage2"]["state_dict_numel"]
    delta_pct = 100 * (stage1_numel - bridge_numel) / bridge_numel
    ax.text(0.02, 0.04, f"BRIDGE → Stage1 adds 0.437% parameters; both retain 8 slots and a 48-token answer budget.\nStage1 vs TRACE has exact key, shape and dtype parity.", transform=ax.transAxes, fontsize=5.45, color=COLORS["muted"], va="bottom")
    assert abs(delta_pct - 0.4373) < 0.01
    panel_label(ax, "c")
    fig.text(0.5, 0.014, "Validation uses every epoch. The epoch7 tie is resolved by shorter validation #L; downstream test and geometry audits do not select the checkpoint.", ha="center", fontsize=5.7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.07, right=0.99, top=0.87, bottom=0.24, wspace=0.38)
    save_figure(fig, out_dir / "fig4_optimization_and_fairness")
    plt.close(fig)
    write_csv(out_dir / "source_data" / "fig4_validation.csv", validation)
    write_csv(out_dir / "source_data" / "fig4_training_guard.csv", per_epoch)


def questionwise_profiles(rows: pd.DataFrame) -> list[dict]:
    result = []
    for dataset in DATASETS:
        group = rows[rows["dataset"] == dataset]
        shorter = group["candidate_L"] < group["reference_L"]
        nonlong = group["candidate_L"] <= group["reference_L"]
        nonworse = group["candidate_acc"] >= group["reference_acc"]
        strict_pareto = ((group["candidate_acc"] > group["reference_acc"]) & nonlong) | (nonworse & shorter)
        dominated = (group["candidate_acc"] < group["reference_acc"]) & (group["candidate_L"] >= group["reference_L"])
        result.append({
            "dataset": dataset,
            "n": len(group),
            "accuracy_nonworse_pct": 100 * float(nonworse.mean()),
            "length_nonincreasing_pct": 100 * float(nonlong.mean()),
            "strict_pareto_improved_pct": 100 * float(strict_pareto.mean()),
            "fully_dominated_pct": 100 * float(dominated.mean()),
        })
    return result


def draw_questionwise_profile(question_rows: pd.DataFrame, out_dir: Path) -> None:
    profiles = questionwise_profiles(question_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.08, 0.92]})
    y = np.arange(len(DATASETS))[::-1]
    ax = axes[0]
    style_axis(ax)
    acc = np.asarray([next(row for row in profiles if row["dataset"] == d)["accuracy_nonworse_pct"] for d in DATASETS])
    length = np.asarray([next(row for row in profiles if row["dataset"] == d)["length_nonincreasing_pct"] for d in DATASETS])
    for yy, a, l in zip(y, acc, length):
        ax.plot([l, a], [yy, yy], color=COLORS["grid"], linewidth=2.2, zorder=1)
        ax.scatter([a], [yy], s=35, marker="o", color=COLORS["blue"], edgecolor=COLORS["text"], linewidth=0.4, zorder=3)
        ax.scatter([l], [yy], s=31, marker="s", color=COLORS["pink"], edgecolor=COLORS["text"], linewidth=0.4, zorder=3)
    for yy, a, l in zip(y, acc, length):
        ax.text(a + 0.35, yy + 0.12, f"{a:.1f}%", ha="left", va="center", fontsize=5.7, color=COLORS["blue"], fontweight="bold")
        ax.text(l - 0.35, yy - 0.16, f"{l:.1f}%", ha="right", va="center", fontsize=5.5, color=COLORS["pink"], fontweight="bold")
    ax.set_yticks(y, DATASETS)
    ax.set_xlim(86.0, 102.0)
    ax.set_ylim(-0.72, 3.25)
    ax.set_xlabel("Paired test questions (%)")
    ax.set_title("Question-wise non-degradation profile", loc="left", fontweight="bold")
    ax.text(86.3, -0.50, "● accuracy non-worse", fontsize=5.7, color=COLORS["blue"], va="center")
    ax.text(94.0, -0.50, "■ #L non-increasing", fontsize=5.7, color=COLORS["pink"], va="center")
    panel_label(ax, "a")

    ax = axes[1]
    style_axis(ax)
    improved = np.asarray([next(row for row in profiles if row["dataset"] == d)["strict_pareto_improved_pct"] for d in DATASETS])
    dominated = np.asarray([next(row for row in profiles if row["dataset"] == d)["fully_dominated_pct"] for d in DATASETS])
    for idx, dataset in enumerate(DATASETS):
        color = DATASET_COLORS[dataset]
        ax.plot([dominated[idx], improved[idx]], [y[idx], y[idx]], color=pale(color, 0.50), linewidth=2.0)
        ax.scatter([improved[idx]], [y[idx]], s=38, color=color, edgecolor=COLORS["text"], linewidth=0.4, zorder=3)
        ax.scatter([dominated[idx]], [y[idx]], s=25, color="white", edgecolor=color, linewidth=1.1, zorder=3)
        ax.text(improved[idx] + 0.8, y[idx], f"{improved[idx]:.1f}%", va="center", fontsize=5.8, color=color, fontweight="bold")
        ax.text(dominated[idx] + 0.5, y[idx] - 0.23, f"{dominated[idx]:.1f}%", va="center", fontsize=5.2, color=COLORS["muted"])
    ax.set_yticks(y, DATASETS)
    ax.set_xlim(-0.8, 42.5)
    ax.set_ylim(-0.72, 3.25)
    ax.set_xlabel("Paired test questions (%)")
    ax.set_title("Strict Pareto gains dominate losses", loc="left", fontweight="bold")
    ax.text(0.2, -0.50, "● strict Pareto improved", fontsize=5.7, color=COLORS["muted"], va="center")
    ax.text(24.0, -0.50, "○ fully dominated", fontsize=5.7, color=COLORS["muted"], va="center")
    panel_label(ax, "b")
    fig.text(0.5, 0.015, "Strict Pareto improvement means higher accuracy with non-increasing #L, or unchanged accuracy with shorter #L. Accuracy is binary per question.", ha="center", fontsize=5.7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.085, right=0.99, top=0.87, bottom=0.29, wspace=0.36)
    save_figure(fig, out_dir / "fig5_questionwise_pareto_profile")
    plt.close(fig)
    write_csv(out_dir / "source_data" / "fig5_questionwise_profile.csv", profiles)


def normalized_assignment_matrix(records: dict[int, dict], indices: list[int], bins: int = 8) -> np.ndarray:
    matrices = []
    for idx in indices:
        assignment = records[idx]["assignment"].float().cpu().numpy()
        n_steps = assignment.shape[1]
        remapped = np.zeros((assignment.shape[0], bins), dtype=np.float64)
        for step_idx, position in enumerate(np.linspace(0.0, bins - 1.0, n_steps)):
            lower = int(np.floor(position))
            upper = min(bins - 1, lower + 1)
            fraction = position - lower
            remapped[:, lower] += assignment[:, step_idx] * (1.0 - fraction)
            remapped[:, upper] += assignment[:, step_idx] * fraction
        matrices.append(remapped)
    return np.mean(matrices, axis=0)


def draw_population_assignment_maps(story: dict, out_dir: Path) -> None:
    labels = ["BRIDGE", "Stage1", "TRACE epoch7"]
    record_sets = {
        label: {int(record["idx"]): record for record in torch.load(story["record_paths"][label], map_location="cpu", weights_only=False)}
        for label in labels
    }
    common = sorted(set.intersection(*(set(records) for records in record_sets.values())))
    eligible = [
        idx for idx in common
        if all(len(record_sets[label][idx]["steps"]) >= 3 for label in labels)
    ]
    matrices = {label: normalized_assignment_matrix(record_sets[label], eligible) for label in labels}
    progress_bins = np.linspace(0.0, 1.0, 8)
    profiles = {label: matrix @ progress_bins for label, matrix in matrices.items()}
    vmax = max(float(matrix.max()) for matrix in matrices.values())
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "trace_assignment",
        [COLORS["white"], pale(COLORS["blue"], 0.62), pale(COLORS["green"], 0.38), COLORS["gold"]],
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.50), sharey=True)
    method_colors = [COLORS["blue"], COLORS["green"], COLORS["gold"]]
    image = None
    for panel_idx, (ax, label, color) in enumerate(zip(axes, labels, method_colors)):
        matrix = matrices[label]
        image = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap=cmap, origin="upper", aspect="auto", interpolation="nearest")
        expected_bins = profiles[label] * 7.0
        ax.plot(expected_bins, np.arange(8), color=COLORS["pink"], linewidth=1.65, marker="o", markersize=2.8, markeredgecolor="white", markeredgewidth=0.45)
        span = float(profiles[label][-1] - profiles[label][0])
        ax.set_title(f"{label}\nprogress span = {span:.3f}", color=color, fontweight="bold", pad=4)
        ax.set_xticks([0, 3.5, 7], ["early", "mid", "late"])
        ax.set_xlabel("Normalized CoT progress")
        ax.set_yticks(np.arange(8), [f"z{i}" for i in range(1, 9)])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(ax, chr(ord("a") + panel_idx))
    axes[0].set_ylabel("Latent slot")
    fig.subplots_adjust(left=0.075, right=0.885, top=0.84, bottom=0.25, wspace=0.16)
    colorbar_ax = fig.add_axes([0.915, 0.25, 0.012, 0.59])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Mean assignment mass")
    colorbar.outline.set_linewidth(0.55)
    fig.text(0.5, 0.018, f"Population average over all {len(eligible)} shared questions with at least three human CoT steps. Pink lines show row-wise expected progress; CoT is used only for evaluation.", ha="center", fontsize=5.7, color=COLORS["muted"])
    save_figure(fig, out_dir / "fig7_population_assignment_maps")
    plt.close(fig)

    matrix_rows = []
    profile_rows = []
    for label in labels:
        for slot in range(8):
            profile_rows.append({"method": label, "latent_slot": slot + 1, "expected_progress": float(profiles[label][slot])})
            for progress_bin in range(8):
                matrix_rows.append({"method": label, "latent_slot": slot + 1, "progress_bin": progress_bin + 1, "mean_assignment_mass": float(matrices[label][slot, progress_bin]), "n_questions": len(eligible)})
    write_csv(out_dir / "source_data" / "fig7_population_assignment_matrix.csv", matrix_rows)
    write_csv(out_dir / "source_data" / "fig7_population_assignment_profile.csv", profile_rows)


def question_gain_summaries(question_rows: pd.DataFrame) -> list[dict]:
    summaries = []
    for dataset in DATASETS:
        rows = question_rows[question_rows["dataset"] == dataset]
        rescued = rows[rows["transition"] == "rescued"]
        regressed = rows[rows["transition"] == "regressed"]
        summaries.append({
            "dataset": dataset,
            "n": int(len(rows)),
            "rescued": int(len(rescued)),
            "regressed": int(len(regressed)),
            "rescued_pct": 100.0 * len(rescued) / len(rows),
            "regressed_pct": 100.0 * len(regressed) / len(rows),
            "net_accuracy_gain_pp": 100.0 * (len(rescued) - len(regressed)) / len(rows),
            "rescue_to_regress_ratio": float(len(rescued) / len(regressed)) if len(regressed) else None,
            "rescues_nonincreasing_L_pct": 100.0 * float((rescued["delta_L"] <= 0).mean()) if len(rescued) else None,
            "rescues_shorter_L_pct": 100.0 * float((rescued["delta_L"] < 0).mean()) if len(rescued) else None,
            "rescues_median_delta_L": float(rescued["delta_L"].median()) if len(rescued) else None,
            "all_nonincreasing_L_pct": 100.0 * float((rows["delta_L"] <= 0).mean()),
        })
    return summaries


def draw_question_level_gain_anatomy(question_rows: pd.DataFrame, out_dir: Path) -> None:
    summaries = question_gain_summaries(question_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [0.96, 1.12]})
    y = np.arange(len(DATASETS))[::-1]

    ax = axes[0]
    style_axis(ax)
    for row_idx, (dataset, yy) in enumerate(zip(DATASETS, y)):
        summary = next(item for item in summaries if item["dataset"] == dataset)
        ax.barh(yy, -summary["regressed_pct"], height=0.46, color=COLORS["pink"], edgecolor="white", linewidth=0.45)
        ax.barh(yy, summary["rescued_pct"], height=0.46, color=COLORS["green"], edgecolor="white", linewidth=0.45)
        if summary["regressed_pct"] >= 2.0:
            ax.text(-summary["regressed_pct"] / 2.0, yy, f"{summary['regressed']} ({summary['regressed_pct']:.1f}%)", ha="center", va="center", fontsize=5.6, color=COLORS["text"], fontweight="bold")
        else:
            ax.text(-summary["regressed_pct"] - 0.18, yy, f"{summary['regressed']} ({summary['regressed_pct']:.1f}%)", ha="right", va="center", fontsize=5.6, color=COLORS["pink"], fontweight="bold")
        ax.text(summary["rescued_pct"] + 0.18, yy, f"{summary['rescued']} ({summary['rescued_pct']:.1f}%)", ha="left", va="center", fontsize=5.6, color=COLORS["green"], fontweight="bold")
        ax.text(9.95, yy - 0.28, f"net +{summary['net_accuracy_gain_pp']:.2f} pp", ha="right", va="center", fontsize=5.1, color=DATASET_COLORS[dataset])
    ax.axvline(0.0, color=COLORS["blue"], linewidth=0.9, linestyle="--")
    ax.set_yticks(y, DATASETS)
    ax.set_xlim(-5.8, 10.4)
    ax.set_xlabel("Paired questions (%)")
    ax.set_title("Rescues consistently exceed regressions", loc="left", fontweight="bold")
    ax.text(-5.6, 3.42, "regressed", color=COLORS["pink"], fontsize=5.7, fontweight="bold", ha="left")
    ax.text(10.2, 3.42, "rescued", color=COLORS["green"], fontsize=5.7, fontweight="bold", ha="right")
    panel_label(ax, "a")

    ax = axes[1]
    style_axis(ax)
    ax.axvspan(-34, 0, color=pale(COLORS["green"], 0.93), zorder=0)
    ax.axvspan(0, 20, color=pale(COLORS["pink"], 0.94), zorder=0)
    ax.axvline(0.0, color=COLORS["muted"], linewidth=0.9, linestyle="--")
    rng = np.random.default_rng(20260714)
    rescued_rows = []
    for dataset, yy in zip(DATASETS, y):
        values = question_rows[(question_rows["dataset"] == dataset) & (question_rows["transition"] == "rescued")]["delta_L"].to_numpy(dtype=float)
        color = DATASET_COLORS[dataset]
        jitter = rng.uniform(-0.16, 0.16, len(values))
        ax.scatter(values, yy + jitter, s=8, color=color, alpha=0.28, edgecolor="none", zorder=2)
        if len(values) > 1:
            q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
            ax.plot([q05, q95], [yy, yy], color=color, linewidth=1.0, zorder=3)
            ax.plot([q25, q75], [yy, yy], color=color, linewidth=4.0, solid_capstyle="butt", zorder=3)
        else:
            median = float(values[0])
        ax.scatter([median], [yy], s=34, color=color, edgecolor=COLORS["text"], linewidth=0.45, zorder=4)
        nonincrease = 100.0 * float((values <= 0).mean())
        ax.text(19.4, yy, f"{nonincrease:.0f}% ≤ 0", ha="right", va="center", fontsize=5.5, color=color, fontweight="bold")
        for value in values:
            rescued_rows.append({"dataset": dataset, "delta_L": float(value)})
    ax.set_yticks(y, DATASETS)
    ax.set_xlim(-34, 20)
    ax.set_xlabel("Rescued-question Δ#L (TRACE − Stage1)")
    ax.set_title("Most rescues require no extra length", loc="left", fontweight="bold")
    ax.text(-33.2, 3.42, "shorter", color=COLORS["green"], fontsize=5.7, fontweight="bold", ha="left")
    ax.text(19.2, 3.42, "longer", color=COLORS["pink"], fontsize=5.7, fontweight="bold", ha="right")
    panel_label(ax, "b")
    fig.text(0.5, 0.018, "Exact-question paired Stage1-to-TRACE evaluation. Negative Δ#L means a shorter TRACE answer; dots show every rescued question and intervals show 5th–95th and 25th–75th percentiles.", ha="center", fontsize=5.7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.25, wspace=0.32)
    save_figure(fig, out_dir / "fig8_question_level_gain_anatomy")
    plt.close(fig)
    write_csv(out_dir / "source_data" / "fig8_question_gain_summary.csv", summaries)
    write_csv(out_dir / "source_data" / "fig8_rescued_length_rows.csv", rescued_rows)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def draw_rollout_consensus_reliability(story: dict, out_dir: Path) -> None:
    labels = ["Stage1", "TRACE epoch7"]
    record_sets = {
        label: {int(record["idx"]): record for record in torch.load(story["record_paths"][label], map_location="cpu", weights_only=False)}
        for label in labels
    }
    common = sorted(set(record_sets[labels[0]]) & set(record_sets[labels[1]]))
    rows = []
    for label in labels:
        for idx in common:
            record = record_sets[label][idx]
            correct_paths = int((record["multiview_acc"].float().cpu().numpy() > 0.5).sum())
            rows.append({"method": label, "idx": idx, "correct_rollouts": correct_paths, "main_correct": int(float(record["acc"]) > 0.5)})
    frame = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.0, 1.06]})
    x = np.arange(9)
    method_colors = {"Stage1": COLORS["green"], "TRACE epoch7": COLORS["gold"]}
    distribution_rows = []
    ax = axes[0]
    style_axis(ax)
    for label in labels:
        subset = frame[frame["method"] == label]
        counts = subset["correct_rollouts"].value_counts().reindex(x, fill_value=0).to_numpy()
        ax.plot(x, counts, marker="o" if label == "Stage1" else "s", markersize=4.2, linewidth=1.45, color=method_colors[label], label=label)
        for correct_rollouts, count in zip(x, counts):
            distribution_rows.append({"method": label, "correct_rollouts": int(correct_rollouts), "question_count": int(count), "n_questions": len(common)})
    ax.annotate("+30 all-correct sets", xy=(8, 88), xytext=(5.25, 93), fontsize=5.9, color=COLORS["pink"], fontweight="bold", arrowprops={"arrowstyle": "-|>", "color": COLORS["pink"], "lw": 1.0, "mutation_scale": 8})
    ax.set_xticks(x)
    ax.set_xlim(-0.25, 8.35)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Correct rollouts per question (of 8)")
    ax.set_ylabel("Questions")
    ax.set_title("Stage2 shifts mass toward 8/8 correct", loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    panel_label(ax, "a")

    ax = axes[1]
    style_axis(ax)
    reliability_rows = []
    gap_rows = []
    method_arrays = {}
    y_positions = {"Stage1": 1.0, "TRACE epoch7": 0.0}
    for label in labels:
        subset = frame[frame["method"] == label].set_index("idx").loc[common]
        correct_rollouts = subset["correct_rollouts"].to_numpy(dtype=int)
        main_correct = subset["main_correct"].to_numpy(dtype=int)
        method_arrays[label] = (correct_rollouts, main_correct)
        group_values = []
        for group_label, mask, marker, category_color in (
            ("low (0–2)", correct_rollouts <= 2, "s", COLORS["pink"]),
            ("high (6–8)", correct_rollouts >= 6, "o", COLORS["green"]),
        ):
            total = int(mask.sum())
            successes = int(main_correct[mask].sum())
            mean = successes / total
            low, high = wilson_interval(successes, total)
            mean_pct, low_pct, high_pct = 100.0 * mean, 100.0 * low, 100.0 * high
            group_values.append(mean_pct)
            ax.errorbar(mean_pct, y_positions[label], xerr=[[max(0.0, mean_pct - low_pct)], [max(0.0, high_pct - mean_pct)]], fmt=marker, ms=6.2, color=category_color, ecolor=category_color, capsize=2.2, linewidth=1.1, markeredgecolor=COLORS["text"], markeredgewidth=0.55, zorder=4)
            ax.text(mean_pct, y_positions[label] - 0.22, f"n={total}", ha="center", va="top", fontsize=5.0, color=COLORS["muted"])
            reliability_rows.append({"method": label, "consensus_group": group_label, "n_questions": total, "main_correct": successes, "main_accuracy_pct": mean_pct, "wilson95_low_pct": low_pct, "wilson95_high_pct": high_pct})
        gap = group_values[1] - group_values[0]
        ax.plot(group_values, [y_positions[label], y_positions[label]], color=method_colors[label], linewidth=2.0, zorder=2)
        ax.text(np.mean(group_values), y_positions[label] + 0.20, f"gap {gap:.1f} pp", ha="center", va="bottom", fontsize=5.7, color=method_colors[label], fontweight="bold")
        gap_rows.append({"method": label, "reliability_gap_pp": gap})

    rng = np.random.default_rng(20260714)
    bootstrap_deltas = []
    for _ in range(10000):
        sampled = rng.integers(0, len(common), len(common))
        gaps = {}
        for label in labels:
            correct_rollouts, main_correct = method_arrays[label]
            sampled_rollouts = correct_rollouts[sampled]
            sampled_correct = main_correct[sampled]
            gaps[label] = 100.0 * (sampled_correct[sampled_rollouts >= 6].mean() - sampled_correct[sampled_rollouts <= 2].mean())
        bootstrap_deltas.append(gaps["TRACE epoch7"] - gaps["Stage1"])
    observed_delta = gap_rows[1]["reliability_gap_pp"] - gap_rows[0]["reliability_gap_pp"]
    delta_low, delta_high = np.quantile(bootstrap_deltas, [0.025, 0.975])
    gap_rows.append({"method": "TRACE − Stage1", "reliability_gap_pp": observed_delta, "bootstrap95_low_pp": float(delta_low), "bootstrap95_high_pp": float(delta_high), "bootstrap_samples": 10000})
    ax.set_yticks([1.0, 0.0], ["Stage1", "TRACE epoch7"])
    ax.set_xlim(-3, 106)
    ax.set_ylim(-0.52, 1.48)
    ax.set_xlabel("Main-answer accuracy (%)")
    ax.set_title("Consensus becomes more diagnostic", loc="left", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=COLORS["pink"], markeredgecolor=COLORS["text"], label="low consensus (0–2)"),
        Line2D([0], [0], marker="o", linestyle="", color=COLORS["green"], markeredgecolor=COLORS["text"], label="high consensus (6–8)"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=0.8)
    panel_label(ax, "b")
    fig.text(0.5, 0.018, f"Same {len(common)} GSM8K questions and eight fixed rollout views. Reliability-gap increase: +{observed_delta:.1f} pp (paired-question bootstrap 95% CI [{delta_low:.1f}, {delta_high:.1f}]); association is diagnostic, not causal.", ha="center", fontsize=5.7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.25, wspace=0.30)
    save_figure(fig, out_dir / "fig9_rollout_consensus_reliability")
    plt.close(fig)
    write_csv(out_dir / "source_data" / "fig9_rollout_count_distribution.csv", distribution_rows)
    write_csv(out_dir / "source_data" / "fig9_rollout_reliability.csv", reliability_rows)
    write_csv(out_dir / "source_data" / "fig9_rollout_reliability_gap.csv", gap_rows)


def draw_booktabs_table(
    out_path: Path,
    title: str,
    subtitle: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    trace_row: int | None = None,
    bridge_row: int | None = None,
    bold_cells: set[tuple[int, int]] | None = None,
    highlight_columns: dict[int, str] | None = None,
    footnote: str = "",
    height: float = 3.0,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.965, title, fontsize=9.2, fontweight="bold", color=COLORS["text"], va="top")
    ax.text(0.01, 0.912, subtitle, fontsize=5.8, color=COLORS["muted"], va="top")
    left, right = 0.012, 0.988
    widths = np.asarray(widths, dtype=float)
    widths = widths / widths.sum() * (right - left)
    centers = left + np.cumsum(widths) - widths / 2
    edges = np.concatenate([[left], left + np.cumsum(widths)])
    header_y = 0.835
    row_top = 0.785
    available = row_top - 0.145
    row_h = available / len(rows)
    ax.plot([left, right], [0.875, 0.875], color=COLORS["text"], linewidth=1.0)
    ax.plot([left, right], [0.805, 0.805], color=COLORS["text"], linewidth=0.75)
    for col, header in enumerate(headers):
        ha = "left" if col == 0 else "center"
        x = edges[col] + 0.006 if col == 0 else centers[col]
        header_color = highlight_columns.get(col, COLORS["text"]) if highlight_columns else COLORS["text"]
        ax.text(x, header_y, header, ha=ha, va="center", fontsize=6.1, fontweight="bold", color=header_color)
    for ridx, row in enumerate(rows):
        y_center = row_top - (ridx + 0.5) * row_h
        y_bottom = row_top - (ridx + 1) * row_h
        if ridx == trace_row:
            ax.add_patch(FancyBboxPatch((left, y_bottom), right - left, row_h, boxstyle="square,pad=0", facecolor=pale(COLORS["pink"], 0.78), edgecolor="none"))
            ax.add_patch(FancyBboxPatch((left, y_bottom), 0.0045, row_h, boxstyle="square,pad=0", facecolor=COLORS["pink"], edgecolor="none"))
        elif ridx == bridge_row:
            ax.add_patch(FancyBboxPatch((left, y_bottom), right - left, row_h, boxstyle="square,pad=0", facecolor=pale(COLORS["gold"], 0.84), edgecolor="none"))
        elif ridx % 2 == 1:
            ax.add_patch(FancyBboxPatch((left, y_bottom), right - left, row_h, boxstyle="square,pad=0", facecolor=COLORS["light"], edgecolor="none"))
        if highlight_columns:
            for col, color in highlight_columns.items():
                ax.add_patch(FancyBboxPatch((edges[col], y_bottom), widths[col], row_h, boxstyle="square,pad=0", facecolor=pale(color, 0.88), edgecolor="none"))
        for col, value in enumerate(row):
            ha = "left" if col == 0 else "center"
            x = edges[col] + 0.006 if col == 0 else centers[col]
            bold = bold_cells is not None and (ridx, col) in bold_cells
            ax.text(x, y_center, value, ha=ha, va="center", fontsize=5.9, color=COLORS["text"], fontweight="bold" if bold else "normal")
    bottom = row_top - len(rows) * row_h
    ax.plot([left, right], [bottom, bottom], color=COLORS["text"], linewidth=1.0)
    ax.text(0.01, 0.055, footnote, fontsize=5.45, color=COLORS["muted"], va="bottom")
    save_figure(fig, out_path)
    plt.close(fig)


def draw_main_results_table(out_dir: Path) -> None:
    source = published_rows()
    rows = []
    for item in source:
        rows.append([
            item["method"],
            f"{item['GSM8K_acc']:.2f} / {item['GSM8K_L']:.2f}",
            f"{item['GSMHard_acc']:.2f} / {item['GSMHard_L']:.2f}",
            f"{item['SVAMP_acc']:.2f} / {item['SVAMP_L']:.2f}",
            f"{item['MultiArith_acc']:.2f} / {item['MultiArith_L']:.2f}",
            f"{item['macro_acc']:.2f} / {item['macro_L']:.2f}",
        ])
    trace_row = next(i for i, item in enumerate(source) if item["method"] == "TRACE epoch7")
    bridge_row = next(i for i, item in enumerate(source) if item["method"] == "BRIDGE")
    bold = {(trace_row, c) for c in (0, 1, 2, 3, 5)} | {(bridge_row, 4)}
    draw_booktabs_table(
        out_dir / "table1_main_results",
        "Table 1 | Main results on in-domain and out-of-distribution arithmetic reasoning",
        "Cells report Accuracy (%) / total reasoning length #L. Higher accuracy and lower #L are preferred.",
        ["Method", "GSM8K", "GSMHard", "SVAMP", "MultiArith", "Macro"],
        rows,
        [1.45, 1.25, 1.25, 1.25, 1.25, 1.15],
        trace_row=trace_row,
        bridge_row=bridge_row,
        bold_cells=bold,
        footnote="Published baselines are five-seed means from the BRIDGE evaluation protocol. TRACE epoch7 is one validation-frozen checkpoint; training-seed replication remains required for the final submission.",
        height=3.15,
    )
    write_csv(out_dir / "source_data" / "table1_main_results.csv", source)


def pvalue_text(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def draw_attribution_table(paired: dict, out_dir: Path) -> None:
    rows = []
    for item in paired["summaries"]:
        rows.append([
            item["dataset"],
            str(item["n"]),
            f"{item['reference_acc']:.2f} / {item['reference_L']:.2f}",
            f"{item['candidate_acc']:.2f} / {item['candidate_L']:.2f}",
            f"+{item['accuracy_delta_pp']:.2f} [{item['accuracy_delta_bootstrap_ci95_low']:.2f}, {item['accuracy_delta_bootstrap_ci95_high']:.2f}]",
            f"{item['length_delta']:.2f} [{item['length_delta_bootstrap_ci95_low']:.2f}, {item['length_delta_bootstrap_ci95_high']:.2f}]",
            f"{item['rescued']} / {item['regressed']}",
            pvalue_text(item["mcnemar_exact_p"]),
        ])
    bold = {(r, c) for r in range(len(rows)) for c in (3, 4, 5)}
    draw_booktabs_table(
        out_dir / "table2_stage2_paired_attribution",
        "Table 2 | Controlled Stage2 attribution from the architecture-identical Stage1 checkpoint",
        "Every comparison is exact-question paired. Accuracy intervals and #L intervals are 95% question-bootstrap confidence intervals.",
        ["Dataset", "n", "Stage1 Acc/#L", "TRACE Acc/#L", "ΔAcc [95% CI]", "Δ#L [95% CI]", "Rescue/regress", "Exact p"],
        rows,
        [0.82, 0.42, 1.02, 1.02, 1.42, 1.42, 0.86, 0.62],
        trace_row=None,
        bridge_row=None,
        bold_cells=bold,
        highlight_columns={3: COLORS["pink"]},
        footnote="Exact p is the two-sided McNemar test. MultiArith is ceiling limited (176/178 questions are correct under both checkpoints); its one rescue is not significance evidence.",
        height=2.60,
    )
    write_csv(out_dir / "source_data" / "table2_stage2_paired_attribution.csv", paired["summaries"])


def draw_stagewise_mechanism_table(trajectory: dict, story: dict, parity: dict, capacity: dict, out_dir: Path) -> None:
    method_stats = trajectory["method_statistics"]
    transition = story["transition"]
    bridge_params = capacity["stage1"]["state_dict_numel"] / 1e6
    stage1_params = parity["stage1"]["state_dict_numel"] / 1e6
    trace_params = parity["stage2"]["state_dict_numel"] / 1e6
    specs = [
        ("BRIDGE", bridge_params, "BRIDGE", None),
        ("TRACE Stage1", stage1_params, "Stage1", transition["stage1_rollout_accuracy"]),
        ("TRACE epoch7", trace_params, "TRACE-epoch7", transition["trace_rollout_accuracy"]),
    ]
    source = []
    rows = []
    for label, params_m, stat_key, rollout_acc in specs:
        span, _, _ = transformed_metric(method_stats[stat_key], "assignment_progress_span")
        monotonicity, _, _ = transformed_metric(method_stats[stat_key], "assignment_progress_inversion_frac")
        step_alignment, _, _ = transformed_metric(method_stats[stat_key], "diag_residual_cos")
        final_alignment, _, _ = transformed_metric(method_stats[stat_key], "final_path_cos")
        source.append({
            "method": label,
            "saved_params_m": params_m,
            "latent_slots": 8,
            "answer_token_budget": 48,
            "progress_span": span,
            "monotonicity": monotonicity,
            "step_alignment": step_alignment,
            "final_alignment": final_alignment,
            "eight_path_rollout_accuracy": rollout_acc,
        })
        rows.append([
            label,
            f"{params_m:.3f}",
            "8",
            "48",
            f"{span:.3f}",
            f"{monotonicity:.3f}",
            f"{step_alignment:.3f}",
            f"{final_alignment:.3f}",
            "—" if rollout_acc is None else f"{rollout_acc:.2f}%",
        ])
    bold = {(1, c) for c in (4, 5, 6, 7)} | {(2, 8)}
    draw_booktabs_table(
        out_dir / "table3_stagewise_mechanism_audit",
        "Table 3 | Stagewise structural and capacity audit",
        "Stage1 constructs ordered latent progress; Stage2 improves rollout success without expanding the inference graph.",
        ["Method", "Params (M)", "Slots", "Budget", "Progress span", "Monotonicity", "Step align.", "Final align.", "8-path acc."],
        rows,
        [1.12, 0.70, 0.46, 0.52, 0.82, 0.82, 0.78, 0.78, 0.80],
        trace_row=2,
        bridge_row=0,
        bold_cells=bold,
        footnote="Structure metrics use the same 200 GSM8K questions; monotonicity is 1 − inversion fraction. Stage1 and TRACE have exact state-key, shape, dtype, slot and answer-budget parity.",
        height=2.35,
    )
    write_csv(out_dir / "source_data" / "table3_stagewise_mechanism_audit.csv", source)
    latex_rows = []
    for row in rows:
        method = r"\textbf{TRACE epoch7}" if row[0] == "TRACE epoch7" else row[0]
        latex_rows.append(method + " & " + " & ".join(row[1:]) + r" \\")
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Stagewise structural and capacity audit. Structure metrics use the same 200 GSM8K questions; monotonicity is one minus the assignment inversion fraction.}",
        r"\label{tab:trace-stagewise}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Params (M) & Slots & Budget & Progress span & Monotonicity & Step align. & Final align. & 8-path acc. \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (out_dir / "table3_stagewise_mechanism_audit.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def draw_question_gain_table(question_rows: pd.DataFrame, out_dir: Path) -> None:
    summaries = question_gain_summaries(question_rows)
    rows = []
    for item in summaries:
        ratio = "Inf*" if item["rescue_to_regress_ratio"] is None else f"{item['rescue_to_regress_ratio']:.2f}x"
        rows.append([
            item["dataset"],
            str(item["n"]),
            f"{item['rescued']} ({item['rescued_pct']:.1f}%)",
            f"{item['regressed']} ({item['regressed_pct']:.1f}%)",
            f"+{item['net_accuracy_gain_pp']:.2f}",
            ratio,
            f"{item['rescues_nonincreasing_L_pct']:.1f}%",
            f"{item['rescues_median_delta_L']:.1f}",
            f"{item['all_nonincreasing_L_pct']:.1f}%",
        ])
    bold = {(row_idx, col) for row_idx in range(len(rows)) for col in (2, 4, 6)}
    draw_booktabs_table(
        out_dir / "table4_question_level_gain_anatomy",
        "Table 4 | Question-level anatomy of Stage2 gains",
        "Rescues outnumber regressions on every dataset, and most rescued answers use no additional reasoning length.",
        ["Dataset", "n", "Rescued", "Regressed", "Net ΔAcc", "R/R ratio", "Rescue #L≤", "Rescue med. Δ#L", "All #L≤"],
        rows,
        [0.78, 0.42, 0.88, 0.88, 0.70, 0.68, 0.80, 0.96, 0.70],
        bold_cells=bold,
        highlight_columns={2: COLORS["green"], 3: COLORS["pink"]},
        footnote="Exact-question paired results. #L≤ reports non-increasing TRACE length. *MultiArith has one rescue and zero regressions at a ceiling-limited baseline; its ratio is not significance evidence.",
        height=2.55,
    )
    write_csv(out_dir / "source_data" / "table4_question_level_gain_anatomy.csv", summaries)
    latex_rows = []
    for row in rows:
        latex_rows.append(" & ".join(row) + r" \\")
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Question-level anatomy of the paired Stage1-to-TRACE gains. Most rescued answers have non-increasing reasoning length.}",
        r"\label{tab:trace-question-anatomy}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Dataset & $n$ & Rescued & Regressed & Net $\Delta$Acc & R/R ratio & Rescue $\#L\leq$ & Rescue med. $\Delta\#L$ & All $\#L\leq$ \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (out_dir / "table4_question_level_gain_anatomy.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def write_latex_tables(out_dir: Path, paired: dict) -> None:
    table1_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main arithmetic-reasoning results. Cells report accuracy (\%)/total reasoning length $\#L$. Published baselines are five-seed means; TRACE is the single validation-frozen epoch7 checkpoint.}",
        r"\label{tab:trace-main}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & GSM8K & GSMHard & SVAMP & MultiArith & Avg. Acc & Avg. $\#L$ \\",
        r"\midrule",
    ]
    for row in published_rows():
        cells = [f"{row[d + '_acc']:.2f}/{row[d + '_L']:.2f}" for d in DATASETS]
        method = r"\textbf{TRACE epoch7}" if row["method"] == "TRACE epoch7" else row["method"]
        table1_lines.append(f"{method} & " + " & ".join(cells) + f" & {row['macro_acc']:.2f} & {row['macro_L']:.2f} " + r"\\")
    table1_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (out_dir / "table1_main_results.tex").write_text("\n".join(table1_lines) + "\n", encoding="utf-8")

    table2_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Exact-question paired Stage1-to-TRACE attribution. Intervals are question-bootstrap 95\% confidence intervals; $p$ is the two-sided exact McNemar test.}",
        r"\label{tab:trace-stage2}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Stage1 Acc/$\#L$ & TRACE Acc/$\#L$ & $\Delta$Acc & $\Delta\#L$ & Rescue/regress & $p$ \\",
        r"\midrule",
    ]
    for row in paired["summaries"]:
        table2_lines.append(
            f"{row['dataset']} & {row['reference_acc']:.2f}/{row['reference_L']:.2f} & {row['candidate_acc']:.2f}/{row['candidate_L']:.2f} & "
            f"+{row['accuracy_delta_pp']:.2f} [{row['accuracy_delta_bootstrap_ci95_low']:.2f}, {row['accuracy_delta_bootstrap_ci95_high']:.2f}] & "
            f"{row['length_delta']:.2f} [{row['length_delta_bootstrap_ci95_low']:.2f}, {row['length_delta_bootstrap_ci95_high']:.2f}] & "
            f"{row['rescued']}/{row['regressed']} & {pvalue_text(row['mcnemar_exact_p'])} " + r"\\"
        )
    table2_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (out_dir / "table2_stage2_paired_attribution.tex").write_text("\n".join(table2_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.evidence_root
    out_dir = args.out_dir or root / "paper_main_candidates_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    paired = json.loads((root / "paired_stage1_vs_epoch7" / "paired_evidence.json").read_text(encoding="utf-8"))
    trajectory = json.loads((root / "trajectory_structure_200" / "trajectory_structure.json").read_text(encoding="utf-8"))
    training = json.loads((root / "training_dynamics" / "training_dynamics.json").read_text(encoding="utf-8"))
    story = json.loads((root / "story_evidence" / "story_evidence_manifest.json").read_text(encoding="utf-8"))
    parity = json.loads((root / "checkpoint_parity" / "checkpoint_parity.json").read_text(encoding="utf-8"))
    capacity = json.loads((root / "bridge_to_stage1_capacity" / "checkpoint_parity.json").read_text(encoding="utf-8"))
    question_rows = pd.read_csv(root / "paired_stage1_vs_epoch7" / "paired_question_rows.csv")

    draw_method_overview(out_dir)
    draw_main_performance(paired, out_dir)
    draw_structure_to_outcome(trajectory, story, out_dir)
    draw_structure_to_outcome(trajectory, story, out_dir, paired_transition=True)
    draw_optimization_fairness(training, parity, capacity, out_dir)
    draw_questionwise_profile(question_rows, out_dir)
    draw_population_assignment_maps(story, out_dir)
    draw_question_level_gain_anatomy(question_rows, out_dir)
    draw_rollout_consensus_reliability(story, out_dir)
    draw_main_results_table(out_dir)
    draw_attribution_table(paired, out_dir)
    draw_stagewise_mechanism_table(trajectory, story, parity, capacity, out_dir)
    draw_question_gain_table(question_rows, out_dir)
    write_latex_tables(out_dir, paired)

    qualitative_source = root / "story_evidence" / "global_pca_rescue_cases" / "trace_exploded_paths_3d_objective"
    for suffix in (".png", ".pdf", ".svg", ".tiff"):
        shutil.copy2(qualitative_source.with_suffix(suffix), out_dir / f"fig6_trace_path_atlas{suffix}")
    shutil.copy2(qualitative_source.with_suffix(".json"), out_dir / "source_data" / "fig6_trace_path_atlas.json")

    manifest = {
        "frozen_checkpoint_sha256": story["frozen_checkpoint_sha256"],
        "source_root": str(root),
        "figures": [
            "fig1_trace_method_overview",
            "fig2_accuracy_efficiency_frontier",
            "fig3_structure_to_outcome",
            "fig3_structure_to_outcome_paired",
            "fig4_optimization_and_fairness",
            "fig5_questionwise_pareto_profile",
            "fig6_trace_path_atlas",
            "fig7_population_assignment_maps",
            "fig8_question_level_gain_anatomy",
            "fig9_rollout_consensus_reliability",
            "table1_main_results",
            "table2_stage2_paired_attribution",
            "table3_stagewise_mechanism_audit",
            "table4_question_level_gain_anatomy",
        ],
        "palette": COLORS,
        "font_stack": FONT_STACK,
        "recommended_main": ["fig1_trace_method_overview", "table1_main_results", "fig2_accuracy_efficiency_frontier", "fig3_structure_to_outcome_paired", "table2_stage2_paired_attribution"],
        "recommended_first_supplement": ["fig9_rollout_consensus_reliability", "fig7_population_assignment_maps", "fig8_question_level_gain_anatomy", "fig4_optimization_and_fairness", "table3_stagewise_mechanism_audit", "table4_question_level_gain_anatomy"],
        "claim_boundary": "Main claims are accuracy-efficiency, Stage1 trajectory structure, Stage2 paired task improvement, rollout-consistency and reliability shifts, question-level rescue dominance, and no Stage2 inference expansion. Outcome-specific geometry remains supplementary and non-population-level.",
    }
    (out_dir / "paper_main_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "figures": manifest["figures"]}, indent=2))


if __name__ == "__main__":
    main()
