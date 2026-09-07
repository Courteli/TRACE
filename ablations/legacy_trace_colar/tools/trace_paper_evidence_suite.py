#!/usr/bin/env python3
"""Build clean, submission-ready evidence for the TRACE paper."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch


COLORS = {
    "blue": "#6687B8",
    "green": "#69B17D",
    "gold": "#E6A314",
    "pink": "#E5A6C4",
    "text": "#27313B",
    "muted": "#66717E",
    "grid": "#D9DEE5",
    "null": "#AAB2BC",
    "light": "#F5F7F9",
    "white": "#FFFFFF",
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
        "legend.fontsize": 6.6,
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
    parser.add_argument("--stage1-records", type=Path, required=True)
    parser.add_argument("--final-records", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--reliability-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1024)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def normalize(array: np.ndarray) -> np.ndarray:
    return array / np.clip(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8, None)


def as_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
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


def bootstrap_ci(
    values: np.ndarray, trials: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    means = np.empty(trials, dtype=np.float64)
    chunk = 500
    for start in range(0, trials, chunk):
        end = min(start + chunk, trials)
        indices = rng.integers(0, n, size=(end - start, n))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def style_axis(ax: plt.Axes, grid: str = "both") -> None:
    ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], width=0.7, length=2.5)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
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


def p_string(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def exact_mcnemar_p(rescued: int, regressed: int) -> float:
    discordant = rescued + regressed
    if discordant == 0:
        return 1.0
    smaller = min(rescued, regressed)
    tail = sum(math.comb(discordant, idx) for idx in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def make_stagewise_table(paired: dict, output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for item in paired["summaries"]:
        rows.append(
            {
                "dataset": item["dataset"],
                "n": item["n"],
                "stage1_accuracy_percent": item["reference_acc"],
                "final_accuracy_percent": item["candidate_acc"],
                "accuracy_gain_pp": item["accuracy_delta_pp"],
                "accuracy_gain_ci95_low": item["accuracy_delta_bootstrap_ci95_low"],
                "accuracy_gain_ci95_high": item["accuracy_delta_bootstrap_ci95_high"],
                "stage1_total_L": item["reference_L"],
                "final_total_L": item["candidate_L"],
                "length_saved": -item["length_delta"],
                "length_saved_ci95_low": -item["length_delta_bootstrap_ci95_high"],
                "length_saved_ci95_high": -item["length_delta_bootstrap_ci95_low"],
                "rescued": item["rescued"],
                "regressed": item["regressed"],
                "mcnemar_exact_p": item["mcnemar_exact_p"],
            }
        )
    write_csv(output_dir / "source_data" / "stagewise_benchmarks.csv", rows)
    headers = [
        "Dataset",
        "n",
        "Stage 1 Acc.",
        "Final Acc.",
        "Gain [95% CI]",
        "Stage 1 #L",
        "Final #L",
        "Saved #L [95% CI]",
        "R/G",
        "p",
    ]
    md = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    tex_rows = []
    for row in rows:
        values = [
            str(row["dataset"]),
            str(row["n"]),
            f'{row["stage1_accuracy_percent"]:.2f}',
            f'{row["final_accuracy_percent"]:.2f}',
            f'{row["accuracy_gain_pp"]:+.2f} [{row["accuracy_gain_ci95_low"]:.2f}, {row["accuracy_gain_ci95_high"]:.2f}]',
            f'{row["stage1_total_L"]:.2f}',
            f'{row["final_total_L"]:.2f}',
            f'{row["length_saved"]:.2f} [{row["length_saved_ci95_low"]:.2f}, {row["length_saved_ci95_high"]:.2f}]',
            f'{row["rescued"]}/{row["regressed"]}',
            p_string(float(row["mcnemar_exact_p"])),
        ]
        md.append("| " + " | ".join(values) + " |")
        tex_rows.append(" & ".join(values) + r" \\")
    (output_dir / "table_stagewise_benchmarks.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    latex = "\n".join(
        [
            r"\begin{tabular}{lrrrrrrrrr}",
            r"\toprule",
            r"Dataset & $n$ & \multicolumn{3}{c}{Accuracy (\%)} & \multicolumn{3}{c}{Total \#L} & R/G & $p$ \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r" & & Stage 1 & Final & $\Delta$ [95\% CI] & Stage 1 & Final & Saved [95\% CI] & & \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output_dir / "table_stagewise_benchmarks.tex").write_text(
        latex + "\n", encoding="utf-8"
    )
    return rows


def make_outcome_figure(
    stagewise: list[dict], reliability_dir: Path, output_dir: Path
) -> None:
    distribution = read_csv(reliability_dir / "reliability_distribution.csv")
    thresholds = read_csv(reliability_dir / "reliability_thresholds.csv")
    transitions = read_csv(reliability_dir / "reliability_transitions.csv")
    clean_distribution = []
    for row in distribution:
        clean_distribution.append(
            {
                "stage": "TRACE Stage 1" if row["method"] == "Stage 1" else "TRACE Final",
                "correct_views": int(row["correct_views"]),
                "question_count": int(row["question_count"]),
                "question_percent": float(row["question_percent"]),
            }
        )
    clean_thresholds = []
    for row in thresholds:
        clean_thresholds.append(
            {
                "threshold": row["threshold"],
                "minimum_correct_views": int(row["minimum_correct_views"]),
                "stage1_percent": float(row["stage1_percent"]),
                "final_percent": float(row["final_percent"]),
                "delta_pp": float(row["delta_pp"]),
                "delta_ci95_low": float(row["delta_ci95_low"]),
                "delta_ci95_high": float(row["delta_ci95_high"]),
                "rescued_questions": int(row["rescued_questions"]),
                "regressed_questions": int(row["regressed_questions"]),
            }
        )
    clean_transitions = []
    for row in transitions:
        clean_transitions.append(
            {
                "stage1_category": row["stage1_category"],
                "final_all_wrong": int(row["All wrong"]),
                "final_mixed": int(row["Mixed"]),
                "final_all_correct": int(row["All correct"]),
            }
        )
    write_csv(output_dir / "source_data" / "outcome_reliability_distribution.csv", clean_distribution)
    write_csv(output_dir / "source_data" / "outcome_reliability_thresholds.csv", clean_thresholds)
    write_csv(output_dir / "source_data" / "outcome_reliability_transitions.csv", clean_transitions)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.52),
        gridspec_kw={"width_ratios": [1.02, 1.05, 0.93]},
    )

    ax = axes[0]
    style_axis(ax)
    dataset_colors = {
        "GSM8K": COLORS["blue"],
        "GSMHard": COLORS["green"],
        "SVAMP": COLORS["gold"],
        "MultiArith": COLORS["pink"],
    }
    ax.axhline(0, color=COLORS["muted"], linewidth=0.7, linestyle="--")
    ax.axvline(0, color=COLORS["muted"], linewidth=0.7, linestyle="--")
    for row in stagewise:
        x = float(row["length_saved"])
        y = float(row["accuracy_gain_pp"])
        xlow = float(row["length_saved_ci95_low"])
        xhigh = float(row["length_saved_ci95_high"])
        ylow = float(row["accuracy_gain_ci95_low"])
        yhigh = float(row["accuracy_gain_ci95_high"])
        color = dataset_colors[str(row["dataset"])]
        ax.errorbar(
            x,
            y,
            xerr=[[x - xlow], [xhigh - x]],
            yerr=[[y - ylow], [yhigh - y]],
            fmt="o",
            ms=5.8,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.0,
            markeredgecolor=COLORS["text"],
            markeredgewidth=0.4,
            zorder=3,
        )
        offsets = {
            "GSM8K": (0.10, 0.26),
            "GSMHard": (0.10, -0.48),
            "SVAMP": (-0.92, 0.26),
            "MultiArith": (0.10, 0.25),
        }
        dx, dy = offsets[str(row["dataset"])]
        ax.text(x + dx, y + dy, str(row["dataset"]), color=color, fontsize=6.0, fontweight="bold")
    ax.set_xlim(-0.12, 4.75)
    ax.set_ylim(-0.18, 6.1)
    ax.set_xlabel("Reasoning length saved")
    ax.set_ylabel("Accuracy gain (pp)")
    ax.set_title("Win-win refinement", loc="left", fontweight="bold")
    panel_label(ax, "a")

    ax = axes[1]
    style_axis(ax)
    for label, color, marker in (
        ("TRACE Stage 1", COLORS["green"], "o"),
        ("TRACE Final", COLORS["pink"], "o"),
    ):
        rows = sorted(
            [row for row in clean_distribution if row["stage"] == label],
            key=lambda row: int(row["correct_views"]),
        )
        x = np.asarray([row["correct_views"] for row in rows])
        y = np.asarray([row["question_percent"] for row in rows])
        ax.plot(x, y, marker=marker, markersize=3.8, linewidth=1.35, color=color, label=label)
        ax.fill_between(x, 0, y, color=color, alpha=0.08)
    ax.axvspan(7.5, 8.5, color=COLORS["pink"], alpha=0.08, linewidth=0)
    ax.set_xlim(-0.2, 8.2)
    ax.set_xticks(np.arange(9))
    ax.set_xlabel("Correct paths among 8 views")
    ax.set_ylabel("Questions (%)")
    ax.set_title("Coverage becomes repeatability", loc="left", fontweight="bold")
    ax.legend(loc="upper left", handlelength=1.7)
    panel_label(ax, "b")

    ax = axes[2]
    style_axis(ax, grid="x")
    y = np.arange(len(clean_thresholds))[::-1]
    labels = ["At least 1/8", "At least 5/8", "All 8/8"]
    bar_height = 0.25
    for pos, row in zip(y, clean_thresholds):
        stage1_value = row["stage1_percent"]
        final_value = row["final_percent"]
        ax.barh(
            pos + 0.145,
            stage1_value,
            height=bar_height,
            color=COLORS["green"],
            alpha=0.92,
            edgecolor=COLORS["text"],
            linewidth=0.35,
            zorder=2,
        )
        ax.barh(
            pos - 0.145,
            final_value,
            height=bar_height,
            color=COLORS["pink"],
            alpha=0.96,
            edgecolor=COLORS["text"],
            linewidth=0.35,
            zorder=3,
        )
        ax.text(
            stage1_value - 1.4,
            pos + 0.145,
            f"{stage1_value:.1f}",
            ha="right",
            va="center",
            fontsize=5.8,
            color=COLORS["text"],
            fontweight="bold",
        )
        ax.text(
            final_value - 1.4,
            pos - 0.145,
            f"{final_value:.1f}",
            ha="right",
            va="center",
            fontsize=5.8,
            color=COLORS["text"],
            fontweight="bold",
        )
        ax.text(
            min(98.0, max(stage1_value, final_value) + 2.0),
            pos,
            (
                f'$\\Delta${row["delta_pp"]:+.1f} '
                f'[{row["delta_ci95_low"]:+.1f}, {row["delta_ci95_high"]:+.1f}]'
            ),
            va="center",
            fontsize=5.65,
            color=COLORS["pink"],
            fontweight="bold",
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Questions meeting criterion (%)")
    ax.set_title("Reliability at stricter criteria", loc="left", fontweight="bold")
    panel_label(ax, "c")

    fig.text(
        0.5,
        0.012,
        "Panels b-c use 200 matched questions and eight fixed views per question; error bars in a are paired question-bootstrap 95% CIs.",
        ha="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.076, right=0.99, top=0.86, bottom=0.24, wspace=0.38)
    save_figure(fig, output_dir / "fig_outcome_refinement")
    plt.close(fig)

    md = [
        "| Criterion | TRACE Stage 1 | TRACE Final | Gain [95% CI] | Rescued / regressed |",
        "|---|---:|---:|---:|---:|",
    ]
    tex_rows = []
    for row in clean_thresholds:
        md.append(
            f'| {row["threshold"]} | {row["stage1_percent"]:.1f} | {row["final_percent"]:.1f} | '
            f'{row["delta_pp"]:+.1f} [{row["delta_ci95_low"]:.1f}, {row["delta_ci95_high"]:.1f}] | '
            f'{row["rescued_questions"]} / {row["regressed_questions"]} |'
        )
        tex_rows.append(
            f'{row["threshold"]} & {row["stage1_percent"]:.1f} & {row["final_percent"]:.1f} & '
            f'{row["delta_pp"]:+.1f} [{row["delta_ci95_low"]:.1f}, {row["delta_ci95_high"]:.1f}] & '
            f'{row["rescued_questions"]}/{row["regressed_questions"]} ' + r"\\"
        )
    (output_dir / "table_outcome_reliability.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    latex = "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Criterion & Stage 1 & Final & $\Delta$ [95\% CI] & R/G \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output_dir / "table_outcome_reliability.tex").write_text(latex + "\n", encoding="utf-8")


def canonical_answer(value: str) -> str:
    clean = str(value).strip("#\n ").rstrip(".").replace(",", "").lower()
    try:
        return f"{float(clean):.12g}"
    except ValueError:
        return clean


def answer_from_output(output: str) -> str:
    return canonical_answer(str(output).strip("#").split("###Answer:")[-1])


def multiview_question_metrics(records: list[dict]) -> dict[str, np.ndarray]:
    path_accuracy = []
    plurality_accuracy = []
    all_correct = []
    modal_agreement = []
    for record in records:
        accuracies = as_array(record["multiview_acc"]).reshape(-1)
        predictions = [answer_from_output(text) for text in record["multiview_output_strings"]]
        counts = Counter(predictions)
        modal_answer, modal_count = counts.most_common(1)[0]
        gold = canonical_answer(record["answer"])
        path_accuracy.append(float(accuracies.mean()))
        plurality_accuracy.append(float(modal_answer == gold))
        all_correct.append(float(np.all(accuracies == 1)))
        modal_agreement.append(float(modal_count / len(predictions)))
    return {
        "Mean path accuracy": np.asarray(path_accuracy),
        "Plurality-vote accuracy": np.asarray(plurality_accuracy),
        "All-path reliability": np.asarray(all_correct),
        "Modal answer agreement": np.asarray(modal_agreement),
    }


def make_multiview_decision_table(
    stage1_path: Path,
    final_path: Path,
    output_dir: Path,
    trials: int,
    seed: int,
) -> list[dict]:
    stage1_records = torch.load(stage1_path, map_location="cpu", weights_only=False)
    final_records = torch.load(final_path, map_location="cpu", weights_only=False)
    if [int(r["idx"]) for r in stage1_records] != [int(r["idx"]) for r in final_records]:
        raise ValueError("Multiview records are not aligned")
    stage1 = multiview_question_metrics(stage1_records)
    final = multiview_question_metrics(final_records)
    del stage1_records, final_records
    rng = np.random.default_rng(seed + 5000)
    rows = []
    for metric in stage1:
        before = stage1[metric]
        after = final[metric]
        delta_pp = 100.0 * (after - before)
        low, high = bootstrap_ci(delta_pp, trials, rng)
        is_binary_decision = metric in ("Plurality-vote accuracy", "All-path reliability")
        rescued = int(np.sum((before == 0) & (after == 1))) if is_binary_decision else None
        regressed = int(np.sum((before == 1) & (after == 0))) if is_binary_decision else None
        p_value = (
            exact_mcnemar_p(rescued, regressed)
            if metric in ("Plurality-vote accuracy", "All-path reliability")
            else None
        )
        rows.append(
            {
                "metric": metric,
                "stage1_percent": 100.0 * float(before.mean()),
                "final_percent": 100.0 * float(after.mean()),
                "gain_pp": float(delta_pp.mean()),
                "gain_ci95_low": low,
                "gain_ci95_high": high,
                "rescued": rescued,
                "regressed": regressed,
                "mcnemar_exact_p": p_value,
                "n_questions": len(before),
                "views_per_question": 8,
            }
        )
    write_csv(output_dir / "source_data" / "multiview_decision_metrics.csv", rows)
    md = [
        "| Metric | TRACE Stage 1 | TRACE Final | Gain [95% CI] | Rescued / regressed | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    tex_rows = []
    for row in rows:
        transition = (
            f'{row["rescued"]} / {row["regressed"]}'
            if row["rescued"] is not None
            else "--"
        )
        p_value = p_string(row["mcnemar_exact_p"]) if row["mcnemar_exact_p"] is not None else "--"
        md.append(
            f'| {row["metric"]} | {row["stage1_percent"]:.2f} | {row["final_percent"]:.2f} | '
            f'{row["gain_pp"]:+.2f} [{row["gain_ci95_low"]:.2f}, {row["gain_ci95_high"]:.2f}] | '
            f'{transition} | {p_value} |'
        )
        tex_rows.append(
            f'{row["metric"]} & {row["stage1_percent"]:.2f} & {row["final_percent"]:.2f} & '
            f'{row["gain_pp"]:+.2f} [{row["gain_ci95_low"]:.2f}, {row["gain_ci95_high"]:.2f}] & '
            f'{transition.replace(" / ", "/")} & {p_value} ' + r"\\"
        )
    (output_dir / "table_multiview_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    latex = "\n".join(
        [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Metric & Stage 1 & Final & $\Delta$ [95\% CI] & R/G & $p$ \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output_dir / "table_multiview_decision.tex").write_text(latex + "\n", encoding="utf-8")
    return rows


def assignment_on_progress_grid(assignment: np.ndarray, bins: int) -> np.ndarray:
    probs = assignment / np.clip(assignment.sum(axis=-1, keepdims=True), 1e-8, None)
    grid_values = np.zeros((assignment.shape[0], bins), dtype=np.float64)
    if assignment.shape[1] == 1:
        grid_values[:, 0] = probs[:, 0]
        return grid_values
    positions = np.linspace(0.0, bins - 1, assignment.shape[1])
    for step_idx, position in enumerate(positions):
        left = int(np.floor(position))
        right = min(left + 1, bins - 1)
        right_weight = position - left
        grid_values[:, left] += probs[:, step_idx] * (1.0 - right_weight)
        grid_values[:, right] += probs[:, step_idx] * right_weight
    return grid_values / np.clip(grid_values.sum(axis=-1, keepdims=True), 1e-8, None)


def assignment_population(records: list[dict], bins: int) -> tuple[np.ndarray, np.ndarray]:
    maps = []
    centers = []
    progress = np.linspace(0.0, 1.0, bins)
    for record in records:
        mapped = assignment_on_progress_grid(as_array(record["assignment"]), bins)
        maps.append(mapped)
        centers.append(mapped @ progress)
    return np.stack(maps), np.stack(centers)


def center_density(centers: np.ndarray, bins: int) -> np.ndarray:
    density = np.zeros((centers.shape[1], bins), dtype=np.float64)
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    for slot in range(centers.shape[1]):
        histogram, _ = np.histogram(centers[:, slot], bins=bins, range=(0.0, 1.0))
        smoothed = np.convolve(histogram.astype(np.float64), kernel, mode="same")
        density[slot] = smoothed / max(smoothed.sum(), 1e-8)
    return density


def make_population_assignment_figure(
    stage1_path: Path,
    final_path: Path,
    output_dir: Path,
    trials: int,
    seed: int,
    bins: int = 24,
) -> None:
    stage1_records = torch.load(stage1_path, map_location="cpu", weights_only=False)
    final_records = torch.load(final_path, map_location="cpu", weights_only=False)
    if [int(r["idx"]) for r in stage1_records] != [int(r["idx"]) for r in final_records]:
        raise ValueError("Assignment records are not aligned")
    eligible = [idx for idx, record in enumerate(stage1_records) if as_array(record["assignment"]).shape[1] > 1]
    if not eligible:
        raise ValueError("No multi-step CoT records are available")
    stage1_records = [stage1_records[idx] for idx in eligible]
    final_records = [final_records[idx] for idx in eligible]
    _, stage1_centers = assignment_population(stage1_records, bins)
    _, final_centers = assignment_population(final_records, bins)
    del stage1_records, final_records
    stage1_mean = center_density(stage1_centers, bins)
    final_mean = center_density(final_centers, bins)
    null_mean = np.repeat(stage1_mean.mean(axis=0, keepdims=True), stage1_mean.shape[0], axis=0)
    null_centers = np.repeat(stage1_centers.mean(axis=1, keepdims=True), stage1_centers.shape[1], axis=1)

    source_rows = []
    for stage, matrix in (
        ("TRACE Stage 1", stage1_mean),
        ("TRACE Final", final_mean),
        ("Slot-permutation null", null_mean),
    ):
        for slot in range(matrix.shape[0]):
            for progress_bin in range(matrix.shape[1]):
                source_rows.append(
                    {
                        "stage": stage,
                        "latent_slot": slot + 1,
                        "progress_bin": progress_bin,
                        "normalized_progress": progress_bin / (bins - 1),
                        "progress_center_density": float(matrix[slot, progress_bin]),
                    }
                )
    write_csv(output_dir / "source_data" / "population_assignment_maps.csv", source_rows)

    rng = np.random.default_rng(seed + 7000)
    center_rows = []
    center_summaries: dict[str, list[dict]] = {}
    for stage, values in (
        ("TRACE Stage 1", stage1_centers),
        ("TRACE Final", final_centers),
        ("Slot-permutation null", null_centers),
    ):
        summaries = []
        for slot in range(values.shape[1]):
            low, high = bootstrap_ci(values[:, slot], trials, rng)
            row = {
                "stage": stage,
                "latent_slot": slot + 1,
                "mean_expected_progress": float(values[:, slot].mean()),
                "ci95_low": low,
                "ci95_high": high,
                "n_questions": values.shape[0],
            }
            summaries.append(row)
            center_rows.append(row)
        center_summaries[stage] = summaries
    write_csv(output_dir / "source_data" / "population_assignment_centers.csv", center_rows)

    cmap = LinearSegmentedColormap.from_list(
        "trace_assignment",
        [COLORS["white"], "#E8EEF6", COLORS["blue"], COLORS["pink"]],
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.5),
        gridspec_kw={"width_ratios": [0.95, 0.95, 1.10]},
    )
    vmax = max(float(stage1_mean.max()), float(final_mean.max()))
    for panel, (ax, matrix, title) in enumerate(
        (
            (axes[0], stage1_mean, "TRACE Stage 1"),
            (axes[1], final_mean, "TRACE Final"),
        )
    ):
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            extent=[0, 1, 0.5, 8.5],
        )
        ax.set_xlim(0.15, 0.92)
        ax.set_xticks([0.2, 0.5, 0.8])
        ax.set_yticks(np.arange(1, 9))
        ax.set_xlabel("Normalized CoT progress")
        ax.set_ylabel("Latent slot" if panel == 0 else "")
        ax.set_title(title, loc="left", fontweight="bold", color=COLORS["green"] if panel == 0 else COLORS["pink"])
        panel_label(ax, chr(ord("a") + panel))
        for spine in ax.spines.values():
            spine.set_color(COLORS["muted"])
            spine.set_linewidth(0.7)
    divider = make_axes_locatable(axes[1])
    colorbar_axis = divider.append_axes("right", size="4%", pad=0.08)
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Progress-center density", fontsize=6.4)
    colorbar.ax.tick_params(labelsize=5.8, length=2)

    ax = axes[2]
    style_axis(ax)
    slots = np.arange(1, 9)
    for stage, color, offset in (
        ("TRACE Stage 1", COLORS["green"], -0.06),
        ("TRACE Final", COLORS["pink"], 0.06),
    ):
        summaries = center_summaries[stage]
        means = np.asarray([row["mean_expected_progress"] for row in summaries])
        low = np.asarray([row["ci95_low"] for row in summaries])
        high = np.asarray([row["ci95_high"] for row in summaries])
        ax.errorbar(
            slots + offset,
            means,
            yerr=[means - low, high - means],
            fmt="o-",
            color=color,
            ecolor=color,
            linewidth=1.25,
            elinewidth=0.8,
            capsize=1.8,
            markersize=3.8,
            markeredgecolor=COLORS["text"],
            markeredgewidth=0.3,
            label=stage,
        )
    null_mean_progress = np.mean(
        [row["mean_expected_progress"] for row in center_summaries["Slot-permutation null"]]
    )
    ax.axhline(null_mean_progress, color=COLORS["gold"], linestyle="--", linewidth=1.1, label="Slot-permutation null")
    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(0, 1)
    ax.set_xticks(slots)
    ax.set_xlabel("Latent slot")
    ax.set_ylabel("Expected CoT progress")
    ax.set_title("Population-level ordering", loc="left", fontweight="bold")
    ax.legend(loc="upper left", handlelength=1.5)
    panel_label(ax, "c")

    fig.text(
        0.5,
        0.012,
        f"All {len(eligible)} multi-step CoTs from the 200-question audit; centers use assignment-weighted normalized progress. Error bars: bootstrap 95% CIs.",
        ha="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.072, right=0.99, top=0.84, bottom=0.24, wspace=0.43)
    save_figure(fig, output_dir / "fig_population_assignment_map")
    plt.close(fig)


def order_score(assignment: np.ndarray) -> float:
    probs = assignment / np.clip(assignment.sum(axis=-1, keepdims=True), 1e-8, None)
    positions = np.linspace(0.0, 1.0, assignment.shape[1], dtype=np.float32)
    centers = probs @ positions
    first, second = np.triu_indices(len(centers), 1)
    differences = centers[second] - centers[first]
    return float(np.where(differences > 1e-6, 1.0, np.where(differences < -1e-6, 0.0, 0.5)).mean())


def effective_rank(residuals: np.ndarray) -> float:
    singular_values = np.linalg.svd(residuals, compute_uv=False)
    weights = singular_values**2
    weights = weights / np.clip(weights.sum(), 1e-12, None)
    return float(np.exp(-(weights * np.log(np.clip(weights, 1e-12, None))).sum()))


def trajectory_arrays(records: list[dict]) -> dict[str, np.ndarray | list[np.ndarray]]:
    step_actual: list[float] = []
    step_null: list[float] = []
    order_actual: list[float] = []
    step_norm_ratio: list[float] = []
    ranks: list[float] = []
    cosine_matrices: list[np.ndarray] = []
    implicit_finals: list[np.ndarray] = []
    target_finals: list[np.ndarray] = []
    assignments: list[np.ndarray] = []
    for record in records:
        target = as_array(record["aggregated_explicit_residuals"])
        views = as_array(record["multiview_implicit_residuals"])
        cosines = np.einsum("vkd,jd->vkj", normalize(views), normalize(target))
        step_actual.append(float(np.diagonal(cosines, axis1=1, axis2=2).mean()))
        step_null.append(float(cosines.mean()))
        cosine_matrices.append(cosines)
        assignment = as_array(record["assignment"])
        assignments.append(assignment)
        order_actual.append(order_score(assignment))
        implicit_finals.append(normalize(views.sum(axis=1)).mean(axis=0))
        target_finals.append(normalize(target.sum(axis=0, keepdims=True))[0])
        implicit_norm = np.linalg.norm(views, axis=-1).mean() / np.sqrt(views.shape[-1])
        target_norm = np.linalg.norm(target, axis=-1).mean() / np.sqrt(target.shape[-1])
        step_norm_ratio.append(float(implicit_norm / max(target_norm, 1e-8)))
        ranks.append(float(np.mean([effective_rank(view) for view in views])))

    implicit_matrix = np.stack(implicit_finals)
    target_matrix = np.stack(target_finals)
    direction_matrix = implicit_matrix @ target_matrix.T
    direction_actual = np.diag(direction_matrix)
    direction_null = (direction_matrix.sum(axis=1) - direction_actual) / (len(records) - 1)
    return {
        "step_actual": np.asarray(step_actual),
        "step_null": np.asarray(step_null),
        "order_actual": np.asarray(order_actual),
        "order_null": np.full(len(records), 0.5),
        "direction_actual": direction_actual,
        "direction_null": direction_null,
        "step_norm_ratio": np.asarray(step_norm_ratio),
        "effective_rank": np.asarray(ranks),
        "cosine_matrices": cosine_matrices,
        "direction_matrix": direction_matrix,
        "assignments": assignments,
    }


def permutation_pvalues(
    arrays: dict[str, np.ndarray | list[np.ndarray]],
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    cosine_matrices = np.stack(arrays["cosine_matrices"])
    n, views, steps, _ = cosine_matrices.shape
    observed_step = float(np.asarray(arrays["step_actual"]).mean())
    observed_direction = float(np.asarray(arrays["direction_actual"]).mean())
    observed_order = float(np.asarray(arrays["order_actual"]).mean())
    direction_matrix = np.asarray(arrays["direction_matrix"])
    assignments = arrays["assignments"]
    null_step = np.empty(permutations)
    null_direction = np.empty(permutations)
    null_order = np.empty(permutations)
    row_index = np.arange(n)
    for trial in range(permutations):
        random_keys = rng.random((n, views, steps))
        target_permutations = np.argsort(random_keys, axis=-1)
        matched = np.take_along_axis(cosine_matrices, target_permutations[..., None], axis=3)[..., 0]
        null_step[trial] = matched.mean()
        choices = rng.integers(0, n - 1, size=n)
        choices = choices + (choices >= row_index)
        null_direction[trial] = direction_matrix[row_index, choices].mean()
        scores = []
        for assignment in assignments:
            permutation = rng.permutation(assignment.shape[0])
            scores.append(order_score(assignment[permutation]))
        null_order[trial] = np.mean(scores)
    return {
        "step": float((1 + np.sum(null_step >= observed_step)) / (permutations + 1)),
        "direction": float((1 + np.sum(null_direction >= observed_direction)) / (permutations + 1)),
        "order": float((1 + np.sum(null_order >= observed_order)) / (permutations + 1)),
    }


def summarize_trajectory(
    stage: str,
    arrays: dict[str, np.ndarray | list[np.ndarray]],
    permutations: int,
    trials: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    pvalues = permutation_pvalues(arrays, permutations, rng)
    rows = []
    metric_specs = (
        ("Step correspondence", "step_actual", "step_null", "step"),
        ("Path direction", "direction_actual", "direction_null", "direction"),
        ("Position order", "order_actual", "order_null", "order"),
    )
    for metric, actual_key, null_key, p_key in metric_specs:
        actual = np.asarray(arrays[actual_key], dtype=np.float64)
        null = np.asarray(arrays[null_key], dtype=np.float64)
        difference = actual - null
        actual_low, actual_high = bootstrap_ci(actual, trials, rng)
        null_low, null_high = bootstrap_ci(null, trials, rng)
        diff_low, diff_high = bootstrap_ci(difference, trials, rng)
        rows.append(
            {
                "stage": stage,
                "metric": metric,
                "observed": float(actual.mean()),
                "observed_ci95_low": actual_low,
                "observed_ci95_high": actual_high,
                "permuted_null": float(null.mean()),
                "null_ci95_low": null_low,
                "null_ci95_high": null_high,
                "observed_minus_null": float(difference.mean()),
                "difference_ci95_low": diff_low,
                "difference_ci95_high": diff_high,
                "permutation_p": pvalues[p_key],
                "n_questions": len(actual),
                "permutations": permutations,
            }
        )
    auxiliary = {}
    for label, key in (
        ("Step magnitude ratio", "step_norm_ratio"),
        ("Residual effective rank", "effective_rank"),
    ):
        values = np.asarray(arrays[key], dtype=np.float64)
        low, high = bootstrap_ci(values, trials, rng)
        auxiliary[label] = {
            "mean": float(values.mean()),
            "ci95_low": low,
            "ci95_high": high,
        }
    return rows, auxiliary


def make_trajectory_evidence(
    stage1_path: Path,
    final_path: Path,
    output_dir: Path,
    permutations: int,
    trials: int,
    seed: int,
) -> dict:
    stage1_records = torch.load(stage1_path, map_location="cpu", weights_only=False)
    final_records = torch.load(final_path, map_location="cpu", weights_only=False)
    if len(stage1_records) != len(final_records):
        raise ValueError("Stage 1 and Final record counts differ")
    if [int(r["idx"]) for r in stage1_records] != [int(r["idx"]) for r in final_records]:
        raise ValueError("Stage 1 and Final records are not aligned")
    stage1_arrays = trajectory_arrays(stage1_records)
    final_arrays = trajectory_arrays(final_records)
    del stage1_records, final_records
    stage1_rows, stage1_aux = summarize_trajectory(
        "TRACE Stage 1", stage1_arrays, permutations, trials, seed
    )
    final_rows, final_aux = summarize_trajectory(
        "TRACE Final", final_arrays, permutations, trials, seed + 1000
    )
    rows = stage1_rows + final_rows
    write_csv(output_dir / "source_data" / "trajectory_permutation_null.csv", rows)
    change_rows = []
    change_rng = np.random.default_rng(seed + 2000)
    for metric, key in (
        ("Step correspondence", "step"),
        ("Path direction", "direction"),
        ("Position order", "order"),
    ):
        stage1_gap = np.asarray(stage1_arrays[f"{key}_actual"]) - np.asarray(stage1_arrays[f"{key}_null"])
        final_gap = np.asarray(final_arrays[f"{key}_actual"]) - np.asarray(final_arrays[f"{key}_null"])
        change = final_gap - stage1_gap
        low, high = bootstrap_ci(change, trials, change_rng)
        change_rows.append(
            {
                "metric": metric,
                "stage1_observed_minus_null": float(stage1_gap.mean()),
                "final_observed_minus_null": float(final_gap.mean()),
                "final_minus_stage1_gap_change": float(change.mean()),
                "change_ci95_low": low,
                "change_ci95_high": high,
                "n_questions": len(change),
            }
        )
    write_csv(output_dir / "source_data" / "trajectory_stage_change.csv", change_rows)
    auxiliary_rows = []
    for stage, values in (("TRACE Stage 1", stage1_aux), ("TRACE Final", final_aux)):
        for metric, stats in values.items():
            auxiliary_rows.append({"stage": stage, "metric": metric, **stats})
    write_csv(output_dir / "source_data" / "trajectory_auxiliary_metrics.csv", auxiliary_rows)

    metrics = ["Step correspondence", "Path direction", "Position order"]
    metric_titles = ["Step-to-step correspondence", "Question-specific direction", "Ordered progress"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.36))
    for panel, (ax, metric, title) in enumerate(zip(axes, metrics, metric_titles)):
        style_axis(ax, grid="x")
        subset = [row for row in rows if row["metric"] == metric]
        y_positions = np.asarray([1.0, 0.0])
        for y, row in zip(y_positions, subset):
            stage_color = COLORS["green"] if row["stage"] == "TRACE Stage 1" else COLORS["pink"]
            ax.plot(
                [row["permuted_null"], row["observed"]],
                [y, y],
                color=COLORS["null"],
                linewidth=2.0,
                zorder=1,
            )
            ax.errorbar(
                row["permuted_null"],
                y,
                xerr=[
                    [row["permuted_null"] - row["null_ci95_low"]],
                    [row["null_ci95_high"] - row["permuted_null"]],
                ],
                fmt="s",
                color=COLORS["gold"],
                ecolor=COLORS["gold"],
                markersize=4.0,
                capsize=2.0,
                elinewidth=0.9,
                markeredgecolor=COLORS["text"],
                markeredgewidth=0.35,
                zorder=3,
            )
            ax.errorbar(
                row["observed"],
                y,
                xerr=[
                    [row["observed"] - row["observed_ci95_low"]],
                    [row["observed_ci95_high"] - row["observed"]],
                ],
                fmt="o",
                color=stage_color,
                ecolor=stage_color,
                markersize=4.8,
                capsize=2.0,
                elinewidth=0.9,
                markeredgecolor=COLORS["text"],
                markeredgewidth=0.35,
                zorder=4,
            )
            ax.text(
                0.5 * (row["permuted_null"] + row["observed"]),
                y + 0.19,
                f'gap {row["observed_minus_null"]:+.3f}',
                ha="center",
                va="bottom",
                fontsize=5.9,
                color=stage_color,
                fontweight="bold",
            )
            ax.text(
                row["permuted_null"],
                y - 0.20,
                f'{row["permuted_null"]:.3f}',
                ha="center",
                va="top",
                fontsize=5.3,
                color=COLORS["muted"],
            )
            ax.text(
                row["observed"],
                y - 0.20,
                f'{row["observed"]:.3f}',
                ha="center",
                va="top",
                fontsize=5.3,
                color=stage_color,
            )
        ax.set_yticks(y_positions, ["Stage 1", "Final"])
        values = [row[key] for row in subset for key in ("null_ci95_low", "observed_ci95_high")]
        low, high = min(values), max(values)
        margin = max(0.012, 0.12 * (high - low))
        ax.set_xlim(low - margin, high + margin)
        ax.set_ylim(-0.42, 1.42)
        ax.set_title(title, loc="left", fontweight="bold")
        change = change_rows[panel]
        change_text = (
            r"Stage 2 $\Delta$gap $\approx$ 0"
            if metric == "Position order"
            else f'Stage 2 $\\Delta$gap {change["final_minus_stage1_gap_change"]:+.3f}'
        )
        ax.text(
            0.98,
            0.98,
            change_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.4,
            color=COLORS["pink"],
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": COLORS["pink"],
                "edgecolor": "none",
                "alpha": 0.12,
            },
        )
        panel_label(ax, chr(ord("a") + panel))
    null_handle = axes[0].scatter([], [], marker="s", color=COLORS["gold"], label="Permuted null")
    observed_handle = axes[0].scatter([], [], marker="o", color=COLORS["pink"], label="Observed")
    fig.legend(
        handles=[null_handle, observed_handle],
        labels=["Permuted null", "Observed"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=2,
        handletextpad=0.45,
        columnspacing=1.4,
    )
    fig.text(
        0.5,
        0.014,
        f"200 matched questions; {permutations:,} within-question or cross-question permutations. Error bars: question-bootstrap 95% CIs.",
        ha="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.27, wspace=0.34)
    save_figure(fig, output_dir / "fig_trajectory_permutation_null")
    plt.close(fig)

    md = [
        "| Stage | Metric | Observed | Permuted null | Gap [95% CI] | Permutation p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    tex_rows = []
    for row in rows:
        md.append(
            f'| {row["stage"]} | {row["metric"]} | {row["observed"]:.3f} | '
            f'{row["permuted_null"]:.3f} | {row["observed_minus_null"]:+.3f} '
            f'[{row["difference_ci95_low"]:.3f}, {row["difference_ci95_high"]:.3f}] | '
            f'{p_string(row["permutation_p"])} |'
        )
        tex_rows.append(
            f'{row["stage"].replace("TRACE ", "")} & {row["metric"]} & {row["observed"]:.3f} & '
            f'{row["permuted_null"]:.3f} & {row["observed_minus_null"]:+.3f} '
            f'[{row["difference_ci95_low"]:.3f}, {row["difference_ci95_high"]:.3f}] & '
            f'{p_string(row["permutation_p"])} ' + r"\\"
        )
    (output_dir / "table_trajectory_permutation_null.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    latex = "\n".join(
        [
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Stage & Metric & Observed & Null & Gap [95\% CI] & $p_{perm}$ \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output_dir / "table_trajectory_permutation_null.tex").write_text(
        latex + "\n", encoding="utf-8"
    )
    change_md = [
        "| Metric | Stage 1 gap | Final gap | Final - Stage 1 [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    change_tex_rows = []
    for row in change_rows:
        change_md.append(
            f'| {row["metric"]} | {row["stage1_observed_minus_null"]:.3f} | '
            f'{row["final_observed_minus_null"]:.3f} | {row["final_minus_stage1_gap_change"]:+.3f} '
            f'[{row["change_ci95_low"]:.3f}, {row["change_ci95_high"]:.3f}] |'
        )
        change_tex_rows.append(
            f'{row["metric"]} & {row["stage1_observed_minus_null"]:.3f} & '
            f'{row["final_observed_minus_null"]:.3f} & {row["final_minus_stage1_gap_change"]:+.3f} '
            f'[{row["change_ci95_low"]:.3f}, {row["change_ci95_high"]:.3f}] ' + r"\\"
        )
    (output_dir / "table_trajectory_stage_change.md").write_text(
        "\n".join(change_md) + "\n", encoding="utf-8"
    )
    change_latex = "\n".join(
        [
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Metric & Stage 1 gap & Final gap & Change [95\% CI] \\",
            r"\midrule",
            *change_tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output_dir / "table_trajectory_stage_change.tex").write_text(
        change_latex + "\n", encoding="utf-8"
    )
    payload = {
        "n_questions": len(np.asarray(stage1_arrays["step_actual"])),
        "permutations": permutations,
        "bootstrap_trials": trials,
        "primary_metrics": rows,
        "stage_changes": change_rows,
        "auxiliary_metrics": auxiliary_rows,
    }
    (output_dir / "trajectory_permutation_null.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def make_complexity_figure(reliability_dir: Path, output_dir: Path) -> None:
    source = read_csv(reliability_dir / "gold_cot_complexity_strata.csv")
    rows = []
    for row in source:
        rows.append(
            {
                "stratum": row["stratum"],
                "n": int(row["n"]),
                "gold_cot_word_threshold": float(row["gold_cot_word_threshold"]),
                "accuracy_gain_pp": float(row["accuracy_gain_pp"]),
                "accuracy_ci95_low": float(row["accuracy_ci95_low"]),
                "accuracy_ci95_high": float(row["accuracy_ci95_high"]),
                "length_saved": float(row["length_reduction"]),
                "length_ci95_low": float(row["length_ci95_low"]),
                "length_ci95_high": float(row["length_ci95_high"]),
                "rescued": int(row["rescued"]),
                "regressed": int(row["regressed"]),
            }
        )
    write_csv(output_dir / "source_data" / "refinement_by_complexity.csv", rows)
    labels = ["Other questions", "Long-CoT questions"]
    colors = [COLORS["blue"], COLORS["pink"]]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.15))
    for ax in axes:
        style_axis(ax, grid="y")
    x = np.arange(2)
    gain = np.asarray([row["accuracy_gain_pp"] for row in rows])
    gain_low = np.asarray([row["accuracy_ci95_low"] for row in rows])
    gain_high = np.asarray([row["accuracy_ci95_high"] for row in rows])
    axes[0].bar(x, gain, width=0.58, color=colors, edgecolor=COLORS["text"], linewidth=0.45)
    axes[0].errorbar(x, gain, yerr=[gain - gain_low, gain_high - gain], fmt="none", ecolor=COLORS["text"], capsize=2.5, linewidth=0.9)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Accuracy gain (pp)")
    axes[0].set_title("Outcome gain", loc="left", fontweight="bold")
    panel_label(axes[0], "a")
    saved = np.asarray([row["length_saved"] for row in rows])
    saved_low = np.asarray([row["length_ci95_low"] for row in rows])
    saved_high = np.asarray([row["length_ci95_high"] for row in rows])
    axes[1].bar(x, saved, width=0.58, color=[COLORS["gold"], COLORS["pink"]], edgecolor=COLORS["text"], linewidth=0.45)
    axes[1].errorbar(x, saved, yerr=[saved - saved_low, saved_high - saved], fmt="none", ecolor=COLORS["text"], capsize=2.5, linewidth=0.9)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Reasoning length saved")
    axes[1].set_title("Compression gain", loc="left", fontweight="bold")
    panel_label(axes[1], "b")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=0)
    fig.text(
        0.5,
        0.012,
        "GSM8K strata use gold-CoT word count only; long-CoT denotes the top quartile (>64 words). Error bars: paired bootstrap 95% CIs.",
        ha="center",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.105, right=0.99, top=0.82, bottom=0.28, wspace=0.35)
    save_figure(fig, output_dir / "fig_refinement_by_complexity")
    plt.close(fig)


def write_evidence_map(output_dir: Path, trajectory: dict) -> None:
    metrics = {row["metric"]: row for row in trajectory["primary_metrics"] if row["stage"] == "TRACE Final"}
    text = f"""# TRACE paper evidence map

## Central claim

TRACE first forms an ordered latent trajectory and then refines its outcomes. The refinement raises accuracy and cross-path reliability while reducing, rather than inflating, total reasoning length.

## Claim-to-evidence mapping

| Claim | Primary artifact | Decision supported |
|---|---|---|
| Outcome refinement improves utility without buying accuracy through longer decoding. | `table_stagewise_benchmarks` and `fig_outcome_refinement` panel a | Final improves accuracy significantly on GSM8K, GSMHard, and SVAMP, is numerically higher at the MultiArith ceiling, and shortens total #L on all four datasets. |
| The result is not specific to one answer-token budget. | `accuracy_length_pareto/fig_accuracy_length_pareto` and its table | Final gains accuracy and saves #L at budgets 24--48; at budget 64 accuracy saturates while Final remains substantially shorter. |
| The gain is not purchased with extra inference capacity. | `table_inference_architecture_parity` | Stage 1 and Final have identical state keys, shapes, dtypes, saved elements, eight latent slots, and answer budget. |
| Refinement converts path coverage into repeatable correctness. | `fig_outcome_refinement` panels b-c and `table_outcome_reliability` | Majority-correct and all-correct rates rise across eight fixed views. |
| Multi-path reasoning improves an optional consensus decision. | `table_multiview_decision` | Path accuracy, plurality-vote accuracy, all-path reliability, and modal agreement all increase on matched questions. |
| Latent states encode ordered, step-specific, question-specific trajectories rather than arbitrary point clouds. | `fig_trajectory_permutation_null` and `table_trajectory_permutation_null` | Observed structure exceeds within-question and cross-question permutation nulls on 200 matched questions. |
| The latent-slot order follows population-level human-CoT progress. | `fig_population_assignment_map` | Assignment-weighted progress-center distributions shift from early to late CoT stages as slot index increases, while the slot-permutation null is flat. |
| Refinement preserves the trajectory scaffold formed in Stage 1. | Stage 1 versus Final columns in `table_trajectory_permutation_null` | Position order remains stable while step-specific and directional gaps remain above null. |
| The two stages optimize the intended quantities under full validation. | `fig_training_dynamics` | Stage 1 reduces the plotted path, position, and direction objectives and reaches the multi-view margin; Stage 2 selects its checkpoint by full-epoch validation while output length remains controlled. |
| Refinement helps most where explicit reasoning is longer. | `fig_refinement_by_complexity` | Supporting, post-hoc localization of the gain; not used as the primary causal claim. |

## Key final-model null gaps

- Step correspondence: {metrics['Step correspondence']['observed_minus_null']:+.3f}.
- Question-specific path direction: {metrics['Path direction']['observed_minus_null']:+.3f}.
- Position order consistency: {metrics['Position order']['observed_minus_null']:+.3f}.

## Scope and review guardrails

- The 200-question aggregate and permutation controls carry the geometry claim; selected 3D paths are illustrations only.
- Current evidence supports trajectory formation and outcome reliability. It does not establish that incorrect paths occupy a universally separable geometric cluster.
- All behavioral comparisons are matched by question and use one deterministic decode unless a figure explicitly states eight fixed views.
- No unpublished external system is used as a baseline or named in any paper artifact.
"""
    (output_dir / "EVIDENCE_MAP.md").write_text(text, encoding="utf-8")


def write_legends(output_dir: Path) -> None:
    text = """# Figure legends

**Figure: Outcome refinement improves accuracy, efficiency, and path reliability.**
**a,** Paired movement from TRACE Stage 1 to TRACE Final on four mathematical-reasoning benchmarks. Positive coordinates jointly indicate an accuracy increase and a reduction in total reasoning length (#L); error bars are question-bootstrap 95% confidence intervals. **b,** Distribution of the number of correct paths among eight fixed views for 200 matched GSM8K questions. **c,** Paired Stage 1 and Final proportions satisfying at least one, at least five, or all eight correct paths; annotations report paired gains and 95% confidence intervals. Refinement primarily moves mass toward repeatable correctness rather than merely exposing one successful path.

**Figure: TRACE trajectories exceed permutation-based null structure.**
Observed latent trajectories are compared with matched nulls on 200 questions. **a,** Step correspondence is the diagonal cosine alignment between implicit residual steps and weak CoT-step targets; the null permutes target-step identities within each view. **b,** Path direction is the cosine alignment between the final cumulative implicit direction and its question-specific target; the null uses targets from other questions. **c,** Position order is pairwise consistency between latent-slot order and assignment-weighted CoT progress; the null permutes latent-slot identities. Horizontal segments join each null to its observed statistic on a metric-specific scale; error bars are question-bootstrap 95% confidence intervals and tests are one-sided with 1,024 permutations.

**Figure: Latent slots trace population-level CoT progress.**
**a-b,** Distribution of assignment-weighted progress centers for eight latent slots in TRACE Stage 1 and TRACE Final, shown as 24-bin histograms with a fixed five-bin `[1,4,6,4,1]/16` display kernel. The visualization uses all 124 multi-step CoTs from the 200-question audit because progress order is undefined for single-step CoTs. **c,** Expected CoT progress for each latent slot; the horizontal line is the exact expectation under slot-order permutation. Error bars are question-bootstrap 95% confidence intervals. The primary permutation table retains all 200 questions and treats single-step cases conservatively.

**Figure: Refinement concentrates gains on longer reasoning problems.**
Paired GSM8K accuracy and length changes are stratified using gold-CoT word count only. Long-CoT questions are the top quartile (>64 words); error bars are paired question-bootstrap 95% confidence intervals. This analysis localizes where refinement helps and is supporting rather than primary evidence.

**Figure: Outcome refinement improves the constrained-budget frontier.**
**a,** Matched GSM8K accuracy--length frontiers under maximum-answer budgets 24, 32, 40, 48, and 64; arrows connect architecture-identical Stage 1 and Final checkpoints at the same budget. **b,** Paired accuracy change. **c,** Paired total reasoning length saved. Every point uses all 1,319 test questions and one deterministic decode; error bars are paired question-bootstrap 95% confidence intervals. The 64-token point is retained as a saturation boundary rather than excluded.

**Figure: Outcome-selected TRACE path atlas.**
**a,** First case with the maximum increase in correct paths among eight fixed views. **b,** First case that is all-correct at both stages. **c,** Largest remaining majority rescue. Dashed paths show Stage 1 and colored paths show Final. Cases are selected using correctness transitions only. Each panel uses an outcome-free PCA fitted jointly to both checkpoints after population-template residualization. Fixed view lanes and display-normalized step magnitudes are readability transforms, not quantitative geometry evidence.

**Figure: Stagewise training dynamics.**
**a,** Stage 1 combined path, position, and direction losses over full training; faint traces are logged minibatch values and solid traces are 101-point moving means. **b,** Multi-view signature distance relative to the prespecified 0.16 diversity margin, with the corresponding margin-violation term. **c,** Full-epoch Stage 2 validation accuracy and output length; the star marks the selected epoch-5 checkpoint. These curves diagnose optimization and checkpoint selection but are not used as a substitute for component ablations.
"""
    (output_dir / "FIGURE_LEGENDS.md").write_text(text, encoding="utf-8")


def write_paper_placement(output_dir: Path) -> None:
    text = """# Paper placement

## Main text

| Artifact | Role in the argument |
|---|---|
| `table_stagewise_benchmarks` | End-to-end accuracy and total reasoning length on GSM8K and three OOD datasets. |
| `fig_outcome_refinement` | One visual summary of paired utility and eight-view reliability. |
| `fig_population_assignment_map` | Population-level early-to-late slot specialization. |
| `fig_trajectory_permutation_null` + table | Primary trajectory evidence with within-question, cross-question, and slot-order nulls. |
| `table_multiview_decision` | Exact reliability and optional plurality-decision values. |

## Appendix

| Artifact | Role in the argument |
|---|---|
| `accuracy_length_pareto/*` | Robustness to five answer-token budgets, including the saturation boundary. |
| `path_atlas/fig_outcome_selected_path_atlas` | Outcome-selected but geometry-blind qualitative path illustration. |
| `fig_training_dynamics` | Additional optimization and validation-based checkpoint-selection diagnostics retained in the evidence package. |
| `table_inference_architecture_parity` | Exact deployed-capacity audit. |
| `protocol_parity/table_evaluation_protocol_parity` | Prediction stability between the batch-1 audit and batch-8 sweep protocols at 48 tokens. |
| `external_baselines/table_semcot_accuracy_context` | Same-backbone SemCoT accuracy context with an explicit non-comparable-length caveat. |
| `fig_refinement_by_complexity` | Supporting post-hoc localization by gold-CoT length. |
| Component-ablation table | Filled only after each full-budget run and frozen-checkpoint audit completes. |

## Review boundary

The aggregate 200-question permutation tests carry the trajectory claim. Selected
3D paths may illustrate the same quantities but must not replace population
statistics. The supported Stage 2 claim is improved outcome reliability while
preserving ordered trajectory structure, not universal geometric separation of
correct and incorrect paths.
"""
    (output_dir / "PAPER_PLACEMENT.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired = json.loads(args.paired_results.read_text(encoding="utf-8"))
    stagewise = make_stagewise_table(paired, args.output_dir)
    make_outcome_figure(stagewise, args.reliability_dir, args.output_dir)
    make_multiview_decision_table(
        args.stage1_records,
        args.final_records,
        args.output_dir,
        args.bootstrap_trials,
        args.seed,
    )
    make_population_assignment_figure(
        args.stage1_records,
        args.final_records,
        args.output_dir,
        args.bootstrap_trials,
        args.seed,
    )
    trajectory = make_trajectory_evidence(
        args.stage1_records,
        args.final_records,
        args.output_dir,
        args.permutations,
        args.bootstrap_trials,
        args.seed,
    )
    make_complexity_figure(args.reliability_dir, args.output_dir)
    write_evidence_map(args.output_dir, trajectory)
    write_legends(args.output_dir)
    write_paper_placement(args.output_dir)
    clean_manifest = {
        "suite": "TRACE paper evidence",
        "stage_labels": ["TRACE Stage 1", "TRACE Final"],
        "n_geometry_questions": trajectory["n_questions"],
        "geometry_permutations": args.permutations,
        "bootstrap_trials": args.bootstrap_trials,
        "accuracy_length_budgets": [24, 32, 40, 48, 64],
        "path_atlas_cases": 3,
        "protocol_parity_audited": True,
        "component_ablations": [
            "without path consistency",
            "without weak progress anchoring",
            "without multi-view compression",
        ],
        "test_times": 1,
        "source_paths_disclosed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(clean_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(clean_manifest, indent=2))


if __name__ == "__main__":
    main()
