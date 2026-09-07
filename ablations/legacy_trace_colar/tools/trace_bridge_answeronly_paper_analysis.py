#!/usr/bin/env python
import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch


BLUE = "#6687B8"
GREEN = "#69B17D"
GOLD = "#E6A314"
PINK = "#E5A6C4"
INK = "#3F4854"
MID = "#7C8794"
LIGHT = "#E9EDF2"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def bootstrap_mean(values, rng, trials):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty array")
    means = np.empty(trials, dtype=np.float64)
    chunk = 256
    for start in range(0, trials, chunk):
        count = min(chunk, trials - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def bootstrap_independent_contrast(first, second, rng, trials):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    values = np.empty(trials, dtype=np.float64)
    for index in range(trials):
        values[index] = (
            rng.choice(first, len(first), replace=True).mean()
            - rng.choice(second, len(second), replace=True).mean()
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return float(first.mean() - second.mean()), float(low), float(high)


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_to_markdown(frame, index=True, float_digits=2):
    table = frame.reset_index() if index else frame.copy()
    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{float_digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def panel_label(axis, label, x=-0.12, y=1.06):
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )


def category(correct_count):
    if correct_count == 0:
        return "All wrong"
    if correct_count == 8:
        return "All correct"
    return "Mixed"


def load_rollout_rows(stage1_path, final_path, max_records):
    stage1_records = torch.load(stage1_path, map_location="cpu", weights_only=False)
    final_records = torch.load(final_path, map_location="cpu", weights_only=False)
    stage1 = {int(row["idx"]): row for row in stage1_records[:max_records]}
    final = {int(row["idx"]): row for row in final_records[:max_records]}
    common = sorted(set(stage1) & set(final))
    if len(common) != max_records:
        raise ValueError(f"Expected {max_records} common rollout records, found {len(common)}")
    rows = []
    for idx in common:
        for key in ("question", "answer", "steps"):
            if stage1[idx].get(key) != final[idx].get(key):
                raise ValueError(f"Mismatched {key} for rollout record {idx}")
        stage1_view_ids = stage1[idx].get("multiview_view_ids")
        final_view_ids = final[idx].get("multiview_view_ids")
        if not torch.is_tensor(stage1_view_ids) or not torch.is_tensor(final_view_ids):
            raise ValueError(f"Missing multiview_view_ids for rollout record {idx}")
        if not torch.equal(stage1_view_ids.cpu(), final_view_ids.cpu()):
            raise ValueError(f"Mismatched multiview_view_ids for rollout record {idx}")
        if int(stage1_view_ids.numel()) != 8:
            raise ValueError(f"Expected eight fixed views for record {idx}")
        stage1_correct = int(stage1[idx]["multiview_acc"].float().sum().item())
        final_correct = int(final[idx]["multiview_acc"].float().sum().item())
        rows.append(
            {
                "idx": idx,
                "stage1_correct_views": stage1_correct,
                "final_correct_views": final_correct,
                "stage1_category": category(stage1_correct),
                "final_category": category(final_correct),
            }
        )
    return pd.DataFrame(rows)


def reliability_analysis(rows, out_dir, rng, trials):
    distribution_rows = []
    for method, column in (("Stage 1", "stage1_correct_views"), ("Refined TRACE", "final_correct_views")):
        counts = rows[column].value_counts().reindex(range(9), fill_value=0)
        for correct_views, count in counts.items():
            distribution_rows.append(
                {
                    "method": method,
                    "correct_views": int(correct_views),
                    "question_count": int(count),
                    "question_percent": 100.0 * count / len(rows),
                }
            )
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(out_dir / "reliability_distribution.csv", index=False)

    thresholds = [("Any correct", 1), ("Majority correct", 5), ("All correct", 8)]
    threshold_rows = []
    for label, threshold in thresholds:
        stage1_success = (rows["stage1_correct_views"].to_numpy() >= threshold).astype(float)
        final_success = (rows["final_correct_views"].to_numpy() >= threshold).astype(float)
        delta = 100.0 * (final_success - stage1_success)
        mean, low, high = bootstrap_mean(delta, rng, trials)
        threshold_rows.append(
            {
                "threshold": label,
                "minimum_correct_views": threshold,
                "stage1_percent": 100.0 * stage1_success.mean(),
                "final_percent": 100.0 * final_success.mean(),
                "delta_pp": mean,
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "rescued_questions": int(((stage1_success == 0) & (final_success == 1)).sum()),
                "regressed_questions": int(((stage1_success == 1) & (final_success == 0)).sum()),
            }
        )
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(out_dir / "reliability_thresholds.csv", index=False)

    order = ["All wrong", "Mixed", "All correct"]
    transition = pd.crosstab(rows["stage1_category"], rows["final_category"]).reindex(
        index=order, columns=order, fill_value=0
    )
    transition.index.name = "stage1_category"
    transition.to_csv(out_dir / "reliability_transitions.csv")

    figure = plt.figure(figsize=(7.2, 4.45), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
    ax_a = figure.add_subplot(grid[:, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 1])

    x = np.arange(9)
    width = 0.35
    stage1_percent = distribution[distribution.method == "Stage 1"].sort_values("correct_views")[
        "question_percent"
    ].to_numpy()
    final_percent = distribution[distribution.method == "Refined TRACE"].sort_values("correct_views")[
        "question_percent"
    ].to_numpy()
    ax_a.bar(x - width / 2, stage1_percent, width, color=BLUE, label="Stage 1", edgecolor="white", linewidth=0.35)
    ax_a.bar(x + width / 2, final_percent, width, color=PINK, label="Refined TRACE", edgecolor="white", linewidth=0.35)
    ax_a.set_xlabel("Correct rollouts among eight fixed views")
    ax_a.set_ylabel("Questions (%)")
    ax_a.set_xticks(x)
    ax_a.set_ylim(0, max(final_percent.max(), stage1_percent.max()) * 1.18)
    ax_a.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax_a.set_axisbelow(True)
    ax_a.set_title("Answer-only refinement makes successful outcomes repeatable", loc="left", pad=8)
    ax_a.legend(loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.2)
    panel_label(ax_a, "a", x=-0.10, y=1.035)

    y = np.arange(len(threshold_frame))[::-1]
    stage_values = threshold_frame["stage1_percent"].to_numpy()
    final_values = threshold_frame["final_percent"].to_numpy()
    for row_index, y_value in enumerate(y):
        ax_b.plot([stage_values[row_index], final_values[row_index]], [y_value, y_value], color=MID, linewidth=1.25)
        ax_b.scatter(stage_values[row_index], y_value, s=23, color=BLUE, zorder=3)
        ax_b.scatter(final_values[row_index], y_value, s=28, color=PINK, marker="s", zorder=3)
        delta = threshold_frame.iloc[row_index]
        ax_b.text(
            max(stage_values[row_index], final_values[row_index]) + 2.0,
            y_value,
            f"{delta.delta_pp:+.1f} [{delta.delta_ci95_low:+.1f}, {delta.delta_ci95_high:+.1f}]",
            va="center",
            ha="left",
            fontsize=6.5,
            color=INK,
        )
    ax_b.set_yticks(y, threshold_frame["threshold"])
    ax_b.set_xlim(20, 103)
    ax_b.set_xlabel("Questions meeting threshold (%)")
    ax_b.set_title("Paired reliability gain, pp [95% CI]", loc="left", pad=5)
    ax_b.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax_b.set_axisbelow(True)
    panel_label(ax_b, "b", x=-0.16, y=1.10)

    matrix = transition.to_numpy()
    ax_c.set_xlim(0, 3)
    ax_c.set_ylim(3, 0)
    ax_c.set_aspect("equal")
    ax_c.spines[:].set_visible(False)
    for row_index in range(3):
        for column_index in range(3):
            value = int(matrix[row_index, column_index])
            if value == 0:
                color = "white"
            elif column_index > row_index:
                color = GREEN
            elif column_index < row_index:
                color = GOLD
            else:
                color = PINK
            alpha = 1.0 if value == 0 else 0.25 + 0.65 * value / max(1, matrix.max())
            ax_c.add_patch(
                Rectangle(
                    (column_index, row_index),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.2,
                    alpha=alpha,
                )
            )
            ax_c.text(column_index + 0.5, row_index + 0.5, str(value), ha="center", va="center", fontsize=8.2)
    ax_c.set_xticks(np.arange(3) + 0.5, ["All wrong", "Mixed", "All correct"], rotation=18, ha="right")
    ax_c.set_yticks(np.arange(3) + 0.5, ["All wrong", "Mixed", "All correct"])
    ax_c.set_xlabel("Refined TRACE")
    ax_c.set_ylabel("Stage 1")
    ax_c.set_title("Paired question transitions", loc="left", pad=5)
    panel_label(ax_c, "c", x=-0.18, y=1.12)

    save_figure(figure, out_dir / "fig_reliability_refinement")
    return distribution, threshold_frame, transition


def ranked_top_quartile(frame, value_column):
    ordered = frame.sort_values([value_column, "idx"], kind="stable").reset_index(drop=True).copy()
    top_count = int(np.ceil(len(ordered) * 0.25))
    ordered["stratum"] = "Lower 75%"
    ordered.loc[len(ordered) - top_count :, "stratum"] = "Top 25%"
    return ordered


def summarize_strata(frame, grouping, rng, trials, source):
    rows = []
    for stratum in ("Lower 75%", "Top 25%"):
        subset = frame[frame[grouping] == stratum]
        accuracy_delta = 100.0 * (subset["candidate_acc"] - subset["reference_acc"]).to_numpy()
        length_reduction = -subset["delta_L"].to_numpy()
        acc_mean, acc_low, acc_high = bootstrap_mean(accuracy_delta, rng, trials)
        len_mean, len_low, len_high = bootstrap_mean(length_reduction, rng, trials)
        rows.append(
            {
                "source": source,
                "dataset": str(subset["dataset"].iloc[0]),
                "stratum": stratum,
                "n": len(subset),
                "accuracy_gain_pp": acc_mean,
                "accuracy_ci95_low": acc_low,
                "accuracy_ci95_high": acc_high,
                "length_reduction": len_mean,
                "length_ci95_low": len_low,
                "length_ci95_high": len_high,
                "rescued": int((subset["transition"] == "rescued").sum()),
                "regressed": int((subset["transition"] == "regressed").sum()),
            }
        )
    return rows


def complexity_analysis(paired_csv, gsm8k_meta, out_dir, rng, trials):
    paired = pd.read_csv(paired_csv)
    stage1_strata = []
    for dataset, dataset_rows in paired.groupby("dataset", sort=False):
        ranked = ranked_top_quartile(dataset_rows, "reference_L")
        stage1_strata.extend(summarize_strata(ranked, "stratum", rng, trials, "Stage1 output #L rank"))
    stage1_frame = pd.DataFrame(stage1_strata)
    stage1_frame.to_csv(out_dir / "stage1_length_strata.csv", index=False)

    metadata = json.loads(gsm8k_meta.read_text(encoding="utf-8"))
    metadata_rows = []
    for idx, record in enumerate(metadata):
        cot_text = " ".join(record.get("steps", []))
        metadata_rows.append(
            {
                "idx": idx,
                "gold_cot_words": len(re.findall(r"\b\w+\b", cot_text)),
                "gold_cot_steps": len(record.get("steps", [])),
            }
        )
    gsm8k = paired[paired["dataset"] == "GSM8K"].merge(
        pd.DataFrame(metadata_rows), on="idx", validate="one_to_one"
    )
    threshold = float(gsm8k["gold_cot_words"].quantile(0.75))
    gsm8k["gold_stratum"] = np.where(
        gsm8k["gold_cot_words"] > threshold, "Long gold CoT (>Q3)", "Other questions"
    )
    gold_rows = []
    for stratum in ("Other questions", "Long gold CoT (>Q3)"):
        subset = gsm8k[gsm8k["gold_stratum"] == stratum]
        accuracy_delta = 100.0 * (subset["candidate_acc"] - subset["reference_acc"]).to_numpy()
        length_reduction = -subset["delta_L"].to_numpy()
        acc_mean, acc_low, acc_high = bootstrap_mean(accuracy_delta, rng, trials)
        len_mean, len_low, len_high = bootstrap_mean(length_reduction, rng, trials)
        gold_rows.append(
            {
                "dataset": "GSM8K",
                "stratum": stratum,
                "gold_cot_word_threshold": threshold,
                "n": len(subset),
                "accuracy_gain_pp": acc_mean,
                "accuracy_ci95_low": acc_low,
                "accuracy_ci95_high": acc_high,
                "length_reduction": len_mean,
                "length_ci95_low": len_low,
                "length_ci95_high": len_high,
                "rescued": int((subset["transition"] == "rescued").sum()),
                "regressed": int((subset["transition"] == "regressed").sum()),
            }
        )
    gold_frame = pd.DataFrame(gold_rows)
    gold_frame.to_csv(out_dir / "gold_cot_complexity_strata.csv", index=False)

    long_subset = gsm8k[gsm8k["gold_stratum"] == "Long gold CoT (>Q3)"]
    other_subset = gsm8k[gsm8k["gold_stratum"] == "Other questions"]
    gold_contrasts = {
        "accuracy_gain_contrast_pp": bootstrap_independent_contrast(
            100.0 * (long_subset["candidate_acc"] - long_subset["reference_acc"]),
            100.0 * (other_subset["candidate_acc"] - other_subset["reference_acc"]),
            rng,
            trials,
        ),
        "length_reduction_contrast": bootstrap_independent_contrast(
            -long_subset["delta_L"], -other_subset["delta_L"], rng, trials
        ),
    }

    figure = plt.figure(figsize=(7.2, 3.25), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.0, 1.55], height_ratios=[1.0, 1.0])
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[:, 1])

    group_order = ["Other questions", "Long gold CoT (>Q3)"]
    group_colors = [BLUE, PINK]
    y_positions = np.array([1, 0])
    for axis, value, low, high, xlabel, title in (
        (
            ax_a,
            "accuracy_gain_pp",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "Accuracy gain (pp)",
            "Long gold solutions receive larger gains",
        ),
        (
            ax_b,
            "length_reduction",
            "length_ci95_low",
            "length_ci95_high",
            "Reasoning-length reduction (-Δ#L)",
            "The same questions become more concise",
        ),
    ):
        for index, stratum in enumerate(group_order):
            row = gold_frame[gold_frame.stratum == stratum].iloc[0]
            axis.errorbar(
                row[value],
                y_positions[index],
                xerr=[[row[value] - row[low]], [row[high] - row[value]]],
                fmt="o",
                color=group_colors[index],
                ecolor=INK,
                elinewidth=1.0,
                capsize=2.5,
                markersize=5.2,
                zorder=3,
            )
            axis.text(row[high] + 0.3, y_positions[index], f"n={int(row.n)}", va="center", fontsize=6.4, color=INK)
        axis.axvline(0, color=MID, linewidth=0.8, linestyle="--")
        axis.set_yticks(y_positions, ["Other", "Gold CoT > Q3"])
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left", pad=5)
        axis.grid(axis="x", color=LIGHT, linewidth=0.7)
        axis.set_axisbelow(True)
    panel_label(ax_a, "a", x=-0.18, y=1.12)
    panel_label(ax_b, "b", x=-0.18, y=1.12)

    dataset_colors = {"GSM8K": BLUE, "GSMHard": GREEN, "SVAMP": GOLD, "MultiArith": PINK}
    label_offsets = {
        "GSM8K": (0.25, 0.20),
        "GSMHard": (0.25, 0.18),
        "SVAMP": (0.25, 0.20),
        "MultiArith": (0.32, -0.02),
    }
    for dataset, color in dataset_colors.items():
        subset = stage1_frame[stage1_frame.dataset == dataset].set_index("stratum")
        lower = subset.loc["Lower 75%"]
        top = subset.loc["Top 25%"]
        ax_c.annotate(
            "",
            xy=(top.length_reduction, top.accuracy_gain_pp),
            xytext=(lower.length_reduction, lower.accuracy_gain_pp),
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.5, "shrinkA": 4, "shrinkB": 4},
        )
        ax_c.scatter(
            lower.length_reduction,
            lower.accuracy_gain_pp,
            s=32,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
        ax_c.scatter(
            top.length_reduction,
            top.accuracy_gain_pp,
            s=38,
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        label_dx, label_dy = label_offsets[dataset]
        ax_c.text(
            top.length_reduction + label_dx,
            top.accuracy_gain_pp + label_dy,
            dataset,
            color=color,
            fontsize=6.8,
            fontweight="bold",
        )
    ax_c.axhline(0, color=MID, linewidth=0.8, linestyle="--")
    ax_c.axvline(0, color=MID, linewidth=0.8, linestyle="--")
    ax_c.set_xlabel("Reasoning-length reduction (-Δ#L)")
    ax_c.set_ylabel("Accuracy gain (pp)")
    ax_c.set_title("Long Stage 1 outputs define a high-opportunity stratum", loc="left", pad=5)
    ax_c.grid(color=LIGHT, linewidth=0.7)
    ax_c.set_axisbelow(True)
    ax_c.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK, label="Lower 75%"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, markeredgecolor="white", label="Top 25%"),
        ],
        loc="upper left",
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    ax_c.text(
        0.98,
        0.02,
        "Post hoc: ranked within dataset by Stage 1 #L; ties use dataset index",
        transform=ax_c.transAxes,
        fontsize=5.8,
        color=MID,
        va="bottom",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )
    panel_label(ax_c, "c", x=-0.11, y=1.06)

    save_figure(figure, out_dir / "fig_where_refinement_helps")
    return gold_frame, stage1_frame, gold_contrasts


def write_report(path, thresholds, transition, gold_frame, stage1_frame, contrasts, trials):
    majority = thresholds[thresholds.threshold == "Majority correct"].iloc[0]
    unanimous = thresholds[thresholds.threshold == "All correct"].iloc[0]
    any_correct = thresholds[thresholds.threshold == "Any correct"].iloc[0]
    gold_long = gold_frame[gold_frame.stratum == "Long gold CoT (>Q3)"].iloc[0]
    gold_other = gold_frame[gold_frame.stratum == "Other questions"].iloc[0]
    lines = [
        "# TRACE Answer-Only Post-Hoc Paper Analysis",
        "",
        "## Outcome coverage becomes reliability",
        "",
        f"On 200 matched GSM8K questions with eight fixed views, the fraction with at least one correct view changes only from {any_correct.stage1_percent:.1f}% to {any_correct.final_percent:.1f}% (delta {any_correct.delta_pp:+.1f} pp, 95% CI [{any_correct.delta_ci95_low:+.1f}, {any_correct.delta_ci95_high:+.1f}]).",
        f"In contrast, majority-correct questions increase from {majority.stage1_percent:.1f}% to {majority.final_percent:.1f}% (delta {majority.delta_pp:+.1f} pp, CI [{majority.delta_ci95_low:+.1f}, {majority.delta_ci95_high:+.1f}]), and all-correct groups increase from {unanimous.stage1_percent:.1f}% to {unanimous.final_percent:.1f}% (delta {unanimous.delta_pp:+.1f} pp, CI [{unanimous.delta_ci95_low:+.1f}, {unanimous.delta_ci95_high:+.1f}]).",
        "",
        "Interpretation: Stage 1 already exposes a correct path on most audited questions; answer-only refinement mainly reallocates probability mass toward successful paths and makes them repeatable. This is outcome reliability, not evidence for geometric correct/wrong modes.",
        "",
        "Paired category transitions (rows Stage 1, columns refined TRACE):",
        "",
        frame_to_markdown(transition, index=True, float_digits=0),
        "",
        "## Where refinement helps",
        "",
        f"Using gold-CoT word count fixed before model inference, GSM8K questions above the 75th percentile gain {gold_long.accuracy_gain_pp:+.2f} pp (CI [{gold_long.accuracy_ci95_low:+.2f}, {gold_long.accuracy_ci95_high:+.2f}]) and save {gold_long.length_reduction:.2f} #L (CI [{gold_long.length_ci95_low:.2f}, {gold_long.length_ci95_high:.2f}]). Other questions gain {gold_other.accuracy_gain_pp:+.2f} pp and save {gold_other.length_reduction:.2f} #L.",
        f"The long-minus-other contrast is {contrasts['accuracy_gain_contrast_pp'][0]:+.2f} pp, CI [{contrasts['accuracy_gain_contrast_pp'][1]:+.2f}, {contrasts['accuracy_gain_contrast_pp'][2]:+.2f}], with {contrasts['length_reduction_contrast'][0]:+.2f} additional #L saved, CI [{contrasts['length_reduction_contrast'][1]:+.2f}, {contrasts['length_reduction_contrast'][2]:+.2f}].",
        "",
        "Stage1-output-length stratification is model-derived and therefore secondary. It is consistent across the four datasets but should be labeled post hoc rather than used as a primary causal result.",
        "",
        frame_to_markdown(stage1_frame, index=False, float_digits=2),
        "",
        "## Recommended captions",
        "",
        "**Outcome coverage becomes reliability.** On the same 200 GSM8K questions and eight fixed view IDs, (a) answer-only outcome refinement shifts the distribution toward eight-of-eight success, (b) majority-correct and all-correct reliability increase while at-least-one-correct coverage is nearly unchanged, and (c) the paired transition matrix shows 28 mixed groups becoming all-correct versus three all-correct groups becoming mixed. Error bars are 95% paired question-bootstrap intervals. The policy signal uses answer correctness and length; the final training objective also retains Stage 1 replay with weight 0.05.",
        "",
        "**Where refinement helps (post hoc).** Panels (a--b) stratify GSM8K by gold-CoT word count, which is fixed before model inference. Questions above the 75th percentile receive a larger accuracy gain and a larger output-length reduction. Panel (c) shows the analogous model-derived Stage 1 output-length quartiles across datasets; open and filled markers denote the lower 75% and top 25%, respectively. This analysis localizes gains but does not replace the matched BRIDGE-plus-RL control.",
        "",
        "## Review boundaries",
        "",
        "- The rollout audit uses 200 questions and eight fixed views, not independent training seeds.",
        "- Gold correctness defines coverage/reliability thresholds; no claim of unsupervised confidence calibration is made.",
        "- Complexity stratification explains where gains occur but does not replace the matched BRIDGE-plus-RL control.",
        "- The deterministic full-test accuracy, the 200-question eight-view audit, and the post-hoc complexity split are distinct protocols and must not be merged into one sample size or uncertainty statement.",
        f"- All confidence intervals use {trials:,} question-level bootstrap resamples.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired_csv", type=Path, required=True)
    parser.add_argument("--stage1_records", type=Path, required=True)
    parser.add_argument("--final_records", type=Path, required=True)
    parser.add_argument("--gsm8k_meta", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rollout_rows = load_rollout_rows(args.stage1_records, args.final_records, args.max_records)
    rollout_rows.to_csv(args.out_dir / "rollout_question_rows.csv", index=False)
    _, thresholds, transition = reliability_analysis(
        rollout_rows, args.out_dir, rng, args.bootstrap_trials
    )
    gold_frame, stage1_frame, contrasts = complexity_analysis(
        args.paired_csv, args.gsm8k_meta, args.out_dir, rng, args.bootstrap_trials
    )
    summary = {
        "bootstrap_trials": args.bootstrap_trials,
        "seed": args.seed,
        "rollout_questions": len(rollout_rows),
        "reliability_thresholds": thresholds.to_dict(orient="records"),
        "transition_matrix": transition.to_dict(),
        "gold_cot_complexity": gold_frame.to_dict(orient="records"),
        "stage1_length_strata": stage1_frame.to_dict(orient="records"),
        "gold_cot_contrasts": contrasts,
        "definitions": {
            "views": "eight fixed same-question views",
            "majority_correct": "at least five of eight views are correct",
            "gold_long": "gold-CoT word count strictly above its GSM8K test-set 75th percentile",
            "stage1_top_quartile": "top 25% ranked by Stage1 #L within dataset; index breaks ties",
        },
        "provenance": {
            "paired_csv": str(args.paired_csv.resolve()),
            "paired_csv_sha256": sha256_file(args.paired_csv),
            "stage1_records": str(args.stage1_records.resolve()),
            "stage1_records_sha256": sha256_file(args.stage1_records),
            "final_records": str(args.final_records.resolve()),
            "final_records_sha256": sha256_file(args.final_records),
            "gsm8k_meta": str(args.gsm8k_meta.resolve()),
            "gsm8k_meta_sha256": sha256_file(args.gsm8k_meta),
            "protocol_check": "question, answer, steps, and eight ordered view IDs match for all 200 records",
        },
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(
        args.out_dir / "PAPER_ANALYSIS.md",
        thresholds,
        transition,
        gold_frame,
        stage1_frame,
        contrasts,
        args.bootstrap_trials,
    )


if __name__ == "__main__":
    main()
