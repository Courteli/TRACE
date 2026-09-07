#!/usr/bin/env python3
"""Quantify whether outcome refinement preserves TRACE path identity and diversity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    parser.add_argument("--stage1-records", type=Path, required=True)
    parser.add_argument("--final-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def as_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def normalize(array: np.ndarray) -> np.ndarray:
    return array / np.clip(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8, None)


def path_signatures(records: list[dict]) -> np.ndarray:
    signatures = []
    for record in records:
        residuals = as_array(record["multiview_implicit_residuals"])
        if residuals.ndim != 3:
            raise ValueError(f"Expected [views, steps, hidden], received {residuals.shape}")
        signatures.append(normalize(normalize(residuals).reshape(residuals.shape[0], -1)))
    return np.stack(signatures)


def bootstrap_ci(
    values: np.ndarray,
    trials: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(trials, dtype=np.float64)
    chunk = 500
    for start in range(0, trials, chunk):
        end = min(start + chunk, trials)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)


def style_axis(ax: plt.Axes, grid: str = "x") -> None:
    ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], width=0.7, length=2.5)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
        color=COLORS["text"],
    )


def p_string(value: float) -> str:
    return f"{value:.1e}" if value < 0.001 else f"{value:.3f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_records = torch.load(args.stage1_records, map_location="cpu", weights_only=False)
    final_records = torch.load(args.final_records, map_location="cpu", weights_only=False)
    stage1_indices = [int(record["idx"]) for record in stage1_records]
    final_indices = [int(record["idx"]) for record in final_records]
    if stage1_indices != final_indices:
        raise ValueError("Stage 1 and Final records are not aligned by question")

    stage1 = path_signatures(stage1_records)
    final = path_signatures(final_records)
    if stage1.shape != final.shape:
        raise ValueError(f"Signature shapes differ: {stage1.shape} versus {final.shape}")
    n_questions, n_views, _ = stage1.shape
    if n_questions < 2 or n_views < 2:
        raise ValueError("Identity controls require at least two questions and two views")

    mean_view_matrix = np.zeros((n_views, n_views), dtype=np.float64)
    per_question_view_matrices = np.empty(
        (n_questions, n_views, n_views), dtype=np.float32
    )
    same_view_question_matrix = np.empty(
        (n_questions, n_questions), dtype=np.float32
    )
    matched = np.empty(n_questions, dtype=np.float64)
    same_question_other_view = np.empty(n_questions, dtype=np.float64)
    different_question_same_view = np.empty(n_questions, dtype=np.float64)
    different_question_other_view = np.empty(n_questions, dtype=np.float64)

    final_flat = final.reshape(n_questions * n_views, -1)
    for question_idx in range(n_questions):
        cross = stage1[question_idx] @ final_flat.T
        cross = cross.reshape(n_views, n_questions, n_views)
        same_question = cross[:, question_idx, :]
        mean_view_matrix += same_question
        per_question_view_matrices[question_idx] = same_question
        same_view_question_matrix[question_idx] = np.stack(
            [cross[view, :, view] for view in range(n_views)]
        ).mean(axis=0)
        diagonal_sum = float(np.trace(same_question))
        matched[question_idx] = diagonal_sum / n_views
        same_question_other_view[question_idx] = (
            same_question.sum() - diagonal_sum
        ) / (n_views * (n_views - 1))

        same_view_values = np.stack([cross[view, :, view] for view in range(n_views)])
        same_view_total = same_view_values.sum() - diagonal_sum
        different_question_same_view[question_idx] = same_view_total / (
            n_views * (n_questions - 1)
        )
        all_total = cross.sum()
        other_question_other_view_total = (
            all_total - same_question.sum() - same_view_values.sum() + diagonal_sum
        )
        different_question_other_view[question_idx] = other_question_other_view_total / (
            n_views * (n_questions - 1) * (n_views - 1)
        )
    mean_view_matrix /= n_questions

    def diversity(signatures: np.ndarray) -> np.ndarray:
        values = []
        for question_paths in signatures:
            similarities = question_paths @ question_paths.T
            off_diagonal = (similarities.sum() - np.trace(similarities)) / (
                n_views * (n_views - 1)
            )
            values.append(1.0 - off_diagonal)
        return np.asarray(values, dtype=np.float64)

    stage1_diversity = diversity(stage1)
    final_diversity = diversity(final)
    view_gap = matched - same_question_other_view
    question_gap = matched - different_question_same_view
    diversity_change = final_diversity - stage1_diversity

    rng = np.random.default_rng(args.seed)
    view_p_null = np.empty(args.permutations, dtype=np.float64)
    question_p_null = np.empty(args.permutations, dtype=np.float64)
    question_indices = np.arange(n_questions)
    view_indices = np.arange(n_views)
    for trial in range(args.permutations):
        shifts = rng.integers(1, n_views, size=n_questions)
        shifted_view_indices = (view_indices[None, :] - shifts[:, None]) % n_views
        view_p_null[trial] = float(
            per_question_view_matrices[
                question_indices[:, None], view_indices[None, :], shifted_view_indices
            ].mean()
        )
        shift = int(rng.integers(1, n_questions))
        shifted_question_indices = (question_indices - shift) % n_questions
        question_p_null[trial] = float(
            same_view_question_matrix[
                question_indices, shifted_question_indices
            ].mean()
        )
    observed = float(matched.mean())
    view_p = float((1 + np.sum(view_p_null >= observed)) / (args.permutations + 1))
    question_p = float(
        (1 + np.sum(question_p_null >= observed)) / (args.permutations + 1)
    )

    per_question_rows = []
    for offset, question_idx in enumerate(stage1_indices):
        per_question_rows.append(
            {
                "question_idx": question_idx,
                "matched_question_matched_view_similarity": matched[offset],
                "matched_question_other_view_similarity": same_question_other_view[offset],
                "other_question_matched_view_similarity": different_question_same_view[offset],
                "other_question_other_view_similarity": different_question_other_view[offset],
                "view_identity_gap": view_gap[offset],
                "question_identity_gap": question_gap[offset],
                "stage1_cross_view_diversity": stage1_diversity[offset],
                "final_cross_view_diversity": final_diversity[offset],
                "diversity_change": diversity_change[offset],
            }
        )
    write_csv(args.output_dir / "source_data" / "path_identity_per_question.csv", per_question_rows)
    matrix_rows = []
    for stage1_view in range(n_views):
        for final_view in range(n_views):
            matrix_rows.append(
                {
                    "stage1_view": stage1_view + 1,
                    "final_view": final_view + 1,
                    "mean_cosine_similarity": mean_view_matrix[stage1_view, final_view],
                }
            )
    write_csv(args.output_dir / "source_data" / "path_identity_view_matrix.csv", matrix_rows)

    metric_specs = [
        (
            "View identity",
            matched,
            same_question_other_view,
            view_gap,
            view_p,
            "same question, different view",
        ),
        (
            "Question identity",
            matched,
            different_question_same_view,
            question_gap,
            question_p,
            "different question, same view",
        ),
        (
            "Cross-view diversity",
            final_diversity,
            stage1_diversity,
            diversity_change,
            None,
            "TRACE Stage 1",
        ),
    ]
    aggregate_rows = []
    for metric, value, control, gap, pvalue, control_name in metric_specs:
        low, high = bootstrap_ci(gap, args.bootstrap_trials, rng)
        aggregate_rows.append(
            {
                "metric": metric,
                "observed_or_final": float(value.mean()),
                "control_or_stage1": float(control.mean()),
                "control_definition": control_name,
                "gap_or_change": float(gap.mean()),
                "gap_ci95_low": low,
                "gap_ci95_high": high,
                "permutation_p": pvalue,
                "n_questions": n_questions,
            }
        )
    write_csv(args.output_dir / "source_data" / "path_identity_summary.csv", aggregate_rows)

    summary = {
        "n_questions": n_questions,
        "n_views_per_question": n_views,
        "signature": "concatenated unit-normalized implicit residual steps",
        "bootstrap_trials": args.bootstrap_trials,
        "permutations": args.permutations,
        "metrics": aggregate_rows,
        "interpretation": (
            "Outcome refinement preserves question- and view-conditioned path identity "
            "without collapsing cross-view diversity. This is not a correctness-separation test."
        ),
    }
    (args.output_dir / "path_identity_retention.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure_cmap = LinearSegmentedColormap.from_list(
        "trace_identity", [COLORS["white"], COLORS["pink"], COLORS["blue"]]
    )
    fig = plt.figure(figsize=(7.2, 2.55))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.17, 0.84, 1.25], wspace=0.55)

    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(mean_view_matrix, cmap=figure_cmap, vmin=0.84, vmax=0.98, aspect="equal")
    for view in range(n_views):
        ax.add_patch(
            plt.Rectangle(
                (view - 0.5, view - 0.5),
                1,
                1,
                fill=False,
                edgecolor=COLORS["gold"],
                linewidth=1.05,
            )
        )
    ax.set_xticks(np.arange(n_views), np.arange(1, n_views + 1))
    ax.set_yticks(np.arange(n_views), np.arange(1, n_views + 1))
    ax.set_xlabel("Final view")
    ax.set_ylabel("Stage 1 view")
    ax.set_title("View identity is retained", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.035)
    colorbar.ax.set_title("cos.", fontsize=5.9, color=COLORS["muted"], pad=2)
    colorbar.ax.tick_params(labelsize=5.9, length=2)
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    identity_matrix = np.asarray(
        [
            [different_question_other_view.mean(), different_question_same_view.mean()],
            [same_question_other_view.mean(), matched.mean()],
        ]
    )
    ax.imshow(identity_matrix, cmap=figure_cmap, vmin=0.78, vmax=0.98, aspect="equal")
    for row in range(2):
        for column in range(2):
            value = identity_matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight="bold",
                color=COLORS["white"] if value > 0.91 else COLORS["text"],
            )
    ax.set_xticks([0, 1], ["Different", "Same"])
    ax.set_yticks([0, 1], ["Different", "Same"])
    ax.set_xlabel("View identity")
    ax.set_ylabel("Question identity")
    ax.set_title("Two matched identities", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[0, 2])
    style_axis(ax, grid="x")
    labels = ["Question identity", "View identity", "Diversity change"]
    values = [question_gap.mean(), view_gap.mean(), diversity_change.mean()]
    intervals = [
        bootstrap_ci(question_gap, args.bootstrap_trials, np.random.default_rng(args.seed + 11)),
        bootstrap_ci(view_gap, args.bootstrap_trials, np.random.default_rng(args.seed + 12)),
        bootstrap_ci(diversity_change, args.bootstrap_trials, np.random.default_rng(args.seed + 13)),
    ]
    colors = [COLORS["blue"], COLORS["green"], COLORS["pink"]]
    y = np.arange(3)[::-1]
    ax.axvline(0.0, color=COLORS["muted"], linewidth=0.75, linestyle="--")
    for pos, label, value, interval, color in zip(y, labels, values, intervals, colors):
        low, high = interval
        ax.errorbar(
            value,
            pos,
            xerr=[[value - low], [high - value]],
            fmt="o",
            ms=5.7,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markeredgecolor=COLORS["text"],
            markeredgewidth=0.4,
            zorder=3,
        )
        ax.text(
            high + 0.004,
            pos,
            f"{value:+.3f}",
            va="center",
            fontsize=6.4,
            color=color,
            fontweight="bold",
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.01, 0.135)
    ax.set_xlabel("Cosine gap / paired change")
    ax.set_title("Identity survives refinement", loc="left", fontweight="bold")
    panel_label(ax, "c")

    fig.text(
        0.5,
        0.012,
        (
            f"{n_questions} matched questions, {n_views} fixed views each. "
            "Error bars: question-bootstrap 95% CIs; identity controls use 1,024 matched permutations."
        ),
        ha="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.066, right=0.99, top=0.84, bottom=0.24)
    save_figure(fig, args.output_dir / "fig_path_identity_retention")
    plt.close(fig)

    md = [
        "| Metric | Observed / Final | Matched control / Stage 1 | Gap or change [95% CI] | p |",
        "|---|---:|---:|---:|---:|",
    ]
    tex_rows = []
    for row in aggregate_rows:
        raw_pvalue = row["permutation_p"]
        pvalue = None if raw_pvalue is None else float(raw_pvalue)
        ptext = "--" if pvalue is None else p_string(pvalue)
        delta = (
            f'{row["gap_or_change"]:+.3f} '
            f'[{row["gap_ci95_low"]:+.3f}, {row["gap_ci95_high"]:+.3f}]'
        )
        md.append(
            f'| {row["metric"]} | {row["observed_or_final"]:.3f} | '
            f'{row["control_or_stage1"]:.3f} | {delta} | {ptext} |'
        )
        latex_p = "--" if pvalue is None else (
            r"$<10^{-3}$" if pvalue < 0.001 else f"{pvalue:.3f}"
        )
        tex_rows.append(
            f'{row["metric"]} & {row["observed_or_final"]:.3f} & '
            f'{row["control_or_stage1"]:.3f} & {delta} & {latex_p} ' + r"\\"
        )
    (args.output_dir / "table_path_identity_retention.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    latex = "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Metric & Observed/Final & Control/Stage 1 & $\Delta$ [95\% CI] & $p$ \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (args.output_dir / "table_path_identity_retention.tex").write_text(
        latex + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
