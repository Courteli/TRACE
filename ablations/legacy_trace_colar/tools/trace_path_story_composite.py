#!/usr/bin/env python3
"""Build the TRACE qualitative-plus-quantitative path story figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import torch

from trace_outcome_selected_path_atlas import (
    pca_fit,
    pca_project,
    residualized_paths,
    select_cases,
    template,
    visible_turn_path,
)


COLORS = {
    "blue": "#6687B8",
    "green": "#69B17D",
    "gold": "#E6A314",
    "pink": "#E5A6C4",
    "text": "#27313B",
    "muted": "#66717E",
    "grid": "#D9DEE5",
    "stage1": "#9CA6B2",
    "null": "#AAB2BC",
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
        "axes.titlesize": 8.1,
        "axes.labelsize": 7.1,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
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
    parser.add_argument("--identity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=200)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def panel_label_2d(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
        color=COLORS["text"],
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], width=0.7, length=2.5)


def draw_path_panel(
    ax: plt.Axes,
    case: dict,
    stage1_record: dict,
    final_record: dict,
    stage1_template: np.ndarray,
    final_template: np.ndarray,
    n_questions: int,
    panel_label: str,
) -> dict:
    stage1_paths = residualized_paths(stage1_record, stage1_template, n_questions)
    final_paths = residualized_paths(final_record, final_template, n_questions)
    fit_points = np.concatenate(
        [
            stage1_paths[:, 1:].reshape(-1, stage1_paths.shape[-1]),
            final_paths[:, 1:].reshape(-1, final_paths.shape[-1]),
        ],
        axis=0,
    )
    mean, components, explained = pca_fit(fit_points)
    stage1_projection = pca_project(
        stage1_paths.reshape(-1, stage1_paths.shape[-1]), mean, components
    ).reshape(8, 9, 2)
    final_projection = pca_project(
        final_paths.reshape(-1, final_paths.shape[-1]), mean, components
    ).reshape(8, 9, 2)
    view_colors = [
        COLORS["blue"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["green"],
        COLORS["gold"],
        COLORS["gold"],
        COLORS["pink"],
        COLORS["pink"],
    ]
    for view_idx in range(8):
        lane_column = view_idx % 4
        lane_row = view_idx // 4
        offset = np.asarray([(lane_column - 1.5) * 5.1, (lane_row - 0.5) * 6.0])
        latent_step = np.arange(9, dtype=np.float32)
        stage1_turn = visible_turn_path(stage1_projection[view_idx])
        final_turn = visible_turn_path(final_projection[view_idx])
        stage1_display = np.column_stack(
            [
                latent_step,
                offset[0] + stage1_turn[:, 0],
                offset[1] + stage1_turn[:, 1],
            ]
        )
        final_display = np.column_stack(
            [
                latent_step,
                offset[0] + final_turn[:, 0],
                offset[1] + final_turn[:, 1],
            ]
        )
        ax.plot(
            *stage1_display.T,
            color=COLORS["stage1"],
            linewidth=0.82,
            linestyle="--",
            alpha=0.72,
        )
        color = view_colors[view_idx]
        ax.plot(*final_display.T, color=color, linewidth=1.8, alpha=0.97)
        ax.scatter(
            *final_display.T,
            color=color,
            s=np.linspace(3.0, 12.5, 9),
            alpha=0.9,
            depthshade=False,
        )
        movement = np.diff(final_display, axis=0)
        ax.quiver(
            final_display[:-1, 0],
            final_display[:-1, 1],
            final_display[:-1, 2],
            movement[:, 0],
            movement[:, 1],
            movement[:, 2],
            color=color,
            alpha=0.68,
            linewidth=0.55,
            arrow_length_ratio=0.20,
        )
        ax.scatter(
            *final_display[0],
            color=COLORS["text"],
            s=7.5,
            marker="s",
            alpha=0.72,
            depthshade=False,
        )
        ax.scatter(
            *final_display[-1],
            color=color,
            s=23,
            edgecolor="white",
            linewidth=0.45,
            depthshade=False,
        )

    concise_selection = (
        "Maximum reliability gain"
        if case["selection"] == "Maximum reliability gain"
        else "Majority rescue"
    )
    ax.set_title(
        f'{concise_selection} | q{case["idx"]}\n'
        f'{case["stage1_correct"]}/8 $\\rightarrow$ {case["final_correct"]}/8 correct paths',
        fontsize=7.5,
        fontweight="bold",
        pad=1.5,
    )
    ax.set_xlim(-0.35, 8.45)
    ax.set_ylim(-10.4, 10.4)
    ax.set_zlim(-5.6, 5.6)
    ax.set_box_aspect((1.45, 1.08, 0.72), zoom=1.13)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-67)
    ax.set_xticks([0, 4, 8], ["z0", "z4", "z8"])
    ax.set_xlabel("latent step", labelpad=-4)
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(COLORS["grid"])
        axis._axinfo["grid"].update(color=COLORS["grid"], linewidth=0.30)
    ax.text2D(
        -0.02,
        1.02,
        panel_label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["text"],
    )
    return {
        **case,
        "pca_pc1_explained": float(explained[0]),
        "pca_pc2_explained": float(explained[1]),
    }


def main() -> None:
    args = parse_args()
    stage1_list = torch.load(
        args.stage1_records, map_location="cpu", weights_only=False
    )[: args.max_records]
    final_list = torch.load(
        args.final_records, map_location="cpu", weights_only=False
    )[: args.max_records]
    stage1 = {int(record["idx"]): record for record in stage1_list}
    final = {int(record["idx"]): record for record in final_list}
    indices = sorted(set(stage1) & set(final))
    if len(indices) != args.max_records:
        raise ValueError(f"Expected {args.max_records} aligned questions, found {len(indices)}")
    all_cases = select_cases(stage1, final, indices)
    cases = [all_cases[0], all_cases[2]]
    stage1_template = template(stage1, indices)
    final_template = template(final, indices)

    matrix_rows = read_csv(
        args.identity_dir / "source_data" / "path_identity_view_matrix.csv"
    )
    summary_rows = read_csv(
        args.identity_dir / "source_data" / "path_identity_summary.csv"
    )
    n_views = max(int(row["stage1_view"]) for row in matrix_rows)
    matrix = np.empty((n_views, n_views), dtype=np.float64)
    for row in matrix_rows:
        matrix[int(row["stage1_view"]) - 1, int(row["final_view"]) - 1] = float(
            row["mean_cosine_similarity"]
        )

    summary = {row["metric"]: row for row in summary_rows}
    metric_order = ["Question identity", "View identity", "Cross-view diversity"]
    metric_labels = ["Question identity", "View identity", "Diversity change"]
    metric_colors = [COLORS["blue"], COLORS["green"], COLORS["pink"]]

    fig = plt.figure(figsize=(7.2, 4.65))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.17, 0.83],
        left=0.045,
        right=0.988,
        top=0.95,
        bottom=0.105,
        hspace=0.38,
        wspace=0.28,
    )
    case_metadata = []
    for panel, case in enumerate(cases):
        ax = fig.add_subplot(grid[0, panel], projection="3d")
        case_metadata.append(
            draw_path_panel(
                ax,
                case,
                stage1[case["idx"]],
                final[case["idx"]],
                stage1_template,
                final_template,
                len(indices),
                chr(ord("a") + panel),
            )
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["stage1"],
            linestyle="--",
            linewidth=1.2,
            label="TRACE Stage 1",
        )
    ]
    for color, label in zip(
        (COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["pink"]),
        ("Final views 1-2", "Final views 3-4", "Final views 5-6", "Final views 7-8"),
    ):
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.8, label=label)
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.522),
        ncol=5,
        handlelength=1.6,
        columnspacing=1.35,
    )

    ax = fig.add_subplot(grid[1, 0])
    identity_cmap = LinearSegmentedColormap.from_list(
        "trace_identity", [COLORS["white"], COLORS["pink"], COLORS["blue"]]
    )
    image = ax.imshow(matrix, cmap=identity_cmap, vmin=0.70, vmax=0.98, aspect="equal")
    for view in range(n_views):
        ax.add_patch(
            plt.Rectangle(
                (view - 0.5, view - 0.5),
                1,
                1,
                fill=False,
                edgecolor=COLORS["gold"],
                linewidth=1.0,
            )
        )
    ax.set_xticks(np.arange(n_views), np.arange(1, n_views + 1))
    ax.set_yticks(np.arange(n_views), np.arange(1, n_views + 1))
    ax.set_xlabel("Final view")
    ax.set_ylabel("Stage 1 view")
    ax.set_title("Matched view identity persists", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.ax.set_title("cos.", fontsize=5.8, color=COLORS["muted"], pad=2)
    colorbar.ax.tick_params(labelsize=5.8, length=2)
    panel_label_2d(ax, "c")

    ax = fig.add_subplot(grid[1, 1])
    style_axis(ax)
    y_positions = np.arange(3)[::-1]
    ax.axvline(0.0, color=COLORS["muted"], linewidth=0.75, linestyle="--")
    for y, metric, label, color in zip(
        y_positions, metric_order, metric_labels, metric_colors
    ):
        row = summary[metric]
        value = float(row["gap_or_change"])
        low = float(row["gap_ci95_low"])
        high = float(row["gap_ci95_high"])
        ax.errorbar(
            value,
            y,
            xerr=[[value - low], [high - value]],
            fmt="o",
            ms=5.8,
            color=color,
            ecolor=color,
            elinewidth=1.05,
            capsize=2.2,
            markeredgecolor=COLORS["text"],
            markeredgewidth=0.4,
            zorder=3,
        )
        ax.text(
            high + 0.004,
            y,
            f"{value:+.3f}",
            va="center",
            fontsize=6.5,
            color=color,
            fontweight="bold",
        )
    ax.set_yticks([])
    for y, label in zip(y_positions, metric_labels):
        label_x = 0.030 if label == "Diversity change" else 0.014
        ax.text(
            label_x,
            y,
            label,
            va="center",
            ha="left",
            fontsize=6.7,
            color=COLORS["muted"],
        )
    ax.set_xlim(-0.01, 0.135)
    ax.set_ylim(-0.35, 2.35)
    ax.set_xlabel("Cosine gap / paired change")
    ax.set_title(
        "Identity survives; diversity does not collapse",
        loc="left",
        fontweight="bold",
    )
    panel_label_2d(ax, "d")

    fig.text(
        0.5,
        0.025,
        (
            "Panels a-b: outcome-only case selection; fixed lanes and display-normalized step magnitudes expose turns but are not distance evidence. "
            "Panels c-d: 200 matched questions; error bars are question-bootstrap 95% CIs."
        ),
        ha="center",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output_dir / "fig_trace_path_story")
    plt.close(fig)

    # Keep the qualitative examples and the population evidence as separate
    # manuscript figures so that each visual answers only one question.
    cases_fig = plt.figure(figsize=(7.2, 3.05))
    cases_grid = cases_fig.add_gridspec(
        1,
        2,
        left=0.018,
        right=0.988,
        top=0.93,
        bottom=0.22,
        wspace=0.015,
    )
    for panel, case in enumerate(cases):
        ax = cases_fig.add_subplot(cases_grid[0, panel], projection="3d")
        draw_path_panel(
            ax,
            case,
            stage1[case["idx"]],
            final[case["idx"]],
            stage1_template,
            final_template,
            len(indices),
            chr(ord("a") + panel),
        )
    cases_fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
        ncol=5,
        handlelength=1.7,
        columnspacing=1.45,
    )
    cases_fig.text(
        0.5,
        0.025,
        (
            "Cases are selected by correctness transitions only. Fixed view lanes and "
            "display-normalized step magnitudes expose turns but are not distance evidence."
        ),
        ha="center",
        fontsize=6.2,
        color=COLORS["muted"],
    )
    save_figure(cases_fig, args.output_dir / "fig_trace_path_cases")
    plt.close(cases_fig)

    identity_fig = plt.figure(figsize=(7.2, 2.84))
    identity_grid = identity_fig.add_gridspec(
        2,
        2,
        width_ratios=[0.92, 1.35],
        height_ratios=[1.12, 0.72],
        left=0.065,
        right=0.985,
        top=0.88,
        bottom=0.23,
        hspace=0.58,
        wspace=0.34,
    )

    ax = identity_fig.add_subplot(identity_grid[:, 0])
    image = ax.imshow(
        matrix,
        cmap=identity_cmap,
        vmin=0.70,
        vmax=0.98,
        aspect="equal",
    )
    for view in range(n_views):
        ax.add_patch(
            plt.Rectangle(
                (view - 0.5, view - 0.5),
                1,
                1,
                fill=False,
                edgecolor=COLORS["gold"],
                linewidth=1.15,
            )
        )
    ax.set_xticks(np.arange(n_views), np.arange(1, n_views + 1))
    ax.set_yticks(np.arange(n_views), np.arange(1, n_views + 1))
    ax.set_xlabel("Final view")
    ax.set_ylabel("Stage 1 view")
    ax.set_title("View identity across stages", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    colorbar = identity_fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.ax.set_title("cos.", fontsize=6.0, color=COLORS["muted"], pad=2)
    colorbar.ax.tick_params(labelsize=6.0, length=2)
    panel_label_2d(ax, "a")

    ax = identity_fig.add_subplot(identity_grid[0, 1])
    style_axis(ax)
    identity_metrics = [
        ("Question identity", "Question identity", COLORS["blue"]),
        ("View identity", "View identity", COLORS["green"]),
    ]
    y_positions = np.asarray([1.0, 0.0])
    for y, (metric, label, color) in zip(y_positions, identity_metrics):
        row = summary[metric]
        matched = float(row["observed_or_final"])
        control = float(row["control_or_stage1"])
        gap = float(row["gap_or_change"])
        low = float(row["gap_ci95_low"])
        high = float(row["gap_ci95_high"])
        ax.plot([control, matched], [y, y], color=COLORS["null"], linewidth=2.0, zorder=1)
        ax.scatter(
            control,
            y,
            marker="s",
            s=31,
            color=COLORS["gold"],
            edgecolor=COLORS["text"],
            linewidth=0.4,
            zorder=3,
        )
        ax.errorbar(
            matched,
            y,
            xerr=[[gap - low], [high - gap]],
            fmt="o",
            ms=6.3,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.5,
            markeredgecolor=COLORS["text"],
            markeredgewidth=0.45,
            zorder=3,
        )
        ax.text(
            0.5 * (control + matched),
            y + 0.20,
            f"gap {gap:+.3f} [{low:.3f}, {high:.3f}]",
            va="bottom",
            ha="center",
            fontsize=6.0,
            color=color,
            fontweight="bold",
        )
        ax.text(
            control,
            y - 0.20,
            f"control {control:.3f}",
            va="top",
            ha="center",
            fontsize=5.3,
            color=COLORS["muted"],
        )
        ax.text(
            matched,
            y - 0.20,
            f"matched {matched:.3f}",
            va="top",
            ha="center",
            fontsize=5.3,
            color=color,
        )
    ax.set_yticks(y_positions, [item[1] for item in identity_metrics])
    ax.set_xlim(0.83, 0.98)
    ax.set_ylim(-0.42, 1.42)
    ax.set_xlabel("Cross-stage path similarity")
    ax.set_title("Matched paths beat identity controls", loc="left", fontweight="bold")
    panel_label_2d(ax, "b")

    ax = identity_fig.add_subplot(identity_grid[1, 1])
    style_axis(ax)
    diversity = summary["Cross-view diversity"]
    stage1_diversity = float(diversity["control_or_stage1"])
    final_diversity = float(diversity["observed_or_final"])
    change = float(diversity["gap_or_change"])
    change_low = float(diversity["gap_ci95_low"])
    change_high = float(diversity["gap_ci95_high"])
    ax.plot(
        [stage1_diversity, final_diversity],
        [0.0, 0.0],
        color=COLORS["null"],
        linewidth=2.0,
        zorder=1,
    )
    ax.scatter(
        stage1_diversity,
        0.0,
        s=37,
        color=COLORS["green"],
        edgecolor=COLORS["text"],
        linewidth=0.4,
        zorder=3,
    )
    ax.errorbar(
        final_diversity,
        0.0,
        xerr=[[change - change_low], [change_high - change]],
        fmt="o",
        ms=6.3,
        color=COLORS["pink"],
        ecolor=COLORS["pink"],
        elinewidth=1.15,
        capsize=2.4,
        markeredgecolor=COLORS["text"],
        markeredgewidth=0.45,
        zorder=4,
    )
    ax.text(
        0.5 * (stage1_diversity + final_diversity),
        0.20,
        f"change {change:+.4f} [{change_low:.4f}, {change_high:.4f}]",
        ha="center",
        va="bottom",
        fontsize=5.9,
        color=COLORS["pink"],
        fontweight="bold",
    )
    ax.text(
        stage1_diversity,
        -0.19,
        f"Stage 1  {stage1_diversity:.3f}",
        ha="right",
        va="top",
        fontsize=5.3,
        color=COLORS["green"],
    )
    ax.text(
        final_diversity,
        -0.19,
        f"Final  {final_diversity:.3f}",
        ha="left",
        va="top",
        fontsize=5.3,
        color=COLORS["pink"],
    )
    ax.set_xlim(0.0715, 0.0825)
    ax.set_ylim(-0.36, 0.36)
    ax.set_yticks([])
    ax.set_xlabel("Cross-view diversity (1 - cosine)")
    ax.set_title("Diversity is retained", loc="left", fontweight="bold")
    panel_label_2d(ax, "c")
    identity_fig.text(
        0.5,
        0.045,
        (
            "200 matched questions; error bars are question-bootstrap 95% CIs; "
            "identity tests use 1,024 matched permutations."
        ),
        ha="center",
        fontsize=6.2,
        color=COLORS["muted"],
    )
    save_figure(
        identity_fig,
        args.output_dir / "fig_trace_path_identity_summary",
    )
    plt.close(identity_fig)

    write_csv(
        args.output_dir / "source_data" / "trace_path_story_cases.csv",
        case_metadata,
    )
    write_csv(
        args.output_dir / "source_data" / "trace_path_story_view_matrix.csv",
        matrix_rows,
    )
    write_csv(
        args.output_dir / "source_data" / "trace_path_story_identity_summary.csv",
        summary_rows,
    )
    manifest = {
        "figure": "TRACE path story figures",
        "manuscript_outputs": [
            "fig_trace_path_cases",
            "fig_trace_path_identity_summary",
        ],
        "archived_composite_output": "fig_trace_path_story",
        "n_population_questions": len(indices),
        "n_fixed_views": n_views,
        "case_selection_uses_geometry": False,
        "case_selection": [case["selection"] for case in cases],
        "path_projection": (
            "per-question PCA jointly fitted to Stage 1 and Final after "
            "outcome-free population-template residualization"
        ),
        "display_transform": (
            "fixed view lanes with direction-preserving step-ratio compression "
            "and fixed radial extent"
        ),
        "quantitative_panels": (
            "all-question cross-stage path identity with matched controls and "
            "question-bootstrap confidence intervals"
        ),
        "claim_boundary": (
            "3D paths are qualitative; the standalone identity summary and the "
            "permutation analyses carry the population claim"
        ),
        "cases": case_metadata,
    }
    (args.output_dir / "trace_path_story.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
