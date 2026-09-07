#!/usr/bin/env python3
"""Build SemCoT-style small-multiple comparisons for complete TRACE paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from trace_submission_figure_redesign import (
    COLORS,
    FIGURE_SERIF,
    cumulative_paths,
    load_aligned_records,
    padded_limits,
    project_paths,
    save_figure,
    select_outcome_blind_camera,
    write_csv,
)


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            FIGURE_SERIF,
            "Times New Roman",
            "Nimbus Roman No9 L",
            "Times",
            "Liberation Serif",
        ],
        "font.size": 7.0,
        "axes.titlesize": 8.2,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 6.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "legend.frameon": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-records", type=Path, required=True)
    parser.add_argument("--final-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=200)
    return parser.parse_args()


def correct_count(record: dict) -> int:
    return int((record["multiview_acc"].float().cpu().numpy() > 0.5).sum())


def select_cases(stage1: list[dict], final: list[dict]) -> list[dict]:
    counts = []
    for position, (stage1_record, final_record) in enumerate(zip(stage1, final)):
        stage1_correct = correct_count(stage1_record)
        final_correct = correct_count(final_record)
        counts.append(
            {
                "position": position,
                "idx": int(final_record["idx"]),
                "stage1_correct": stage1_correct,
                "final_correct": final_correct,
                "gain": final_correct - stage1_correct,
            }
        )

    maximum_gain = sorted(
        counts,
        key=lambda row: (-row["gain"], -row["final_correct"], row["idx"]),
    )[0]
    stable_all_correct = sorted(
        [
            row
            for row in counts
            if row["stage1_correct"] == 8 and row["final_correct"] == 8
        ],
        key=lambda row: row["idx"],
    )[0]
    majority_rescue = sorted(
        [
            row
            for row in counts
            if row["position"] not in {maximum_gain["position"], stable_all_correct["position"]}
            and row["stage1_correct"] < 4
            and 4 < row["final_correct"] < 8
        ],
        key=lambda row: (-row["gain"], -row["final_correct"], row["idx"]),
    )[0]

    cases = [
        {**maximum_gain, "selection": "Maximum rescue"},
        {**majority_rescue, "selection": "Majority rescue"},
        {**stable_all_correct, "selection": "Stable control"},
    ]
    return cases


def fit_global_pca_excluding(
    stage1: list[dict], final: list[dict], excluded_positions: set[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for records in (stage1, final):
        for position, record in enumerate(records):
            if position in excluded_positions:
                continue
            paths = cumulative_paths(record)[:, 1:, :]
            arrays.append(torch.from_numpy(paths.reshape(-1, paths.shape[-1])))
    points = torch.cat(arrays, dim=0).float()
    torch.manual_seed(0)
    mean = points.mean(dim=0)
    _, singular_values, components = torch.pca_lowrank(
        points, q=3, center=True, niter=5
    )
    centered_ss = ((points - mean) ** 2).sum().item()
    explained = (singular_values**2 / max(centered_ss, 1e-12)).cpu().numpy()
    return mean.cpu().numpy(), components.cpu().numpy(), explained


def prepare_projected_cases(
    stage1: list[dict], final: list[dict], cases: list[dict]
) -> tuple[list[dict], np.ndarray, float, float, float, dict[str, float]]:
    excluded = {int(case["position"]) for case in cases}
    mean, components, explained = fit_global_pca_excluding(stage1, final, excluded)
    projected_cases = []
    aggregate_displacement = []
    for case in cases:
        position = int(case["position"])
        stage1_projection = project_paths(
            cumulative_paths(stage1[position]), mean, components
        )
        final_projection = project_paths(
            cumulative_paths(final[position]), mean, components
        )
        aggregate_displacement.append(
            final_projection[:, -1].mean(axis=0)
            - final_projection[:, 0].mean(axis=0)
        )
        projected_cases.append(
            {
                **case,
                "stage1_projection": stage1_projection,
                "final_projection": final_projection,
                "stage1_outcomes": (
                    stage1[position]["multiview_acc"].float().cpu().numpy() > 0.5
                ),
                "final_outcomes": (
                    final[position]["multiview_acc"].float().cpu().numpy() > 0.5
                ),
            }
        )

    orientation_sign = 1.0
    if np.mean(aggregate_displacement, axis=0)[0] < 0:
        orientation_sign = -1.0
        for case in projected_cases:
            case["stage1_projection"][..., 0] *= -1.0
            case["final_projection"][..., 0] *= -1.0

    centered_paths = []
    for case in projected_cases:
        shared_origin = np.concatenate(
            [
                case["stage1_projection"][:, :1],
                case["final_projection"][:, :1],
            ],
            axis=0,
        ).mean(axis=0, keepdims=True)
        case["stage1_projection"] -= shared_origin
        case["final_projection"] -= shared_origin
        centered_paths.extend(
            [case["stage1_projection"], case["final_projection"]]
        )

    camera_paths = np.concatenate(centered_paths, axis=0)
    camera_elev, camera_azim, camera_score, camera_components = (
        select_outcome_blind_camera(camera_paths)
    )
    return (
        projected_cases,
        explained,
        orientation_sign,
        camera_elev,
        camera_azim,
        {"score": camera_score, **camera_components},
    )


def style_3d_axis(
    ax: plt.Axes,
    points: np.ndarray,
    elev: float,
    azim: float,
    *,
    show_axis_labels: bool = True,
) -> None:
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(*padded_limits(points[..., 0], padding=0.06))
    ax.set_ylim(*padded_limits(points[..., 1], padding=0.06))
    ax.set_zlim(*padded_limits(points[..., 2], padding=0.06))
    ranges = np.ptp(points, axis=(0, 1))
    ranges = np.clip(ranges, max(float(ranges.max()) * 0.52, 1e-6), None)
    ax.set_box_aspect(tuple(ranges), zoom=1.12)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
    ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
    if show_axis_labels:
        ax.set_xlabel("PC1", labelpad=-8)
        ax.set_ylabel("PC2", labelpad=-7)
        ax.set_zlabel("PC3", labelpad=-7)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    ax.tick_params(colors=COLORS["muted"], labelsize=5.2, pad=-1, width=0.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(COLORS["grid"])
        axis._axinfo["grid"].update(color=COLORS["grid"], linewidth=0.3)


def draw_paths(
    ax: plt.Axes,
    projection: np.ndarray,
    outcomes: np.ndarray,
    stage: str,
    *,
    show_step_markers: bool = True,
    endpoint_scale: float = 1.0,
) -> None:
    for view, (path, correct) in enumerate(zip(projection, outcomes)):
        if stage == "stage1":
            color = COLORS["blue"]
            linestyle = (0, (3.0, 2.0))
            alpha = 0.40
            linewidth = 0.90
        else:
            color = COLORS["pink"] if correct else COLORS["orange"]
            linestyle = "-"
            alpha = 0.78 if correct else 0.98
            linewidth = 1.25 if correct else 1.45
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2 if stage == "stage1" else 4,
        )
        if show_step_markers:
            ax.scatter(
                path[1:-1, 0],
                path[1:-1, 1],
                path[1:-1, 2],
                s=5.5,
                color=color,
                alpha=min(alpha + 0.08, 1.0),
                edgecolor=COLORS["white"],
                linewidth=0.2,
                zorder=5,
            )
        endpoint_color = COLORS["pink"] if correct else COLORS["orange"]
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            marker="*" if correct else "X",
            s=(36 if correct else 31) * endpoint_scale,
            color=endpoint_color,
            edgecolor=COLORS["text"],
            linewidth=0.4,
            zorder=8,
        )
    origin = projection[0, 0]
    ax.scatter(
        origin[0],
        origin[1],
        origin[2],
        marker="s",
        s=23 * endpoint_scale,
        color=COLORS["text"],
        edgecolor=COLORS["white"],
        linewidth=0.4,
        zorder=9,
    )


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS["blue"],
            linestyle=(0, (3.0, 2.0)),
            linewidth=1.4,
            label="Stage 1 path",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["pink"],
            linewidth=1.6,
            label="Final correct",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["orange"],
            linewidth=1.6,
            label="Final wrong",
        ),
    ]


def endpoint_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=COLORS["text"],
            markeredgecolor=COLORS["white"],
            label="shared start",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            markersize=7.5,
            linestyle="none",
            markerfacecolor=COLORS["pink"],
            markeredgecolor=COLORS["text"],
            label="correct endpoint",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            markersize=6.0,
            linestyle="none",
            markerfacecolor=COLORS["orange"],
            markeredgecolor=COLORS["text"],
            label="wrong endpoint",
        ),
    ]


def case_limits(case: dict) -> np.ndarray:
    return np.concatenate(
        [case["stage1_projection"], case["final_projection"]], axis=0
    )


def make_q45_split(
    cases: list[dict], output_dir: Path, elev: float, azim: float, explained: np.ndarray
) -> None:
    case = next(item for item in cases if item["selection"] == "Majority rescue")
    limits = case_limits(case)
    fig = plt.figure(figsize=(7.2, 2.74))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    draw_paths(axes[0], case["stage1_projection"], case["stage1_outcomes"], "stage1", endpoint_scale=1.15)
    draw_paths(axes[1], case["final_projection"], case["final_outcomes"], "final", endpoint_scale=1.15)
    style_3d_axis(axes[0], limits, elev, azim, show_axis_labels=True)
    style_3d_axis(axes[1], limits, elev, azim, show_axis_labels=False)
    axes[0].set_title(
        f"Stage 1 | {case['stage1_correct']}/8 correct",
        loc="left",
        fontweight="bold",
        color=COLORS["blue"],
        pad=4,
    )
    axes[1].set_title(
        f"TRACE Final | {case['final_correct']}/8 correct",
        loc="left",
        fontweight="bold",
        color=COLORS["pink"],
        pad=4,
    )
    axes[0].text2D(-0.09, 1.02, "a", transform=axes[0].transAxes, fontsize=9.2, fontweight="bold")
    axes[1].text2D(-0.09, 1.02, "b", transform=axes[1].transAxes, fontsize=9.2, fontweight="bold")
    fig.text(
        0.02,
        0.965,
        f"q{case['idx']} | majority rescue",
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.98,
        0.965,
        f"held-out global PCA: PC1-3 {100 * explained.sum():.1f}% | same camera and limits",
        ha="right",
        va="top",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    fig.legend(
        handles=endpoint_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        handlelength=1.0,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.82, bottom=0.12, wspace=-0.03)
    save_figure(fig, output_dir / "fig_trace_paths_q45_split")
    plt.close(fig)


def make_three_case_overlay(
    cases: list[dict], output_dir: Path, elev: float, azim: float, explained: np.ndarray
) -> None:
    fig = plt.figure(figsize=(7.2, 2.66))
    for panel, case in enumerate(cases, start=1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        draw_paths(ax, case["stage1_projection"], case["stage1_outcomes"], "stage1", endpoint_scale=0.9)
        draw_paths(ax, case["final_projection"], case["final_outcomes"], "final", endpoint_scale=0.9)
        style_3d_axis(ax, case_limits(case), elev, azim, show_axis_labels=panel == 1)
        ax.set_title(
            f"{case['selection']} | q{case['idx']}\n"
            f"{case['stage1_correct']}/8 to {case['final_correct']}/8 correct",
            loc="left",
            fontsize=7.5,
            fontweight="bold",
            pad=2,
        )
        ax.text2D(-0.10, 1.04, chr(96 + panel), transform=ax.transAxes, fontsize=8.8, fontweight="bold")
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        handlelength=2.0,
        columnspacing=1.1,
    )
    fig.text(
        0.98,
        0.975,
        f"held-out global PCA: PC1-3 {100 * explained.sum():.1f}% | one camera",
        ha="right",
        va="top",
        fontsize=5.6,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.13, wspace=-0.05)
    save_figure(fig, output_dir / "fig_trace_paths_three_cases_overlay")
    plt.close(fig)


def make_stage_by_case(
    cases: list[dict], output_dir: Path, elev: float, azim: float, explained: np.ndarray
) -> None:
    fig = plt.figure(figsize=(7.2, 4.05))
    for column, case in enumerate(cases):
        limits = case_limits(case)
        for row, stage in enumerate(("stage1", "final")):
            panel = row * 3 + column + 1
            ax = fig.add_subplot(2, 3, panel, projection="3d")
            projection = case["stage1_projection"] if stage == "stage1" else case["final_projection"]
            outcomes = case["stage1_outcomes"] if stage == "stage1" else case["final_outcomes"]
            draw_paths(ax, projection, outcomes, stage, endpoint_scale=1.0)
            style_3d_axis(
                ax,
                limits,
                elev,
                azim,
                show_axis_labels=(row == 1 and column == 0),
            )
            if row == 0:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            ax.text2D(
                0.01,
                0.97,
                chr(97 + row * 3 + column),
                transform=ax.transAxes,
                fontsize=7.8,
                fontweight="bold",
                va="top",
            )
    column_x = [0.183, 0.506, 0.829]
    for x_position, case in zip(column_x, cases):
        fig.text(
            x_position,
            0.975,
            f"{case['selection']} | q{case['idx']}\n"
            f"{case['stage1_correct']}/8 $\\rightarrow$ {case['final_correct']}/8 correct",
            ha="center",
            va="top",
            fontsize=7.4,
            fontweight="bold",
            color=COLORS["text"],
        )
    fig.text(0.015, 0.68, "Stage 1", rotation=90, ha="center", va="center", fontsize=8.4, fontweight="bold", color=COLORS["blue"])
    fig.text(0.015, 0.29, "TRACE Final", rotation=90, ha="center", va="center", fontsize=8.4, fontweight="bold", color=COLORS["pink"])
    fig.legend(
        handles=endpoint_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=3,
        handlelength=1.0,
        columnspacing=1.0,
        fontsize=5.9,
    )
    fig.text(
        0.985,
        0.022,
        f"global PCA PC1-3 {100 * explained.sum():.1f}% | same camera; each column shares limits across rows",
        ha="right",
        fontsize=5.3,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.035, right=0.995, top=0.88, bottom=0.08, wspace=-0.07, hspace=-0.10)
    save_figure(fig, output_dir / "fig_trace_paths_stage_by_case")
    plt.close(fig)


def write_source_data(cases: list[dict], output_dir: Path) -> None:
    rows = []
    for case in cases:
        for stage, projection, outcomes in (
            ("Stage 1", case["stage1_projection"], case["stage1_outcomes"]),
            ("Final", case["final_projection"], case["final_outcomes"]),
        ):
            for view in range(projection.shape[0]):
                for step in range(projection.shape[1]):
                    rows.append(
                        {
                            "selection": case["selection"],
                            "question_idx": case["idx"],
                            "stage": stage,
                            "view": view + 1,
                            "correct": int(outcomes[view]),
                            "latent_step": step,
                            "global_pc1_centered": float(projection[view, step, 0]),
                            "global_pc2_centered": float(projection[view, step, 1]),
                            "global_pc3_centered": float(projection[view, step, 2]),
                        }
                    )
    write_csv(output_dir / "source_data" / "trace_path_comparison_cases.csv", rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1, final = load_aligned_records(
        args.stage1_records, args.final_records, args.max_records
    )
    cases = select_cases(stage1, final)
    (
        projected_cases,
        explained,
        orientation_sign,
        camera_elev,
        camera_azim,
        camera_metrics,
    ) = prepare_projected_cases(stage1, final, cases)
    make_q45_split(
        projected_cases,
        args.output_dir,
        camera_elev,
        camera_azim,
        explained,
    )
    make_three_case_overlay(
        projected_cases,
        args.output_dir,
        camera_elev,
        camera_azim,
        explained,
    )
    make_stage_by_case(
        projected_cases,
        args.output_dir,
        camera_elev,
        camera_azim,
        explained,
    )
    write_source_data(projected_cases, args.output_dir)
    manifest = {
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key
                in {
                    "selection",
                    "position",
                    "idx",
                    "stage1_correct",
                    "final_correct",
                    "gain",
                }
            }
            for case in projected_cases
        ],
        "pca_fit_questions": len(stage1) - len(projected_cases),
        "pca_uses_outcome_labels": False,
        "pca_explained_variance_ratio": [float(value) for value in explained],
        "pc1_orientation_sign": orientation_sign,
        "question_centering": "translation by the shared projected z0 only",
        "path_offsets": False,
        "per_path_scaling": False,
        "camera_selection": "outcome-blind screen-space visibility objective over all displayed paths",
        "camera_elev": camera_elev,
        "camera_azim": camera_azim,
        "camera_metrics": camera_metrics,
        "candidates": [
            "fig_trace_paths_q45_split",
            "fig_trace_paths_three_cases_overlay",
            "fig_trace_paths_stage_by_case",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
