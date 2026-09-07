#!/usr/bin/env python3
"""Iterate outcome-legible, SemCoT-style views of complete TRACE paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from trace_global_pca_comparison_iterations import (
    prepare_projected_cases,
    style_3d_axis,
)
from trace_submission_figure_redesign import (
    COLORS,
    FIGURE_SERIF,
    camera_projection,
    load_aligned_records,
    save_figure,
    write_csv,
)


WRONG = "#C98200"


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
        "font.size": 7.1,
        "axes.titlesize": 8.2,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "legend.fontsize": 6.2,
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


def select_mixed_outcome_cases(
    stage1: list[dict], final: list[dict]
) -> list[dict]:
    rows = []
    for position, (stage1_record, final_record) in enumerate(zip(stage1, final)):
        stage1_correct = correct_count(stage1_record)
        final_correct = correct_count(final_record)
        rows.append(
            {
                "position": position,
                "idx": int(final_record["idx"]),
                "stage1_correct": stage1_correct,
                "final_correct": final_correct,
                "gain": final_correct - stage1_correct,
            }
        )

    specifications = [
        ("Near-consensus rescue", 7),
        ("Majority rescue", 6),
        ("Balanced rescue", 4),
    ]
    cases = []
    used_positions: set[int] = set()
    for label, target_final_count in specifications:
        candidates = [
            row
            for row in rows
            if row["final_correct"] == target_final_count
            and row["gain"] > 0
            and row["position"] not in used_positions
        ]
        selected = sorted(
            candidates,
            key=lambda row: (-row["gain"], row["idx"]),
        )[0]
        cases.append({**selected, "selection": label})
        used_positions.add(int(selected["position"]))
    return cases


def final_limits(case: dict) -> np.ndarray:
    return case["final_projection"]


def combined_limits(case: dict) -> np.ndarray:
    return np.concatenate(
        [case["stage1_projection"], case["final_projection"]], axis=0
    )


def display_scaled_paths(paths: np.ndarray) -> np.ndarray:
    """Match the axis-box scaling used by the rendered 3D panel."""
    minimum = paths.min(axis=(0, 1))
    maximum = paths.max(axis=(0, 1))
    spans = np.maximum(maximum - minimum, 1e-6)
    box_spans = np.clip(spans, max(float(spans.max()) * 0.52, 1e-6), None)
    return (paths - (minimum + maximum) / 2.0) * (box_spans / spans)


def endpoint_camera_metrics(
    cases: list[dict],
    elev: float,
    azim: float,
) -> dict[str, object]:
    mean_nearest = []
    lower_quartile_nearest = []
    path_retention = []
    for case in cases:
        points = display_scaled_paths(case["final_projection"])
        projected = camera_projection(points, elev, azim)
        diagonal = max(
            float(np.linalg.norm(np.ptp(projected.reshape(-1, 2), axis=0))),
            1e-6,
        )
        endpoints = projected[:, -1]
        distances = np.linalg.norm(
            endpoints[:, None, :] - endpoints[None, :, :],
            axis=-1,
        )
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1) / diagonal
        mean_nearest.append(float(nearest.mean()))
        lower_quartile_nearest.append(float(np.quantile(nearest, 0.25)))

        projected_length = float(
            np.linalg.norm(np.diff(projected, axis=1), axis=-1)
            .sum(axis=1)
            .mean()
        )
        spatial_length = float(
            np.linalg.norm(np.diff(points, axis=1), axis=-1)
            .sum(axis=1)
            .mean()
        )
        path_retention.append(projected_length / max(spatial_length, 1e-6))

    score = 0.65 * float(np.mean(mean_nearest)) + 0.35 * float(
        np.mean(lower_quartile_nearest)
    )
    return {
        "score": score,
        "mean_nearest_endpoint_distance": mean_nearest,
        "lower_quartile_nearest_endpoint_distance": lower_quartile_nearest,
        "path_length_retention": path_retention,
    }


def select_endpoint_readable_camera(
    cases: list[dict],
    *,
    minimum_path_retention: float = 0.80,
) -> tuple[float, float, dict[str, object]]:
    """Maximize endpoint visibility without labels or path foreshortening."""
    best: tuple[float, float, float, dict[str, object]] | None = None
    for elev in np.arange(0.0, 81.0, 2.0):
        for azim in np.arange(-180.0, 181.0, 2.0):
            metrics = endpoint_camera_metrics(cases, float(elev), float(azim))
            if min(metrics["path_length_retention"]) < minimum_path_retention:
                continue
            candidate = (
                float(metrics["score"]),
                float(elev),
                float(azim),
                metrics,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None

    coarse_elev, coarse_azim = best[1], best[2]
    for elev in np.arange(max(0.0, coarse_elev - 2.0), min(80.0, coarse_elev + 2.0) + 0.1, 1.0):
        for azim in np.arange(coarse_azim - 2.0, coarse_azim + 2.1, 1.0):
            metrics = endpoint_camera_metrics(cases, float(elev), float(azim))
            if min(metrics["path_length_retention"]) < minimum_path_retention:
                continue
            candidate = (
                float(metrics["score"]),
                float(elev),
                float(azim),
                metrics,
            )
            if candidate[0] > best[0]:
                best = candidate
    return best[1], best[2], best[3]


def endpoint_crop_metrics(
    case: dict,
    elev: float,
    azim: float,
) -> dict[str, float]:
    """Measure terminal visibility after the endpoint-only crop is applied."""
    points = display_scaled_paths(case["final_projection"])
    endpoints = camera_projection(points, elev, azim)[:, -1]
    distances = np.linalg.norm(
        endpoints[:, None, :] - endpoints[None, :, :],
        axis=-1,
    )
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    crop_span = max(
        float(np.ptp(endpoints[:, 0])),
        float(np.ptp(endpoints[:, 1])),
        1e-6,
    )
    normalized = nearest / crop_span
    minimum = float(normalized.min())
    lower_quartile = float(np.quantile(normalized, 0.25))
    mean = float(normalized.mean())
    return {
        "score": 0.60 * minimum + 0.25 * lower_quartile + 0.15 * mean,
        "minimum_nearest_endpoint_distance": minimum,
        "lower_quartile_nearest_endpoint_distance": lower_quartile,
        "mean_nearest_endpoint_distance": mean,
    }


def select_endpoint_crop_camera(
    case: dict,
) -> tuple[float, float, dict[str, float]]:
    """Choose a label-free view that prioritizes the most crowded endpoint."""
    best: tuple[float, float, float, dict[str, float]] | None = None
    for elev in np.arange(0.0, 81.0, 2.0):
        for azim in np.arange(-180.0, 181.0, 2.0):
            metrics = endpoint_crop_metrics(case, float(elev), float(azim))
            candidate = (
                float(metrics["score"]),
                float(elev),
                float(azim),
                metrics,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None

    coarse_elev, coarse_azim = best[1], best[2]
    for elev in np.arange(
        max(0.0, coarse_elev - 2.0),
        min(80.0, coarse_elev + 2.0) + 0.1,
        0.5,
    ):
        for azim in np.arange(
            coarse_azim - 2.0,
            coarse_azim + 2.1,
            0.5,
        ):
            metrics = endpoint_crop_metrics(case, float(elev), float(azim))
            candidate = (
                float(metrics["score"]),
                float(elev),
                float(azim),
                metrics,
            )
            if candidate[0] > best[0]:
                best = candidate
    return best[1], best[2], best[3]


def draw_start(ax: plt.Axes, projection: np.ndarray, scale: float = 1.0) -> None:
    origin = projection[0, 0]
    ax.scatter(
        origin[0],
        origin[1],
        origin[2],
        marker="s",
        s=24 * scale,
        color=COLORS["text"],
        edgecolor=COLORS["white"],
        linewidth=0.45,
        zorder=20,
    )


def draw_stage1_context(ax: plt.Axes, projection: np.ndarray) -> None:
    for path in projection:
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=COLORS["blue"],
            linestyle=(0, (3.0, 2.2)),
            linewidth=0.72,
            alpha=0.20,
            zorder=1,
        )


def draw_final_paths(
    ax: plt.Axes,
    projection: np.ndarray,
    outcomes: np.ndarray,
    *,
    include_correct: bool = True,
    include_wrong: bool = True,
    endpoint_scale: float = 1.0,
) -> None:
    # Correct paths form the translucent context. Wrong paths are deliberately
    # drawn last with a white under-stroke so genuine overlap remains visible.
    for path, correct in zip(projection, outcomes):
        if not correct or not include_correct:
            continue
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=COLORS["pink"],
            linestyle="-",
            linewidth=1.18,
            alpha=0.65,
            zorder=4,
        )
        ax.scatter(
            path[1:-1, 0],
            path[1:-1, 1],
            path[1:-1, 2],
            marker="o",
            s=6.2,
            color=COLORS["pink"],
            alpha=0.82,
            edgecolor=COLORS["white"],
            linewidth=0.25,
            zorder=5,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            marker="*",
            s=44 * endpoint_scale,
            color=COLORS["pink"],
            edgecolor=COLORS["text"],
            linewidth=0.45,
            zorder=12,
        )

    for path, correct in zip(projection, outcomes):
        if correct or not include_wrong:
            continue
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=COLORS["white"],
            linestyle="-",
            linewidth=3.3,
            alpha=0.98,
            zorder=8,
        )
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=WRONG,
            linestyle=(0, (4.0, 1.55)),
            linewidth=2.05,
            alpha=1.0,
            zorder=9,
        )
        ax.scatter(
            path[1:-1, 0],
            path[1:-1, 1],
            path[1:-1, 2],
            marker="D",
            s=9.5,
            color=WRONG,
            alpha=1.0,
            edgecolor=COLORS["white"],
            linewidth=0.35,
            zorder=10,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            marker="X",
            s=42 * endpoint_scale,
            color=WRONG,
            edgecolor=COLORS["text"],
            linewidth=0.5,
            zorder=13,
        )
    draw_start(ax, projection, endpoint_scale)


def add_endpoint_zoom(
    ax: plt.Axes,
    projection: np.ndarray,
    outcomes: np.ndarray,
    elev: float,
    azim: float,
) -> None:
    points = display_scaled_paths(projection)
    endpoints = camera_projection(points, elev, azim)[:, -1]
    inset = ax.inset_axes([0.64, 0.48, 0.32, 0.29])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.94))
    for endpoint, correct in zip(endpoints, outcomes):
        inset.scatter(
            endpoint[0],
            endpoint[1],
            marker="o" if correct else "D",
            s=9.5 if correct else 10.5,
            color=COLORS["pink"] if correct else WRONG,
            edgecolor=COLORS["text"],
            linewidth=0.32,
            zorder=3,
        )
    x_low, x_high = endpoints[:, 0].min(), endpoints[:, 0].max()
    y_low, y_high = endpoints[:, 1].min(), endpoints[:, 1].max()
    x_span = max(float(x_high - x_low), 1e-6)
    y_span = max(float(y_high - y_low), 1e-6)
    inset.set_xlim(x_low - 0.22 * x_span, x_high + 0.22 * x_span)
    inset.set_ylim(y_low - 0.22 * y_span, y_high + 0.22 * y_span)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(
        "endpoint view",
        fontsize=5.0,
        color=COLORS["muted"],
        pad=1.0,
    )
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(0.6)


def outcome_handles(*, include_stage1: bool = False) -> list[Line2D]:
    handles = []
    if include_stage1:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["blue"],
                linestyle=(0, (3.0, 2.2)),
                linewidth=1.2,
                alpha=0.55,
                label="Stage 1 reference",
            )
        )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=COLORS["pink"],
                marker="o",
                markerfacecolor=COLORS["pink"],
                markeredgecolor=COLORS["white"],
                linewidth=1.8,
                label="Final correct path",
            ),
            Line2D(
                [0],
                [0],
                color=WRONG,
                linestyle=(0, (4.0, 1.55)),
                marker="D",
                markerfacecolor=WRONG,
                markeredgecolor=COLORS["white"],
                linewidth=2.1,
                label="Final wrong path",
            ),
            Line2D(
                [0],
                [0],
                color="none",
                marker="s",
                markerfacecolor=COLORS["text"],
                markeredgecolor=COLORS["white"],
                label="Shared start",
            ),
        ]
    )
    return handles


def add_case_title(ax: plt.Axes, case: dict, panel: int) -> None:
    wrong_count = 8 - int(case["final_correct"])
    ax.set_title(
        f"{case['selection']} | q{case['idx']}\n"
        f"Stage 1 {case['stage1_correct']}/8 $\\rightarrow$ "
        f"Final {case['final_correct']}/8",
        loc="left",
        fontsize=7.35,
        fontweight="bold",
        pad=3,
        color=COLORS["text"],
    )
    ax.text2D(
        0.98,
        0.90,
        f"{case['final_correct']} correct",
        transform=ax.transAxes,
        fontsize=6.0,
        fontweight="bold",
        color=COLORS["pink"],
        ha="right",
    )
    ax.text2D(
        0.98,
        0.83,
        f"{wrong_count} wrong",
        transform=ax.transAxes,
        fontsize=6.0,
        fontweight="bold",
        color=WRONG,
        ha="right",
    )
    ax.text2D(
        -0.10,
        1.04,
        chr(97 + panel),
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
    )


def make_final_outcome_overlay(
    cases: list[dict],
    output_dir: Path,
    elev: float,
    azim: float,
    explained: np.ndarray,
    *,
    endpoint_zoom_cameras: list[tuple[float, float]],
) -> None:
    fig = plt.figure(figsize=(7.2, 2.48))
    for panel, case in enumerate(cases):
        ax = fig.add_subplot(1, 3, panel + 1, projection="3d")
        draw_final_paths(
            ax,
            case["final_projection"],
            case["final_outcomes"],
            endpoint_scale=1.05,
        )
        add_endpoint_zoom(
            ax,
            case["final_projection"],
            case["final_outcomes"],
            endpoint_zoom_cameras[panel][0],
            endpoint_zoom_cameras[panel][1],
        )
        style_3d_axis(
            ax,
            final_limits(case),
            elev,
            azim,
            show_axis_labels=panel == 0,
        )
        add_case_title(ax, case, panel)
    fig.legend(
        handles=outcome_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=3,
        handlelength=2.25,
        columnspacing=1.25,
    )
    fig.text(
        0.985,
        0.982,
        f"held-out global PCA PC1-3 {100 * explained.sum():.1f}% | "
        "fixed main camera, label-free endpoint views",
        ha="right",
        va="top",
        fontsize=5.5,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(
        left=0.012,
        right=0.992,
        top=0.825,
        bottom=0.135,
        wspace=-0.045,
    )
    save_figure(fig, output_dir / "fig_final_correct_wrong_overlay")
    plt.close(fig)


def make_outcome_rows(
    cases: list[dict],
    output_dir: Path,
    elev: float,
    azim: float,
    explained: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(7.2, 4.02))
    for column, case in enumerate(cases):
        limits = final_limits(case)
        for row, outcome_name in enumerate(("correct", "wrong")):
            panel = row * 3 + column
            ax = fig.add_subplot(2, 3, panel + 1, projection="3d")
            draw_final_paths(
                ax,
                case["final_projection"],
                case["final_outcomes"],
                include_correct=outcome_name == "correct",
                include_wrong=outcome_name == "wrong",
                endpoint_scale=1.1,
            )
            style_3d_axis(
                ax,
                limits,
                elev,
                azim,
                show_axis_labels=row == 1 and column == 0,
            )
            if row == 0:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            ax.text2D(
                0.01,
                0.97,
                chr(97 + panel),
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
            f"{case['final_correct']} correct / {8 - case['final_correct']} wrong",
            ha="center",
            va="top",
            fontsize=7.3,
            fontweight="bold",
            color=COLORS["text"],
        )
    fig.text(
        0.015,
        0.68,
        "Final correct paths",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["pink"],
    )
    fig.text(
        0.015,
        0.29,
        "Final wrong paths",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=WRONG,
    )
    fig.legend(
        handles=outcome_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.001),
        ncol=3,
        handlelength=2.25,
        columnspacing=1.2,
        fontsize=5.9,
    )
    fig.text(
        0.985,
        0.022,
        f"global PCA PC1-3 {100 * explained.sum():.1f}% | each column shares limits across rows",
        ha="right",
        fontsize=5.2,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(
        left=0.035,
        right=0.995,
        top=0.88,
        bottom=0.08,
        wspace=-0.07,
        hspace=-0.10,
    )
    save_figure(fig, output_dir / "fig_final_correct_wrong_rows")
    plt.close(fig)


def make_stage1_context_overlay(
    cases: list[dict],
    output_dir: Path,
    elev: float,
    azim: float,
    explained: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(7.2, 2.72))
    for panel, case in enumerate(cases):
        ax = fig.add_subplot(1, 3, panel + 1, projection="3d")
        draw_stage1_context(ax, case["stage1_projection"])
        draw_final_paths(
            ax,
            case["final_projection"],
            case["final_outcomes"],
            endpoint_scale=1.05,
        )
        style_3d_axis(
            ax,
            combined_limits(case),
            elev,
            azim,
            show_axis_labels=panel == 0,
        )
        add_case_title(ax, case, panel)
    fig.legend(
        handles=outcome_handles(include_stage1=True),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=4,
        handlelength=1.9,
        columnspacing=0.9,
        fontsize=5.8,
    )
    fig.text(
        0.985,
        0.975,
        f"held-out global PCA PC1-3 {100 * explained.sum():.1f}% | one camera",
        ha="right",
        va="top",
        fontsize=5.5,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(
        left=0.012,
        right=0.992,
        top=0.84,
        bottom=0.13,
        wspace=-0.045,
    )
    save_figure(fig, output_dir / "fig_final_outcomes_with_stage1_context")
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
    write_csv(
        output_dir / "source_data" / "outcome_legible_complete_paths.csv",
        rows,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1, final = load_aligned_records(
        args.stage1_records,
        args.final_records,
        args.max_records,
    )
    cases = select_mixed_outcome_cases(stage1, final)
    (
        projected_cases,
        explained,
        orientation_sign,
        path_camera_elev,
        path_camera_azim,
        path_camera_metrics,
    ) = prepare_projected_cases(stage1, final, cases)
    camera_elev, camera_azim, camera_metrics = (
        select_endpoint_readable_camera(
            projected_cases,
            minimum_path_retention=0.80,
        )
    )
    endpoint_zoom_results = [
        select_endpoint_crop_camera(case) for case in projected_cases
    ]
    endpoint_zoom_cameras = [
        (result[0], result[1]) for result in endpoint_zoom_results
    ]
    endpoint_zoom_metrics = [result[2] for result in endpoint_zoom_results]
    (
        shared_endpoint_elev,
        shared_endpoint_azim,
        _,
    ) = select_endpoint_readable_camera(
        projected_cases,
        minimum_path_retention=0.0,
    )
    shared_endpoint_crop_metrics = [
        endpoint_crop_metrics(
            case,
            shared_endpoint_elev,
            shared_endpoint_azim,
        )
        for case in projected_cases
    ]

    make_final_outcome_overlay(
        projected_cases,
        args.output_dir,
        camera_elev,
        camera_azim,
        explained,
        endpoint_zoom_cameras=endpoint_zoom_cameras,
    )
    make_outcome_rows(
        projected_cases,
        args.output_dir,
        camera_elev,
        camera_azim,
        explained,
    )
    make_stage1_context_overlay(
        projected_cases,
        args.output_dir,
        camera_elev,
        camera_azim,
        explained,
    )
    write_source_data(projected_cases, args.output_dir)

    manifest = {
        "figure_contract": (
            "Complete Final trajectories are encoded by rollout outcome; "
            "correct and wrong paths must remain distinguishable under overlap."
        ),
        "selection_rule": [
            "Within Final=7/8, choose the largest positive Stage1-to-Final gain; tie-break by index.",
            "Within Final=6/8, choose the largest positive Stage1-to-Final gain; tie-break by index.",
            "Within Final=4/8, choose the largest positive Stage1-to-Final gain; tie-break by index.",
        ],
        "selection_uses_geometry": False,
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
        "stage1_records": str(args.stage1_records.resolve()),
        "final_records": str(args.final_records.resolve()),
        "pca_fit_questions": len(stage1) - len(projected_cases),
        "pca_uses_outcome_labels": False,
        "pca_explained_variance_ratio": [
            float(value) for value in explained
        ],
        "pc1_orientation_sign": orientation_sign,
        "question_centering": "translation by the shared projected z0 only",
        "path_offsets": False,
        "per_path_scaling": False,
        "camera_selection": (
            "outcome-blind mean and lower-quartile nearest-endpoint "
            "visibility, constrained to retain at least 80% of each "
            "question's projected path length"
        ),
        "camera_elev": camera_elev,
        "camera_azim": camera_azim,
        "camera_metrics": camera_metrics,
        "path_visibility_camera_candidate": {
            "camera_elev": path_camera_elev,
            "camera_azim": path_camera_azim,
            "camera_metrics": path_camera_metrics,
        },
        "endpoint_zoom": (
            "same frozen PCA; each inset independently maximizes the "
            "outcome-blind minimum nearest-endpoint distance after cropping; "
            "no point displacement"
        ),
        "endpoint_zoom_cameras": [
            {
                "question_idx": int(case["idx"]),
                "camera_elev": elev,
                "camera_azim": azim,
            }
            for case, (elev, azim) in zip(
                projected_cases,
                endpoint_zoom_cameras,
            )
        ],
        "endpoint_zoom_camera_metrics": endpoint_zoom_metrics,
        "endpoint_zoom_shared_camera_baseline": {
            "camera_elev": shared_endpoint_elev,
            "camera_azim": shared_endpoint_azim,
            "camera_metrics": shared_endpoint_crop_metrics,
        },
        "endpoint_zoom_minimum_distance_gain_vs_shared": [
            final_metrics["minimum_nearest_endpoint_distance"]
            / max(
                shared_metrics["minimum_nearest_endpoint_distance"],
                1e-6,
            )
            for final_metrics, shared_metrics in zip(
                endpoint_zoom_metrics,
                shared_endpoint_crop_metrics,
            )
        ],
        "visual_encoding": {
            "final_correct": "pink solid line, circle steps, star endpoint",
            "final_wrong": (
                "dark-orange dashed line with white under-stroke, "
                "diamond steps, X endpoint"
            ),
            "stage1_reference": "low-opacity blue dashed line",
        },
        "candidates": [
            "fig_final_correct_wrong_overlay",
            "fig_final_correct_wrong_rows",
            "fig_final_outcomes_with_stage1_context",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
