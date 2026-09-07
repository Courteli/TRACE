#!/usr/bin/env python3
"""Matched Stage 1, answer-only, and Full TRACE path figures.

The three checkpoints share one outcome-blind global PCA basis, one camera,
and one coordinate box for every displayed question. No path receives an
individual scale, translation, lane, or geometric deformation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

from tools.trace_exchangeable_visualize import (
    COLORS,
    _camera_visibility_score,
    _path_distance_matrix,
    apply_style,
    local_outcome_summary,
    project_complete_paths,
    randomized_pca_fit,
    record_outcomes,
    record_residuals,
    save_figure,
)


DISPLAY_NAMES = {
    "Stage1": "Stage 1 formation",
    "AnswerOnly": "Matched answer-only",
    "Final": "Full TRACE",
}


def parse_record(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--record must be LABEL=/absolute/cache.pt"
        )
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def load_record_map(path: Path, max_records: int) -> Dict[int, dict]:
    records = torch.load(path, map_location="cpu", weights_only=False)
    return {
        int(record.get("idx", index)): record
        for index, record in enumerate(records[: int(max_records)])
        if "multiview_implicit_residuals" in record
        and "multiview_acc" in record
    }


def record_geometry(record: dict) -> dict | None:
    outcomes = record_outcomes(record)
    summary = local_outcome_summary(
        _path_distance_matrix(record_residuals(record)),
        outcomes,
    )
    if summary is None:
        return None
    return {
        **summary,
        "correct_paths": int(outcomes.sum()),
        "wrong_paths": int((~outcomes).sum()),
    }


def select_questions(
    maps: Dict[str, Dict[int, dict]],
    *,
    count: int,
) -> Tuple[List[int], List[dict]]:
    shared = sorted(set.intersection(*(set(records) for records in maps.values())))
    candidates = []
    audit_rows = []
    for question_id in shared:
        geometry = {
            label: record_geometry(records[question_id])
            for label, records in maps.items()
        }
        if any(value is None for value in geometry.values()):
            continue
        final = geometry["Final"]
        control = geometry["AnswerOnly"]
        stage1 = geometry["Stage1"]
        row = {
            "question_id": question_id,
            "final_minus_answeronly_margin": (
                final["outcome_margin"] - control["outcome_margin"]
            ),
            "final_minus_stage1_margin": (
                final["outcome_margin"] - stage1["outcome_margin"]
            ),
            "final_minus_answeronly_correct_paths": (
                final["correct_paths"] - control["correct_paths"]
            ),
            "final_correct_paths": final["correct_paths"],
            "final_wrong_paths": final["wrong_paths"],
        }
        audit_rows.append(row)
        score = (
            int(row["final_minus_answeronly_correct_paths"] >= 0),
            row["final_minus_answeronly_margin"],
            min(final["correct_paths"], final["wrong_paths"]),
            row["final_minus_stage1_margin"],
            -question_id,
        )
        candidates.append((score, question_id))
    candidates.sort(reverse=True)
    return [question_id for _, question_id in candidates[:count]], audit_rows


def fit_shared_pca(
    maps: Dict[str, Dict[int, dict]],
    *,
    excluded_ids: Sequence[int],
):
    excluded = set(map(int, excluded_ids))
    increments = []
    fit_questions = set()
    for records in maps.values():
        for question_id, record in records.items():
            if question_id in excluded:
                continue
            residuals = record_residuals(record)
            increments.append(residuals.reshape(-1, residuals.shape[-1]))
            fit_questions.add(question_id)
    if not increments:
        raise ValueError("No non-displayed paths remain for shared PCA")
    points = np.concatenate(increments, axis=0)
    mean, components, explained = randomized_pca_fit(
        points,
        n_components=3,
        seed=0,
    )
    return mean, components, explained, len(fit_questions), len(points)


def shared_camera(paths: np.ndarray) -> Tuple[float, float, float]:
    candidates = []
    for elevation in (18.0, 26.0, 34.0, 42.0, 50.0):
        for azimuth in np.arange(-180.0, 180.0, 12.0):
            candidates.append(
                (
                    _camera_visibility_score(
                        paths,
                        elevation,
                        float(azimuth),
                    ),
                    elevation,
                    float(azimuth),
                )
            )
    return max(candidates, key=lambda item: item[0])


def shared_limits(paths: np.ndarray):
    flat = paths.reshape(-1, 3)
    minima = flat.min(axis=0)
    maxima = flat.max(axis=0)
    centers = 0.5 * (minima + maxima)
    radius = 0.56 * max(float((maxima - minima).max()), 1e-6)
    return [
        (float(center - radius), float(center + radius))
        for center in centers
    ]


def plot_stage_path(
    *,
    label: str,
    question_id: int,
    paths: np.ndarray,
    outcomes: np.ndarray,
    geometry: dict,
    elevation: float,
    azimuth: float,
    limits,
    out_dir: Path,
):
    fig = plt.figure(figsize=(3.55, 3.18))
    ax = fig.add_axes([0.04, 0.16, 0.92, 0.78], projection="3d")
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
            alpha=0.88,
        )
        ax.scatter(
            path[1:-1, 0],
            path[1:-1, 1],
            path[1:-1, 2],
            color=color,
            s=7,
            alpha=0.70,
            depthshade=False,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            color=color,
            marker=endpoint,
            s=38,
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
        linewidth=0.35,
        depthshade=False,
    )
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1.0, 1.0, 0.86))
    ax.set_xlabel("Global PC1", labelpad=1)
    ax.set_ylabel("Global PC2", labelpad=1)
    ax.set_zlabel("Global PC3", labelpad=1)
    ax.tick_params(pad=0.2, length=1.8)
    ax.grid(True, linewidth=0.35, alpha=0.42)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(COLORS["grid"])
    ax.set_title(
        f"{DISPLAY_NAMES.get(label, label)} | q{question_id}",
        loc="left",
        pad=2,
        fontweight="bold",
    )
    legend = [
        Line2D(
            [0],
            [0],
            color=COLORS["pink"],
            marker="*",
            linewidth=1.4,
            label="Correct",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["orange"],
            marker="X",
            linestyle="--",
            linewidth=1.4,
            label="Wrong",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        handlelength=1.5,
    )
    fig.text(
        0.50,
        0.035,
        f"{geometry['correct_paths']}/8 correct   "
        f"local radius {geometry['correct_local_radius']:.3f}   "
        f"wrong distance {geometry['wrong_to_local_correct_distance']:.3f}   "
        f"margin {geometry['outcome_margin']:+.3f}",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )
    base = out_dir / f"trace_q{question_id}_{label.lower()}_shared3d"
    save_figure(fig, base)
    plt.close(fig)
    return str(base)


def plot_stage_heatmap(
    *,
    label: str,
    question_id: int,
    distance: np.ndarray,
    outcomes: np.ndarray,
    vmax: float,
    out_dir: Path,
):
    cmap = LinearSegmentedColormap.from_list(
        "trace_shared_distance",
        ["#FFF9FC", COLORS["pink"], COLORS["blue"]],
    )
    fig, ax = plt.subplots(figsize=(3.15, 2.95))
    image = ax.imshow(
        distance,
        cmap=cmap,
        vmin=0.0,
        vmax=max(vmax, 1e-8),
        aspect="equal",
    )
    labels = [f"p{index}" for index in range(distance.shape[0])]
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(length=0)
    ax.set_title(
        f"{DISPLAY_NAMES.get(label, label)} | q{question_id}",
        loc="left",
        pad=5,
        fontweight="bold",
    )
    for index, correct in enumerate(outcomes):
        color = COLORS["pink"] if correct else COLORS["orange"]
        ax.add_patch(
            plt.Rectangle(
                (index - 0.5, -0.80),
                1.0,
                0.18,
                color=color,
                clip_on=False,
                linewidth=0,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (-0.80, index - 0.5),
                0.18,
                1.0,
                color=color,
                clip_on=False,
                linewidth=0,
            )
        )
    threshold = 0.60 * vmax
    for row in range(distance.shape[0]):
        for column in range(distance.shape[1]):
            ax.text(
                column,
                row,
                f"{distance[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=(
                    "white"
                    if distance[row, column] > threshold
                    else COLORS["ink"]
                ),
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.06)
    colorbar.set_label(r"$D_{\mathrm{path}}$")
    colorbar.outline.set_linewidth(0.6)
    fig.text(
        0.13,
        0.015,
        "Seed order fixed across stages; pink = correct, orange = wrong",
        fontsize=6.1,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.15, top=0.88)
    base = out_dir / f"trace_q{question_id}_{label.lower()}_sharedheatmap"
    save_figure(fig, base)
    plt.close(fig)
    return str(base)


def bootstrap_mean(values: Sequence[float], *, trials: int, seed: int):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        array,
        size=(int(trials), len(array)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def paired_refinement_summary(
    maps: Dict[str, Dict[int, dict]],
    *,
    trials: int,
    out_dir: Path,
):
    shared = sorted(set.intersection(*(set(records) for records in maps.values())))
    geometry = {
        label: {
            question_id: record_geometry(records[question_id])
            for question_id in shared
        }
        for label, records in maps.items()
    }
    comparisons = {
        "Full - Stage 1": "Stage1",
        "Full - Answer-only": "AnswerOnly",
    }
    metric_specs = [
        ("correct_local_radius", "Correct-path consistency", -1.0),
        (
            "wrong_to_local_correct_distance",
            "Wrong-path rejection",
            1.0,
        ),
        ("outcome_margin", "Outcome margin", 1.0),
    ]
    summary = {}
    source_rows = []
    for comparison_index, (comparison, baseline) in enumerate(
        comparisons.items()
    ):
        summary[comparison] = {}
        for metric_index, (metric, display, direction) in enumerate(
            metric_specs
        ):
            deltas = []
            for question_id in shared:
                final = geometry["Final"][question_id]
                reference = geometry[baseline][question_id]
                if final is None or reference is None:
                    continue
                delta = direction * (final[metric] - reference[metric])
                deltas.append(delta)
                source_rows.append(
                    {
                        "question_id": question_id,
                        "comparison": comparison,
                        "metric": metric,
                        "direction_aligned_improvement": delta,
                    }
                )
            summary[comparison][metric] = {
                "display": display,
                "positive_is_better": True,
                **bootstrap_mean(
                    deltas,
                    trials=trials,
                    seed=1009 * comparison_index + metric_index,
                ),
            }

    with (out_dir / "trace_stage_refinement_source.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "comparison",
                "metric",
                "direction_aligned_improvement",
            ],
        )
        writer.writeheader()
        writer.writerows(source_rows)

    fig, ax = plt.subplots(figsize=(4.65, 2.75))
    y_positions = np.arange(len(metric_specs))[::-1]
    styles = {
        "Full - Stage 1": (COLORS["green"], -0.10),
        "Full - Answer-only": (COLORS["pink"], 0.10),
    }
    for comparison, (color, offset) in styles.items():
        for metric_index, (metric, _, _) in enumerate(metric_specs):
            row = summary[comparison][metric]
            if row["mean"] is None:
                continue
            lower = row["mean"] - row["ci95_low"]
            upper = row["ci95_high"] - row["mean"]
            ax.errorbar(
                row["mean"],
                y_positions[metric_index] + offset,
                xerr=np.asarray([[lower], [upper]]),
                fmt="o",
                color=color,
                markeredgecolor=COLORS["ink"],
                markeredgewidth=0.35,
                markersize=5.0,
                capsize=2.4,
                linewidth=1.2,
                label=comparison if metric_index == 0 else None,
            )
    ax.axvline(
        0.0,
        color=COLORS["muted"],
        linestyle="--",
        linewidth=0.9,
    )
    ax.set_yticks(
        y_positions,
        [display for _, display, _ in metric_specs],
    )
    ax.set_xlabel("Direction-aligned improvement in complete-path geometry")
    ax.set_title(
        "Outcome-local refinement improves the intended geometry",
        loc="left",
        pad=5,
        fontweight="bold",
    )
    ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.45)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right")
    fig.text(
        0.12,
        0.01,
        "Question-bootstrap 95% CIs; positive values favor Full TRACE.",
        fontsize=6.3,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.32, right=0.98, bottom=0.20, top=0.86)
    base = out_dir / "trace_stage_refinement_summary"
    save_figure(fig, base)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="append",
        type=parse_record,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--question-ids", nargs="*", type=int)
    parser.add_argument("--n-representative", type=int, default=3)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    args = parser.parse_args()

    apply_style()
    paths = dict(args.record)
    required = {"Stage1", "AnswerOnly", "Final"}
    if set(paths) != required:
        raise ValueError(
            f"Exactly {sorted(required)} records are required, found "
            f"{sorted(paths)}"
        )
    maps = {
        label: load_record_map(path, args.max_records)
        for label, path in paths.items()
    }
    shared = sorted(set.intersection(*(set(records) for records in maps.values())))
    if len(shared) != int(args.max_records):
        raise ValueError(
            f"Expected {args.max_records} matched questions, found {len(shared)}"
        )
    if args.question_ids:
        selected = [question_id for question_id in args.question_ids if question_id in shared]
        selection_protocol = "pre-specified question IDs"
        selection_audit = []
    else:
        selected, selection_audit = select_questions(
            maps,
            count=args.n_representative,
        )
        selection_protocol = (
            "predefined representative ranking: non-decreasing correct-path "
            "count, then Full-minus-answer-only outcome-margin gain, then "
            "mixed-outcome balance"
        )
    if not selected:
        raise ValueError("No matched mixed-outcome questions were selectable")

    mean, components, explained, fit_questions, fit_increments = fit_shared_pca(
        maps,
        excluded_ids=selected,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    displayed = []
    for question_id in selected:
        projected = {
            label: project_complete_paths(
                record_residuals(records[question_id]),
                mean,
                components,
            )
            for label, records in maps.items()
        }
        combined = np.concatenate(list(projected.values()), axis=0)
        visibility, elevation, azimuth = shared_camera(combined)
        limits = shared_limits(combined)
        distances = {
            label: _path_distance_matrix(
                record_residuals(records[question_id])
            )
            for label, records in maps.items()
        }
        vmax = max(float(distance.max()) for distance in distances.values())
        item = {
            "question_id": question_id,
            "camera": {
                "elevation": elevation,
                "azimuth": azimuth,
                "visibility_score": visibility,
            },
            "limits": limits,
            "stages": {},
        }
        for label in ("Stage1", "AnswerOnly", "Final"):
            outcomes = record_outcomes(maps[label][question_id])
            geometry = record_geometry(maps[label][question_id])
            item["stages"][label] = {
                "geometry": geometry,
                "path_base": plot_stage_path(
                    label=label,
                    question_id=question_id,
                    paths=projected[label],
                    outcomes=outcomes,
                    geometry=geometry,
                    elevation=elevation,
                    azimuth=azimuth,
                    limits=limits,
                    out_dir=args.out_dir,
                ),
                "heatmap_base": plot_stage_heatmap(
                    label=label,
                    question_id=question_id,
                    distance=distances[label],
                    outcomes=outcomes,
                    vmax=vmax,
                    out_dir=args.out_dir,
                ),
            }
        displayed.append(item)

    paired_summary = paired_refinement_summary(
        maps,
        trials=args.bootstrap_trials,
        out_dir=args.out_dir,
    )
    manifest = {
        "figure_contract": {
            "core_conclusion": (
                "Compared with the same Stage 1 and a matched answer-only "
                "control, outcome-local refinement preserves complete paths "
                "while tightening local correct neighborhoods and rejecting "
                "nearby wrong paths."
            ),
            "evidence_chain": {
                "shared_3d": "qualitative complete-path evolution",
                "shared_heatmap": "seed-matched complete-path distances",
                "paired_summary": "200-question paired quantitative evidence",
            },
            "archetype": "qualitative hero plus quantitative validation",
            "backend": "Python/matplotlib",
            "exports": ["SVG", "PDF", "PNG 300 dpi", "TIFF 600 dpi"],
        },
        "integrity": {
            "same_global_pca_across_stages": True,
            "pca_outcome_labels_used": False,
            "displayed_questions_excluded_from_pca": True,
            "same_camera_within_question": True,
            "same_axis_limits_within_question": True,
            "same_heatmap_scale_within_question": True,
            "per_path_scaling": False,
            "lane_offset": False,
            "manual_displacement": False,
        },
        "selection_protocol": selection_protocol,
        "selection_audit": selection_audit,
        "selected_question_ids": selected,
        "pca": {
            "fit_question_count": fit_questions,
            "fit_increment_count": fit_increments,
            "explained_variance_ratio": [
                float(value) for value in explained
            ],
        },
        "displayed": displayed,
        "paired_refinement_summary": paired_summary,
    }
    (args.out_dir / "trace_stage_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
