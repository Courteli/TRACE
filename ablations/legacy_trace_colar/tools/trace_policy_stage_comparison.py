#!/usr/bin/env python3
"""Paired Stage-1 versus Stage-2 TRACE evidence.

Figure contract
---------------
Core conclusion:
    Outcome refinement should improve deterministic and rollout reliability
    without increasing answer length, while increasing the complete-path
    margin between local correct modes and wrong-answer trajectories.
Evidence logic:
    Every statistic is paired by the same 200 GSM8K test questions. Geometry
    differences use the intersection of questions that have at least two
    correct rollouts and one wrong rollout in both stages.
Review risks:
    Outcome labels mean greedy-answer correctness, not a proof that a
    trajectory is logically valid. Geometry is measured in D_TRACE, not in
    display PCA coordinates.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from trace_policy_geometry_summary import (
    choose_camera,
    common_axis_limits,
    plot_trajectory,
    project_residuals,
)


PINK = "#E5A3BF"
PINK_DARK = "#B95C88"
BLUE = "#6687B8"
GREEN = "#69AD7C"
ORANGE = "#E5A11A"
INK = "#30343B"
GRID = "#DDE2E8"


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Liberation Serif",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def save_figure(fig, output_base: Path):
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def load_records(path: Path):
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or len(records) < 200:
        raise ValueError(f"{path} must contain at least 200 records")
    records = records[:200]
    required = (
        "idx",
        "map_correct",
        "map_output_length",
        "answer_latent_attention_access",
        "rollout_correctness",
        "rollout_schema",
        "visualization_contract",
    )
    for row, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"record {row} is missing {missing}")
        if record["rollout_schema"] != "iid_conditional_gaussian":
            raise ValueError(f"record {row} is not an IID rollout")
        if len(record["rollout_correctness"]) != 8:
            raise ValueError(f"record {row} does not contain eight paths")
        if int(record["answer_latent_attention_access"]) != 8:
            raise ValueError(f"record {row} does not use eight latent states")
        contract = record["visualization_contract"]
        if contract.get("manual_offsets") is not False:
            raise ValueError(f"record {row} permits manual offsets")
        if contract.get("per_path_rescaling") is not False:
            raise ValueError(f"record {row} permits per-path rescaling")
    return records


def load_geometry_rows(path: Path) -> Dict[int, dict]:
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(float(row.pop("idx")))
            rows[index] = {
                key: float(value) for key, value in row.items()
            }
    if len(rows) != 200:
        raise ValueError(f"{path} must contain exactly 200 geometry rows")
    return rows


def paired_ci(
    stage1: Sequence[float],
    final: Sequence[float],
    *,
    rng: np.random.Generator,
    bootstrap: int,
) -> Dict[str, object]:
    left = np.asarray(stage1, dtype=np.float64)
    right = np.asarray(final, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired vectors must have identical shapes")
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size == 0:
        return {
            "stage1": float("nan"),
            "final": float("nan"),
            "delta": float("nan"),
            "delta_ci95": [float("nan"), float("nan")],
            "n_questions": 0,
        }
    indices = rng.integers(
        0,
        left.size,
        size=(int(bootstrap), left.size),
    )
    left_means = left[indices].mean(axis=1)
    right_means = right[indices].mean(axis=1)
    deltas = (right[indices] - left[indices]).mean(axis=1)
    return {
        "stage1": float(left.mean()),
        "stage1_ci95": [
            float(np.quantile(left_means, 0.025)),
            float(np.quantile(left_means, 0.975)),
        ],
        "final": float(right.mean()),
        "final_ci95": [
            float(np.quantile(right_means, 0.025)),
            float(np.quantile(right_means, 0.975)),
        ],
        "delta": float((right - left).mean()),
        "delta_ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "n_questions": int(left.size),
    }


def stage_arrays(records):
    rollout = np.asarray(
        [record["rollout_correctness"] for record in records],
        dtype=np.float64,
    )
    correct_count = rollout.sum(axis=1)
    return {
        "map_accuracy": np.asarray(
            [record["map_correct"] for record in records],
            dtype=np.float64,
        ),
        "map_output_length": np.asarray(
            [record["map_output_length"] for record in records],
            dtype=np.float64,
        ),
        "map_total_length": np.asarray(
            [
                record["map_output_length"]
                + record["answer_latent_attention_access"]
                for record in records
            ],
            dtype=np.float64,
        ),
        "rollout_accuracy": rollout.mean(axis=1),
        "any_correct": (correct_count >= 1).astype(np.float64),
        "majority_correct": (correct_count >= 5).astype(np.float64),
        "all_correct": (correct_count == 8).astype(np.float64),
        "correct_count": correct_count,
    }


def plot_reliability(
    stage1: Dict[str, np.ndarray],
    final: Dict[str, np.ndarray],
    output_base: Path,
):
    keys = (
        "map_accuracy",
        "any_correct",
        "majority_correct",
        "all_correct",
    )
    labels = ("MAP", "Any / 8", "Majority / 8", "All / 8")
    left = np.asarray([stage1[key].mean() for key in keys]) * 100.0
    right = np.asarray([final[key].mean() for key in keys]) * 100.0
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    ax.plot(
        x,
        left,
        color=GREEN,
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        label="Stage 1",
    )
    ax.plot(
        x,
        right,
        color=PINK_DARK,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="Final",
    )
    ax.fill_between(x, left, right, color=PINK, alpha=0.16)
    lower = max(0.0, min(left.min(), right.min()) - 7.0)
    upper = min(100.0, max(left.max(), right.max()) + 9.0)
    for index, delta in enumerate(right - left):
        if abs(delta) < 0.05:
            continue
        anchor = max(left[index], right[index])
        if anchor >= upper - 2.0:
            y = anchor - 1.8
            vertical_alignment = "top"
        else:
            y = anchor + 1.6
            vertical_alignment = "bottom"
        ax.text(
            index,
            y,
            f"{delta:+.1f}",
            color=PINK_DARK,
            ha="center",
            va=vertical_alignment,
            fontweight="bold",
        )
    ax.set_ylim(lower, upper)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Questions meeting criterion (%)")
    ax.set_title(
        "Outcome reliability across eight IID paths",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_base)


def plot_accuracy_length(
    stage1: Dict[str, np.ndarray],
    final: Dict[str, np.ndarray],
    output_base: Path,
):
    rng = np.random.default_rng(17)
    accuracy_metric = paired_ci(
        stage1["map_accuracy"] * 100.0,
        final["map_accuracy"] * 100.0,
        rng=rng,
        bootstrap=2000,
    )
    length_metric = paired_ci(
        stage1["map_total_length"],
        final["map_total_length"],
        rng=rng,
        bootstrap=2000,
    )
    colors = [GREEN, PINK]
    edges = [GREEN, PINK_DARK]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.15))
    for ax, metric, title, ylabel, suffix in (
        (
            axes[0],
            accuracy_metric,
            "Deterministic accuracy",
            "Accuracy (%)",
            " pp",
        ),
        (
            axes[1],
            length_metric,
            r"Total reasoning length $\#L$",
            "Latent + generated tokens",
            "",
        ),
    ):
        values = [metric["stage1"], metric["final"]]
        intervals = [
            metric["stage1_ci95"],
            metric["final_ci95"],
        ]
        errors = np.asarray(
            [
                [
                    values[index] - intervals[index][0]
                    for index in range(2)
                ],
                [
                    intervals[index][1] - values[index]
                    for index in range(2)
                ],
            ]
        )
        ax.bar(
            [0, 1],
            values,
            color=colors,
            edgecolor=edges,
            linewidth=0.8,
            width=0.56,
            zorder=2,
        )
        ax.errorbar(
            [0, 1],
            values,
            yerr=errors,
            color=INK,
            fmt="none",
            capsize=2.5,
            linewidth=0.9,
            zorder=3,
        )
        ax.text(
            0.5,
            max(values),
            f"{metric['delta']:+.2f}{suffix}",
            ha="center",
            va="bottom",
            color=PINK_DARK,
            fontweight="bold",
        )
        ax.set_xticks([0, 1], ["Stage 1", "Final"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.55)
    fig.tight_layout(pad=0.7, w_pad=1.2)
    save_figure(fig, output_base)


def plot_geometry(
    metrics: Dict[str, dict],
    output_base: Path,
):
    keys = (
        "correct_local_radius",
        "wrong_to_correct_distance",
        "outcome_margin",
        "path_diversity",
    )
    labels = (
        "Correct\nradius",
        "Wrong-to-\ncorrect",
        "Outcome\nmargin",
        "All-path\ndiversity",
    )
    stage1 = np.asarray([metrics[key]["stage1"] for key in keys])
    final = np.asarray([metrics[key]["final"] for key in keys])
    x = np.arange(len(keys))
    stage1_errors = np.asarray(
        [
            [
                metrics[key]["stage1"]
                - metrics[key]["stage1_ci95"][0]
                for key in keys
            ],
            [
                metrics[key]["stage1_ci95"][1]
                - metrics[key]["stage1"]
                for key in keys
            ],
        ]
    )
    final_errors = np.asarray(
        [
            [
                metrics[key]["final"]
                - metrics[key]["final_ci95"][0]
                for key in keys
            ],
            [
                metrics[key]["final_ci95"][1]
                - metrics[key]["final"]
                for key in keys
            ],
        ]
    )
    fig, ax = plt.subplots(figsize=(3.55, 2.6))
    ax.axhline(0.0, color=INK, linewidth=0.8, linestyle="--")
    ax.errorbar(
        x - 0.08,
        stage1,
        yerr=stage1_errors,
        color=GREEN,
        marker="o",
        linestyle="none",
        linewidth=1.2,
        markersize=4.5,
        capsize=2.2,
        label="Stage 1",
    )
    ax.errorbar(
        x + 0.08,
        final,
        yerr=final_errors,
        color=PINK_DARK,
        marker="o",
        linestyle="none",
        linewidth=1.2,
        markersize=4.5,
        capsize=2.2,
        label="Final",
    )
    for index, key in enumerate(keys):
        if abs(metrics[key]["delta"]) < 5e-4:
            continue
        ax.text(
            index,
            max(stage1[index], final[index]),
            f"{metrics[key]['delta']:+.3f}",
            ha="center",
            va="bottom",
            color=PINK_DARK,
            fontweight="bold",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Complete-path distance $D_{\mathrm{TRACE}}$")
    ax.set_title(
        "Paired outcome geometry on common eligible questions",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_base)


def figure_qa(output_dir: Path, *, expected_figures: int) -> dict:
    from PIL import Image, ImageStat

    figures = []
    for svg in sorted(output_dir.glob("*.svg")):
        pdf = svg.with_suffix(".pdf")
        tiff = svg.with_suffix(".tiff")
        if "<text" not in svg.read_text():
            raise ValueError(f"{svg} does not preserve editable text")
        if not pdf.exists() or not tiff.exists():
            raise ValueError(f"incomplete export bundle for {svg.stem}")
        image = Image.open(tiff).convert("RGB")
        extrema = ImageStat.Stat(image).extrema
        if all(low == high for low, high in extrema):
            raise ValueError(f"{tiff} is visually blank")
        figures.append(
            {
                "name": svg.stem,
                "svg_editable_text": True,
                "pdf_exists": True,
                "tiff_exists": True,
                "tiff_pixels": list(image.size),
                "nonblank": True,
            }
        )
    if len(figures) != int(expected_figures):
        raise ValueError(
            f"expected {expected_figures} paired figures, found {len(figures)}"
        )
    return {"status": "PASS", "figures": figures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-records", type=Path, required=True)
    parser.add_argument("--final-records", type=Path, required=True)
    parser.add_argument("--stage1-geometry", type=Path, required=True)
    parser.add_argument("--final-geometry", type=Path, required=True)
    parser.add_argument("--shared-pca", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    stage1_records = load_records(args.stage1_records)
    final_records = load_records(args.final_records)
    stage1_indices = [int(record["idx"]) for record in stage1_records]
    final_indices = [int(record["idx"]) for record in final_records]
    if stage1_indices != final_indices:
        raise ValueError("Stage-1 and final records are not question-paired")
    pca = torch.load(
        args.shared_pca,
        map_location="cpu",
        weights_only=False,
    )
    if not {"mean", "components", "explained_ratio"} <= set(pca):
        raise ValueError("shared PCA state is incomplete")
    stage1 = stage_arrays(stage1_records)
    final = stage_arrays(final_records)
    rng = np.random.default_rng(20260719)

    behavior = {
        key: paired_ci(
            stage1[key],
            final[key],
            rng=rng,
            bootstrap=args.bootstrap,
        )
        for key in stage1
    }
    stage1_rows = load_geometry_rows(args.stage1_geometry)
    final_rows = load_geometry_rows(args.final_geometry)
    if list(stage1_rows) != list(final_rows):
        raise ValueError("Stage-1 and final geometry rows are not paired")
    common = [
        index
        for index in stage1_rows
        if stage1_rows[index]["eligible"] > 0.5
        and final_rows[index]["eligible"] > 0.5
    ]
    if not common:
        raise ValueError("no common outcome-geometry eligible questions")
    geometry = {}
    for key in (
        "correct_local_radius",
        "wrong_to_correct_distance",
        "outcome_margin",
    ):
        geometry[key] = paired_ci(
            [stage1_rows[index][key] for index in common],
            [final_rows[index][key] for index in common],
            rng=rng,
            bootstrap=args.bootstrap,
        )
    geometry["path_diversity"] = paired_ci(
        [stage1_rows[index]["path_diversity"] for index in stage1_indices],
        [final_rows[index]["path_diversity"] for index in stage1_indices],
        rng=rng,
        bootstrap=args.bootstrap,
    )

    rescued = int(
        np.sum(
            (stage1["map_accuracy"] == 0)
            & (final["map_accuracy"] == 1)
        )
    )
    regressed = int(
        np.sum(
            (stage1["map_accuracy"] == 1)
            & (final["map_accuracy"] == 0)
        )
    )
    report = {
        "status": "PASS",
        "paired_questions": 200,
        "common_geometry_questions": len(common),
        "behavior": behavior,
        "geometry": geometry,
        "map_rescued_questions": rescued,
        "map_regressed_questions": regressed,
        "claim_boundary": (
            "Correct and wrong denote greedy-answer outcomes. This paired "
            "summary does not independently verify logical path validity."
        ),
    }

    stage1_counts = stage1["correct_count"]
    final_counts = final["correct_count"]
    selected = [
        index
        for index in range(200)
        if 0 < stage1_counts[index] < 8
        and 0 < final_counts[index] < 8
    ][:3]
    if not selected:
        selected = [
            index
            for index in range(200)
            if 0 < final_counts[index] < 8
        ][:3]
    stage1_residuals = torch.stack(
        [
            record["rollout_implicit_residuals"].float()
            for record in stage1_records
        ]
    )
    final_residuals = torch.stack(
        [
            record["rollout_implicit_residuals"].float()
            for record in final_records
        ]
    )
    stage1_projected = project_residuals(stage1_residuals, pca)[..., :3]
    final_projected = project_residuals(final_residuals, pca)[..., :3]
    stage1_maps = project_residuals(
        torch.stack(
            [
                record["map_implicit_residuals"].float()
                for record in stage1_records
            ]
        ),
        pca,
    )[..., :3]
    final_maps = project_residuals(
        torch.stack(
            [
                record["map_implicit_residuals"].float()
                for record in final_records
            ]
        ),
        pca,
    )[..., :3]
    trajectory_audit = []
    for index in selected:
        stage1_paths = np.concatenate(
            [
                np.zeros((8, 1, 3)),
                stage1_projected[index].cumsum(dim=1).numpy(),
            ],
            axis=1,
        )
        final_paths = np.concatenate(
            [
                np.zeros((8, 1, 3)),
                final_projected[index].cumsum(dim=1).numpy(),
            ],
            axis=1,
        )
        stage1_map = np.concatenate(
            [
                np.zeros((1, 3)),
                stage1_maps[index].cumsum(dim=0).numpy(),
            ],
            axis=0,
        )
        final_map = np.concatenate(
            [
                np.zeros((1, 3)),
                final_maps[index].cumsum(dim=0).numpy(),
            ],
            axis=0,
        )
        camera = choose_camera(
            np.concatenate([stage1_paths, final_paths], axis=0)
        )
        limits = common_axis_limits(
            [
                stage1_paths,
                final_paths,
                stage1_map[None, ...],
                final_map[None, ...],
            ]
        )
        question_id = int(stage1_records[index]["idx"])
        plot_trajectory(
            stage1_records[index],
            stage1_projected[index].numpy(),
            stage1_maps[index].numpy(),
            camera=camera,
            limits=limits,
            output_base=(
                args.output_dir / f"trajectory_q{question_id}_stage1"
            ),
            stage_label="Stage 1",
        )
        plot_trajectory(
            final_records[index],
            final_projected[index].numpy(),
            final_maps[index].numpy(),
            camera=camera,
            limits=limits,
            output_base=(
                args.output_dir / f"trajectory_q{question_id}_final"
            ),
            stage_label="Final",
        )
        trajectory_audit.append(
            {
                "question_index": question_id,
                "stage1_correct_paths": int(stage1_counts[index]),
                "final_correct_paths": int(final_counts[index]),
                "camera_score": camera[0],
                "elevation": camera[1],
                "azimuth": camera[2],
                "axis_limits": [
                    [float(lower), float(upper)]
                    for lower, upper in limits
                ],
            }
        )
    report["paired_qualitative_trajectories"] = {
        "selection": (
            "first three questions with mixed outcomes in both stages; "
            "outcome counts only"
        ),
        "shared_projection_camera_and_limits": True,
        "manual_offsets": False,
        "per_path_rescaling": False,
        "questions": trajectory_audit,
    }
    (args.output_dir / "stage_comparison.json").write_text(
        json.dumps(report, indent=2)
    )
    lines = [
        "# Paired TRACE Stage Comparison",
        "",
        "| Metric | Stage 1 | Final | Delta | 95% CI | N |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for section, names in (
        (
            behavior,
            (
                "map_accuracy",
                "map_total_length",
                "map_output_length",
                "rollout_accuracy",
                "any_correct",
                "majority_correct",
                "all_correct",
            ),
        ),
        (
            geometry,
            (
                "correct_local_radius",
                "wrong_to_correct_distance",
                "outcome_margin",
                "path_diversity",
            ),
        ),
    ):
        for name in names:
            value = section[name]
            lines.append(
                f"| {name} | {value['stage1']:.4f} | "
                f"{value['final']:.4f} | {value['delta']:+.4f} | "
                f"[{value['delta_ci95'][0]:+.4f}, "
                f"{value['delta_ci95'][1]:+.4f}] | "
                f"{value['n_questions']} |"
            )
    lines.extend(
        [
            "",
            f"- MAP rescue/regression: **{rescued}/{regressed}** questions.",
            f"- Common geometry-eligible set: **{len(common)}/200** questions.",
            "",
            report["claim_boundary"],
        ]
    )
    (args.output_dir / "stage_comparison.md").write_text(
        "\n".join(lines)
    )
    plot_reliability(
        stage1,
        final,
        args.output_dir / "stage_reliability",
    )
    plot_accuracy_length(
        stage1,
        final,
        args.output_dir / "stage_accuracy_length",
    )
    plot_geometry(
        geometry,
        args.output_dir / "stage_outcome_geometry",
    )
    np.savez_compressed(
        args.output_dir / "stage_comparison_source_data.npz",
        question_indices=np.asarray(stage1_indices),
        stage1_map_accuracy=stage1["map_accuracy"],
        final_map_accuracy=final["map_accuracy"],
        stage1_output_length=stage1["map_output_length"],
        final_output_length=final["map_output_length"],
        stage1_total_L=stage1["map_total_length"],
        final_total_L=final["map_total_length"],
        stage1_rollout_correctness=np.asarray(
            [
                record["rollout_correctness"]
                for record in stage1_records
            ]
        ),
        final_rollout_correctness=np.asarray(
            [
                record["rollout_correctness"]
                for record in final_records
            ]
        ),
        common_geometry_indices=np.asarray(common),
    )
    contract = {
        "core_conclusion": (
            "Stage-2 outcome refinement should increase accuracy and "
            "reliability at stable total #L while improving "
            "question-paired complete-path outcome geometry."
        ),
        "paired_questions": 200,
        "geometry_subset": "eligible in both Stage 1 and Final",
        "uncertainty": "question-bootstrap 95% confidence intervals",
        "outcome_definition": "greedy decoded answer correctness",
        "path_distance": "D_TRACE; no PCA display distance",
        "shared_pca": str(args.shared_pca),
        "paired_trajectory_selection": (
            "outcome counts only; geometry unused for question selection"
        ),
        "manual_offsets": False,
        "per_path_rescaling": False,
        "exports": ["SVG", "PDF", "TIFF 600 dpi"],
    }
    (args.output_dir / "figure_contract.json").write_text(
        json.dumps(contract, indent=2)
    )
    qa = figure_qa(
        args.output_dir,
        expected_figures=3 + 2 * len(selected),
    )
    (args.output_dir / "figure_qa.json").write_text(
        json.dumps(qa, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
