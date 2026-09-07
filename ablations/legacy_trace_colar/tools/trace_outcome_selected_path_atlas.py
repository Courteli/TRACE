#!/usr/bin/env python3
"""Draw outcome-selected TRACE path cases with an outcome-free PCA basis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    "stage1": "#9CA6B2",
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
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.3,
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
    parser.add_argument("--max-records", type=int, default=200)
    return parser.parse_args()


def as_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def normalized_paths(record: dict) -> np.ndarray:
    residuals = as_array(record["multiview_implicit_residuals"])
    paths = np.cumsum(residuals, axis=1)
    paths = np.concatenate([np.zeros_like(paths[:, :1]), paths], axis=1)
    final_norm = np.linalg.norm(paths[:, -1], axis=-1, keepdims=True)
    return (paths / np.clip(final_norm[:, None, :], 1e-8, None)).astype(np.float32)


def template(records: dict[int, dict], indices: list[int]) -> np.ndarray:
    total = None
    for idx in indices:
        value = normalized_paths(records[idx])
        total = value if total is None else total + value
    return total / len(indices)


def residualized_paths(
    record: dict, population_template: np.ndarray, n_questions: int
) -> np.ndarray:
    paths = normalized_paths(record)
    paths = (n_questions / max(n_questions - 1, 1)) * (paths - population_template)
    paths = paths - paths.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(paths * paths, axis=-1)))
    return (paths / max(float(scale), 1e-8)).astype(np.float32)


def pca_fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    centered = points - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    explained = variance[:2] / max(float(variance.sum()), 1e-12)
    return mean, vh[:2], explained


def pca_project(points: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (points - mean) @ components.T


def visible_turn_path(projected: np.ndarray) -> np.ndarray:
    path = projected - projected[0]
    steps = np.diff(path, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    nonzero = lengths[lengths > 1e-8]
    reference = float(np.median(nonzero)) if len(nonzero) else 1.0
    visible_lengths = np.clip(np.sqrt(lengths / max(reference, 1e-8)), 0.55, 1.55)
    directions = steps / np.clip(lengths[:, None], 1e-8, None)
    visible = directions * visible_lengths[:, None]
    result = np.concatenate([np.zeros((1, 2)), np.cumsum(visible, axis=0)], axis=0)
    arc = np.linalg.norm(np.diff(result, axis=0), axis=1).sum()
    result *= 5.8 / max(float(arc), 1e-8)
    radius = np.linalg.norm(result, axis=1).max()
    result *= 2.15 / max(float(radius), 1e-8)
    return result


def correct_count(record: dict) -> int:
    return int((as_array(record["multiview_acc"]).reshape(-1) > 0.5).sum())


def select_cases(
    stage1: dict[int, dict], final: dict[int, dict], indices: list[int]
) -> list[dict]:
    counts = [(idx, correct_count(stage1[idx]), correct_count(final[idx])) for idx in indices]
    max_gain = max(final_count - stage1_count for _, stage1_count, final_count in counts)
    first_max = min(item for item in counts if item[2] - item[1] == max_gain)
    stable_all = min(item for item in counts if item[1] == 8 and item[2] == 8)
    majority_rescues = [item for item in counts if item[1] < 5 and item[2] >= 5 and item != first_max]
    largest_remaining_gain = max(item[2] - item[1] for item in majority_rescues)
    first_majority = min(
        item for item in majority_rescues if item[2] - item[1] == largest_remaining_gain
    )
    return [
        {"selection": "Maximum reliability gain", "idx": first_max[0], "stage1_correct": first_max[1], "final_correct": first_max[2]},
        {"selection": "Stable all-correct", "idx": stable_all[0], "stage1_correct": stable_all[1], "final_correct": stable_all[2]},
        {"selection": "Largest remaining majority rescue", "idx": first_majority[0], "stage1_correct": first_majority[1], "final_correct": first_majority[2]},
    ]


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)


def main() -> None:
    args = parse_args()
    stage1_list = torch.load(args.stage1_records, map_location="cpu", weights_only=False)[: args.max_records]
    final_list = torch.load(args.final_records, map_location="cpu", weights_only=False)[: args.max_records]
    stage1 = {int(record["idx"]): record for record in stage1_list}
    final = {int(record["idx"]): record for record in final_list}
    indices = sorted(set(stage1) & set(final))
    if len(indices) != args.max_records:
        raise ValueError(f"Expected {args.max_records} aligned records, found {len(indices)}")
    cases = select_cases(stage1, final, indices)
    stage1_template = template(stage1, indices)
    final_template = template(final, indices)

    fig = plt.figure(figsize=(7.2, 2.78))
    view_colors = [COLORS["blue"], COLORS["blue"], COLORS["green"], COLORS["green"], COLORS["gold"], COLORS["gold"], COLORS["pink"], COLORS["pink"]]
    metadata = []
    for panel, case in enumerate(cases):
        idx = case["idx"]
        stage1_paths = residualized_paths(stage1[idx], stage1_template, len(indices))
        final_paths = residualized_paths(final[idx], final_template, len(indices))
        fit_points = np.concatenate(
            [stage1_paths[:, 1:].reshape(-1, stage1_paths.shape[-1]), final_paths[:, 1:].reshape(-1, final_paths.shape[-1])],
            axis=0,
        )
        mean, components, explained = pca_fit(fit_points)
        projections = {
            "stage1": pca_project(stage1_paths.reshape(-1, stage1_paths.shape[-1]), mean, components).reshape(8, 9, 2),
            "final": pca_project(final_paths.reshape(-1, final_paths.shape[-1]), mean, components).reshape(8, 9, 2),
        }
        ax = fig.add_subplot(1, 3, panel + 1, projection="3d")
        for view_idx in range(8):
            lane_col = view_idx % 4
            lane_row = view_idx // 4
            offset = np.asarray([(lane_col - 1.5) * 5.1, (lane_row - 0.5) * 6.0])
            latent_step = np.arange(9, dtype=np.float32)
            stage1_turn = visible_turn_path(projections["stage1"][view_idx])
            final_turn = visible_turn_path(projections["final"][view_idx])
            stage1_display = np.column_stack([latent_step, offset[0] + stage1_turn[:, 0], offset[1] + stage1_turn[:, 1]])
            final_display = np.column_stack([latent_step, offset[0] + final_turn[:, 0], offset[1] + final_turn[:, 1]])
            ax.plot(*stage1_display.T, color=COLORS["stage1"], linewidth=0.85, linestyle="--", alpha=0.75)
            color = view_colors[view_idx]
            ax.plot(*final_display.T, color=color, linewidth=1.85, alpha=0.96)
            ax.scatter(*final_display.T, color=color, s=np.linspace(3.5, 13, 9), alpha=0.88)
            movement = np.diff(final_display, axis=0)
            ax.quiver(
                final_display[:-1, 0], final_display[:-1, 1], final_display[:-1, 2],
                movement[:, 0], movement[:, 1], movement[:, 2],
                color=color, alpha=0.75, linewidth=0.65, arrow_length_ratio=0.22,
            )
            ax.scatter(*final_display[0], color=COLORS["text"], s=8, marker="s", alpha=0.7)
            ax.scatter(*final_display[-1], color=color, s=24, edgecolor="white", linewidth=0.4)

        ax.set_title(
            f'{case["selection"]} | q{idx}\n{case["stage1_correct"]}/8 -> {case["final_correct"]}/8 correct paths',
            fontsize=6.8,
            fontweight="bold",
            pad=2,
        )
        ax.set_xlim(-0.35, 8.45)
        ax.set_ylim(-10.4, 10.4)
        ax.set_zlim(-5.6, 5.6)
        ax.set_box_aspect((1.38, 1.05, 0.76), zoom=1.15)
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
        ax.text2D(-0.02, 1.02, chr(ord("a") + panel), transform=ax.transAxes, fontsize=8.0, fontweight="bold")
        metadata.append({**case, "local_pca_explained_variance_ratio": [float(value) for value in explained]})

    handles = [Line2D([0], [0], color=COLORS["stage1"], linestyle="--", linewidth=1.2, label="TRACE Stage 1")]
    for color, label in zip((COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["pink"]), ("Final views 1-2", "Final views 3-4", "Final views 5-6", "Final views 7-8")):
        handles.append(Line2D([0], [0], color=color, linewidth=1.8, label=label))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.078), handlelength=1.6)
    fig.text(
        0.5,
        0.018,
        "Outcome-only case selection. Fixed lanes expose all eight z0-to-z8 paths; lane distance and display-normalized step magnitude are not geometry evidence.",
        ha="center",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.90, bottom=0.19, wspace=0.02)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output_dir / "fig_outcome_selected_path_atlas")
    plt.close(fig)

    source_path = args.output_dir / "source_data" / "outcome_selected_path_cases.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    manifest = {
        "figure": "TRACE outcome-selected path atlas",
        "n_population_questions": len(indices),
        "selection_uses_geometry": False,
        "selection_rule": [item["selection"] for item in cases],
        "pca": "per-question basis fitted jointly to Stage 1 and Final paths after outcome-free population-template residualization",
        "display_transform": "fixed view lanes with direction-preserving step-ratio compression and fixed radial extent",
        "claim_boundary": "qualitative path illustration only; aggregate permutation tests carry the trajectory claim",
        "cases": metadata,
    }
    (args.output_dir / "outcome_selected_path_atlas.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
