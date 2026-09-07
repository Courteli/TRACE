#!/usr/bin/env python
"""Publication figures for exchangeable TRACE.

Figure contract
---------------
Core conclusion: complete latent paths are visible without geometric editing,
while outcome separation and teacher-relative structure are quantified by the
same D_path used for training.
Archetype: separate qualitative hero path and quantitative validation figures.
Integrity: PCA is fit without outcome labels on non-displayed questions; paths
receive no per-path scaling, translation, lane offset, or manual deformation.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

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

from src.models.trace_exchangeable import trace_path_distance


COLORS = {
    "blue": "#6789B7",
    "green": "#69B482",
    "orange": "#E6A019",
    "pink": "#E7A1BF",
    "ink": "#303640",
    "muted": "#687383",
    "grid": "#DCE2EA",
    "paper": "#FFFFFF",
}


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
        }
    )


def save_figure(fig, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        base_path.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def as_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_records(paths: Sequence[Path], max_records: int) -> List[dict]:
    records = []
    for path in paths:
        records.extend(torch.load(path, map_location="cpu", weights_only=False))
    return records[: int(max_records)]


def record_residuals(record: dict) -> np.ndarray:
    if "multiview_implicit_residuals" in record:
        return as_array(record["multiview_implicit_residuals"])
    return as_array(record["implicit_residuals"])[None, ...]


def record_teacher_residuals(record: dict) -> np.ndarray:
    paths = record_residuals(record)
    if "rationale_teacher_residuals" in record:
        teachers = as_array(record["rationale_teacher_residuals"])
        path_tensor = torch.from_numpy(paths).float()
        teacher_tensor = torch.from_numpy(teachers).float()
        distance = trace_path_distance(
            path_tensor[:, None, :, :],
            teacher_tensor[None, :, :, :],
            anchor_count=3,
            position_weight=0.45,
            direction_weight=0.35,
            step_weight=0.15,
        )
        nearest = distance.argmin(dim=1).numpy()
        return teachers[nearest]
    if "multiview_teacher_residuals" in record:
        return as_array(record["multiview_teacher_residuals"])
    teacher = as_array(record["aggregated_explicit_residuals"])
    return np.repeat(teacher[None, ...], paths.shape[0], axis=0)


def record_outcomes(record: dict) -> np.ndarray:
    if "multiview_acc" not in record:
        return np.asarray([bool(record.get("acc", 0.0) > 0.5)])
    return as_array(record["multiview_acc"]).reshape(-1) > 0.5


def randomized_pca_fit(
    points: np.ndarray,
    *,
    n_components: int = 3,
    oversamples: int = 16,
    n_iter: int = 2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = points.mean(axis=0, keepdims=True).astype(np.float32)
    centered = (points - mean).astype(np.float32, copy=False)
    n_samples, n_features = centered.shape
    rank = min(n_components + oversamples, n_samples, n_features)
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(n_features, rank)).astype(np.float32)
    sketch = centered @ omega
    for _ in range(max(0, n_iter)):
        sketch = centered @ (centered.T @ sketch)
    q, _ = np.linalg.qr(sketch, mode="reduced")
    compressed = q.T @ centered
    _, singular_values, vh = np.linalg.svd(compressed, full_matrices=False)
    components = vh[:n_components].T.astype(np.float32)
    total_variance = float((centered * centered).sum())
    explained = (
        singular_values[:n_components] ** 2 / total_variance
        if total_variance > 0
        else np.zeros(n_components, dtype=np.float32)
    )
    return mean, components, explained.astype(np.float32)


def fit_global_increment_pca(
    records: Sequence[dict],
    *,
    excluded_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    excluded = set(map(int, excluded_ids))
    fit_chunks = [
        record_residuals(record).reshape(-1, record_residuals(record).shape[-1])
        for record in records
        if int(record.get("idx", -1)) not in excluded
    ]
    if not fit_chunks:
        raise ValueError("No non-displayed records remain for global PCA fitting")
    points = np.concatenate(fit_chunks, axis=0)
    mean, components, explained = randomized_pca_fit(points)
    return mean, components, explained, int(points.shape[0])


def project_complete_paths(
    residuals: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    del mean
    # Residuals are displacement vectors. PCA centering determines the basis,
    # but subtracting the fitted mean from every transition would accumulate an
    # artificial t * mean drift and would not map the zero vector to the origin.
    projected_increments = residuals @ components
    cumulative = np.cumsum(projected_increments, axis=1)
    origin = np.zeros((residuals.shape[0], 1, 3), dtype=np.float32)
    return np.concatenate([origin, cumulative], axis=1)


def _camera_basis(elevation: float, azimuth: float):
    elev = math.radians(elevation)
    azim = math.radians(azimuth)
    view = np.asarray(
        [
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ],
        dtype=np.float64,
    )
    horizontal = np.asarray([-math.sin(azim), math.cos(azim), 0.0])
    vertical = np.cross(view, horizontal)
    horizontal /= max(np.linalg.norm(horizontal), 1e-12)
    vertical /= max(np.linalg.norm(vertical), 1e-12)
    return horizontal, vertical


def _camera_visibility_score(paths: np.ndarray, elevation: float, azimuth: float) -> float:
    horizontal, vertical = _camera_basis(elevation, azimuth)
    projection = np.stack(
        [
            np.tensordot(paths, horizontal, axes=([-1], [0])),
            np.tensordot(paths, vertical, axes=([-1], [0])),
        ],
        axis=-1,
    )
    spread = np.linalg.norm(
        np.ptp(projection.reshape(-1, 2), axis=0)
    )
    spread = max(float(spread), 1e-8)
    pairwise = []
    endpoints = []
    for left in range(paths.shape[0]):
        for right in range(left + 1, paths.shape[0]):
            pairwise.append(
                np.linalg.norm(
                    projection[left] - projection[right],
                    axis=-1,
                ).mean()
            )
            endpoints.append(
                np.linalg.norm(projection[left, -1] - projection[right, -1])
            )
    if not pairwise:
        return 0.0
    projected_length = np.linalg.norm(
        np.diff(projection, axis=1),
        axis=-1,
    ).sum(axis=1)
    true_length = np.linalg.norm(np.diff(paths, axis=1), axis=-1).sum(axis=1)
    length_retention = np.mean(projected_length / np.clip(true_length, 1e-8, None))
    return float(
        0.50 * np.mean(pairwise) / spread
        + 0.30 * np.mean(endpoints) / spread
        + 0.20 * length_retention
    )


def select_unlabeled_camera(paths: np.ndarray) -> Tuple[float, float, float]:
    candidates = []
    for elevation in (18.0, 26.0, 34.0, 42.0, 50.0):
        for azimuth in np.arange(-180.0, 180.0, 12.0):
            candidates.append(
                (
                    _camera_visibility_score(paths, elevation, float(azimuth)),
                    elevation,
                    float(azimuth),
                )
            )
    score, elevation, azimuth = max(candidates, key=lambda item: item[0])
    return elevation, azimuth, score


def _set_equal_3d_limits(ax, paths: np.ndarray):
    flat = paths.reshape(-1, 3)
    minima = flat.min(axis=0)
    maxima = flat.max(axis=0)
    centers = 0.5 * (minima + maxima)
    radius = 0.55 * max(float((maxima - minima).max()), 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 0.86))


def plot_single_path_hero(
    record: dict,
    *,
    mean: np.ndarray,
    components: np.ndarray,
    explained: np.ndarray,
    pca_fit_question_count: int,
    pca_fit_increment_count: int,
    out_dir: Path,
) -> dict:
    residuals = record_residuals(record)
    paths = project_complete_paths(residuals, mean, components)
    outcomes = record_outcomes(record)
    elevation, azimuth, visibility = select_unlabeled_camera(paths)

    fig = plt.figure(figsize=(5.75, 3.55))
    ax = fig.add_axes([0.01, 0.13, 0.72, 0.80], projection="3d")
    for path, correct in zip(paths, outcomes):
        if correct:
            color = COLORS["pink"]
            linestyle = "-"
            endpoint = "*"
        else:
            color = COLORS["orange"]
            linestyle = "--"
            endpoint = "X"
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            linestyle=linestyle,
            linewidth=1.35,
            alpha=0.88,
            zorder=3,
        )
        ax.scatter(
            path[1:-1, 0],
            path[1:-1, 1],
            path[1:-1, 2],
            color=color,
            s=8,
            alpha=0.72,
            depthshade=False,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            color=color,
            marker=endpoint,
            s=42,
            edgecolor=COLORS["ink"],
            linewidth=0.35,
            depthshade=False,
            zorder=5,
        )
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        color=COLORS["blue"],
        s=28,
        marker="o",
        edgecolor=COLORS["ink"],
        linewidth=0.4,
        depthshade=False,
        zorder=6,
    )
    ax.text(0.0, 0.0, 0.0, " start", color=COLORS["blue"], fontsize=6.5)

    ax.view_init(elev=elevation, azim=azimuth)
    _set_equal_3d_limits(ax, paths)
    ax.set_xlabel("Global PC1", labelpad=2)
    ax.set_ylabel("Global PC2", labelpad=2)
    ax.set_zlabel("Global PC3", labelpad=2)
    ax.tick_params(pad=0.5, length=2)
    ax.grid(True, linewidth=0.35, alpha=0.45)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(COLORS["grid"])
    ax.set_title(
        f"Complete latent paths: {int(outcomes.sum())}/{len(outcomes)} correct",
        loc="left",
        pad=3,
        fontweight="bold",
    )
    legend = [
        Line2D([0], [0], color=COLORS["pink"], lw=1.6, marker="*", label="Correct path"),
        Line2D([0], [0], color=COLORS["orange"], lw=1.6, ls="--", marker="X", label="Wrong path"),
        Line2D([0], [0], color=COLORS["blue"], lw=0, marker="o", label="Shared origin"),
    ]
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.735, 0.82),
        handlelength=1.7,
    )
    local_summary = local_outcome_summary(
        _path_distance_matrix(residuals),
        outcomes,
    )
    if local_summary is not None:
        fig.text(
            0.745,
            0.60,
            "Question-local geometry\n"
            f"Correct radius   {local_summary['correct_local_radius']:.3f}\n"
            f"Wrong distance   {local_summary['wrong_to_local_correct_distance']:.3f}\n"
            f"Outcome margin  {local_summary['outcome_margin']:+.3f}",
            ha="left",
            va="top",
            fontsize=7,
            linespacing=1.35,
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "#FFF8FB",
                "edgecolor": COLORS["pink"],
                "linewidth": 0.75,
            },
        )

    horizontal, vertical = _camera_basis(elevation, azimuth)
    terminal_projection = np.stack(
        [
            np.tensordot(paths, horizontal, axes=([-1], [0])),
            np.tensordot(paths, vertical, axes=([-1], [0])),
        ],
        axis=-1,
    )
    terminal_ax = fig.add_axes([0.745, 0.20, 0.235, 0.31])
    for path_projection, correct in zip(terminal_projection, outcomes):
        color = COLORS["pink"] if correct else COLORS["orange"]
        linestyle = "-" if correct else "--"
        marker = "*" if correct else "X"
        terminal_ax.plot(
            path_projection[-4:, 0],
            path_projection[-4:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=1.15,
            alpha=0.82,
        )
        terminal_ax.scatter(
            path_projection[-1, 0],
            path_projection[-1, 1],
            color=color,
            marker=marker,
            s=28,
            edgecolor=COLORS["ink"],
            linewidth=0.3,
            zorder=4,
        )
    terminal_ax.set_title("Last three transitions", loc="left", fontsize=7.2, pad=3)
    terminal_ax.set_xticks([])
    terminal_ax.set_yticks([])
    terminal_ax.set_aspect("equal", adjustable="datalim")
    terminal_ax.grid(True, color=COLORS["grid"], linewidth=0.4)
    terminal_ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.07,
        0.015,
        f"PCA fit on {pca_fit_question_count} non-displayed questions "
        f"({pca_fit_increment_count:,} increments); no path scaling or offset.",
        fontsize=6.3,
        color=COLORS["muted"],
    )
    base = out_dir / f"trace_complete_path_q{int(record.get('idx', -1))}"
    save_figure(fig, base)
    plt.close(fig)
    metadata = {
        "question_id": int(record.get("idx", -1)),
        "correct_paths": int(outcomes.sum()),
        "wrong_paths": int((~outcomes).sum()),
        "pca_fit_question_count": pca_fit_question_count,
        "pca_fit_increment_count": pca_fit_increment_count,
        "explained_variance_ratio": [float(value) for value in explained],
        "increment_projection": "origin-preserving linear projection delta_z @ PC",
        "camera_protocol": "outcome-blind grid search maximizing projected path visibility",
        "camera_elevation": elevation,
        "camera_azimuth": azimuth,
        "camera_visibility_score": visibility,
        "question_local_geometry": local_summary,
        "per_path_normalization": False,
        "lane_offset": False,
        "manual_path_displacement": False,
    }
    base.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def _path_distance_matrix(residuals: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(residuals).float()
    distance = trace_path_distance(
        tensor[:, None, :, :],
        tensor[None, :, :, :],
        anchor_count=3,
        position_weight=0.45,
        direction_weight=0.35,
        step_weight=0.15,
    )
    return distance.numpy()


def local_outcome_summary(distance: np.ndarray, outcomes: np.ndarray):
    correct = np.flatnonzero(outcomes)
    wrong = np.flatnonzero(~outcomes)
    if len(correct) < 2 or len(wrong) < 1:
        return None
    correct_pairwise = distance[np.ix_(correct, correct)].copy()
    np.fill_diagonal(correct_pairwise, np.inf)
    radii = []
    wrong_distances = []
    for wrong_index in wrong:
        correct_rank = int(np.argmin(distance[correct, wrong_index]))
        correct_index = int(correct[correct_rank])
        radii.append(float(correct_pairwise[correct_rank].min()))
        wrong_distances.append(float(distance[correct_index, wrong_index]))
    radius = float(np.mean(radii))
    wrong_distance = float(np.mean(wrong_distances))
    return {
        "correct_local_radius": radius,
        "wrong_to_local_correct_distance": wrong_distance,
        "outcome_margin": wrong_distance - radius,
    }


def plot_path_distance_heatmap(record: dict, out_dir: Path) -> dict:
    residuals = record_residuals(record)
    outcomes = record_outcomes(record)
    order = np.concatenate([np.flatnonzero(outcomes), np.flatnonzero(~outcomes)])
    distance = _path_distance_matrix(residuals)
    ordered = distance[np.ix_(order, order)]
    ordered_outcomes = outcomes[order]
    labels = []
    correct_number = 0
    wrong_number = 0
    for correct in ordered_outcomes:
        if correct:
            correct_number += 1
            labels.append(f"C{correct_number}")
        else:
            wrong_number += 1
            labels.append(f"W{wrong_number}")

    cmap = LinearSegmentedColormap.from_list(
        "trace_distance",
        ["#FFF9FC", COLORS["pink"], COLORS["blue"]],
    )
    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    image = ax.imshow(ordered, cmap=cmap, aspect="equal")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(length=0)
    ax.set_title("Complete-path distance", loc="left", pad=6, fontweight="bold")
    for index, correct in enumerate(ordered_outcomes):
        color = COLORS["pink"] if correct else COLORS["orange"]
        ax.add_patch(
            plt.Rectangle(
                (index - 0.5, -0.82),
                1.0,
                0.22,
                color=color,
                clip_on=False,
                linewidth=0,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (-0.82, index - 0.5),
                0.22,
                1.0,
                color=color,
                clip_on=False,
                linewidth=0,
            )
        )
    threshold = 0.60 * float(np.nanmax(ordered)) if ordered.size else 0.0
    for row in range(ordered.shape[0]):
        for column in range(ordered.shape[1]):
            ax.text(
                column,
                row,
                f"{ordered[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if ordered[row, column] > threshold else COLORS["ink"],
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.06)
    colorbar.set_label(r"$D_{\mathrm{path}}$", rotation=90)
    colorbar.outline.set_linewidth(0.6)
    fig.text(
        0.13,
        0.015,
        "Pink: correct rollout   Orange: wrong rollout",
        fontsize=6.4,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.14, top=0.88)
    base = out_dir / f"trace_path_distance_q{int(record.get('idx', -1))}"
    save_figure(fig, base)
    plt.close(fig)
    return {
        "question_id": int(record.get("idx", -1)),
        "ordered_original_indices": order.astype(int).tolist(),
        "outcome_ordered_for_display": True,
        "distance_definition": "complete eight-transition D_path",
        "distance_matrix": distance.astype(float).tolist(),
    }


def plot_teacher_relation(records: Sequence[dict], out_dir: Path) -> dict:
    model_values = []
    teacher_values = []
    for record in records:
        model = _path_distance_matrix(record_residuals(record))
        teacher = _path_distance_matrix(record_teacher_residuals(record))
        upper = np.triu(np.ones_like(model, dtype=bool), k=1)
        model_values.extend(model[upper].tolist())
        teacher_values.extend(teacher[upper].tolist())
    model_values = np.asarray(model_values, dtype=np.float64)
    teacher_values = np.asarray(teacher_values, dtype=np.float64)
    finite = np.isfinite(model_values) & np.isfinite(teacher_values)
    model_values = model_values[finite]
    teacher_values = teacher_values[finite]
    correlation = (
        float(np.corrcoef(teacher_values, model_values)[0, 1])
        if len(model_values) > 1
        and model_values.std() > 1e-8
        and teacher_values.std() > 1e-8
        else None
    )
    mae = float(np.abs(model_values - teacher_values).mean()) if len(model_values) else None

    fig, ax = plt.subplots(figsize=(3.55, 2.85))
    ax.scatter(
        teacher_values,
        model_values,
        s=8,
        color=COLORS["pink"],
        alpha=0.34,
        edgecolors="none",
        rasterized=True,
    )
    if len(model_values):
        lower = min(float(teacher_values.min()), float(model_values.min()))
        upper = max(float(teacher_values.max()), float(model_values.max()))
        ax.plot(
            [lower, upper],
            [lower, upper],
            color=COLORS["green"],
            linewidth=1.2,
            linestyle="--",
            label="Ideal relation",
        )
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
    annotation = f"MAE = {mae:.3f}" if mae is not None else "MAE = NA"
    if correlation is not None:
        annotation += f"\nr = {correlation:.3f}"
    ax.text(
        0.04,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={
            "boxstyle": "square,pad=0.25",
            "facecolor": "white",
            "edgecolor": COLORS["grid"],
            "linewidth": 0.6,
        },
    )
    ax.set_xlabel(r"Teacher $D_{\mathrm{path}}$")
    ax.set_ylabel(r"Model $D_{\mathrm{path}}$")
    ax.set_title("Teacher-relative path relations", loc="left", pad=5, fontweight="bold")
    ax.grid(True, color=COLORS["grid"], linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    base = out_dir / "trace_teacher_relation"
    save_figure(fig, base)
    plt.close(fig)
    metadata = {
        "question_count": len(records),
        "pair_count": int(len(model_values)),
        "mae": mae,
        "pearson_r": correlation,
        "distance_definition": "complete eight-transition D_path",
    }
    base.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def select_representative_records(records: Sequence[dict], count: int) -> List[dict]:
    scored = []
    for record in records:
        outcomes = record_outcomes(record)
        if not outcomes.any() or outcomes.all():
            continue
        distance = _path_distance_matrix(record_residuals(record))
        correct = np.flatnonzero(outcomes)
        wrong = np.flatnonzero(~outcomes)
        if len(correct) < 2:
            continue
        correct_distance = distance[np.ix_(correct, correct)].copy()
        np.fill_diagonal(correct_distance, np.inf)
        margins = []
        for wrong_index in wrong:
            correct_rank = int(np.argmin(distance[correct, wrong_index]))
            correct_index = int(correct[correct_rank])
            radius = float(correct_distance[correct_rank].min())
            margins.append(float(distance[correct_index, wrong_index] - radius))
        balanced = min(int(outcomes.sum()), int((~outcomes).sum()))
        score = (balanced, float(np.mean(margins)), -int(record.get("idx", -1)))
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:count]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument("--question_ids", nargs="*", type=int, default=None)
    parser.add_argument("--n_representative", type=int, default=3)
    args = parser.parse_args()

    apply_style()
    records = load_records(
        [Path(path) for path in args.records],
        max_records=args.max_records,
    )
    if args.question_ids:
        requested = set(args.question_ids)
        selected = [
            record
            for record in records
            if int(record.get("idx", -1)) in requested
        ]
        selection_protocol = "pre-specified question IDs"
    else:
        selected = select_representative_records(
            records,
            count=args.n_representative,
        )
        selection_protocol = (
            "predefined mixed-group ranking: balance, then complete-path outcome margin"
        )
    if not selected:
        raise ValueError("No eligible records were available for the path hero")

    selected_ids = [int(record.get("idx", -1)) for record in selected]
    mean, components, explained, fit_increment_count = fit_global_increment_pca(
        records,
        excluded_ids=selected_ids,
    )
    fit_question_count = sum(
        int(record.get("idx", -1)) not in set(selected_ids)
        for record in records
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heroes = []
    heatmaps = []
    for record in selected:
        heroes.append(
            plot_single_path_hero(
                record,
                mean=mean,
                components=components,
                explained=explained,
                pca_fit_question_count=fit_question_count,
                pca_fit_increment_count=fit_increment_count,
                out_dir=out_dir,
            )
        )
        heatmaps.append(plot_path_distance_heatmap(record, out_dir))
    teacher_relation = plot_teacher_relation(records, out_dir)

    manifest = {
        "figure_contract": {
            "core_conclusion": (
                "Complete paths are shown without geometric editing; outcome "
                "separation and teacher relation are validated with D_path."
            ),
            "archetype": "separate qualitative hero plus quantitative validation",
            "backend": "Python/matplotlib",
            "font_request": "Times New Roman",
            "font_fallback_if_unavailable": "Liberation Serif",
            "exports": ["SVG", "PDF", "PNG", "TIFF 600 dpi"],
        },
        "selection_protocol": selection_protocol,
        "selected_question_ids": selected_ids,
        "pca_protocol": {
            "fit_unit": "latent residual increments",
            "fit_question_count": fit_question_count,
            "fit_increment_count": fit_increment_count,
            "displayed_questions_excluded": True,
            "outcome_labels_used_for_pca": False,
            "explained_variance_ratio": [float(value) for value in explained],
            "per_path_normalization": False,
            "manual_offset_or_lane": False,
        },
        "heroes": heroes,
        "heatmaps": heatmaps,
        "teacher_relation": teacher_relation,
    }
    manifest_name = (
        "trace_final_visual_manifest.json"
        if any("path_bottleneck" in record for record in records)
        else "trace_exchangeable_visual_manifest.json"
    )
    (out_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
