#!/usr/bin/env python3
"""Build the quantitative and qualitative TRACE trajectory evidence bundle.

Figure contract
---------------
Core conclusion:
    Held-out questions produce non-collapsed IID latent trajectories whose
    outcome-valid local modes reject nearby invalid paths, while the answer
    decoder remains bottlenecked through the complete eight-step path.
Evidence chain:
    1. A 200-question outcome-geometry summary and within-question label null.
    2. A question-held-out outcome probe in a train-fit global PCA space.
    3. Pre-registered mixed-outcome full-path plots and distance heatmaps.
Archetype:
    Asymmetric mixed-modality evidence bundle with separate, reusable panels.
Review risks:
    Three-dimensional projections are qualitative; all claims defer to the
    200-question statistics, label null, and held-out probe. PCA is fit only
    on a separate unlabeled training-question cache. No path is shifted or
    rescaled for display.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
    "Liberation Serif",
    "Nimbus Roman",
    "DejaVu Serif",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["mathtext.fontset"] = "stix"

PINK = "#E5A3BF"
PINK_DARK = "#B95C88"
BLUE = "#6687B8"
GREEN = "#69AD7C"
ORANGE = "#E5A11A"
INK = "#30343B"
GRID = "#DDE2E8"
LIGHT = "#F6F7F9"


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
            "lines.solid_capstyle": "round",
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


def load_records(path: Path, expected: int = 200) -> List[dict]:
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a record list")
    if len(records) < expected:
        raise ValueError(
            f"{path} contains {len(records)} records, expected {expected}"
        )
    records = records[:expected]
    required = (
        "rollout_schema",
        "rollout_innovations",
        "rollout_implicit_residuals",
        "rollout_action_log_stds",
        "rollout_correctness",
        "teacher_assignments",
        "visualization_contract",
    )
    for row, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"record {row} is missing {missing}")
        if record["rollout_schema"] != "iid_conditional_gaussian":
            raise ValueError(f"record {row} is not an IID policy rollout")
        contract = record["visualization_contract"]
        if contract.get("manual_offsets") is not False:
            raise ValueError(f"record {row} permits manual path offsets")
        if contract.get("per_path_rescaling") is not False:
            raise ValueError(f"record {row} permits per-path rescaling")
        residuals = record["rollout_implicit_residuals"]
        if tuple(residuals.shape[:2]) != (8, 8):
            raise ValueError(
                f"record {row} has path shape {tuple(residuals.shape)}"
            )
        if len(record["rollout_correctness"]) != 8:
            raise ValueError(f"record {row} has invalid outcome labels")
    return records


def stack_tensor(records: Sequence[dict], key: str) -> torch.Tensor:
    return torch.stack(
        [record[key].float() for record in records],
        dim=0,
    )


def fit_global_pca(
    fit_residuals: torch.Tensor,
    *,
    components: int,
) -> Dict[str, torch.Tensor]:
    flat = fit_residuals.reshape(-1, fit_residuals.shape[-1]).float()
    mean = flat.mean(dim=0)
    centered = flat - mean
    torch.manual_seed(20260719)
    _, singular_values, vectors = torch.pca_lowrank(
        centered,
        q=int(components),
        center=False,
        niter=4,
    )
    explained = singular_values.square()
    total = centered.square().sum().clamp_min(1e-12)
    return {
        "mean": mean,
        "components": vectors,
        "explained_ratio": explained / total,
    }


def project_residuals(
    residuals: torch.Tensor,
    pca: Dict[str, torch.Tensor],
) -> torch.Tensor:
    centered = residuals.float() - pca["mean"]
    return torch.einsum(
        "...h,hc->...c",
        centered,
        pca["components"],
    )


def mean_ci(
    values: Sequence[float],
    *,
    bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = rng.choice(array, size=(int(bootstrap), array.size), replace=True)
    means = draws.mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def question_geometry(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    labels = labels.astype(bool)
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    off_diagonal = ~np.eye(len(labels), dtype=bool)
    output = {
        "correct_count": float(positive.size),
        "path_diversity": float(distance_matrix[off_diagonal].mean()),
        "eligible": float(positive.size >= 2 and negative.size >= 1),
    }
    if positive.size >= 2:
        correct_matrix = distance_matrix[np.ix_(positive, positive)].copy()
        np.fill_diagonal(correct_matrix, np.inf)
        correct_radius = correct_matrix.min(axis=1)
        output["correct_local_radius"] = float(correct_radius.mean())
        output["correct_pair_distance"] = float(
            correct_matrix[np.isfinite(correct_matrix)].mean()
        )
    else:
        output["correct_local_radius"] = float("nan")
        output["correct_pair_distance"] = float("nan")
    if positive.size and negative.size:
        wrong_to_correct = distance_matrix[np.ix_(negative, positive)].min(
            axis=1
        )
        output["wrong_to_correct_distance"] = float(
            wrong_to_correct.mean()
        )
        output["hard_negative_distance"] = float(
            wrong_to_correct.min()
        )
    else:
        output["wrong_to_correct_distance"] = float("nan")
        output["hard_negative_distance"] = float("nan")
    if output["eligible"]:
        output["outcome_margin"] = (
            output["wrong_to_correct_distance"]
            - output["correct_local_radius"]
        )
        output["hard_outcome_margin"] = (
            output["hard_negative_distance"]
            - output["correct_local_radius"]
        )
    else:
        output["outcome_margin"] = float("nan")
        output["hard_outcome_margin"] = float("nan")
    if negative.size >= 2:
        wrong_matrix = distance_matrix[np.ix_(negative, negative)]
        wrong_off_diagonal = ~np.eye(negative.size, dtype=bool)
        output["wrong_pair_distance"] = float(
            wrong_matrix[wrong_off_diagonal].mean()
        )
    else:
        output["wrong_pair_distance"] = float("nan")
    return output


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positive = labels.sum()
    negative = labels.size - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while (
            end < labels.size
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[labels == 1].sum()
    return float(
        (rank_sum - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positive = labels.sum()
    if positive == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(
        1,
        labels.size + 1,
    )
    return float((precision * sorted_labels).sum() / positive)


def roc_curve(labels: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.int64)
    positive = max(1, int(sorted_labels.sum()))
    negative = max(1, int(sorted_labels.size - sorted_labels.sum()))
    tpr = np.concatenate([[0.0], np.cumsum(sorted_labels) / positive])
    fpr = np.concatenate(
        [[0.0], np.cumsum(1 - sorted_labels) / negative]
    )
    return fpr, tpr


def path_features(
    projected_residuals: torch.Tensor,
    original_residuals: torch.Tensor,
) -> np.ndarray:
    directions = F_normalize(projected_residuals)
    step_norms = original_residuals.float().norm(dim=-1) / math.sqrt(
        original_residuals.shape[-1]
    )
    features = torch.cat(
        [
            directions.flatten(start_dim=2),
            step_norms,
        ],
        dim=-1,
    )
    return features.numpy()


def F_normalize(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def fit_ridge_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate(
        [standardized, np.ones((len(features), 1))],
        axis=1,
    )
    labels_pm = labels.astype(np.float64) * 2.0 - 1.0
    positives = max(1, int(labels.sum()))
    negatives = max(1, int(len(labels) - labels.sum()))
    weights = np.where(
        labels > 0,
        len(labels) / (2.0 * positives),
        len(labels) / (2.0 * negatives),
    )
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = labels_pm * np.sqrt(weights)
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return coefficients, mean, scale


def question_heldout_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int = 5,
    ridge: float = 1.0,
) -> Dict[str, object]:
    n_questions, group_size, feature_size = features.shape
    scores = np.zeros((n_questions, group_size), dtype=np.float64)
    for fold in range(int(folds)):
        test_questions = np.arange(n_questions) % int(folds) == fold
        train_questions = ~test_questions
        train_x = features[train_questions].reshape(-1, feature_size)
        train_y = labels[train_questions].reshape(-1)
        test_x = features[test_questions].reshape(-1, feature_size)
        coefficients, mean, scale = fit_ridge_classifier(
            train_x,
            train_y,
            ridge=ridge,
        )
        standardized = (test_x - mean) / scale
        design = np.concatenate(
            [standardized, np.ones((len(test_x), 1))],
            axis=1,
        )
        fold_scores = design @ coefficients
        scores[test_questions] = fold_scores.reshape(-1, group_size)
    flat_labels = labels.reshape(-1)
    flat_scores = scores.reshape(-1)
    predictions = flat_scores >= 0.0
    positive_mask = flat_labels == 1
    negative_mask = ~positive_mask
    balanced_accuracy = 0.5 * (
        predictions[positive_mask].mean()
        + (~predictions[negative_mask]).mean()
    )
    return {
        "scores": scores,
        "auroc": auroc(flat_labels, flat_scores),
        "auprc": auprc(flat_labels, flat_scores),
        "balanced_accuracy": float(balanced_accuracy),
    }


def question_bootstrap_probe_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values = []
    for _ in range(int(bootstrap)):
        indices = rng.integers(0, labels.shape[0], size=labels.shape[0])
        value = auroc(
            labels[indices].reshape(-1),
            scores[indices].reshape(-1),
        )
        if np.isfinite(value):
            values.append(value)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def permutation_null(
    distance_matrices: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    observed_rows = [
        question_geometry(distance, outcome)
        for distance, outcome in zip(distance_matrices, labels)
    ]
    observed = np.mean(
        [
            row["outcome_margin"]
            for row in observed_rows
            if row["eligible"]
        ]
    )
    null = []
    for _ in range(int(permutations)):
        values = []
        for distance, outcome in zip(distance_matrices, labels):
            shuffled = rng.permutation(outcome)
            row = question_geometry(distance, shuffled)
            if row["eligible"]:
                values.append(row["outcome_margin"])
        null.append(float(np.mean(values)))
    null = np.asarray(null)
    p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    return {
        "observed_margin": float(observed),
        "null_mean": float(null.mean()),
        "excess": float(observed - null.mean()),
        "p_value": p_value,
        "null_ci": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
    }


def choose_camera(paths: np.ndarray) -> Tuple[float, float, float]:
    best = None
    for elevation in (15.0, 25.0, 35.0, 45.0):
        for azimuth in np.arange(0.0, 360.0, 20.0):
            elev = np.deg2rad(elevation)
            azim = np.deg2rad(azimuth)
            view = np.array(
                [
                    np.cos(elev) * np.cos(azim),
                    np.cos(elev) * np.sin(azim),
                    np.sin(elev),
                ]
            )
            right = np.cross(view, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(right) < 1e-6:
                continue
            right /= np.linalg.norm(right)
            up = np.cross(right, view)
            projected = np.stack(
                [paths @ right, paths @ up],
                axis=-1,
            )
            endpoint = projected[:, -1]
            endpoint_distances = np.linalg.norm(
                endpoint[:, None] - endpoint[None, :],
                axis=-1,
            )
            path_distances = np.linalg.norm(
                projected[:, None] - projected[None, :],
                axis=-1,
            ).mean(axis=-1)
            mask = ~np.eye(paths.shape[0], dtype=bool)
            score = (
                np.median(endpoint_distances[mask])
                + np.median(path_distances[mask])
            )
            candidate = (float(score), elevation, azimuth)
            if best is None or candidate > best:
                best = candidate
    return best


def common_axis_limits(path_sets: Sequence[np.ndarray]):
    points = np.concatenate(
        [paths.reshape(-1, 3) for paths in path_sets],
        axis=0,
    )
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    spans = upper - lower
    fallback = max(float(spans.max()) * 0.02, 1e-3)
    padding = np.maximum(spans * 0.08, fallback)
    return list(zip(lower - padding, upper + padding))


def plot_trajectory(
    record: dict,
    projected_paths: np.ndarray,
    projected_map: np.ndarray,
    *,
    camera: Tuple[float, float, float],
    limits,
    output_base: Path,
    stage_label: str = "",
):
    labels = np.asarray(record["rollout_correctness"], dtype=bool)
    paths = np.concatenate(
        [np.zeros((len(projected_paths), 1, 3)), projected_paths.cumsum(1)],
        axis=1,
    )
    map_path = np.concatenate(
        [np.zeros((1, 3)), projected_map.cumsum(0)],
        axis=0,
    )
    fig = plt.figure(figsize=(3.55, 3.15))
    ax = fig.add_subplot(111, projection="3d")
    for index, (path, correct) in enumerate(zip(paths, labels)):
        color = PINK if correct else ORANGE
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=color,
            linewidth=1.35,
            alpha=0.82,
        )
        ax.scatter(
            path[1:-1, 0],
            path[1:-1, 1],
            path[1:-1, 2],
            color=color,
            s=5,
            alpha=0.75,
            depthshade=False,
        )
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            path[-1, 2],
            color=color,
            marker="*" if correct else "X",
            s=36 if correct else 27,
            edgecolor=INK,
            linewidth=0.35,
            depthshade=False,
        )
    ax.plot(
        map_path[:, 0],
        map_path[:, 1],
        map_path[:, 2],
        color=BLUE,
        linewidth=2.0,
        linestyle="--",
        label="MAP path",
    )
    ax.scatter(
        [0],
        [0],
        [0],
        color=GREEN,
        edgecolor=INK,
        linewidth=0.4,
        s=24,
        label="Shared origin",
        depthshade=False,
    )
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1.20, 1.0, 0.82))
    ax.set_xlabel("Global PC1", labelpad=-2)
    ax.set_ylabel("Global PC2", labelpad=-8)
    ax.set_zlabel("Global PC3", labelpad=-8)
    ax.view_init(elev=camera[1], azim=camera[2])
    ax.tick_params(pad=0)
    ax.grid(True, color=GRID, linewidth=0.45)
    title_prefix = f"{stage_label} | " if stage_label else ""
    ax.set_title(
        f"{title_prefix}Question {record['idx']} | "
        f"{int(labels.sum())}/8 correct paths",
        loc="left",
        fontweight="bold",
        pad=3,
    )
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=PINK,
            marker="*",
            label="Outcome-correct",
        ),
        mpl.lines.Line2D(
            [],
            [],
            color=ORANGE,
            marker="X",
            label="Wrong",
        ),
        mpl.lines.Line2D(
            [],
            [],
            color=BLUE,
            linestyle="--",
            label="MAP",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=3,
        handlelength=1.4,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.13, top=0.89)
    save_figure(fig, output_base)


def plot_distance_heatmap(record: dict, output_base: Path):
    matrix = record["rollout_path_distance_matrix"].float().numpy()
    labels = np.asarray(record["rollout_correctness"], dtype=np.int64)
    order = np.argsort(-labels, kind="stable")
    matrix = matrix[np.ix_(order, order)]
    labels = labels[order]
    cmap = LinearSegmentedColormap.from_list(
        "trace_distance",
        [BLUE, "#F7F7F8", PINK],
    )
    fig, ax = plt.subplots(figsize=(2.75, 2.45))
    image = ax.imshow(matrix, cmap=cmap, aspect="equal")
    tick_labels = [
        f"{'C' if label else 'W'}{rank + 1}"
        for rank, label in enumerate(labels)
    ]
    ax.set_xticks(range(8), tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(8), tick_labels)
    for tick, label in zip(ax.get_xticklabels(), labels):
        tick.set_color(PINK_DARK if label else ORANGE)
    for tick, label in zip(ax.get_yticklabels(), labels):
        tick.set_color(PINK_DARK if label else ORANGE)
    ax.set_title(
        f"Complete-path distance | q{record['idx']}",
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("IID rollout")
    ax.set_ylabel("IID rollout")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(r"$D_{\mathrm{TRACE}}$")
    fig.tight_layout(pad=0.5)
    save_figure(fig, output_base)


def plot_teacher_assignments(record: dict, output_base: Path):
    assignments = [
        assignment.float().numpy()
        for assignment in record["teacher_assignments"]
    ]
    mode_ids = record["teacher_semantic_mode_ids"].tolist()
    cmap = LinearSegmentedColormap.from_list(
        "assignment",
        ["#F7F7F8", PINK, PINK_DARK],
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(4.8, 3.65),
        layout="constrained",
    )
    for view, (ax, assignment, mode_id) in enumerate(
        zip(axes.flat, assignments, mode_ids)
    ):
        image = ax.imshow(
            assignment,
            cmap=cmap,
            vmin=0.0,
            vmax=max(0.25, float(assignment.max())),
            aspect="auto",
        )
        ax.set_title(
            f"T{view + 1} | rationale mode {mode_id + 1}",
            loc="left",
            fontsize=7,
            fontweight="bold",
        )
        ax.set_yticks(range(0, assignment.shape[0], 2))
        ax.set_xticks(range(assignment.shape[1]))
        if view < 2:
            ax.set_xticklabels([])
        if view % 2:
            ax.set_yticklabels([])
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
        label="Assignment weight",
    )
    fig.suptitle(
        f"Stochastic monotone teacher compression | q{record['idx']}",
        x=0.02,
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    fig.supxlabel("CoT step", fontsize=7)
    fig.supylabel("Latent transition", fontsize=7)
    save_figure(fig, output_base)


def plot_geometry_separation(rows: Sequence[dict], output_base: Path):
    eligible = [row for row in rows if row["eligible"]]
    radius = np.asarray(
        [row["correct_local_radius"] for row in eligible]
    )
    wrong = np.asarray(
        [row["wrong_to_correct_distance"] for row in eligible]
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    for left, right in zip(radius, wrong):
        ax.plot(
            [0, 1],
            [left, right],
            color="#C9CED5",
            linewidth=0.45,
            alpha=0.45,
            zorder=1,
        )
    jitter = np.linspace(-0.07, 0.07, len(radius))
    ax.scatter(
        np.zeros_like(radius) + jitter,
        radius,
        s=10,
        color=PINK,
        edgecolor=INK,
        linewidth=0.25,
        alpha=0.75,
        zorder=2,
    )
    ax.scatter(
        np.ones_like(wrong) + jitter,
        wrong,
        s=10,
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.25,
        alpha=0.75,
        zorder=2,
    )
    ax.plot(
        [0, 1],
        [radius.mean(), wrong.mean()],
        color=INK,
        linewidth=2.0,
        marker="o",
        markersize=4,
        zorder=3,
    )
    ax.set_xticks(
        [0, 1],
        ["Nearest correct peer", "Wrong-to-correct"],
    )
    ax.set_ylabel(r"Complete-path distance $D_{\mathrm{TRACE}}$")
    ax.set_title(
        "Wrong-answer paths lie outside local correct modes",
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.98,
        f"mean margin = {np.mean(wrong - radius):+.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=PINK_DARK,
        fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_base)


def plot_step_profile(
    records: Sequence[dict],
    residuals: torch.Tensor,
    output_base: Path,
):
    log_stds = stack_tensor(records, "rollout_action_log_stds")
    policy_std = torch.exp(log_stds).mean(dim=(0, 1, 3)).numpy()
    step_norm = (
        residuals.norm(dim=-1) / math.sqrt(residuals.shape[-1])
    ).mean(dim=(0, 1)).numpy()
    steps = np.arange(1, residuals.shape[2] + 1)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.plot(
        steps,
        policy_std,
        color=PINK_DARK,
        marker="o",
        markersize=4,
        linewidth=1.7,
        label="Policy std",
    )
    ax.set_xlabel("Latent transition")
    ax.set_ylabel("Action standard deviation", color=PINK_DARK)
    ax.tick_params(axis="y", colors=PINK_DARK)
    ax.set_xticks(steps)
    second = ax.twinx()
    second.plot(
        steps,
        step_norm,
        color=GREEN,
        marker="s",
        markersize=3.5,
        linewidth=1.5,
        label="Step norm",
    )
    second.set_ylabel("Normalized path step", color=GREEN)
    second.tick_params(axis="y", colors=GREEN)
    ax.set_title(
        "Diversity is retained without transition collapse",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    handles = ax.get_lines() + second.get_lines()
    ax.legend(handles, [line.get_label() for line in handles], loc="best")
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_base)


def plot_probe_roc(
    labels: np.ndarray,
    scores: np.ndarray,
    auroc_value: float,
    output_base: Path,
):
    fpr, tpr = roc_curve(labels.reshape(-1), scores.reshape(-1))
    fig, ax = plt.subplots(figsize=(2.85, 2.65))
    ax.plot(
        fpr,
        tpr,
        color=PINK_DARK,
        linewidth=2.0,
        label=f"TRACE path probe ({auroc_value:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color=BLUE,
        linestyle="--",
        linewidth=1.0,
        label="Chance",
    )
    ax.fill_between(fpr, 0, tpr, color=PINK, alpha=0.18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(
        "Outcome transfers across held-out questions",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(color=GRID, linewidth=0.5)
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_base)


def write_rows(rows: Sequence[dict], path: Path):
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def figure_qa(output_dir: Path) -> dict:
    from PIL import Image, ImageStat

    figures = []
    for svg in sorted(output_dir.glob("*.svg")):
        pdf = svg.with_suffix(".pdf")
        tiff = svg.with_suffix(".tiff")
        svg_text = svg.read_text()
        if "<text" not in svg_text:
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
    return {"status": "PASS", "figures": figures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-records", type=Path, required=True)
    parser.add_argument("--additional-fit-records", type=Path)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    fit_records = load_records(args.fit_records)
    additional_fit_records = (
        load_records(args.additional_fit_records)
        if args.additional_fit_records is not None
        else None
    )
    if additional_fit_records is not None:
        fit_indices = [int(record["idx"]) for record in fit_records]
        additional_indices = [
            int(record["idx"]) for record in additional_fit_records
        ]
        if fit_indices != additional_indices:
            raise ValueError(
                "shared PCA fit caches do not contain the same questions"
            )
    records = load_records(args.records)
    fit_tensors = [
        stack_tensor(fit_records, "rollout_implicit_residuals")
    ]
    if additional_fit_records is not None:
        fit_tensors.append(
            stack_tensor(
                additional_fit_records,
                "rollout_implicit_residuals",
            )
        )
    fit_residuals = torch.cat(fit_tensors, dim=0)
    residuals = stack_tensor(records, "rollout_implicit_residuals")
    labels = np.asarray(
        [record["rollout_correctness"] for record in records],
        dtype=np.int64,
    )
    pca = fit_global_pca(fit_residuals, components=16)
    torch.save(pca, args.output_dir / "global_train_fit_pca.pt")
    projected = project_residuals(residuals, pca)
    projected_map = project_residuals(
        torch.stack(
            [record["map_implicit_residuals"].float() for record in records]
        ),
        pca,
    )
    distance_matrices = np.stack(
        [
            record["rollout_path_distance_matrix"].float().numpy()
            for record in records
        ]
    )
    rows = [
        {
            "idx": int(record["idx"]),
            **question_geometry(distance, outcome),
        }
        for record, distance, outcome in zip(
            records,
            distance_matrices,
            labels,
        )
    ]
    write_rows(rows, args.output_dir / "question_geometry.csv")

    rng = np.random.default_rng(20260719)
    eligible_rows = [row for row in rows if row["eligible"]]
    metrics = {}
    for key in (
        "correct_local_radius",
        "wrong_to_correct_distance",
        "outcome_margin",
        "hard_outcome_margin",
        "path_diversity",
        "wrong_pair_distance",
    ):
        values = [
            row[key]
            for row in (
                eligible_rows if key != "path_diversity" else rows
            )
            if np.isfinite(row[key])
        ]
        mean, lower, upper = mean_ci(
            values,
            bootstrap=args.bootstrap,
            rng=rng,
        )
        metrics[key] = {
            "mean": mean,
            "ci95": [lower, upper],
            "n_questions": len(values),
        }
    metrics["eligible_questions"] = len(eligible_rows)
    metrics["mixed_questions"] = int(
        sum(0 < row["correct_count"] < 8 for row in rows)
    )
    metrics["all_correct_questions"] = int(
        sum(row["correct_count"] == 8 for row in rows)
    )
    metrics["all_wrong_questions"] = int(
        sum(row["correct_count"] == 0 for row in rows)
    )
    metrics["permutation_null"] = permutation_null(
        distance_matrices,
        labels,
        permutations=args.permutations,
        rng=rng,
    )

    features = path_features(projected[..., :16], residuals)
    probe = question_heldout_probe(features, labels)
    probe_ci = question_bootstrap_probe_ci(
        labels,
        probe["scores"],
        bootstrap=args.bootstrap,
        rng=rng,
    )
    metrics["question_heldout_probe"] = {
        "auroc": probe["auroc"],
        "auroc_ci95": list(probe_ci),
        "auprc": probe["auprc"],
        "balanced_accuracy": probe["balanced_accuracy"],
        "folds": 5,
        "feature_projection": "16-PC train-fit residual directions + step norms",
    }
    metrics["pca"] = {
        "fit_questions": len(fit_records),
        "fit_split": "training",
        "fit_model_stages": len(fit_tensors),
        "fit_trajectory_sets": int(fit_residuals.shape[0]),
        "outcome_labels_used": False,
        "explained_ratio_first3": pca["explained_ratio"][:3].tolist(),
        "explained_ratio_first16": float(
            pca["explained_ratio"][:16].sum()
        ),
    }

    selected = [
        index
        for index, row in enumerate(rows)
        if 2 <= row["correct_count"] <= 7
    ][:3]
    if len(selected) < 3:
        selected.extend(
            [
                index
                for index, row in enumerate(rows)
                if 0 < row["correct_count"] < 8 and index not in selected
            ][: 3 - len(selected)]
        )
    if not selected:
        selected = [0]
    display_parameters = {}
    display_audit = []
    for index in selected:
        rollout_paths = np.concatenate(
            [
                np.zeros((8, 1, 3)),
                projected[index, ..., :3].cumsum(dim=1).numpy(),
            ],
            axis=1,
        )
        map_path = np.concatenate(
            [
                np.zeros((1, 3)),
                projected_map[index, ..., :3].cumsum(dim=0).numpy(),
            ],
            axis=0,
        )
        camera = choose_camera(rollout_paths)
        limits = common_axis_limits(
            [rollout_paths, map_path[None, ...]]
        )
        display_parameters[index] = (camera, limits)
        display_audit.append(
            {
                "record_index": int(records[index]["idx"]),
                "camera_score": camera[0],
                "elevation": camera[1],
                "azimuth": camera[2],
                "axis_limits": [
                    [float(lower), float(upper)]
                    for lower, upper in limits
                ],
            }
        )
    metrics["qualitative_selection"] = {
        "rule": "first three test questions with 2-7 correct IID paths",
        "outcome_counts_only": True,
        "geometry_used_for_selection": False,
        "record_indices": [int(records[index]["idx"]) for index in selected],
        "display_parameters": display_audit,
        "display_rule": (
            "Per-question camera and axis limits use all eight unlabeled "
            "paths. Axis-box aspect is standardized for readability; no "
            "individual path is moved or rescaled and numeric PC coordinates "
            "remain unchanged."
        ),
    }

    plot_geometry_separation(
        rows,
        args.output_dir / "geometry_separation",
    )
    plot_step_profile(
        records,
        residuals,
        args.output_dir / "policy_step_profile",
    )
    plot_probe_roc(
        labels,
        probe["scores"],
        probe["auroc"],
        args.output_dir / "outcome_probe_roc",
    )
    for index in selected:
        record = records[index]
        camera, limits = display_parameters[index]
        plot_trajectory(
            record,
            projected[index, ..., :3].numpy(),
            projected_map[index, ..., :3].numpy(),
            camera=camera,
            limits=limits,
            output_base=(
                args.output_dir / f"trajectory_q{record['idx']}"
            ),
        )
        plot_distance_heatmap(
            record,
            args.output_dir / f"path_distance_q{record['idx']}",
        )
        plot_teacher_assignments(
            record,
            args.output_dir / f"teacher_assignment_q{record['idx']}",
        )

    np.savez_compressed(
        args.output_dir / "source_data.npz",
        labels=labels,
        projected_residuals=projected.numpy(),
        projected_map_residuals=projected_map.numpy(),
        distance_matrices=distance_matrices,
        probe_scores=probe["scores"],
    )
    contract = {
        "core_conclusion": (
            "Outcome-valid IID trajectories form coherent local modes that "
            "reject invalid paths without collapsing transition diversity."
        ),
        "archetype": "asymmetric_mixed_modality_evidence_bundle",
        "pca_fit": (
            "same 200 separate training questions from all supplied model "
            "stages; labels unused"
        ),
        "displayed_questions": metrics["qualitative_selection"],
        "manual_offsets": False,
        "per_path_rescaling": False,
        "outcome_used_for_projection": False,
        "exports": ["SVG", "PDF", "TIFF 600 dpi"],
        "review_risk": (
            "3D panels are qualitative and cannot override the question-level "
            "bootstrap, permutation null, or held-out outcome probe."
        ),
    }
    (args.output_dir / "figure_contract.json").write_text(
        json.dumps(contract, indent=2)
    )
    (args.output_dir / "geometry_summary.json").write_text(
        json.dumps(metrics, indent=2)
    )
    markdown = [
        "# TRACE 200-Question Geometry Summary",
        "",
        f"- Eligible correct/correct/wrong questions: "
        f"**{metrics['eligible_questions']} / 200**.",
        f"- Correct local radius: "
        f"**{metrics['correct_local_radius']['mean']:.4f}** "
        f"(95% CI {metrics['correct_local_radius']['ci95'][0]:.4f}, "
        f"{metrics['correct_local_radius']['ci95'][1]:.4f}).",
        f"- Wrong-to-correct distance: "
        f"**{metrics['wrong_to_correct_distance']['mean']:.4f}** "
        f"(95% CI {metrics['wrong_to_correct_distance']['ci95'][0]:.4f}, "
        f"{metrics['wrong_to_correct_distance']['ci95'][1]:.4f}).",
        f"- Outcome margin: **{metrics['outcome_margin']['mean']:+.4f}** "
        f"(95% CI {metrics['outcome_margin']['ci95'][0]:+.4f}, "
        f"{metrics['outcome_margin']['ci95'][1]:+.4f}).",
        f"- Within-question label-null excess: "
        f"**{metrics['permutation_null']['excess']:+.4f}**, "
        f"`p={metrics['permutation_null']['p_value']:.4g}`.",
        f"- Question-held-out path probe AUROC: "
        f"**{probe['auroc']:.3f}** "
        f"(95% CI {probe_ci[0]:.3f}, {probe_ci[1]:.3f}).",
        "",
        "PCA was fit on the same 200 separate training questions from all "
        "supplied model stages without labels. "
        "Displayed test questions were selected only by rollout outcome count. "
        "No path offset or per-path rescaling was applied.",
    ]
    (args.output_dir / "geometry_summary.md").write_text(
        "\n".join(markdown)
    )
    qa = figure_qa(args.output_dir)
    (args.output_dir / "figure_qa.json").write_text(
        json.dumps(qa, indent=2)
    )
    print(json.dumps({"metrics": metrics, "figure_qa": qa}, indent=2))


if __name__ == "__main__":
    main()
