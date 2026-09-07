#!/usr/bin/env python3
"""Build the quantitative and qualitative TRACE trajectory evidence bundle.

Figure contract
---------------
Core conclusion:
    The fixed PLAN--SOLVE--REFINE program preserves its Stage-1 semantic
    formation while retaining role-conditioned exploration, and the final
    COMMIT transition is conditionally deterministic and is the only latent
    state exposed to answer generation.
Evidence chain:
    1. Role-level semantic similarity on 200 questions x eight IID paths.
    2. Role-level Gaussian standard deviation/entropy and COMMIT audit.
    3. Diagnostic-only outcome/corridor geometry in a shared train-fit PCA.
Archetype:
    Asymmetric mixed-modality evidence bundle with separate, reusable panels.
Review risks:
    Teacher cosine similarity measures semantic alignment, not answer
    correctness. Three-dimensional projections and legacy corridors are
    diagnostic only. PCA is fit only on a separate unlabeled training cache;
    no path is shifted or rescaled for display.
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
ROLE_SCHEMA = (
    "PLAN,SOLVE1,SOLVE2,SOLVE3,SOLVE4,SOLVE5,REFINE,COMMIT"
)
ROLE_NAMES = (
    "PLAN",
    "SOLVE1",
    "SOLVE2",
    "SOLVE3",
    "SOLVE4",
    "SOLVE5",
    "REFINE",
    "COMMIT",
)
GAUSSIAN_ENTROPY_CONSTANT = 0.5 * math.log(2.0 * math.pi * math.e)
COMMIT_AUDIT_TOLERANCE = 5e-4


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
        "idx",
        "rollout_schema",
        "rollout_innovations",
        "rollout_actions",
        "rollout_action_means",
        "rollout_implicit_residuals",
        "map_implicit_residuals",
        "rollout_action_log_stds",
        "rollout_path_distance_matrix",
        "rollout_correctness",
        "role_schema",
        "role_teacher_plan",
        "role_teacher_solve",
        "role_teacher_solve_mask",
        "role_teacher_summary",
        "map_role_rewards",
        "rollout_role_rewards",
        "answer_latent_attention_access",
        "answer_latent_attention_role",
        "visualization_contract",
    )
    for row, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"record {row} is missing {missing}")
        if record["rollout_schema"] != (
            "iid_role_conditioned_gaussian_with_deterministic_commit"
        ):
            raise ValueError(f"record {row} is not a role-aware IID rollout")
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
        if record["role_schema"] != ROLE_SCHEMA:
            raise ValueError(f"record {row} has invalid role schema")
        actions = record["rollout_actions"]
        means = record["rollout_action_means"]
        log_stds = record["rollout_action_log_stds"]
        if tuple(actions.shape[:2]) != (8, 8):
            raise ValueError(f"record {row} has invalid rollout actions")
        if means.shape != actions.shape or log_stds.shape != actions.shape:
            raise ValueError(f"record {row} has misaligned policy tensors")
        if tuple(record["map_role_rewards"].shape) != (8,):
            raise ValueError(f"record {row} has invalid MAP role rewards")
        if tuple(record["rollout_role_rewards"].shape) != (8, 8):
            raise ValueError(f"record {row} has invalid rollout role rewards")
        plan = record["role_teacher_plan"]
        solves = record["role_teacher_solve"]
        solve_mask = record["role_teacher_solve_mask"]
        summary = record["role_teacher_summary"]
        if plan.ndim != 1 or summary.shape != plan.shape:
            raise ValueError(f"record {row} has invalid teacher endpoints")
        if tuple(solves.shape) != (5, plan.numel()):
            raise ValueError(f"record {row} has invalid SOLVE targets")
        if tuple(solve_mask.shape) != (5,) or not bool(solve_mask.any()):
            raise ValueError(f"record {row} has invalid SOLVE mask")
        if int(record["answer_latent_attention_access"]) != 1:
            raise ValueError(f"record {row} does not use COMMIT-only readout")
        if record["answer_latent_attention_role"] != "COMMIT":
            raise ValueError(f"record {row} has invalid answer readout role")
    return records


def stack_tensor(records: Sequence[dict], key: str) -> torch.Tensor:
    return torch.stack(
        [record[key].float() for record in records],
        dim=0,
    )


def role_question_metrics(record: dict) -> Dict[str, float]:
    """Reduce one strict 8-path record to auditable role-level evidence."""
    map_rewards = record["map_role_rewards"].float().numpy()
    rollout_rewards = record["rollout_role_rewards"].float().numpy()
    solve_mask = record["role_teacher_solve_mask"].bool().numpy()
    log_stds = record["rollout_action_log_stds"].float().numpy()
    actions = record["rollout_actions"].float().numpy()
    means = record["rollout_action_means"].float().numpy()

    solve_steps = np.flatnonzero(solve_mask) + 1
    semantic_groups = {
        "plan": np.asarray([0], dtype=np.int64),
        "solve": solve_steps.astype(np.int64),
        "refine": np.asarray([6], dtype=np.int64),
    }
    exploration_groups = {
        "plan": np.asarray([0], dtype=np.int64),
        "solve": np.arange(1, 6, dtype=np.int64),
        "refine": np.asarray([6], dtype=np.int64),
    }
    output = {
        "role_schema_valid": float(record["role_schema"] == ROLE_SCHEMA),
        "active_solve_slots": float(solve_mask.sum()),
        "corridor_diagnostic_only": 1.0,
    }
    for role, semantic_indices in semantic_groups.items():
        policy_indices = exploration_groups[role]
        selected_log_stds = log_stds[:, policy_indices, :]
        output[f"map_{role}_similarity"] = float(
            map_rewards[semantic_indices].mean()
        )
        output[f"rollout_{role}_similarity"] = float(
            rollout_rewards[:, semantic_indices].mean()
        )
        output[f"{role}_action_std"] = float(
            np.exp(selected_log_stds).mean()
        )
        output[f"{role}_action_entropy"] = float(
            (selected_log_stds + GAUSSIAN_ENTROPY_CONSTANT).mean()
        )

    commit_action_gap = float(np.max(np.abs(actions[:, 7] - means[:, 7])))
    commit_reward_gap = float(
        max(
            abs(float(map_rewards[7])),
            float(np.max(np.abs(rollout_rewards[:, 7]))),
        )
    )
    commit_log_std_placeholder = float(np.max(np.abs(log_stds[:, 7])))
    commit_readout_only = float(
        int(record["answer_latent_attention_access"]) == 1
        and record["answer_latent_attention_role"] == "COMMIT"
    )
    output.update(
        {
            "commit_action_mean_abs_max": commit_action_gap,
            "commit_role_reward_abs_max": commit_reward_gap,
            "commit_log_std_placeholder_abs_max": (
                commit_log_std_placeholder
            ),
            "commit_readout_only": commit_readout_only,
            "commit_deterministic": float(
                commit_action_gap <= COMMIT_AUDIT_TOLERANCE
                and commit_reward_gap <= COMMIT_AUDIT_TOLERANCE
                and commit_log_std_placeholder <= COMMIT_AUDIT_TOLERANCE
                and commit_readout_only > 0.5
            ),
        }
    )
    return output


def summarize_role_metrics(
    rows: Sequence[dict],
    *,
    bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Bootstrap role evidence at the question level (never by path)."""

    def summary(key: str) -> Dict[str, object]:
        values = [row[key] for row in rows if np.isfinite(row[key])]
        mean, lower, upper = mean_ci(
            values,
            bootstrap=bootstrap,
            rng=rng,
        )
        return {
            "mean": mean,
            "ci95": [lower, upper],
            "n_questions": len(values),
        }

    semantics = {}
    exploration = {}
    for role in ("plan", "solve", "refine"):
        semantics[role.upper()] = {
            "map_teacher_similarity": summary(
                f"map_{role}_similarity"
            ),
            "rollout_teacher_similarity": summary(
                f"rollout_{role}_similarity"
            ),
        }
        exploration[role.upper()] = {
            "action_std": summary(f"{role}_action_std"),
            "gaussian_entropy_per_dimension": summary(
                f"{role}_action_entropy"
            ),
        }

    deterministic = np.asarray(
        [row["commit_deterministic"] for row in rows],
        dtype=np.float64,
    )
    return {
        "semantics": semantics,
        "exploration": exploration,
        "commit": {
            "deterministic_pass_rate": float(deterministic.mean()),
            "passing_questions": int(deterministic.sum()),
            "n_questions": len(rows),
            "max_action_minus_mean_abs": float(
                max(row["commit_action_mean_abs_max"] for row in rows)
            ),
            "max_role_reward_abs": float(
                max(row["commit_role_reward_abs_max"] for row in rows)
            ),
            "max_log_std_placeholder_abs": float(
                max(
                    row["commit_log_std_placeholder_abs_max"]
                    for row in rows
                )
            ),
            "commit_only_readout_rate": float(
                np.mean([row["commit_readout_only"] for row in rows])
            ),
            "audit_tolerance": COMMIT_AUDIT_TOLERANCE,
            "audit_definition": (
                "COMMIT action equals its conditional mean; COMMIT semantic "
                "reward and shape-compatible log-std placeholder are zero; "
                "answer generation reads only the COMMIT state."
            ),
        },
    }


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


def pairwise_euclidean_matrix(signatures: np.ndarray) -> np.ndarray:
    signatures = np.asarray(signatures, dtype=np.float64)
    if signatures.ndim != 2:
        raise ValueError("signatures must have shape [path, feature]")
    differences = signatures[:, None, :] - signatures[None, :, :]
    return np.linalg.norm(differences, axis=-1) / math.sqrt(
        max(1, signatures.shape[-1])
    )


def matrix_distance_correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("distance matrices must have equal square shapes")
    upper = np.triu_indices(left.shape[0], k=1)
    left_values = left[upper]
    right_values = right[upper]
    left_values = left_values - left_values.mean()
    right_values = right_values - right_values.mean()
    denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_values, right_values) / denominator)


def formation_geometry(
    actions: np.ndarray,
    progress_centers: np.ndarray,
    path_distances: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    actions = np.asarray(actions, dtype=np.float64)
    centers = np.asarray(progress_centers, dtype=np.float64)
    if actions.ndim != 3:
        raise ValueError("actions must have shape [path, step, action]")
    if centers.shape != actions.shape[:2]:
        raise ValueError("progress centers must align with action paths")
    if path_distances.shape != (actions.shape[0], actions.shape[0]):
        raise ValueError("path distance matrix has an invalid shape")
    if not np.all(np.diff(centers, axis=1) > 0):
        raise ValueError("action-conditioned progress is not strictly ordered")
    action_distances = pairwise_euclidean_matrix(
        actions.reshape(actions.shape[0], -1)
    )
    center_distances = pairwise_euclidean_matrix(centers)
    upper = np.triu_indices(actions.shape[0], k=1)
    return (
        {
            "progress_schedule_diversity": float(
                centers.std(axis=0).mean()
            ),
            "progress_pair_distance": float(
                center_distances[upper].mean()
            ),
            "minimum_progress_gap": float(
                np.diff(
                    np.concatenate(
                        [
                            np.zeros((centers.shape[0], 1)),
                            centers,
                        ],
                        axis=1,
                    ),
                    axis=1,
                ).min()
            ),
            "action_path_distance_correlation": (
                matrix_distance_correlation(
                    action_distances,
                    path_distances,
                )
            ),
        },
        action_distances,
    )


def action_path_permutation_null(
    action_distances: np.ndarray,
    path_distances: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    observed_by_question = np.asarray(
        [
            matrix_distance_correlation(action, path)
            for action, path in zip(action_distances, path_distances)
        ],
        dtype=np.float64,
    )
    observed = float(observed_by_question.mean())
    null = np.empty(int(permutations), dtype=np.float64)
    group_size = action_distances.shape[1]
    for draw in range(int(permutations)):
        values = []
        for action, path in zip(action_distances, path_distances):
            permutation = rng.permutation(group_size)
            shuffled_path = path[np.ix_(permutation, permutation)]
            values.append(
                matrix_distance_correlation(action, shuffled_path)
            )
        null[draw] = float(np.mean(values))
    return {
        "observed_mean": observed,
        "observed_question_values": observed_by_question.tolist(),
        "null_mean": float(null.mean()),
        "excess": float(observed - null.mean()),
        "p_value": float(
            (1 + np.sum(null >= observed)) / (len(null) + 1)
        ),
        "null_ci95": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "null_values": null.tolist(),
    }


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
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 3:
        raise ValueError("features must have shape [question, path, feature]")
    if labels.shape != features.shape[:2]:
        raise ValueError("labels must align with question and path axes")
    n_questions, group_size, feature_size = features.shape
    folds = int(folds)
    if folds < 2 or folds > n_questions:
        raise ValueError(
            f"folds must be in [2, {n_questions}], received {folds}"
        )
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("probe labels must be binary")
    if np.unique(labels).size != 2:
        raise ValueError("probe requires both correct and wrong paths")
    scores = np.zeros((n_questions, group_size), dtype=np.float64)
    for fold in range(folds):
        test_questions = np.arange(n_questions) % folds == fold
        train_questions = ~test_questions
        train_x = features[train_questions].reshape(-1, feature_size)
        train_y = labels[train_questions].reshape(-1)
        test_x = features[test_questions].reshape(-1, feature_size)
        if test_x.size == 0:
            raise ValueError(f"fold {fold} has no held-out questions")
        if np.unique(train_y).size != 2:
            raise ValueError(
                f"fold {fold} training partition lacks one outcome class"
            )
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
    if not values:
        raise ValueError(
            "question bootstrap produced no finite AUROC estimates"
        )
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


def plot_corridor_assignments(record: dict, output_base: Path):
    assignments = [
        assignment.float().numpy()
        for assignment in record["corridor_assignments"]
    ]
    cmap = LinearSegmentedColormap.from_list(
        "assignment",
        ["#F7F7F8", PINK, PINK_DARK],
    )
    columns = min(4, len(assignments))
    rows = int(math.ceil(len(assignments) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.15 * columns, 2.0 * rows),
        layout="constrained",
        squeeze=False,
    )
    for view, (ax, assignment) in enumerate(
        zip(axes.flat, assignments)
    ):
        image = ax.imshow(
            assignment,
            cmap=cmap,
            vmin=0.0,
            vmax=max(0.25, float(assignment.max())),
            aspect="auto",
        )
        ax.set_title(
            f"IID path {view + 1}",
            loc="left",
            fontsize=7,
            fontweight="bold",
        )
        ax.set_yticks(range(0, assignment.shape[0], 2))
        ax.set_xticks(range(assignment.shape[1]))
        if view < (rows - 1) * columns:
            ax.set_xticklabels([])
        if view % columns:
            ax.set_yticklabels([])
    for ax in list(axes.flat)[len(assignments) :]:
        ax.set_visible(False)
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
        label="Assignment weight",
    )
    fig.suptitle(
        f"Diagnostic only: legacy CoT corridor | q{record['idx']}",
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
        "Diagnostic only: outcome-path separation",
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
    # COMMIT's zero log-std is a shape-compatible placeholder, not log(1).
    # Its effective policy standard deviation is exactly zero because
    # realize_action returns the conditional mean without sampling.
    policy_std[-1] = 0.0
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
    ax.set_xlabel("Role-conditioned latent transition")
    ax.set_ylabel("Action standard deviation", color=PINK_DARK)
    ax.tick_params(axis="y", colors=PINK_DARK)
    ax.set_xticks(steps, ("P", "S1", "S2", "S3", "S4", "S5", "R", "D"))
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
        "Stochastic roles explore; COMMIT (D) is deterministic",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    handles = ax.get_lines() + second.get_lines()
    ax.legend(handles, [line.get_label() for line in handles], loc="best")
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_base)


def plot_role_evidence(
    rows: Sequence[dict],
    role_metrics: Dict[str, object],
    output_base: Path,
):
    """Primary semantic/exploration evidence plus deterministic audit."""
    roles = ("PLAN", "SOLVE", "REFINE")
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.45))
    positions = np.arange(3)
    semantics = role_metrics["semantics"]
    map_mean = np.asarray(
        [semantics[role]["map_teacher_similarity"]["mean"] for role in roles]
    )
    rollout_mean = np.asarray(
        [
            semantics[role]["rollout_teacher_similarity"]["mean"]
            for role in roles
        ]
    )
    map_error = np.asarray(
        [
            [
                semantics[role]["map_teacher_similarity"]["mean"]
                - semantics[role]["map_teacher_similarity"]["ci95"][0],
                semantics[role]["map_teacher_similarity"]["ci95"][1]
                - semantics[role]["map_teacher_similarity"]["mean"],
            ]
            for role in roles
        ]
    ).T
    rollout_error = np.asarray(
        [
            [
                semantics[role]["rollout_teacher_similarity"]["mean"]
                - semantics[role]["rollout_teacher_similarity"]["ci95"][0],
                semantics[role]["rollout_teacher_similarity"]["ci95"][1]
                - semantics[role]["rollout_teacher_similarity"]["mean"],
            ]
            for role in roles
        ]
    ).T
    axes[0].errorbar(
        positions - 0.06,
        map_mean,
        yerr=map_error,
        marker="o",
        color=BLUE,
        linewidth=1.5,
        capsize=2,
        label="MAP",
    )
    axes[0].errorbar(
        positions + 0.06,
        rollout_mean,
        yerr=rollout_error,
        marker="s",
        color=PINK_DARK,
        linewidth=1.5,
        capsize=2,
        label="8-path mean",
    )
    axes[0].axhline(0.0, color=GRID, linewidth=0.8)
    axes[0].set_xticks(positions, roles)
    axes[0].set_ylabel("Teacher cosine similarity")
    axes[0].set_title("Role semantics", loc="left", fontweight="bold")
    axes[0].legend(loc="best")

    exploration = role_metrics["exploration"]
    role_std = [exploration[role]["action_std"]["mean"] for role in roles]
    role_entropy = [
        exploration[role]["gaussian_entropy_per_dimension"]["mean"]
        for role in roles
    ]
    axes[1].plot(
        positions,
        role_std,
        color=GREEN,
        marker="o",
        linewidth=1.6,
        label="Std",
    )
    entropy_axis = axes[1].twinx()
    entropy_axis.plot(
        positions,
        role_entropy,
        color=ORANGE,
        marker="s",
        linewidth=1.4,
        label="Entropy / dim",
    )
    axes[1].set_xticks(positions, roles)
    axes[1].set_ylabel("Action std", color=GREEN)
    axes[1].tick_params(axis="y", colors=GREEN)
    entropy_axis.set_ylabel("Gaussian entropy", color=ORANGE)
    entropy_axis.tick_params(axis="y", colors=ORANGE)
    axes[1].set_title("Role-conditioned exploration", loc="left", fontweight="bold")
    handles = axes[1].get_lines() + entropy_axis.get_lines()
    axes[1].legend(handles, [line.get_label() for line in handles], loc="best")

    commit = role_metrics["commit"]
    audit_values = np.asarray(
        [
            commit["max_action_minus_mean_abs"],
            commit["max_role_reward_abs"],
            commit["max_log_std_placeholder_abs"],
        ],
        dtype=np.float64,
    )
    display_values = np.maximum(audit_values, 1e-8)
    axes[2].bar(
        np.arange(3),
        display_values,
        color=(BLUE, PINK, GREEN),
        width=0.68,
    )
    axes[2].axhline(
        commit["audit_tolerance"],
        color=ORANGE,
        linewidth=1.2,
        linestyle="--",
        label="Audit tolerance",
    )
    axes[2].set_yscale("log")
    axes[2].set_xticks(
        np.arange(3),
        ("|a-μ|", "|reward|", "|log-std|"),
    )
    axes[2].set_ylabel("Maximum absolute value")
    axes[2].set_title("Deterministic COMMIT audit", loc="left", fontweight="bold")
    axes[2].text(
        0.98,
        0.96,
        f"{commit['passing_questions']}/{commit['n_questions']} pass\n"
        f"readout={commit['commit_only_readout_rate']:.0%}",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        color=INK,
        fontweight="bold",
    )
    axes[2].legend(loc="lower right")
    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.16,
            1.06,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.75)
    fig.tight_layout(pad=0.55, w_pad=0.8)
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


def plot_formation_evidence(
    progress_centers: np.ndarray,
    action_distances: np.ndarray,
    path_distances: np.ndarray,
    coupling_null: Dict[str, object],
    output_base: Path,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.08, 2.25),
        gridspec_kw={"width_ratios": [1.15, 1.2, 0.9]},
    )
    steps = np.arange(1, progress_centers.shape[-1] + 1)
    flattened_centers = progress_centers.reshape(
        -1,
        progress_centers.shape[-1],
    )
    lower, median, upper = np.quantile(
        flattened_centers,
        [0.05, 0.5, 0.95],
        axis=0,
    )
    axes[0].fill_between(
        steps,
        lower,
        upper,
        color=PINK,
        alpha=0.25,
        label="5--95% across paths",
    )
    axes[0].plot(
        steps,
        median,
        color=PINK_DARK,
        linewidth=1.8,
        label="Median",
    )
    sample_colors = (BLUE, GREEN, ORANGE, PINK_DARK)
    for index, center in enumerate(progress_centers[0]):
        axes[0].plot(
            steps,
            center,
            color=sample_colors[index % len(sample_colors)],
            linewidth=0.75,
            alpha=0.75,
        )
    axes[0].set(
        xlabel="Latent transition",
        ylabel="Normalized CoT progress",
        xticks=steps,
        ylim=(0.0, 1.03),
    )
    axes[0].set_title(
        "Diagnostic only: ordered corridors",
        loc="left",
        fontweight="bold",
    )
    axes[0].legend(loc="upper left")

    upper_index = np.triu_indices(action_distances.shape[1], k=1)
    action_pairs = action_distances[:, upper_index[0], upper_index[1]].reshape(
        -1
    )
    path_pairs = path_distances[:, upper_index[0], upper_index[1]].reshape(-1)
    axes[1].scatter(
        action_pairs,
        path_pairs,
        s=6,
        color=PINK,
        alpha=0.18,
        edgecolors="none",
        rasterized=True,
    )
    if np.std(action_pairs) > 1e-12:
        slope, intercept = np.polyfit(action_pairs, path_pairs, deg=1)
        x_line = np.linspace(action_pairs.min(), action_pairs.max(), 100)
        axes[1].plot(
            x_line,
            intercept + slope * x_line,
            color=GREEN,
            linewidth=1.8,
        )
    axes[1].set(
        xlabel="Pairwise action distance",
        ylabel="Pairwise realized-path distance",
    )
    axes[1].set_title(
        "Diagnostic only: action--path coupling",
        loc="left",
        fontweight="bold",
    )

    observed_values = np.asarray(
        coupling_null["observed_question_values"],
        dtype=np.float64,
    )
    null_values = np.asarray(
        coupling_null["null_values"],
        dtype=np.float64,
    )
    axes[2].hist(
        null_values,
        bins=24,
        density=True,
        color=BLUE,
        alpha=0.35,
        label="Within-question null",
    )
    axes[2].axvline(
        coupling_null["observed_mean"],
        color=PINK_DARK,
        linewidth=2.0,
        label="Observed mean",
    )
    axes[2].axvspan(
        np.quantile(observed_values, 0.025),
        np.quantile(observed_values, 0.975),
        color=PINK,
        alpha=0.17,
    )
    axes[2].set(
        xlabel="Action--path distance correlation",
        ylabel="Density",
    )
    axes[2].set_title(
        "Diagnostic only: shuffled pairing",
        loc="left",
        fontweight="bold",
    )
    axes[2].legend(loc="upper left")
    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.15,
            1.06,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        ax.grid(color=GRID, linewidth=0.5, alpha=0.75)
    fig.tight_layout(pad=0.55, w_pad=0.9)
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
    legacy_corridor_available = all(
        "corridor_progress_centers" in record
        and "corridor_assignments" in record
        for record in records
    )
    progress_centers = (
        np.stack(
            [
                record["corridor_progress_centers"].float().numpy()
                for record in records
            ]
        )
        if legacy_corridor_available
        else None
    )
    action_tensors = np.stack(
        [record["rollout_actions"].float().numpy() for record in records]
    )
    action_mean_tensors = np.stack(
        [
            record["rollout_action_means"].float().numpy()
            for record in records
        ]
    )
    action_log_std_tensors = np.stack(
        [
            record["rollout_action_log_stds"].float().numpy()
            for record in records
        ]
    )
    map_role_rewards = np.stack(
        [record["map_role_rewards"].float().numpy() for record in records]
    )
    rollout_role_rewards = np.stack(
        [
            record["rollout_role_rewards"].float().numpy()
            for record in records
        ]
    )
    role_solve_masks = np.stack(
        [
            record["role_teacher_solve_mask"].bool().numpy()
            for record in records
        ]
    )
    rows = []
    action_distance_matrices = []
    for record_index, (record, actions, distance, outcome) in enumerate(zip(
        records,
        action_tensors,
        distance_matrices,
        labels,
    )):
        action_distances = pairwise_euclidean_matrix(
            actions.reshape(actions.shape[0], -1)
        )
        formation = {
            "progress_schedule_diversity": float("nan"),
            "progress_pair_distance": float("nan"),
            "minimum_progress_gap": float("nan"),
            "action_path_distance_correlation": (
                matrix_distance_correlation(action_distances, distance)
            ),
        }
        if legacy_corridor_available:
            formation, action_distances = formation_geometry(
                actions,
                progress_centers[record_index],
                distance,
            )
        rows.append(
            {
                "idx": int(record["idx"]),
                **role_question_metrics(record),
                **question_geometry(distance, outcome),
                **formation,
            }
        )
        action_distance_matrices.append(action_distances)
    action_distance_matrices = np.stack(action_distance_matrices)
    write_rows(rows, args.output_dir / "question_geometry.csv")

    rng = np.random.default_rng(20260719)
    eligible_rows = [row for row in rows if row["eligible"]]
    role_metrics = summarize_role_metrics(
        rows,
        bootstrap=args.bootstrap,
        rng=rng,
    )
    metrics = {
        "evidence_priority": {
            "primary": (
                "strict COMMIT bottleneck and stochastic action-path coupling"
            ),
            "supporting": (
                "role-conditioned Gaussian exploration and deterministic "
                "COMMIT audit"
            ),
            "retention_diagnostic": (
                "PLAN/SOLVE/REFINE frozen-text-CoT semantic similarity"
            ),
            "diagnostic_only": "legacy corridor and display PCA geometry",
            "sampling_unit": "question",
            "questions": len(records),
            "paths_per_question": 8,
            "bootstrap_replicates": int(args.bootstrap),
            "permutation_replicates": int(args.permutations),
        },
        "role_semantics": role_metrics["semantics"],
        "role_exploration": role_metrics["exploration"],
        "commit_determinism": role_metrics["commit"],
    }
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
    for key in (
        "progress_schedule_diversity",
        "progress_pair_distance",
        "minimum_progress_gap",
        "action_path_distance_correlation",
    ):
        values = [row[key] for row in rows if np.isfinite(row[key])]
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
    metrics["action_path_permutation_null"] = action_path_permutation_null(
        action_distance_matrices,
        distance_matrices,
        permutations=args.permutations,
        rng=rng,
    )
    metrics["commit_reward_exact_zero"] = bool(
        np.abs(map_role_rewards[:, -1]).max() <= COMMIT_AUDIT_TOLERANCE
        and np.abs(rollout_role_rewards[..., -1]).max()
        <= COMMIT_AUDIT_TOLERANCE
    )
    metrics["corridor_diagnostics"] = {
        "evidence_status": (
            "diagnostic_only" if legacy_corridor_available else "not_recorded"
        ),
        "available_in_records": legacy_corridor_available,
        "reason": (
            "Legacy corridor modules are not part of the trained role-aware "
            "mechanism. Old fields are accepted only for backward-compatible "
            "inspection and are never required by the formal evaluation."
        ),
        "reported_metrics": [
            "progress_schedule_diversity",
            "progress_pair_distance",
            "minimum_progress_gap",
            "action_path_distance_correlation",
            "action_path_permutation_null",
        ],
    }
    metrics["outcome_geometry_diagnostics"] = {
        "evidence_status": "diagnostic_only",
        "reason": (
            "Answer labels do not supervise or identify individual latent "
            "roles; these metrics are retained as secondary diagnostics."
        ),
    }

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

    plot_role_evidence(
        rows,
        role_metrics,
        args.output_dir / "role_mechanism_evidence",
    )
    plot_step_profile(
        records,
        residuals,
        args.output_dir / "policy_step_profile",
    )
    plot_geometry_separation(
        rows,
        args.output_dir / "diagnostic_outcome_geometry",
    )
    plot_probe_roc(
        labels,
        probe["scores"],
        probe["auroc"],
        args.output_dir / "outcome_probe_roc",
    )
    if legacy_corridor_available:
        plot_formation_evidence(
            progress_centers,
            action_distance_matrices,
            distance_matrices,
            metrics["action_path_permutation_null"],
            args.output_dir / "diagnostic_legacy_corridor_coupling",
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
        if legacy_corridor_available:
            plot_corridor_assignments(
                record,
                args.output_dir
                / f"diagnostic_legacy_corridor_q{record['idx']}",
            )

    np.savez_compressed(
        args.output_dir / "source_data.npz",
        labels=labels,
        projected_residuals=projected.numpy(),
        projected_map_residuals=projected_map.numpy(),
        distance_matrices=distance_matrices,
        action_distance_matrices=action_distance_matrices,
        legacy_progress_centers=(
            progress_centers
            if legacy_corridor_available
            else np.empty((0, 8, 8), dtype=np.float32)
        ),
        role_solve_masks=role_solve_masks,
        map_role_rewards=map_role_rewards,
        rollout_role_rewards=rollout_role_rewards,
        rollout_action_means=action_mean_tensors,
        rollout_action_log_stds=action_log_std_tensors,
        commit_action_mean_abs=(
            np.abs(action_tensors[:, :, 7] - action_mean_tensors[:, :, 7])
        ),
        probe_scores=probe["scores"],
    )
    contract = {
        "core_conclusion": (
            "PLAN, masked SOLVE, and REFINE retain their Stage-1 semantic "
            "formation; their role-conditioned policies produce measurable "
            "action-path coupling; COMMIT is conditionally deterministic and "
            "is the answer generator's only latent readout."
        ),
        "primary_evidence": (
            "numerical COMMIT audit and question-level action-path coupling; "
            "PLAN/SOLVE/REFINE cosine is reported only as retention evidence"
        ),
        "sampling": "200 questions x eight IID role-conditioned paths",
        "uncertainty": (
            f"{args.bootstrap} question-bootstrap replicates; "
            f"{args.permutations} within-question permutations"
        ),
        "commit_audit": role_metrics["commit"],
        "legacy_corridor_evidence_status": (
            "diagnostic_only" if legacy_corridor_available else "not_recorded"
        ),
        "outcome_geometry_evidence_status": "diagnostic_only",
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
            "Teacher cosine similarity is semantic alignment rather than "
            "answer correctness. Legacy corridors and 3D panels are "
            "diagnostic only and cannot override role-level evidence."
        ),
    }
    (args.output_dir / "figure_contract.json").write_text(
        json.dumps(contract, indent=2)
    )
    (args.output_dir / "geometry_summary.json").write_text(
        json.dumps(metrics, indent=2)
    )
    markdown = [
        "# TRACE 200-Question Role-Aware Evidence Summary",
        "",
        "## Primary mechanism evidence",
        "",
    ]
    for role in ("PLAN", "SOLVE", "REFINE"):
        semantic = metrics["role_semantics"][role]
        exploration = metrics["role_exploration"][role]
        map_value = semantic["map_teacher_similarity"]
        rollout_value = semantic["rollout_teacher_similarity"]
        markdown.extend(
            [
                f"- **{role} semantic similarity:** MAP "
                f"{map_value['mean']:.4f} "
                f"(95% CI {map_value['ci95'][0]:.4f}, "
                f"{map_value['ci95'][1]:.4f}); eight-path mean "
                f"{rollout_value['mean']:.4f} "
                f"(95% CI {rollout_value['ci95'][0]:.4f}, "
                f"{rollout_value['ci95'][1]:.4f}).",
                f"  Role exploration: std "
                f"{exploration['action_std']['mean']:.4f}; Gaussian entropy "
                f"{exploration['gaussian_entropy_per_dimension']['mean']:.4f} "
                "per action dimension.",
            ]
        )
    commit = metrics["commit_determinism"]
    markdown.extend(
        [
            f"- **COMMIT deterministic audit:** "
            f"{commit['passing_questions']}/{commit['n_questions']} questions "
            f"pass at tolerance {commit['audit_tolerance']:.1e}; max "
            f"|action-mean|={commit['max_action_minus_mean_abs']:.3e}, max "
            f"|role reward|={commit['max_role_reward_abs']:.3e}, "
            f"COMMIT-only readout={commit['commit_only_readout_rate']:.0%}.",
            "",
            "Teacher cosine similarity measures alignment to externally "
            "available textual-CoT targets; it is not itself answer accuracy.",
            "",
            "## Diagnostic-only legacy evidence",
            "",
            f"- Eligible correct/correct/wrong questions: "
            f"**{metrics['eligible_questions']} / 200**.",
            f"- Outcome margin: **{metrics['outcome_margin']['mean']:+.4f}** "
            f"(95% CI {metrics['outcome_margin']['ci95'][0]:+.4f}, "
            f"{metrics['outcome_margin']['ci95'][1]:+.4f}); label-null "
            f"`p={metrics['permutation_null']['p_value']:.4g}`.",
            f"- Question-held-out path probe AUROC: **{probe['auroc']:.3f}** "
            f"(95% CI {probe_ci[0]:.3f}, {probe_ci[1]:.3f}).",
            f"- Legacy action--corridor coupling: "
            f"r={metrics['action_path_distance_correlation']['mean']:.3f}; "
            f"pairing-null `p="
            f"{metrics['action_path_permutation_null']['p_value']:.4g}`.",
            "",
            f"All estimates use 200 questions x eight paths, "
            f"{args.bootstrap} question-bootstrap replicates, and "
            f"{args.permutations} within-question permutations. PCA was fit "
            "on separate training questions from all supplied stages without "
            "labels. No path offset or per-path rescaling was applied.",
        ]
    )
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
