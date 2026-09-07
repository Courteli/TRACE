#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.trace_exchangeable import trace_path_distance, trace_path_distance_components


DISTANCE_KWARGS = {
    "anchor_count": 3,
    "position_weight": 0.45,
    "direction_weight": 0.35,
    "step_weight": 0.15,
}


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    return torch.as_tensor(value, dtype=torch.float32)


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float()
    right = right.float()
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    similarity = torch.nn.functional.cosine_similarity(left, right, dim=-1, eps=1e-8)
    both_zero = (left_norm <= 1e-8) & (right_norm <= 1e-8)
    exactly_one_zero = (left_norm <= 1e-8) ^ (right_norm <= 1e-8)
    similarity = torch.where(both_zero, torch.ones_like(similarity), similarity)
    similarity = torch.where(exactly_one_zero, torch.zeros_like(similarity), similarity)
    return torch.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)


def _assignment_structure(assignment: torch.Tensor) -> dict:
    assignment = assignment.float()
    assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    step_position = torch.linspace(0.0, 1.0, assignment.shape[-1])
    centers = (assignment * step_position).sum(dim=-1)
    differences = centers.diff()
    return {
        "progress_span": float((centers.max() - centers.min()).item()),
        "progress_center_std": float(centers.std(unbiased=False).item()),
        "position_order_consistency": (
            float((differences >= -1e-4).float().mean().item())
            if differences.numel()
            else 1.0
        ),
    }


def _structure_metrics(
    record: dict,
    paths: torch.Tensor,
    teachers: torch.Tensor,
    *,
    permutations: int,
    seed: int,
) -> dict:
    if "rationale_teacher_assignments" in record:
        assignments = [
            _as_tensor(assignment)
            for assignment in record["rationale_teacher_assignments"]
        ]
    elif "multiview_teacher_assignments" in record:
        assignments = _as_tensor(record["multiview_teacher_assignments"])
    else:
        assignments = _as_tensor(record["assignment"]).unsqueeze(0)
    assignment_rows = [_assignment_structure(assignment) for assignment in assignments]

    step_alignment = float(_cosine_similarity(paths, teachers).mean().item())
    normalized_paths = torch.nn.functional.normalize(paths.float(), dim=-1)
    normalized_teachers = torch.nn.functional.normalize(teachers.float(), dim=-1)
    step_pairwise = torch.einsum(
        "vth,vsh->vts",
        normalized_paths,
        normalized_teachers,
    )
    final_path_alignment = float(
        _cosine_similarity(paths.sum(dim=-2), teachers.sum(dim=-2)).mean().item()
    )

    rng = np.random.default_rng(int(seed))
    order_null = []
    step_null = []
    n_steps = int(paths.shape[-2])
    for _ in range(int(permutations)):
        per_view_order = []
        for assignment in assignments:
            permutation = torch.as_tensor(
                rng.permutation(assignment.shape[0]),
                dtype=torch.long,
            )
            permuted = assignment.index_select(0, permutation)
            per_view_order.append(
                _assignment_structure(permuted)["position_order_consistency"]
            )
        order_null.append(float(np.mean(per_view_order)))

        transition_permutation = torch.as_tensor(
            rng.permutation(n_steps),
            dtype=torch.long,
        )
        step_null.append(
            float(
                step_pairwise[
                    :,
                    torch.arange(n_steps),
                    transition_permutation,
                ].mean().item()
            )
        )

    progress_span = float(np.mean([row["progress_span"] for row in assignment_rows]))
    progress_center_std = float(
        np.mean([row["progress_center_std"] for row in assignment_rows])
    )
    position_order = float(
        np.mean([row["position_order_consistency"] for row in assignment_rows])
    )
    position_order_null = float(np.mean(order_null)) if order_null else None
    step_alignment_null = float(np.mean(step_null)) if step_null else None
    return {
        "assignment_progress_span": progress_span,
        "assignment_progress_center_std": progress_center_std,
        "position_order_consistency": position_order,
        "position_order_permutation_null": position_order_null,
        "position_order_excess_over_null": (
            None if position_order_null is None else position_order - position_order_null
        ),
        "step_alignment_cos": step_alignment,
        "step_alignment_permutation_null": step_alignment_null,
        "step_alignment_excess_over_null": (
            None if step_alignment_null is None else step_alignment - step_alignment_null
        ),
        "final_path_alignment_cos": final_path_alignment,
    }


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    trials: int,
    seed: int,
) -> Dict[str, Optional[float]]:
    array = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    if array.size == 1:
        value = float(array[0])
        return {"n": 1, "mean": value, "ci95_low": value, "ci95_high": value}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(int(trials), array.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _sign_flip_pvalue(
    values: Iterable[float],
    *,
    trials: int,
    seed: int,
) -> Optional[float]:
    array = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if array.size == 0:
        return None
    observed = float(array.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(int(trials), array.size))
    null = (signs * array.reshape(1, -1)).mean(axis=1)
    return float((1 + np.count_nonzero(null >= observed)) / (int(trials) + 1))


def _holm_adjust(pvalues: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    family_size = len(ordered)
    for rank, label in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * float(pvalues[label]))
        running = max(running, candidate)
        adjusted[label] = running
    return adjusted


def _pairwise_distance(paths: torch.Tensor) -> torch.Tensor:
    return trace_path_distance(
        paths[:, None, :, :],
        paths[None, :, :, :],
        **DISTANCE_KWARGS,
    )


def _local_outcome_metrics(
    distance: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    margin: float,
) -> Optional[dict]:
    outcomes = outcomes.bool().view(-1)
    correct = torch.nonzero(outcomes, as_tuple=False).flatten()
    wrong = torch.nonzero(~outcomes, as_tuple=False).flatten()
    if correct.numel() < 2 or wrong.numel() < 1:
        return None

    correct_pairwise = distance.index_select(0, correct).index_select(1, correct).clone()
    correct_pairwise.fill_diagonal_(float("inf"))
    correct_wrong = distance.index_select(0, correct).index_select(1, wrong)
    nearest_correct = correct_wrong.argmin(dim=0)
    radii = []
    wrong_distances = []
    hinges = []
    for wrong_rank, correct_rank_tensor in enumerate(nearest_correct):
        correct_rank = int(correct_rank_tensor.item())
        peer_rank = int(correct_pairwise[correct_rank].argmin().item())
        radius = correct_pairwise[correct_rank, peer_rank]
        wrong_distance = correct_wrong[correct_rank, wrong_rank]
        radii.append(radius)
        wrong_distances.append(wrong_distance)
        hinges.append(F_relu(float(margin) + radius - wrong_distance))

    radius = torch.stack(radii).mean()
    wrong_distance = torch.stack(wrong_distances).mean()
    return {
        "correct_local_radius": float(radius.item()),
        "wrong_to_local_correct_distance": float(wrong_distance.item()),
        "outcome_margin": float((wrong_distance - radius).item()),
        "ranking_hinge": float(torch.stack(hinges).mean().item()),
        "triplet_count": int(wrong.numel()),
    }


def F_relu(value: torch.Tensor) -> torch.Tensor:
    return torch.clamp(value, min=0.0)


def _off_diagonal_mean(matrix: torch.Tensor) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    return float(matrix[mask].mean().item())


def _relation_metrics(
    model_distance: torch.Tensor,
    teacher_distance: torch.Tensor,
) -> dict:
    if model_distance.shape[0] <= 1:
        return {
            "model_path_diversity": 0.0,
            "teacher_path_diversity": 0.0,
            "teacher_relation_mae": 0.0,
            "teacher_relation_correlation": None,
        }
    upper = torch.triu(
        torch.ones_like(model_distance, dtype=torch.bool),
        diagonal=1,
    )
    model = model_distance[upper].numpy()
    teacher = teacher_distance[upper].numpy()
    correlation = None
    if model.size > 1 and model.std() > 1e-8 and teacher.std() > 1e-8:
        correlation = float(np.corrcoef(model, teacher)[0, 1])
    return {
        "model_path_diversity": float(model.mean()),
        "teacher_path_diversity": float(teacher.mean()),
        "teacher_relation_mae": float(np.abs(model - teacher).mean()),
        "teacher_relation_correlation": correlation,
    }


def _record_paths_and_teachers(record: dict):
    if "multiview_implicit_residuals" in record:
        paths = _as_tensor(record["multiview_implicit_residuals"])
    else:
        paths = _as_tensor(record["implicit_residuals"]).unsqueeze(0)
    if "rationale_teacher_residuals" in record:
        teacher_set = _as_tensor(record["rationale_teacher_residuals"])
        model_to_teacher = trace_path_distance(
            paths[:, None, :, :],
            teacher_set[None, :, :, :],
            **DISTANCE_KWARGS,
        )
        nearest_teacher = model_to_teacher.argmin(dim=1)
        teachers = teacher_set.index_select(0, nearest_teacher)
    elif "multiview_teacher_residuals" in record:
        teachers = _as_tensor(record["multiview_teacher_residuals"])
    else:
        teacher = _as_tensor(record["aggregated_explicit_residuals"])
        teachers = teacher.unsqueeze(0).expand(paths.shape[0], -1, -1)
    if paths.shape != teachers.shape:
        raise ValueError(
            f"Model/teacher path shape mismatch: {tuple(paths.shape)} vs {tuple(teachers.shape)}"
        )
    return paths, teachers


def _rationale_set_metrics(record: dict, paths: torch.Tensor) -> dict:
    if "rationale_teacher_residuals" not in record:
        return {
            "n_verified_rationales": None,
            "set_model_to_teacher_distance": None,
            "set_teacher_coverage_distance": None,
            "set_teacher_path_diversity": None,
            "set_relation_mae": None,
            "set_relation_correlation": None,
        }
    teachers = _as_tensor(record["rationale_teacher_residuals"])
    distance = trace_path_distance(
        paths[:, None, :, :],
        teachers[None, :, :, :],
        **DISTANCE_KWARGS,
    )
    teacher_pairwise = _pairwise_distance(teachers)
    diversity = _off_diagonal_mean(teacher_pairwise)
    relation_mae = None
    relation_correlation = None
    if teachers.shape[0] > 1:
        nearest_model = distance.argmin(dim=0)
        matched_models = paths.index_select(0, nearest_model)
        model_pairwise = _pairwise_distance(matched_models)
        upper = torch.triu(
            torch.ones_like(teacher_pairwise, dtype=torch.bool),
            diagonal=1,
        )
        model_relations = model_pairwise[upper].numpy()
        teacher_relations = teacher_pairwise[upper].numpy()
        relation_mae = float(
            np.abs(model_relations - teacher_relations).mean()
        )
        if (
            model_relations.size > 1
            and model_relations.std() > 1e-8
            and teacher_relations.std() > 1e-8
        ):
            relation_correlation = float(
                np.corrcoef(model_relations, teacher_relations)[0, 1]
            )
    return {
        "n_verified_rationales": int(teachers.shape[0]),
        "set_model_to_teacher_distance": float(
            distance.min(dim=1).values.mean().item()
        ),
        "set_teacher_coverage_distance": float(
            distance.min(dim=0).values.mean().item()
        ),
        "set_teacher_path_diversity": diversity,
        "set_relation_mae": relation_mae,
        "set_relation_correlation": relation_correlation,
    }


def record_metrics(
    record: dict,
    *,
    label_permutations: int,
    ranking_margin: float,
    seed: int,
) -> dict:
    paths, teachers = _record_paths_and_teachers(record)
    model_distance = _pairwise_distance(paths)
    teacher_distance = _pairwise_distance(teachers)
    relation_metrics = _relation_metrics(
        model_distance,
        teacher_distance,
    )
    rationale_set_metrics = _rationale_set_metrics(record, paths)
    if rationale_set_metrics["n_verified_rationales"] is not None:
        relation_metrics["teacher_path_diversity"] = (
            rationale_set_metrics["set_teacher_path_diversity"]
        )
        relation_metrics["teacher_relation_mae"] = (
            rationale_set_metrics["set_relation_mae"]
        )
        relation_metrics["teacher_relation_correlation"] = (
            rationale_set_metrics["set_relation_correlation"]
        )
    alignment = trace_path_distance_components(
        paths,
        teachers,
        **DISTANCE_KWARGS,
    )
    row = {
        "idx": int(record.get("idx", -1)),
        "n_paths": int(paths.shape[0]),
        "path_alignment": float(alignment["total"].mean().item()),
        "position_alignment_distance": float(alignment["position"].mean().item()),
        "direction_alignment_distance": float(alignment["direction"].mean().item()),
        "step_alignment_distance": float(alignment["step"].mean().item()),
        **relation_metrics,
        **rationale_set_metrics,
        **_structure_metrics(
            record,
            paths,
            teachers,
            permutations=label_permutations,
            seed=int(seed) + 2000003 * (int(record.get("idx", -1)) + 1),
        ),
    }

    outcomes = None
    if "multiview_acc" in record:
        outcomes = _as_tensor(record["multiview_acc"]).view(-1) > 0.5
        if outcomes.numel() != paths.shape[0]:
            raise ValueError("multiview_acc does not match the number of paths")
        output_lengths = _as_tensor(
            record.get("multiview_output_lengths", torch.zeros_like(outcomes))
        ).view(-1)
        row.update(
            {
                "rollout_accuracy": float(outcomes.float().mean().item()),
                "output_length": float(output_lengths.float().mean().item()),
                "L": float(paths.shape[1] + output_lengths.float().mean().item()),
                "n_correct": int(outcomes.sum().item()),
                "n_wrong": int((~outcomes).sum().item()),
                "ranking_eligible": int(
                    outcomes.sum().item() >= 2 and (~outcomes).sum().item() >= 1
                ),
            }
        )
        local = _local_outcome_metrics(
            model_distance,
            outcomes,
            margin=ranking_margin,
        )
        if local is not None:
            row.update(local)
            question_rng = np.random.default_rng(
                int(seed) + 1000003 * (row["idx"] + 1)
            )
            null_margins = []
            for _ in range(int(label_permutations)):
                permutation = torch.as_tensor(
                    question_rng.permutation(paths.shape[0]),
                    dtype=torch.long,
                )
                permuted = _local_outcome_metrics(
                    model_distance,
                    outcomes.index_select(0, permutation),
                    margin=ranking_margin,
                )
                if permuted is not None:
                    null_margins.append(permuted["outcome_margin"])
            null_mean = float(np.mean(null_margins)) if null_margins else None
            row["outcome_margin_label_null"] = null_mean
            row["outcome_margin_excess_over_null"] = (
                None if null_mean is None else row["outcome_margin"] - null_mean
            )

            reverse = torch.arange(paths.shape[0] - 1, -1, -1)
            reversed_local = _local_outcome_metrics(
                model_distance.index_select(0, reverse).index_select(1, reverse),
                outcomes.index_select(0, reverse),
                margin=ranking_margin,
            )
            row["seed_order_permutation_error"] = max(
                abs(local[key] - reversed_local[key])
                for key in (
                    "correct_local_radius",
                    "wrong_to_local_correct_distance",
                    "outcome_margin",
                    "ranking_hinge",
                )
            )
    else:
        row.update(
            {
                "rollout_accuracy": None,
                "output_length": None,
                "L": None,
                "n_correct": None,
                "n_wrong": None,
                "ranking_eligible": 0,
            }
        )
    return row


def _seed_index_probe(records: List[dict]) -> dict:
    eligible = [
        record
        for record in records
        if "multiview_implicit_residuals" in record
    ]
    if not eligible:
        return {
            "n_questions": 0,
            "n_seed_positions": 0,
            "crossfit_accuracy": None,
            "chance_accuracy": None,
            "outcome_rate_std_by_position": None,
        }
    n_positions = int(_as_tensor(eligible[0]["multiview_implicit_residuals"]).shape[0])
    eligible = [
        record
        for record in eligible
        if _as_tensor(record["multiview_implicit_residuals"]).shape[0] == n_positions
    ]
    features = []
    question_ids = []
    outcome_rates = []
    for position, record in enumerate(eligible):
        paths = _as_tensor(record["multiview_implicit_residuals"]).flatten(1)
        paths = torch.nn.functional.normalize(paths, dim=-1)
        features.append(paths)
        question_ids.append(int(record.get("idx", position)))
        if "multiview_acc" in record:
            outcome_rates.append(_as_tensor(record["multiview_acc"]).view(-1))
    feature_tensor = torch.stack(features, dim=0)
    folds = torch.as_tensor(question_ids, dtype=torch.long).remainder(2)
    fold_accuracies = []
    targets = torch.arange(n_positions).view(1, -1)
    for fold in (0, 1):
        train = folds != fold
        test = folds == fold
        if not train.any() or not test.any():
            continue
        templates = torch.nn.functional.normalize(
            feature_tensor[train].mean(dim=0),
            dim=-1,
        )
        scores = torch.einsum(
            "nvd,wd->nvw",
            feature_tensor[test],
            templates,
        )
        predictions = scores.argmax(dim=-1)
        fold_accuracies.append(
            float((predictions == targets).float().mean().item())
        )
    outcome_rate_std = None
    if outcome_rates:
        outcome_rate_std = float(
            torch.stack(outcome_rates).float().mean(dim=0).std().item()
        )
    return {
        "n_questions": len(eligible),
        "n_seed_positions": n_positions,
        "crossfit_accuracy": (
            float(np.mean(fold_accuracies)) if fold_accuracies else None
        ),
        "chance_accuracy": 1.0 / n_positions,
        "outcome_rate_std_by_position": outcome_rate_std,
    }


def _aggregate_rows(rows: List[dict], bootstrap_trials: int, seed: int) -> dict:
    metric_names = [
        "rollout_accuracy",
        "L",
        "path_alignment",
        "position_alignment_distance",
        "direction_alignment_distance",
        "step_alignment_distance",
        "model_path_diversity",
        "teacher_path_diversity",
        "teacher_relation_mae",
        "teacher_relation_correlation",
        "n_verified_rationales",
        "set_model_to_teacher_distance",
        "set_teacher_coverage_distance",
        "set_teacher_path_diversity",
        "set_relation_mae",
        "set_relation_correlation",
        "correct_local_radius",
        "wrong_to_local_correct_distance",
        "outcome_margin",
        "ranking_hinge",
        "outcome_margin_excess_over_null",
        "seed_order_permutation_error",
        "assignment_progress_span",
        "assignment_progress_center_std",
        "position_order_consistency",
        "position_order_permutation_null",
        "position_order_excess_over_null",
        "step_alignment_cos",
        "step_alignment_permutation_null",
        "step_alignment_excess_over_null",
        "final_path_alignment_cos",
    ]
    summary = {
        metric: _bootstrap_mean_ci(
            [row.get(metric) for row in rows],
            trials=bootstrap_trials,
            seed=seed + metric_index,
        )
        for metric_index, metric in enumerate(metric_names)
    }
    summary["question_count"] = len(rows)
    summary["ranking_eligible_count"] = sum(
        int(row.get("ranking_eligible", 0)) for row in rows
    )
    summary["ranking_eligible_fraction"] = (
        summary["ranking_eligible_count"] / len(rows) if rows else None
    )
    summary["outcome_margin_null_signflip_p"] = _sign_flip_pvalue(
        [row.get("outcome_margin_excess_over_null") for row in rows],
        trials=bootstrap_trials,
        seed=seed + 7919,
    )
    return summary


def _paired_delta(
    baseline_rows: List[dict],
    target_rows: List[dict],
    *,
    bootstrap_trials: int,
    seed: int,
) -> dict:
    baseline = {row["idx"]: row for row in baseline_rows}
    target = {row["idx"]: row for row in target_rows}
    shared = sorted(set(baseline) & set(target))
    metrics = [
        "rollout_accuracy",
        "L",
        "path_alignment",
        "teacher_relation_mae",
        "set_model_to_teacher_distance",
        "set_teacher_coverage_distance",
        "set_relation_mae",
        "correct_local_radius",
        "wrong_to_local_correct_distance",
        "outcome_margin",
        "outcome_margin_excess_over_null",
        "assignment_progress_span",
        "position_order_excess_over_null",
        "step_alignment_excess_over_null",
        "final_path_alignment_cos",
    ]
    output = {"matched_questions": len(shared), "metrics": {}}
    for metric_index, metric in enumerate(metrics):
        deltas = []
        for index in shared:
            left = baseline[index].get(metric)
            right = target[index].get(metric)
            if left is not None and right is not None:
                deltas.append(float(right) - float(left))
        output["metrics"][metric] = _bootstrap_mean_ci(
            deltas,
            trials=bootstrap_trials,
            seed=seed + metric_index,
        )
    return output


def _factorial_interaction(
    rows_by_label: Dict[str, List[dict]],
    *,
    bootstrap_trials: int,
    seed: int,
) -> Optional[dict]:
    required = ("M00", "M01", "M10", "M11")
    if not all(label in rows_by_label for label in required):
        return None
    maps = {
        label: {row["idx"]: row for row in rows_by_label[label]}
        for label in required
    }
    shared = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    metrics = [
        "rollout_accuracy",
        "L",
        "path_alignment",
        "teacher_relation_mae",
        "correct_local_radius",
        "wrong_to_local_correct_distance",
        "outcome_margin",
        "outcome_margin_excess_over_null",
        "assignment_progress_span",
        "position_order_excess_over_null",
        "step_alignment_excess_over_null",
        "final_path_alignment_cos",
    ]
    output = {"matched_questions": len(shared), "definition": "(M11-M10)-(M01-M00)", "metrics": {}}
    for metric_index, metric in enumerate(metrics):
        interactions = []
        for index in shared:
            values = [maps[label][index].get(metric) for label in required]
            if all(value is not None for value in values):
                m00, m01, m10, m11 = map(float, values)
                interactions.append((m11 - m10) - (m01 - m00))
        output["metrics"][metric] = _bootstrap_mean_ci(
            interactions,
            trials=bootstrap_trials,
            seed=seed + metric_index,
        )
    return output


def _formation_retention(
    rows_by_label: Dict[str, List[dict]],
    *,
    bootstrap_trials: int,
    seed: int,
) -> Optional[dict]:
    required = ("S1P", "S1F", "M00", "M01", "M10", "M11")
    if not all(label in rows_by_label for label in required):
        return None
    maps = {
        label: {row["idx"]: row for row in rows_by_label[label]}
        for label in required
    }
    shared = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    metrics = (
        "assignment_progress_span",
        "position_order_excess_over_null",
        "step_alignment_excess_over_null",
    )
    rng = np.random.default_rng(seed)
    output = {
        "matched_questions": len(shared),
        "definition": {
            "stage1_effect": "S1F-S1P",
            "answer_only_formation_effect": "M10-M00",
            "ranked_formation_effect": "M11-M01",
            "ranking_retention_ratio": "(M11-M01)/(M10-M00)",
        },
        "metrics": {},
    }
    for metric in metrics:
        valid = [
            index
            for index in shared
            if all(maps[label][index].get(metric) is not None for label in required)
        ]
        stage1 = np.asarray(
            [maps["S1F"][index][metric] - maps["S1P"][index][metric] for index in valid],
            dtype=np.float64,
        )
        answer = np.asarray(
            [maps["M10"][index][metric] - maps["M00"][index][metric] for index in valid],
            dtype=np.float64,
        )
        ranked = np.asarray(
            [maps["M11"][index][metric] - maps["M01"][index][metric] for index in valid],
            dtype=np.float64,
        )
        ratios = []
        if valid:
            remaining = int(bootstrap_trials)
            while remaining:
                chunk = min(1000, remaining)
                indices = rng.integers(0, len(valid), size=(chunk, len(valid)))
                numerator = ranked[indices].mean(axis=1)
                denominator = answer[indices].mean(axis=1)
                stable = np.abs(denominator) > 1e-8
                ratios.extend((numerator[stable] / denominator[stable]).tolist())
                remaining -= chunk
        point_ratio = (
            float(ranked.mean() / answer.mean())
            if valid and abs(float(answer.mean())) > 1e-8
            else None
        )
        if ratios:
            ratio_low, ratio_high = np.quantile(np.asarray(ratios), (0.025, 0.975))
            ratio_ci = (float(ratio_low), float(ratio_high))
        else:
            ratio_ci = (None, None)
        output["metrics"][metric] = {
            "n": len(valid),
            "stage1_formation_effect": _bootstrap_mean_ci(
                stage1,
                trials=bootstrap_trials,
                seed=seed + 11,
            ),
            "answer_only_formation_effect": _bootstrap_mean_ci(
                answer,
                trials=bootstrap_trials,
                seed=seed + 23,
            ),
            "ranked_formation_effect": _bootstrap_mean_ci(
                ranked,
                trials=bootstrap_trials,
                seed=seed + 37,
            ),
            "ranking_retention_ratio": point_ratio,
            "ranking_retention_ratio_ci95_low": ratio_ci[0],
            "ranking_retention_ratio_ci95_high": ratio_ci[1],
            "retains_at_least_95_percent_point_estimate": (
                point_ratio is not None and point_ratio >= 0.95
            ),
            "retains_at_least_95_percent_ci": (
                ratio_ci[0] is not None and ratio_ci[0] >= 0.95
            ),
        }
    return output


def _parse_record_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--record must be LABEL=/absolute/cache.pt")
    label, path = value.split("=", 1)
    return label, Path(path)


def _parse_comparison_arg(value: str):
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "--comparison must be BASELINE:TARGET"
        )
    baseline, target = value.split(":", 1)
    if not baseline or not target:
        raise argparse.ArgumentTypeError(
            "--comparison must name both BASELINE and TARGET"
        )
    return baseline, target


def _write_rows(path: Path, rows_by_label: Dict[str, List[dict]]):
    all_rows = []
    for label, rows in rows_by_label.items():
        all_rows.extend({"method": label, **row} for row in rows)
    if not all_rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in all_rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def _write_markdown(path: Path, payload: dict):
    lines = [
        "# TRACE Complete-Path Geometry Summary",
        "",
        "All geometry uses the training-time eight-transition `D_path`; no compressed signature, prototype, or per-path normalization is used.",
        "",
        "| Method | Questions | Eligible | Acc | #L | Local radius | Wrong distance | Outcome margin | Margin excess/null | Relation MAE | Seed-index probe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, method in payload["methods"].items():
        aggregate = method["aggregate"]

        def mean(name):
            value = aggregate[name]["mean"]
            return "NA" if value is None else f"{value:.4f}"

        probe = method["seed_index_probe"]["crossfit_accuracy"]
        probe_text = "NA" if probe is None else f"{probe:.3f}"
        lines.append(
            f"| {label} | {aggregate['question_count']} | "
            f"{aggregate['ranking_eligible_count']} | {mean('rollout_accuracy')} | "
            f"{mean('L')} | {mean('correct_local_radius')} | "
            f"{mean('wrong_to_local_correct_distance')} | {mean('outcome_margin')} | "
            f"{mean('outcome_margin_excess_over_null')} | {mean('teacher_relation_mae')} | "
            f"{probe_text} |"
        )
    lines.extend(
        [
            "",
            "## Multi-rationale set formation",
            "",
            "| Method | Verified rationales | Model-to-set distance | Teacher coverage distance | Teacher diversity | Relation MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, method in payload["methods"].items():
        aggregate = method["aggregate"]

        def set_mean(name):
            value = aggregate[name]["mean"]
            return "NA" if value is None else f"{value:.4f}"

        lines.append(
            f"| {label} | {set_mean('n_verified_rationales')} | "
            f"{set_mean('set_model_to_teacher_distance')} | "
            f"{set_mean('set_teacher_coverage_distance')} | "
            f"{set_mean('set_teacher_path_diversity')} | "
            f"{set_mean('set_relation_mae')} |"
        )
    if payload.get("factorial_interaction") is not None:
        lines.extend(
            [
                "",
                "## 2x2 Interaction",
                "",
                "`(M11-M10)-(M01-M00)` is reported in the JSON with question-bootstrap 95% confidence intervals.",
            ]
        )
    if payload.get("formation_retention") is not None:
        lines.extend(
            [
                "",
                "## Formation retention",
                "",
                "| Metric | Stage 1 effect | Answer-only effect | Ranked effect | Ranked/answer retention (95% CI) | 95% gate |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for metric, values in payload["formation_retention"]["metrics"].items():
            stage1 = values["stage1_formation_effect"]["mean"]
            answer = values["answer_only_formation_effect"]["mean"]
            ranked = values["ranked_formation_effect"]["mean"]
            ratio = values["ranking_retention_ratio"]
            low = values["ranking_retention_ratio_ci95_low"]
            high = values["ranking_retention_ratio_ci95_high"]
            ratio_text = (
                "NA"
                if ratio is None
                else f"{ratio:.3f} [{low:.3f}, {high:.3f}]"
            )
            lines.append(
                f"| {metric} | {stage1:+.4f} | {answer:+.4f} | {ranked:+.4f} | "
                f"{ratio_text} | "
                f"{'PASS' if values['retains_at_least_95_percent_ci'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Ordered trajectory structure",
            "",
            "| Method | Progress span | Position order | Order excess/null | Step alignment | Step excess/null | Final-path alignment |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, method in payload["methods"].items():
        aggregate = method["aggregate"]

        def structure_mean(name):
            value = aggregate[name]["mean"]
            return "NA" if value is None else f"{value:.4f}"

        lines.append(
            f"| {label} | {structure_mean('assignment_progress_span')} | "
            f"{structure_mean('position_order_consistency')} | "
            f"{structure_mean('position_order_excess_over_null')} | "
            f"{structure_mean('step_alignment_cos')} | "
            f"{structure_mean('step_alignment_excess_over_null')} | "
            f"{structure_mean('final_path_alignment_cos')} |"
        )
    if payload.get("familywise_outcome_null"):
        lines.extend(
            [
                "",
                "## Outcome-label null tests",
                "",
                "| Method | One-sided sign-flip p | Holm-adjusted p |",
                "| --- | ---: | ---: |",
            ]
        )
        for label, values in payload["familywise_outcome_null"].items():
            lines.append(
                f"| {label} | {values['raw_p']:.5f} | {values['holm_p']:.5f} |"
            )
    if payload.get("paired_comparisons"):
        lines.extend(
            [
                "",
                "## Paired mechanism comparisons",
                "",
                "| Comparison | Accuracy delta | Local-radius delta | Wrong-distance delta | Outcome-margin delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, comparison in payload["paired_comparisons"].items():
            metrics = comparison["metrics"]

            def paired_mean(name):
                value = metrics[name]["mean"]
                return "NA" if value is None else f"{value:+.4f}"

            lines.append(
                f"| {label} | {paired_mean('rollout_accuracy')} | "
                f"{paired_mean('correct_local_radius')} | "
                f"{paired_mean('wrong_to_local_correct_distance')} | "
                f"{paired_mean('outcome_margin')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", type=_parse_record_arg, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument("--label_permutations", type=int, default=128)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--ranking_margin", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline_label", default=None)
    parser.add_argument("--target_label", default=None)
    parser.add_argument(
        "--comparison",
        action="append",
        type=_parse_comparison_arg,
        default=[],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_label: Dict[str, List[dict]] = {}
    methods = {}
    for label, record_path in args.record:
        records = torch.load(
            record_path,
            map_location="cpu",
            weights_only=False,
        )[: args.max_records]
        rows = [
            record_metrics(
                record,
                label_permutations=args.label_permutations,
                ranking_margin=args.ranking_margin,
                seed=args.seed,
            )
            for record in records
        ]
        rows_by_label[label] = rows
        methods[label] = {
            "record_path": str(record_path),
            "aggregate": _aggregate_rows(
                rows,
                bootstrap_trials=args.bootstrap_trials,
                seed=args.seed,
            ),
            "seed_index_probe": _seed_index_probe(records),
        }

    raw_pvalues = {
        label: method["aggregate"]["outcome_margin_null_signflip_p"]
        for label, method in methods.items()
        if method["aggregate"]["outcome_margin_null_signflip_p"] is not None
    }
    holm_pvalues = _holm_adjust(raw_pvalues)
    familywise_outcome_null = {
        label: {
            "raw_p": raw_pvalues[label],
            "holm_p": holm_pvalues[label],
        }
        for label in raw_pvalues
    }

    paired = None
    if args.baseline_label is not None or args.target_label is not None:
        if args.baseline_label not in rows_by_label or args.target_label not in rows_by_label:
            raise ValueError("Both baseline_label and target_label must name supplied records")
        paired = _paired_delta(
            rows_by_label[args.baseline_label],
            rows_by_label[args.target_label],
            bootstrap_trials=args.bootstrap_trials,
            seed=args.seed,
        )
    comparisons = {}
    requested_comparisons = list(args.comparison)
    if (
        args.baseline_label is not None
        and args.target_label is not None
        and (args.baseline_label, args.target_label)
        not in requested_comparisons
    ):
        requested_comparisons.insert(
            0,
            (args.baseline_label, args.target_label),
        )
    for comparison_index, (baseline, target) in enumerate(
        requested_comparisons
    ):
        if baseline not in rows_by_label or target not in rows_by_label:
            raise ValueError(
                f"Comparison {baseline}:{target} does not name supplied "
                "records"
            )
        comparisons[f"{baseline}_to_{target}"] = _paired_delta(
            rows_by_label[baseline],
            rows_by_label[target],
            bootstrap_trials=args.bootstrap_trials,
            seed=args.seed + 1009 * comparison_index,
        )
    payload = {
        "metric_definition": {
            "path": "0.45 position + 0.35 direction + 0.15 symmetric-relative step length",
            "ranking_margin": args.ranking_margin,
            "label_permutations_per_question": args.label_permutations,
            "bootstrap_trials": args.bootstrap_trials,
            "no_signature": True,
            "no_prototype": True,
            "no_per_path_normalization": True,
        },
        "methods": methods,
        "paired_delta": paired,
        "paired_comparisons": comparisons,
        "factorial_interaction": _factorial_interaction(
            rows_by_label,
            bootstrap_trials=args.bootstrap_trials,
            seed=args.seed,
        ),
        "formation_retention": _formation_retention(
            rows_by_label,
            bootstrap_trials=args.bootstrap_trials,
            seed=args.seed + 50021,
        ),
        "familywise_outcome_null": familywise_outcome_null,
    }
    _write_rows(out_dir / "trace_exchangeable_geometry_rows.csv", rows_by_label)
    (out_dir / "trace_exchangeable_geometry_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_markdown(out_dir / "trace_exchangeable_geometry_summary.md", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
