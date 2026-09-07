#!/usr/bin/env python3
"""Question-held-out and OOD probes for complete TRACE path outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def parse_labeled_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected LABEL=/absolute/trace_final_visual_test.pt"
        )
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def parse_transfer(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Expected TRAIN_LABEL:EVAL_LABEL"
        )
    return tuple(value.split(":", 1))


def as_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_records(path: Path, max_records: int) -> List[dict]:
    records = torch.load(path, map_location="cpu", weights_only=False)
    records = [
        record
        for record in records[: int(max_records)]
        if "multiview_implicit_residuals" in record
        and "multiview_acc" in record
    ]
    if not records:
        raise ValueError(f"No multi-path outcome records in {path}")
    return records


def record_arrays(records: Sequence[dict]):
    paths = np.stack(
        [as_array(record["multiview_implicit_residuals"]) for record in records]
    )
    labels = np.stack(
        [as_array(record["multiview_acc"]).reshape(-1) for record in records]
    ).astype(np.int64)
    question_ids = np.asarray(
        [int(record.get("idx", index)) for index, record in enumerate(records)],
        dtype=np.int64,
    )
    if paths.shape[:2] != labels.shape:
        raise ValueError(
            f"Path/label mismatch: {paths.shape[:2]} versus {labels.shape}"
        )
    return paths.astype(np.float32), labels, question_ids


def randomized_pca(
    points: np.ndarray,
    *,
    n_components: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = points.mean(axis=0, keepdims=True).astype(np.float32)
    centered = (points - mean).astype(np.float32, copy=False)
    rank = min(
        int(n_components) + 12,
        centered.shape[0],
        centered.shape[1],
    )
    rng = np.random.default_rng(seed)
    omega = rng.normal(
        size=(centered.shape[1], rank)
    ).astype(np.float32)
    sketch = centered @ omega
    for _ in range(2):
        sketch = centered @ (centered.T @ sketch)
    q, _ = np.linalg.qr(sketch, mode="reduced")
    _, _, vh = np.linalg.svd(q.T @ centered, full_matrices=False)
    return mean, vh[: int(n_components)].T.astype(np.float32)


def path_features(
    paths: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    centered = paths - mean.reshape(1, 1, 1, -1)
    projected = centered @ components
    positions = np.cumsum(projected, axis=2)
    anchors = np.asarray([2, 5, 7], dtype=np.int64)
    anchors = np.clip(anchors, 0, paths.shape[2] - 1)
    anchor_positions = positions[:, :, anchors, :]
    step_norms = np.linalg.norm(paths, axis=-1) / np.sqrt(paths.shape[-1])
    return np.concatenate(
        [
            projected.reshape(paths.shape[0], paths.shape[1], -1),
            anchor_positions.reshape(paths.shape[0], paths.shape[1], -1),
            step_norms,
        ],
        axis=-1,
    ).astype(np.float64)


def fit_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float,
) -> dict:
    x = features.reshape(-1, features.shape[-1]).astype(np.float64)
    y = labels.reshape(-1).astype(np.int64)
    if np.unique(y).size != 2:
        raise ValueError("Probe training requires both outcomes")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (x - mean) / scale
    design = np.concatenate(
        [standardized, np.ones((standardized.shape[0], 1))],
        axis=1,
    )
    counts = np.bincount(y, minlength=2).astype(np.float64)
    sample_weight = 1.0 / counts[y].clip(min=1.0)
    sample_weight *= len(y) / sample_weight.sum()
    target = 2.0 * y.astype(np.float64) - 1.0
    weighted = design * sample_weight[:, None]
    gram = design.T @ weighted
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    regularizer[-1, -1] = 0.0
    rhs = design.T @ (sample_weight * target)
    weights = np.linalg.solve(gram + regularizer, rhs)
    return {
        "mean": mean,
        "scale": scale,
        "weights": weights,
    }


def predict_probe(probe: dict, features: np.ndarray) -> np.ndarray:
    x = features.reshape(-1, features.shape[-1]).astype(np.float64)
    standardized = (x - probe["mean"]) / probe["scale"]
    design = np.concatenate(
        [standardized, np.ones((standardized.shape[0], 1))],
        axis=1,
    )
    return design @ probe["weights"]


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = labels.reshape(-1).astype(np.int64)
    scores = scores.reshape(-1).astype(np.float64)
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return {
            "auroc": None,
            "auprc": None,
            "balanced_accuracy": None,
        }

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while (
            stop < len(order)
            and scores[order[stop]] == scores[order[start]]
        ):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    auroc = (
        ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)

    descending = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[descending]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    auprc = float(precision[sorted_labels == 1].mean())

    predictions = scores >= 0.0
    sensitivity = float(predictions[positives].mean())
    specificity = float((~predictions[negatives]).mean())
    return {
        "auroc": float(auroc),
        "auprc": auprc,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
    }


def question_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    trials: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    metrics = {
        "auroc": [],
        "auprc": [],
        "balanced_accuracy": [],
    }
    for _ in range(int(trials)):
        indices = rng.integers(0, labels.shape[0], size=labels.shape[0])
        sampled = binary_metrics(labels[indices], scores[indices])
        for key in metrics:
            if sampled[key] is not None:
                metrics[key].append(sampled[key])
    output = {}
    points = binary_metrics(labels, scores)
    for key, values in metrics.items():
        if not values:
            output[key] = {
                "value": points[key],
                "ci95_low": None,
                "ci95_high": None,
            }
            continue
        low, high = np.quantile(values, [0.025, 0.975])
        output[key] = {
            "value": points[key],
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return output


def prepare_folds(
    paths: np.ndarray,
    labels: np.ndarray,
    question_ids: np.ndarray,
    *,
    pca_components: int,
    folds: int,
    seed: int,
) -> List[dict]:
    fold_ids = question_ids % int(folds)
    prepared = []
    for fold in range(int(folds)):
        train_questions = fold_ids != fold
        test_questions = fold_ids == fold
        if not train_questions.any() or not test_questions.any():
            continue
        fit_points = paths[train_questions].reshape(-1, paths.shape[-1])
        mean, components = randomized_pca(
            fit_points,
            n_components=pca_components,
            seed=seed + fold,
        )
        prepared.append(
            {
                "fold": fold,
                "train_questions": train_questions,
                "test_questions": test_questions,
                "train_features": path_features(
                    paths[train_questions],
                    mean,
                    components,
                ),
                "test_features": path_features(
                    paths[test_questions],
                    mean,
                    components,
                ),
            }
        )
    return prepared


def crossfit_scores(
    prepared: Sequence[dict],
    labels: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    scores = np.full(labels.shape, np.nan, dtype=np.float64)
    for fold in prepared:
        train_labels = labels[fold["train_questions"]]
        if np.unique(train_labels).size != 2:
            continue
        probe = fit_linear_probe(
            fold["train_features"],
            train_labels,
            ridge=ridge,
        )
        fold_scores = predict_probe(
            probe,
            fold["test_features"],
        ).reshape(labels[fold["test_questions"]].shape)
        scores[fold["test_questions"]] = fold_scores
    if not np.isfinite(scores).all():
        raise ValueError("Cross-fitting did not score every question")
    return scores


def within_question_permutation_pvalue(
    prepared: Sequence[dict],
    labels: np.ndarray,
    observed_auroc: float,
    *,
    ridge: float,
    trials: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(trials)):
        permuted = np.stack(
            [rng.permutation(row) for row in labels],
            axis=0,
        )
        scores = crossfit_scores(prepared, permuted, ridge=ridge)
        value = binary_metrics(permuted, scores)["auroc"]
        if value is not None:
            null.append(value)
    pvalue = (
        (1 + sum(value >= observed_auroc for value in null))
        / (1 + len(null))
        if null
        else None
    )
    return {
        "trials": len(null),
        "mean": float(np.mean(null)) if null else None,
        "p_one_sided": pvalue,
    }


def crossfit_report(
    records: Sequence[dict],
    *,
    pca_components: int,
    folds: int,
    ridge: float,
    bootstrap_trials: int,
    permutation_trials: int,
    seed: int,
) -> dict:
    paths, labels, question_ids = record_arrays(records)
    prepared = prepare_folds(
        paths,
        labels,
        question_ids,
        pca_components=pca_components,
        folds=folds,
        seed=seed,
    )
    scores = crossfit_scores(prepared, labels, ridge=ridge)
    intervals = question_bootstrap(
        labels,
        scores,
        trials=bootstrap_trials,
        seed=seed + 1009,
    )
    observed = intervals["auroc"]["value"]
    null = within_question_permutation_pvalue(
        prepared,
        labels,
        observed,
        ridge=ridge,
        trials=permutation_trials,
        seed=seed + 2017,
    )
    return {
        "question_count": int(paths.shape[0]),
        "path_count": int(paths.shape[0] * paths.shape[1]),
        "correct_rate": float(labels.mean()),
        "question_held_out": True,
        "pca_fit_on_training_questions_only": True,
        "pca_components": int(pca_components),
        "metrics": intervals,
        "within_question_label_permutation_null": null,
    }


def fit_transfer_source(
    records: Sequence[dict],
    *,
    pca_components: int,
    ridge: float,
    seed: int,
) -> dict:
    paths, labels, _ = record_arrays(records)
    mean, components = randomized_pca(
        paths.reshape(-1, paths.shape[-1]),
        n_components=pca_components,
        seed=seed,
    )
    features = path_features(paths, mean, components)
    return {
        "mean": mean,
        "components": components,
        "probe": fit_linear_probe(features, labels, ridge=ridge),
    }


def transfer_report(
    source_records: Sequence[dict],
    target_records: Sequence[dict],
    *,
    pca_components: int,
    ridge: float,
    bootstrap_trials: int,
    seed: int,
) -> dict:
    fitted = fit_transfer_source(
        source_records,
        pca_components=pca_components,
        ridge=ridge,
        seed=seed,
    )
    paths, labels, _ = record_arrays(target_records)
    features = path_features(
        paths,
        fitted["mean"],
        fitted["components"],
    )
    scores = predict_probe(fitted["probe"], features).reshape(labels.shape)
    return {
        "question_count": int(paths.shape[0]),
        "path_count": int(paths.shape[0] * paths.shape[1]),
        "correct_rate": float(labels.mean()),
        "projection_refit_on_target": False,
        "probe_refit_on_target": False,
        "metrics": question_bootstrap(
            labels,
            scores,
            trials=bootstrap_trials,
            seed=seed + 3019,
        ),
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# TRACE Outcome Probe",
        "",
        "The probe uses complete eight-transition position, direction, and step features. Cross-fitting holds out entire questions; OOD transfer refits neither PCA nor the probe.",
        "",
        "## Question-held-out",
        "",
        "| Checkpoint/data | Paths | AUROC | AUPRC | Balanced acc | Permutation p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, report in payload["crossfit"].items():
        metric = report["metrics"]
        null = report["within_question_label_permutation_null"]

        def interval(name):
            row = metric[name]
            return (
                f"{row['value']:.3f} "
                f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
            )

        lines.append(
            f"| {label} | {report['path_count']} | {interval('auroc')} | "
            f"{interval('auprc')} | {interval('balanced_accuracy')} | "
            f"{null['p_one_sided']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen OOD transfer",
            "",
            "| Transfer | Paths | AUROC | AUPRC | Balanced acc |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, report in payload["transfer"].items():
        metric = report["metrics"]

        def interval(name):
            row = metric[name]
            if row["value"] is None:
                return "NA"
            return (
                f"{row['value']:.3f} "
                f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
            )

        lines.append(
            f"| {label} | {report['path_count']} | {interval('auroc')} | "
            f"{interval('auprc')} | {interval('balanced_accuracy')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="append",
        type=parse_labeled_path,
        required=True,
    )
    parser.add_argument(
        "--transfer",
        action="append",
        type=parse_transfer,
        default=[],
    )
    parser.add_argument(
        "--crossfit-label",
        action="append",
        default=[],
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    parser.add_argument("--permutation-trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records: Dict[str, List[dict]] = {}
    for label, path in args.record:
        if label in records:
            raise ValueError(f"Duplicate record label {label}")
        records[label] = load_records(path, args.max_records)

    crossfit_labels = (
        args.crossfit_label
        if args.crossfit_label
        else list(records)
    )
    unknown_crossfit = [
        label for label in crossfit_labels if label not in records
    ]
    if unknown_crossfit:
        raise ValueError(f"Unknown crossfit labels: {unknown_crossfit}")
    crossfit = {
        label: crossfit_report(
            records[label],
            pca_components=args.pca_components,
            folds=args.folds,
            ridge=args.ridge,
            bootstrap_trials=args.bootstrap_trials,
            permutation_trials=args.permutation_trials,
            seed=args.seed,
        )
        for label in crossfit_labels
    }
    transfer = {}
    for source_label, target_label in args.transfer:
        if source_label not in records or target_label not in records:
            raise ValueError(
                f"Unknown transfer labels {source_label}:{target_label}"
            )
        transfer[f"{source_label}->{target_label}"] = transfer_report(
            records[source_label],
            records[target_label],
            pca_components=args.pca_components,
            ridge=args.ridge,
            bootstrap_trials=args.bootstrap_trials,
            seed=args.seed,
        )

    payload = {
        "feature_definition": (
            "global-PCA transition directions + cumulative early/mid/late "
            "positions + all eight step magnitudes"
        ),
        "crossfit": crossfit,
        "transfer": transfer,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trace_final_outcome_probe.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.out_dir / "trace_final_outcome_probe.md",
        payload,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
