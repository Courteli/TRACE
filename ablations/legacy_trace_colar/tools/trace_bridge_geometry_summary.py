#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def load_records(paths):
    records = []
    for path in paths:
        records.extend(torch.load(path, map_location="cpu", weights_only=False))
    return records


def as_float(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def prepare_group_signatures(signatures, representation="raw", raw_mix=0.25):
    raw = normalize(signatures)
    if representation == "raw":
        return raw
    if representation != "stage2_centered":
        raise ValueError(f"Unknown signature representation: {representation}")
    centered = normalize(raw - raw.mean(axis=0, keepdims=True))
    if raw_mix <= 0:
        return centered
    return normalize(np.concatenate([centered, raw_mix * raw], axis=-1))


def mean_pairwise_distance(signatures):
    if signatures.shape[0] <= 1:
        return 0.0
    sig = normalize(signatures)
    sim = sig @ sig.T
    mask = ~np.eye(sig.shape[0], dtype=bool)
    return float((1.0 - sim[mask]).mean())


def path_signature(path):
    mean_state = path.mean(axis=0)
    first = path[0]
    last = path[-1]
    trend = last - first
    deltas = path[1:] - path[:-1] if path.shape[0] > 1 else np.zeros_like(path[:1])
    delta = deltas.mean(axis=0)
    weights = (0.5, 0.5, 1.0, 1.0)
    parts = [
        weight * normalize(x.reshape(1, -1)).reshape(-1)
        for weight, x in zip(weights, (mean_state, last, trend, delta))
    ]
    sig = np.concatenate(parts)
    return normalize(sig.reshape(1, -1)).reshape(-1)


def record_signatures(record):
    if "multiview_implicit_residuals" in record:
        residual_views = as_float(record["multiview_implicit_residuals"])
    else:
        residual_views = as_float(record["implicit_residuals"])[None, ...]
    signatures = []
    for residuals in residual_views:
        signatures.append(path_signature(np.cumsum(residuals, axis=0)))
    return np.stack(signatures, axis=0)


def assignment_progress_metrics(assignment_probs):
    if assignment_probs.shape[1] <= 1:
        return {
            "assignment_progress_span": 0.0,
            "assignment_progress_inversion_frac": 0.0,
            "assignment_progress_center_std": 0.0,
        }
    step_pos = np.linspace(0.0, 1.0, assignment_probs.shape[1], dtype=np.float32)
    centers = (assignment_probs * step_pos.reshape(1, -1)).sum(axis=-1)
    diffs = np.diff(centers)
    return {
        "assignment_progress_span": float(centers.max() - centers.min()),
        "assignment_progress_inversion_frac": float((diffs < -1e-4).mean()) if diffs.size else 0.0,
        "assignment_progress_center_std": float(centers.std()),
    }


def record_metrics(record):
    target_res = as_float(record["aggregated_explicit_residuals"])
    target_path = np.cumsum(target_res, axis=0)
    if "multiview_implicit_residuals" in record:
        residual_views = as_float(record["multiview_implicit_residuals"])
    else:
        residual_views = as_float(record["implicit_residuals"])[None, ...]

    diag_cos = []
    final_path_cos = []
    step_norm = []
    target_step_norm = float(np.linalg.norm(target_res, axis=-1).mean() / np.sqrt(target_res.shape[-1]))
    signatures = []
    for residuals in residual_views:
        implicit_path = np.cumsum(residuals, axis=0)
        diag = (normalize(residuals) * normalize(target_res)).sum(axis=-1).mean()
        final_cos = float((normalize(implicit_path[-1:]) * normalize(target_path[-1:])).sum())
        diag_cos.append(float(diag))
        final_path_cos.append(final_cos)
        step_norm.append(float(np.linalg.norm(residuals, axis=-1).mean() / np.sqrt(residuals.shape[-1])))
        signatures.append(path_signature(implicit_path))

    assignment = as_float(record["assignment"])
    assignment_probs = assignment / np.clip(assignment.sum(axis=-1, keepdims=True), 1e-8, None)
    entropy = -(assignment_probs * np.log(np.clip(assignment_probs, 1e-8, None))).sum(axis=-1)
    norm_entropy = entropy / max(np.log(max(assignment.shape[1], 2)), 1e-8)
    progress_metrics = assignment_progress_metrics(assignment_probs)
    relation = as_float(record["relation_probs"])

    output_length = float(record.get("output_length", 0.0))
    n_latents = float(target_res.shape[0])
    return {
        "idx": int(record.get("idx", -1)),
        "acc": float(record.get("acc", 0.0)),
        "output_length": output_length,
        "n_latents": n_latents,
        "L": n_latents + output_length,
        "diag_residual_cos": float(np.mean(diag_cos)),
        "final_path_cos": float(np.mean(final_path_cos)),
        "implicit_step_norm": float(np.mean(step_norm)),
        "target_step_norm": target_step_norm,
        "step_norm_ratio": float(np.mean(step_norm) / max(target_step_norm, 1e-8)),
        "view_signature_distance": mean_pairwise_distance(np.stack(signatures, axis=0)),
        "assignment_entropy": float(norm_entropy.mean()),
        **progress_metrics,
        "relation_density": float(relation.mean()),
        "n_views": float(residual_views.shape[0]),
    }


def aggregate(rows):
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0].keys() if k != "idx"]
    out = {"n": len(rows)}
    for key in keys:
        out[key] = float(np.mean([row[key] for row in rows]))
    return out


def pairwise_distance(a, b=None, offdiag=False):
    if a.shape[0] == 0:
        return None
    a = normalize(a)
    b = a if b is None else normalize(b)
    if b.shape[0] == 0:
        return None
    dist = 1.0 - (a @ b.T)
    if offdiag:
        if dist.shape[0] <= 1:
            return None
        dist = dist[~np.eye(dist.shape[0], dtype=bool)]
    return float(dist.mean()) if dist.size else None


def build_modes(signatures, max_modes=3, merge_threshold=0.65):
    signatures = normalize(signatures)
    prototypes = [signatures[0]]
    while len(prototypes) < min(max_modes, signatures.shape[0]):
        proto = normalize(np.stack(prototypes, axis=0))
        nearest = (signatures @ proto.T).max(axis=1)
        candidate = int(np.argmin(nearest))
        if nearest[candidate] >= merge_threshold:
            break
        prototypes.append(signatures[candidate])
    prototypes = normalize(np.stack(prototypes, axis=0))
    assignments = np.zeros(signatures.shape[0], dtype=np.int64)
    for _ in range(2):
        assignments = np.argmax(signatures @ prototypes.T, axis=1)
        updated = []
        for mode_idx in range(prototypes.shape[0]):
            members = signatures[assignments == mode_idx]
            updated.append(normalize(members.mean(axis=0, keepdims=True))[0] if len(members) else prototypes[mode_idx])
        prototypes = np.stack(updated, axis=0)
    counts = np.asarray([(assignments == i).sum() for i in range(prototypes.shape[0])], dtype=np.float32)
    probs = counts / max(float(counts.sum()), 1.0)
    effective_modes = float(np.exp(-(probs * np.log(np.clip(probs, 1e-8, None))).sum()))
    return prototypes, assignments, counts, effective_modes


def wrong_rejection_auc_from_signatures(signatures, outcomes, max_modes=3, merge_threshold=0.65):
    positive = signatures[outcomes]
    negative = signatures[~outcomes]
    if len(positive) <= 1 or len(negative) == 0:
        return None
    prototypes, _, _, _ = build_modes(
        positive,
        max_modes=max_modes,
        merge_threshold=merge_threshold,
    )
    wrong_scores = 1.0 - (normalize(negative) @ normalize(prototypes).T).max(axis=1)
    correct_scores = []
    for positive_idx in range(len(positive)):
        leave_one_out = np.delete(positive, positive_idx, axis=0)
        loo_prototypes, _, _, _ = build_modes(
            leave_one_out,
            max_modes=max_modes,
            merge_threshold=merge_threshold,
        )
        correct_score = 1.0 - (
            normalize(positive[positive_idx : positive_idx + 1]) @ normalize(loo_prototypes).T
        ).max()
        correct_scores.append(float(correct_score))
    comparisons = wrong_scores[:, None] - np.asarray(correct_scores, dtype=np.float64)[None, :]
    return float((comparisons > 0).mean() + 0.5 * (np.abs(comparisons) <= 1e-12).mean())


def summarize_values(values):
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95": None,
            "bootstrap_ci95_low": None,
            "bootstrap_ci95_high": None,
        }
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if values.size > 1:
        rng = np.random.default_rng(0)
        bootstrap_means = rng.choice(values, size=(10000, values.size), replace=True).mean(axis=1)
        bootstrap_low, bootstrap_high = np.quantile(bootstrap_means, [0.025, 0.975])
    else:
        bootstrap_low = bootstrap_high = float(values[0])
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(values.size)) if values.size > 1 else 0.0,
        "bootstrap_ci95_low": float(bootstrap_low),
        "bootstrap_ci95_high": float(bootstrap_high),
    }


def signature_separation(
    records,
    max_modes=3,
    merge_threshold=0.65,
    permutation_null_trials=32,
    signature_representation="raw",
    signature_raw_mix=0.25,
):
    metric_values = {
        "correct_within_distance": [],
        "wrong_within_distance": [],
        "correct_wrong_distance": [],
        "wrong_nearest_correct_mode_distance": [],
        "correct_within_mode_distance": [],
        "correct_inter_mode_distance": [],
        "mode_count": [],
        "effective_modes": [],
        "cross_minus_correct_within_mode": [],
        "wrong_minus_correct_within_mode": [],
        "wrong_rejection_auc": [],
        "wrong_rejection_auc_null_mean": [],
        "wrong_rejection_auc_excess_over_null": [],
    }
    n_records = 0
    n_paths = 0
    mixed_count = 0
    all_correct_count = 0
    all_wrong_count = 0
    per_question = []

    for record in records:
        if "multiview_acc" not in record:
            continue
        signatures = prepare_group_signatures(
            record_signatures(record),
            representation=signature_representation,
            raw_mix=signature_raw_mix,
        )
        outcomes = as_float(record["multiview_acc"]).reshape(-1) > 0.5
        if signatures.shape[0] != outcomes.shape[0]:
            continue
        question_row = {
            "idx": int(record.get("idx", -1)),
            "n_paths": int(signatures.shape[0]),
            "n_correct": int(outcomes.sum()),
            "n_wrong": int((~outcomes).sum()),
            "correct_within_distance": None,
            "wrong_within_distance": None,
            "correct_wrong_distance": None,
            "wrong_nearest_correct_mode_distance": None,
            "correct_within_mode_distance": None,
            "correct_inter_mode_distance": None,
            "mode_count": None,
            "effective_modes": None,
            "cross_minus_correct_within_mode": None,
            "wrong_minus_correct_within_mode": None,
            "wrong_rejection_auc": None,
            "wrong_rejection_auc_null_mean": None,
            "wrong_rejection_auc_excess_over_null": None,
        }
        n_records += 1
        n_paths += int(signatures.shape[0])
        positive = signatures[outcomes]
        negative = signatures[~outcomes]
        if len(positive) == 0:
            all_wrong_count += 1
            wrong_within = pairwise_distance(negative, offdiag=True)
            metric_values["wrong_within_distance"].append(wrong_within)
            question_row["wrong_within_distance"] = wrong_within
            per_question.append(question_row)
            continue
        if len(negative) == 0:
            all_correct_count += 1
        else:
            mixed_count += 1

        correct_within = pairwise_distance(positive, offdiag=True)
        wrong_within = pairwise_distance(negative, offdiag=True)
        cross = pairwise_distance(positive, negative)
        question_row["correct_within_distance"] = correct_within
        question_row["wrong_within_distance"] = wrong_within
        question_row["correct_wrong_distance"] = cross
        metric_values["correct_within_distance"].append(correct_within)
        metric_values["wrong_within_distance"].append(wrong_within)
        metric_values["correct_wrong_distance"].append(cross)

        prototypes, assignments, counts, effective_modes = build_modes(
            positive,
            max_modes=max_modes,
            merge_threshold=merge_threshold,
        )
        metric_values["mode_count"].append(float(len(prototypes)))
        metric_values["effective_modes"].append(effective_modes)
        question_row["mode_count"] = float(len(prototypes))
        question_row["effective_modes"] = effective_modes

        within_mode = []
        for mode_idx in range(len(prototypes)):
            members = positive[assignments == mode_idx]
            value = pairwise_distance(members, offdiag=True)
            if value is not None:
                within_mode.append(value)
        correct_within_mode = float(np.mean(within_mode)) if within_mode else None
        correct_inter_mode = pairwise_distance(prototypes, offdiag=True)
        metric_values["correct_within_mode_distance"].append(correct_within_mode)
        metric_values["correct_inter_mode_distance"].append(correct_inter_mode)
        question_row["correct_within_mode_distance"] = correct_within_mode
        question_row["correct_inter_mode_distance"] = correct_inter_mode

        if len(negative):
            nearest_distance = 1.0 - (normalize(negative) @ normalize(prototypes).T).max(axis=1)
            wrong_nearest = float(nearest_distance.mean())
            metric_values["wrong_nearest_correct_mode_distance"].append(wrong_nearest)
            question_row["wrong_nearest_correct_mode_distance"] = wrong_nearest
            if len(positive) > 1:
                auc = wrong_rejection_auc_from_signatures(
                    signatures,
                    outcomes,
                    max_modes=max_modes,
                    merge_threshold=merge_threshold,
                )
                metric_values["wrong_rejection_auc"].append(auc)
                question_row["wrong_rejection_auc"] = auc
                null_aucs = []
                rng = np.random.default_rng(int(record.get("idx", 0)))
                for _ in range(max(0, permutation_null_trials)):
                    null_auc = wrong_rejection_auc_from_signatures(
                        signatures,
                        rng.permutation(outcomes),
                        max_modes=max_modes,
                        merge_threshold=merge_threshold,
                    )
                    if null_auc is not None:
                        null_aucs.append(null_auc)
                if null_aucs:
                    null_mean = float(np.mean(null_aucs))
                    excess = float(auc - null_mean)
                    metric_values["wrong_rejection_auc_null_mean"].append(null_mean)
                    metric_values["wrong_rejection_auc_excess_over_null"].append(excess)
                    question_row["wrong_rejection_auc_null_mean"] = null_mean
                    question_row["wrong_rejection_auc_excess_over_null"] = excess
        if cross is not None and correct_within_mode is not None:
            cross_gap = cross - correct_within_mode
            metric_values["cross_minus_correct_within_mode"].append(cross_gap)
            question_row["cross_minus_correct_within_mode"] = cross_gap
        if wrong_within is not None and correct_within_mode is not None:
            wrong_gap = wrong_within - correct_within_mode
            metric_values["wrong_minus_correct_within_mode"].append(wrong_gap)
            question_row["wrong_minus_correct_within_mode"] = wrong_gap
        per_question.append(question_row)

    return {
        "definition": "same-question per-rollout outcomes only",
        "signature_representation": signature_representation,
        "signature_raw_mix": signature_raw_mix if signature_representation == "stage2_centered" else None,
        "n_records_with_path_outcomes": n_records,
        "n_paths": n_paths,
        "mixed_count": mixed_count,
        "mixed_frac": float(mixed_count / n_records) if n_records else None,
        "all_correct_count": all_correct_count,
        "all_wrong_count": all_wrong_count,
        "metrics": {key: summarize_values(values) for key, values in metric_values.items()},
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument("--max_modes", type=int, default=3)
    parser.add_argument("--mode_merge_threshold", type=float, default=0.65)
    parser.add_argument("--permutation_null_trials", type=int, default=32)
    parser.add_argument(
        "--signature_representation",
        choices=("raw", "stage2_centered"),
        default="raw",
    )
    parser.add_argument("--signature_raw_mix", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records([Path(p) for p in args.records])[: args.max_records]
    rows = [record_metrics(record) for record in records]

    csv_path = out_dir / "trace_bridge_geometry_rows.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "all": aggregate(rows),
        "correct": aggregate([row for row in rows if row["acc"] > 0.5]),
        "wrong": aggregate([row for row in rows if row["acc"] <= 0.5]),
        "rollout_signature_separation": signature_separation(
            records,
            max_modes=args.max_modes,
            merge_threshold=args.mode_merge_threshold,
            permutation_null_trials=args.permutation_null_trials,
            signature_representation=args.signature_representation,
            signature_raw_mix=args.signature_raw_mix,
        ),
    }
    (out_dir / "trace_bridge_geometry_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
