#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def path_signatures(residuals, prefix_length):
    path = np.cumsum(residuals[:, :prefix_length], axis=1)
    mean_state = path.mean(axis=1)
    first = path[:, 0]
    last = path[:, -1]
    trend = last - first
    if prefix_length > 1:
        delta = np.diff(path, axis=1).mean(axis=1)
    else:
        delta = np.zeros_like(mean_state)
    parts = (
        0.5 * normalize(mean_state),
        0.5 * normalize(last),
        normalize(trend),
        normalize(delta),
    )
    return normalize(np.concatenate(parts, axis=-1))


def prepare_stage2(signatures, raw_mix):
    raw = normalize(signatures)
    centered = normalize(raw - raw.mean(axis=0, keepdims=True))
    return normalize(np.concatenate([centered, raw_mix * raw], axis=-1))


def hash_project(features, buckets, signs, output_dim):
    projected = np.zeros((features.shape[0], output_dim), dtype=np.float32)
    for bucket in range(output_dim):
        mask = buckets == bucket
        if mask.any():
            projected[:, bucket] = (features[:, mask] * signs[mask]).sum(axis=1)
    return normalize(projected)


def auc(scores, labels):
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (np.abs(comparisons) <= 1e-12).mean())


def fit_ridge(train_x, train_y, ridge):
    train_x = np.concatenate(
        [train_x, np.ones((len(train_x), 1), dtype=np.float32)], axis=1
    )
    positive = max(int(train_y.sum()), 1)
    negative = max(int((~train_y).sum()), 1)
    weights = np.where(train_y, 0.5 / positive, 0.5 / negative) * len(train_y)
    weighted_x = train_x * np.sqrt(weights[:, None])
    target = np.where(train_y, 1.0, -1.0) * np.sqrt(weights)
    penalty = ridge * np.eye(train_x.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(weighted_x.T @ weighted_x + penalty, weighted_x.T @ target)


def predict_ridge(test_x, weights):
    test_x = np.concatenate(
        [test_x, np.ones((len(test_x), 1), dtype=np.float32)], axis=1
    )
    return test_x @ weights


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--raw_mix", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = torch.load(args.records, map_location="cpu", weights_only=False)[
        : args.max_records
    ]
    mixed = []
    for record in records:
        if "multiview_acc" not in record:
            continue
        outcomes = np.asarray(record["multiview_acc"], dtype=np.float32).reshape(-1) > 0.5
        if outcomes.any() and (~outcomes).any():
            mixed.append(
                {
                    "idx": int(record["idx"]),
                    "residuals": np.asarray(
                        record["multiview_implicit_residuals"], dtype=np.float32
                    ),
                    "outcomes": outcomes,
                }
            )
    if len(mixed) < 20:
        raise ValueError(f"Need at least 20 mixed questions, found {len(mixed)}")

    rng = np.random.default_rng(args.seed)
    split_orders = [rng.permutation(len(mixed)) for _ in range(args.repeats)]
    results = {"raw": {}, "stage2_centered": {}, "permuted_train_null": {}}
    max_steps = min(item["residuals"].shape[1] for item in mixed)

    for prefix in range(1, max_steps + 1):
        raw_groups = [path_signatures(item["residuals"], prefix) for item in mixed]
        centered_groups = [prepare_stage2(group, args.raw_mix) for group in raw_groups]
        representation_groups = {"raw": raw_groups, "stage2_centered": centered_groups}
        projected_groups = {}
        for representation_idx, (representation, groups) in enumerate(
            representation_groups.items()
        ):
            feature_dim = groups[0].shape[1]
            projection_rng = np.random.default_rng(
                args.seed + 1009 * prefix + 7919 * representation_idx
            )
            buckets = projection_rng.integers(
                0, args.projection_dim, size=feature_dim
            )
            signs = projection_rng.choice(
                np.asarray([-1.0, 1.0], dtype=np.float32), size=feature_dim
            )
            group_sizes = [len(group) for group in groups]
            all_projected = hash_project(
                np.concatenate(groups, axis=0),
                buckets,
                signs,
                args.projection_dim,
            )
            split_points = np.cumsum(group_sizes)[:-1]
            projected_groups[representation] = np.split(all_projected, split_points)
        prefix_values = {"raw": [], "stage2_centered": [], "permuted_train_null": []}

        for repeat, order in enumerate(split_orders):
            split = len(order) // 2
            train_questions = set(order[:split].tolist())
            test_questions = set(order[split:].tolist())
            permutation_rng = np.random.default_rng(
                args.seed + 1009 * repeat + 17 * prefix
            )

            for representation, projected in projected_groups.items():
                train_x = np.concatenate(
                    [projected[idx] for idx in sorted(train_questions)], axis=0
                )
                train_y = np.concatenate(
                    [mixed[idx]["outcomes"] for idx in sorted(train_questions)]
                )
                test_x = np.concatenate(
                    [projected[idx] for idx in sorted(test_questions)], axis=0
                )
                test_y = np.concatenate(
                    [mixed[idx]["outcomes"] for idx in sorted(test_questions)]
                )
                weights = fit_ridge(train_x, train_y, args.ridge)
                value = auc(predict_ridge(test_x, weights), test_y)
                if value is not None:
                    prefix_values[representation].append(value)

                if representation == "stage2_centered":
                    permuted_y = np.concatenate(
                        [
                            permutation_rng.permutation(mixed[idx]["outcomes"])
                            for idx in sorted(train_questions)
                        ]
                    )
                    null_weights = fit_ridge(train_x, permuted_y, args.ridge)
                    null_value = auc(predict_ridge(test_x, null_weights), test_y)
                    if null_value is not None:
                        prefix_values["permuted_train_null"].append(null_value)

        for key, values in prefix_values.items():
            results[key][str(prefix)] = summarize(values)

    output = {
        "definition": (
            "linear ridge probe trained and tested on disjoint mixed-question groups; "
            "features are deterministic sparse random projections of latent-prefix signatures"
        ),
        "n_mixed_questions": len(mixed),
        "question_indices": [item["idx"] for item in mixed],
        "repeats": args.repeats,
        "projection_dim": args.projection_dim,
        "ridge": args.ridge,
        "raw_mix": args.raw_mix,
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prefix_probe.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    styles = {
        "raw": ("#6b7280", "Raw signature"),
        "stage2_centered": ("#15803d", "Exact Stage2 signature"),
        "permuted_train_null": ("#dc2626", "Within-question label-permutation null"),
    }
    x = np.arange(1, max_steps + 1)
    for key, (color, label) in styles.items():
        means = np.asarray([results[key][str(step)]["mean"] for step in x])
        low = np.asarray([results[key][str(step)]["ci95_low"] for step in x])
        high = np.asarray([results[key][str(step)]["ci95_high"] for step in x])
        ax.plot(x, means, marker="o", color=color, label=label)
        ax.fill_between(x, low, high, color=color, alpha=0.14)
    ax.axhline(0.5, color="#111827", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("Latent prefix length")
    ax.set_ylabel("Held-out rollout correctness AUROC")
    ax.set_ylim(0.35, 0.75)
    ax.set_title("Does the latent trajectory progressively reveal final correctness?")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "prefix_probe.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
