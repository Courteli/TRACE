#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finite(value):
    return value is not None and np.isfinite(float(value))


def load_separation(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["rollout_signature_separation"]


def ecdf(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    return values, np.arange(1, len(values) + 1, dtype=np.float64) / len(values)


def summarize(separation):
    rows = separation["per_question"]
    mixed = [row for row in rows if row["n_correct"] and row["n_wrong"]]
    paired = [
        row
        for row in mixed
        if finite(row.get("correct_within_mode_distance"))
        and finite(row.get("wrong_nearest_correct_mode_distance"))
    ]
    auc_rows = [row for row in mixed if finite(row.get("wrong_rejection_auc"))]
    null_rows = [
        row
        for row in auc_rows
        if finite(row.get("wrong_rejection_auc_null_mean"))
    ]
    balanced = [row for row in mixed if 2 <= row["n_correct"] <= 6]
    strict = [
        row
        for row in balanced
        if float(row.get("mode_count") or 0) >= 2
        and float(row.get("wrong_nearest_correct_mode_distance") or 0) > 0.1
        and float(row.get("wrong_rejection_auc") or 0) >= 0.75
    ]
    return {
        "representation": separation.get("signature_representation", "raw"),
        "n_questions": separation["n_records_with_path_outcomes"],
        "n_paths": separation["n_paths"],
        "mixed_questions": separation["mixed_count"],
        "paired_margin_questions": len(paired),
        "wrong_farther_than_correct_mode_fraction": (
            float(
                np.mean(
                    [
                        row["wrong_nearest_correct_mode_distance"]
                        > row["correct_within_mode_distance"]
                        for row in paired
                    ]
                )
            )
            if paired
            else None
        ),
        "auc_questions": len(auc_rows),
        "auc_above_chance_fraction": (
            float(np.mean([row["wrong_rejection_auc"] > 0.5 for row in auc_rows]))
            if auc_rows
            else None
        ),
        "auc_at_least_075_count": sum(
            row["wrong_rejection_auc"] >= 0.75 for row in auc_rows
        ),
        "auc_excess_over_null_positive_fraction": (
            float(
                np.mean(
                    [
                        row["wrong_rejection_auc"]
                        > row["wrong_rejection_auc_null_mean"]
                        for row in null_rows
                    ]
                )
            )
            if null_rows
            else None
        ),
        "balanced_mixed_questions": len(balanced),
        "strict_multimode_and_rejection_count": len(strict),
        "strict_question_indices": [int(row["idx"]) for row in strict],
        "aggregate_metrics": separation["metrics"],
    }


def plot_evidence(raw, centered, out_path):
    raw_rows = {int(row["idx"]): row for row in raw["per_question"]}
    centered_rows = {int(row["idx"]): row for row in centered["per_question"]}
    centered_mixed = [
        row
        for row in centered_rows.values()
        if row["n_correct"]
        and row["n_wrong"]
        and finite(row.get("correct_within_mode_distance"))
        and finite(row.get("wrong_nearest_correct_mode_distance"))
    ]
    auc_rows = [
        row
        for row in centered_rows.values()
        if finite(row.get("wrong_rejection_auc"))
        and finite(row.get("wrong_rejection_auc_null_mean"))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    scatter = axes[0, 0].scatter(
        [row["correct_within_mode_distance"] for row in centered_mixed],
        [row["wrong_nearest_correct_mode_distance"] for row in centered_mixed],
        c=[float(row.get("wrong_rejection_auc") or 0.5) for row in centered_mixed],
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.4,
    )
    limit = max(
        [row["wrong_nearest_correct_mode_distance"] for row in centered_mixed]
        + [row["correct_within_mode_distance"] for row in centered_mixed]
    )
    axes[0, 0].plot([0, limit], [0, limit], color="#4b5563", linestyle="--")
    axes[0, 0].set_xlabel("Correct within-mode distance")
    axes[0, 0].set_ylabel("Wrong distance to nearest correct mode")
    axes[0, 0].set_title("Mixed questions under the exact Stage2 signature metric")
    axes[0, 0].grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes[0, 0], label="Leave-one-out wrong-rejection AUC")

    axes[0, 1].scatter(
        [row["wrong_rejection_auc_null_mean"] for row in auc_rows],
        [row["wrong_rejection_auc"] for row in auc_rows],
        color="#7c3aed",
        alpha=0.72,
        edgecolors="white",
        linewidths=0.4,
    )
    axes[0, 1].plot([0, 1], [0, 1], color="#4b5563", linestyle="--")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Per-question permutation-null AUC")
    axes[0, 1].set_ylabel("Observed AUC")
    axes[0, 1].set_title("Observed outcome rejection versus its null")
    axes[0, 1].grid(alpha=0.2)

    correct_distance = [row["correct_within_mode_distance"] for row in centered_mixed]
    wrong_distance = [row["wrong_nearest_correct_mode_distance"] for row in centered_mixed]
    for values, label, color in (
        (correct_distance, "correct rollout to its mode", "#15803d"),
        (wrong_distance, "wrong rollout to nearest correct mode", "#dc2626"),
    ):
        x, y = ecdf(values)
        axes[1, 0].step(x, y, where="post", label=label, color=color, linewidth=2)
    axes[1, 0].set_xlabel("Cosine distance")
    axes[1, 0].set_ylabel("Empirical CDF across mixed questions")
    axes[1, 0].set_title("Correct compactness and wrong-path distance")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    raw_auc = [
        row["wrong_rejection_auc"]
        for row in raw_rows.values()
        if finite(row.get("wrong_rejection_auc"))
    ]
    centered_auc = [row["wrong_rejection_auc"] for row in auc_rows]
    centered_null = [row["wrong_rejection_auc_null_mean"] for row in auc_rows]
    axes[1, 1].boxplot(
        [raw_auc, centered_auc, centered_null],
        tick_labels=["raw", "Stage2 metric", "permutation null"],
        showmeans=True,
    )
    axes[1, 1].axhline(0.5, color="#b91c1c", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Leave-one-out wrong-rejection AUC")
    axes[1, 1].set_title("Representation audit: outcome ranking remains the hard case")
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.suptitle(
        "TRACE epoch7: 200-question outcome-geometry audit without qualitative selection",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_summary", required=True)
    parser.add_argument("--stage2_summary", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_separation(args.raw_summary)
    centered = load_separation(args.stage2_summary)
    summary = {"raw": summarize(raw), "stage2_centered": summarize(centered)}
    (out_dir / "geometry_evidence.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_evidence(raw, centered, out_dir / "geometry_evidence_dashboard.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
