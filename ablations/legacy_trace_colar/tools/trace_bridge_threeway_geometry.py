#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trace_bridge_stage_geometry_delta import METRICS, bootstrap_mean_ci, composition, load_summary


def parse_summary(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--summary must be LABEL=/absolute/summary.json")
    label, path = value.split("=", 1)
    return label, Path(path)


def method_statistics(rows_by_method, common, rng, trials):
    output = {}
    for label, rows in rows_by_method.items():
        accuracies = np.asarray(
            [rows[idx]["n_correct"] / rows[idx]["n_paths"] for idx in common],
            dtype=np.float64,
        )
        ci_low, ci_high = bootstrap_mean_ci(100.0 * accuracies, rng, trials)
        selected_rows = {idx: rows[idx] for idx in common}
        output[label] = {
            "question_count": len(common),
            "rollout_accuracy": 100.0 * float(accuracies.mean()),
            "rollout_accuracy_bootstrap_ci95_low": ci_low,
            "rollout_accuracy_bootstrap_ci95_high": ci_high,
            "composition": composition(selected_rows),
        }
    return output


def paired_geometry(reference_rows, target_rows, rng, trials):
    metrics = []
    shared = sorted(set(reference_rows) & set(target_rows))
    for key, label, direction in METRICS:
        pairs = []
        for idx in shared:
            reference = reference_rows[idx].get(key)
            target = target_rows[idx].get(key)
            if reference is None or target is None:
                continue
            if not np.isfinite(reference) or not np.isfinite(target):
                continue
            pairs.append((float(reference), float(target)))
        if not pairs:
            metrics.append({"metric": key, "label": label, "n": 0})
            continue
        reference = np.asarray([item[0] for item in pairs], dtype=np.float64)
        target = np.asarray([item[1] for item in pairs], dtype=np.float64)
        raw_delta = target - reference
        oriented = direction * raw_delta
        raw_low, raw_high = bootstrap_mean_ci(raw_delta, rng, trials)
        oriented_low, oriented_high = bootstrap_mean_ci(oriented, rng, trials)
        scale = float(oriented.std(ddof=1)) if len(oriented) > 1 else 0.0
        metrics.append(
            {
                "metric": key,
                "label": label,
                "direction": "lower_is_better" if direction < 0 else "higher_is_better",
                "n": len(pairs),
                "reference_mean": float(reference.mean()),
                "target_mean": float(target.mean()),
                "raw_delta": float(raw_delta.mean()),
                "raw_delta_bootstrap_ci95_low": raw_low,
                "raw_delta_bootstrap_ci95_high": raw_high,
                "oriented_improvement": float(oriented.mean()),
                "oriented_bootstrap_ci95_low": oriented_low,
                "oriented_bootstrap_ci95_high": oriented_high,
                "standardized_paired_effect": float(oriented.mean() / scale) if scale > 0 else 0.0,
                "standardized_ci95_low": float(oriented_low / scale) if scale > 0 else 0.0,
                "standardized_ci95_high": float(oriented_high / scale) if scale > 0 else 0.0,
                "paired_improved_fraction": float((oriented > 0).mean()),
            }
        )
    return metrics


def plot_dashboard(payload, out_path):
    labels = payload["method_order"]
    target = payload["target_label"]
    stats = payload["method_statistics"]
    comparisons = payload["comparisons"]
    colors = ("#475569", "#d97706", "#15803d", "#2563eb")
    method_colors = {label: colors[idx % len(colors)] for idx, label in enumerate(labels)}
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.6))

    positions = np.arange(len(labels))
    means = np.asarray([stats[label]["rollout_accuracy"] for label in labels])
    low = means - np.asarray([stats[label]["rollout_accuracy_bootstrap_ci95_low"] for label in labels])
    high = np.asarray([stats[label]["rollout_accuracy_bootstrap_ci95_high"] for label in labels]) - means
    axes[0, 0].bar(positions, means, color=[method_colors[label] for label in labels], width=0.62)
    axes[0, 0].errorbar(positions, means, yerr=np.vstack([low, high]), fmt="none", color="#111827", capsize=4)
    axes[0, 0].set_xticks(positions, labels=labels)
    axes[0, 0].set_ylabel("Mean correct rollouts per question (%)")
    axes[0, 0].set_title("Same 200 questions and eight rollout views")
    axes[0, 0].grid(axis="y", alpha=0.2)

    categories = ("all_wrong", "mixed", "all_correct")
    category_colors = ("#dc2626", "#d97706", "#15803d")
    bottoms = np.zeros(len(labels), dtype=np.float64)
    for category, color in zip(categories, category_colors):
        values = np.asarray([stats[label]["composition"][category] for label in labels], dtype=np.float64)
        axes[0, 1].bar(positions, values, bottom=bottoms, color=color, label=category.replace("_", " "))
        bottoms += values
    axes[0, 1].set_xticks(positions, labels=labels)
    axes[0, 1].set_ylabel("Questions")
    axes[0, 1].set_title("Outcome composition")
    axes[0, 1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.2)

    metric_labels = [item[1] for item in METRICS]
    y = np.arange(len(metric_labels), dtype=np.float64)
    comparison_items = list(comparisons.items())
    offsets = np.linspace(-0.16, 0.16, max(1, len(comparison_items)))
    for comparison_idx, (reference, metrics) in enumerate(comparison_items):
        by_key = {row["metric"]: row for row in metrics if row.get("n", 0)}
        values = np.asarray([by_key[key]["standardized_paired_effect"] for key, _, _ in METRICS])
        ci_low = np.asarray([by_key[key]["standardized_ci95_low"] for key, _, _ in METRICS])
        ci_high = np.asarray([by_key[key]["standardized_ci95_high"] for key, _, _ in METRICS])
        xerr = np.vstack([values - ci_low, ci_high - values])
        axes[1, 0].errorbar(
            values,
            y + offsets[comparison_idx],
            xerr=xerr,
            fmt="o",
            capsize=3,
            color=method_colors[reference],
            label=f"{target} vs {reference}",
        )
    axes[1, 0].axvline(0.0, color="#111827", linewidth=1)
    axes[1, 0].set_yticks(y, labels=metric_labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Oriented paired effect / paired SD (positive favors TRACE)")
    axes[1, 0].set_title("Question-paired geometry with bootstrap 95% CI")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="x", alpha=0.2)

    width = 0.72 / max(1, len(comparison_items))
    for comparison_idx, (reference, metrics) in enumerate(comparison_items):
        by_key = {row["metric"]: row for row in metrics if row.get("n", 0)}
        values = np.asarray([by_key[key]["paired_improved_fraction"] for key, _, _ in METRICS])
        x = np.arange(len(metric_labels)) - 0.36 + width / 2 + comparison_idx * width
        axes[1, 1].bar(x, values, width, color=method_colors[reference], label=f"vs {reference}")
    axes[1, 1].axhline(0.5, color="#111827", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(np.arange(len(metric_labels)), labels=metric_labels, rotation=18, ha="right")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Fraction of paired questions improved")
    axes[1, 1].set_title("Per-question direction of change")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)

    representation = payload["signature_representation"]
    fig.suptitle(f"BRIDGE to TRACE: 200-question geometry scorecard ({representation})", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, payload):
    labels = payload["method_order"]
    target = payload["target_label"]
    stats = payload["method_statistics"]
    lines = [
        "# BRIDGE / Stage1 / TRACE 200-Question Geometry Scorecard",
        "",
        "All methods use the common question intersection, eight rollout views, and the same signature definition. Geometry checkpoint selection did not use this comparison.",
        "",
        "Compactness is conditional on noncollapsed modes. BRIDGE maps all audit views to one identical signature mode, so its near-zero within-mode distance is a degenerate no-diversity value rather than evidence of superior compact multi-path reasoning.",
        "",
        "## Rollout outcomes",
        "",
        "| Method | Questions | Rollout accuracy (95% CI) | All wrong | Mixed | All correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        row = stats[label]
        comp = row["composition"]
        lines.append(
            f"| {label} | {row['question_count']} | {row['rollout_accuracy']:.2f}% "
            f"[{row['rollout_accuracy_bootstrap_ci95_low']:.2f}, {row['rollout_accuracy_bootstrap_ci95_high']:.2f}] | "
            f"{comp['all_wrong']} | {comp['mixed']} | {comp['all_correct']} |"
        )
    for reference, metrics in payload["comparisons"].items():
        lines.extend(
            [
                "",
                f"## {target} versus {reference}",
                "",
                "Only questions where the metric exists for both methods are included.",
                "",
                "| Metric | n | Reference | TRACE | Raw delta (95% CI) | Paired improved |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in metrics:
            if not row.get("n", 0):
                continue
            lines.append(
                f"| {row['label']} ({row['direction']}) | {row['n']} | {row['reference_mean']:.4f} | "
                f"{row['target_mean']:.4f} | {row['raw_delta']:+.4f} "
                f"[{row['raw_delta_bootstrap_ci95_low']:+.4f}, {row['raw_delta_bootstrap_ci95_high']:+.4f}] | "
                f"{100.0 * row['paired_improved_fraction']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "Positive oriented effects favor TRACE: compactness is oriented downward; distance, margin, dispersion, and AUC-excess are oriented upward. Confidence intervals crossing zero are inconclusive. Collective dispersion and single-path rejection remain separate claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=parse_summary, required=True)
    parser.add_argument("--target_label", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summaries = dict(args.summary)
    if args.target_label not in summaries:
        raise ValueError(f"target label {args.target_label!r} is not among {list(summaries)}")
    rows_by_method = {}
    representations = set()
    for label, path in summaries.items():
        _, separation, rows = load_summary(path)
        rows_by_method[label] = rows
        representations.add(separation.get("signature_representation", "raw"))
    if len(representations) != 1:
        raise ValueError(f"signature representations differ: {sorted(representations)}")
    common = sorted(set.intersection(*(set(rows) for rows in rows_by_method.values())))
    if not common:
        raise ValueError("no common question IDs")

    rng = np.random.default_rng(args.seed)
    target_rows = rows_by_method[args.target_label]
    comparisons = {
        label: paired_geometry(rows, target_rows, rng, args.bootstrap_trials)
        for label, rows in rows_by_method.items()
        if label != args.target_label
    }
    payload = {
        "method_order": list(summaries),
        "target_label": args.target_label,
        "common_question_count": len(common),
        "common_question_indices": common,
        "signature_representation": next(iter(representations)),
        "bootstrap_trials": args.bootstrap_trials,
        "seed": args.seed,
        "method_statistics": method_statistics(rows_by_method, common, rng, args.bootstrap_trials),
        "comparisons": comparisons,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "threeway_geometry.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "threeway_geometry.md", payload)
    plot_dashboard(payload, args.out_dir / "threeway_geometry_dashboard.png")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
