#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trace_bridge_stage_geometry_delta import bootstrap_mean_ci


METRICS = (
    ("assignment_progress_span", "CoT progress span", 1.0),
    ("assignment_progress_inversion_frac", "Progress inversion fraction", -1.0),
    ("assignment_progress_center_std", "Step-center coverage", 1.0),
    ("diag_residual_cos", "Step residual alignment", 1.0),
    ("final_path_cos", "Final path alignment", 1.0),
)


def parse_rows(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--rows must be LABEL=/absolute/trace_bridge_geometry_rows.csv")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_rows(path):
    output = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[int(row["idx"])] = {
                key: float(row[key])
                for key, _, _ in METRICS
            }
    return output


def mean_ci(rows, common, key, rng, trials):
    values = np.asarray([rows[idx][key] for idx in common], dtype=np.float64)
    low, high = bootstrap_mean_ci(values, rng, trials)
    return {
        "mean": float(values.mean()),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
    }


def compare(reference_rows, target_rows, common, rng, trials):
    output = []
    for key, label, direction in METRICS:
        reference = np.asarray([reference_rows[idx][key] for idx in common], dtype=np.float64)
        target = np.asarray([target_rows[idx][key] for idx in common], dtype=np.float64)
        raw_delta = target - reference
        oriented = direction * raw_delta
        raw_low, raw_high = bootstrap_mean_ci(raw_delta, rng, trials)
        oriented_low, oriented_high = bootstrap_mean_ci(oriented, rng, trials)
        scale = float(raw_delta.std(ddof=1)) if len(raw_delta) > 1 else 0.0
        output.append(
            {
                "metric": key,
                "label": label,
                "direction": "lower_is_better" if direction < 0 else "higher_is_better",
                "n": len(common),
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
    return output


def plot_dashboard(payload, out_path):
    labels = payload["method_order"]
    stats = payload["method_statistics"]
    colors = ("#5B7DB1", "#66B07A", "#E6A516", "#E5A6C4")
    color_map = {label: colors[idx % len(colors)] for idx, label in enumerate(labels)}
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4))
    positions = np.arange(len(labels), dtype=np.float64)

    for axis, key, title, ylabel in (
        (axes[0, 0], "assignment_progress_span", "Latent slots cover CoT progress", "Mean normalized progress span"),
        (axes[0, 1], "assignment_progress_inversion_frac", "Latent order follows CoT direction", "Mean inversion fraction"),
    ):
        means = np.asarray([stats[label][key]["mean"] for label in labels])
        low = means - np.asarray([stats[label][key]["bootstrap_ci95_low"] for label in labels])
        high = np.asarray([stats[label][key]["bootstrap_ci95_high"] for label in labels]) - means
        axis.bar(positions, means, color=[color_map[label] for label in labels], width=0.62)
        axis.errorbar(positions, means, yerr=np.vstack([low, high]), fmt="none", color="#111827", capsize=4)
        axis.set_xticks(positions, labels=labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)

    width = 0.34
    for offset, key, metric_label in (
        (-width / 2, "diag_residual_cos", "Step residual"),
        (width / 2, "final_path_cos", "Final path"),
    ):
        values = np.asarray([stats[label][key]["mean"] for label in labels])
        axes[1, 0].bar(positions + offset, values, width, label=metric_label)
    axes[1, 0].set_xticks(positions, labels=labels)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Mean cosine to human-CoT compression target")
    axes[1, 0].set_title("Compression-path alignment")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.2)

    comparisons = list(payload["comparisons"].items())
    y = np.arange(len(METRICS), dtype=np.float64)
    offsets = np.linspace(-0.12, 0.12, max(1, len(comparisons)))
    for comp_idx, (name, rows) in enumerate(comparisons):
        values = np.asarray([row["standardized_paired_effect"] for row in rows])
        low = np.asarray([row["standardized_ci95_low"] for row in rows])
        high = np.asarray([row["standardized_ci95_high"] for row in rows])
        axes[1, 1].errorbar(
            values,
            y + offsets[comp_idx],
            xerr=np.vstack([values - low, high - values]),
            fmt="o",
            capsize=3,
            label=name,
        )
    axes[1, 1].axvline(0.0, color="#111827", linewidth=1)
    axes[1, 1].set_yticks(y, labels=[label for _, label, _ in METRICS])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Oriented paired effect / paired SD")
    axes[1, 1].set_title("Stage1 construction and Stage2 change")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="x", alpha=0.2)

    fig.suptitle("TRACE latent-trajectory structure: full 200-question audit", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, payload):
    lines = [
        "# TRACE 200-Question Latent-Trajectory Structure Audit",
        "",
        "All methods use the same 200 questions. CoT assignments are evaluation-only diagnostics; human CoT is absent at inference. These statistics establish structural alignment, not causal necessity for answer accuracy.",
        "",
        "## Aggregate structure",
        "",
        "| Method | Progress span | Inversion fraction | Step-center coverage | Step alignment | Final-path alignment |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in payload["method_order"]:
        row = payload["method_statistics"][label]
        lines.append(
            f"| {label} | {row['assignment_progress_span']['mean']:.4f} | "
            f"{row['assignment_progress_inversion_frac']['mean']:.4f} | "
            f"{row['assignment_progress_center_std']['mean']:.4f} | "
            f"{row['diag_residual_cos']['mean']:.4f} | {row['final_path_cos']['mean']:.4f} |"
        )
    for comparison, rows in payload["comparisons"].items():
        lines.extend(
            [
                "",
                f"## {comparison}",
                "",
                "| Metric | n | Reference | Target | Raw delta (95% CI) | Improved questions |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['label']} ({row['direction']}) | {row['n']} | {row['reference_mean']:.4f} | "
                f"{row['target_mean']:.4f} | {row['raw_delta']:+.4f} "
                f"[{row['raw_delta_bootstrap_ci95_low']:+.4f}, {row['raw_delta_bootstrap_ci95_high']:+.4f}] | "
                f"{100.0 * row['paired_improved_fraction']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Progress span, inversion, and alignment diagnose the Stage1 weak-anchor/path-consistency construction. The paired statistics above report what the target checkpoint preserves or changes. These geometry diagnostics establish structural organization; causal necessity must be assessed separately with frozen-checkpoint interventions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", action="append", type=parse_rows, required=True)
    parser.add_argument("--comparison", action="append", required=True, help="NAME=REFERENCE,TARGET")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    row_paths = dict(args.rows)
    rows_by_method = {label: load_rows(path) for label, path in row_paths.items()}
    common = sorted(set.intersection(*(set(rows) for rows in rows_by_method.values())))
    if not common:
        raise ValueError("no common question IDs")
    rng = np.random.default_rng(args.seed)
    method_statistics = {
        label: {
            key: mean_ci(rows, common, key, rng, args.bootstrap_trials)
            for key, _, _ in METRICS
        }
        for label, rows in rows_by_method.items()
    }
    comparisons = {}
    for value in args.comparison:
        if "=" not in value or "," not in value:
            raise ValueError("--comparison must be NAME=REFERENCE,TARGET")
        name, pair = value.split("=", 1)
        reference, target = pair.split(",", 1)
        comparisons[name] = compare(
            rows_by_method[reference],
            rows_by_method[target],
            common,
            rng,
            args.bootstrap_trials,
        )
    payload = {
        "method_order": list(row_paths),
        "common_question_count": len(common),
        "common_question_indices": common,
        "bootstrap_trials": args.bootstrap_trials,
        "seed": args.seed,
        "method_statistics": method_statistics,
        "comparisons": comparisons,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trajectory_structure.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "trajectory_structure.md", payload)
    plot_dashboard(payload, args.out_dir / "trajectory_structure_dashboard.png")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
