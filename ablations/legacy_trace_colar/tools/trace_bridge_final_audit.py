#!/usr/bin/env python
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_summary(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\\n", "\n")
    rows = {}
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line or "Dataset" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 6:
            continue
        try:
            rows[cells[1]] = {
                "acc": float(cells[2]),
                "n_latent": float(cells[3]),
                "output_length": float(cells[4]),
                "L": float(cells[5]),
            }
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No result rows found in {path}")
    return {
        "datasets": rows,
        "average_acc": sum(row["acc"] for row in rows.values()) / len(rows),
        "average_L": sum(row["L"] for row in rows.values()) / len(rows),
    }


def load_geometry(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["rollout_signature_separation"]


def finite(value):
    return value is not None and math.isfinite(float(value))


def paired_delta(candidate, reference, metric, direction):
    candidate_rows = {int(row["idx"]): row for row in candidate.get("per_question", [])}
    reference_rows = {int(row["idx"]): row for row in reference.get("per_question", [])}
    values = []
    for idx in sorted(set(candidate_rows) & set(reference_rows)):
        candidate_value = candidate_rows[idx].get(metric)
        reference_value = reference_rows[idx].get(metric)
        if finite(candidate_value) and finite(reference_value):
            raw_delta = float(candidate_value) - float(reference_value)
            values.append(raw_delta if direction == "higher" else -raw_delta)
    if not values:
        return {
            "n": 0,
            "improvement": None,
            "ci95": None,
            "lower95": None,
            "upper95": None,
        }
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) > 1:
        variance = float(((values - mean) ** 2).sum() / (len(values) - 1))
        ci95 = 1.96 * math.sqrt(variance / len(values))
        rng = np.random.default_rng(0)
        bootstrap_means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
        lower95, upper95 = np.quantile(bootstrap_means, [0.025, 0.975])
    else:
        ci95 = 0.0
        lower95 = upper95 = mean
    return {
        "n": len(values),
        "improvement": mean,
        "ci95": ci95,
        "lower95": float(lower95),
        "upper95": float(upper95),
    }


def geometry_mean(geometry, metric):
    return geometry.get("metrics", {}).get(metric, {}).get("mean")


def geometry_lower95(geometry, metric):
    summary = geometry.get("metrics", {}).get(metric, {})
    bootstrap_low = summary.get("bootstrap_ci95_low")
    if finite(bootstrap_low):
        return float(bootstrap_low)
    mean = summary.get("mean")
    ci95 = summary.get("ci95")
    if not finite(mean) or not finite(ci95):
        return None
    return float(mean) - float(ci95)


def plot_accuracy_and_paired_geometry(summaries, paired, out_path):
    method_colors = {
        "BRIDGE baseline": "#6b7280",
        "TRACE Stage1": "#2563eb",
        "answer-only Stage2": "#d97706",
        "TRACE Stage2": "#15803d",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    for ax, title, acc_key, length_key in (
        (axes[0, 0], "Average accuracy at the same length budget", "average_acc", "average_L"),
        (axes[0, 1], "GSM8K accuracy at the same length budget", "gsm_acc", "gsm_L"),
    ):
        for name, summary in summaries.items():
            if acc_key == "average_acc":
                accuracy = summary["average_acc"]
                length = summary["average_L"]
            else:
                accuracy = summary["datasets"]["GSM8K-Aug"]["acc"]
                length = summary["datasets"]["GSM8K-Aug"]["L"]
            ax.scatter(length, accuracy, s=70, color=method_colors[name], label=name, zorder=3)
            ax.annotate(name, (length, accuracy), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("#L (latent slots + generated tokens; lower is better)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.25)

    metric_labels = {
        "correct_within_mode_distance": "Correct-mode compactness",
        "wrong_nearest_correct_mode_distance": "Wrong-path rejection",
        "cross_minus_correct_within_mode": "Correct-wrong margin",
        "wrong_minus_correct_within_mode": "Wrong-path dispersion gap",
        "wrong_rejection_auc": "Wrong-path rejection AUC",
    }
    for ax, reference_name in zip(axes[1], ("TRACE Stage1", "answer-only Stage2")):
        comparisons = paired[reference_name]
        labels = list(metric_labels.values())
        positions = np.arange(len(labels))
        for pos, (metric, label) in enumerate(metric_labels.items()):
            result = comparisons[metric]
            if not finite(result["improvement"]):
                continue
            value = float(result["improvement"])
            lower = float(result["lower95"])
            upper = float(result["upper95"])
            ax.errorbar(
                value,
                pos,
                xerr=[[value - lower], [upper - value]],
                fmt="o",
                color="#15803d",
                ecolor="#4b5563",
                capsize=4,
            )
            ax.annotate(f'n={result["n"]}', (upper, pos), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
        ax.axvline(0.0, color="#b91c1c", linestyle="--", linewidth=1.0)
        ax.set_yticks(positions, labels=labels)
        ax.set_xlabel("Paired improvement (positive favors TRACE)")
        ax.set_title(f"200-question geometry vs {reference_name}")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("TRACE advantage: accuracy, length, and outcome-aware latent geometry", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_means(geometries, out_path):
    method_colors = {
        "BRIDGE baseline": "#6b7280",
        "TRACE Stage1": "#2563eb",
        "answer-only Stage2": "#d97706",
        "TRACE Stage2": "#15803d",
    }
    metrics = (
        ("correct_within_mode_distance", "Correct-mode compactness", "lower is better"),
        ("wrong_nearest_correct_mode_distance", "Wrong distance to nearest correct mode", "higher is better"),
        ("cross_minus_correct_within_mode", "Correct-wrong separation margin", "higher is better"),
        ("wrong_minus_correct_within_mode", "Wrong dispersion minus correct compactness", "higher is better"),
        ("wrong_rejection_auc", "Leave-one-out wrong-path rejection AUC", "higher is better"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 11.2))
    names = [name for name in method_colors if name in geometries]
    for ax, (metric, title, direction) in zip(axes.flat, metrics):
        for position, name in enumerate(names):
            summary = geometries[name].get("metrics", {}).get(metric, {})
            mean = summary.get("mean")
            low = summary.get("bootstrap_ci95_low")
            high = summary.get("bootstrap_ci95_high")
            if not finite(mean) or not finite(low) or not finite(high):
                continue
            mean = float(mean)
            low = float(low)
            high = float(high)
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                markersize=7,
                color=method_colors[name],
                ecolor=method_colors[name],
                capsize=4,
            )
            ax.annotate(f'n={summary.get("n", 0)}', (position, high), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xticks(range(len(names)), labels=names, rotation=15, ha="right")
        ax.set_title(f"{title} ({direction})")
        ax.set_ylabel("AUC" if metric == "wrong_rejection_auc" else "Cosine distance")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.flat[len(metrics) :]:
        ax.axis("off")
    fig.suptitle("Same-question rollout geometry over 200 questions (bootstrap 95% CI)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_summary", required=True)
    parser.add_argument("--baseline_geometry")
    parser.add_argument("--stage1_summary", required=True)
    parser.add_argument("--answer_summary", required=True)
    parser.add_argument("--trace_summary", required=True)
    parser.add_argument("--stage1_geometry", required=True)
    parser.add_argument("--answer_geometry", required=True)
    parser.add_argument("--trace_geometry", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--length_tolerance", type=float, default=1.0)
    parser.add_argument(
        "--minimum_mixed_questions",
        type=int,
        default=25,
        help="Minimum same-question correct/wrong rollout groups required for an outcome-geometry claim.",
    )
    parser.add_argument(
        "--minimum_paired_questions",
        type=int,
        default=25,
        help="Minimum paired questions required before a geometry improvement can satisfy a final gate.",
    )
    args = parser.parse_args()

    summaries = {
        "BRIDGE baseline": parse_summary(args.baseline_summary),
        "TRACE Stage1": parse_summary(args.stage1_summary),
        "answer-only Stage2": parse_summary(args.answer_summary),
        "TRACE Stage2": parse_summary(args.trace_summary),
    }
    geometries = {
        "TRACE Stage1": load_geometry(args.stage1_geometry),
        "answer-only Stage2": load_geometry(args.answer_geometry),
        "TRACE Stage2": load_geometry(args.trace_geometry),
    }
    if args.baseline_geometry:
        geometries = {"BRIDGE baseline": load_geometry(args.baseline_geometry), **geometries}

    trace = summaries["TRACE Stage2"]
    stage1 = summaries["TRACE Stage1"]
    answer = summaries["answer-only Stage2"]
    baseline = summaries["BRIDGE baseline"]
    trace_geo = geometries["TRACE Stage2"]

    paired_metrics = {
        "correct_within_mode_distance": "lower",
        "wrong_nearest_correct_mode_distance": "higher",
        "cross_minus_correct_within_mode": "higher",
        "wrong_minus_correct_within_mode": "higher",
        "wrong_rejection_auc": "higher",
    }
    paired = {}
    for reference_name in ("TRACE Stage1", "answer-only Stage2"):
        paired[reference_name] = {
            metric: paired_delta(trace_geo, geometries[reference_name], metric, direction)
            for metric, direction in paired_metrics.items()
        }

    gsm_name = "GSM8K-Aug"
    gates = {
        "average_accuracy_above_stage1": trace["average_acc"] > stage1["average_acc"],
        "average_accuracy_above_answer_only": trace["average_acc"] > answer["average_acc"],
        "gsm8k_accuracy_above_stage1": trace["datasets"][gsm_name]["acc"] > stage1["datasets"][gsm_name]["acc"],
        "gsm8k_accuracy_above_answer_only": trace["datasets"][gsm_name]["acc"] > answer["datasets"][gsm_name]["acc"],
        "average_length_within_stage1_budget": trace["average_L"] <= stage1["average_L"] + args.length_tolerance,
        "gsm8k_length_within_stage1_budget": trace["datasets"][gsm_name]["L"] <= stage1["datasets"][gsm_name]["L"] + args.length_tolerance,
        "outcome_geometry_has_sufficient_mixed_questions": int(trace_geo.get("mixed_count", 0)) >= args.minimum_mixed_questions,
        "correct_wrong_gap_positive_95": (geometry_lower95(trace_geo, "cross_minus_correct_within_mode") or -math.inf) > 0,
        "wrong_dispersion_gap_positive_95": (geometry_lower95(trace_geo, "wrong_minus_correct_within_mode") or -math.inf) > 0,
        "wrong_rejection_auc_above_chance_95": (
            geometry_lower95(trace_geo, "wrong_rejection_auc") or -math.inf
        ) > 0.5,
        "wrong_rejection_auc_above_permuted_null_95": (
            geometry_lower95(trace_geo, "wrong_rejection_auc_excess_over_null") or -math.inf
        ) > 0,
    }
    for reference_name, comparisons in paired.items():
        slug = re.sub(r"[^a-z0-9]+", "_", reference_name.lower()).strip("_")
        for metric, result in comparisons.items():
            gates[f"{metric}_improves_vs_{slug}_at_95"] = (
                result["n"] >= args.minimum_paired_questions
                and finite(result["lower95"])
                and result["lower95"] > 0
            )

    report = {
        "success_definition": {
            "accuracy": "TRACE Stage2 exceeds the same Stage1 and answer-only control",
            "length": f"#L increases by at most {args.length_tolerance:.2f}",
            "geometry": "same-question correct modes compact; wrong paths farther from correct modes and relatively dispersed",
            "minimum_mixed_questions": int(args.minimum_mixed_questions),
            "minimum_paired_questions": int(args.minimum_paired_questions),
        },
        "summaries": summaries,
        "geometry_means": {
            name: {metric: geometry_mean(geometry, metric) for metric in paired_metrics}
            for name, geometry in geometries.items()
        },
        "geometry_negative_controls": {
            name: {
                "wrong_rejection_auc_null_mean": geometry_mean(geometry, "wrong_rejection_auc_null_mean"),
                "wrong_rejection_auc_excess_over_null": geometry_mean(
                    geometry,
                    "wrong_rejection_auc_excess_over_null",
                ),
            }
            for name, geometry in geometries.items()
        },
        "geometry_coverage": {
            name: {
                "n_records_with_path_outcomes": int(geometry.get("n_records_with_path_outcomes", 0)),
                "n_paths": int(geometry.get("n_paths", 0)),
                "mixed_count": int(geometry.get("mixed_count", 0)),
                "mixed_frac": geometry.get("mixed_frac"),
            }
            for name, geometry in geometries.items()
        },
        "paired_trace_improvements": paired,
        "gates": gates,
        "all_required_gates_pass": all(gates.values()),
        "baseline_context": {
            "trace_minus_bridge_average_acc": trace["average_acc"] - baseline["average_acc"],
            "trace_minus_bridge_average_L": trace["average_L"] - baseline["average_L"],
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_and_paired_geometry(
        summaries,
        paired,
        out_dir / "trace_accuracy_length_geometry_advantage.png",
    )
    plot_geometry_means(
        geometries,
        out_dir / "trace_geometry_outcome_summary_200.png",
    )
    report["visual_artifacts"] = {
        "accuracy_length_and_paired_geometry": "trace_accuracy_length_geometry_advantage.png",
        "geometry_means_with_bootstrap_ci": "trace_geometry_outcome_summary_200.png",
    }
    (out_dir / "trace_final_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# TRACE final audit",
        "",
        "| Method | Avg Acc | Avg #L | GSM8K Acc | GSM8K #L |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['average_acc']:.2f} | {summary['average_L']:.2f} | "
            f"{summary['datasets'][gsm_name]['acc']:.2f} | {summary['datasets'][gsm_name]['L']:.2f} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in gates.items())
    lines.extend(["", f"All required gates pass: **{report['all_required_gates_pass']}**", ""])
    (out_dir / "trace_final_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
