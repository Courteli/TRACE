#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("correct_within_mode_distance", "Correct within-mode compactness", -1.0),
    ("wrong_nearest_correct_mode_distance", "Wrong-to-correct-mode distance", 1.0),
    ("cross_minus_correct_within_mode", "Correct/wrong margin", 1.0),
    ("wrong_minus_correct_within_mode", "Wrong dispersion gap", 1.0),
    ("wrong_rejection_auc_excess_over_null", "Rejection AUC above null", 1.0),
)


def load_summary(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    separation = payload["rollout_signature_separation"]
    rows = {int(row["idx"]): row for row in separation["per_question"]}
    return payload, separation, rows


def bootstrap_mean_ci(values, rng, trials):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        value = float(values[0])
        return value, value
    means = np.empty(trials, dtype=np.float64)
    chunk_size = 256
    for start in range(0, trials, chunk_size):
        count = min(chunk_size, trials - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def composition(rows):
    counts = {"all_wrong": 0, "mixed": 0, "all_correct": 0}
    for row in rows.values():
        if row["n_correct"] == 0:
            counts["all_wrong"] += 1
        elif row["n_wrong"] == 0:
            counts["all_correct"] += 1
        else:
            counts["mixed"] += 1
    return counts


def compare_pair(stage1_rows, trace_rows, rng, trials):
    common = sorted(set(stage1_rows) & set(trace_rows))
    stage1_rollout_acc = np.asarray(
        [stage1_rows[idx]["n_correct"] / stage1_rows[idx]["n_paths"] for idx in common],
        dtype=np.float64,
    )
    trace_rollout_acc = np.asarray(
        [trace_rows[idx]["n_correct"] / trace_rows[idx]["n_paths"] for idx in common],
        dtype=np.float64,
    )
    rollout_delta = trace_rollout_acc - stage1_rollout_acc
    rollout_low, rollout_high = bootstrap_mean_ci(100.0 * rollout_delta, rng, trials)
    return {
        "common_questions": len(common),
        "indices": common,
        "stage1_rollout_accuracy": 100.0 * float(stage1_rollout_acc.mean()),
        "trace_rollout_accuracy": 100.0 * float(trace_rollout_acc.mean()),
        "rollout_accuracy_delta_pp": 100.0 * float(rollout_delta.mean()),
        "rollout_accuracy_delta_bootstrap_ci95_low": rollout_low,
        "rollout_accuracy_delta_bootstrap_ci95_high": rollout_high,
        "stage1_rollout_accuracy_per_question": stage1_rollout_acc.tolist(),
        "trace_rollout_accuracy_per_question": trace_rollout_acc.tolist(),
    }


def compare_metrics(stage1_rows, trace_rows, rng, trials):
    output = []
    for key, label, direction in METRICS:
        paired = []
        for idx in sorted(set(stage1_rows) & set(trace_rows)):
            before = stage1_rows[idx].get(key)
            after = trace_rows[idx].get(key)
            if before is None or after is None:
                continue
            if not np.isfinite(before) or not np.isfinite(after):
                continue
            paired.append((float(before), float(after)))
        if not paired:
            output.append({"metric": key, "label": label, "n": 0})
            continue
        before = np.asarray([item[0] for item in paired], dtype=np.float64)
        after = np.asarray([item[1] for item in paired], dtype=np.float64)
        raw_delta = after - before
        oriented_improvement = direction * raw_delta
        low, high = bootstrap_mean_ci(raw_delta, rng, trials)
        effect_scale = float(raw_delta.std(ddof=1)) if len(raw_delta) > 1 else 0.0
        output.append(
            {
                "metric": key,
                "label": label,
                "direction": "lower_is_better" if direction < 0 else "higher_is_better",
                "n": len(paired),
                "stage1_mean": float(before.mean()),
                "trace_mean": float(after.mean()),
                "raw_delta": float(raw_delta.mean()),
                "raw_delta_bootstrap_ci95_low": low,
                "raw_delta_bootstrap_ci95_high": high,
                "oriented_improvement": float(oriented_improvement.mean()),
                "paired_improved_fraction": float((oriented_improvement > 0).mean()),
                "paired_tied_fraction": float((np.abs(oriented_improvement) <= 1e-12).mean()),
                "standardized_paired_effect": (
                    float(oriented_improvement.mean() / effect_scale) if effect_scale > 0 else None
                ),
            }
        )
    return output


def plot_dashboard(payload, out_path, target_label):
    centered = payload["centered"]
    raw = payload["raw"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))

    x = np.asarray(centered["paired_rollout"]["stage1_rollout_accuracy_per_question"]) * 100.0
    y = np.asarray(centered["paired_rollout"]["trace_rollout_accuracy_per_question"]) * 100.0
    axes[0, 0].scatter(x, y, s=20, alpha=0.55, color="#2563eb", edgecolors="none")
    axes[0, 0].plot([0, 100], [0, 100], linestyle="--", color="#4b5563", linewidth=1)
    axes[0, 0].set_xlim(-3, 103)
    axes[0, 0].set_ylim(-3, 103)
    axes[0, 0].set_xlabel("Stage1 rollout accuracy per question (%)")
    axes[0, 0].set_ylabel(f"{target_label} rollout accuracy per question (%)")
    axes[0, 0].set_title("Matched seeds, views, noise, and questions")
    axes[0, 0].grid(alpha=0.2)

    categories = ("all_wrong", "mixed", "all_correct")
    colors = ("#dc2626", "#d97706", "#15803d")
    for model_idx, model in enumerate(("stage1", "trace")):
        bottom = 0
        for category, color in zip(categories, colors):
            value = centered[f"{model}_composition"][category]
            axes[0, 1].bar(model_idx, value, bottom=bottom, color=color, label=category if model_idx == 0 else None)
            bottom += value
    axes[0, 1].set_xticks([0, 1], labels=["Stage1", target_label])
    axes[0, 1].set_ylabel("Questions (8 rollouts each)")
    axes[0, 1].set_title("Outcome composition over the same 200 questions")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.2)

    centered_metrics = [row for row in centered["metrics"] if row.get("n", 0)]
    labels = [row["label"] for row in centered_metrics]
    positions = np.arange(len(labels))
    effects = np.asarray([row["standardized_paired_effect"] or 0.0 for row in centered_metrics])
    axes[1, 0].barh(positions, effects, color=["#15803d" if value > 0 else "#dc2626" for value in effects])
    axes[1, 0].axvline(0.0, color="#111827", linewidth=1)
    axes[1, 0].set_yticks(positions, labels=labels)
    axes[1, 0].set_xlabel("Oriented paired effect / SD (positive favors TRACE)")
    axes[1, 0].set_title("Exact Stage2-centered signature geometry")
    axes[1, 0].grid(axis="x", alpha=0.2)

    width = 0.35
    raw_map = {row["metric"]: row for row in raw["metrics"] if row.get("n", 0)}
    raw_fraction = np.asarray([raw_map.get(row["metric"], {}).get("paired_improved_fraction", np.nan) for row in centered_metrics])
    centered_fraction = np.asarray([row["paired_improved_fraction"] for row in centered_metrics])
    axes[1, 1].bar(positions - width / 2, raw_fraction, width, label="raw signature", color="#64748b")
    axes[1, 1].bar(positions + width / 2, centered_fraction, width, label="Stage2-centered", color="#7c3aed")
    axes[1, 1].axhline(0.5, color="#111827", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(positions, labels=labels, rotation=18, ha="right")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Fraction of paired questions improved")
    axes[1, 1].set_title("Raw-space and trained-metric robustness check")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.suptitle(f"Stage1 to {target_label}: matched rollout geometry delta", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, payload, target_label):
    centered = payload["centered"]
    paired = centered["paired_rollout"]
    lines = [
        f"# Stage1 to {target_label} Geometry Delta",
        "",
        "Both checkpoints use the same 200 questions, eight view IDs, latent-noise seed, sampling seed, temperature, and top-p. Checkpoint selection did not use these geometry results.",
        "",
        "## Rollout outcomes",
        "",
        f"Stage1 mean rollout accuracy: **{paired['stage1_rollout_accuracy']:.2f}%**  ",
        f"{target_label} mean rollout accuracy: **{paired['trace_rollout_accuracy']:.2f}%**  ",
        f"Paired delta: **{paired['rollout_accuracy_delta_pp']:+.2f} pp** "
        f"(question bootstrap 95% CI [{paired['rollout_accuracy_delta_bootstrap_ci95_low']:+.2f}, "
        f"{paired['rollout_accuracy_delta_bootstrap_ci95_high']:+.2f}]).",
        "",
        "| Model | All wrong | Mixed | All correct |",
        "| --- | ---: | ---: | ---: |",
        f"| Stage1 | {centered['stage1_composition']['all_wrong']} | {centered['stage1_composition']['mixed']} | {centered['stage1_composition']['all_correct']} |",
        f"| {target_label} | {centered['trace_composition']['all_wrong']} | {centered['trace_composition']['mixed']} | {centered['trace_composition']['all_correct']} |",
        "",
        "## Exact Stage2 metric",
        "",
        "Only questions for which a metric is defined for both checkpoints enter that metric's paired comparison; `n` is reported explicitly to avoid survivorship ambiguity.",
        "",
        "| Metric | n | Stage1 | TRACE | Raw delta (95% CI) | Improved fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in centered["metrics"]:
        if not row.get("n", 0):
            continue
        lines.append(
            f"| {row['label']} ({row['direction']}) | {row['n']} | {row['stage1_mean']:.4f} | "
            f"{row['trace_mean']:.4f} | {row['raw_delta']:+.4f} "
            f"[{row['raw_delta_bootstrap_ci95_low']:+.4f}, {row['raw_delta_bootstrap_ci95_high']:+.4f}] | "
            f"{100.0 * row['paired_improved_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule: compactness improves when it decreases; distance, margin, dispersion, and AUC-excess improve when they increase. A confidence interval crossing zero is inconclusive and must not be presented as a positive result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def representation_payload(stage1_path, trace_path, rng, trials):
    _, stage1_sep, stage1_rows = load_summary(stage1_path)
    _, trace_sep, trace_rows = load_summary(trace_path)
    return {
        "stage1_path": str(stage1_path),
        "trace_path": str(trace_path),
        "signature_representation": trace_sep.get("signature_representation", "raw"),
        "stage1_composition": composition(stage1_rows),
        "trace_composition": composition(trace_rows),
        "paired_rollout": compare_pair(stage1_rows, trace_rows, rng, trials),
        "metrics": compare_metrics(stage1_rows, trace_rows, rng, trials),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_raw", type=Path, required=True)
    parser.add_argument("--trace_raw", type=Path, required=True)
    parser.add_argument("--stage1_centered", type=Path, required=True)
    parser.add_argument("--trace_centered", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_label", default="TRACE target")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    payload = {
        "bootstrap_trials": args.bootstrap_trials,
        "seed": args.seed,
        "raw": representation_payload(args.stage1_raw, args.trace_raw, rng, args.bootstrap_trials),
        "centered": representation_payload(
            args.stage1_centered,
            args.trace_centered,
            rng,
            args.bootstrap_trials,
        ),
    }
    (args.out_dir / "stage_geometry_delta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "stage_geometry_delta.md", payload, args.target_label)
    plot_dashboard(payload, args.out_dir / "stage_geometry_delta_dashboard.png", args.target_label)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
