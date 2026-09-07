#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

BLUE = "#6687B8"
GREEN = "#69B17D"
GOLD = "#E6A314"
PINK = "#E5A6C4"
INK = "#3F4854"


def parse_intervention(value):
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("--intervention must be LABEL=/absolute/results.json")
    return label, Path(path)


def scalar(value):
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def first_string(value):
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)


def load_records(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for key, value in payload.items():
        if not isinstance(value, dict) or "question" not in value:
            continue
        records[value["question"]] = {
            "idx": int(key),
            "question": value["question"],
            "acc": scalar(value["acc"]),
            "output_length": scalar(value["output_length"]),
            "n_latents": scalar(value["n_latent_forward"]),
            "pred_answer": first_string(value.get("pred_answer", "")),
            "output_string": first_string(value.get("output_string", "")),
        }
    if not records:
        raise ValueError(f"No question records found in {path}")
    return records


def exact_mcnemar_p(broken, rescued):
    discordant = broken + rescued
    if discordant == 0:
        return 1.0
    smaller = min(broken, rescued)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_mean_ci(values, rng, trials):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = np.empty(trials, dtype=np.float64)
    chunk_size = 256
    for start in range(0, trials, chunk_size):
        count = min(chunk_size, trials - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def compare(reference, intervention, label, rng, bootstrap_trials):
    common = sorted(set(reference) & set(intervention))
    if not common:
        raise ValueError(f"Reference and {label} have no common questions")
    rows = []
    for question in common:
        normal = reference[question]
        changed = intervention[question]
        normal_correct = normal["acc"] > 0.5
        changed_correct = changed["acc"] > 0.5
        if normal_correct and changed_correct:
            transition = "both_correct"
        elif normal_correct and not changed_correct:
            transition = "broken"
        elif not normal_correct and changed_correct:
            transition = "rescued"
        else:
            transition = "both_wrong"
        normal_L = normal["n_latents"] + normal["output_length"]
        changed_L = changed["n_latents"] + changed["output_length"]
        rows.append(
            {
                "intervention": label,
                "idx": normal["idx"],
                "question": question,
                "normal_acc": normal["acc"],
                "intervention_acc": changed["acc"],
                "normal_L": normal_L,
                "intervention_L": changed_L,
                "delta_L": changed_L - normal_L,
                "normal_pred": normal["pred_answer"],
                "intervention_pred": changed["pred_answer"],
                "prediction_changed": normal["pred_answer"] != changed["pred_answer"],
                "output_changed": normal["output_string"] != changed["output_string"],
                "transition": transition,
            }
        )

    normal_acc = np.asarray([row["normal_acc"] for row in rows], dtype=np.float64)
    changed_acc = np.asarray([row["intervention_acc"] for row in rows], dtype=np.float64)
    acc_delta_pp = 100.0 * (changed_acc - normal_acc)
    length_delta = np.asarray([row["delta_L"] for row in rows], dtype=np.float64)
    acc_low, acc_high = bootstrap_mean_ci(acc_delta_pp, rng, bootstrap_trials)
    length_low, length_high = bootstrap_mean_ci(length_delta, rng, bootstrap_trials)
    prediction_changed = np.asarray([row["prediction_changed"] for row in rows], dtype=np.float64)
    output_changed = np.asarray([row["output_changed"] for row in rows], dtype=np.float64)
    prediction_low, prediction_high = bootstrap_mean_ci(100.0 * prediction_changed, rng, bootstrap_trials)
    output_low, output_high = bootstrap_mean_ci(100.0 * output_changed, rng, bootstrap_trials)
    transitions = {
        name: sum(row["transition"] == name for row in rows)
        for name in ("both_correct", "broken", "rescued", "both_wrong")
    }
    normal_correct_count = transitions["both_correct"] + transitions["broken"]
    return rows, {
        "intervention": label,
        "n": len(rows),
        "normal_acc": 100.0 * float(normal_acc.mean()),
        "intervention_acc": 100.0 * float(changed_acc.mean()),
        "accuracy_delta_pp": float(acc_delta_pp.mean()),
        "accuracy_delta_bootstrap_ci95_low": acc_low,
        "accuracy_delta_bootstrap_ci95_high": acc_high,
        "normal_L": float(np.mean([row["normal_L"] for row in rows])),
        "intervention_L": float(np.mean([row["intervention_L"] for row in rows])),
        "length_delta": float(length_delta.mean()),
        "length_delta_bootstrap_ci95_low": length_low,
        "length_delta_bootstrap_ci95_high": length_high,
        "prediction_changed_count": int(prediction_changed.sum()),
        "prediction_changed_fraction": float(prediction_changed.mean()),
        "prediction_changed_percent_bootstrap_ci95_low": prediction_low,
        "prediction_changed_percent_bootstrap_ci95_high": prediction_high,
        "output_changed_count": int(output_changed.sum()),
        "output_changed_fraction": float(output_changed.mean()),
        "output_changed_percent_bootstrap_ci95_low": output_low,
        "output_changed_percent_bootstrap_ci95_high": output_high,
        **transitions,
        "normal_correct_destroyed_fraction": transitions["broken"] / max(1, normal_correct_count),
        "mcnemar_exact_p": exact_mcnemar_p(transitions["broken"], transitions["rescued"]),
    }


def plot_dashboard(summaries, out_path):
    labels = [row["intervention"] for row in summaries]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))

    normal = np.asarray([row["normal_acc"] for row in summaries])
    changed = np.asarray([row["intervention_acc"] for row in summaries])
    width = 0.36
    axes[0, 0].bar(y - width / 2, normal, width, label="intact", color=BLUE)
    axes[0, 0].bar(y + width / 2, changed, width, label="intervened", color=PINK)
    axes[0, 0].set_xticks(y, labels=labels, rotation=15, ha="right")
    axes[0, 0].set_ylabel("GSM8K accuracy (%)")
    axes[0, 0].set_title("Frozen model, latent-input intervention")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.25)

    delta = np.asarray([row["accuracy_delta_pp"] for row in summaries])
    low = np.asarray([row["accuracy_delta_bootstrap_ci95_low"] for row in summaries])
    high = np.asarray([row["accuracy_delta_bootstrap_ci95_high"] for row in summaries])
    axes[0, 1].errorbar(
        delta,
        y,
        xerr=np.vstack((delta - low, high - delta)),
        fmt="o",
        color=PINK,
        ecolor=INK,
        capsize=4,
    )
    axes[0, 1].axvline(0.0, color=INK, linestyle="--", linewidth=1)
    axes[0, 1].set_yticks(y, labels=labels)
    axes[0, 1].set_xlabel("Accuracy change (percentage points)")
    axes[0, 1].set_title("Paired question bootstrap 95% CI")
    axes[0, 1].grid(axis="x", alpha=0.25)

    broken = np.asarray([row["broken"] for row in summaries])
    rescued = np.asarray([row["rescued"] for row in summaries])
    axes[1, 0].bar(y - width / 2, broken, width, label="broken", color=GOLD)
    axes[1, 0].bar(y + width / 2, rescued, width, label="rescued", color=GREEN)
    axes[1, 0].set_xticks(y, labels=labels, rotation=15, ha="right")
    axes[1, 0].set_ylabel("Paired question count")
    axes[1, 0].set_title("Destroyed versus rescued answers")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    length = np.asarray([row["length_delta"] for row in summaries])
    length_low = np.asarray([row["length_delta_bootstrap_ci95_low"] for row in summaries])
    length_high = np.asarray([row["length_delta_bootstrap_ci95_high"] for row in summaries])
    axes[1, 1].errorbar(
        length,
        y,
        xerr=np.vstack((length - length_low, length_high - length)),
        fmt="o",
        color=PINK,
        ecolor=INK,
        capsize=4,
    )
    axes[1, 1].axvline(0.0, color=INK, linestyle="--", linewidth=1)
    axes[1, 1].set_yticks(y, labels=labels)
    axes[1, 1].set_xlabel("Delta #L (intervention - normal)")
    axes[1, 1].set_title("Response-length side effect")
    axes[1, 1].grid(axis="x", alpha=0.25)

    fig.suptitle("TRACE latent-slot intervention audit", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, summaries):
    lines = [
        "# TRACE Latent-Slot Intervention Audit",
        "",
        "The checkpoint and decoding budget are frozen. Each control changes only the latent input-embedding sequence before the causal latent forward pass; hidden trajectory states are then recomputed. Every result is paired by exact question text.",
        "",
        "| Intervention | Normal Acc | Intervened Acc | Delta Acc (95% CI) | Broken | Rescued | Exact p | Delta #L (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {intervention} | {normal_acc:.2f} | {intervention_acc:.2f} | "
            "{accuracy_delta_pp:+.2f} [{accuracy_delta_bootstrap_ci95_low:+.2f}, {accuracy_delta_bootstrap_ci95_high:+.2f}] | "
            "{broken} | {rescued} | {mcnemar_exact_p:.3g} | "
            "{length_delta:+.2f} [{length_delta_bootstrap_ci95_low:+.2f}, {length_delta_bootstrap_ci95_high:+.2f}] |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Behavioral sensitivity",
            "",
            "These diagnostics do not replace the accuracy test: they only show whether an intervention changes the numeric prediction or the exact generated reasoning text.",
            "",
            "| Intervention | Numeric prediction changed | Exact generated text changed |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['intervention']} | {100.0 * row['prediction_changed_fraction']:.2f}% "
            f"[{row['prediction_changed_percent_bootstrap_ci95_low']:.2f}, "
            f"{row['prediction_changed_percent_bootstrap_ci95_high']:.2f}] | "
            f"{100.0 * row['output_changed_fraction']:.2f}% "
            f"[{row['output_changed_percent_bootstrap_ci95_low']:.2f}, "
            f"{row['output_changed_percent_bootstrap_ci95_high']:.2f}] |"
        )
    lines.extend(
        [
            "",
            "Controls:",
            "- `reverse`: preserves the latent input-embedding multiset and reverses its order before the latent forward pass.",
            "- `shuffle`: preserves that input multiset and uses a deterministic question-specific permutation.",
            "- `mean_repeat`: preserves the input-embedding centroid, replaces every slot by that centroid, and removes slot-specific input content.",
            "- `random_direction`: preserves each latent input's distance from the input-embedding centroid while replacing its learned offset direction.",
            "",
            "A significant accuracy drop supports sensitivity to the corresponding latent-input property. A null result is evidence that the tested input property is not necessary under this intervention; it is not a direct intervention on frozen hidden trajectory states.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--intervention", action="append", type=parse_intervention, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_records(args.reference)
    rng = np.random.default_rng(args.seed)
    summaries = []
    all_rows = []
    for label, path in args.intervention:
        rows, summary = compare(reference, load_records(path), label, rng, args.bootstrap_trials)
        all_rows.extend(rows)
        summaries.append(summary)

    payload = {
        "reference": str(args.reference),
        "bootstrap_trials": args.bootstrap_trials,
        "seed": args.seed,
        "summaries": summaries,
    }
    (args.out_dir / "causal_evidence.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "causal_question_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_markdown(args.out_dir / "causal_evidence.md", summaries)
    plot_dashboard(summaries, args.out_dir / "causal_evidence_dashboard.png")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
