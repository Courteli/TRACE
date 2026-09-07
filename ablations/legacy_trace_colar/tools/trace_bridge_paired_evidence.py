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


def parse_dataset(value):
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--dataset must be LABEL=/absolute/reference.json=/absolute/candidate.json"
        )
    return parts[0], Path(parts[1]), Path(parts[2])


def scalar(value):
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def load_records(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for key, value in payload.items():
        if not isinstance(value, dict) or "question" not in value:
            continue
        records[value["question"]] = {
            "idx": int(key),
            "question": value["question"],
            "answer": str(value.get("answer", "")),
            "pred_answer": str(value.get("pred_answer", [""])[0]),
            "output": str(value.get("output_string", [""])[0]),
            "acc": scalar(value["acc"]),
            "output_length": scalar(value["output_length"]),
            "n_latents": scalar(value["n_latent_forward"]),
        }
    if not records:
        raise ValueError(f"No question records found in {path}")
    return records


def exact_mcnemar_p(rescued, regressed):
    discordant = rescued + regressed
    if discordant == 0:
        return 1.0
    smaller = min(rescued, regressed)
    tail = sum(math.comb(discordant, idx) for idx in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_mean_ci(values, rng, trials):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
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


def paired_rows(reference, candidate):
    common = sorted(set(reference) & set(candidate))
    rows = []
    for question in common:
        ref = reference[question]
        cand = candidate[question]
        ref_L = ref["n_latents"] + ref["output_length"]
        cand_L = cand["n_latents"] + cand["output_length"]
        if ref["acc"] > 0.5 and cand["acc"] > 0.5:
            transition = "both_correct"
        elif ref["acc"] <= 0.5 and cand["acc"] > 0.5:
            transition = "rescued"
        elif ref["acc"] > 0.5 and cand["acc"] <= 0.5:
            transition = "regressed"
        else:
            transition = "both_wrong"
        rows.append(
            {
                "idx": ref["idx"],
                "question": question,
                "answer": ref["answer"],
                "reference_pred": ref["pred_answer"],
                "candidate_pred": cand["pred_answer"],
                "reference_output": ref["output"],
                "candidate_output": cand["output"],
                "reference_acc": ref["acc"],
                "candidate_acc": cand["acc"],
                "reference_L": ref_L,
                "candidate_L": cand_L,
                "delta_L": cand_L - ref_L,
                "transition": transition,
            }
        )
    if not rows:
        raise ValueError("Reference and candidate have no common questions")
    return rows


def summarize(label, rows, rng, bootstrap_trials):
    transitions = {
        name: sum(row["transition"] == name for row in rows)
        for name in ("both_correct", "rescued", "regressed", "both_wrong")
    }
    acc_delta = np.asarray(
        [row["candidate_acc"] - row["reference_acc"] for row in rows],
        dtype=np.float64,
    )
    length_delta = np.asarray([row["delta_L"] for row in rows], dtype=np.float64)
    acc_low, acc_high = bootstrap_mean_ci(100.0 * acc_delta, rng, bootstrap_trials)
    length_low, length_high = bootstrap_mean_ci(length_delta, rng, bootstrap_trials)
    return {
        "dataset": label,
        "n": len(rows),
        "reference_acc": 100.0 * float(np.mean([row["reference_acc"] for row in rows])),
        "candidate_acc": 100.0 * float(np.mean([row["candidate_acc"] for row in rows])),
        "accuracy_delta_pp": 100.0 * float(acc_delta.mean()),
        "accuracy_delta_bootstrap_ci95_low": acc_low,
        "accuracy_delta_bootstrap_ci95_high": acc_high,
        "reference_L": float(np.mean([row["reference_L"] for row in rows])),
        "candidate_L": float(np.mean([row["candidate_L"] for row in rows])),
        "length_delta": float(length_delta.mean()),
        "length_delta_bootstrap_ci95_low": length_low,
        "length_delta_bootstrap_ci95_high": length_high,
        **transitions,
        "net_rescues": transitions["rescued"] - transitions["regressed"],
        "mcnemar_exact_p": exact_mcnemar_p(
            transitions["rescued"], transitions["regressed"]
        ),
    }


def select_representative_samples(rows, per_class):
    selected = []
    for transition in ("rescued", "regressed", "both_correct", "both_wrong"):
        candidates = [row for row in rows if row["transition"] == transition]
        if not candidates:
            continue
        median_delta = float(np.median([row["delta_L"] for row in candidates]))
        candidates.sort(key=lambda row: (abs(row["delta_L"] - median_delta), row["idx"]))
        for rank, row in enumerate(candidates[:per_class], start=1):
            selected.append(
                {
                    **row,
                    "selection_rank": rank,
                    "class_median_delta_L": median_delta,
                    "selection_rule": (
                        "transition class first; then smallest absolute distance to the "
                        "class-median delta_L; dataset index breaks ties"
                    ),
                }
            )
    return selected


def plot_dashboard(summaries, reference_name, candidate_name, out_path):
    labels = [row["dataset"] for row in summaries]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))

    acc = np.asarray([row["accuracy_delta_pp"] for row in summaries])
    acc_low = np.asarray([row["accuracy_delta_bootstrap_ci95_low"] for row in summaries])
    acc_high = np.asarray([row["accuracy_delta_bootstrap_ci95_high"] for row in summaries])
    axes[0, 0].errorbar(
        acc,
        y,
        xerr=np.vstack((acc - acc_low, acc_high - acc)),
        fmt="o",
        color="#15803d",
        ecolor="#4b5563",
        capsize=4,
    )
    axes[0, 0].axvline(0.0, color="#b91c1c", linestyle="--", linewidth=1)
    axes[0, 0].set_yticks(y, labels=labels)
    axes[0, 0].set_xlabel("Accuracy gain (percentage points)")
    axes[0, 0].set_title("Paired accuracy gain with bootstrap 95% CI")
    axes[0, 0].grid(axis="x", alpha=0.25)

    length = np.asarray([row["length_delta"] for row in summaries])
    length_low = np.asarray([row["length_delta_bootstrap_ci95_low"] for row in summaries])
    length_high = np.asarray([row["length_delta_bootstrap_ci95_high"] for row in summaries])
    axes[0, 1].errorbar(
        length,
        y,
        xerr=np.vstack((length - length_low, length_high - length)),
        fmt="o",
        color="#2563eb",
        ecolor="#4b5563",
        capsize=4,
    )
    axes[0, 1].axvline(0.0, color="#b91c1c", linestyle="--", linewidth=1)
    axes[0, 1].set_yticks(y, labels=labels)
    axes[0, 1].set_xlabel("Delta #L (candidate - reference; lower is better)")
    axes[0, 1].set_title("Paired length change with bootstrap 95% CI")
    axes[0, 1].grid(axis="x", alpha=0.25)

    rescued = np.asarray([row["rescued"] for row in summaries])
    regressed = np.asarray([row["regressed"] for row in summaries])
    width = 0.36
    axes[1, 0].bar(y - width / 2, rescued, width, label="wrong to correct", color="#15803d")
    axes[1, 0].bar(y + width / 2, regressed, width, label="correct to wrong", color="#dc2626")
    axes[1, 0].set_xticks(y, labels=labels, rotation=15, ha="right")
    axes[1, 0].set_ylabel("Number of questions")
    axes[1, 0].set_title("Paired rescue versus regression counts")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    colors = ("#2563eb", "#15803d", "#d97706", "#7c3aed")
    for color, row in zip(colors, summaries):
        axes[1, 1].annotate(
            "",
            xy=(row["candidate_L"], row["candidate_acc"]),
            xytext=(row["reference_L"], row["reference_acc"]),
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.8},
        )
        axes[1, 1].scatter(row["reference_L"], row["reference_acc"], color=color, marker="o", s=42)
        axes[1, 1].scatter(row["candidate_L"], row["candidate_acc"], color=color, marker="*", s=90)
        axes[1, 1].annotate(row["dataset"], (row["candidate_L"], row["candidate_acc"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1, 1].set_xlabel("#L (lower is better)")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title(f"Paired Pareto movement: {reference_name} to {candidate_name}")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(
        f"{candidate_name}: paired outcome and efficiency evidence",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, summaries, selected, reference_name, candidate_name):
    lines = [
        f"# Paired Evidence: {reference_name} vs {candidate_name}",
        "",
        "Samples are paired by exact question text. Accuracy intervals are question-level bootstrap intervals; p-values are exact paired McNemar tests.",
        "",
        "| Dataset | Ref Acc | Candidate Acc | Delta Acc (95% CI) | Ref #L | Candidate #L | Delta #L (95% CI) | Rescued | Regressed | Exact p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {reference_acc:.2f} | {candidate_acc:.2f} | "
            "{accuracy_delta_pp:+.2f} [{accuracy_delta_bootstrap_ci95_low:+.2f}, {accuracy_delta_bootstrap_ci95_high:+.2f}] | "
            "{reference_L:.2f} | {candidate_L:.2f} | "
            "{length_delta:+.2f} [{length_delta_bootstrap_ci95_low:+.2f}, {length_delta_bootstrap_ci95_high:+.2f}] | "
            "{rescued} | {regressed} | {mcnemar_exact_p:.3g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Objective Case Selection",
            "",
            "Cases are selected by correctness-transition class first, then by proximity to the class-median length change. Geometry is not used for selection.",
            "",
            "| Dataset | Class | Index | Delta #L | Gold | Ref prediction | Candidate prediction | Question |",
            "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in selected:
        question = " ".join(row["question"].split())[:180].replace("|", "\\|")
        lines.append(
            f'| {row["dataset"]} | {row["transition"]} | {row["idx"]} | '
            f'{row["delta_L"]:+.1f} | {row["answer"]} | {row["reference_pred"]} | '
            f'{row["candidate_pred"]} | {question} |'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reference_name", default="TRACE Stage1")
    parser.add_argument("--candidate_name", default="TRACE epoch7")
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--samples_per_class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summaries = []
    selected = []
    all_rows = []
    for label, reference_path, candidate_path in args.dataset:
        rows = paired_rows(load_records(reference_path), load_records(candidate_path))
        for row in rows:
            row["dataset"] = label
        all_rows.extend(rows)
        summaries.append(summarize(label, rows, rng, args.bootstrap_trials))
        selected.extend(select_representative_samples(rows, args.samples_per_class))
        for row in selected:
            row.setdefault("dataset", label)

    (out_dir / "paired_evidence.json").write_text(
        json.dumps(
            {
                "reference": args.reference_name,
                "candidate": args.candidate_name,
                "bootstrap_trials": args.bootstrap_trials,
                "summaries": summaries,
                "selection_policy": (
                    "transition class first; proximity to class-median delta_L second; "
                    "geometry never used"
                ),
                "selected_samples": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / "paired_question_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "dataset",
            "idx",
            "transition",
            "reference_acc",
            "candidate_acc",
            "reference_L",
            "candidate_L",
            "delta_L",
            "answer",
            "reference_pred",
            "candidate_pred",
            "question",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    write_markdown(
        out_dir / "paired_evidence.md",
        summaries,
        selected,
        args.reference_name,
        args.candidate_name,
    )
    plot_dashboard(
        summaries,
        args.reference_name,
        args.candidate_name,
        out_dir / "paired_evidence_dashboard.png",
    )
    print(json.dumps({"summaries": summaries, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
