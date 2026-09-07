#!/usr/bin/env python3
"""Build the paired GSM8K/OOD accuracy and total-#L main table."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


DATASETS = {
    "gsm8k": ("gsm8k_geometry200", 1319),
    "gsmhard": ("gsmhard", 1319),
    "svamp": ("svamp", 1000),
    "multiarith": ("multiarith", 180),
}


def scalar(value) -> float:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("formal evaluation requires exactly one replication")
    return float(value[0])


def load_eval(
    directory: Path,
    *,
    expected_count: int,
    expected_checkpoint: Path,
) -> Dict[int, dict]:
    files = sorted(directory.rglob("test_*.json"))
    if len(files) != 1:
        raise ValueError(
            f"{directory} must contain exactly one test JSON, found {len(files)}"
        )
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    metadata = payload.get("test_metadata", {})
    if int(metadata.get("test_times", -1)) != 1:
        raise ValueError(f"{files[0]} was not evaluated with test_times=1")
    actual_checkpoint = Path(str(metadata.get("ckpt_path", ""))).resolve()
    if actual_checkpoint != expected_checkpoint.resolve():
        raise ValueError(
            f"{files[0]} uses {actual_checkpoint}, expected "
            f"{expected_checkpoint.resolve()}"
        )

    rows = {}
    for key, record in payload.items():
        if not str(key).isdigit():
            continue
        index = int(key)
        latent = scalar(record["n_latent_forward"])
        if latent != 8.0:
            raise ValueError(f"{files[0]} question {index} has {latent} latents")
        output_length = scalar(record["output_length"])
        rows[index] = {
            "accuracy": scalar(record["acc"]),
            "latent_length": latent,
            "output_length": output_length,
            "total_L": latent + output_length,
        }
    if len(rows) != int(expected_count):
        raise ValueError(
            f"{files[0]} has {len(rows)} questions, expected {expected_count}"
        )
    if sorted(rows) != list(range(int(expected_count))):
        raise ValueError(f"{files[0]} question indices are incomplete")
    return rows


def bootstrap_mean(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(
        0,
        len(values),
        size=(int(draws), len(values)),
    )
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def exact_sign_p(rescued: int, regressed: int) -> float:
    discordant = int(rescued) + int(regressed)
    if discordant == 0:
        return 1.0
    tail = min(int(rescued), int(regressed))
    probability = sum(
        math.comb(discordant, index)
        for index in range(tail + 1)
    ) / (2.0**discordant)
    return float(min(1.0, 2.0 * probability))


def summarize_pair(
    stage1: Dict[int, dict],
    final: Dict[int, dict],
    *,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    if sorted(stage1) != sorted(final):
        raise ValueError("Stage-1 and final evaluations are not question-paired")
    indices = sorted(stage1)
    stage1_accuracy = np.asarray(
        [stage1[index]["accuracy"] for index in indices],
        dtype=np.float64,
    )
    final_accuracy = np.asarray(
        [final[index]["accuracy"] for index in indices],
        dtype=np.float64,
    )
    stage1_length = np.asarray(
        [stage1[index]["total_L"] for index in indices],
        dtype=np.float64,
    )
    final_length = np.asarray(
        [final[index]["total_L"] for index in indices],
        dtype=np.float64,
    )
    rescued = int(((stage1_accuracy == 0) & (final_accuracy == 1)).sum())
    regressed = int(((stage1_accuracy == 1) & (final_accuracy == 0)).sum())
    return {
        "questions": len(indices),
        "latent_length": 8,
        "stage1_accuracy": bootstrap_mean(
            stage1_accuracy,
            rng=rng,
            draws=draws,
        ),
        "final_accuracy": bootstrap_mean(
            final_accuracy,
            rng=rng,
            draws=draws,
        ),
        "accuracy_delta": bootstrap_mean(
            final_accuracy - stage1_accuracy,
            rng=rng,
            draws=draws,
        ),
        "stage1_total_L": bootstrap_mean(
            stage1_length,
            rng=rng,
            draws=draws,
        ),
        "final_total_L": bootstrap_mean(
            final_length,
            rng=rng,
            draws=draws,
        ),
        "total_L_delta": bootstrap_mean(
            final_length - stage1_length,
            rng=rng,
            draws=draws,
        ),
        "rescued": rescued,
        "regressed": regressed,
        "exact_sign_p": exact_sign_p(rescued, regressed),
    }


def write_csv(path: Path, rows: dict):
    fields = (
        "dataset",
        "questions",
        "stage1_accuracy",
        "final_accuracy",
        "accuracy_delta",
        "accuracy_delta_ci95_low",
        "accuracy_delta_ci95_high",
        "stage1_total_L",
        "final_total_L",
        "total_L_delta",
        "total_L_delta_ci95_low",
        "total_L_delta_ci95_high",
        "rescued",
        "regressed",
        "exact_sign_p",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, row in rows.items():
            writer.writerow(
                {
                    "dataset": dataset,
                    "questions": row["questions"],
                    "stage1_accuracy": row["stage1_accuracy"]["mean"],
                    "final_accuracy": row["final_accuracy"]["mean"],
                    "accuracy_delta": row["accuracy_delta"]["mean"],
                    "accuracy_delta_ci95_low": row["accuracy_delta"]["ci95"][0],
                    "accuracy_delta_ci95_high": row["accuracy_delta"]["ci95"][1],
                    "stage1_total_L": row["stage1_total_L"]["mean"],
                    "final_total_L": row["final_total_L"]["mean"],
                    "total_L_delta": row["total_L_delta"]["mean"],
                    "total_L_delta_ci95_low": row["total_L_delta"]["ci95"][0],
                    "total_L_delta_ci95_high": row["total_L_delta"]["ci95"][1],
                    "rescued": row["rescued"],
                    "regressed": row["regressed"],
                    "exact_sign_p": row["exact_sign_p"],
                }
            )


def markdown_row(dataset: str, row: dict) -> str:
    accuracy = row["accuracy_delta"]
    length = row["total_L_delta"]
    return (
        f"| {dataset} | {row['questions']} | "
        f"{100 * row['stage1_accuracy']['mean']:.2f} | "
        f"{100 * row['final_accuracy']['mean']:.2f} | "
        f"{100 * accuracy['mean']:+.2f} "
        f"[{100 * accuracy['ci95'][0]:+.2f}, "
        f"{100 * accuracy['ci95'][1]:+.2f}] | "
        f"{row['stage1_total_L']['mean']:.2f} | "
        f"{row['final_total_L']['mean']:.2f} | "
        f"{length['mean']:+.2f} "
        f"[{length['ci95'][0]:+.2f}, {length['ci95'][1]:+.2f}] | "
        f"{row['rescued']}/{row['regressed']} | "
        f"{row['exact_sign_p']:.3g} |"
    )


def write_markdown(path: Path, report: dict):
    lines = [
        "# TRACE Policy Task Summary",
        "",
        "All rows compare validation-selected checkpoints once on paired full "
        "test sets. Total #L is eight latent states plus generated answer tokens.",
        "",
        "| Dataset | N | Stage 1 acc. (%) | Final acc. (%) | Delta acc. "
        "(pp, 95% CI) | Stage 1 #L | Final #L | Delta #L (95% CI) | "
        "Rescue/regress | Exact p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    lines.extend(
        markdown_row(dataset, row)
        for dataset, row in report["datasets"].items()
    )
    gate = report["primary_gate"]
    lines.extend(
        [
            "",
            f"Primary GSM8K gate: **{'PASS' if gate['pass'] else 'FAIL'}**.",
            "",
            gate["criterion"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: dict):
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Stage 1 & Final & $\Delta$ Acc. & Stage 1 \#L "
        r"& Final \#L & $\Delta$\#L \\",
        r"\midrule",
    ]
    for dataset, row in rows.items():
        lines.append(
            f"{dataset} & "
            f"{100 * row['stage1_accuracy']['mean']:.2f} & "
            f"{100 * row['final_accuracy']['mean']:.2f} & "
            f"{100 * row['accuracy_delta']['mean']:+.2f} & "
            f"{row['stage1_total_L']['mean']:.2f} & "
            f"{row['final_total_L']['mean']:.2f} & "
            f"{row['total_L_delta']['mean']:+.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260719)
    summaries = {}
    for dataset, (directory_suffix, expected_count) in DATASETS.items():
        stage1 = load_eval(
            args.evidence_root / f"stage1_{directory_suffix}",
            expected_count=expected_count,
            expected_checkpoint=args.stage1_checkpoint,
        )
        final = load_eval(
            args.evidence_root / f"final_{directory_suffix}",
            expected_count=expected_count,
            expected_checkpoint=args.final_checkpoint,
        )
        summaries[dataset] = summarize_pair(
            stage1,
            final,
            rng=rng,
            draws=args.bootstrap,
        )

    gsm8k = summaries["gsm8k"]
    primary_gate = {
        "pass": (
            gsm8k["accuracy_delta"]["ci95"][0] > 0.0
            and gsm8k["total_L_delta"]["ci95"][1] <= 1.0
        ),
        "criterion": (
            "The paired GSM8K accuracy-gain 95% CI must be above zero, "
            "and the 95% upper bound on total-#L increase must be at most 1."
        ),
    }
    report = {
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "final_checkpoint": str(args.final_checkpoint.resolve()),
        "test_times": 1,
        "total_L_definition": "8 latent states + generated answer tokens",
        "datasets": summaries,
        "macro_accuracy_delta": float(
            np.mean(
                [
                    row["accuracy_delta"]["mean"]
                    for row in summaries.values()
                ]
            )
        ),
        "primary_gate": primary_gate,
    }
    (args.output_dir / "task_summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "task_summary.csv", summaries)
    write_markdown(args.output_dir / "task_summary.md", report)
    write_latex(args.output_dir / "task_summary.tex", summaries)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
