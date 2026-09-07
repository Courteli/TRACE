#!/usr/bin/env python3
"""Paired Stage 1 versus Stage 2 TRACE task and length summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATASETS = ("gsm8k", "gsmhard", "svamp", "multiarith")


def scalar(value) -> float:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Formal TRACE evaluation requires test_times=1")
        value = value[0]
    return float(value)


def load_eval(path: Path) -> dict:
    files = sorted((path / "logs" / "tb" / "run").glob("test_*.json"))
    if not files:
        raise FileNotFoundError(f"No test result in {path}")
    result_path = max(
        files,
        key=lambda item: (item.stat().st_mtime_ns, str(item)),
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if int(payload.get("test_metadata", {}).get("test_times", -1)) != 1:
        raise ValueError(f"{path}: test_times is not exactly 1")
    return {
        int(key): {
            "acc": scalar(record["acc"]),
            "L": scalar(record["n_latent_forward"])
            + scalar(record["output_length"]),
        }
        for key, record in payload.items()
        if str(key).isdigit()
    }


def bootstrap(values, *, trials: int, seed: int):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        values,
        size=(int(trials), len(values)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def summarize_triplet(
    stage1: dict,
    answeronly: dict,
    final: dict,
    *,
    trials: int,
    seed: int,
):
    shared = sorted(set(stage1) & set(answeronly) & set(final))
    stage1_accuracy = float(
        np.mean([stage1[index]["acc"] for index in shared])
    )
    answeronly_accuracy = float(
        np.mean([answeronly[index]["acc"] for index in shared])
    )
    final_accuracy = float(
        np.mean([final[index]["acc"] for index in shared])
    )
    stage1_length = float(
        np.mean([stage1[index]["L"] for index in shared])
    )
    answeronly_length = float(
        np.mean([answeronly[index]["L"] for index in shared])
    )
    final_length = float(
        np.mean([final[index]["L"] for index in shared])
    )
    return {
        "question_count": len(shared),
        "stage1_accuracy": stage1_accuracy,
        "answeronly_accuracy": answeronly_accuracy,
        "final_accuracy": final_accuracy,
        "stage1_L": stage1_length,
        "answeronly_L": answeronly_length,
        "final_L": final_length,
        "answeronly_vs_stage1_accuracy_delta": bootstrap(
            [
                answeronly[index]["acc"] - stage1[index]["acc"]
                for index in shared
            ],
            trials=trials,
            seed=seed,
        ),
        "final_vs_stage1_accuracy_delta": bootstrap(
            [
                final[index]["acc"] - stage1[index]["acc"]
                for index in shared
            ],
            trials=trials,
            seed=seed + 101,
        ),
        "final_vs_answeronly_accuracy_delta": bootstrap(
            [
                final[index]["acc"] - answeronly[index]["acc"]
                for index in shared
            ],
            trials=trials,
            seed=seed + 211,
        ),
        "final_vs_stage1_L_delta": bootstrap(
            [
                final[index]["L"] - stage1[index]["L"]
                for index in shared
            ],
            trials=trials,
            seed=seed + 307,
        ),
        "final_vs_answeronly_L_delta": bootstrap(
            [
                final[index]["L"] - answeronly[index]["L"]
                for index in shared
            ],
            trials=trials,
            seed=seed + 401,
        ),
    }


def write_markdown(path: Path, payload: dict):
    lines = [
        "# TRACE Stage 2 Task Summary",
        "",
        "All comparisons use the same Stage 1 lineage, data, seed, budget, deterministic center-path inference, full test sets, and `test_times=1`. The matched control removes only outcome-local path ranking.",
        "",
        "| Dataset | Stage 1 acc | Answer-only acc | Full TRACE acc | Full - Stage 1 (95% CI) | Full - Answer-only | Stage 1 #L | Answer-only #L | Full #L |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, row in payload["datasets"].items():
        acc = row["final_vs_stage1_accuracy_delta"]
        control = row["final_vs_answeronly_accuracy_delta"]
        lines.append(
            f"| {dataset} | {100 * row['stage1_accuracy']:.2f}% | "
            f"{100 * row['answeronly_accuracy']:.2f}% | "
            f"{100 * row['final_accuracy']:.2f}% | "
            f"{100 * acc['mean']:+.2f} "
            f"[{100 * acc['ci95_low']:+.2f}, "
            f"{100 * acc['ci95_high']:+.2f}] | "
            f"{100 * control['mean']:+.2f} | "
            f"{row['stage1_L']:.2f} | {row['answeronly_L']:.2f} | "
            f"{row['final_L']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Macro accuracy delta: {100 * payload['macro_accuracy_delta']:+.2f} pp.",
            "",
            f"Primary gate (GSM8K accuracy rises and #L grows by no more than 1): **{'PASS' if payload['primary_gate']['pass'] else 'FAIL'}**.",
            "",
            f"Matched-control task gate (Full TRACE is no less accurate than answer-only and adds at most one token): **{'PASS' if payload['matched_control_task_gate']['pass'] else 'FAIL'}**.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    datasets = {}
    for dataset_index, dataset in enumerate(DATASETS):
        stage1 = load_eval(args.evidence_root / "stage1" / dataset)
        answeronly = load_eval(
            args.evidence_root / "answeronly" / dataset
        )
        final = load_eval(args.evidence_root / "final" / dataset)
        datasets[dataset] = summarize_triplet(
            stage1,
            answeronly,
            final,
            trials=args.bootstrap_trials,
            seed=args.seed + dataset_index * 1009,
        )
    macro_delta = float(
        np.mean(
            [
                row["final_accuracy"] - row["stage1_accuracy"]
                for row in datasets.values()
            ]
        )
    )
    gsm = datasets["gsm8k"]
    primary_accuracy = gsm["final_vs_stage1_accuracy_delta"]
    primary_length = gsm["final_vs_stage1_L_delta"]
    primary_gate = {
        "pass": (
            primary_accuracy["ci95_low"] > 0.0
            and primary_length["ci95_high"] <= 1.0
        ),
        "criterion": (
            "GSM8K paired accuracy-gain 95% CI is above zero and the "
            "95% upper bound on mean #L increase is at most 1"
        ),
    }
    matched_control_task_gate = {
        "pass": (
            gsm["final_accuracy"] >= gsm["answeronly_accuracy"]
            and gsm["final_L"] - gsm["answeronly_L"] <= 1.0
        ),
        "criterion": (
            "Full TRACE is no less accurate than the matched answer-only "
            "control and increases mean #L by at most 1"
        ),
    }
    payload = {
        "datasets": datasets,
        "macro_accuracy_delta": macro_delta,
        "primary_gate": primary_gate,
        "matched_control_task_gate": matched_control_task_gate,
        "test_times": 1,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trace_final_task_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "trace_final_task_summary.md", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
