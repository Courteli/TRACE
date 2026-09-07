#!/usr/bin/env python3
"""Paired causal audit for TRACE's answer-stage path bottleneck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scalar(value) -> float:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                f"Expected test_times=1, found {len(value)} values"
            )
        value = value[0]
    return float(value)


def load_condition(path: Path) -> dict:
    log_dir = path / "logs" / "tb" / "run"
    files = sorted(log_dir.glob("test_*.json"))
    if not files:
        raise FileNotFoundError(f"No test JSON under {log_dir}")
    result_path = max(
        files,
        key=lambda item: (item.stat().st_mtime_ns, str(item)),
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = payload.get("test_metadata", {})
    if int(metadata.get("test_times", -1)) != 1:
        raise ValueError(f"{path.name}: test_times is not exactly 1")
    records = {
        int(key): {
            "acc": scalar(value["acc"]),
            "output_length": scalar(value["output_length"]),
        }
        for key, value in payload.items()
        if str(key).isdigit()
    }
    return {
        "result_path": str(result_path),
        "records": records,
    }


def bootstrap_delta(
    full: dict,
    condition: dict,
    *,
    trials: int,
    seed: int,
) -> dict:
    shared = sorted(set(full) & set(condition))
    delta = np.asarray(
        [full[index]["acc"] - condition[index]["acc"] for index in shared],
        dtype=np.float64,
    )
    length_delta = np.asarray(
        [
            full[index]["output_length"]
            - condition[index]["output_length"]
            for index in shared
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        delta,
        size=(int(trials), len(delta)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "matched_questions": len(shared),
        "condition_accuracy": float(
            np.mean([condition[index]["acc"] for index in shared])
        ),
        "full_minus_condition_accuracy": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "full_minus_condition_output_length": float(length_delta.mean()),
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# TRACE Causal Path Audit",
        "",
        "All rows are paired on the same 200 GSM8K questions with `test_times=1`. Positive delta means the intact eight-state path is more accurate.",
        "",
        "| Intervention | Accuracy | Full - intervention | 95% CI |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, row in payload["conditions"].items():
        lines.append(
            f"| {label} | {100 * row['condition_accuracy']:.2f}% | "
            f"{100 * row['full_minus_condition_accuracy']:+.2f} | "
            f"[{100 * row['ci95_low']:+.2f}, "
            f"{100 * row['ci95_high']:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Claim gates",
            "",
            "| Claim | Status | Criterion |",
            "| --- | --- | --- |",
        ]
    )
    for label, gate in payload["claim_gates"].items():
        lines.append(
            f"| {label} | {'PASS' if gate['pass'] else 'FAIL'} | "
            f"{gate['criterion']} |"
        )
    lines.extend(
        [
            "",
            "A failed per-state gate narrows the claim; it is not replaced by a qualitative 3D figure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    condition_dirs = sorted(
        path
        for path in args.suite_root.iterdir()
        if path.is_dir() and (path / "manifest.txt").is_file()
    )
    loaded = {
        path.name: load_condition(path)
        for path in condition_dirs
    }
    if "full" not in loaded:
        raise ValueError("The causal suite must contain an intact `full` run")
    full = loaded["full"]["records"]
    if len(full) != 200:
        raise ValueError(f"Expected 200 full-path records, found {len(full)}")

    reports = {}
    for condition_index, (label, item) in enumerate(loaded.items()):
        if label == "full":
            continue
        reports[label] = bootstrap_delta(
            full,
            item["records"],
            trials=args.bootstrap_trials,
            seed=args.seed + condition_index,
        )

    required = [
        "no_path",
        "reverse",
        "shuffle",
        "mean_repeat",
        "random_direction",
        "same_norm_random_path",
        "same_question_swap",
        "cross_question_swap",
    ]
    missing = [label for label in required if label not in reports]
    if missing:
        raise ValueError(f"Missing causal conditions: {missing}")
    drops = [reports[f"drop_z{index}"] for index in range(8)]
    replacements = [
        reports[f"replace_t{index}"] for index in range(8)
    ]
    prefix_accuracy = {
        0: reports["no_path"]["condition_accuracy"],
        8: float(np.mean([record["acc"] for record in full.values()])),
    }
    for index in range(1, 8):
        prefix_accuracy[index] = reports[
            f"prefix_{index}"
        ]["condition_accuracy"]
    prefix_values = np.asarray(
        [prefix_accuracy[index] for index in range(9)],
        dtype=np.float64,
    )
    same_vs_cross = bootstrap_delta(
        loaded["same_question_swap"]["records"],
        loaded["cross_question_swap"]["records"],
        trials=args.bootstrap_trials,
        seed=args.seed + 100003,
    )

    claim_gates = {
        "Path is a necessary answer mediator": {
            "pass": reports["no_path"]["ci95_low"] > 0.0,
            "criterion": "95% CI for full - no-path is strictly positive",
        },
        "Ordered path structure is used": {
            "pass": (
                reports["reverse"]["ci95_low"] > 0.0
                and reports["shuffle"]["ci95_low"] > 0.0
            ),
            "criterion": (
                "Both reverse and shuffle have strictly positive paired "
                "95% lower bounds"
            ),
        },
        "All eight states are individually necessary": {
            "pass": all(row["ci95_low"] > 0.0 for row in drops),
            "criterion": (
                "Every one-state answer-readout deletion has a strictly "
                "positive paired 95% lower bound"
            ),
        },
        "All eight transition directions matter": {
            "pass": all(
                row["ci95_low"] > 0.0 for row in replacements
            ),
            "criterion": (
                "Every same-norm transition replacement has a strictly "
                "positive paired 95% lower bound"
            ),
        },
        "More ordered prefix states improve accuracy": {
            "pass": (
                np.corrcoef(np.arange(9), prefix_values)[0, 1] > 0.8
                and prefix_values[-1] > prefix_values[0]
            ),
            "criterion": (
                "Pearson r(k, accuracy) > 0.8 and full exceeds no-path"
            ),
        },
        "Path content is question-specific": {
            "pass": (
                reports["cross_question_swap"]["ci95_low"] > 0.0
                and same_vs_cross["ci95_low"] > 0.0
            ),
            "criterion": (
                "Cross-question path replacement hurts intact accuracy and "
                "same-question donor paths outperform cross-question donors"
            ),
        },
        "Path effect is not a norm artifact": {
            "pass": reports["same_norm_random_path"]["ci95_low"] > 0.0,
            "criterion": (
                "The paired 95% lower bound for full minus a same-norm "
                "random path is strictly positive"
            ),
        },
    }
    payload = {
        "full_accuracy": float(
            np.mean([record["acc"] for record in full.values()])
        ),
        "question_count": len(full),
        "test_times": 1,
        "prefix_accuracy": {
            str(index): prefix_accuracy[index] for index in range(9)
        },
        "prefix_accuracy_pearson_r": float(
            np.corrcoef(np.arange(9), prefix_values)[0, 1]
        ),
        "conditions": reports,
        "same_question_minus_cross_question": same_vs_cross,
        "claim_gates": claim_gates,
        "interpretation_rule": (
            "Only passed gates may be stated as established claims; failed "
            "state-level gates must narrow the wording."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trace_final_causal_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.out_dir / "trace_final_causal_summary.md",
        payload,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
