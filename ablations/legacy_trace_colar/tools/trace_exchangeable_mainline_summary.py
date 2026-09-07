#!/usr/bin/env python3
"""Paired Stage-1-to-Stage-2 task summary for the final TRACE mainline."""

import argparse
import json
from pathlib import Path

from trace_exchangeable_2x2_task_summary import (
    _aggregate,
    _find_test_json,
    _holm_adjust,
    _load_question_rows,
    _paired,
    _write_csv,
)


EXPECTED_COUNTS = {
    "gsm8k": 1319,
    "gsmhard": 1319,
    "svamp": 1000,
    "multiarith": 180,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_root", type=Path, required=True)
    parser.add_argument("--final_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--length_tolerance", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "comparison": "Final TRACE Stage 2 minus its own Stage 1 checkpoint",
            "training_seeds": 1,
            "test_times": 1,
            "checkpoint_selection": "final monitored checkpoint; no test selection",
        },
        "length_tolerance": args.length_tolerance,
        "datasets": {},
    }
    flat_rows = []
    raw_pvalues = {}
    for dataset_index, (dataset, expected_count) in enumerate(
        EXPECTED_COUNTS.items()
    ):
        stage1_path = _find_test_json(args.stage1_root, dataset)
        final_path = _find_test_json(args.final_root, dataset)
        stage1 = _load_question_rows(stage1_path)
        final = _load_question_rows(final_path)
        if len(stage1) != expected_count or len(final) != expected_count:
            raise ValueError(
                f"{dataset}: expected {expected_count} full-test rows, "
                f"found Stage1={len(stage1)}, Final={len(final)}"
            )
        comparison = _paired(
            stage1,
            final,
            trials=args.bootstrap_trials,
            seed=args.seed + 10000 * dataset_index,
        )
        accuracy = comparison["metrics"]["accuracy"]
        total_length = comparison["metrics"]["L"]
        raw_pvalues[dataset] = accuracy["paired_signflip_p_one_sided"]
        payload["datasets"][dataset] = {
            "stage1": {
                "test_json": str(stage1_path),
                "aggregate": _aggregate(
                    stage1,
                    trials=args.bootstrap_trials,
                    seed=args.seed + 10000 * dataset_index + 1000,
                ),
            },
            "final": {
                "test_json": str(final_path),
                "aggregate": _aggregate(
                    final,
                    trials=args.bootstrap_trials,
                    seed=args.seed + 10000 * dataset_index + 2000,
                ),
            },
            "final_minus_stage1": comparison,
            "acceptance": {
                "accuracy_ci_lower_above_zero": accuracy["ci95_low"] > 0.0,
                "length_ci_upper_le_tolerance": (
                    total_length["ci95_high"] <= args.length_tolerance
                ),
            },
        }
        for label, rows in (("Stage1", stage1), ("TRACE", final)):
            flat_rows.extend(
                {"dataset": dataset, "checkpoint": label, **row}
                for row in rows.values()
            )

    holm = _holm_adjust(raw_pvalues)
    payload["accuracy_familywise_test"] = {
        dataset: {"raw_p": raw_pvalues[dataset], "holm_p": holm[dataset]}
        for dataset in raw_pvalues
    }
    _write_csv(args.out_dir / "trace_mainline_task_rows.csv", flat_rows)
    (args.out_dir / "trace_mainline_task_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# TRACE Mainline: Stage 1 to Stage 2",
        "",
        "One training seed and one complete deterministic test pass per checkpoint.",
        "",
        "| Dataset | Stage 1 Acc. | Final Acc. | Delta Acc. (pp, 95% CI) | "
        "Stage 1 #L | Final #L | Delta #L (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, values in payload["datasets"].items():
        stage1 = values["stage1"]["aggregate"]
        final = values["final"]["aggregate"]
        delta_acc = values["final_minus_stage1"]["metrics"]["accuracy"]
        delta_length = values["final_minus_stage1"]["metrics"]["L"]
        lines.append(
            f"| {dataset} | {100 * stage1['accuracy']['mean']:.2f}% | "
            f"{100 * final['accuracy']['mean']:.2f}% | "
            f"{100 * delta_acc['mean']:+.2f} "
            f"[{100 * delta_acc['ci95_low']:+.2f}, "
            f"{100 * delta_acc['ci95_high']:+.2f}] | "
            f"{stage1['L']['mean']:.2f} | {final['L']['mean']:.2f} | "
            f"{delta_length['mean']:+.2f} "
            f"[{delta_length['ci95_low']:+.2f}, "
            f"{delta_length['ci95_high']:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "The task comparison uses every test question. The separate "
            "200-question caches are used only for matched path geometry.",
        ]
    )
    (args.out_dir / "trace_mainline_task_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
