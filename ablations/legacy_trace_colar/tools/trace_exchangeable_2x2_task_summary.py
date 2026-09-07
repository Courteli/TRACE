#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_DATASETS = ("gsm8k", "gsmhard", "svamp", "multiarith")
CENTRAL_COMPARISONS = {
    "ranking_on_formed_path": ("M10", "M11"),
    "ranking_without_formation": ("M00", "M01"),
    "formation_with_answer_only": ("M00", "M10"),
}


def _parse_cell(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--cell must be LABEL=/absolute/evidence/root")
    label, root = value.split("=", 1)
    return label, Path(root)


def _mean_ci(
    values: Iterable[float],
    *,
    trials: int,
    seed: int,
) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    if array.size == 1:
        value = float(array[0])
        return {"n": 1, "mean": value, "ci95_low": value, "ci95_high": value}
    rng = np.random.default_rng(seed)
    samples = []
    remaining = int(trials)
    while remaining:
        chunk = min(1000, remaining)
        indices = rng.integers(0, array.size, size=(chunk, array.size))
        samples.append(array[indices].mean(axis=1))
        remaining -= chunk
    bootstrap = np.concatenate(samples)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _sign_flip_pvalue(values: Iterable[float], *, trials: int, seed: int) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return 1.0
    observed = float(array.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = int(trials)
    while remaining:
        chunk = min(1000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(chunk, array.size))
        extreme += int(np.count_nonzero((signs * array).mean(axis=1) >= observed))
        remaining -= chunk
    return float((1 + extreme) / (int(trials) + 1))


def _holm_adjust(pvalues: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    family_size = len(ordered)
    for rank, label in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * float(pvalues[label]))
        running = max(running, candidate)
        adjusted[label] = running
    return adjusted


def _find_test_json(root: Path, dataset: str) -> Path:
    dataset_root = root / f"{dataset}_seed271828"
    candidates = sorted(
        dataset_root.rglob("test_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    if not candidates:
        raise FileNotFoundError(f"No test JSON found under {dataset_root}")
    return candidates[-1]


def _load_question_rows(path: Path) -> Dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for key, value in payload.items():
        if not str(key).isdigit() or not isinstance(value, dict):
            continue
        accuracies = value.get("acc")
        output_lengths = value.get("output_length")
        latent_lengths = value.get("n_latent_forward")
        if not all(isinstance(item, list) for item in (accuracies, output_lengths, latent_lengths)):
            continue
        if not (len(accuracies) == len(output_lengths) == len(latent_lengths) == 1):
            raise ValueError(
                f"{path} contains test_times={len(accuracies)} for question {key}; "
                "the frozen protocol requires exactly one test pass"
            )
        rows[int(key)] = {
            "idx": int(key),
            "accuracy": float(accuracies[0]),
            "output_length": float(output_lengths[0]),
            "latent_length": float(latent_lengths[0]),
            "L": float(output_lengths[0]) + float(latent_lengths[0]),
        }
    if not rows:
        raise ValueError(f"No per-question rows found in {path}")
    return rows


def _aggregate(rows: Dict[int, dict], *, trials: int, seed: int) -> dict:
    return {
        metric: _mean_ci(
            [row[metric] for row in rows.values()],
            trials=trials,
            seed=seed + metric_index,
        )
        for metric_index, metric in enumerate(
            ("accuracy", "output_length", "latent_length", "L")
        )
    }


def _paired(
    left: Dict[int, dict],
    right: Dict[int, dict],
    *,
    trials: int,
    seed: int,
) -> dict:
    common = sorted(set(left) & set(right))
    output = {"matched_questions": len(common), "metrics": {}}
    for metric_index, metric in enumerate(("accuracy", "output_length", "L")):
        deltas = [right[index][metric] - left[index][metric] for index in common]
        output["metrics"][metric] = _mean_ci(
            deltas,
            trials=trials,
            seed=seed + metric_index,
        )
        output["metrics"][metric]["paired_signflip_p_one_sided"] = _sign_flip_pvalue(
            deltas,
            trials=trials,
            seed=seed + 100 + metric_index,
        )
    return output


def _interaction(
    rows_by_cell: Dict[str, Dict[int, dict]],
    *,
    trials: int,
    seed: int,
) -> Optional[dict]:
    required = ("M00", "M01", "M10", "M11")
    if not all(cell in rows_by_cell for cell in required):
        return None
    common = sorted(
        set.intersection(*(set(rows_by_cell[cell]) for cell in required))
    )
    output = {
        "definition": "(M11-M10)-(M01-M00)",
        "matched_questions": len(common),
        "metrics": {},
    }
    for metric_index, metric in enumerate(("accuracy", "output_length", "L")):
        values = [
            (
                rows_by_cell["M11"][index][metric]
                - rows_by_cell["M10"][index][metric]
                - rows_by_cell["M01"][index][metric]
                + rows_by_cell["M00"][index][metric]
            )
            for index in common
        ]
        output["metrics"][metric] = _mean_ci(
            values,
            trials=trials,
            seed=seed + metric_index,
        )
        output["metrics"][metric]["paired_signflip_p_one_sided"] = _sign_flip_pvalue(
            values,
            trials=trials,
            seed=seed + 100 + metric_index,
        )
    return output


def _write_csv(path: Path, rows: List[dict]):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict):
    lines = [
        "# TRACE Frozen-Checkpoint Task Summary",
        "",
        "All entries use deterministic center-seed inference (`u=0`) and exactly one test pass.",
        "",
        "| Dataset | Cell | Accuracy | Total #L | Output #L |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset in payload["datasets"]:
        for cell, values in payload["datasets"][dataset]["cells"].items():
            aggregate = values["aggregate"]
            lines.append(
                f"| {dataset} | {cell} | {100.0 * aggregate['accuracy']['mean']:.2f}% | "
                f"{aggregate['L']['mean']:.2f} | {aggregate['output_length']['mean']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Central matched comparison",
            "",
            "| Dataset | M11-M10 accuracy (pp, 95% CI) | M11-M10 #L (95% CI) | Accuracy gate | Length gate |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for dataset, values in payload["datasets"].items():
        central = values["comparisons"].get("ranking_on_formed_path")
        if central is None:
            continue
        accuracy = central["metrics"]["accuracy"]
        length = central["metrics"]["L"]
        gate = values["acceptance"]
        lines.append(
            f"| {dataset} | {100.0 * accuracy['mean']:+.2f} "
            f"[{100.0 * accuracy['ci95_low']:+.2f}, {100.0 * accuracy['ci95_high']:+.2f}] | "
            f"{length['mean']:+.2f} [{length['ci95_low']:+.2f}, {length['ci95_high']:+.2f}] | "
            f"{'PASS' if gate['accuracy_ci_lower_above_zero'] else 'FAIL'} | "
            f"{'PASS' if gate['length_ci_upper_le_tolerance'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Pre-registered total-length tolerance: +{payload['length_tolerance']:.2f} tokens.",
            "A failed gate is a result, not a plotting problem; no checkpoint is selected on test.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", type=_parse_cell, required=True)
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--length_tolerance", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cells = dict(args.cell)
    datasets = args.dataset or list(DEFAULT_DATASETS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "test_times": 1,
            "inference_seed": "u=0",
            "checkpoint_selection": "frozen before test",
            "accuracy_unit": "fraction; markdown reports percentage points",
        },
        "bootstrap_trials": args.bootstrap_trials,
        "length_tolerance": args.length_tolerance,
        "datasets": {},
    }
    flat_rows = []
    central_accuracy_p = {}
    for dataset_index, dataset in enumerate(datasets):
        rows_by_cell = {}
        cell_payload = {}
        for cell_index, (cell, root) in enumerate(cells.items()):
            test_json = _find_test_json(root, dataset)
            rows = _load_question_rows(test_json)
            rows_by_cell[cell] = rows
            aggregate = _aggregate(
                rows,
                trials=args.bootstrap_trials,
                seed=args.seed + 10000 * dataset_index + 100 * cell_index,
            )
            cell_payload[cell] = {
                "evidence_root": str(root),
                "test_json": str(test_json),
                "aggregate": aggregate,
            }
            flat_rows.extend(
                {"dataset": dataset, "cell": cell, **row}
                for row in rows.values()
            )

        comparisons = {}
        for comparison_index, (name, (left, right)) in enumerate(
            CENTRAL_COMPARISONS.items()
        ):
            if left in rows_by_cell and right in rows_by_cell:
                comparisons[name] = _paired(
                    rows_by_cell[left],
                    rows_by_cell[right],
                    trials=args.bootstrap_trials,
                    seed=args.seed
                    + 10000 * dataset_index
                    + 1000 * comparison_index,
                )
        interaction = _interaction(
            rows_by_cell,
            trials=args.bootstrap_trials,
            seed=args.seed + 10000 * dataset_index + 9000,
        )
        central = comparisons.get("ranking_on_formed_path")
        acceptance = None
        if central is not None:
            accuracy = central["metrics"]["accuracy"]
            total_length = central["metrics"]["L"]
            acceptance = {
                "accuracy_ci_lower_above_zero": accuracy["ci95_low"] > 0.0,
                "length_ci_upper_le_tolerance": (
                    total_length["ci95_high"] <= args.length_tolerance
                ),
            }
            central_accuracy_p[dataset] = accuracy[
                "paired_signflip_p_one_sided"
            ]
        payload["datasets"][dataset] = {
            "cells": cell_payload,
            "comparisons": comparisons,
            "factorial_interaction": interaction,
            "acceptance": acceptance,
        }

    adjusted = _holm_adjust(central_accuracy_p)
    payload["central_accuracy_familywise_test"] = {
        dataset: {
            "raw_p": central_accuracy_p[dataset],
            "holm_p": adjusted[dataset],
        }
        for dataset in central_accuracy_p
    }
    _write_csv(args.out_dir / "trace_exchangeable_task_rows.csv", flat_rows)
    (args.out_dir / "trace_exchangeable_2x2_task_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_markdown(
        args.out_dir / "trace_exchangeable_2x2_task_summary.md",
        payload,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
