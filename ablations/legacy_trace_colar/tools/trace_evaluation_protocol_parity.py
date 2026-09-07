#!/usr/bin/env python3
"""Audit TRACE prediction stability across the audit and batched sweep protocols."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-audit", type=Path, required=True)
    parser.add_argument("--stage1-batched", type=Path, required=True)
    parser.add_argument("--final-audit", type=Path, required=True)
    parser.add_argument("--final-batched", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scalar(value, default: float = 0.0) -> float:
    if isinstance(value, list):
        return float(value[0]) if value else default
    return float(value) if value is not None else default


def load(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for value in payload.values():
        if not isinstance(value, dict) or "question" not in value or "acc" not in value:
            continue
        rows[str(value["question"])] = {
            "accuracy": scalar(value.get("acc")),
            "prediction": str(value.get("pred_answer", "")),
            "total_L": scalar(value.get("output_length"))
            + scalar(value.get("n_latent_forward"), 8.0),
        }
    return rows


def compare(label: str, audit_path: Path, batched_path: Path) -> dict:
    audit = load(audit_path)
    batched = load(batched_path)
    questions = sorted(set(audit) & set(batched))
    if set(audit) != set(batched):
        raise ValueError(f"Question sets differ for {label}")
    audit_acc = np.asarray([audit[q]["accuracy"] for q in questions])
    batched_acc = np.asarray([batched[q]["accuracy"] for q in questions])
    audit_length = np.asarray([audit[q]["total_L"] for q in questions])
    batched_length = np.asarray([batched[q]["total_L"] for q in questions])
    return {
        "checkpoint": label,
        "n": len(questions),
        "audit_accuracy_percent": 100.0 * float(audit_acc.mean()),
        "batched_accuracy_percent": 100.0 * float(batched_acc.mean()),
        "batched_minus_audit_accuracy_pp": 100.0 * float((batched_acc - audit_acc).mean()),
        "correctness_agreement_percent": 100.0 * float((batched_acc == audit_acc).mean()),
        "prediction_agreement_percent": 100.0
        * float(np.mean([audit[q]["prediction"] == batched[q]["prediction"] for q in questions])),
        "audit_total_L": float(audit_length.mean()),
        "batched_total_L": float(batched_length.mean()),
        "batched_minus_audit_total_L": float((batched_length - audit_length).mean()),
        "audit_correct_batched_wrong": int(np.sum((audit_acc == 1) & (batched_acc == 0))),
        "audit_wrong_batched_correct": int(np.sum((audit_acc == 0) & (batched_acc == 1))),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        compare("TRACE Stage 1", args.stage1_audit, args.stage1_batched),
        compare("TRACE Final", args.final_audit, args.final_batched),
    ]
    source = args.output_dir / "source_data" / "evaluation_protocol_parity.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headers = ["Checkpoint", "Audit Acc.", "Batched Acc.", "Delta", "Correctness agree", "Prediction agree", "Audit #L", "Batched #L"]
    markdown = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    tex_rows = []
    for row in rows:
        values = [
            row["checkpoint"],
            f'{row["audit_accuracy_percent"]:.2f}',
            f'{row["batched_accuracy_percent"]:.2f}',
            f'{row["batched_minus_audit_accuracy_pp"]:+.2f}',
            f'{row["correctness_agreement_percent"]:.2f}',
            f'{row["prediction_agreement_percent"]:.2f}',
            f'{row["audit_total_L"]:.2f}',
            f'{row["batched_total_L"]:.2f}',
        ]
        markdown.append("| " + " | ".join(values) + " |")
        tex_rows.append(" & ".join(values) + r" \\")
    (args.output_dir / "table_evaluation_protocol_parity.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex = "\n".join(
        [
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Checkpoint & Audit Acc. & Batched Acc. & $\Delta$ & Corr. agree & Pred. agree & Audit \#L & Batched \#L \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (args.output_dir / "table_evaluation_protocol_parity.tex").write_text(latex + "\n", encoding="utf-8")
    payload = {
        "experiment": "evaluation_protocol_parity",
        "audit_batch_size": 1,
        "batched_sweep_batch_size": 8,
        "max_answer_tokens": 48,
        "test_times": 1,
        "rows": rows,
    }
    (args.output_dir / "evaluation_protocol_parity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
