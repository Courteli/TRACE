#!/usr/bin/env python3
"""Create an accuracy-only TRACE/SemCoT same-backbone context table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semcot-gsm8k", type=Path, required=True)
    parser.add_argument("--semcot-gsmhard-a", type=Path, required=True)
    parser.add_argument("--semcot-gsmhard-b", type=Path, required=True)
    parser.add_argument("--semcot-svamp", type=Path, required=True)
    parser.add_argument("--semcot-multiarith", type=Path, required=True)
    parser.add_argument("--stagewise", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-table", type=Path)
    return parser.parse_args()


def load_last_json_line(path: Path) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gsm8k = load_last_json_line(args.semcot_gsm8k)
    hard_a = load_last_json_line(args.semcot_gsmhard_a)
    hard_b = load_last_json_line(args.semcot_gsmhard_b)
    svamp = load_last_json_line(args.semcot_svamp)
    multiarith = load_last_json_line(args.semcot_multiarith)
    hard_n_a = int(hard_a["num_eval"])
    hard_n_b = int(hard_b["num_eval"])
    hard_accuracy = (
        hard_n_a * float(hard_a["numerical_accuracy"])
        + hard_n_b * float(hard_b["numerical_accuracy"])
    ) / (hard_n_a + hard_n_b)

    paired = json.loads(args.stagewise.read_text(encoding="utf-8"))
    summaries = {row["dataset"]: row for row in paired["summaries"]}
    rows = [
        {
            "method": "SemCoT (40-token local adaptation)",
            "gsm8k_accuracy_percent": 100.0 * float(gsm8k["numerical_accuracy"]),
            "gsmhard_accuracy_percent": 100.0 * hard_accuracy,
            "svamp_accuracy_percent": 100.0 * float(svamp["numerical_accuracy"]),
            "multiarith_accuracy_percent": 100.0 * float(multiarith["numerical_accuracy"]),
        },
        {
            "method": "TRACE Stage 1",
            "gsm8k_accuracy_percent": summaries["GSM8K"]["reference_acc"],
            "gsmhard_accuracy_percent": summaries["GSMHard"]["reference_acc"],
            "svamp_accuracy_percent": summaries["SVAMP"]["reference_acc"],
            "multiarith_accuracy_percent": summaries["MultiArith"]["reference_acc"],
        },
        {
            "method": "TRACE Final",
            "gsm8k_accuracy_percent": summaries["GSM8K"]["candidate_acc"],
            "gsmhard_accuracy_percent": summaries["GSMHard"]["candidate_acc"],
            "svamp_accuracy_percent": summaries["SVAMP"]["candidate_acc"],
            "multiarith_accuracy_percent": summaries["MultiArith"]["candidate_acc"],
        },
    ]
    for row in rows:
        row["macro_accuracy_percent"] = sum(
            row[key]
            for key in (
                "gsm8k_accuracy_percent",
                "gsmhard_accuracy_percent",
                "svamp_accuracy_percent",
                "multiarith_accuracy_percent",
            )
        ) / 4.0

    source = args.output_dir / "source_data" / "semcot_accuracy_context.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| Method | GSM8K | GSMHard | SVAMP | MultiArith | Macro avg. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    tex_rows = []
    for row in rows:
        values = [
            row["method"],
            f'{row["gsm8k_accuracy_percent"]:.2f}',
            f'{row["gsmhard_accuracy_percent"]:.2f}',
            f'{row["svamp_accuracy_percent"]:.2f}',
            f'{row["multiarith_accuracy_percent"]:.2f}',
            f'{row["macro_accuracy_percent"]:.2f}',
        ]
        markdown.append("| " + " | ".join(values) + " |")
        latex_label = {
            "SemCoT (40-token local adaptation)": r"SemCoT (local, max 40)",
            "TRACE Stage 1": r"\trace Stage~1",
            "TRACE Final": r"\textbf{\trace Final}",
        }[row["method"]]
        tex_values = [latex_label, *values[1:]]
        if row["method"] == "TRACE Final":
            tex_values[1:] = [rf"\textbf{{{value}}}" for value in tex_values[1:]]
        tex_rows.append(" & ".join(tex_values) + r" \\")
    (args.output_dir / "table_semcot_accuracy_context.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    latex = "\n".join(
        [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Same-backbone accuracy-only context. The SemCoT local adaptation permits at most 40 contemplation tokens; its length semantics differ from TRACE, so this table is not used for paired accuracy--length claims.}",
            r"\label{tab:semcot-context}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.8pt}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Method & GSM8K & GSMHard & SVAMP & MultiArith & Avg. \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
        ]
    )
    (args.output_dir / "table_semcot_accuracy_context.tex").write_text(latex + "\n", encoding="utf-8")
    if args.paper_table:
        args.paper_table.parent.mkdir(parents=True, exist_ok=True)
        args.paper_table.write_text(latex + "\n", encoding="utf-8")
    payload = {
        "experiment": "same-backbone accuracy-only context",
        "semcot_max_contemplation_tokens": 40,
        "length_comparison_allowed": False,
        "rows": rows,
    }
    (args.output_dir / "semcot_accuracy_context.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
