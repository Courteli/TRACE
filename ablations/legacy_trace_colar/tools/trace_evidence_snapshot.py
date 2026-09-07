#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

from trace_compare_diagnostics import summarize as summarize_diagnostic
from trace_gate_report import load_run
from trace_summarize_runs import find_result_files, fmt, summarize_json


DEFAULT_RESULT_ROOTS = [
    "logs/cot_qwen3_instruct",
    "logs/colar_qwen3_instruct",
    "logs/trace_colar_qwen3_instruct",
]
REQUIRED_TRACE_OOD_DATASETS = {"gsm8k_aug_nl", "gsmhard", "svamp", "multiarith"}
TRACE_OOD_MIN_ITEMS = {
    "gsm8k_aug_nl": 1000,
    "gsmhard": 1000,
    "svamp": 900,
    "multiarith": 150,
}


def load_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def collect_results(log_roots, latest_per_root, include_train):
    rows = []
    for root in log_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        paths = find_result_files(root_path, include_train=include_train)
        if latest_per_root > 0:
            paths = paths[-latest_per_root:]
        for path in paths:
            try:
                row = summarize_json(path)
            except (json.JSONDecodeError, OSError, ValueError):
                continue
            row["root"] = str(root_path)
            rows.append(row)
    return sorted(rows, key=lambda row: row["mtime"])


def collect_diagnostics(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(Path().glob(pattern))
    rows = []
    for path in sorted(set(paths), key=lambda p: p.stat().st_mtime):
        try:
            row = summarize_diagnostic(path)
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            continue
        row["path"] = str(path)
        row["mtime"] = path.stat().st_mtime
        rows.append(row)
    return rows


def collect_gate_rows(log_root: str, latest: int):
    root = Path(log_root)
    if not root.exists():
        return []
    run_dirs = [p for p in root.glob("*") if p.is_dir()]
    rows = [load_run(run_dir) for run_dir in sorted(run_dirs, key=lambda p: p.stat().st_mtime)]
    if latest > 0:
        rows = rows[-latest:]
    return rows


def status_from_gate(gate_rows, result_rows):
    active_trace = [
        row
        for row in gate_rows
        if "trace_v2_rl" in row.get("run", "") and row.get("gate", {}).get("status") in {"observed", "weak"}
    ]
    answered = [row for row in gate_rows if "answer_only" in row.get("run", "") and row.get("checkpoint", {}).get("count", 0)]
    full_ood_by_eval = {}
    for row in result_rows:
        if "trace_colar_qwen3_instruct" not in row.get("root", ""):
            continue
        run = row.get("run", "")
        if "trace_v2" not in run:
            continue
        if not str(row.get("file", "")).startswith("test_"):
            continue
        ckpt_path = row.get("ckpt_path") or ""
        if not ckpt_path or not str(ckpt_path).endswith(".ckpt"):
            continue
        if (row.get("effective_test_times") or row.get("test_times") or 0) < 5:
            continue
        dataset = row.get("dataset")
        if (row.get("n_items") or 0) < TRACE_OOD_MIN_ITEMS.get(dataset, 500):
            continue
        if dataset in REQUIRED_TRACE_OOD_DATASETS:
            eval_signature = row.get("eval_signature") or "legacy_default"
            eval_key = f"{run}::{ckpt_path}::{eval_signature}"
            full_ood_by_eval.setdefault(eval_key, set()).add(dataset)
    full_ood_runs = {
        eval_key: sorted(datasets)
        for eval_key, datasets in full_ood_by_eval.items()
        if REQUIRED_TRACE_OOD_DATASETS.issubset(datasets)
    }
    return {
        "trace_v2_active_or_observed": bool(active_trace),
        "answer_only_checkpoint_available": bool(answered),
        "trace_test_available": bool(full_ood_runs),
        "trace_full_ood_by_eval": {eval_key: sorted(datasets) for eval_key, datasets in full_ood_by_eval.items()},
        "trace_full_ood_runs": full_ood_runs,
        "trace_ood_required_datasets": sorted(REQUIRED_TRACE_OOD_DATASETS),
        "missing": [
            name
            for name, ok in [
                ("answer_only_checkpoint_available", bool(answered)),
                ("trace_test_available", bool(full_ood_runs)),
            ]
            if not ok
        ],
    }


def write_markdown(snapshot, path):
    lines = []
    lines.append("# TRACE Evidence Snapshot")
    lines.append("")
    lines.append("## Gate Status")
    lines.append("")
    lines.append("| run | gate | monitor | grad_nz | trace_nz | ckpt | notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in snapshot["gate_rows"]:
        tags = row.get("tags", {})
        monitor = row.get("checkpoint", {}).get("best_monitor")
        if monitor is None:
            monitor = tags.get("monitor", {}).get("last")
        grad = tags.get("train/grad_norm", {}).get("nonzero_frac")
        trace = tags.get("train/trace/bonus", {}).get("nonzero_frac")
        ckpt = row.get("checkpoint", {}).get("count", 0)
        gate = row.get("gate", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("run", "-"),
                    gate.get("status", "-"),
                    fmt(monitor, 3),
                    fmt(grad, 3),
                    fmt(trace, 3),
                    str(ckpt),
                    gate.get("notes", "-") or "-",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Result Files")
    lines.append("")
    lines.append("| root | run | file | dataset | times | n_pred | acc | #L | out_len |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in snapshot["result_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("root", "-"),
                    row.get("run", "-"),
                    row.get("file", "-"),
                    row.get("dataset", "-"),
                    fmt(row.get("effective_test_times"), 0),
                    str(row.get("n_predictions", "-")),
                    fmt(row.get("acc")),
                    fmt(row.get("n_latent_forward"), 2),
                    fmt(row.get("output_length"), 2),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append("| file | n | G | mixed_frac | acc | raw_pn | ctr_pn | raw_pp | ctr_pp |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in snapshot["diagnostic_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("file", "-"),
                    fmt(row.get("n"), 0),
                    fmt(row.get("group_size"), 0),
                    fmt(row.get("mixed_frac")),
                    fmt(row.get("group_acc")),
                    fmt(row.get("pos_neg_sim")),
                    fmt(row.get("centered_pos_neg_sim")),
                    fmt(row.get("pos_pos_sim")),
                    fmt(row.get("centered_pos_pos_sim")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Missing Evidence")
    lines.append("")
    missing = snapshot["status"].get("missing", [])
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", action="append", default=None)
    parser.add_argument("--latest_per_root", type=int, default=40)
    parser.add_argument("--include_train", action="store_true")
    parser.add_argument("--gate_json", default="run_outputs/trace/gate_report_latest.json")
    parser.add_argument(
        "--gate_log_root",
        default="logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl",
    )
    parser.add_argument("--gate_latest", type=int, default=80)
    parser.add_argument("--no_refresh_gate", action="store_true")
    parser.add_argument(
        "--diagnostics_glob",
        action="append",
        default=["run_outputs/trace/diagnostics/*.json"],
    )
    parser.add_argument("--json_out", default="run_outputs/trace/evidence_snapshot.json")
    parser.add_argument("--md_out", default="run_outputs/trace/evidence_snapshot.md")
    args = parser.parse_args()

    log_roots = args.log_root if args.log_root is not None else DEFAULT_RESULT_ROOTS
    gate_path = Path(args.gate_json)
    if args.no_refresh_gate:
        gate_rows = load_json(gate_path, [])
    else:
        gate_rows = collect_gate_rows(args.gate_log_root, args.gate_latest)
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(gate_rows, indent=2), encoding="utf-8")
    result_rows = collect_results(log_roots, args.latest_per_root, args.include_train)
    snapshot = {
        "gate_json": args.gate_json,
        "gate_rows": gate_rows,
        "status": status_from_gate(gate_rows, result_rows),
        "result_rows": result_rows,
        "diagnostic_rows": collect_diagnostics(args.diagnostics_glob),
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    write_markdown(snapshot, Path(args.md_out))
    print(f"wrote {json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
