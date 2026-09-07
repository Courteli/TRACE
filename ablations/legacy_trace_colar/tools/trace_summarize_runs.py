#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


KNOWN_DATASETS = ("gsm8k_aug_nl", "gsmhard", "multiarith", "svamp")
EVAL_SIGNATURE_KEYS = (
    "max_n_latent_forward",
    "min_n_latent_forward",
    "compression_factor",
    "latent_temperature",
    "eol_temperature",
)


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def flatten_metric(data, key):
    vals = []
    for sample_key, sample in data.items():
        if sample_key in {"test_result", "test_metadata"} or not isinstance(sample, dict):
            continue
        value = sample.get(key)
        if isinstance(value, list):
            vals.extend(float(v) for v in value if v is not None)
        elif value is not None:
            vals.append(float(value))
    return vals


def infer_dataset_from_path(path):
    name = path.name.lower()
    for dataset in KNOWN_DATASETS:
        if dataset.lower() in name:
            return dataset
    if path.parent.parent == path.parent:
        return "-"
    pair = path.parent.parent.name
    if "-" in pair:
        return pair.split("-")[-1]
    return pair or "-"


def infer_repetitions(data, key="acc"):
    counts = []
    for sample_key, sample in data.items():
        if sample_key in {"test_result", "test_metadata"} or not isinstance(sample, dict):
            continue
        value = sample.get(key)
        if isinstance(value, list):
            counts.append(len(value))
        elif value is not None:
            counts.append(1)
    return max(counts) if counts else None


def eval_signature(metadata):
    latent_config = metadata.get("latent_generation_config", {})
    if not isinstance(latent_config, dict) or not latent_config:
        return ""
    parts = [f"{key}={latent_config.get(key)}" for key in EVAL_SIGNATURE_KEYS if key in latent_config]
    return ",".join(parts)


def summarize_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    acc = flatten_metric(data, "acc")
    latent = flatten_metric(data, "n_latent_forward")
    output_len = flatten_metric(data, "output_length")
    metadata = data.get("test_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    dm = metadata.get("data_module", {}) if isinstance(metadata, dict) else {}
    inferred_test_times = infer_repetitions(data)
    test_times = metadata.get("test_times")
    return {
        "path": str(path),
        "run": path.parent.name,
        "file": path.name,
        "dataset": metadata.get("dataset") or dm.get("dataset_name") or infer_dataset_from_path(path),
        "ckpt_path": metadata.get("ckpt_path", ""),
        "eval_signature": eval_signature(metadata),
        "test_times": metadata.get("test_times"),
        "inferred_test_times": inferred_test_times,
        "effective_test_times": test_times or inferred_test_times,
        "n_items": len(
            [k for k, v in data.items() if isinstance(v, dict) and k not in {"test_result", "test_metadata"}]
        ),
        "n_predictions": len(acc),
        "acc": mean(acc),
        "n_latent_forward": mean(latent),
        "output_length": mean(output_len),
        "mtime": path.stat().st_mtime,
    }


def find_result_files(log_root, include_train=False):
    patterns = ["test_*.json"]
    if include_train:
        patterns.append("train.json")
    files = []
    for pattern in patterns:
        files.extend(log_root.glob(f"**/{pattern}"))
    return sorted(files, key=lambda p: p.stat().st_mtime)


def print_markdown(rows):
    headers = ["run", "file", "dataset", "times", "n_pred", "acc", "#L", "out_len"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row["run"],
                    row["file"],
                    row["dataset"],
                    fmt(row.get("effective_test_times"), 0),
                    str(row["n_predictions"]),
                    fmt(row["acc"]),
                    fmt(row["n_latent_forward"], 2),
                    fmt(row["output_length"], 2),
                ]
            )
            + " |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_root",
        default="logs/trace_colar_qwen3_instruct",
        help="Root containing Lightning run directories.",
    )
    parser.add_argument("--include_train", action="store_true")
    parser.add_argument("--run_contains", default="", help="Only keep run dirs whose path contains this string.")
    parser.add_argument("--latest", type=int, default=0, help="Keep the latest N result files after filtering.")
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    rows = []
    for path in find_result_files(log_root, include_train=args.include_train):
        if args.run_contains and args.run_contains not in str(path.parent):
            continue
        rows.append(summarize_json(path))
    if args.latest > 0:
        rows = rows[-args.latest :]

    print_markdown(rows)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
