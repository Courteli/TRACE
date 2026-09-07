#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text())
    acc = []
    lengths = []
    output_lengths = []
    n_items = 0
    for key, value in payload.items():
        if not isinstance(value, dict) or "acc" not in value:
            continue
        n_items += 1
        item_acc = value.get("acc", [])
        item_l = value.get("n_latent_forward", [])
        item_out = value.get("output_length", [])
        if not isinstance(item_acc, list):
            item_acc = [item_acc]
        if not isinstance(item_l, list):
            item_l = [item_l]
        if not isinstance(item_out, list):
            item_out = [item_out]
        acc.extend(float(x) for x in item_acc)
        lengths.extend(float(x) for x in item_l)
        output_lengths.extend(float(x) for x in item_out)

    metadata = payload.get("test_metadata", {})
    latent_cfg = metadata.get("latent_generation_config", {}) if isinstance(metadata, dict) else {}
    answer_cfg = metadata.get("answer_generation_config", {}) if isinstance(metadata, dict) else {}
    data_cfg = metadata.get("data_module", {}) if isinstance(metadata, dict) else {}
    return {
        "path": str(path),
        "dataset": data_cfg.get("dataset_name"),
        "test_file": data_cfg.get("test_file"),
        "n_items": n_items,
        "n_predictions": len(acc),
        "acc": sum(acc) / len(acc) if acc else None,
        "avg_L": sum(lengths) / len(lengths) if lengths else None,
        "avg_output_len": sum(output_lengths) / len(output_lengths) if output_lengths else None,
        "max_L": latent_cfg.get("max_n_latent_forward"),
        "min_L": latent_cfg.get("min_n_latent_forward"),
        "latent_temperature": latent_cfg.get("latent_temperature"),
        "eol_temperature": latent_cfg.get("eol_temperature"),
        "compression_factor": latent_cfg.get("compression_factor"),
        "max_new_tokens": answer_cfg.get("max_new_tokens"),
        "ckpt": metadata.get("ckpt_path") if isinstance(metadata, dict) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    rows = [summarize(path) for path in args.paths]
    if args.csv:
        fieldnames = [
            "path",
            "dataset",
            "test_file",
            "n_items",
            "n_predictions",
            "acc",
            "avg_L",
            "avg_output_len",
            "max_L",
            "min_L",
            "latent_temperature",
            "eol_temperature",
            "compression_factor",
            "max_new_tokens",
            "ckpt",
        ]
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
