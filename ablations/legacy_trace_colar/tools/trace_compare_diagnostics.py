#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    samples = data.get("samples", [])
    mixed_count = summary.get("mixed_count")
    mixed_frac = summary.get("mixed_frac")
    if mixed_count is None:
        mixed_count = sum(1 for row in samples if row.get("pos_count", 0) > 0 and row.get("neg_count", 0) > 0)
    if mixed_frac is None:
        mixed_frac = mixed_count / max(len(samples), 1)
    return {
        "file": path.name,
        "ckpt": summary.get("ckpt", ""),
        "n": summary.get("num_questions"),
        "group_size": summary.get("group_size"),
        "mixed_count": mixed_count,
        "mixed_frac": mixed_frac,
        "group_acc": summary.get("mean_group_acc"),
        "pos_count": summary.get("mean_pos_count"),
        "neg_count": summary.get("mean_neg_count"),
        "latent_len": summary.get("mean_latent_len"),
        "pos_neg_sim": summary.get("mean_pos_neg_sim"),
        "centered_pos_neg_sim": summary.get("mean_centered_pos_neg_sim"),
        "pos_pos_sim": summary.get("mean_pos_pos_sim"),
        "centered_pos_pos_sim": summary.get("mean_centered_pos_pos_sim"),
    }


def print_markdown(rows):
    headers = [
        "file",
        "n",
        "G",
        "mixed",
        "mixed_frac",
        "acc",
        "pos",
        "neg",
        "#L",
        "raw_pn",
        "ctr_pn",
        "raw_pp",
        "ctr_pp",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row["file"],
                    fmt(row["n"], 0),
                    fmt(row["group_size"], 0),
                    fmt(row["mixed_count"], 0),
                    fmt(row["mixed_frac"]),
                    fmt(row["group_acc"]),
                    fmt(row["pos_count"]),
                    fmt(row["neg_count"]),
                    fmt(row["latent_len"], 2),
                    fmt(row["pos_neg_sim"]),
                    fmt(row["centered_pos_neg_sim"]),
                    fmt(row["pos_pos_sim"]),
                    fmt(row["centered_pos_pos_sim"]),
                ]
            )
            + " |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    rows = [summarize(Path(path)) for path in args.paths]
    print_markdown(rows)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
