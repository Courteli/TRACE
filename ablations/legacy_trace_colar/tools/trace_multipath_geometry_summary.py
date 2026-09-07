#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from trace_multipath_visualize import DEFAULT_DATASET, collect_group, load_dataset, load_model


NUMERIC_KEYS = [
    "acc",
    "pos_count",
    "neg_count",
    "hard_count",
    "mode_count",
    "effective_modes",
    "mode_entropy",
    "mean_n_latent_forward",
    "mean_step_coherence",
    "mean_delta_norm",
    "mean_nearest_proto_sim",
    "pos_proto_sim",
    "neg_proto_sim",
    "pos_neg_margin",
    "pos_pair_sim",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_questions", type=int, default=200)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--sample_random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_l", type=int, default=40)
    parser.add_argument("--min_l", type=int, default=0)
    parser.add_argument("--latent_temperature", type=float, default=None)
    parser.add_argument("--eol_temperature", type=float, default=None)
    parser.add_argument("--compression_factor", type=int, default=None)
    parser.add_argument("--lp_deterministic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_rollouts", action="store_true")
    return parser.parse_args()


def choose_indices(dataset_size, args):
    if args.indices:
        return [idx for idx in args.indices if 0 <= idx < dataset_size]
    if args.sample_random:
        rng = np.random.default_rng(args.seed)
        count = min(args.num_questions, dataset_size)
        return sorted(rng.choice(np.arange(dataset_size), size=count, replace=False).tolist())
    end = min(args.start_idx + args.num_questions, dataset_size)
    return list(range(args.start_idx, end))


def numeric_stats(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=0))
    ci95 = 0.0 if len(arr) == 1 else float(1.96 * std / math.sqrt(len(arr)))
    return {"n": int(len(arr)), "mean": float(arr.mean()), "std": std, "ci95": ci95}


def build_summary(rows, args, indices):
    metrics = {key: numeric_stats([row.get(key) for row in rows]) for key in NUMERIC_KEYS}
    mixed_count = sum(1 for row in rows if row["pos_count"] > 0 and row["neg_count"] > 0)
    all_correct_count = sum(1 for row in rows if row["pos_count"] == args.group_size)
    all_wrong_count = sum(1 for row in rows if row["pos_count"] == 0)
    has_hard_count = sum(1 for row in rows if row["hard_count"] > 0)
    return {
        "ckpt": str(Path(args.ckpt).resolve()),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "num_questions": len(rows),
        "requested_num_questions": args.num_questions,
        "group_size": args.group_size,
        "max_l": args.max_l,
        "min_l": args.min_l,
        "latent_temperature": args.latent_temperature,
        "eol_temperature": args.eol_temperature,
        "compression_factor": args.compression_factor,
        "seed": args.seed,
        "indices": indices,
        "mixed_count": mixed_count,
        "mixed_frac": mixed_count / max(len(rows), 1),
        "all_correct_count": all_correct_count,
        "all_wrong_count": all_wrong_count,
        "has_hard_count": has_hard_count,
        "has_hard_frac": has_hard_count / max(len(rows), 1),
        "metrics": metrics,
    }


def write_metrics_csv(rows, path):
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_md(summary, path):
    selected = [
        "acc",
        "mode_count",
        "effective_modes",
        "pos_count",
        "neg_count",
        "hard_count",
        "mean_n_latent_forward",
        "mean_step_coherence",
        "mean_delta_norm",
        "pos_neg_margin",
        "pos_pair_sim",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# TRACE MultiPath Geometry Summary\n\n")
        f.write(f"- ckpt: `{summary['ckpt']}`\n")
        f.write(f"- num_questions: {summary['num_questions']}\n")
        f.write(f"- group_size: {summary['group_size']}\n")
        f.write(f"- mixed_frac: {summary['mixed_frac']:.6f}\n")
        f.write(f"- has_hard_frac: {summary['has_hard_frac']:.6f}\n\n")
        f.write("| metric | mean | std | ci95 | n |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for key in selected:
            stat = summary["metrics"][key]
            mean = "" if stat["mean"] is None else f"{stat['mean']:.6f}"
            std = "" if stat["std"] is None else f"{stat['std']:.6f}"
            ci95 = "" if stat["ci95"] is None else f"{stat['ci95']:.6f}"
            f.write(f"| {key} | {mean} | {std} | {ci95} | {stat['n']} |\n")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(Path(args.dataset_path))
    indices = choose_indices(len(dataset), args)
    if not indices:
        raise SystemExit("no valid question indices selected")

    model = load_model(args)
    rows = []
    rollouts_path = out_dir / "trace_multipath_geometry_rollouts.jsonl"
    rollouts_file = None if args.no_rollouts else rollouts_path.open("w", encoding="utf-8")
    try:
        for offset, idx in enumerate(indices, start=1):
            group = collect_group(model, dataset[idx], args.group_size)
            metric_row = dict(group["metrics"])
            metric_row["source_id"] = group["item"].get("source_id", idx)
            rows.append(metric_row)
            if rollouts_file is not None:
                item = group["item"]
                for rollout_idx, pred in enumerate(group["pred_strings"]):
                    rollouts_file.write(
                        json.dumps(
                            {
                                "idx": int(item["idx"]),
                                "source_id": item.get("source_id", item["idx"]),
                                "rollout": rollout_idx,
                                "correct": float(group["accuracies"][rollout_idx]),
                                "hard_negative": bool(group["hard_mask"][rollout_idx]),
                                "mode": int(group["mode_assign"][rollout_idx]),
                                "prediction": pred,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if offset % 10 == 0 or offset == len(indices):
                print(f"[geometry] collected {offset}/{len(indices)} questions")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if rollouts_file is not None:
            rollouts_file.close()

    summary = build_summary(rows, args, indices)
    metrics_path = out_dir / "trace_multipath_geometry_metrics.csv"
    summary_json_path = out_dir / "trace_multipath_geometry_summary.json"
    summary_md_path = out_dir / "trace_multipath_geometry_summary.md"
    write_metrics_csv(rows, metrics_path)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(summary, summary_md_path)

    print(f"[done] wrote {metrics_path}")
    print(f"[done] wrote {summary_json_path}")
    print(f"[done] wrote {summary_md_path}")
    if not args.no_rollouts:
        print(f"[done] wrote {rollouts_path}")


if __name__ == "__main__":
    main()
