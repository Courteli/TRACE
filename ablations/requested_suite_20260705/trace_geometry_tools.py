#!/usr/bin/env python3
import argparse
import csv
import gc
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path("/disk1/dingxukai/trace_colar")
VIS_PATH = ROOT / "tools" / "trace_multipath_visualize.py"


def load_vis_module():
    spec = importlib.util.spec_from_file_location("trace_multipath_visualize_local", VIS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    return module


VIS = load_vis_module()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_args(ckpt, device, group_size, max_l, min_l=0):
    return SimpleNamespace(
        ckpt=str(ckpt),
        dataset_path=VIS.DEFAULT_DATASET,
        indices=[],
        auto_select=False,
        candidate_count=0,
        num_questions=0,
        seed=0,
        group_size=group_size,
        max_l=max_l,
        min_l=min_l,
        device=device,
        out_dir="",
        dpi=180,
        trajectory_space="residual",
        pca_scope="global",
    )


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_for_ckpt(ckpt, indices, device, group_size, max_l, label, seed):
    set_seed(seed)
    args = make_args(ckpt, device=device, group_size=group_size, max_l=max_l)
    model = VIS.load_model(args)
    dataset = VIS.load_dataset(Path(VIS.DEFAULT_DATASET))
    groups = []
    for offset, idx in enumerate(indices):
        set_seed(seed + offset)
        group = VIS.collect_group(model, dataset[idx], group_size)
        group["method"] = label
        groups.append(group)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    release_model(model)
    return groups


def all_valid_points(method_groups):
    points = []
    transformed = {}
    for method, groups in method_groups.items():
        transformed[method] = []
        for group in groups:
            latents = VIS.transform_group_latents(group["latent_embeds"], group["latent_mask"], "residual")
            transformed[method].append(latents)
            mask = group["latent_mask"]
            for i in range(latents.shape[0]):
                valid = mask[i].astype(bool)
                if valid.any():
                    points.append(latents[i, valid])
    return transformed, points


def plot_global_pca(method_groups, out_dir, dpi=180):
    plt, _ = VIS.ensure_matplotlib()
    transformed, points = all_valid_points(method_groups)
    stacked = np.vstack(points)
    mean, components = VIS.fit_pca(stacked, dim=3)
    methods = list(method_groups.keys())
    ncols = len(next(iter(method_groups.values())))
    fig = plt.figure(figsize=(4.4 * ncols, 4.0 * len(methods)), constrained_layout=True)
    mode_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf"]
    wrong_color = "#d95f02"
    hard_color = "#d62728"
    gray = "#7f7f7f"
    for row, method in enumerate(methods):
        for col, group in enumerate(method_groups[method]):
            ax_idx = row * ncols + col + 1
            ax = fig.add_subplot(len(methods), ncols, ax_idx, projection="3d")
            plot_latents = transformed[method][col]
            order = VIS.sorted_indices(group)
            any_pos = group["accuracies"].sum() > 0
            for rollout_idx in order:
                valid = group["latent_mask"][rollout_idx].astype(bool)
                if not valid.any():
                    continue
                xyz = VIS.project(plot_latents[rollout_idx, valid], mean, components, dim=3)
                is_correct = group["accuracies"][rollout_idx] > 0.5
                is_hard = bool(group["hard_mask"][rollout_idx])
                mode = int(group["mode_assign"][rollout_idx])
                if is_correct:
                    color = mode_colors[mode % len(mode_colors)] if mode >= 0 else mode_colors[0]
                    marker = "o"
                    label = f"correct mode {mode}" if mode >= 0 else "correct"
                    lw = 1.7
                    alpha = 0.95
                elif is_hard:
                    color = hard_color
                    marker = "x"
                    label = "hard negative"
                    lw = 1.5
                    alpha = 0.9
                else:
                    color = wrong_color if any_pos else gray
                    marker = "s"
                    label = "wrong"
                    lw = 1.1
                    alpha = 0.65
                ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=lw, alpha=alpha)
                ax.scatter(xyz[-1:, 0], xyz[-1:, 1], xyz[-1:, 2], color=color, s=30, marker=marker, label=label)
            metrics = group["metrics"]
            if col == 0:
                ax.text2D(-0.22, 0.95, method, transform=ax.transAxes, fontsize=11, fontweight="bold")
            ax.set_title(
                f"q{metrics['idx']} acc={metrics['acc']:.2f} modes={metrics['mode_count']} hard={metrics['hard_count']}",
                fontsize=9,
            )
            ax.set_xlabel("global PC1")
            ax.set_ylabel("global PC2")
            ax.set_zlabel("global PC3")
            ax.view_init(elev=22, azim=-58)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    dedup = {}
    for h, label in zip(handles, labels):
        dedup.setdefault(label, h)
    fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=min(5, len(dedup)), frameon=False)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "same_question_global_pca_3d.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def write_metrics(method_groups, out_dir):
    out_dir = Path(out_dir)
    keys = ["method"] + sorted(next(iter(method_groups.values()))[0]["metrics"].keys())
    path = out_dir / "same_question_global_pca_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for method, groups in method_groups.items():
            for group in groups:
                row = {"method": method}
                row.update(group["metrics"])
                writer.writerow(row)
    return path


def command_global_pca(args):
    indices = args.indices
    method_groups = {
        "CoLaR origin": collect_for_ckpt(args.origin_ckpt, indices, args.device, args.group_size, args.max_l, "CoLaR origin", args.seed),
        "answer-only RL": collect_for_ckpt(args.answer_ckpt, indices, args.device, args.group_size, args.max_l, "answer-only RL", args.seed + 10000),
        "TRACE v3": collect_for_ckpt(args.trace_ckpt, indices, args.device, args.group_size, args.max_l, "TRACE v3", args.seed + 20000),
    }
    plot_path = plot_global_pca(method_groups, args.out_dir, dpi=args.dpi)
    metrics_path = write_metrics(method_groups, args.out_dir)
    print(f"[done] wrote {plot_path}")
    print(f"[done] wrote {metrics_path}")


def summarize_rows(rows):
    numeric_keys = [
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
    out = {"n_questions": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        if not values:
            continue
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    out["mixed_group_rate"] = float(np.mean([(row["pos_count"] > 0 and row["neg_count"] > 0) for row in rows])) if rows else 0.0
    out["all_correct_rate"] = float(np.mean([row["neg_count"] == 0 for row in rows])) if rows else 0.0
    out["all_wrong_rate"] = float(np.mean([row["pos_count"] == 0 for row in rows])) if rows else 0.0
    out["has_hard_negative_rate"] = float(np.mean([row["hard_count"] > 0 for row in rows])) if rows else 0.0
    out["multi_mode_positive_rate"] = float(np.mean([row["mode_count"] > 1 for row in rows if row["pos_count"] > 1])) if any(row["pos_count"] > 1 for row in rows) else 0.0
    return out


def command_summary(args):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.candidate_seed)
    dataset = VIS.load_dataset(Path(VIS.DEFAULT_DATASET))
    if args.indices:
        indices = args.indices
    else:
        count = min(args.num_questions, len(dataset))
        indices = rng.choice(np.arange(len(dataset)), size=count, replace=False).tolist()
    model_args = make_args(args.ckpt, device=args.device, group_size=args.group_size, max_l=args.max_l)
    model = VIS.load_model(model_args)
    rows = []
    rollouts_path = out_dir / f"{args.method}_geometry_rollouts.jsonl"
    with rollouts_path.open("w", encoding="utf-8") as rollouts:
        for n, idx in enumerate(indices, start=1):
            set_seed(args.seed + n - 1)
            group = VIS.collect_group(model, dataset[idx], args.group_size)
            row = {"method": args.method}
            row.update(group["metrics"])
            rows.append(row)
            for i, pred in enumerate(group["pred_strings"]):
                rollouts.write(
                    json.dumps(
                        {
                            "method": args.method,
                            "idx": int(group["metrics"]["idx"]),
                            "rollout": i,
                            "correct": float(group["accuracies"][i]),
                            "hard_negative": bool(group["hard_mask"][i]),
                            "mode": int(group["mode_assign"][i]),
                            "prediction": pred,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if n % 10 == 0:
                print(f"[summary] {args.method}: {n}/{len(indices)} groups")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    release_model(model)
    keys = ["method"] + sorted([k for k in rows[0].keys() if k != "method"])
    csv_path = out_dir / f"{args.method}_geometry_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary = summarize_rows(rows)
    summary_path = out_dir / f"{args.method}_geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {rollouts_path}")
    print(f"[done] wrote {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    gp = sub.add_parser("global-pca")
    gp.add_argument("--origin_ckpt", required=True)
    gp.add_argument("--answer_ckpt", required=True)
    gp.add_argument("--trace_ckpt", required=True)
    gp.add_argument("--indices", nargs="+", type=int, required=True)
    gp.add_argument("--group_size", type=int, default=8)
    gp.add_argument("--max_l", type=int, default=40)
    gp.add_argument("--device", default="cuda:0")
    gp.add_argument("--out_dir", required=True)
    gp.add_argument("--dpi", type=int, default=180)
    gp.add_argument("--seed", type=int, default=0)
    gp.set_defaults(func=command_global_pca)

    sm = sub.add_parser("summary")
    sm.add_argument("--method", required=True)
    sm.add_argument("--ckpt", required=True)
    sm.add_argument("--indices", nargs="*", type=int, default=[])
    sm.add_argument("--num_questions", type=int, default=200)
    sm.add_argument("--candidate_seed", type=int, default=0)
    sm.add_argument("--group_size", type=int, default=8)
    sm.add_argument("--max_l", type=int, default=40)
    sm.add_argument("--device", default="cuda:0")
    sm.add_argument("--out_dir", required=True)
    sm.add_argument("--seed", type=int, default=0)
    sm.set_defaults(func=command_summary)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
