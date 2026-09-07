#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trace_bridge_stage_geometry_delta import METRICS, bootstrap_mean_ci, composition, load_summary


def finite_metric(rows, key):
    values = {}
    for idx, row in rows.items():
        value = row.get(key)
        if value is not None and np.isfinite(value):
            values[idx] = float(value)
    return values


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(a, b, rank=False):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return None
    if rank:
        a = rankdata(a)
        b = rankdata(b)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def mean_summary(values, rng, trials):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "bootstrap_ci95_low": None, "bootstrap_ci95_high": None}
    low, high = bootstrap_mean_ci(values, rng, trials)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
    }


def compare_rollouts(seed0_rows, seed1_rows, rng, trials):
    common = sorted(set(seed0_rows) & set(seed1_rows))
    seed0 = np.asarray(
        [seed0_rows[idx]["n_correct"] / seed0_rows[idx]["n_paths"] for idx in common],
        dtype=np.float64,
    )
    seed1 = np.asarray(
        [seed1_rows[idx]["n_correct"] / seed1_rows[idx]["n_paths"] for idx in common],
        dtype=np.float64,
    )
    delta_pp = 100.0 * (seed1 - seed0)
    delta_low, delta_high = bootstrap_mean_ci(delta_pp, rng, trials)
    seed0_ci = bootstrap_mean_ci(100.0 * seed0, rng, trials)
    seed1_ci = bootstrap_mean_ci(100.0 * seed1, rng, trials)
    return {
        "common_questions": len(common),
        "indices": common,
        "seed0_accuracy": 100.0 * float(seed0.mean()),
        "seed0_bootstrap_ci95_low": seed0_ci[0],
        "seed0_bootstrap_ci95_high": seed0_ci[1],
        "seed1_accuracy": 100.0 * float(seed1.mean()),
        "seed1_bootstrap_ci95_low": seed1_ci[0],
        "seed1_bootstrap_ci95_high": seed1_ci[1],
        "delta_pp": float(delta_pp.mean()),
        "delta_bootstrap_ci95_low": delta_low,
        "delta_bootstrap_ci95_high": delta_high,
        "pearson_r": correlation(seed0, seed1),
        "spearman_rho": correlation(seed0, seed1, rank=True),
        "exact_match_fraction": float((seed0 == seed1).mean()),
        "seed0_per_question": seed0.tolist(),
        "seed1_per_question": seed1.tolist(),
    }


def compare_metrics(seed0_rows, seed1_rows, rng, trials):
    output = []
    for key, label, direction in METRICS:
        seed0_map = finite_metric(seed0_rows, key)
        seed1_map = finite_metric(seed1_rows, key)
        common = sorted(set(seed0_map) & set(seed1_map))
        seed0_all = mean_summary(list(seed0_map.values()), rng, trials)
        seed1_all = mean_summary(list(seed1_map.values()), rng, trials)
        row = {
            "metric": key,
            "label": label,
            "direction": "lower_is_better" if direction < 0 else "higher_is_better",
            "seed0": seed0_all,
            "seed1": seed1_all,
            "paired_n": len(common),
        }
        if common:
            a = np.asarray([seed0_map[idx] for idx in common], dtype=np.float64)
            b = np.asarray([seed1_map[idx] for idx in common], dtype=np.float64)
            delta = b - a
            low, high = bootstrap_mean_ci(delta, rng, trials)
            row.update(
                {
                    "paired_seed0_mean": float(a.mean()),
                    "paired_seed1_mean": float(b.mean()),
                    "paired_delta": float(delta.mean()),
                    "paired_delta_bootstrap_ci95_low": low,
                    "paired_delta_bootstrap_ci95_high": high,
                    "pearson_r": correlation(a, b),
                    "spearman_rho": correlation(a, b, rank=True),
                }
            )
        output.append(row)
    return output


def representation_payload(seed0_path, seed1_path, rng, trials):
    _, seed0_sep, seed0_rows = load_summary(seed0_path)
    _, seed1_sep, seed1_rows = load_summary(seed1_path)
    rep0 = seed0_sep.get("signature_representation", "raw")
    rep1 = seed1_sep.get("signature_representation", "raw")
    if rep0 != rep1:
        raise ValueError(f"signature representations differ: {rep0!r} vs {rep1!r}")
    common = sorted(set(seed0_rows) & set(seed1_rows))
    return {
        "seed0_path": str(seed0_path),
        "seed1_path": str(seed1_path),
        "signature_representation": rep0,
        "seed0_composition": composition({idx: seed0_rows[idx] for idx in common}),
        "seed1_composition": composition({idx: seed1_rows[idx] for idx in common}),
        "rollout": compare_rollouts(seed0_rows, seed1_rows, rng, trials),
        "metrics": compare_metrics(seed0_rows, seed1_rows, rng, trials),
    }


def plot_dashboard(payload, out_path):
    centered = payload["centered"]
    rollout = centered["rollout"]
    metrics = centered["metrics"]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))

    x = np.asarray(rollout["seed0_per_question"]) * 100.0
    y = np.asarray(rollout["seed1_per_question"]) * 100.0
    axes[0, 0].scatter(x, y, s=24, alpha=0.55, color="#2563eb", edgecolors="none")
    axes[0, 0].plot([0, 100], [0, 100], linestyle="--", color="#374151", linewidth=1)
    axes[0, 0].set_xlim(-3, 103)
    axes[0, 0].set_ylim(-3, 103)
    axes[0, 0].set_xlabel("Seed 0 correct rollouts per question (%)")
    axes[0, 0].set_ylabel("Seed 1 correct rollouts per question (%)")
    rho = rollout["spearman_rho"]
    rho_text = "NA" if rho is None else f"{rho:.2f}"
    axes[0, 0].set_title(f"Independent rollout/noise seeds (Spearman {rho_text})")
    axes[0, 0].grid(alpha=0.2)

    categories = ("all_wrong", "mixed", "all_correct")
    colors = ("#dc2626", "#d97706", "#15803d")
    for seed_idx, seed_name in enumerate(("seed0", "seed1")):
        bottom = 0
        for category, color in zip(categories, colors):
            value = centered[f"{seed_name}_composition"][category]
            axes[0, 1].bar(seed_idx, value, bottom=bottom, color=color, label=category.replace("_", " ") if seed_idx == 0 else None)
            bottom += value
    axes[0, 1].set_xticks([0, 1], labels=["Seed 0", "Seed 1"])
    axes[0, 1].set_ylabel("Questions (8 rollouts each)")
    axes[0, 1].set_title("Outcome composition on the same 200 questions")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=0.2)

    positions = np.arange(len(metrics), dtype=np.float64)
    width = 0.36
    for offset, key, label, color in (
        (-width / 2, "seed0", "Seed 0", "#475569"),
        (width / 2, "seed1", "Seed 1", "#2563eb"),
    ):
        means = np.asarray([row[key]["mean"] for row in metrics], dtype=np.float64)
        low = means - np.asarray([row[key]["bootstrap_ci95_low"] for row in metrics])
        high = np.asarray([row[key]["bootstrap_ci95_high"] for row in metrics]) - means
        axes[1, 0].bar(positions + offset, means, width, color=color, label=label)
        axes[1, 0].errorbar(positions + offset, means, yerr=np.vstack([low, high]), fmt="none", color="#111827", capsize=3)
    axes[1, 0].axhline(0.0, color="#111827", linewidth=1)
    axes[1, 0].set_xticks(positions, labels=[row["label"] for row in metrics], rotation=18, ha="right")
    axes[1, 0].set_ylabel("Metric mean with question-bootstrap 95% CI")
    axes[1, 0].set_title("Stage2-centered geometry replication")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.2)

    paired = [row for row in metrics if row["paired_n"]]
    y_pos = np.arange(len(paired), dtype=np.float64)
    delta = np.asarray([row["paired_delta"] for row in paired])
    low = np.asarray([row["paired_delta_bootstrap_ci95_low"] for row in paired])
    high = np.asarray([row["paired_delta_bootstrap_ci95_high"] for row in paired])
    axes[1, 1].errorbar(delta, y_pos, xerr=np.vstack([delta - low, high - delta]), fmt="o", color="#7c3aed", capsize=4)
    axes[1, 1].axvline(0.0, color="#111827", linewidth=1)
    axes[1, 1].set_yticks(y_pos, labels=[f"{row['label']} (n={row['paired_n']})" for row in paired])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Paired seed1 - seed0 mean difference (95% CI)")
    axes[1, 1].set_title("Zero indicates seed stability, not model superiority")
    axes[1, 1].grid(axis="x", alpha=0.2)

    fig.suptitle("TRACE epoch7 geometry: independent-seed robustness", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt_corr(value):
    return "NA" if value is None else f"{value:.3f}"


def write_markdown(path, payload):
    centered = payload["centered"]
    rollout = centered["rollout"]
    lines = [
        "# TRACE Epoch7 Independent-Seed Geometry Robustness",
        "",
        "The frozen checkpoint is evaluated twice on the same 200 questions with eight paths per question. Seed 1 changes both sampling and latent-noise seed; no checkpoint selection or retraining uses this replication.",
        "",
        "## Rollout outcomes",
        "",
        f"Seed 0 rollout accuracy: **{rollout['seed0_accuracy']:.2f}%** "
        f"[{rollout['seed0_bootstrap_ci95_low']:.2f}, {rollout['seed0_bootstrap_ci95_high']:.2f}]  ",
        f"Seed 1 rollout accuracy: **{rollout['seed1_accuracy']:.2f}%** "
        f"[{rollout['seed1_bootstrap_ci95_low']:.2f}, {rollout['seed1_bootstrap_ci95_high']:.2f}]  ",
        f"Paired seed difference: **{rollout['delta_pp']:+.2f} pp** "
        f"[{rollout['delta_bootstrap_ci95_low']:+.2f}, {rollout['delta_bootstrap_ci95_high']:+.2f}].  ",
        f"Question-level Spearman rho: **{fmt_corr(rollout['spearman_rho'])}**; exact 8-path accuracy match: **{100.0 * rollout['exact_match_fraction']:.1f}%**.",
        "",
        "| Seed | All wrong | Mixed | All correct |",
        "| --- | ---: | ---: | ---: |",
        f"| 0 | {centered['seed0_composition']['all_wrong']} | {centered['seed0_composition']['mixed']} | {centered['seed0_composition']['all_correct']} |",
        f"| 1 | {centered['seed1_composition']['all_wrong']} | {centered['seed1_composition']['mixed']} | {centered['seed1_composition']['all_correct']} |",
        "",
        "## Stage2-centered geometry",
        "",
        "Marginal columns use every question on which that seed's metric is defined. The paired delta uses only the shared eligible question subset; this prevents differing mixed-outcome composition from being hidden.",
        "",
        "| Metric | Seed 0 mean (95% CI), n | Seed 1 mean (95% CI), n | Paired n | Seed1 - seed0 (95% CI) | Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in centered["metrics"]:
        a = row["seed0"]
        b = row["seed1"]
        delta = "NA"
        if row["paired_n"]:
            delta = (
                f"{row['paired_delta']:+.4f} "
                f"[{row['paired_delta_bootstrap_ci95_low']:+.4f}, {row['paired_delta_bootstrap_ci95_high']:+.4f}]"
            )
        lines.append(
            f"| {row['label']} ({row['direction']}) | {a['mean']:.4f} "
            f"[{a['bootstrap_ci95_low']:.4f}, {a['bootstrap_ci95_high']:.4f}], {a['n']} | "
            f"{b['mean']:.4f} [{b['bootstrap_ci95_low']:.4f}, {b['bootstrap_ci95_high']:.4f}], {b['n']} | "
            f"{row['paired_n']} | {delta} | {fmt_corr(row.get('spearman_rho'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Replication supports a geometry claim only when its qualitative sign survives both seeds. In particular, positive correct/wrong margin and wrong-dispersion gap are collective separation claims. Rejection AUC above a permuted-label null is the separate individual-path claim; if either confidence interval crosses zero, it remains inconclusive. Per-question correlation is secondary because the two evaluations intentionally resample paths.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0_raw", type=Path, required=True)
    parser.add_argument("--seed1_raw", type=Path, required=True)
    parser.add_argument("--seed0_centered", type=Path, required=True)
    parser.add_argument("--seed1_centered", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    payload = {
        "bootstrap_trials": args.bootstrap_trials,
        "bootstrap_seed": args.seed,
        "raw": representation_payload(args.seed0_raw, args.seed1_raw, rng, args.bootstrap_trials),
        "centered": representation_payload(
            args.seed0_centered,
            args.seed1_centered,
            rng,
            args.bootstrap_trials,
        ),
    }
    (args.out_dir / "geometry_seed_robustness.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "geometry_seed_robustness.md", payload)
    plot_dashboard(payload, args.out_dir / "geometry_seed_robustness_dashboard.png")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
