#!/usr/bin/env python3
"""Build the full-budget TRACE Stage 1 component-ablation report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trace_paper_evidence_suite import summarize_trajectory, trajectory_arrays


COLORS = {
    "blue": "#6687B8",
    "green": "#69B17D",
    "gold": "#E6A314",
    "pink": "#E5A6C4",
    "text": "#27313B",
    "muted": "#66717E",
    "grid": "#D9DEE5",
}

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Liberation Serif",
            "Nimbus Roman No9 L",
            "DejaVu Serif",
        ],
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


VARIANTS = (
    ("full", "TRACE Stage 1", "Full", COLORS["pink"]),
    ("no_path_consistency", "without path consistency", "No path\nconsistency", COLORS["blue"]),
    ("no_progress_anchor", "without weak progress anchoring", "No progress\nanchor", COLORS["gold"]),
    ("no_multiview", "without multi-view compression", "No multi-view", COLORS["green"]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-gsm8k", type=Path, required=True)
    parser.add_argument("--baseline-gsmhard", type=Path, required=True)
    parser.add_argument("--baseline-svamp", type=Path, required=True)
    parser.add_argument("--baseline-multiarith", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--paper-table", type=Path)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def scalar(value, default: float = 0.0) -> float:
    if isinstance(value, list):
        return float(value[0]) if value else default
    return float(value) if value is not None else default


def load_test(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    accuracy = []
    total_length = []
    for value in payload.values():
        if not isinstance(value, dict) or "acc" not in value:
            continue
        accuracy.append(scalar(value.get("acc")))
        total_length.append(
            scalar(value.get("output_length"))
            + scalar(value.get("n_latent_forward"), 8.0)
        )
    if not accuracy:
        raise ValueError(f"No sample records in {path}")
    return {
        "accuracy": np.asarray(accuracy, dtype=np.float64),
        "length": np.asarray(total_length, dtype=np.float64),
    }


def find_result(directory: Path) -> Path:
    candidates = []
    for current, _, names in os.walk(directory, followlinks=True):
        for name in names:
            if name.startswith("test_") and "_gsm_pid" in name and name.endswith(".json"):
                candidates.append(Path(current) / name)
    if not candidates:
        raise FileNotFoundError(f"No test result under {directory}")
    return sorted(candidates)[-1]


def bootstrap_ci(
    values: np.ndarray, trials: int, rng: np.random.Generator
) -> tuple[float, float]:
    means = np.empty(trials, dtype=np.float64)
    for start in range(0, trials, 500):
        end = min(start + 500, trials)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def macro_bootstrap_ci(
    datasets: list[np.ndarray], trials: int, rng: np.random.Generator
) -> tuple[float, float]:
    means = np.empty(trials, dtype=np.float64)
    for trial in range(trials):
        means[trial] = np.mean(
            [values[rng.integers(0, len(values), size=len(values))].mean() for values in datasets]
        )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_variant(
    key: str,
    label: str,
    tests: dict[str, Path],
    records_path: Path,
    trials: int,
    permutations: int,
    seed: int,
) -> dict:
    loaded = {name: load_test(path) for name, path in tests.items()}
    rng = np.random.default_rng(seed)
    gsm_acc = 100.0 * loaded["gsm8k"]["accuracy"]
    gsm_length = loaded["gsm8k"]["length"]
    ood_values = [100.0 * loaded[name]["accuracy"] for name in ("gsmhard", "svamp", "multiarith")]
    gsm_acc_ci = bootstrap_ci(gsm_acc, trials, rng)
    gsm_length_ci = bootstrap_ci(gsm_length, trials, rng)
    ood_ci = macro_bootstrap_ci(ood_values, trials, rng)

    records = torch.load(records_path, map_location="cpu", weights_only=False)
    arrays = trajectory_arrays(records)
    geometry_rows, _ = summarize_trajectory(label, arrays, permutations, trials, seed + 500)
    geometry = {row["metric"]: row for row in geometry_rows}
    return {
        "key": key,
        "variant": label,
        "gsm8k_accuracy_percent": float(gsm_acc.mean()),
        "gsm8k_accuracy_percent_ci95_low": gsm_acc_ci[0],
        "gsm8k_accuracy_percent_ci95_high": gsm_acc_ci[1],
        "ood_macro_accuracy_percent": float(np.mean([values.mean() for values in ood_values])),
        "ood_macro_accuracy_percent_ci95_low": ood_ci[0],
        "ood_macro_accuracy_percent_ci95_high": ood_ci[1],
        "gsm8k_total_L": float(gsm_length.mean()),
        "gsm8k_total_L_ci95_low": gsm_length_ci[0],
        "gsm8k_total_L_ci95_high": gsm_length_ci[1],
        "step_null_gap": geometry["Step correspondence"]["observed_minus_null"],
        "step_null_gap_ci95_low": geometry["Step correspondence"]["difference_ci95_low"],
        "step_null_gap_ci95_high": geometry["Step correspondence"]["difference_ci95_high"],
        "order_null_gap": geometry["Position order"]["observed_minus_null"],
        "order_null_gap_ci95_low": geometry["Position order"]["difference_ci95_low"],
        "order_null_gap_ci95_high": geometry["Position order"]["difference_ci95_high"],
        "n_gsm8k": int(len(gsm_acc)),
        "n_geometry": int(len(records)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_text(rows: list[dict]) -> tuple[str, str]:
    md = [
        "| Variant | GSM8K Acc. | OOD Acc. | GSM8K #L | Step-null gap | Order-null gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    tex_rows = []
    for row in rows:
        values = [
            row["variant"],
            f'{row["gsm8k_accuracy_percent"]:.2f}',
            f'{row["ood_macro_accuracy_percent"]:.2f}',
            f'{row["gsm8k_total_L"]:.2f}',
            f'{row["step_null_gap"]:.3f}',
            f'{row["order_null_gap"]:.3f}',
        ]
        md.append("| " + " | ".join(values) + " |")
        latex_label = {
            "TRACE Stage 1": r"\trace Stage~1",
            "without path consistency": r"\quad without path consistency",
            "without weak progress anchoring": r"\quad without weak progress anchoring",
            "without multi-view compression": r"\quad without multi-view compression",
        }[row["variant"]]
        tex_rows.append(" & ".join([latex_label, *values[1:]]) + r" \\")
    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Full-budget Stage~1 component ablations. Every variant starts from the same CoT-SFT initialization, trains on all 6,726 examples per epoch, validates on all 747 questions, and uses its validation-selected checkpoint. OOD is the unweighted GSMHard/SVAMP/MultiArith average.}",
            r"\label{tab:ablation}",
            r"\small",
            r"\setlength{\tabcolsep}{5.0pt}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Variant & GSM8K Acc. & OOD Acc. & GSM8K \#L & Step-null gap & Order-null gap \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(md) + "\n", latex + "\n"


def style_axis(ax: plt.Axes, grid: str = "y") -> None:
    ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], width=0.7, length=2.5)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.07, label, transform=ax.transAxes, fontsize=9.2, fontweight="bold", va="top")


def draw_figure(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [item[2] for item in VARIANTS]
    colors = [item[3] for item in VARIANTS]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.12, 0.82, 1.12]})
    for ax in axes:
        style_axis(ax)

    width = 0.34
    for offset, key, color, name in (
        (-width / 2, "gsm8k_accuracy_percent", COLORS["blue"], "GSM8K"),
        (width / 2, "ood_macro_accuracy_percent", COLORS["pink"], "OOD macro"),
    ):
        values = np.asarray([row[key] for row in rows])
        low = np.asarray([row[f"{key}_ci95_low"] for row in rows])
        high = np.asarray([row[f"{key}_ci95_high"] for row in rows])
        axes[0].bar(x + offset, values, width, color=color, edgecolor=COLORS["text"], linewidth=0.4, label=name)
        axes[0].errorbar(x + offset, values, yerr=[values - low, high - values], fmt="none", ecolor=COLORS["text"], linewidth=0.65, capsize=1.5)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Task performance", loc="left", fontweight="bold")
    axes[0].legend(loc="lower left")
    panel_label(axes[0], "a")

    length = np.asarray([row["gsm8k_total_L"] for row in rows])
    length_low = np.asarray([row["gsm8k_total_L_ci95_low"] for row in rows])
    length_high = np.asarray([row["gsm8k_total_L_ci95_high"] for row in rows])
    axes[1].bar(x, length, 0.58, color=colors, edgecolor=COLORS["text"], linewidth=0.4)
    axes[1].errorbar(x, length, yerr=[length - length_low, length_high - length], fmt="none", ecolor=COLORS["text"], linewidth=0.65, capsize=1.5)
    axes[1].set_ylabel("GSM8K total #L")
    axes[1].set_title("Reasoning length", loc="left", fontweight="bold")
    panel_label(axes[1], "b")

    for offset, key, color, name in (
        (-width / 2, "step_null_gap", COLORS["gold"], "Step correspondence"),
        (width / 2, "order_null_gap", COLORS["green"], "Position order"),
    ):
        values = np.asarray([row[key] for row in rows])
        low = np.asarray([row[f"{key}_ci95_low"] for row in rows])
        high = np.asarray([row[f"{key}_ci95_high"] for row in rows])
        axes[2].bar(x + offset, values, width, color=color, edgecolor=COLORS["text"], linewidth=0.4, label=name)
        axes[2].errorbar(x + offset, values, yerr=[values - low, high - values], fmt="none", ecolor=COLORS["text"], linewidth=0.65, capsize=1.5)
    axes[2].axhline(0, color=COLORS["muted"], linestyle="--", linewidth=0.7)
    axes[2].set_ylabel("Observed-minus-null gap")
    axes[2].set_title("Trajectory structure", loc="left", fontweight="bold")
    axes[2].legend(loc="upper right")
    panel_label(axes[2], "c")

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", length=0)
    fig.text(0.5, 0.012, "Full-data training and full-epoch validation; error bars are question-bootstrap 95% confidence intervals.", ha="center", fontsize=5.8, color=COLORS["muted"])
    fig.subplots_adjust(left=0.073, right=0.99, top=0.84, bottom=0.31, wspace=0.34)
    base = output_dir / "fig_stage1_component_ablation"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_tests = {
        "gsm8k": args.baseline_gsm8k,
        "gsmhard": args.baseline_gsmhard,
        "svamp": args.baseline_svamp,
        "multiarith": args.baseline_multiarith,
    }
    rows = []
    for idx, (key, label, _, _) in enumerate(VARIANTS):
        if key == "full":
            tests = baseline_tests
            records = args.baseline_records
        else:
            tests = {
                dataset: find_result(args.root / key / "eval" / dataset)
                for dataset in ("gsm8k", "gsmhard", "svamp", "multiarith")
            }
            records = args.root / key / "eval" / "gsm8k" / "logs" / "tb" / "run" / "trace_bridge_visual_test.pt"
            if not records.exists():
                raise FileNotFoundError(records)
        rows.append(
            summarize_variant(
                key,
                label,
                tests,
                records,
                args.bootstrap_trials,
                args.permutations,
                args.seed + idx * 1000,
            )
        )

    write_csv(args.output_dir / "source_data" / "stage1_component_ablations.csv", rows)
    markdown, latex = table_text(rows)
    (args.output_dir / "table_stage1_component_ablations.md").write_text(markdown, encoding="utf-8")
    (args.output_dir / "table_stage1_component_ablations.tex").write_text(latex, encoding="utf-8")
    if args.paper_table:
        args.paper_table.parent.mkdir(parents=True, exist_ok=True)
        args.paper_table.write_text(latex, encoding="utf-8")
    draw_figure(args.output_dir, rows)
    payload = {
        "experiment": "TRACE Stage 1 full-budget component ablations",
        "bootstrap_trials": args.bootstrap_trials,
        "permutations": args.permutations,
        "test_times": 1,
        "rows": rows,
    }
    (args.output_dir / "stage1_component_ablations.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
