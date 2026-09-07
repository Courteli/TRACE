#!/usr/bin/env python3
"""Analyze matched TRACE Stage 1 and Final decoding-budget evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#6687B8",
    "green": "#69B17D",
    "gold": "#E6A314",
    "pink": "#E5A6C4",
    "text": "#27313B",
    "muted": "#66717E",
    "grid": "#D9DEE5",
    "null": "#AAB2BC",
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
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def scalar(value, default: float = 0.0) -> float:
    if isinstance(value, list):
        return float(value[0]) if value else default
    return float(value) if value is not None else default


def find_result(directory: Path) -> Path | None:
    candidates = sorted(directory.rglob("test_*_gsm_pid*.json"))
    return candidates[-1] if candidates else None


def load_result(path: Path) -> dict[int, dict[str, float | str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[int, dict[str, float | str]] = {}
    for fallback_idx, (key, item) in enumerate(payload.items()):
        if not isinstance(item, dict) or "acc" not in item:
            continue
        try:
            idx = int(key)
        except ValueError:
            idx = fallback_idx
        output_length = scalar(item.get("output_length"))
        n_latents = scalar(item.get("n_latent_forward"), 8.0)
        rows[idx] = {
            "acc": scalar(item.get("acc")),
            "output_length": output_length,
            "n_latents": n_latents,
            "L": output_length + n_latents,
            "question": str(item.get("question", "")),
        }
    return rows


def bootstrap_mean_ci(
    values: np.ndarray, trials: int, rng: np.random.Generator
) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(trials, dtype=np.float64)
    chunk = 500
    for start in range(0, trials, chunk):
        end = min(start + chunk, trials)
        indices = rng.integers(0, n, size=(end - start, n))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_mcnemar_p(rescued: int, regressed: int) -> float:
    discordant = rescued + regressed
    if discordant == 0:
        return 1.0
    smaller = min(rescued, regressed)
    tail = sum(math.comb(discordant, idx) for idx in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def matched_summary(
    budget: int,
    stage1: dict[int, dict[str, float | str]],
    final: dict[int, dict[str, float | str]],
    trials: int,
    seed: int,
) -> dict[str, float | int]:
    ids = sorted(set(stage1) & set(final))
    if not ids:
        raise ValueError(f"No matched examples for budget {budget}")
    for idx in ids:
        if stage1[idx]["question"] != final[idx]["question"]:
            raise ValueError(f"Question mismatch at budget={budget}, idx={idx}")
    s_acc = np.asarray([stage1[idx]["acc"] for idx in ids], dtype=np.float64)
    f_acc = np.asarray([final[idx]["acc"] for idx in ids], dtype=np.float64)
    s_len = np.asarray([stage1[idx]["L"] for idx in ids], dtype=np.float64)
    f_len = np.asarray([final[idx]["L"] for idx in ids], dtype=np.float64)
    acc_delta = 100.0 * (f_acc - s_acc)
    length_delta = f_len - s_len
    rng = np.random.default_rng(seed + budget)
    acc_low, acc_high = bootstrap_mean_ci(acc_delta, trials, rng)
    len_low, len_high = bootstrap_mean_ci(length_delta, trials, rng)
    rescued = int(np.sum((s_acc == 0) & (f_acc == 1)))
    regressed = int(np.sum((s_acc == 1) & (f_acc == 0)))
    p_value = exact_mcnemar_p(rescued, regressed)
    return {
        "max_new_tokens": budget,
        "n": len(ids),
        "stage1_accuracy_percent": 100.0 * float(s_acc.mean()),
        "final_accuracy_percent": 100.0 * float(f_acc.mean()),
        "accuracy_gain_pp": float(acc_delta.mean()),
        "accuracy_gain_ci95_low": acc_low,
        "accuracy_gain_ci95_high": acc_high,
        "stage1_total_L": float(s_len.mean()),
        "final_total_L": float(f_len.mean()),
        "length_delta": float(length_delta.mean()),
        "length_delta_ci95_low": len_low,
        "length_delta_ci95_high": len_high,
        "length_saved": -float(length_delta.mean()),
        "length_saved_ci95_low": -len_high,
        "length_saved_ci95_high": -len_low,
        "rescued": rescued,
        "regressed": regressed,
        "mcnemar_exact_p": p_value,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def p_string(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def write_tables(output_dir: Path, rows: list[dict]) -> None:
    headers = [
        "Budget",
        "Stage 1 Acc.",
        "Final Acc.",
        "Gain",
        "Stage 1 #L",
        "Final #L",
        "Saved #L",
        "R/G",
        "p",
    ]
    markdown = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    tex_rows = []
    for row in rows:
        values = [
            str(row["max_new_tokens"]),
            f'{row["stage1_accuracy_percent"]:.2f}',
            f'{row["final_accuracy_percent"]:.2f}',
            f'{row["accuracy_gain_pp"]:+.2f}',
            f'{row["stage1_total_L"]:.2f}',
            f'{row["final_total_L"]:.2f}',
            f'{row["length_saved"]:+.2f}',
            f'{row["rescued"]}/{row["regressed"]}',
            p_string(float(row["mcnemar_exact_p"])),
        ]
        markdown.append("| " + " | ".join(values) + " |")
        tex_rows.append(" & ".join(values) + r" \\")
    (output_dir / "table_accuracy_length_pareto.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    latex = "\n".join(
        [
            r"\begin{tabular}{rcccccccc}",
            r"\toprule",
            r"Budget & \multicolumn{3}{c}{Accuracy (\%)} & \multicolumn{3}{c}{Total \#L} & R/G & $p$ \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r" & Stage 1 & Final & $\Delta$ & Stage 1 & Final & Saved & & \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output_dir / "table_accuracy_length_pareto.tex").write_text(
        latex + "\n", encoding="utf-8"
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=COLORS["grid"], linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], width=0.7, length=2.5)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
        color=COLORS["text"],
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)


def draw_figure(output_dir: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: int(row["max_new_tokens"]))
    budgets = np.asarray([row["max_new_tokens"] for row in rows], dtype=float)
    stage1_acc = np.asarray([row["stage1_accuracy_percent"] for row in rows])
    final_acc = np.asarray([row["final_accuracy_percent"] for row in rows])
    stage1_len = np.asarray([row["stage1_total_L"] for row in rows])
    final_len = np.asarray([row["final_total_L"] for row in rows])
    gain = np.asarray([row["accuracy_gain_pp"] for row in rows])
    gain_low = np.asarray([row["accuracy_gain_ci95_low"] for row in rows])
    gain_high = np.asarray([row["accuracy_gain_ci95_high"] for row in rows])
    saved = np.asarray([row["length_saved"] for row in rows])
    saved_low = np.asarray([row["length_saved_ci95_low"] for row in rows])
    saved_high = np.asarray([row["length_saved_ci95_high"] for row in rows])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.42),
        gridspec_kw={"width_ratios": [1.18, 0.91, 0.91]},
    )
    for ax in axes:
        style_axis(ax)

    ax = axes[0]
    label_offsets = {
        24: (-10, 0),
        32: (-13, 5),
        40: (-13, -10),
        48: (-7, 10),
        64: (8, -1),
    }
    for idx, budget in enumerate(budgets):
        ax.annotate(
            "",
            xy=(final_len[idx], final_acc[idx]),
            xytext=(stage1_len[idx], stage1_acc[idx]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["null"],
                "linewidth": 0.85,
                "mutation_scale": 8,
                "shrinkA": 5,
                "shrinkB": 5,
            },
            zorder=1,
        )
        ax.annotate(
            f"{int(budget)}",
            xy=(final_len[idx], final_acc[idx]),
            xytext=label_offsets[int(budget)],
            textcoords="offset points",
            fontsize=5.8,
            color=COLORS["pink"],
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.3},
            zorder=5,
        )
    ax.plot(
        stage1_len,
        stage1_acc,
        "o-",
        color=COLORS["green"],
        markeredgecolor=COLORS["text"],
        markeredgewidth=0.35,
        markersize=4.6,
        linewidth=1.25,
        label="TRACE Stage 1",
        zorder=3,
    )
    ax.plot(
        final_len,
        final_acc,
        "o-",
        color=COLORS["pink"],
        markeredgecolor=COLORS["text"],
        markeredgewidth=0.35,
        markersize=4.8,
        linewidth=1.45,
        label="TRACE Final",
        zorder=4,
    )
    ax.set_xlabel("Realized total reasoning length #L")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("Matched accuracy-length frontier", loc="left", fontweight="bold")
    ax.legend(loc="lower right", handlelength=1.8)
    panel_label(ax, "a")

    ax = axes[1]
    ax.axhline(0, color=COLORS["muted"], linestyle="--", linewidth=0.75)
    ax.errorbar(
        budgets,
        gain,
        yerr=[gain - gain_low, gain_high - gain],
        fmt="o-",
        color=COLORS["pink"],
        ecolor=COLORS["blue"],
        markeredgecolor=COLORS["text"],
        markeredgewidth=0.35,
        markersize=4.8,
        linewidth=1.35,
        elinewidth=0.95,
        capsize=2.0,
    )
    ax.set_xticks(budgets)
    ax.set_xlabel("Maximum answer tokens")
    ax.set_ylabel("Accuracy gain (pp)")
    ax.set_title("Outcome gain", loc="left", fontweight="bold")
    panel_label(ax, "b")

    ax = axes[2]
    ax.axhline(0, color=COLORS["muted"], linestyle="--", linewidth=0.75)
    ax.errorbar(
        budgets,
        saved,
        yerr=[saved - saved_low, saved_high - saved],
        fmt="o-",
        color=COLORS["gold"],
        ecolor=COLORS["pink"],
        markeredgecolor=COLORS["text"],
        markeredgewidth=0.35,
        markersize=4.8,
        linewidth=1.35,
        elinewidth=0.95,
        capsize=2.0,
    )
    ax.set_xticks(budgets)
    ax.set_xlabel("Maximum answer tokens")
    ax.set_ylabel("Reasoning length saved")
    ax.set_title("No length inflation", loc="left", fontweight="bold")
    panel_label(ax, "c")

    fig.text(
        0.5,
        0.012,
        "Full GSM8K test set; one deterministic decode per question. Error bars: paired question-bootstrap 95% CIs.",
        ha="center",
        fontsize=5.9,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.072, right=0.99, top=0.86, bottom=0.25, wspace=0.35)
    save_figure(fig, output_dir / "fig_accuracy_length_pareto")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for budget_dir in sorted((args.raw_root / "stage1").glob("budget*")):
        try:
            budget = int(budget_dir.name.removeprefix("budget"))
        except ValueError:
            continue
        stage1_path = find_result(budget_dir)
        final_path = find_result(args.raw_root / "final" / f"budget{budget}")
        if stage1_path is None or final_path is None:
            continue
        summaries.append(
            matched_summary(
                budget,
                load_result(stage1_path),
                load_result(final_path),
                args.bootstrap_trials,
                args.seed,
            )
        )
    summaries.sort(key=lambda row: int(row["max_new_tokens"]))
    write_csv(args.output_dir / "source_data" / "accuracy_length_pareto.csv", summaries)
    payload = {
        "experiment": "matched_accuracy_length_pareto",
        "dataset": "GSM8K test",
        "test_times": 1,
        "bootstrap_trials": args.bootstrap_trials,
        "rows": summaries,
    }
    (args.output_dir / "accuracy_length_pareto.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if summaries:
        write_tables(args.output_dir, summaries)
        draw_figure(args.output_dir, summaries)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
