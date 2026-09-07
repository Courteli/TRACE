#!/usr/bin/env python3
"""Summarize and plot the TRACE hidden-prefix causal-access experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("stage1", "final")
MODEL_LABELS = {
    "stage1": "TRACE Stage 1",
    "final": "TRACE Final",
}
COLORS = {
    "stage1": "#69B17D",
    "final": "#E5A6C4",
    "gold": "#E6A314",
    "ink": "#2E3440",
    "muted": "#667080",
    "grid": "#D9DEE7",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def numeric_records(payload: dict) -> dict[str, dict]:
    return {key: value for key, value in payload.items() if str(key).isdigit() and isinstance(value, dict)}


def scalar(value, default=float("nan")) -> float:
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def latest_result(run_dir: Path) -> Path | None:
    marker = run_dir / "completed_result.txt"
    if marker.exists():
        candidate = Path(marker.read_text().strip())
        if candidate.exists():
            return candidate
    candidates = sorted(run_dir.glob("**/test_*_gsm_pid*.json"))
    return candidates[-1] if candidates else None


def discover_runs(root: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    full_root = root / "full"
    for model in MODEL_ORDER:
        model_root = full_root / model
        if not model_root.exists():
            continue
        for run_dir in sorted(model_root.glob("k*")):
            try:
                k = int(run_dir.name[1:])
            except ValueError:
                continue
            result = latest_result(run_dir)
            if result is None:
                continue
            payload = json.loads(result.read_text())
            records = numeric_records(payload)
            if records:
                runs[model][k] = {"path": result, "records": records}
    return runs


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return center - half, center + half


def paired_bootstrap_ci(delta: np.ndarray, seed: int = 20260716, reps: int = 10000) -> tuple[float, float]:
    if delta.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    chunk = 500
    for start in range(0, reps, chunk):
        stop = min(start + chunk, reps)
        indices = rng.integers(0, delta.size, size=(stop - start, delta.size))
        means[start:stop] = delta[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def exact_mcnemar_p(rescued: int, lost: int) -> float:
    discordant = rescued + lost
    if discordant == 0:
        return 1.0
    tail = min(rescued, lost)
    probability = sum(math.comb(discordant, i) for i in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def aligned_arrays(left: dict, right: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(set(left).intersection(right), key=int)
    a = np.asarray([scalar(left[idx].get("acc"), 0.0) for idx in ids], dtype=np.float64)
    b = np.asarray([scalar(right[idx].get("acc"), 0.0) for idx in ids], dtype=np.float64)
    return ids, a, b


def summarize(runs: dict[str, dict[int, dict]]) -> tuple[list[dict], list[dict], list[dict]]:
    summary_rows: list[dict] = []
    pair_rows: list[dict] = []
    source_rows: list[dict] = []

    for model in MODEL_ORDER:
        for k, run in sorted(runs.get(model, {}).items()):
            records = run["records"]
            ids = sorted(records, key=int)
            acc = np.asarray([scalar(records[idx].get("acc"), 0.0) for idx in ids])
            lengths = np.asarray([scalar(records[idx].get("output_length")) for idx in ids])
            successes = int(acc.sum())
            low, high = wilson_interval(successes, len(ids))
            summary_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "k": k,
                    "n": len(ids),
                    "correct": successes,
                    "accuracy": float(acc.mean()),
                    "accuracy_ci_low": low,
                    "accuracy_ci_high": high,
                    "mean_output_length": float(np.nanmean(lengths)),
                    "result_json": str(run["path"]),
                }
            )
            for idx in ids:
                record = records[idx]
                source_rows.append(
                    {
                        "model": model,
                        "k": k,
                        "sample_id": idx,
                        "acc": scalar(record.get("acc"), 0.0),
                        "output_length": scalar(record.get("output_length")),
                        "pred_answer": scalar_or_text(record.get("pred_answer")),
                        "answer": scalar_or_text(record.get("answer")),
                        "question": scalar_or_text(record.get("question")),
                    }
                )

        available = sorted(runs.get(model, {}))
        for left_k, right_k in zip(available[:-1], available[1:]):
            _, left, right = aligned_arrays(
                runs[model][left_k]["records"],
                runs[model][right_k]["records"],
            )
            delta = right - left
            rescued = int(np.sum((left == 0) & (right == 1)))
            lost = int(np.sum((left == 1) & (right == 0)))
            low, high = paired_bootstrap_ci(delta, seed=20260716 + left_k * 101 + right_k)
            pair_rows.append(
                {
                    "model": model,
                    "left_k": left_k,
                    "right_k": right_k,
                    "n": int(delta.size),
                    "accuracy_delta": float(delta.mean()),
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "rescued": rescued,
                    "lost": lost,
                    "mcnemar_p": exact_mcnemar_p(rescued, lost),
                }
            )
    return summary_rows, pair_rows, source_rows


def scalar_or_text(value) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_figure(out_dir: Path, summary_rows: list[dict], pair_rows: list[dict]) -> list[str]:
    if len(summary_rows) < 2:
        return []
    configure_style()
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        by_model[row["model"]].append(row)
    for values in by_model.values():
        values.sort(key=lambda row: row["k"])

    fig = plt.figure(figsize=(7.2, 3.05), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.72, 1.0), wspace=0.33)
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_transition = fig.add_subplot(grid[0, 1])

    for model in MODEL_ORDER:
        rows = by_model.get(model, [])
        if not rows:
            continue
        x = np.asarray([row["k"] for row in rows], dtype=float)
        y = 100.0 * np.asarray([row["accuracy"] for row in rows])
        low = 100.0 * np.asarray([row["accuracy_ci_low"] for row in rows])
        high = 100.0 * np.asarray([row["accuracy_ci_high"] for row in rows])
        linewidth = 2.25 if model == "final" else 1.65
        markersize = 5.2 if model == "final" else 4.4
        ax_curve.fill_between(x, low, high, color=COLORS[model], alpha=0.13, linewidth=0)
        ax_curve.plot(
            x,
            y,
            color=COLORS[model],
            marker="o",
            markersize=markersize,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=linewidth,
            label=MODEL_LABELS[model],
            zorder=3,
        )

    ax_curve.set_title("Causal access to the latent trajectory", loc="left", pad=8)
    ax_curve.set_xlabel("Latent states accessible to the decoder, $k$")
    ax_curve.set_ylabel("GSM8K accuracy (%)")
    ax_curve.set_xticks(range(9))
    ax_curve.grid(axis="y", color=COLORS["grid"], linewidth=0.65, alpha=0.8)
    ax_curve.tick_params(axis="both", length=3, color=COLORS["muted"])
    ax_curve.legend(loc="best", handlelength=2.0)
    ax_curve.text(-0.13, 1.04, "a", transform=ax_curve.transAxes, fontsize=11, fontweight="bold")

    final_pairs = [row for row in pair_rows if row["model"] == "final"]
    final_pairs.sort(key=lambda row: row["left_k"])
    if final_pairs:
        x = np.arange(len(final_pairs))
        n = np.asarray([row["n"] for row in final_pairs], dtype=float)
        rescued = 100.0 * np.asarray([row["rescued"] for row in final_pairs]) / n
        lost = -100.0 * np.asarray([row["lost"] for row in final_pairs]) / n
        ax_transition.bar(x - 0.18, rescued, width=0.36, color=COLORS["final"], edgecolor=COLORS["ink"], linewidth=0.45, label="rescued")
        ax_transition.bar(x + 0.18, lost, width=0.36, color=COLORS["gold"], edgecolor=COLORS["ink"], linewidth=0.45, label="lost")
        ax_transition.axhline(0, color=COLORS["ink"], linewidth=0.75)
        labels = [f"{row['left_k']}→{row['right_k']}" for row in final_pairs]
        ax_transition.set_xticks(x, labels, rotation=45, ha="right")
        bound = max(1.0, float(max(np.max(rescued), np.max(np.abs(lost)))) * 1.25)
        ax_transition.set_ylim(-bound, bound)
        ax_transition.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=0.9)
    else:
        ax_transition.text(0.5, 0.5, "Final-model transitions pending", transform=ax_transition.transAxes, ha="center", va="center", color=COLORS["muted"])
        ax_transition.set_xticks([])
    ax_transition.set_title("Answer transitions in TRACE Final", loc="left", pad=8)
    ax_transition.set_xlabel("Newly accessible latent segment")
    ax_transition.set_ylabel("Questions changing outcome (%)")
    ax_transition.grid(axis="y", color=COLORS["grid"], linewidth=0.65, alpha=0.8)
    ax_transition.text(-0.20, 1.04, "b", transform=ax_transition.transAxes, fontsize=11, fontweight="bold")

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.86)
    stem = out_dir / "fig_hidden_prefix_causal_access"
    outputs = []
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600}),
    ):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def endpoint_report(runs: dict[str, dict[int, dict]], model: str) -> dict | None:
    model_runs = runs.get(model, {})
    if 0 not in model_runs or 8 not in model_runs:
        return None
    _, start, end = aligned_arrays(model_runs[0]["records"], model_runs[8]["records"])
    delta = end - start
    rescued = int(np.sum((start == 0) & (end == 1)))
    lost = int(np.sum((start == 1) & (end == 0)))
    low, high = paired_bootstrap_ci(delta, seed=20260716 + MODEL_ORDER.index(model))
    return {
        "model": model,
        "n": int(delta.size),
        "k0_accuracy": float(start.mean()),
        "k8_accuracy": float(end.mean()),
        "delta": float(delta.mean()),
        "delta_ci_low": low,
        "delta_ci_high": high,
        "rescued": rescued,
        "lost": lost,
        "mcnemar_p": exact_mcnemar_p(rescued, lost),
    }


def write_report(path: Path, runs: dict[str, dict[int, dict]], summary_rows: list[dict], figures: list[str]) -> None:
    lines = [
        "# Hidden-prefix causal-access report",
        "",
        "This intervention keeps all eight latent slots and answer positions fixed while limiting decoder access to the first k slots.",
        "",
        "## Completion",
        "",
        "| Model | Completed k |",
        "|---|---|",
    ]
    for model in MODEL_ORDER:
        completed = ", ".join(str(k) for k in sorted(runs.get(model, {}))) or "pending"
        lines.append(f"| {MODEL_LABELS[model]} | {completed} |")
    lines.extend(["", "## Endpoint effects", "", "| Model | k=0 | k=8 | Delta (95% paired CI) | Rescued / lost | McNemar p |", "|---|---:|---:|---:|---:|---:|"])
    endpoint_count = 0
    for model in MODEL_ORDER:
        report = endpoint_report(runs, model)
        if report is None:
            continue
        endpoint_count += 1
        lines.append(
            f"| {MODEL_LABELS[model]} | {100*report['k0_accuracy']:.2f}% | {100*report['k8_accuracy']:.2f}% | "
            f"{100*report['delta']:+.2f} [{100*report['delta_ci_low']:+.2f}, {100*report['delta_ci_high']:+.2f}] | "
            f"{report['rescued']} / {report['lost']} | {report['mcnemar_p']:.4g} |"
        )
    if not endpoint_count:
        lines.append("| pending | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A positive, progressive k-to-accuracy relation supports behavioral use of the ordered latent path. Flat or unstable curves are reported as evidence against that stronger claim; the k=8 parity preflight only validates the intervention implementation.",
            "",
            f"Completed result points: {len(summary_rows)} / 18.",
            f"Figure exports: {len(figures)}.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(root)
    summary_rows, pair_rows, source_rows = summarize(runs)
    write_csv(out_dir / "hidden_prefix_summary.csv", summary_rows)
    write_csv(out_dir / "hidden_prefix_pairwise.csv", pair_rows)
    write_csv(out_dir / "hidden_prefix_source_data.csv", source_rows)
    figures = plot_figure(out_dir, summary_rows, pair_rows)
    write_report(out_dir / "HIDDEN_PREFIX_REPORT.md", runs, summary_rows, figures)

    contract = {
        "core_conclusion": "Ordered latent states should provide progressively useful information to answer generation, and outcome refinement should preserve or strengthen this relation relative to TRACE Stage 1.",
        "figure_archetype": "quantitative grid",
        "target": "KDD two-column manuscript",
        "backend": "Python/matplotlib",
        "final_size_inches": [7.2, 3.05],
        "panel_a": "Absolute GSM8K accuracy as decoder access expands from k=0 to k=8.",
        "panel_b": "Per-segment wrong-to-correct rescues and correct-to-wrong losses for TRACE Final.",
        "statistics": "Wilson 95% intervals, paired bootstrap delta intervals, exact McNemar tests.",
        "source_data": "Per-question deterministic test JSON; test_times=1.",
        "reviewer_risk": "The intervention establishes causal access/use, but does not by itself assign human-readable semantics to individual states.",
    }
    (out_dir / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest = {
        "root": str(root),
        "completed_points": len(summary_rows),
        "expected_points": 18,
        "models": {model: sorted(runs.get(model, {})) for model in MODEL_ORDER},
        "figures": figures,
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
