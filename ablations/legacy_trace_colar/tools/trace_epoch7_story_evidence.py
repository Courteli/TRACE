#!/usr/bin/env python3
"""Mine and visualize reproducible TRACE epoch7 success stories.

This script intentionally separates representative positive cases from population
claims. Every case is selected by a recorded rule and accompanied by its eligible
set size, so qualitative figures remain useful without becoming hidden cherry-picks.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch


COLORS = {
    "blue": "#5B7DB1",
    "green": "#66B07A",
    "gold": "#E6A516",
    "pink": "#E5A6C4",
    "bridge": "#5B7DB1",
    "stage1": "#66B07A",
    "trace": "#E6A516",
    "seed1": "#E5A6C4",
    "correct": "#66B07A",
    "wrong": "#E5A6C4",
    "mixed": "#E6A516",
    "grid": "#DDE1E6",
    "text": "#29313A",
    "muted": "#69727D",
}

FONT_STACK = ["Times New Roman", "Liberation Serif", "Nimbus Roman No9 L", "DejaVu Serif"]
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": FONT_STACK,
    "font.size": 7.5,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})

SIMILARITY_CMAP = LinearSegmentedColormap.from_list(
    "trace_blue_pink",
    [COLORS["blue"], "#F7F7F5", COLORS["pink"]],
)
ASSIGNMENT_CMAP = LinearSegmentedColormap.from_list(
    "trace_assignment",
    ["#FAFAF8", COLORS["green"], COLORS["blue"]],
)

METHOD_COLORS = {
    "BRIDGE": COLORS["bridge"],
    "Stage1": COLORS["stage1"],
    "TRACE epoch7": COLORS["trace"],
    "TRACE seed1": COLORS["seed1"],
}


def load_records(path: Path) -> dict[int, dict]:
    records = torch.load(path, map_location="cpu", weights_only=False)
    return {int(record["idx"]): record for record in records}


def load_geometry(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rollout_signature_separation"]["per_question"]
    return {int(row["idx"]): row for row in rows}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized(array: np.ndarray) -> np.ndarray:
    return array / np.clip(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8, None)


def path_signature(path: np.ndarray) -> np.ndarray:
    mean_state = path.mean(axis=0)
    first = path[0]
    last = path[-1]
    trend = last - first
    delta = np.diff(path, axis=0).mean(axis=0) if len(path) > 1 else np.zeros_like(first)
    pieces = []
    for weight, value in zip((0.5, 0.5, 1.0, 1.0), (mean_state, last, trend, delta)):
        pieces.append(weight * normalized(value.reshape(1, -1))[0])
    return normalized(np.concatenate(pieces).reshape(1, -1))[0]


def stage2_signatures(record: dict, raw_mix: float = 0.25) -> np.ndarray:
    residuals = record["multiview_implicit_residuals"].float().numpy()
    raw = np.stack([path_signature(np.cumsum(view, axis=0)) for view in residuals])
    raw = normalized(raw)
    centered = normalized(raw - raw.mean(axis=0, keepdims=True))
    return normalized(np.concatenate([centered, raw_mix * raw], axis=-1))


def pca_scores(arrays: list[np.ndarray], n_components: int = 2) -> tuple[list[np.ndarray], np.ndarray]:
    lengths = [len(array) for array in arrays]
    stacked = np.concatenate(arrays, axis=0).astype(np.float64)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :n_components] * singular[:n_components]
    singular_sq = np.square(singular)
    explained = singular_sq / np.clip(singular_sq.sum(), 1e-12, None)
    split = []
    cursor = 0
    for length in lengths:
        split.append(scores[cursor : cursor + length])
        cursor += length
    return split, explained[:n_components]


def pairwise_distance(left: np.ndarray, right: np.ndarray | None = None, offdiag: bool = False) -> float | None:
    if len(left) == 0:
        return None
    left = normalized(left)
    right = left if right is None else normalized(right)
    if len(right) == 0:
        return None
    distance = 1.0 - left @ right.T
    if offdiag:
        if len(left) <= 1:
            return None
        distance = distance[~np.eye(len(left), dtype=bool)]
    return float(distance.mean()) if distance.size else None


def build_modes(signatures: np.ndarray, max_modes: int = 2, merge_threshold: float = 0.65) -> np.ndarray:
    signatures = normalized(signatures)
    prototypes = [signatures[0]]
    while len(prototypes) < min(max_modes, len(signatures)):
        current = normalized(np.stack(prototypes))
        nearest = (signatures @ current.T).max(axis=1)
        candidate = int(np.argmin(nearest))
        if nearest[candidate] >= merge_threshold:
            break
        prototypes.append(signatures[candidate])
    prototypes = normalized(np.stack(prototypes))
    for _ in range(2):
        assignments = np.argmax(signatures @ prototypes.T, axis=1)
        updated = []
        for mode_idx in range(len(prototypes)):
            members = signatures[assignments == mode_idx]
            updated.append(normalized(members.mean(axis=0, keepdims=True))[0] if len(members) else prototypes[mode_idx])
        prototypes = np.stack(updated)
    return prototypes


def wrong_rejection_auc(signatures: np.ndarray, outcomes: np.ndarray) -> float | None:
    positive = signatures[outcomes]
    negative = signatures[~outcomes]
    if len(positive) <= 1 or len(negative) == 0:
        return None
    prototypes = build_modes(positive)
    wrong_scores = 1.0 - (normalized(negative) @ normalized(prototypes).T).max(axis=1)
    correct_scores = []
    for positive_idx in range(len(positive)):
        leave_one_out = np.delete(positive, positive_idx, axis=0)
        leave_one_out_prototypes = build_modes(leave_one_out)
        score = 1.0 - (
            normalized(positive[positive_idx : positive_idx + 1]) @ normalized(leave_one_out_prototypes).T
        ).max()
        correct_scores.append(float(score))
    comparison = wrong_scores[:, None] - np.asarray(correct_scores)[None, :]
    return float((comparison > 0).mean() + 0.5 * (np.abs(comparison) <= 1e-12).mean())


def exact_outcome_null(signatures: np.ndarray, outcomes: np.ndarray) -> tuple[float, float, float]:
    observed = wrong_rejection_auc(signatures, outcomes)
    if observed is None:
        return np.nan, np.nan, np.nan
    null_values = []
    for correct_indices in combinations(range(len(outcomes)), int(outcomes.sum())):
        labels = np.zeros(len(outcomes), dtype=bool)
        labels[list(correct_indices)] = True
        value = wrong_rejection_auc(signatures, labels)
        if value is not None:
            null_values.append(value)
    null_mean = float(np.mean(null_values))
    return float(observed), null_mean, float(observed - null_mean)


def view_residual_metric(signatures: np.ndarray, outcomes: np.ndarray) -> dict | None:
    positive = signatures[outcomes]
    negative = signatures[~outcomes]
    if len(positive) < 2 or len(negative) < 2:
        return None
    correct_within = pairwise_distance(positive, offdiag=True)
    wrong_within = pairwise_distance(negative, offdiag=True)
    cross = pairwise_distance(positive, negative)
    auc, null_mean, auc_excess = exact_outcome_null(signatures, outcomes)
    return {
        "n_correct": int(len(positive)),
        "n_wrong": int(len(negative)),
        "correct_within_distance": correct_within,
        "wrong_within_distance": wrong_within,
        "correct_wrong_distance": cross,
        "correct_wrong_margin": float(cross - correct_within),
        "wrong_dispersion_gap": float(wrong_within - correct_within),
        "wrong_rejection_auc": auc,
        "wrong_rejection_auc_exact_null_mean": null_mean,
        "wrong_rejection_auc_excess_over_exact_null": auc_excess,
    }


def build_view_residualized_diagnostics(record_sets: dict[str, dict[int, dict]]) -> dict:
    signatures_by_seed = {}
    metrics_by_seed = {}
    for seed_label, records in record_sets.items():
        raw = {idx: normalized(np.stack([
            path_signature(np.cumsum(view, axis=0))
            for view in record["multiview_implicit_residuals"].float().numpy()
        ])) for idx, record in records.items()}
        common_indices = sorted(raw)
        signature_sum = np.stack([raw[idx] for idx in common_indices]).sum(axis=0)
        residualized = {}
        metrics = {}
        for idx in common_indices:
            leave_one_out_view_template = (signature_sum - raw[idx]) / max(len(common_indices) - 1, 1)
            residual = normalized(raw[idx] - leave_one_out_view_template)
            residual = normalized(residual - residual.mean(axis=0, keepdims=True))
            residualized[idx] = residual
            outcomes = records[idx]["multiview_acc"].float().numpy() > 0.5
            metrics[idx] = view_residual_metric(residual, outcomes)
        signatures_by_seed[seed_label] = residualized
        metrics_by_seed[seed_label] = metrics
    return {
        "definition": (
            "For each seed and question, subtract each view's mean raw signature estimated from the other 199 questions, "
            "then center and normalize within the question. Outcome metrics use all eight residualized views."
        ),
        "null": "Exact enumeration of every label assignment preserving the question's number correct.",
        "signatures": signatures_by_seed,
        "metrics": metrics_by_seed,
    }


def style_axis(ax, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#555555")
    ax.spines[["left", "bottom"]].set_linewidth(0.75)
    ax.tick_params(colors="#4B5563", direction="out", length=2.5, width=0.7)
    ax.grid(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)


def save_figure(fig, output_path: Path, dpi: int = 400) -> None:
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="left", va="bottom")


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    return np.quantile(bootstrap_means, 0.025, axis=0), np.quantile(bootstrap_means, 0.975, axis=0)


def assignment_progress(record: dict) -> np.ndarray:
    assignment = record["assignment"].float().numpy()
    positions = np.linspace(0.0, 1.0, assignment.shape[1])
    return assignment @ positions


def plot_population_progress_profile(output_path: Path, records: dict[str, dict[int, dict]]) -> dict:
    labels = ["BRIDGE", "Stage1", "TRACE epoch7", "TRACE seed1"]
    profiles = {}
    spans = {}
    inversions = {}
    for label in labels:
        common = sorted(records[label])
        profile = np.stack([assignment_progress(records[label][idx]) for idx in common])
        profiles[label] = profile
        spans[label] = profile.max(axis=1) - profile.min(axis=1)
        inversions[label] = (np.diff(profile, axis=1) < -1e-4).mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.50, 0.90, 0.95]})
    x = np.arange(1, profiles["BRIDGE"].shape[1] + 1)
    line_styles = {"BRIDGE": "-", "Stage1": "--", "TRACE epoch7": "-", "TRACE seed1": ":"}
    markers = {"BRIDGE": "o", "Stage1": "s", "TRACE epoch7": "o", "TRACE seed1": "D"}
    zorders = {"BRIDGE": 2, "Stage1": 5, "TRACE epoch7": 3, "TRACE seed1": 6}
    for seed, label in enumerate(labels, start=71):
        mean = profiles[label].mean(axis=0)
        low, high = bootstrap_mean_ci(profiles[label], seed=seed)
        axes[0].plot(
            x,
            mean,
            marker=markers[label],
            linestyle=line_styles[label],
            linewidth=1.35,
            markersize=2.8,
            color=METHOD_COLORS[label],
            label=label,
            zorder=zorders[label],
        )
        axes[0].fill_between(x, low, high, color=METHOD_COLORS[label], alpha=0.075, linewidth=0, zorder=1)
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Latent slot")
    axes[0].set_ylabel("Expected CoT progress")
    axes[0].set_ylim(0.15, 0.52)
    axes[0].set_title("Latent progress", loc="left", fontweight="bold", pad=4)
    add_panel_label(axes[0], "a")
    style_axis(axes[0])

    box = axes[1].boxplot(
        [spans[label] for label in labels],
        tick_labels=labels,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.1},
        whiskerprops={"color": COLORS["muted"], "linewidth": 0.75},
        capprops={"color": COLORS["muted"], "linewidth": 0.75},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(METHOD_COLORS[label])
        patch.set_edgecolor("#4A4A4A")
        patch.set_linewidth(0.7)
        patch.set_alpha(0.92)
    axes[1].set_ylabel("Progress span")
    axes[1].set_title("Coverage", loc="left", fontweight="bold", pad=4)
    axes[1].tick_params(axis="x", rotation=28)
    add_panel_label(axes[1], "b")
    style_axis(axes[1])

    means = [100.0 * (1.0 - inversions[label].mean()) for label in labels]
    bars = axes[2].bar(
        labels,
        means,
        color=[METHOD_COLORS[label] for label in labels],
        edgecolor="#4A4A4A",
        linewidth=0.65,
        width=0.64,
    )
    axes[2].set_ylabel("Adjacent order consistency (%)")
    axes[2].set_title("Order consistency", loc="left", fontweight="bold", pad=4)
    axes[2].tick_params(axis="x", rotation=28)
    axes[2].set_ylim(70, 101.5)
    add_panel_label(axes[2], "c")
    for bar, label, consistency in zip(bars, labels, means):
        monotonic = 100.0 * float((inversions[label] == 0).mean())
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.65,
            f"{consistency:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=COLORS["text"],
        )
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            72.0,
            f"{monotonic:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=COLORS["text"],
        )
    style_axis(axes[2])

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.text(0.995, 0.01, "n=200; bands, 95% bootstrap CI", ha="right", va="bottom", fontsize=6.2, color=COLORS["muted"])
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.28, wspace=0.42)
    save_figure(fig, output_path)
    plt.close(fig)

    return {
        label: {
            "mean_progress_profile": [float(value) for value in profiles[label].mean(axis=0)],
            "mean_progress_span": float(spans[label].mean()),
            "mean_inversion_rate": float(inversions[label].mean()),
            "zero_inversion_fraction": float((inversions[label] == 0).mean()),
            "n_questions": int(len(profiles[label])),
        }
        for label in labels
    }


def plot_stage2_positive_dashboard(output_path: Path, paired_evidence_path: Path) -> dict:
    evidence = json.loads(paired_evidence_path.read_text(encoding="utf-8"))
    order = ["GSM8K", "GSMHard", "SVAMP", "MultiArith"]
    summaries = {row["dataset"]: row for row in evidence["summaries"]}
    rows = [summaries[dataset] for dataset in order]
    dataset_colors = {
        "GSM8K": COLORS["blue"],
        "GSMHard": COLORS["green"],
        "SVAMP": COLORS["gold"],
        "MultiArith": COLORS["pink"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.55))
    y = np.arange(len(rows))
    gains = np.asarray([row["accuracy_delta_pp"] for row in rows])
    gain_low = np.asarray([row["accuracy_delta_bootstrap_ci95_low"] for row in rows])
    gain_high = np.asarray([row["accuracy_delta_bootstrap_ci95_high"] for row in rows])
    axes[0, 0].errorbar(
        gains,
        y,
        xerr=np.vstack([gains - gain_low, gain_high - gains]),
        fmt="none",
        ecolor="#606A78",
        capsize=2.5,
        linewidth=0.9,
    )
    for idx, row in enumerate(rows):
        axes[0, 0].scatter(gains[idx], idx, s=25, color=dataset_colors[row["dataset"]], edgecolor="#4A4A4A", linewidth=0.35, zorder=3)
        axes[0, 0].text(gain_high[idx] + 0.12, idx, f"+{gains[idx]:.2f}", va="center", fontsize=6.4)
    axes[0, 0].axvline(0, color="#7A838E", linestyle="--", linewidth=0.75)
    axes[0, 0].set_yticks(y, order)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Paired accuracy gain (pp)")
    axes[0, 0].set_title("Accuracy gain with 95% CI", loc="left", fontweight="bold", pad=3)
    add_panel_label(axes[0, 0], "a")
    style_axis(axes[0, 0])

    saved = -np.asarray([row["length_delta"] for row in rows])
    saved_low = -np.asarray([row["length_delta_bootstrap_ci95_high"] for row in rows])
    saved_high = -np.asarray([row["length_delta_bootstrap_ci95_low"] for row in rows])
    axes[0, 1].errorbar(
        saved,
        y,
        xerr=np.vstack([saved - saved_low, saved_high - saved]),
        fmt="none",
        ecolor="#606A78",
        capsize=2.5,
        linewidth=0.9,
    )
    for idx, row in enumerate(rows):
        axes[0, 1].scatter(saved[idx], idx, s=25, color=dataset_colors[row["dataset"]], edgecolor="#4A4A4A", linewidth=0.35, zorder=3)
        axes[0, 1].text(saved_high[idx] + 0.10, idx, f"{saved[idx]:.2f}", va="center", fontsize=6.4)
    axes[0, 1].axvline(0, color="#7A838E", linestyle="--", linewidth=0.75)
    axes[0, 1].set_yticks(y, order)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Paired #L reduction")
    axes[0, 1].set_title("Reasoning-length reduction", loc="left", fontweight="bold", pad=3)
    add_panel_label(axes[0, 1], "b")
    style_axis(axes[0, 1])

    net_wins = np.asarray([row["net_rescues"] for row in rows])
    bars = axes[1, 0].barh(
        y,
        net_wins,
        color=[dataset_colors[dataset] for dataset in order],
        edgecolor="#4A4A4A",
        linewidth=0.65,
        height=0.56,
    )
    axes[1, 0].set_yticks(y, order)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Net paired answer improvements")
    axes[1, 0].set_title("Question-level net improvements", loc="left", fontweight="bold", pad=3)
    add_panel_label(axes[1, 0], "c")
    axes[1, 0].set_xlim(0, max(net_wins) * 1.28)
    for bar, row in zip(bars, rows):
        p_value = float(row["mcnemar_exact_p"])
        p_text = "ceiling set" if row["dataset"] == "MultiArith" else f"p={p_value:.1e}" if p_value < 0.001 else f"p={p_value:.3f}"
        axes[1, 0].text(
            bar.get_width() + max(net_wins) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"+{int(row['net_rescues'])}; {p_text}",
            ha="left",
            va="center",
            fontsize=6.0,
        )
    style_axis(axes[1, 0])

    for row in rows:
        color = dataset_colors[row["dataset"]]
        dataset = row["dataset"]
        point_x = -float(row["length_delta"])
        point_y = float(row["accuracy_delta_pp"])
        axes[1, 1].scatter(point_x, point_y, marker="o", s=34, color=color, edgecolor="#4A4A4A", linewidth=0.4, zorder=3)
        x_offset = 0.10
        y_offset = 0.10 if dataset != "MultiArith" else 0.18
        axes[1, 1].text(point_x + x_offset, point_y + y_offset, dataset, fontsize=6.1, color=COLORS["text"])
    axes[1, 1].axvline(0, color="#7A838E", linestyle="--", linewidth=0.70)
    axes[1, 1].axhline(0, color="#7A838E", linestyle="--", linewidth=0.70)
    axes[1, 1].set_xlim(-0.15, max(saved) + 0.75)
    axes[1, 1].set_ylim(-0.20, max(gains) + 0.85)
    axes[1, 1].set_xlabel("#L reduction (right is better)")
    axes[1, 1].set_ylabel("Accuracy gain (pp)")
    axes[1, 1].set_title("Accuracy-efficiency gain", loc="left", fontweight="bold", pad=3)
    add_panel_label(axes[1, 1], "d")
    axes[1, 1].text(
        0.98,
        0.94,
        "all four datasets are win-win",
        transform=axes[1, 1].transAxes,
        fontsize=6.2,
        color=COLORS["muted"],
        ha="right",
    )
    style_axis(axes[1, 1])

    fig.text(
        0.995,
        0.005,
        "Stage1 vs TRACE epoch7; paired bootstrap 95% CI; exact McNemar test.",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.115, right=0.985, top=0.93, bottom=0.145, hspace=0.68, wspace=0.46)
    save_figure(fig, output_path)
    plt.close(fig)
    return {
        "reference": evidence["reference"],
        "candidate": evidence["candidate"],
        "datasets": rows,
        "display_policy": "Main-text positive aggregate: gains, savings, net paired wins, and Pareto movement. Full transition counts remain in the audit files.",
    }


def compact_output(value: str, width: int = 92) -> str:
    return textwrap.shorten(" ".join(str(value).split()), width=width, placeholder=" ...")


def composition(row: dict) -> str:
    n_correct = int(row["n_correct"])
    if n_correct == 0:
        return "all_wrong"
    if n_correct == int(row["n_paths"]):
        return "all_correct"
    return "mixed"


def geometry_ideal(row0: dict, row1: dict) -> bool:
    keys = (
        "cross_minus_correct_within_mode",
        "wrong_minus_correct_within_mode",
        "wrong_rejection_auc_excess_over_null",
    )
    return all(row0.get(key) is not None and row1.get(key) is not None for key in keys) and all(
        row0[key] > 0 and row1[key] > 0 for key in keys
    )


def select_cases(
    paired_rows: list[dict],
    bridge_records: dict[int, dict],
    stage1_records: dict[int, dict],
    trace_records: dict[int, dict],
    seed1_records: dict[int, dict],
    stage1_geometry: dict[int, dict],
    trace_geometry: dict[int, dict],
    seed1_geometry: dict[int, dict],
    view_residualized: dict,
    causal_rows: list[dict],
) -> dict:
    paired = {
        int(row["idx"]): row
        for row in paired_rows
        if row["dataset"] == "GSM8K" and int(row["idx"]) in trace_records
    }
    rescues = [
        idx
        for idx, row in paired.items()
        if row["transition"] == "rescued" and float(row["delta_L"]) <= 0
    ]
    robust_geometry = [
        idx
        for idx in sorted(set(trace_geometry) & set(seed1_geometry))
        if geometry_ideal(trace_geometry[idx], seed1_geometry[idx])
    ]

    all_wrong_recovery = [
        idx
        for idx in rescues
        if idx in robust_geometry
        and int(stage1_geometry[idx]["n_correct"]) == 0
        and int(trace_geometry[idx]["n_correct"]) > 0
    ]
    all_wrong_recovery.sort(
        key=lambda idx: (
            int(trace_geometry[idx]["n_correct"]) - int(stage1_geometry[idx]["n_correct"]),
            -float(paired[idx]["delta_L"]),
        ),
        reverse=True,
    )
    recovery_case = all_wrong_recovery[0]

    robust_rescues = [idx for idx in rescues if idx in robust_geometry and idx != recovery_case]
    robust_rescues.sort(
        key=lambda idx: (
            -float(paired[idx]["delta_L"]),
            int(trace_geometry[idx]["n_correct"]) - int(stage1_geometry[idx]["n_correct"]),
        ),
        reverse=True,
    )
    robust_rescue_case = robust_rescues[0]

    consensus_candidates = [
        idx
        for idx in rescues
        if int(trace_geometry[idx]["n_correct"]) == 8
        and int(seed1_geometry[idx]["n_correct"]) == 8
    ]
    consensus_candidates.sort(
        key=lambda idx: (
            int(trace_geometry[idx]["n_correct"]) - int(stage1_geometry[idx]["n_correct"]),
            -float(paired[idx]["delta_L"]),
        ),
        reverse=True,
    )
    consensus_case = consensus_candidates[0]

    structural_scores = []
    for idx in rescues:
        bridge_assignment = bridge_records[idx]["assignment"].float().numpy()
        trace_assignment = trace_records[idx]["assignment"].float().numpy()
        if trace_assignment.shape[1] <= 1:
            continue
        positions = np.linspace(0.0, 1.0, trace_assignment.shape[1])
        bridge_center = bridge_assignment @ positions
        trace_center = trace_assignment @ positions
        bridge_span = float(bridge_center.max() - bridge_center.min())
        trace_span = float(trace_center.max() - trace_center.min())
        bridge_inv = float((np.diff(bridge_center) < -1e-4).mean())
        trace_inv = float((np.diff(trace_center) < -1e-4).mean())
        score = (trace_span - bridge_span) + (bridge_inv - trace_inv)
        structural_scores.append((score, idx))
    structural_scores.sort(reverse=True)
    structure_case = structural_scores[0][1]

    residual_metrics0 = view_residualized["metrics"]["seed0"]
    residual_metrics1 = view_residualized["metrics"]["seed1"]
    eligible_geometry = [
        idx
        for idx in sorted(set(residual_metrics0) & set(residual_metrics1))
        if residual_metrics0[idx] is not None and residual_metrics1[idx] is not None
    ]
    required_residual = (
        "correct_wrong_margin",
        "wrong_dispersion_gap",
        "wrong_rejection_auc_excess_over_exact_null",
    )
    strong_geometry = [
        idx
        for idx in eligible_geometry
        if all(
            residual_metrics0[idx][key] > 0 and residual_metrics1[idx][key] > 0
            for key in required_residual
        )
    ]
    local_geometry_case = max(
        strong_geometry,
        key=lambda idx: min(
            residual_metrics0[idx]["wrong_rejection_auc_excess_over_exact_null"],
            residual_metrics1[idx]["wrong_rejection_auc_excess_over_exact_null"],
        ),
    )
    residualized_rescue_cases = [idx for idx in rescues if idx in strong_geometry]

    causal_by_idx = defaultdict(list)
    for row in causal_rows:
        causal_by_idx[int(row["idx"])].append(row)
    normal_correct = sum(float(rows[0]["normal_acc"]) > 0.5 for rows in causal_by_idx.values())
    fully_sensitive = []
    harmed_at_least_three = []
    for idx, rows in causal_by_idx.items():
        harms = sum(float(row["normal_acc"]) > 0.5 and float(row["intervention_acc"]) < 0.5 for row in rows)
        if harms == 4:
            fully_sensitive.append(idx)
        if harms >= 3:
            harmed_at_least_three.append(idx)

    return {
        "structure_case": structure_case,
        "rescue_cases": [recovery_case, robust_rescue_case, consensus_case],
        "local_geometry_case": local_geometry_case,
        "local_geometry_rescue_case": residualized_rescue_cases[0] if residualized_rescue_cases else None,
        "causal_cases": sorted(fully_sensitive),
        "sets": {
            "first200_shorter_main_rescues": sorted(rescues),
            "view_residualized_two_seed_geometry_eligible": eligible_geometry,
            "view_residualized_two_seed_geometry_ideal": strong_geometry,
            "view_residualized_geometry_rescues": residualized_rescue_cases,
            "two_seed_consensus_rescue_candidates": consensus_candidates,
            "all_wrong_recovery_candidates": all_wrong_recovery,
            "fully_intervention_sensitive": sorted(fully_sensitive),
            "harmed_by_at_least_three_interventions": sorted(harmed_at_least_three),
        },
        "counts": {
            "causal_normal_correct": normal_correct,
            "causal_all_questions": len(causal_by_idx),
        },
        "rules": {
            "structure_case": (
                "Among first-200 Stage1-to-epoch7 main-answer rescues with non-increasing #L and at least two CoT steps, "
                "maximize the increase in assignment progress span plus the reduction in inversion versus BRIDGE."
            ),
            "rescue_case_1": (
                "Among shorter main-answer rescues with positive margin, wrong dispersion, and AUC excess on both seeds, "
                "select the largest Stage1 all-wrong recovery."
            ),
            "rescue_case_2": (
                "Among remaining shorter main-answer rescues passing the same two-seed local geometry rule, "
                "select the largest #L reduction."
            ),
            "rescue_case_3": (
                "Among shorter main-answer rescues that are 8/8 correct on both epoch7 seeds, "
                "select the largest Stage1-to-epoch7 rollout gain."
            ),
            "local_geometry_case": (
                "After leave-one-question-out subtraction of every fixed view template, select questions eligible for margin, "
                "wrong dispersion, and exact-null AUC excess on both seeds and positive on all three; maximize the worse-seed AUC excess."
            ),
            "causal_cases": "Show every question whose correct normal prediction is broken by all four frozen interventions.",
        },
    }


def export_audit_tables(
    output_dir: Path,
    selection: dict,
    paired: dict[int, dict],
    records: dict[str, dict[int, dict]],
    stage1_geometry: dict[int, dict],
    trace_geometry: dict[int, dict],
    view_residualized: dict,
    causal_rows: list[dict],
) -> dict:
    category_rank = {"all_wrong": 0, "mixed": 1, "all_correct": 2}
    transition_rows = []
    for idx in sorted(set(stage1_geometry) & set(trace_geometry)):
        stage1_category = composition(stage1_geometry[idx])
        trace_category = composition(trace_geometry[idx])
        rank_delta = category_rank[trace_category] - category_rank[stage1_category]
        pair = paired.get(idx, {})
        transition_rows.append({
            "idx": idx,
            "question": records["TRACE epoch7"][idx].get("question", ""),
            "stage1_n_correct": int(stage1_geometry[idx]["n_correct"]),
            "trace_n_correct": int(trace_geometry[idx]["n_correct"]),
            "delta_n_correct": int(trace_geometry[idx]["n_correct"] - stage1_geometry[idx]["n_correct"]),
            "stage1_category": stage1_category,
            "trace_category": trace_category,
            "category_transition": "improved" if rank_delta > 0 else "regressed" if rank_delta < 0 else "stable",
            "stage1_main_acc": pair.get("reference_acc", ""),
            "trace_main_acc": pair.get("candidate_acc", ""),
            "stage1_main_L": pair.get("reference_L", ""),
            "trace_main_L": pair.get("candidate_L", ""),
            "main_delta_L": pair.get("delta_L", ""),
            "selected_rescue_figure": idx in selection["rescue_cases"],
        })
    write_csv(output_dir / "rollout_transition_rows.csv", transition_rows)

    structure_rows = []
    for idx in selection["sets"]["first200_shorter_main_rescues"]:
        row = {"idx": idx, "question": records["TRACE epoch7"][idx].get("question", "")}
        centers = {}
        for label in ("BRIDGE", "Stage1", "TRACE epoch7"):
            assignment = records[label][idx]["assignment"].float().numpy()
            positions = np.linspace(0.0, 1.0, assignment.shape[1])
            centers[label] = assignment @ positions
            prefix = label.lower().replace(" ", "_")
            row[f"{prefix}_span"] = float(centers[label].max() - centers[label].min())
            row[f"{prefix}_inversion_rate"] = float((np.diff(centers[label]) < -1e-4).mean())
        row["selection_score"] = (
            row["trace_epoch7_span"] - row["bridge_span"]
            + row["bridge_inversion_rate"] - row["trace_epoch7_inversion_rate"]
        )
        row["selected_structure_figure"] = idx == selection["structure_case"]
        structure_rows.append(row)
    structure_rows.sort(key=lambda row: row["selection_score"], reverse=True)
    write_csv(output_dir / "stage1_structure_candidates.csv", structure_rows)

    seed0_metrics = view_residualized["metrics"]["seed0"]
    seed1_metrics = view_residualized["metrics"]["seed1"]
    ideal = set(selection["sets"]["view_residualized_two_seed_geometry_ideal"])
    geometry_rows = []
    for idx in selection["sets"]["view_residualized_two_seed_geometry_eligible"]:
        pair = paired.get(idx, {})
        row = {
            "idx": idx,
            "question": records["TRACE epoch7"][idx].get("question", ""),
            "passes_all_three_signs_both_seeds": idx in ideal,
            "is_shorter_main_answer_rescue": idx in selection["sets"]["first200_shorter_main_rescues"],
            "selected_local_geometry_figure": idx in {
                selection["local_geometry_case"], selection["local_geometry_rescue_case"]
            },
            "main_transition": pair.get("transition", ""),
            "main_delta_L": pair.get("delta_L", ""),
        }
        for seed_label, metrics in (("seed0", seed0_metrics[idx]), ("seed1", seed1_metrics[idx])):
            for key, value in metrics.items():
                row[f"{seed_label}_{key}"] = value
        row["worst_seed_auc_excess"] = min(
            seed0_metrics[idx]["wrong_rejection_auc_excess_over_exact_null"],
            seed1_metrics[idx]["wrong_rejection_auc_excess_over_exact_null"],
        )
        geometry_rows.append(row)
    geometry_rows.sort(key=lambda row: row["worst_seed_auc_excess"], reverse=True)
    write_csv(output_dir / "residualized_geometry_candidates.csv", geometry_rows)

    causal_by_idx = defaultdict(list)
    for row in causal_rows:
        causal_by_idx[int(row["idx"])].append(row)
    causal_candidates = []
    for idx, rows in causal_by_idx.items():
        if float(rows[0]["normal_acc"]) <= 0.5:
            continue
        harmed = [row["intervention"] for row in rows if float(row["intervention_acc"]) < 0.5]
        causal_candidates.append({
            "idx": idx,
            "question": rows[0]["question"],
            "normal_pred": rows[0]["normal_pred"],
            "harm_count": len(harmed),
            "harming_interventions": ";".join(sorted(harmed)),
            "selected_all_four_figure": idx in selection["causal_cases"],
        })
    causal_candidates.sort(key=lambda row: (-row["harm_count"], row["idx"]))
    write_csv(output_dir / "causal_normal_correct_candidates.csv", causal_candidates)

    return {
        "rollout_transition_rows": len(transition_rows),
        "stage1_structure_candidates": len(structure_rows),
        "residualized_geometry_candidates": len(geometry_rows),
        "causal_normal_correct_candidates": len(causal_candidates),
    }


def plot_transition_matrix(
    output_path: Path,
    stage1_geometry: dict[int, dict],
    trace_geometry: dict[int, dict],
) -> dict:
    labels = ["All wrong", "Mixed", "All correct"]
    keys = ["all_wrong", "mixed", "all_correct"]
    matrix = np.zeros((3, 3), dtype=int)
    for idx in sorted(set(stage1_geometry) & set(trace_geometry)):
        matrix[keys.index(composition(stage1_geometry[idx])), keys.index(composition(trace_geometry[idx]))] += 1

    improved = int(sum(matrix[row, col] for row in range(3) for col in range(3) if col > row))
    regressed = int(sum(matrix[row, col] for row in range(3) for col in range(3) if col < row))
    stable = int(np.trace(matrix))
    stage1_acc = np.mean([stage1_geometry[idx]["n_correct"] / 8 for idx in stage1_geometry]) * 100
    trace_acc = np.mean([trace_geometry[idx]["n_correct"] / 8 for idx in trace_geometry]) * 100

    fig, (ax, side) = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.18, 0.82]})
    background = np.zeros((3, 3, 3), dtype=float)
    for row in range(3):
        for col in range(3):
            if col > row:
                background[row, col] = 0.30 * np.asarray(matplotlib.colors.to_rgb(COLORS["green"])) + 0.70
            elif col < row:
                background[row, col] = 0.30 * np.asarray(matplotlib.colors.to_rgb(COLORS["pink"])) + 0.70
            else:
                background[row, col] = 0.24 * np.asarray(matplotlib.colors.to_rgb(COLORS["blue"])) + 0.76
    ax.imshow(background)
    for row in range(3):
        for col in range(3):
            suffix = "improve" if col > row else "regress" if col < row else "stable"
            ax.text(col, row, f"{matrix[row, col]}\n{suffix}", ha="center", va="center", fontsize=7.0, fontweight="bold")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("TRACE epoch7 rollout state")
    ax.set_ylabel("Stage1 rollout state")
    ax.set_title("Rollout-state transitions", loc="left", fontweight="bold", pad=4)
    add_panel_label(ax, "a")
    for spine in ax.spines.values():
        spine.set_visible(False)

    side.axis("off")
    side.text(0.0, 0.94, "Net Stage2 movement", fontsize=8.2, fontweight="bold", transform=side.transAxes)
    side.text(-0.08, 1.02, "b", fontsize=8.5, fontweight="bold", transform=side.transAxes)
    side.text(0.0, 0.72, f"+{improved - regressed}", fontsize=18, fontweight="bold", color=COLORS["green"], transform=side.transAxes)
    side.text(0.25, 0.75, "net category improvements", fontsize=7.0, color=COLORS["muted"], transform=side.transAxes)
    all_correct_gain = int(matrix[:, 2].sum() - matrix[2, :].sum())
    side.text(0.0, 0.50, f"+{all_correct_gain}", fontsize=18, fontweight="bold", color=COLORS["pink"], transform=side.transAxes)
    side.text(0.25, 0.53, "all-correct groups", fontsize=7.0, color=COLORS["muted"], transform=side.transAxes)
    side.text(
        0.0,
        0.22,
        f"Rollout accuracy: {stage1_acc:.2f}%  ->  {trace_acc:.2f}%\n"
        f"Paired gain: {trace_acc - stage1_acc:+.2f} pp\n"
        f"All-correct groups: {all_correct_gain:+d}",
        fontsize=7.0,
        linespacing=1.35,
        color=COLORS["text"],
        transform=side.transAxes,
    )
    side.text(
        0.0,
        0.04,
        "Same 200 GSM8K questions and the same 8 rollout views.",
        fontsize=6.0,
        color=COLORS["muted"],
        transform=side.transAxes,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.90, bottom=0.22, wspace=0.30)
    save_figure(fig, output_path)
    plt.close(fig)
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "improved": improved,
        "regressed": regressed,
        "stable": stable,
        "stage1_rollout_accuracy": float(stage1_acc),
        "trace_rollout_accuracy": float(trace_acc),
    }


def plot_structure_case(
    output_path: Path,
    idx: int,
    records: dict[str, dict[int, dict]],
    selection_rule: str,
) -> dict:
    methods = ["BRIDGE", "Stage1", "TRACE epoch7", "TRACE seed1"]
    record = records["TRACE epoch7"][idx]
    n_steps = len(record["steps"])
    fig = plt.figure(figsize=(7.2, 4.05))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.68], hspace=0.50, wspace=0.30)
    metadata = {}
    image = None
    centers_by_method = {}
    heat_axes = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(grid[0, col])
        heat_axes.append(ax)
        assignment = records[method][idx]["assignment"].float().numpy()
        assignment = assignment / np.clip(assignment.sum(axis=1, keepdims=True), 1e-8, None)
        image = ax.imshow(assignment, cmap=ASSIGNMENT_CMAP, vmin=0, vmax=max(0.75, float(assignment.max())), aspect="auto")
        positions = np.linspace(0.0, 1.0, assignment.shape[1])
        centers = assignment @ positions
        center_steps = centers * max(assignment.shape[1] - 1, 1)
        centers_by_method[method] = centers
        ax.plot(center_steps, np.arange(assignment.shape[0]), color="white", linewidth=1.1, marker="o", markersize=1.8)
        span = float(centers.max() - centers.min())
        inversion = float((np.diff(centers) < -1e-4).mean())
        metadata[method] = {"progress_span": span, "inversion_fraction": inversion}
        ax.set_title(f"{method}\nspan={span:.3f}; inv.={inversion:.3f}", fontweight="bold", color=METHOD_COLORS[method], pad=3)
        ax.set_xlabel("CoT step")
        if col == 0:
            ax.set_ylabel("Latent slot")
            add_panel_label(ax, "a", x=-0.34, y=1.04)
        else:
            ax.set_yticklabels([])
        ax.set_xticks(range(n_steps), [f"S{step + 1}" for step in range(n_steps)])
        ax.set_yticks(range(8), [f"z{slot + 1}" for slot in range(8)])
    fig.colorbar(image, ax=heat_axes, fraction=0.012, pad=0.012, label="Assignment")

    progress_ax = fig.add_subplot(grid[1, :2])
    slots = np.arange(1, 9)
    for method in methods:
        progress_ax.plot(
            slots,
            1.0 + centers_by_method[method] * (n_steps - 1),
            color=METHOD_COLORS[method],
            marker={"BRIDGE": "o", "Stage1": "s", "TRACE epoch7": "o", "TRACE seed1": "D"}[method],
            linestyle={"BRIDGE": "-", "Stage1": "--", "TRACE epoch7": "-", "TRACE seed1": ":"}[method],
            linewidth=1.2,
            markersize=2.6,
            label=method,
        )
    progress_ax.set_xticks(slots)
    progress_ax.set_yticks(range(1, n_steps + 1))
    progress_ax.set_xlabel("Latent slot")
    progress_ax.set_ylabel("Expected CoT step")
    progress_ax.legend(ncol=2, loc="upper left", handlelength=1.8, columnspacing=0.8)
    progress_ax.set_title("Early-to-late progress", loc="left", fontweight="bold", pad=3)
    add_panel_label(progress_ax, "b")
    style_axis(progress_ax)

    text_ax = fig.add_subplot(grid[1, 2:])
    text_ax.axis("off")
    stage1 = records["Stage1"][idx]
    trace = records["TRACE epoch7"][idx]
    step_lines = "\n".join(
        f"S{step_idx + 1}: {compact_output(step, 52)}" for step_idx, step in enumerate(record["steps"])
    )
    text_ax.text(
        0,
        1,
        f"q{idx}: {compact_output(record['question'], 135)}\n\n"
        f"Stage1: pred={stage1.get('pred_answer')} ({'correct' if stage1.get('acc') else 'wrong'}), #L={8 + int(stage1.get('output_length', 0))}\n"
        f"Epoch7: pred={trace.get('pred_answer')} ({'correct' if trace.get('acc') else 'wrong'}), #L={8 + int(trace.get('output_length', 0))}\n\n"
        f"{step_lines}",
        va="top",
        fontsize=6.6,
        linespacing=1.22,
        color=COLORS["text"],
        transform=text_ax.transAxes,
    )
    text_ax.set_title("Representative trace", loc="left", fontweight="bold", pad=3)
    add_panel_label(text_ax, "c", x=-0.06, y=1.04)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.91, bottom=0.10)
    save_figure(fig, output_path)
    plt.close(fig)
    return {"idx": idx, "metrics": metadata, "selection_rule": selection_rule}


def draw_outcome_strip(ax, y: float, outcomes: np.ndarray, label: str, color: str) -> None:
    for view_idx, outcome in enumerate(outcomes):
        ax.scatter(
            view_idx,
            y,
            s=58,
            marker="o" if outcome else "X",
            facecolor=COLORS["correct"] if outcome else COLORS["wrong"],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        ax.text(view_idx, y, str(view_idx), ha="center", va="center", color="white", fontsize=4.8, fontweight="bold", zorder=4)
    ax.text(-0.55, y, label, ha="right", va="center", fontsize=6.2, color=color, fontweight="bold")
    ax.text(7.55, y, f"{int(outcomes.sum())}/8", ha="left", va="center", fontsize=6.2, color=color, fontweight="bold")


def plot_rescue_cases(
    output_path: Path,
    case_ids: list[int],
    records: dict[str, dict[int, dict]],
    paired: dict[int, dict],
    rules: list[str],
) -> list[dict]:
    fig, axes = plt.subplots(2, len(case_ids), figsize=(7.2, 3.05), gridspec_kw={"height_ratios": [0.78, 1.22]})
    outputs = []
    category_titles = [
        "All-wrong recovery",
        "Replicated mixed rescue",
        "Two-seed consensus",
    ]
    for col, (idx, rule) in enumerate(zip(case_ids, rules)):
        text_ax = axes[0, col]
        text_ax.axis("off")
        stage1 = records["Stage1"][idx]
        trace = records["TRACE epoch7"][idx]
        row = paired[idx]
        question_text = textwrap.fill(compact_output(trace["question"], 76), width=38)
        stage1_output = textwrap.fill(compact_output(stage1.get("output_string", ""), 58), width=38)
        trace_output = textwrap.fill(compact_output(trace.get("output_string", ""), 58), width=38)
        text_ax.text(
            0,
            1,
            f"q{idx} | GT={trace['answer']} | delta #L={float(row['delta_L']):+.0f}\n"
            f"{question_text}\n"
            f"Stage1: wrong | #L={int(float(row['reference_L']))}\n"
            f"{stage1_output}\n"
            f"Epoch7: correct | #L={int(float(row['candidate_L']))}\n"
            f"{trace_output}",
            va="top",
            fontsize=5.55,
            linespacing=1.05,
            color=COLORS["text"],
            transform=text_ax.transAxes,
        )
        text_ax.set_title(category_titles[col], loc="left", fontweight="bold", pad=3)
        add_panel_label(text_ax, chr(ord("a") + col), x=-0.08, y=1.04)

        ax = axes[1, col]
        methods = ["BRIDGE", "Stage1", "TRACE epoch7", "TRACE seed1"]
        y_positions = [3, 2, 1, 0]
        for method, y in zip(methods, y_positions):
            outcomes = records[method][idx]["multiview_acc"].float().numpy() > 0.5
            draw_outcome_strip(ax, y, outcomes, method, METHOD_COLORS[method])
        ax.set_xlim(-1.35, 8.2)
        ax.set_ylim(-0.65, 3.65)
        ax.set_xticks(range(8), [f"v{view}" for view in range(8)], fontsize=5.8)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        outputs.append(
            {
                "idx": idx,
                "rule": rule,
                "main_delta_L": float(row["delta_L"]),
                "rollout_correct": {
                    method: int((records[method][idx]["multiview_acc"].float().numpy() > 0.5).sum())
                    for method in methods
                },
            }
        )
    fig.text(
        0.995,
        0.01,
        "Circle, correct; cross, wrong. Independent-seed replication is shown in pink.",
        ha="right",
        fontsize=6.0,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.18, hspace=0.01, wspace=0.34)
    save_figure(fig, output_path)
    plt.close(fig)
    return outputs


def plot_local_geometry_case(
    output_path: Path,
    idx: int,
    trace_record: dict,
    seed1_record: dict,
    geometry0: dict,
    geometry1: dict,
    selection_rule: str,
    ideal_count: int,
    eligible_count: int,
) -> dict:
    signatures = [stage2_signatures(trace_record), stage2_signatures(seed1_record)]
    scores, explained = pca_scores(signatures)
    outcomes = [
        trace_record["multiview_acc"].float().numpy() > 0.5,
        seed1_record["multiview_acc"].float().numpy() > 0.5,
    ]
    fig = plt.figure(figsize=(14.5, 10.5))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.32)
    panels = []
    for col, (label, score, signature, outcome, row) in enumerate(
        zip(("seed0", "seed1"), scores, signatures, outcomes, (geometry0, geometry1))
    ):
        scatter = fig.add_subplot(grid[0, col])
        for view_idx, (point, correct) in enumerate(zip(score, outcome)):
            scatter.scatter(
                point[0],
                point[1],
                s=95,
                marker="o" if correct else "X",
                color=COLORS["correct"] if correct else COLORS["wrong"],
                edgecolor="white",
                linewidth=1.1,
            )
            scatter.text(point[0], point[1], f" v{view_idx}", fontsize=8, va="bottom")
        scatter.set_title(f"{label}: correct={int(outcome.sum())}/8", fontsize=11, fontweight="bold")
        scatter.set_xlabel(f"local PC1 ({100 * explained[0]:.1f}%)")
        scatter.set_ylabel(f"local PC2 ({100 * explained[1]:.1f}%)")
        style_axis(scatter)

        order = np.concatenate([np.flatnonzero(outcome), np.flatnonzero(~outcome)])
        similarity = signature[order] @ signature[order].T
        heat = fig.add_subplot(grid[1, col])
        image = heat.imshow(similarity, vmin=-1, vmax=1, cmap=SIMILARITY_CMAP)
        heat_labels = [f"C:v{view}" if outcome[view] else f"W:v{view}" for view in order]
        heat.set_xticks(range(8), heat_labels, rotation=45, ha="right", fontsize=7.5)
        heat.set_yticks(range(8), heat_labels, fontsize=7.5)
        heat.set_title(f"{label}: centered signature similarity", fontsize=10)
        panels.append(
            {
                "seed": label,
                "n_correct": int(outcome.sum()),
                "margin": row["cross_minus_correct_within_mode"],
                "wrong_dispersion_gap": row["wrong_minus_correct_within_mode"],
                "auc_excess": row["wrong_rejection_auc_excess_over_null"],
            }
        )

    text_ax = fig.add_subplot(grid[:, 2])
    text_ax.axis("off")
    metrics_text = []
    for label, row in zip(("seed0", "seed1"), (geometry0, geometry1)):
        metrics_text.append(
            f"{label}\n"
            f"  correct/wrong margin: {row['cross_minus_correct_within_mode']:+.3f}\n"
            f"  wrong dispersion gap: {row['wrong_minus_correct_within_mode']:+.3f}\n"
            f"  rejection AUC excess: {row['wrong_rejection_auc_excess_over_null']:+.3f}"
        )
    text_ax.text(
        0,
        1,
        f"q{idx}\n{compact_output(trace_record['question'], 95)}\n\n"
        + "\n\n".join(metrics_text)
        + f"\n\nCoverage\n  {ideal_count}/{eligible_count} ({100 * ideal_count / eligible_count:.1f}%) eligible questions "
        "show positive margin, dispersion, and AUC excess on both seeds."
        "\n\nInterpretation\n  This is a reproducible local capability example, not a replacement for the negative global outcome-null result."
        "\n\nSelection\n  "
        + textwrap.fill(selection_rule, width=46),
        va="top",
        fontsize=9.2,
        linespacing=1.4,
        color=COLORS["text"],
        transform=text_ax.transAxes,
    )
    colorbar_ax = fig.add_axes([0.63, 0.11, 0.012, 0.27])
    fig.colorbar(image, cax=colorbar_ax, label="cosine similarity")
    fig.suptitle(
        f"Selected two-seed local outcome geometry capability (q{idx})",
        fontsize=16,
        fontweight="bold",
    )
    save_figure(fig, output_path)
    plt.close(fig)
    return {
        "idx": idx,
        "eligible_count": eligible_count,
        "ideal_count": ideal_count,
        "explained_variance_ratio": explained.tolist(),
        "panels": panels,
        "selection_rule": selection_rule,
    }


def plot_view_residualized_geometry_cases(
    output_path: Path,
    case_ids: list[int],
    records: dict[str, dict[int, dict]],
    diagnostics: dict,
    selection_rule: str,
    ideal_count: int,
    eligible_count: int,
) -> dict:
    fig, axes = plt.subplots(len(case_ids), 4, figsize=(7.2, 2.55 * len(case_ids)), squeeze=False)
    metadata = []
    last_image = None
    for row_idx, idx in enumerate(case_ids):
        signatures = [
            diagnostics["signatures"]["seed0"][idx],
            diagnostics["signatures"]["seed1"][idx],
        ]
        metrics = [
            diagnostics["metrics"]["seed0"][idx],
            diagnostics["metrics"]["seed1"][idx],
        ]
        scores, explained = pca_scores(signatures)
        outcomes = [
            records["TRACE epoch7"][idx]["multiview_acc"].float().numpy() > 0.5,
            records["TRACE seed1"][idx]["multiview_acc"].float().numpy() > 0.5,
        ]
        case_label = "strongest residualized case" if row_idx == 0 else "residualized geometry + main-answer rescue"
        question = records["TRACE epoch7"][idx]["question"]
        fig.text(
            0.06,
            0.93 - row_idx * (0.46 if len(case_ids) == 2 else 0.88 / max(len(case_ids), 1)),
            f"q{idx} | {case_label} | {compact_output(question, 78)}",
            fontsize=6.8,
            fontweight="bold",
            color=COLORS["text"],
        )
        panel_metadata = []
        for seed_idx, seed_label in enumerate(("seed0", "seed1")):
            scatter = axes[row_idx, seed_idx]
            for view_idx, (point, correct) in enumerate(zip(scores[seed_idx], outcomes[seed_idx])):
                scatter.scatter(
                    point[0],
                    point[1],
                    s=22,
                    marker="o" if correct else "X",
                    color=COLORS["correct"] if correct else COLORS["wrong"],
                    edgecolor="white",
                    linewidth=0.45,
                )
                scatter.text(point[0], point[1], f" v{view_idx}", fontsize=4.8, va="bottom")
            metric = metrics[seed_idx]
            scatter.set_title(
                f"{seed_label} | C={metric['n_correct']}/8\n"
                f"margin={metric['correct_wrong_margin']:+.3f} | gap={metric['wrong_dispersion_gap']:+.3f} | "
                f"AUCx={metric['wrong_rejection_auc_excess_over_exact_null']:+.3f}",
                fontsize=5.9,
                fontweight="bold",
            )
            scatter.set_xlabel(f"local PC1 ({100 * explained[0]:.1f}%)")
            scatter.set_ylabel(f"local PC2 ({100 * explained[1]:.1f}%)")
            style_axis(scatter)

            order = np.concatenate([np.flatnonzero(outcomes[seed_idx]), np.flatnonzero(~outcomes[seed_idx])])
            similarity = signatures[seed_idx][order] @ signatures[seed_idx][order].T
            heat = axes[row_idx, seed_idx + 2]
            last_image = heat.imshow(similarity, vmin=-1, vmax=1, cmap=SIMILARITY_CMAP)
            labels = [
                f"C:v{view_idx}" if outcomes[seed_idx][view_idx] else f"W:v{view_idx}"
                for view_idx in order
            ]
            heat.set_xticks(range(8), labels, rotation=45, ha="right", fontsize=4.6)
            heat.set_yticks(range(8), labels, fontsize=4.6)
            heat.set_title(f"{seed_label}: residualized similarity", fontsize=6.1)
            panel_metadata.append({"seed": seed_label, **metric})
        metadata.append({
            "idx": idx,
            "case_label": case_label,
            "local_pca_explained_variance_ratio": explained.tolist(),
            "seeds": panel_metadata,
        })

    if last_image is not None:
        colorbar_ax = fig.add_axes([0.935, 0.30, 0.010, 0.36])
        fig.colorbar(last_image, cax=colorbar_ax, label="cosine similarity")
    fig.text(
        0.99,
        0.01,
        f"Leave-one-question-out view subtraction; exact label null; {ideal_count}/{eligible_count} pass on both seeds.",
        ha="right",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.06, right=0.91, top=0.86, bottom=0.10, wspace=0.43, hspace=0.76)
    save_figure(fig, output_path)
    plt.close(fig)
    return {
        "case_ids": case_ids,
        "eligible_count": eligible_count,
        "ideal_count": ideal_count,
        "definition": diagnostics["definition"],
        "null": diagnostics["null"],
        "selection_rule": selection_rule,
        "panels": metadata,
    }


def plot_causal_microcases(
    output_path: Path,
    case_ids: list[int],
    causal_rows: list[dict],
    normal_correct_count: int,
    harmed_at_least_three_count: int,
    selection_rule: str,
) -> dict:
    interventions = ["normal", "reverse", "shuffle", "mean_repeat", "random_direction"]
    by_idx = defaultdict(dict)
    for row in causal_rows:
        by_idx[int(row["idx"])][row["intervention"]] = row
    values = np.zeros((len(case_ids), len(interventions)), dtype=float)
    predictions = []
    questions = []
    for row_idx, idx in enumerate(case_ids):
        rows = by_idx[idx]
        first = next(iter(rows.values()))
        questions.append(first["question"])
        row_predictions = [first["normal_pred"]]
        values[row_idx, 0] = float(first["normal_acc"])
        for col_idx, intervention in enumerate(interventions[1:], start=1):
            values[row_idx, col_idx] = float(rows[intervention]["intervention_acc"])
            row_predictions.append(rows[intervention]["intervention_pred"])
        predictions.append(row_predictions)

    fig, (ax, prevalence_ax) = plt.subplots(1, 2, figsize=(7.2, 2.25), gridspec_kw={"width_ratios": [1.42, 0.58]})
    rgb = np.empty((len(case_ids), len(interventions), 3), dtype=float)
    for row in range(len(case_ids)):
        for col in range(len(interventions)):
            rgb[row, col] = matplotlib.colors.to_rgb(COLORS["green"] if values[row, col] > 0.5 else COLORS["pink"])
    ax.imshow(rgb, aspect="auto")
    for row_idx, idx in enumerate(case_ids):
        for col_idx, prediction in enumerate(predictions[row_idx]):
            status = "correct" if values[row_idx, col_idx] > 0.5 else "wrong"
            ax.text(col_idx, row_idx, f"{prediction}\n{status}", ha="center", va="center", fontsize=6.2, fontweight="bold")
    ax.set_xticks(range(len(interventions)), ["Normal", "Reverse", "Shuffle", "Mean-repeat", "Random-dir"], rotation=18, ha="right")
    ax.set_yticks(range(len(case_ids)), [f"q{idx}" for idx in case_ids])
    ax.set_title("Path interventions alter selected decisions", loc="left", fontweight="bold", pad=4)
    add_panel_label(ax, "a", x=-0.12, y=1.04)
    for spine in ax.spines.values():
        spine.set_visible(False)

    rates = np.asarray([
        100 * harmed_at_least_three_count / normal_correct_count,
        100 * len(case_ids) / normal_correct_count,
    ])
    prevalence_bars = prevalence_ax.barh(
        [0, 1],
        rates,
        color=[COLORS["green"], COLORS["pink"]],
        edgecolor="#4A4A4A",
        linewidth=0.6,
        height=0.52,
    )
    prevalence_ax.set_yticks([0, 1], ["At least 3", "All 4"])
    prevalence_ax.invert_yaxis()
    prevalence_ax.set_xlim(0, max(rates) * 1.42)
    prevalence_ax.set_xlabel("Normal-correct questions affected (%)")
    prevalence_ax.set_title("Population prevalence", loc="left", fontweight="bold", pad=4)
    add_panel_label(prevalence_ax, "b", x=-0.16, y=1.04)
    counts = [harmed_at_least_three_count, len(case_ids)]
    for bar, count, rate in zip(prevalence_bars, counts, rates):
        prevalence_ax.text(
            bar.get_width() + max(rates) * 0.035,
            bar.get_y() + bar.get_height() / 2,
            f"{count}/{normal_correct_count} ({rate:.2f}%)",
            va="center",
            fontsize=6.0,
        )
    prevalence_ax.text(
        0.98,
        0.50,
        "Average frozen intervention effect: nonsignificant",
        transform=prevalence_ax.transAxes,
        fontsize=5.8,
        color=COLORS["muted"],
        ha="right",
    )
    style_axis(prevalence_ax)
    fig.text(
        0.99,
        0.01,
        "Selected existence cases only; complete candidate questions and selection rule are exported in CSV/JSON.",
        ha="right",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.25, wspace=0.34)
    save_figure(fig, output_path)
    plt.close(fig)
    return {
        "case_ids": case_ids,
        "normal_correct_count": normal_correct_count,
        "harmed_by_all_four_count": len(case_ids),
        "harmed_by_at_least_three_count": harmed_at_least_three_count,
        "selection_rule": selection_rule,
    }


def write_story_markdown(output_dir: Path, manifest: dict) -> None:
    selection = manifest["selection"]
    counts = selection["counts"]
    sets = selection["sets"]
    rescue_ids = selection["rescue_cases"]
    local_id = selection["local_geometry_case"]
    local_rescue_id = selection["local_geometry_rescue_case"]
    structure_id = selection["structure_case"]
    causal_ids = selection["causal_cases"]
    text = f"""# TRACE Epoch7 Story Evidence

## Purpose

This bundle presents reproducible positive cases from the frozen epoch7 checkpoint. It complements rather than replaces the full-population audits. Every case has a declared selection rule and a denominator.

Every primary figure is available as editable SVG, PDF, high-resolution PNG, and 600-dpi TIFF with the same basename. The requested Times New Roman font is first in the SVG font stack; this machine currently renders the metrically compatible Liberation Serif fallback because the proprietary font file is not installed.

## Stage2 Accuracy-Efficiency Gain

Against the architecture-identical Stage1 checkpoint, epoch7 improves accuracy and reduces #L on all four datasets. Net paired answer improvements are `+58`, `+44`, `+51`, and `+1` on GSM8K, GSMHard, SVAMP, and MultiArith.

![Stage2 positive dashboard](stage2_positive_dashboard.png)

## Population Stage1 Trajectory

Across the same 200 questions, BRIDGE assigns all eight latent slots to almost one CoT position, whereas Stage1 and epoch7 advance from early to late CoT progress. The zero-inversion fraction rises from `{manifest['population_progress']['BRIDGE']['zero_inversion_fraction'] * 100:.1f}%` to `{manifest['population_progress']['Stage1']['zero_inversion_fraction'] * 100:.1f}%`.

![Population trajectory profile](population_progress_profile.png)

## Stage1 Trace Case

Question `q{structure_id}` is selected from the first-200 shorter Stage1-to-epoch7 main-answer rescues by the largest BRIDGE-to-TRACE assignment-progress improvement. BRIDGE assigns every latent slot to nearly the same CoT region; Stage1 and epoch7 progress monotonically from early to late CoT steps.

![Stage1 progress case](stage1_progress_case.png)

## Stage2 Rescue Cases

The three cases `q{rescue_ids[0]}`, `q{rescue_ids[1]}`, and `q{rescue_ids[2]}` represent an all-wrong recovery with replicated local geometry, a second replicated local-geometry rescue with the largest remaining length saving, and a two-seed 8/8-consensus rescue. The first-200 set contains `{len(sets['first200_shorter_main_rescues'])}` shorter main-answer rescues.

![Stage2 rescue cases](stage2_rescue_cases.png)

The same three examples are also rendered under one global PCA fitted on all 200 shared BRIDGE/Stage1/epoch7 records in `global_pca_rescue_cases/`.

## Local Geometry Capability

After subtracting each fixed view template using the other 199 questions, `q{local_id}` maximizes the worse-seed exact-null rejection-AUC excess. `q{local_rescue_id}` satisfies the same two-seed residualized signs and is also a shorter Stage1-to-epoch7 main-answer rescue. In total, `{len(sets['view_residualized_two_seed_geometry_ideal'])}` of `{len(sets['view_residualized_two_seed_geometry_eligible'])}` eligible questions have positive correct/wrong margin, wrong-dispersion gap, and exact-null AUC excess on both seeds. This is a local existence and reproducibility result; the global fixed-view outcome-null remains negative.

![Local geometry case](local_geometry_case.png)

## Causal Micro-cases

Exactly `{len(causal_ids)}` of `{counts['causal_normal_correct']}` normal-correct GSM8K questions are broken by reverse, shuffle, mean-repeat, and random-direction interventions. `{len(sets['harmed_by_at_least_three_interventions'])}` are broken by at least three interventions.

![Causal micro-cases](causal_microcases.png)

## Audit Tables

The figures are backed by `rollout_transition_rows.csv`, `stage1_structure_candidates.csv`, `residualized_geometry_candidates.csv`, and `causal_normal_correct_candidates.csv`. These tables expose the full candidate pools, selection flags, and denominators used by the gallery. The complete all-wrong/mixed/all-correct transition matrix remains available as `rollout_transition_matrix.png` for supplementary audit rather than as a main positive figure.

## Safe Story

The supported visual story is: Stage1 converts BRIDGE's flat compression into ordered latent progress; Stage2 improves answers, shortens reasoning, and shifts rollout groups toward correctness; selected questions exhibit repeatable local outcome geometry and causal path dependence. The figures must not be described as evidence that outcome-aware geometry or causal dependence holds for every question.
"""
    (output_dir / "STORY_EVIDENCE.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--trace-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.evidence_root.resolve()
    output_dir = (args.output_dir or root / "story_evidence").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    record_paths = {
        "BRIDGE": root / "matched_bridge_rollouts" / "logs" / "tb" / "run" / "trace_bridge_visual_test.pt",
        "Stage1": root / "matched_stage1_rollouts" / "logs" / "tb" / "run" / "trace_bridge_visual_test.pt",
        "TRACE epoch7": args.trace_record.resolve(),
        "TRACE seed1": root / "geometry_replication_seed1" / "logs" / "tb" / "run" / "trace_bridge_visual_test.pt",
    }
    records = {label: load_records(path) for label, path in record_paths.items()}

    geometry_paths = {
        "Stage1": root / "matched_stage1_rollouts" / "geometry_stage2_metric" / "trace_bridge_geometry_summary.json",
        "TRACE epoch7": root / "geometry_epoch7_stage2_metric" / "trace_bridge_geometry_summary.json",
        "TRACE seed1": root / "geometry_replication_seed1" / "geometry_stage2_metric_null1024" / "trace_bridge_geometry_summary.json",
    }
    geometry = {label: load_geometry(path) for label, path in geometry_paths.items()}
    paired_evidence_path = root / "paired_stage1_vs_epoch7" / "paired_evidence.json"
    paired_rows = read_csv(root / "paired_stage1_vs_epoch7" / "paired_question_rows.csv")
    paired = {int(row["idx"]): row for row in paired_rows if row["dataset"] == "GSM8K"}
    causal_rows = read_csv(root / "causal_interventions" / "summary" / "causal_question_rows.csv")
    view_residualized = build_view_residualized_diagnostics(
        {
            "seed0": records["TRACE epoch7"],
            "seed1": records["TRACE seed1"],
        }
    )

    selection = select_cases(
        paired_rows=paired_rows,
        bridge_records=records["BRIDGE"],
        stage1_records=records["Stage1"],
        trace_records=records["TRACE epoch7"],
        seed1_records=records["TRACE seed1"],
        stage1_geometry=geometry["Stage1"],
        trace_geometry=geometry["TRACE epoch7"],
        seed1_geometry=geometry["TRACE seed1"],
        view_residualized=view_residualized,
        causal_rows=causal_rows,
    )

    population_progress = plot_population_progress_profile(
        output_dir / "population_progress_profile.png",
        records,
    )
    stage2_positive = plot_stage2_positive_dashboard(
        output_dir / "stage2_positive_dashboard.png",
        paired_evidence_path,
    )
    transition = plot_transition_matrix(
        output_dir / "rollout_transition_matrix.png",
        geometry["Stage1"],
        geometry["TRACE epoch7"],
    )
    structure = plot_structure_case(
        output_dir / "stage1_progress_case.png",
        selection["structure_case"],
        records,
        selection["rules"]["structure_case"],
    )
    rescue_rules = [
        selection["rules"]["rescue_case_1"],
        selection["rules"]["rescue_case_2"],
        selection["rules"]["rescue_case_3"],
    ]
    rescues = plot_rescue_cases(
        output_dir / "stage2_rescue_cases.png",
        selection["rescue_cases"],
        records,
        paired,
        rescue_rules,
    )
    local_case_ids = [selection["local_geometry_case"]]
    if selection["local_geometry_rescue_case"] is not None and selection["local_geometry_rescue_case"] not in local_case_ids:
        local_case_ids.append(selection["local_geometry_rescue_case"])
    local = plot_view_residualized_geometry_cases(
        output_dir / "local_geometry_case.png",
        local_case_ids,
        records,
        view_residualized,
        selection["rules"]["local_geometry_case"],
        len(selection["sets"]["view_residualized_two_seed_geometry_ideal"]),
        len(selection["sets"]["view_residualized_two_seed_geometry_eligible"]),
    )
    causal = plot_causal_microcases(
        output_dir / "causal_microcases.png",
        selection["causal_cases"],
        causal_rows,
        selection["counts"]["causal_normal_correct"],
        len(selection["sets"]["harmed_by_at_least_three_interventions"]),
        selection["rules"]["causal_cases"],
    )
    audit_tables = export_audit_tables(
        output_dir=output_dir,
        selection=selection,
        paired=paired,
        records=records,
        stage1_geometry=geometry["Stage1"],
        trace_geometry=geometry["TRACE epoch7"],
        view_residualized=view_residualized,
        causal_rows=causal_rows,
    )

    manifest = {
        "frozen_checkpoint_sha256": "9c166dd2069753f891446851b9b0eb829136e4ad2803cf1daabce57df5f74f3f",
        "record_paths": {label: str(path) for label, path in record_paths.items()},
        "geometry_paths": {label: str(path) for label, path in geometry_paths.items()},
        "selection": selection,
        "population_progress": population_progress,
        "stage2_positive": stage2_positive,
        "transition": transition,
        "structure": structure,
        "rescues": rescues,
        "local_geometry": local,
        "causal": causal,
        "audit_tables": audit_tables,
        "claim_boundary": (
            "Representative success cases and subset coverage support capability and mechanism illustrations; "
            "they do not replace full-population outcome-null or causal tests."
        ),
    }
    (output_dir / "story_evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_story_markdown(output_dir, manifest)
    print(json.dumps({
        "output_dir": str(output_dir),
        "structure_case": selection["structure_case"],
        "rescue_cases": selection["rescue_cases"],
        "local_geometry_case": selection["local_geometry_case"],
        "causal_cases": selection["causal_cases"],
    }, indent=2))


if __name__ == "__main__":
    main()
