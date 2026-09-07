#!/usr/bin/env python3
"""Paired Stage-1 versus Stage-2 role-aware TRACE evidence.

Figure contract
---------------
Core conclusion:
    Stage-2 should improve PLAN/SOLVE/CHECK teacher-semantic alignment while
    preserving role-conditioned stochastic exploration and a strictly
    deterministic, answer-visible COMMIT transition.
Evidence logic:
    Every statistic is paired by the same 200 GSM8K test questions and eight
    IID role-conditioned paths per question. The primary evidence is the
    recorded role reward, with SOLVE aggregated only over observed teacher
    chunks. Policy standard deviation and diagonal-Gaussian entropy describe
    only the seven stochastic roles. Outcome/corridor geometry is retained as
    a diagnostic under one shared train-fit PCA contract.
Review risks:
    Cosine role reward measures teacher-semantic alignment, not reasoning
    correctness. COMMIT determinism must be established from action=mean,
    zero COMMIT role reward, and answer readout restricted to COMMIT. Geometry
    is measured in D_TRACE, not display PCA coordinates.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from trace_policy_geometry_summary import (
    choose_camera,
    common_axis_limits,
    plot_trajectory,
    project_residuals,
)


PINK = "#E5A3BF"
PINK_DARK = "#B95C88"
BLUE = "#6687B8"
GREEN = "#69AD7C"
ORANGE = "#E5A11A"
INK = "#30343B"
GRID = "#DDE2E8"

ROLE_SCHEMA = (
    "PLAN,SOLVE1,SOLVE2,SOLVE3,SOLVE4,SOLVE5,CHECK,COMMIT"
)
ROLE_LABELS = (
    "PLAN",
    "SOLVE1",
    "SOLVE2",
    "SOLVE3",
    "SOLVE4",
    "SOLVE5",
    "CHECK",
    "COMMIT",
)
STOCHASTIC_ROLE_LABELS = ROLE_LABELS[:7]
COMMIT_TOLERANCE = 1e-6
GAUSSIAN_ENTROPY_CONSTANT = 0.5 * np.log(2.0 * np.pi * np.e)


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Liberation Serif",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def save_figure(fig, output_base: Path):
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def load_records(path: Path):
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or len(records) != 200:
        raise ValueError(f"{path} must contain exactly 200 records")
    required = (
        "idx",
        "map_correct",
        "map_output_length",
        "answer_latent_attention_access",
        "answer_latent_attention_role",
        "rollout_correctness",
        "rollout_schema",
        "rollout_actions",
        "rollout_action_means",
        "rollout_action_log_stds",
        "rollout_implicit_residuals",
        "map_implicit_residuals",
        "role_schema",
        "role_teacher_plan",
        "role_teacher_solve",
        "role_teacher_solve_mask",
        "role_teacher_summary",
        "map_role_rewards",
        "rollout_role_rewards",
        "visualization_contract",
    )
    seen_indices = set()
    for row, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"record {row} is missing {missing}")
        question_index = int(record["idx"])
        if question_index in seen_indices:
            raise ValueError(f"record {row} repeats idx={question_index}")
        seen_indices.add(question_index)
        if record["rollout_schema"] != (
            "iid_role_conditioned_gaussian_with_deterministic_commit"
        ):
            raise ValueError(
                f"record {row} is not a role-conditioned IID rollout"
            )
        if record["role_schema"] != ROLE_SCHEMA:
            raise ValueError(f"record {row} has an invalid role schema")
        correctness = torch.as_tensor(record["rollout_correctness"])
        if correctness.shape != (8,):
            raise ValueError(
                f"record {row} does not contain exactly eight paths"
            )
        map_rewards = torch.as_tensor(record["map_role_rewards"])
        rollout_rewards = torch.as_tensor(
            record["rollout_role_rewards"]
        )
        if map_rewards.shape != (8,):
            raise ValueError(
                f"record {row} map_role_rewards must have shape [8]"
            )
        if rollout_rewards.shape != (8, 8):
            raise ValueError(
                f"record {row} rollout_role_rewards must have shape [8,8]"
            )
        teacher_plan = torch.as_tensor(record["role_teacher_plan"])
        teacher_solve = torch.as_tensor(record["role_teacher_solve"])
        teacher_mask = torch.as_tensor(record["role_teacher_solve_mask"])
        teacher_summary = torch.as_tensor(record["role_teacher_summary"])
        if teacher_plan.ndim != 1 or teacher_plan.numel() == 0:
            raise ValueError(f"record {row} has an invalid PLAN teacher")
        hidden_size = int(teacher_plan.shape[0])
        if teacher_solve.shape != (5, hidden_size):
            raise ValueError(f"record {row} has invalid SOLVE teachers")
        if teacher_mask.shape != (5,) or not bool(teacher_mask.bool().any()):
            raise ValueError(f"record {row} has an invalid SOLVE mask")
        if teacher_summary.shape != (hidden_size,):
            raise ValueError(f"record {row} has an invalid CHECK summary")
        actions = torch.as_tensor(record["rollout_actions"])
        means = torch.as_tensor(record["rollout_action_means"])
        log_stds = torch.as_tensor(record["rollout_action_log_stds"])
        if (
            actions.ndim != 3
            or actions.shape[:2] != (8, 8)
            or actions.shape[-1] <= 0
            or means.shape != actions.shape
            or log_stds.shape != actions.shape
        ):
            raise ValueError(
                f"record {row} rollout policy tensors must have shape [8,8,A]"
            )
        rollout_residuals = torch.as_tensor(
            record["rollout_implicit_residuals"]
        )
        map_residuals = torch.as_tensor(record["map_implicit_residuals"])
        if (
            rollout_residuals.ndim != 3
            or rollout_residuals.shape[:2] != (8, 8)
            or rollout_residuals.shape[-1] != hidden_size
            or map_residuals.shape != rollout_residuals.shape[1:]
        ):
            raise ValueError(
                f"record {row} residuals must have [8,8,H] and [8,H] shapes"
            )
        tensors = (
            map_rewards,
            rollout_rewards,
            teacher_plan,
            teacher_solve,
            teacher_summary,
            actions,
            means,
            log_stds,
            rollout_residuals,
            map_residuals,
        )
        if any(not bool(torch.isfinite(tensor.float()).all()) for tensor in tensors):
            raise ValueError(f"record {row} contains non-finite role data")
        commit_action_error = (
            actions[:, 7].float() - means[:, 7].float()
        ).abs().max()
        commit_reward_error = torch.maximum(
            rollout_rewards[:, 7].float().abs().max(),
            map_rewards[7].float().abs(),
        )
        if float(commit_action_error) > COMMIT_TOLERANCE:
            raise ValueError(
                f"record {row} samples its deterministic COMMIT action"
            )
        if float(commit_reward_error) > COMMIT_TOLERANCE:
            raise ValueError(f"record {row} has a non-zero COMMIT reward")
        if int(record["answer_latent_attention_access"]) != 1:
            raise ValueError(
                f"record {row} answer readout is not COMMIT-only"
            )
        if str(record["answer_latent_attention_role"]) != "COMMIT":
            raise ValueError(
                f"record {row} answer readout role is not COMMIT"
            )
        contract = record["visualization_contract"]
        if contract.get("manual_offsets") is not False:
            raise ValueError(f"record {row} permits manual offsets")
        if contract.get("per_path_rescaling") is not False:
            raise ValueError(f"record {row} permits per-path rescaling")
    return records


def load_geometry_rows(path: Path) -> Dict[int, dict]:
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(float(row.pop("idx")))
            rows[index] = {
                key: float(value) for key, value in row.items()
            }
    if len(rows) != 200:
        raise ValueError(f"{path} must contain exactly 200 geometry rows")
    return rows


def paired_ci(
    stage1: Sequence[float],
    final: Sequence[float],
    *,
    rng: np.random.Generator,
    bootstrap: int,
) -> Dict[str, object]:
    left = np.asarray(stage1, dtype=np.float64)
    right = np.asarray(final, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired vectors must have identical shapes")
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size == 0:
        return {
            "stage1": float("nan"),
            "final": float("nan"),
            "delta": float("nan"),
            "delta_ci95": [float("nan"), float("nan")],
            "n_questions": 0,
        }
    indices = rng.integers(
        0,
        left.size,
        size=(int(bootstrap), left.size),
    )
    left_means = left[indices].mean(axis=1)
    right_means = right[indices].mean(axis=1)
    deltas = (right[indices] - left[indices]).mean(axis=1)
    return {
        "stage1": float(left.mean()),
        "stage1_ci95": [
            float(np.quantile(left_means, 0.025)),
            float(np.quantile(left_means, 0.975)),
        ],
        "final": float(right.mean()),
        "final_ci95": [
            float(np.quantile(right_means, 0.025)),
            float(np.quantile(right_means, 0.975)),
        ],
        "delta": float((right - left).mean()),
        "delta_ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "n_questions": int(left.size),
    }


def stage_arrays(records):
    rollout = np.asarray(
        [record["rollout_correctness"] for record in records],
        dtype=np.float64,
    )
    correct_count = rollout.sum(axis=1)
    return {
        "map_accuracy": np.asarray(
            [record["map_correct"] for record in records],
            dtype=np.float64,
        ),
        "map_output_length": np.asarray(
            [record["map_output_length"] for record in records],
            dtype=np.float64,
        ),
        "map_total_length": np.asarray(
            [
                record["map_output_length"]
                + record["answer_latent_attention_access"]
                for record in records
            ],
            dtype=np.float64,
        ),
        "rollout_accuracy": rollout.mean(axis=1),
        "any_correct": (correct_count >= 1).astype(np.float64),
        "majority_correct": (correct_count >= 5).astype(np.float64),
        "all_correct": (correct_count == 8).astype(np.float64),
        "correct_count": correct_count,
    }


def role_arrays(records):
    """Build one paired question-level observation for every role metric."""
    map_rewards = np.stack(
        [
            torch.as_tensor(record["map_role_rewards"])
            .float()
            .numpy()
            for record in records
        ]
    ).astype(np.float64)
    rollout_rewards = np.stack(
        [
            torch.as_tensor(record["rollout_role_rewards"])
            .float()
            .numpy()
            for record in records
        ]
    ).astype(np.float64)
    solve_mask = np.stack(
        [
            torch.as_tensor(record["role_teacher_solve_mask"])
            .bool()
            .numpy()
            for record in records
        ]
    )
    solve_denominator = solve_mask.sum(axis=1).astype(np.float64)
    if np.any(solve_denominator <= 0):
        raise ValueError("every question must supervise at least one SOLVE slot")
    map_solve = (
        map_rewards[:, 1:6] * solve_mask
    ).sum(axis=1) / solve_denominator
    rollout_solve = (
        rollout_rewards[:, :, 1:6] * solve_mask[:, None, :]
    ).sum(axis=(1, 2)) / (8.0 * solve_denominator)

    log_stds = np.stack(
        [
            torch.as_tensor(record["rollout_action_log_stds"])
            .float()
            .numpy()
            for record in records
        ]
    ).astype(np.float64)
    stochastic_log_stds = log_stds[:, :, :7, :]
    position_std = np.exp(stochastic_log_stds).mean(axis=(1, 3))
    position_entropy = (
        stochastic_log_stds + GAUSSIAN_ENTROPY_CONSTANT
    ).mean(axis=(1, 3))
    if not (
        np.isfinite(position_std).all()
        and np.isfinite(position_entropy).all()
    ):
        raise ValueError("role-conditioned policy statistics are non-finite")
    # All five SOLVE transitions are executed and stochastic even when a short
    # teacher CoT leaves some semantic targets masked. Exploration therefore
    # covers every actual SOLVE slot; only semantic similarity is mask-aware.
    solve_std = position_std[:, 1:6].mean(axis=1)
    solve_entropy = position_entropy[:, 1:6].mean(axis=1)

    commit_action_error = np.asarray(
        [
            float(
                (
                    torch.as_tensor(record["rollout_actions"])[
                        :, 7
                    ].float()
                    - torch.as_tensor(record["rollout_action_means"])[
                        :, 7
                    ].float()
                )
                .abs()
                .max()
            )
            for record in records
        ],
        dtype=np.float64,
    )
    commit_reward_error = np.asarray(
        [
            max(
                float(
                    torch.as_tensor(record["map_role_rewards"])[7]
                    .float()
                    .abs()
                ),
                float(
                    torch.as_tensor(record["rollout_role_rewards"])[
                        :, 7
                    ]
                    .float()
                    .abs()
                    .max()
                ),
            )
            for record in records
        ],
        dtype=np.float64,
    )
    commit_readout_only = np.asarray(
        [
            int(record["answer_latent_attention_access"]) == 1
            and str(record["answer_latent_attention_role"]) == "COMMIT"
            for record in records
        ],
        dtype=np.float64,
    )
    commit_deterministic = (
        (commit_action_error <= COMMIT_TOLERANCE)
        & (commit_reward_error <= COMMIT_TOLERANCE)
        & (commit_readout_only > 0.5)
    ).astype(np.float64)
    return {
        "map_plan_similarity": map_rewards[:, 0],
        "map_solve_similarity": map_solve,
        "map_check_similarity": map_rewards[:, 6],
        "rollout_plan_similarity": rollout_rewards[:, :, 0].mean(axis=1),
        "rollout_solve_similarity": rollout_solve,
        "rollout_check_similarity": rollout_rewards[:, :, 6].mean(axis=1),
        "plan_action_std": position_std[:, 0],
        "solve_action_std": solve_std,
        "check_action_std": position_std[:, 6],
        "plan_gaussian_entropy": position_entropy[:, 0],
        "solve_gaussian_entropy": solve_entropy,
        "check_gaussian_entropy": position_entropy[:, 6],
        "commit_action_mean_abs_error": commit_action_error,
        "commit_role_reward_abs_max": commit_reward_error,
        "commit_readout_only": commit_readout_only,
        "commit_deterministic": commit_deterministic,
        "map_role_rewards": map_rewards,
        "rollout_role_rewards": rollout_rewards,
        "solve_mask": solve_mask.astype(np.uint8),
        "stochastic_action_std_by_position": position_std,
        "stochastic_gaussian_entropy_by_position": position_entropy,
    }


def commit_audit_summary(role: Dict[str, np.ndarray]) -> Dict[str, object]:
    action_error = role["commit_action_mean_abs_error"]
    reward_error = role["commit_role_reward_abs_max"]
    readout_only = role["commit_readout_only"]
    deterministic = role["commit_deterministic"]
    summary = {
        "max_action_mean_abs_error": float(action_error.max()),
        "max_commit_role_reward": float(reward_error.max()),
        "action_mismatch_rate": float(
            np.mean(action_error > COMMIT_TOLERANCE)
        ),
        "nonzero_reward_rate": float(
            np.mean(reward_error > COMMIT_TOLERANCE)
        ),
        "wrong_readout_rate": float(np.mean(readout_only < 0.5)),
        "deterministic_pass_rate": float(deterministic.mean()),
        "n_questions": int(deterministic.size),
    }
    summary["pass"] = bool(
        summary["max_action_mean_abs_error"] <= COMMIT_TOLERANCE
        and summary["max_commit_role_reward"] <= COMMIT_TOLERANCE
        and summary["wrong_readout_rate"] == 0.0
        and summary["deterministic_pass_rate"] == 1.0
    )
    return summary


def _stage_values_and_errors(metrics, keys, stage):
    values = np.asarray([metrics[key][stage] for key in keys])
    intervals = np.asarray(
        [metrics[key][f"{stage}_ci95"] for key in keys]
    )
    errors = np.stack(
        [values - intervals[:, 0], intervals[:, 1] - values],
        axis=0,
    )
    return values, errors


def plot_role_comparison(
    semantics: Dict[str, dict],
    exploration: Dict[str, dict],
    commit_audit: Dict[str, dict],
    output_base: Path,
):
    """Primary role-evidence figure for the paired stage comparison."""
    role_names = ("PLAN", "SOLVE", "CHECK")
    x = np.arange(3)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))

    for ax, prefix, panel, title in (
        (axes[0, 0], "map", "a", "MAP-path teacher alignment"),
        (
            axes[0, 1],
            "rollout",
            "b",
            "Eight-path teacher alignment",
        ),
    ):
        keys = tuple(
            f"{prefix}_{role.lower()}_similarity"
            for role in role_names
        )
        for stage, offset, color, marker, label in (
            ("stage1", -0.07, GREEN, "o", "Stage 1"),
            ("final", 0.07, PINK_DARK, "s", "Final"),
        ):
            values, errors = _stage_values_and_errors(
                semantics,
                keys,
                stage,
            )
            ax.errorbar(
                x + offset,
                values,
                yerr=errors,
                color=color,
                marker=marker,
                markersize=4.2,
                linewidth=1.25,
                capsize=2.2,
                label=label,
            )
        ax.axhline(0.0, color=GRID, linewidth=0.7)
        ax.set_xticks(x, role_names)
        ax.set_ylabel("Teacher cosine similarity")
        ax.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    axes[0, 0].legend(loc="best")

    ax = axes[1, 0]
    entropy_ax = ax.twinx()
    std_keys = tuple(
        f"{role.lower()}_action_std" for role in role_names
    )
    entropy_keys = tuple(
        f"{role.lower()}_gaussian_entropy" for role in role_names
    )
    legend_handles = []
    for stage, offset, color, marker, label in (
        ("stage1", -0.06, GREEN, "o", "Stage 1"),
        ("final", 0.06, PINK_DARK, "s", "Final"),
    ):
        std_values, std_errors = _stage_values_and_errors(
            exploration,
            std_keys,
            stage,
        )
        entropy_values, entropy_errors = _stage_values_and_errors(
            exploration,
            entropy_keys,
            stage,
        )
        std_handle = ax.errorbar(
            x + offset,
            std_values,
            yerr=std_errors,
            color=color,
            marker=marker,
            markersize=4.1,
            linewidth=1.25,
            capsize=2.0,
            label=f"{label} std",
        )
        entropy_handle = entropy_ax.errorbar(
            x + offset,
            entropy_values,
            yerr=entropy_errors,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markersize=3.7,
            linewidth=1.0,
            linestyle="--",
            capsize=1.8,
            alpha=0.78,
            label=f"{label} entropy",
        )
        legend_handles.extend([std_handle, entropy_handle])
    ax.set_xticks(x, role_names)
    ax.set_ylabel("Mean policy std")
    entropy_ax.set_ylabel("Gaussian entropy / action dim")
    ax.set_title(
        "c  Role-conditioned exploration (stochastic slots only)",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="best",
        ncol=2,
        fontsize=5.7,
    )

    ax = axes[1, 1]
    audit_labels = (
        "action != mean",
        "non-zero reward",
        "wrong readout",
    )
    for stage, color, marker, label in (
        ("stage1", GREEN, "o", "Stage 1"),
        ("final", PINK_DARK, "s", "Final"),
    ):
        values = 100.0 * np.asarray(
            [
                commit_audit[stage]["action_mismatch_rate"],
                commit_audit[stage]["nonzero_reward_rate"],
                commit_audit[stage]["wrong_readout_rate"],
            ]
        )
        ax.plot(
            x,
            values,
            color=color,
            marker=marker,
            markersize=4.2,
            linewidth=1.25,
            label=label,
        )
    upper = max(
        1.0,
        1.15
        * 100.0
        * max(
            commit_audit[stage][key]
            for stage in ("stage1", "final")
            for key in (
                "action_mismatch_rate",
                "nonzero_reward_rate",
                "wrong_readout_rate",
            )
        ),
    )
    ax.set_ylim(-0.04 * upper, upper)
    ax.set_xticks(x, audit_labels)
    ax.tick_params(axis="x", rotation=14)
    ax.set_ylabel("Questions violating contract (%)")
    ax.set_title(
        "d  COMMIT deterministic/readout audit",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.legend(loc="best")
    for row, stage in enumerate(("stage1", "final")):
        summary = commit_audit[stage]
        ax.text(
            0.02,
            0.96 - 0.11 * row,
            (
                f"{stage.replace('stage1', 'Stage 1').replace('final', 'Final')}: "
                f"max |a-μ|={summary['max_action_mean_abs_error']:.2e}; "
                f"max |r|={summary['max_commit_role_reward']:.2e}; "
                f"{'PASS' if summary['pass'] else 'FAIL'}"
            ),
            transform=ax.transAxes,
            color=GREEN if stage == "stage1" else PINK_DARK,
            fontsize=5.7,
            va="top",
        )

    fig.suptitle(
        "Paired role-aware TRACE evidence (200 questions × 8 paths)",
        x=0.01,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.8, w_pad=1.2, h_pad=1.0)
    save_figure(fig, output_base)


def plot_reliability(
    stage1: Dict[str, np.ndarray],
    final: Dict[str, np.ndarray],
    output_base: Path,
):
    keys = (
        "map_accuracy",
        "any_correct",
        "majority_correct",
        "all_correct",
    )
    labels = ("MAP", "Any / 8", "Majority / 8", "All / 8")
    left = np.asarray([stage1[key].mean() for key in keys]) * 100.0
    right = np.asarray([final[key].mean() for key in keys]) * 100.0
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    ax.plot(
        x,
        left,
        color=GREEN,
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        label="Stage 1",
    )
    ax.plot(
        x,
        right,
        color=PINK_DARK,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="Final",
    )
    ax.fill_between(x, left, right, color=PINK, alpha=0.16)
    lower = max(0.0, min(left.min(), right.min()) - 7.0)
    upper = min(100.0, max(left.max(), right.max()) + 9.0)
    for index, delta in enumerate(right - left):
        if abs(delta) < 0.05:
            continue
        anchor = max(left[index], right[index])
        if anchor >= upper - 2.0:
            y = anchor - 1.8
            vertical_alignment = "top"
        else:
            y = anchor + 1.6
            vertical_alignment = "bottom"
        ax.text(
            index,
            y,
            f"{delta:+.1f}",
            color=PINK_DARK,
            ha="center",
            va=vertical_alignment,
            fontweight="bold",
        )
    ax.set_ylim(lower, upper)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Questions meeting criterion (%)")
    ax.set_title(
        "Secondary outcome reliability across eight IID paths",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_base)


def plot_accuracy_length(
    stage1: Dict[str, np.ndarray],
    final: Dict[str, np.ndarray],
    output_base: Path,
    *,
    bootstrap: int,
):
    rng = np.random.default_rng(17)
    accuracy_metric = paired_ci(
        stage1["map_accuracy"] * 100.0,
        final["map_accuracy"] * 100.0,
        rng=rng,
        bootstrap=bootstrap,
    )
    length_metric = paired_ci(
        stage1["map_total_length"],
        final["map_total_length"],
        rng=rng,
        bootstrap=bootstrap,
    )
    colors = [GREEN, PINK]
    edges = [GREEN, PINK_DARK]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.15))
    for ax, metric, title, ylabel, suffix in (
        (
            axes[0],
            accuracy_metric,
            "Deterministic accuracy",
            "Accuracy (%)",
            " pp",
        ),
        (
            axes[1],
            length_metric,
            "Total reasoning length #L",
            "Latent + generated tokens",
            "",
        ),
    ):
        values = [metric["stage1"], metric["final"]]
        intervals = [
            metric["stage1_ci95"],
            metric["final_ci95"],
        ]
        errors = np.asarray(
            [
                [
                    values[index] - intervals[index][0]
                    for index in range(2)
                ],
                [
                    intervals[index][1] - values[index]
                    for index in range(2)
                ],
            ]
        )
        ax.bar(
            [0, 1],
            values,
            color=colors,
            edgecolor=edges,
            linewidth=0.8,
            width=0.56,
            zorder=2,
        )
        ax.errorbar(
            [0, 1],
            values,
            yerr=errors,
            color=INK,
            fmt="none",
            capsize=2.5,
            linewidth=0.9,
            zorder=3,
        )
        ax.text(
            0.5,
            max(values),
            f"{metric['delta']:+.2f}{suffix}",
            ha="center",
            va="bottom",
            color=PINK_DARK,
            fontweight="bold",
        )
        ax.set_xticks([0, 1], ["Stage 1", "Final"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.55)
    fig.tight_layout(pad=0.7, w_pad=1.2)
    save_figure(fig, output_base)


def plot_geometry(
    metrics: Dict[str, dict],
    output_base: Path,
):
    keys = (
        "correct_local_radius",
        "wrong_to_correct_distance",
        "outcome_margin",
        "path_diversity",
    )
    labels = (
        "Correct\nradius",
        "Wrong-to-\ncorrect",
        "Outcome\nmargin",
        "All-path\ndiversity",
    )
    stage1 = np.asarray([metrics[key]["stage1"] for key in keys])
    final = np.asarray([metrics[key]["final"] for key in keys])
    x = np.arange(len(keys))
    stage1_errors = np.asarray(
        [
            [
                metrics[key]["stage1"]
                - metrics[key]["stage1_ci95"][0]
                for key in keys
            ],
            [
                metrics[key]["stage1_ci95"][1]
                - metrics[key]["stage1"]
                for key in keys
            ],
        ]
    )
    final_errors = np.asarray(
        [
            [
                metrics[key]["final"]
                - metrics[key]["final_ci95"][0]
                for key in keys
            ],
            [
                metrics[key]["final_ci95"][1]
                - metrics[key]["final"]
                for key in keys
            ],
        ]
    )
    fig, ax = plt.subplots(figsize=(3.55, 2.6))
    ax.axhline(0.0, color=INK, linewidth=0.8, linestyle="--")
    ax.errorbar(
        x - 0.08,
        stage1,
        yerr=stage1_errors,
        color=GREEN,
        marker="o",
        linestyle="none",
        linewidth=1.2,
        markersize=4.5,
        capsize=2.2,
        label="Stage 1",
    )
    ax.errorbar(
        x + 0.08,
        final,
        yerr=final_errors,
        color=PINK_DARK,
        marker="o",
        linestyle="none",
        linewidth=1.2,
        markersize=4.5,
        capsize=2.2,
        label="Final",
    )
    for index, key in enumerate(keys):
        if abs(metrics[key]["delta"]) < 5e-4:
            continue
        ax.text(
            index,
            max(stage1[index], final[index]),
            f"{metrics[key]['delta']:+.3f}",
            ha="center",
            va="bottom",
            color=PINK_DARK,
            fontweight="bold",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Complete-path distance $D_{\mathrm{TRACE}}$")
    ax.set_title(
        "Diagnostic only: paired outcome geometry",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.legend(loc="best")
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_base)


def figure_qa(output_dir: Path, *, expected_figures: int) -> dict:
    from PIL import Image, ImageStat

    figures = []
    for svg in sorted(output_dir.glob("*.svg")):
        pdf = svg.with_suffix(".pdf")
        tiff = svg.with_suffix(".tiff")
        if "<text" not in svg.read_text():
            raise ValueError(f"{svg} does not preserve editable text")
        if not pdf.exists() or not tiff.exists():
            raise ValueError(f"incomplete export bundle for {svg.stem}")
        image = Image.open(tiff).convert("RGB")
        extrema = ImageStat.Stat(image).extrema
        if all(low == high for low, high in extrema):
            raise ValueError(f"{tiff} is visually blank")
        figures.append(
            {
                "name": svg.stem,
                "svg_editable_text": True,
                "pdf_exists": True,
                "tiff_exists": True,
                "tiff_pixels": list(image.size),
                "nonblank": True,
            }
        )
    if len(figures) != int(expected_figures):
        raise ValueError(
            f"expected {expected_figures} paired figures, found {len(figures)}"
        )
    return {"status": "PASS", "figures": figures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-records", type=Path, required=True)
    parser.add_argument("--final-records", type=Path, required=True)
    parser.add_argument("--stage1-geometry", type=Path, required=True)
    parser.add_argument("--final-geometry", type=Path, required=True)
    parser.add_argument("--shared-pca", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if int(args.bootstrap) <= 0:
        raise ValueError("--bootstrap must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    stage1_records = load_records(args.stage1_records)
    final_records = load_records(args.final_records)
    stage1_indices = [int(record["idx"]) for record in stage1_records]
    final_indices = [int(record["idx"]) for record in final_records]
    if stage1_indices != final_indices:
        raise ValueError("Stage-1 and final records are not question-paired")
    pca = torch.load(
        args.shared_pca,
        map_location="cpu",
        weights_only=False,
    )
    if not {"mean", "components", "explained_ratio"} <= set(pca):
        raise ValueError("shared PCA state is incomplete")
    stage1 = stage_arrays(stage1_records)
    final = stage_arrays(final_records)
    stage1_role = role_arrays(stage1_records)
    final_role = role_arrays(final_records)
    rng = np.random.default_rng(20260719)

    semantic_names = (
        "map_plan_similarity",
        "map_solve_similarity",
        "map_check_similarity",
        "rollout_plan_similarity",
        "rollout_solve_similarity",
        "rollout_check_similarity",
    )
    role_semantics = {
        key: paired_ci(
            stage1_role[key],
            final_role[key],
            rng=rng,
            bootstrap=args.bootstrap,
        )
        for key in semantic_names
    }
    exploration_names = (
        "plan_action_std",
        "solve_action_std",
        "check_action_std",
        "plan_gaussian_entropy",
        "solve_gaussian_entropy",
        "check_gaussian_entropy",
    )
    role_exploration = {
        key: paired_ci(
            stage1_role[key],
            final_role[key],
            rng=rng,
            bootstrap=args.bootstrap,
        )
        for key in exploration_names
    }
    commit_metric_names = (
        "commit_action_mean_abs_error",
        "commit_role_reward_abs_max",
        "commit_readout_only",
        "commit_deterministic",
    )
    commit_audit = {
        "tolerance": COMMIT_TOLERANCE,
        "stage1": commit_audit_summary(stage1_role),
        "final": commit_audit_summary(final_role),
        "paired_metrics": {
            key: paired_ci(
                stage1_role[key],
                final_role[key],
                rng=rng,
                bootstrap=args.bootstrap,
            )
            for key in commit_metric_names
        },
        "definition": (
            "PASS requires rollout COMMIT action=conditional mean, MAP and "
            "rollout COMMIT role reward=0, and answer readout access=1 with "
            "role COMMIT for every question."
        ),
    }

    behavior = {
        key: paired_ci(
            stage1[key],
            final[key],
            rng=rng,
            bootstrap=args.bootstrap,
        )
        for key in stage1
    }
    stage1_rows = load_geometry_rows(args.stage1_geometry)
    final_rows = load_geometry_rows(args.final_geometry)
    if list(stage1_rows) != list(final_rows):
        raise ValueError("Stage-1 and final geometry rows are not paired")
    common = [
        index
        for index in stage1_rows
        if stage1_rows[index]["eligible"] > 0.5
        and final_rows[index]["eligible"] > 0.5
    ]
    if not common:
        raise ValueError("no common outcome-geometry eligible questions")
    geometry = {}
    for key in (
        "correct_local_radius",
        "wrong_to_correct_distance",
        "outcome_margin",
    ):
        geometry[key] = paired_ci(
            [stage1_rows[index][key] for index in common],
            [final_rows[index][key] for index in common],
            rng=rng,
            bootstrap=args.bootstrap,
        )
    geometry["path_diversity"] = paired_ci(
        [stage1_rows[index]["path_diversity"] for index in stage1_indices],
        [final_rows[index]["path_diversity"] for index in stage1_indices],
        rng=rng,
        bootstrap=args.bootstrap,
    )

    rescued = int(
        np.sum(
            (stage1["map_accuracy"] == 0)
            & (final["map_accuracy"] == 1)
        )
    )
    regressed = int(
        np.sum(
            (stage1["map_accuracy"] == 1)
            & (final["map_accuracy"] == 0)
        )
    )
    report = {
        "status": "PASS",
        "core_conclusion": (
            "Primary evidence compares PLAN/SOLVE/CHECK teacher-semantic "
            "alignment, role-conditioned stochastic exploration, and the "
            "deterministic COMMIT contract from Stage 1 to Final."
        ),
        "evidence_priority": {
            "primary": "role_semantics",
            "supporting": ["role_exploration", "commit_determinism"],
            "secondary": "behavior",
            "diagnostic_only": "geometry",
        },
        "paired_questions": 200,
        "paths_per_question": 8,
        "paired_bootstrap_resamples": int(args.bootstrap),
        "role_semantics": role_semantics,
        "role_exploration": role_exploration,
        "commit_determinism": commit_audit,
        "common_geometry_questions": len(common),
        "behavior": behavior,
        "geometry": geometry,
        "geometry_evidence_status": "diagnostic_only",
        "map_rescued_questions": rescued,
        "map_regressed_questions": regressed,
        "role_claim_boundary": (
            "Role rewards are cosine semantic-alignment measurements and do "
            "not by themselves establish answer correctness or logical "
            "validity. SOLVE averages exclude masked teacher slots."
        ),
        "claim_boundary": (
            "Correct and wrong denote greedy-answer outcomes. This paired "
            "summary does not independently verify logical path validity. "
            "Outcome/corridor geometry is diagnostic-only."
        ),
    }

    stage1_counts = stage1["correct_count"]
    final_counts = final["correct_count"]
    selected = [
        index
        for index in range(200)
        if 0 < stage1_counts[index] < 8
        and 0 < final_counts[index] < 8
    ][:3]
    if not selected:
        selected = [
            index
            for index in range(200)
            if 0 < final_counts[index] < 8
        ][:3]
    stage1_residuals = torch.stack(
        [
            record["rollout_implicit_residuals"].float()
            for record in stage1_records
        ]
    )
    final_residuals = torch.stack(
        [
            record["rollout_implicit_residuals"].float()
            for record in final_records
        ]
    )
    stage1_projected = project_residuals(stage1_residuals, pca)[..., :3]
    final_projected = project_residuals(final_residuals, pca)[..., :3]
    stage1_maps = project_residuals(
        torch.stack(
            [
                record["map_implicit_residuals"].float()
                for record in stage1_records
            ]
        ),
        pca,
    )[..., :3]
    final_maps = project_residuals(
        torch.stack(
            [
                record["map_implicit_residuals"].float()
                for record in final_records
            ]
        ),
        pca,
    )[..., :3]
    trajectory_audit = []
    for index in selected:
        stage1_paths = np.concatenate(
            [
                np.zeros((8, 1, 3)),
                stage1_projected[index].cumsum(dim=1).numpy(),
            ],
            axis=1,
        )
        final_paths = np.concatenate(
            [
                np.zeros((8, 1, 3)),
                final_projected[index].cumsum(dim=1).numpy(),
            ],
            axis=1,
        )
        stage1_map = np.concatenate(
            [
                np.zeros((1, 3)),
                stage1_maps[index].cumsum(dim=0).numpy(),
            ],
            axis=0,
        )
        final_map = np.concatenate(
            [
                np.zeros((1, 3)),
                final_maps[index].cumsum(dim=0).numpy(),
            ],
            axis=0,
        )
        camera = choose_camera(
            np.concatenate([stage1_paths, final_paths], axis=0)
        )
        limits = common_axis_limits(
            [
                stage1_paths,
                final_paths,
                stage1_map[None, ...],
                final_map[None, ...],
            ]
        )
        question_id = int(stage1_records[index]["idx"])
        plot_trajectory(
            stage1_records[index],
            stage1_projected[index].numpy(),
            stage1_maps[index].numpy(),
            camera=camera,
            limits=limits,
            output_base=(
                args.output_dir / f"trajectory_q{question_id}_stage1"
            ),
            stage_label="Stage 1",
        )
        plot_trajectory(
            final_records[index],
            final_projected[index].numpy(),
            final_maps[index].numpy(),
            camera=camera,
            limits=limits,
            output_base=(
                args.output_dir / f"trajectory_q{question_id}_final"
            ),
            stage_label="Final",
        )
        trajectory_audit.append(
            {
                "question_index": question_id,
                "stage1_correct_paths": int(stage1_counts[index]),
                "final_correct_paths": int(final_counts[index]),
                "camera_score": camera[0],
                "elevation": camera[1],
                "azimuth": camera[2],
                "axis_limits": [
                    [float(lower), float(upper)]
                    for lower, upper in limits
                ],
            }
        )
    report["paired_qualitative_trajectories"] = {
        "selection": (
            "first three questions with mixed outcomes in both stages; "
            "outcome counts only"
        ),
        "shared_projection_camera_and_limits": True,
        "manual_offsets": False,
        "per_path_rescaling": False,
        "questions": trajectory_audit,
    }
    (args.output_dir / "stage_comparison.json").write_text(
        json.dumps(report, indent=2)
    )
    lines = [
        "# Paired Role-Aware TRACE Stage Comparison",
        "",
        report["core_conclusion"],
        "",
        (
            "All intervals use a paired question bootstrap over the same "
            f"200 questions ({args.bootstrap} resamples); each question has "
            "eight IID role-conditioned paths."
        ),
    ]

    def append_metric_table(title, section, names, note=None):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
            ]
        )
        if note is not None:
            lines.extend([note, ""])
        lines.extend(
            [
                "| Metric | Stage 1 | Final | Delta | 95% CI | N |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in names:
            value = section[name]
            lines.append(
                f"| {name} | {value['stage1']:.4f} | "
                f"{value['final']:.4f} | {value['delta']:+.4f} | "
                f"[{value['delta_ci95'][0]:+.4f}, "
                f"{value['delta_ci95'][1]:+.4f}] | "
                f"{value['n_questions']} |"
            )

    append_metric_table(
        "Primary: PLAN/SOLVE/CHECK teacher-semantic alignment",
        role_semantics,
        semantic_names,
        (
            "SOLVE is averaged only across teacher-observed slots. These "
            "cosine similarities measure semantic alignment, not accuracy."
        ),
    )
    append_metric_table(
        "Supporting: role-conditioned exploration",
        role_exploration,
        exploration_names,
        (
            "Standard deviation and diagonal-Gaussian entropy are computed "
            "only for stochastic PLAN/SOLVE/CHECK slots; COMMIT is excluded."
        ),
    )
    lines.extend(
        [
            "",
            "## Supporting: deterministic COMMIT audit",
            "",
            "| Stage | max |action - mean| | max |COMMIT reward| | "
            "readout-only rate | deterministic pass rate | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage, label in (("stage1", "Stage 1"), ("final", "Final")):
        audit = commit_audit[stage]
        lines.append(
            f"| {label} | {audit['max_action_mean_abs_error']:.3e} | "
            f"{audit['max_commit_role_reward']:.3e} | "
            f"{1.0 - audit['wrong_readout_rate']:.4f} | "
            f"{audit['deterministic_pass_rate']:.4f} | "
            f"{'PASS' if audit['pass'] else 'FAIL'} |"
        )
    append_metric_table(
        "Secondary: behavioral outcomes",
        behavior,
        (
            "map_accuracy",
            "map_total_length",
            "map_output_length",
            "rollout_accuracy",
            "any_correct",
            "majority_correct",
            "all_correct",
        ),
    )
    append_metric_table(
        "Diagnostic only: outcome/corridor geometry",
        geometry,
        (
            "correct_local_radius",
            "wrong_to_correct_distance",
            "outcome_margin",
            "path_diversity",
        ),
        (
            "These legacy outcome/corridor measurements are retained for "
            "continuity and are not the primary mechanism evidence."
        ),
    )
    lines.extend(
        [
            "",
            f"- MAP rescue/regression: **{rescued}/{regressed}** questions.",
            f"- Common geometry-eligible set: **{len(common)}/200** questions.",
            "",
            report["role_claim_boundary"],
            "",
            report["claim_boundary"],
        ]
    )
    (args.output_dir / "stage_comparison.md").write_text(
        "\n".join(lines)
    )
    plot_role_comparison(
        role_semantics,
        role_exploration,
        commit_audit,
        args.output_dir / "stage_role_evidence",
    )
    plot_reliability(
        stage1,
        final,
        args.output_dir / "stage_reliability",
    )
    plot_accuracy_length(
        stage1,
        final,
        args.output_dir / "stage_accuracy_length",
        bootstrap=args.bootstrap,
    )
    plot_geometry(
        geometry,
        args.output_dir / "stage_outcome_geometry",
    )
    np.savez_compressed(
        args.output_dir / "stage_comparison_source_data.npz",
        question_indices=np.asarray(stage1_indices),
        stage1_map_accuracy=stage1["map_accuracy"],
        final_map_accuracy=final["map_accuracy"],
        stage1_output_length=stage1["map_output_length"],
        final_output_length=final["map_output_length"],
        stage1_total_L=stage1["map_total_length"],
        final_total_L=final["map_total_length"],
        stage1_rollout_correctness=np.asarray(
            [
                record["rollout_correctness"]
                for record in stage1_records
            ]
        ),
        final_rollout_correctness=np.asarray(
            [
                record["rollout_correctness"]
                for record in final_records
            ]
        ),
        role_labels=np.asarray(ROLE_LABELS),
        stochastic_role_labels=np.asarray(STOCHASTIC_ROLE_LABELS),
        stage1_role_solve_mask=stage1_role["solve_mask"],
        final_role_solve_mask=final_role["solve_mask"],
        stage1_map_role_rewards=stage1_role["map_role_rewards"],
        final_map_role_rewards=final_role["map_role_rewards"],
        stage1_rollout_role_rewards=stage1_role[
            "rollout_role_rewards"
        ],
        final_rollout_role_rewards=final_role[
            "rollout_role_rewards"
        ],
        stage1_stochastic_action_std_by_role=stage1_role[
            "stochastic_action_std_by_position"
        ],
        final_stochastic_action_std_by_role=final_role[
            "stochastic_action_std_by_position"
        ],
        stage1_stochastic_gaussian_entropy_by_role=stage1_role[
            "stochastic_gaussian_entropy_by_position"
        ],
        final_stochastic_gaussian_entropy_by_role=final_role[
            "stochastic_gaussian_entropy_by_position"
        ],
        stage1_commit_action_mean_abs_error=stage1_role[
            "commit_action_mean_abs_error"
        ],
        final_commit_action_mean_abs_error=final_role[
            "commit_action_mean_abs_error"
        ],
        stage1_commit_role_reward_abs_max=stage1_role[
            "commit_role_reward_abs_max"
        ],
        final_commit_role_reward_abs_max=final_role[
            "commit_role_reward_abs_max"
        ],
        stage1_commit_readout_only=stage1_role[
            "commit_readout_only"
        ],
        final_commit_readout_only=final_role["commit_readout_only"],
        stage1_commit_deterministic=stage1_role[
            "commit_deterministic"
        ],
        final_commit_deterministic=final_role[
            "commit_deterministic"
        ],
        common_geometry_indices=np.asarray(common),
    )
    contract = {
        "core_conclusion": (
            "Stage-2 should improve PLAN/SOLVE/CHECK teacher-semantic "
            "alignment while preserving role-conditioned exploration and "
            "a strictly deterministic, answer-visible COMMIT transition."
        ),
        "figure_archetype": "quantitative grid",
        "evidence_hierarchy": {
            "hero": "stage_role_evidence",
            "supporting": [
                "stage_reliability",
                "stage_accuracy_length",
            ],
            "diagnostic_only": [
                "stage_outcome_geometry",
                "paired_qualitative_trajectories",
            ],
        },
        "panel_map": {
            "a": "MAP PLAN/SOLVE/CHECK semantic similarity",
            "b": "eight-path PLAN/SOLVE/CHECK semantic similarity",
            "c": (
                "role-conditioned std and diagonal-Gaussian entropy for "
                "the seven stochastic slots"
            ),
            "d": (
                "COMMIT action=mean, zero reward, and COMMIT-only answer "
                "readout audit"
            ),
        },
        "paired_questions": 200,
        "paths_per_question": 8,
        "role_schema": ROLE_SCHEMA,
        "solve_aggregation": "teacher-mask-aware question mean",
        "gaussian_entropy": (
            "mean(log_std + 0.5*log(2*pi*e)) per action dimension; "
            "stochastic slots 0:7 only"
        ),
        "commit_audit_tolerance": COMMIT_TOLERANCE,
        "geometry_subset": "eligible in both Stage 1 and Final",
        "geometry_evidence_status": "diagnostic_only",
        "uncertainty": (
            f"paired question-bootstrap 95% confidence intervals; "
            f"{args.bootstrap} resamples"
        ),
        "outcome_definition": "greedy decoded answer correctness",
        "path_distance": "D_TRACE; no PCA display distance",
        "shared_pca": str(args.shared_pca),
        "paired_trajectory_selection": (
            "outcome counts only; geometry unused for question selection"
        ),
        "manual_offsets": False,
        "per_path_rescaling": False,
        "source_data": "stage_comparison_source_data.npz",
        "review_risks": [
            (
                "Role cosine similarity is semantic alignment, not answer "
                "accuracy or proof of logical validity."
            ),
            (
                "COMMIT is excluded from stochastic std/entropy and must "
                "pass the explicit determinism/readout audit."
            ),
            (
                "Outcome/corridor geometry is legacy diagnostic evidence, "
                "not the role-mechanism conclusion."
            ),
        ],
        "exports": ["SVG", "PDF", "TIFF 600 dpi"],
    }
    (args.output_dir / "figure_contract.json").write_text(
        json.dumps(contract, indent=2)
    )
    qa = figure_qa(
        args.output_dir,
        expected_figures=4 + 2 * len(selected),
    )
    (args.output_dir / "figure_qa.json").write_text(
        json.dumps(qa, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
