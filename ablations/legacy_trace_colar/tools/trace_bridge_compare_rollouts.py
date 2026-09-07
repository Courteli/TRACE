#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from trace_bridge_geometry_summary import build_modes, prepare_group_signatures
from trace_bridge_visualize import as_float, path_from_residuals, path_signature, pca_fit, pca_project


COLORS = {
    "blue": "#5B7DB1",
    "green": "#66B07A",
    "gold": "#E6A516",
    "pink": "#E5A6C4",
    "text": "#29313A",
    "muted": "#69727D",
    "grid": "#DDE1E6",
}
METHOD_COLORS = {
    "BRIDGE": COLORS["blue"],
    "Stage1": COLORS["green"],
    "TRACE-epoch7": COLORS["gold"],
    "TRACE epoch7": COLORS["gold"],
    "TRACE": COLORS["gold"],
    "TRACE seed1": COLORS["pink"],
}
FONT_STACK = ["Times New Roman", "Liberation Serif", "Nimbus Roman No9 L", "DejaVu Serif"]
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": FONT_STACK,
    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def save_figure(fig, out_path):
    out_path = Path(out_path)
    for suffix, kwargs in (
        (".png", {"dpi": 400}),
        (".pdf", {}),
        (".svg", {}),
        (".tiff", {"dpi": 600}),
    ):
        fig.savefig(out_path.with_suffix(suffix), bbox_inches="tight", pad_inches=0.04, **kwargs)


def figure_exports_complete(out_path):
    out_path = Path(out_path)
    return all(out_path.with_suffix(suffix).is_file() for suffix in (".png", ".pdf", ".svg", ".tiff"))


def parse_record_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--record must be LABEL=/absolute/path.pt")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_records(path, max_records):
    records = torch.load(path, map_location="cpu", weights_only=False)
    return {int(record["idx"]): record for record in records[:max_records]}


def record_paths(record, normalize_for_plot):
    paths = [("target", path_from_residuals(as_float(record["aggregated_explicit_residuals"]), normalize_for_plot))]
    if "multiview_implicit_residuals" in record:
        residuals = as_float(record["multiview_implicit_residuals"])
        outcomes = as_float(record.get("multiview_acc", np.zeros(residuals.shape[0]))).reshape(-1) > 0.5
        for view_idx, item in enumerate(residuals):
            paths.append((f"view{view_idx}", path_from_residuals(item, normalize_for_plot), bool(outcomes[view_idx])))
    else:
        paths.append(("view0", path_from_residuals(as_float(record["implicit_residuals"]), normalize_for_plot), None))
    return paths


def record_correct_mode_assignments(
    record,
    max_modes=3,
    merge_threshold=0.65,
    signature_representation="raw",
    signature_raw_mix=0.25,
):
    if "multiview_implicit_residuals" not in record or "multiview_acc" not in record:
        return np.asarray([], dtype=np.int64)
    residuals = as_float(record["multiview_implicit_residuals"])
    outcomes = as_float(record["multiview_acc"]).reshape(-1) > 0.5
    assignments = np.full(len(outcomes), -1, dtype=np.int64)
    if outcomes.any():
        signatures = np.stack([path_signature(np.cumsum(item, axis=0)) for item in residuals], axis=0)
        signatures = prepare_group_signatures(
            signatures,
            representation=signature_representation,
            raw_mix=signature_raw_mix,
        )[outcomes]
        _, positive_assignments, _, _ = build_modes(
            signatures,
            max_modes=max_modes,
            merge_threshold=merge_threshold,
        )
        assignments[outcomes] = positive_assignments
    return assignments


def ordered_rollout_similarity(
    record,
    max_modes=3,
    merge_threshold=0.65,
    signature_representation="raw",
    signature_raw_mix=0.25,
):
    if "multiview_implicit_residuals" not in record or "multiview_acc" not in record:
        return None
    residuals = as_float(record["multiview_implicit_residuals"])
    outcomes = as_float(record["multiview_acc"]).reshape(-1) > 0.5
    if residuals.shape[0] != outcomes.shape[0]:
        return None
    signatures = np.stack(
        [path_signature(np.cumsum(item, axis=0)) for item in residuals],
        axis=0,
    )
    signatures = prepare_group_signatures(
        signatures,
        representation=signature_representation,
        raw_mix=signature_raw_mix,
    )
    mode_ids = record_correct_mode_assignments(
        record,
        max_modes=max_modes,
        merge_threshold=merge_threshold,
        signature_representation=signature_representation,
        signature_raw_mix=signature_raw_mix,
    )
    original_indices = np.arange(len(outcomes))
    order = np.lexsort((original_indices, mode_ids, ~outcomes))
    ordered_signatures = signatures[order]
    ordered_outcomes = outcomes[order]
    ordered_modes = mode_ids[order]
    ordered_indices = original_indices[order]
    labels = [
        f"M{ordered_modes[pos] + 1}:v{ordered_indices[pos]}"
        if is_correct
        else f"W:v{ordered_indices[pos]}"
        for pos, is_correct in enumerate(ordered_outcomes)
    ]
    return {
        "similarity": ordered_signatures @ ordered_signatures.T,
        "labels": labels,
        "outcomes": ordered_outcomes,
        "mode_count": len(set(ordered_modes[ordered_modes >= 0].tolist())),
    }


def select_outcome_balanced_indices(record_sets, common_indices, count=3):
    scored = []
    for idx in common_indices:
        mixed_methods = 0
        balance = 0
        for records in record_sets.values():
            outcomes = as_float(records[idx].get("multiview_acc", np.asarray([]))).reshape(-1) > 0.5
            n_correct = int(outcomes.sum())
            n_wrong = int(len(outcomes) - n_correct)
            if n_correct and n_wrong:
                mixed_methods += 1
                balance += min(n_correct, n_wrong)
        scored.append((mixed_methods, balance, -idx, idx))
    selected = [item[-1] for item in sorted(scored, reverse=True)[:count]]
    return selected if selected else common_indices[:count]


def make_plot(
    record_sets,
    common_indices,
    shown_indices,
    out_path,
    pca_fit_records,
    normalize_for_plot,
    selection_rule,
    signature_representation,
    signature_raw_mix,
):
    fit_indices = common_indices[:pca_fit_records]
    all_points = []
    for records in record_sets.values():
        for idx in fit_indices:
            for path_item in record_paths(records[idx], normalize_for_plot):
                all_points.append(path_item[1])
    stacked = np.concatenate(all_points, axis=0)
    mean, components, explained = pca_fit(stacked, n_components=3)
    shown_points = []
    for records in record_sets.values():
        for idx in shown_indices:
            for path_item in record_paths(records[idx], normalize_for_plot):
                shown_points.append(pca_project(path_item[1], mean, components))
    shown_projection = np.concatenate(shown_points, axis=0)
    axis_min = shown_projection.min(axis=0)
    axis_max = shown_projection.max(axis=0)
    axis_range = np.maximum(axis_max - axis_min, 1e-6)
    axis_min = axis_min - 0.05 * axis_range
    axis_max = axis_max + 0.05 * axis_range

    rows = len(record_sets)
    cols = len(shown_indices)
    fig = plt.figure(figsize=(7.2, 5.85))
    for row_idx, (label, records) in enumerate(record_sets.items()):
        for col_idx, idx in enumerate(shown_indices):
            ax = fig.add_subplot(rows, cols, row_idx * cols + col_idx + 1, projection="3d")
            correct_modes_labeled = set()
            wrong_labeled = False
            mode_assignments = record_correct_mode_assignments(
                records[idx],
                signature_representation=signature_representation,
                signature_raw_mix=signature_raw_mix,
            )
            mode_colors = (COLORS["blue"], COLORS["green"], COLORS["gold"])
            for path_item in record_paths(records[idx], normalize_for_plot):
                name, path = path_item[:2]
                projection = pca_project(path, mean, components)
                if name == "target":
                    ax.plot(*projection.T, color=COLORS["text"], linestyle="--", linewidth=1.15, label="teacher target")
                    ax.scatter(*projection.T, color=COLORS["text"], s=4, alpha=0.70)
                    continue
                outcome = path_item[2]
                view_idx = int(name.removeprefix("view"))
                mode_idx = int(mode_assignments[view_idx]) if view_idx < len(mode_assignments) else -1
                color = (
                    mode_colors[mode_idx % len(mode_colors)]
                    if outcome
                    else COLORS["pink"] if outcome is not None else COLORS["blue"]
                )
                legend = None
                if outcome and mode_idx not in correct_modes_labeled:
                    legend = f"correct mode {mode_idx + 1}"
                    correct_modes_labeled.add(mode_idx)
                elif outcome is False and not wrong_labeled:
                    legend, wrong_labeled = "wrong rollout", True
                ax.plot(*projection.T, color=color, linewidth=1.0, alpha=0.82, label=legend)
                ax.scatter(*projection.T, color=color, s=np.linspace(3, 10, len(projection)), alpha=0.68)
                if len(projection) > 1:
                    movement = np.diff(projection, axis=0)
                    ax.quiver(
                        projection[:-1, 0],
                        projection[:-1, 1],
                        projection[:-1, 2],
                        movement[:, 0],
                        movement[:, 1],
                        movement[:, 2],
                        color=color,
                        alpha=0.45,
                        linewidth=0.35,
                        arrow_length_ratio=0.16,
                        normalize=False,
                    )
                endpoint_marker = "o" if outcome else "X" if outcome is not None else "o"
                ax.scatter(*projection[-1], color=color, s=15, marker=endpoint_marker)
            outcomes = as_float(records[idx].get("multiview_acc", np.asarray([]))).reshape(-1)
            rollout_acc = float(outcomes.mean()) if outcomes.size else float(records[idx].get("acc", 0.0))
            n_modes = len(set(mode_assignments[mode_assignments >= 0].tolist()))
            ax.set_title(
                f"{label} | q{idx}\nacc={rollout_acc:.2f}; modes={n_modes}",
                fontsize=6.3,
                color=METHOD_COLORS.get(label, COLORS["text"]),
                fontweight="bold",
                pad=0,
            )
            ax.set_xlabel("PC1" if row_idx == rows - 1 else "", labelpad=-2)
            ax.set_ylabel("PC2" if row_idx == rows - 1 else "", labelpad=-2)
            ax.set_zlabel("PC3" if col_idx == 0 else "", labelpad=-3)
            ax.set_xlim(float(axis_min[0]), float(axis_max[0]))
            ax.set_ylim(float(axis_min[1]), float(axis_max[1]))
            ax.set_zlim(float(axis_min[2]), float(axis_max[2]))
            # Keep common numeric limits while avoiding a physically elongated 3D box.
            ax.set_box_aspect((1.0, 0.90, 0.68), zoom=1.12)
            ax.view_init(elev=21, azim=-58)
            ax.tick_params(axis="both", which="major", labelsize=4.5, pad=-2)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(3))
                axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                axis.pane.set_edgecolor(COLORS["grid"])
                axis._axinfo["grid"].update(color=COLORS["grid"], linewidth=0.35)
            ax.text2D(-0.02, 1.00, chr(ord("a") + row_idx * cols + col_idx), transform=ax.transAxes, fontsize=7.2, fontweight="bold")
    legend_handles = [
        Line2D([0], [0], color=COLORS["text"], linestyle="--", linewidth=1.15, label="teacher target"),
        Line2D([0], [0], color=COLORS["blue"], linewidth=1.4, label="correct mode 1"),
        Line2D([0], [0], color=COLORS["green"], linewidth=1.4, label="correct mode 2"),
        Line2D([0], [0], color=COLORS["gold"], linewidth=1.4, label="correct mode 3"),
        Line2D([0], [0], color=COLORS["pink"], linewidth=1.4, label="wrong rollout"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.003), handlelength=1.7)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.080, hspace=0.10, wspace=0.02)
    save_figure(fig, out_path)
    plt.close(fig)
    return {
        "labels": list(record_sets),
        "common_question_count": len(common_indices),
        "pca_fit_question_indices": [int(idx) for idx in fit_indices],
        "shown_indices": shown_indices,
        "selection_rule": selection_rule,
        "pca_fit_question_count": len(fit_indices),
        "pca_fit_point_count": int(stacked.shape[0]),
        "path_definition": "origin followed by the cumulative implicit-residual path",
        "outcome_annotation": (
            "each rollout is labeled by exact answer verification for that question; "
            "correct-mode labels are clustered within that question only"
        ),
        "projection_scope": "one PCA fit jointly over every method and the same fit-question IDs",
        "explained_variance_ratio": [float(value) for value in explained],
        "shared_axis_limits": {
            "min": [float(value) for value in axis_min],
            "max": [float(value) for value in axis_max],
            "scope": "all points in every displayed panel",
        },
        "normalization": "final_path_norm" if normalize_for_plot else "raw_cumulative_residual",
        "mode_signature_representation": signature_representation,
        "mode_signature_raw_mix": signature_raw_mix if signature_representation == "stage2_centered" else None,
    }


def normalized_rollout_path_tensor(record):
    residuals = as_float(record["multiview_implicit_residuals"])
    paths = np.cumsum(residuals, axis=1)
    paths = np.concatenate([np.zeros_like(paths[:, :1]), paths], axis=1)
    final_norm = np.linalg.norm(paths[:, -1], axis=-1, keepdims=True)
    paths = paths / np.clip(final_norm[:, None, :], 1e-8, None)
    return paths.astype(np.float32, copy=False)


def residualized_path_tensor(records, common_indices):
    paths = np.stack([normalized_rollout_path_tensor(records[idx]) for idx in common_indices])
    # Leave-one-question-out view-template subtraction has this closed form.
    factor = len(common_indices) / max(len(common_indices) - 1, 1)
    paths = factor * (paths - paths.mean(axis=0, keepdims=True))
    # Remove the question's shared path so the plot shows only branching deviations.
    paths = paths - paths.mean(axis=1, keepdims=True)
    question_scale = np.sqrt(np.mean(np.sum(paths * paths, axis=-1), axis=(1, 2), keepdims=True))
    paths = paths / np.clip(question_scale[..., None], 1e-8, None)
    return paths.astype(np.float32, copy=False)


def make_view_residualized_path_plot(
    record_sets,
    common_indices,
    shown_indices,
    out_path,
    selection_rule,
    signature_representation,
    signature_raw_mix,
):
    tensors = {
        label: residualized_path_tensor(records, common_indices)
        for label, records in record_sets.items()
    }
    index_positions = {idx: position for position, idx in enumerate(common_indices)}
    projections = {label: {} for label in tensors}
    max_abs_by_question = {}
    explained_by_question = {}
    for idx in shown_indices:
        question_paths = {
            label: tensor[index_positions[idx]]
            for label, tensor in tensors.items()
        }
        fit_points = np.concatenate(
            [paths[:, 1:, :].reshape(-1, paths.shape[-1]) for paths in question_paths.values()],
            axis=0,
        )
        mean, components, explained = pca_fit(fit_points, n_components=3)
        fit_scores = pca_project(fit_points, mean, components)
        axis_scale = np.clip(fit_scores.std(axis=0), 1e-8, None)
        shown_points = []
        for label, paths in question_paths.items():
            scores = pca_project(paths.reshape(-1, paths.shape[-1]), mean, components) / axis_scale
            scores = scores.reshape(paths.shape[0], paths.shape[1], 3)
            projections[label][idx] = scores
            shown_points.append(scores.reshape(-1, 3))
        shown_points = np.concatenate(shown_points, axis=0)
        max_abs_by_question[idx] = np.maximum(np.abs(shown_points).max(axis=0) * 1.08, 0.25)
        explained_by_question[idx] = [float(value) for value in explained]

    rows = len(record_sets)
    cols = len(shown_indices)
    fig = plt.figure(figsize=(7.2, 5.85))
    panel_metadata = []
    for row_idx, (label, records) in enumerate(record_sets.items()):
        for col_idx, idx in enumerate(shown_indices):
            ax = fig.add_subplot(rows, cols, row_idx * cols + col_idx + 1, projection="3d")
            outcomes = as_float(records[idx]["multiview_acc"]).reshape(-1) > 0.5
            assignments = record_correct_mode_assignments(
                records[idx],
                signature_representation=signature_representation,
                signature_raw_mix=signature_raw_mix,
            )
            mode_colors = (COLORS["blue"], COLORS["green"], COLORS["gold"])
            scores = projections[label][idx]
            max_abs = max_abs_by_question[idx]
            for view_idx, path in enumerate(scores):
                outcome = bool(outcomes[view_idx])
                mode_idx = int(assignments[view_idx]) if outcome else -1
                color = mode_colors[mode_idx % len(mode_colors)] if outcome else COLORS["pink"]
                ax.plot(*path.T, color=color, linewidth=1.25 if outcome else 1.0, alpha=0.82)
                ax.scatter(*path.T, color=color, s=np.linspace(3, 11, len(path)), alpha=0.72)
                movement = np.diff(path, axis=0)
                if len(movement):
                    ax.quiver(
                        path[:-1, 0], path[:-1, 1], path[:-1, 2],
                        movement[:, 0], movement[:, 1], movement[:, 2],
                        color=color, alpha=0.38, linewidth=0.35,
                        arrow_length_ratio=0.16, normalize=False,
                    )
                ax.scatter(*path[-1], color=color, s=18, marker="o" if outcome else "X", edgecolor="white", linewidth=0.35)
            rollout_acc = float(outcomes.mean())
            n_modes = len(set(assignments[assignments >= 0].tolist()))
            ax.set_title(
                f"{label} | q{idx}\nacc={rollout_acc:.2f}; modes={n_modes}",
                fontsize=6.3,
                color=METHOD_COLORS.get(label, COLORS["text"]),
                fontweight="bold",
                pad=0,
            )
            ax.set_xlim(-max_abs[0], max_abs[0])
            ax.set_ylim(-max_abs[1], max_abs[1])
            ax.set_zlim(-max_abs[2], max_abs[2])
            ax.set_xlabel("rPC1" if row_idx == rows - 1 else "", labelpad=-2)
            ax.set_ylabel("rPC2" if row_idx == rows - 1 else "", labelpad=-2)
            ax.set_zlabel("rPC3" if col_idx == 0 else "", labelpad=-3)
            ax.set_box_aspect((1.0, 0.92, 0.72), zoom=1.12)
            ax.view_init(elev=22, azim=-57)
            ax.tick_params(axis="both", which="major", labelsize=4.5, pad=-2)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(3))
                axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                axis.pane.set_edgecolor(COLORS["grid"])
                axis._axinfo["grid"].update(color=COLORS["grid"], linewidth=0.35)
            if row_idx < rows - 1:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            if col_idx > 0:
                ax.set_zticklabels([])
            ax.text2D(-0.02, 1.00, chr(ord("a") + row_idx * cols + col_idx), transform=ax.transAxes, fontsize=7.2, fontweight="bold")
            panel_metadata.append({
                "method": label,
                "idx": int(idx),
                "rollout_accuracy": rollout_acc,
                "correct_mode_count": n_modes,
            })

    legend_handles = [
        Line2D([0], [0], color=COLORS["blue"], linewidth=1.5, label="correct mode 1"),
        Line2D([0], [0], color=COLORS["green"], linewidth=1.5, label="correct mode 2"),
        Line2D([0], [0], color=COLORS["gold"], linewidth=1.5, label="correct mode 3"),
        Line2D([0], [0], color=COLORS["pink"], linewidth=1.5, label="wrong rollout"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.012), handlelength=1.8)
    fig.text(
        0.5,
        0.003,
        "LOO view templates use all 200 questions; each column uses one outcome-free PCA shared by BRIDGE, Stage1 and TRACE.",
        ha="center",
        fontsize=5.6,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.095, hspace=0.10, wspace=0.02)
    save_figure(fig, out_path)
    plt.close(fig)
    return {
        "labels": list(record_sets),
        "common_question_count": len(common_indices),
        "shown_indices": [int(idx) for idx in shown_indices],
        "selection_rule": selection_rule,
        "projection": "for each shown question, joint PCA over the three methods' LOO view-template-residualized, within-question-centered normalized rollout paths",
        "axis_scaling": "each PCA coordinate divided by its standard deviation within the question; coordinates are shared down each column but not across columns",
        "outcome_usage": "outcomes color paths after projection; outcomes are not used to fit residualization, PCA, or axis scaling",
        "explained_variance_ratio_by_question": explained_by_question,
        "panels": panel_metadata,
    }


def make_trace_exploded_path_plot(
    record_sets,
    common_indices,
    shown_indices,
    out_path,
    selection_rule,
    signature_representation,
    signature_raw_mix,
):
    tensors = {
        label: residualized_path_tensor(records, common_indices)
        for label, records in record_sets.items()
    }
    index_positions = {idx: position for position, idx in enumerate(common_indices)}
    trace_label = next(label for label in record_sets if label.lower().startswith("trace"))
    stage1_label = next(label for label in record_sets if label.lower() == "stage1")

    fig = plt.figure(figsize=(7.2, 2.72))
    metadata = []
    for col_idx, idx in enumerate(shown_indices):
        question_paths = {
            label: tensor[index_positions[idx]]
            for label, tensor in tensors.items()
        }
        fit_points = np.concatenate(
            [paths[:, 1:, :].reshape(-1, paths.shape[-1]) for paths in question_paths.values()],
            axis=0,
        )
        mean, components, explained = pca_fit(fit_points, n_components=2)
        trace_paths = question_paths[trace_label]
        scores = pca_project(trace_paths.reshape(-1, trace_paths.shape[-1]), mean, components)
        scores = scores.reshape(trace_paths.shape[0], trace_paths.shape[1], 2)

        ax = fig.add_subplot(1, len(shown_indices), col_idx + 1, projection="3d")
        outcomes = as_float(record_sets[trace_label][idx]["multiview_acc"]).reshape(-1) > 0.5
        assignments = record_correct_mode_assignments(
            record_sets[trace_label][idx],
            signature_representation=signature_representation,
            signature_raw_mix=signature_raw_mix,
        )
        mode_colors = (COLORS["blue"], COLORS["green"], COLORS["gold"])
        for view_idx, path in enumerate(scores):
            path = path - path[0]
            projected_steps = np.diff(path, axis=0)
            step_lengths = np.linalg.norm(projected_steps, axis=1)
            nonzero = step_lengths[step_lengths > 1e-8]
            reference_length = float(np.median(nonzero)) if len(nonzero) else 1.0
            # Preserve every projected step direction while compressing extreme
            # length ratios so all eight latent transitions remain visible.
            visible_lengths = np.clip(
                np.sqrt(step_lengths / max(reference_length, 1e-8)),
                0.55,
                1.55,
            )
            visible_steps = projected_steps / np.clip(step_lengths[:, None], 1e-8, None)
            visible_steps = visible_steps * visible_lengths[:, None]
            direction_path = np.concatenate(
                [np.zeros((1, 2), dtype=visible_steps.dtype), np.cumsum(visible_steps, axis=0)],
                axis=0,
            )
            visible_arc = np.linalg.norm(np.diff(direction_path, axis=0), axis=1).sum()
            direction_path = direction_path * (5.8 / max(float(visible_arc), 1e-8))
            deviation_radius = np.linalg.norm(direction_path, axis=1).max()
            direction_path = direction_path * (2.15 / max(float(deviation_radius), 1e-8))
            lane_col = view_idx % 4
            lane_row = view_idx // 4
            offset = np.asarray([
                (lane_col - 1.5) * 5.1,
                (lane_row - 0.5) * 6.0,
            ])
            latent_step = np.arange(len(path), dtype=np.float32)
            display_path = np.column_stack([
                latent_step,
                offset[0] + direction_path[:, 0],
                offset[1] + direction_path[:, 1],
            ])
            outcome = bool(outcomes[view_idx])
            mode_idx = int(assignments[view_idx]) if outcome else -1
            color = mode_colors[mode_idx % len(mode_colors)] if outcome else COLORS["pink"]
            ax.plot(
                [latent_step[0], latent_step[-1]],
                [offset[0], offset[0]],
                [offset[1], offset[1]],
                color=COLORS["grid"],
                linewidth=0.45,
                alpha=0.75,
                zorder=0,
            )
            ax.plot(*display_path.T, color=color, linewidth=2.05, alpha=0.96)
            ax.scatter(*display_path.T, color=color, s=np.linspace(4, 15, len(display_path)), alpha=0.86)
            movement = np.diff(display_path, axis=0)
            ax.quiver(
                display_path[:-1, 0], display_path[:-1, 1], display_path[:-1, 2],
                movement[:, 0], movement[:, 1], movement[:, 2],
                color=color, alpha=0.82, linewidth=0.82,
                arrow_length_ratio=0.22, normalize=False,
            )
            ax.scatter(*display_path[0], color=COLORS["text"], s=10, marker="s", alpha=0.75)
            ax.scatter(*display_path[-1], color=color, s=29, marker="o" if outcome else "X", edgecolor="white", linewidth=0.45)
            ax.text(*display_path[0], f" v{view_idx}", fontsize=5.0, color=COLORS["text"])

        bridge_correct = int((as_float(record_sets["BRIDGE"][idx]["multiview_acc"]).reshape(-1) > 0.5).sum())
        stage1_correct = int((as_float(record_sets[stage1_label][idx]["multiview_acc"]).reshape(-1) > 0.5).sum())
        trace_correct = int(outcomes.sum())
        ax.set_title(
            f"q{idx}: BRIDGE {bridge_correct}/8 | Stage1 {stage1_correct}/8 | TRACE {trace_correct}/8",
            fontsize=6.6,
            color=COLORS["text"],
            fontweight="bold",
            pad=2,
        )
        ax.set_xlim(-0.35, 8.45)
        ax.set_ylim(-10.4, 10.4)
        ax.set_zlim(-5.6, 5.6)
        ax.set_box_aspect((1.38, 1.05, 0.76), zoom=1.18)
        ax.set_proj_type("ortho")
        ax.view_init(elev=22, azim=-67)
        ax.set_xticks([0.0, 4.0, 8.0], ["z0", "z4", "z8"], fontsize=5.2)
        ax.set_xlabel("latent step", labelpad=-4, fontsize=6.0)
        ax.set_yticks([])
        ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor(COLORS["grid"])
            axis._axinfo["grid"].update(color=COLORS["grid"], linewidth=0.30)
        ax.text2D(-0.02, 1.02, chr(ord("a") + col_idx), transform=ax.transAxes, fontsize=7.4, fontweight="bold")
        metadata.append({
            "idx": int(idx),
            "bridge_correct": bridge_correct,
            "stage1_correct": stage1_correct,
            "trace_correct": trace_correct,
            "local_pca_explained_variance_ratio": [float(value) for value in explained],
        })

    legend_handles = [
        Line2D([0], [0], color=COLORS["blue"], linewidth=1.7, label="correct mode 1"),
        Line2D([0], [0], color=COLORS["green"], linewidth=1.7, label="correct mode 2"),
        Line2D([0], [0], color=COLORS["gold"], linewidth=1.7, label="correct mode 3"),
        Line2D([0], [0], color=COLORS["pink"], linewidth=1.7, label="wrong rollout"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.075), handlelength=1.8)
    fig.text(
        0.5,
        0.018,
        "Each line advances from z0 to z8; PC turns retain all projected step directions while extreme step-length ratios are compressed. Fixed lanes are not geometry evidence.",
        ha="center",
        fontsize=5.7,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.92, bottom=0.18, wspace=0.02)
    save_figure(fig, out_path)
    plt.close(fig)
    return {
        "shown_indices": [int(idx) for idx in shown_indices],
        "selection_rule": selection_rule,
        "residualization": "LOO view-template subtraction and within-question path centering before local outcome-free 2D PCA",
        "display_transform": "latent step is the longitudinal axis; fixed 4x2 view-index lanes carry projected PC turns after direction preservation, square-root magnitude compression and clipping, and fixed radial display extent",
        "claim_boundary": "display transform exposes the ordered direction sequence of each path; absolute step magnitudes and inter-lane distances are not quantitative geometry",
        "outcome_usage": "outcomes only set colors and endpoint markers after projection and lane placement",
        "panels": metadata,
    }


def make_comparative_similarity_plot(
    record_sets,
    shown_indices,
    out_path,
    selection_rule,
    signature_representation,
    signature_raw_mix,
):
    rows = len(record_sets)
    cols = len(shown_indices)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.25 * cols, 2.15 * rows),
        squeeze=False,
    )
    image = None
    panel_metadata = []
    for row_idx, (label, records) in enumerate(record_sets.items()):
        for col_idx, idx in enumerate(shown_indices):
            ax = axes[row_idx, col_idx]
            ordered = ordered_rollout_similarity(
                records[idx],
                signature_representation=signature_representation,
                signature_raw_mix=signature_raw_mix,
            )
            if ordered is None:
                ax.axis("off")
                continue
            similarity = ordered["similarity"]
            labels = ordered["labels"]
            outcomes = ordered["outcomes"]
            similarity_cmap = LinearSegmentedColormap.from_list(
                "trace_blue_pink",
                [COLORS["blue"], "#F7F7F5", COLORS["pink"]],
            )
            image = ax.imshow(similarity, vmin=-1.0, vmax=1.0, cmap=similarity_cmap)
            ax.set_xticks(range(len(labels)), labels=labels, fontsize=5.1, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)), labels=labels, fontsize=5.1)
            ax.set_title(
                f"{label} | q{idx}\nacc={outcomes.mean():.2f}; modes={ordered['mode_count']}",
                fontsize=6.3,
                color=METHOD_COLORS.get(label, COLORS["text"]),
                fontweight="bold",
                pad=2,
            )
            ax.text(-0.22, 1.08, chr(ord("a") + row_idx * cols + col_idx), transform=ax.transAxes, fontsize=7.2, fontweight="bold")
            panel_metadata.append(
                {
                    "method": label,
                    "idx": int(idx),
                    "rollout_accuracy": float(outcomes.mean()),
                    "correct_mode_count": int(ordered["mode_count"]),
                    "ordered_labels": labels,
                }
            )
    fig.subplots_adjust(
        top=0.965,
        bottom=0.085,
        left=0.06,
        right=0.91,
        hspace=0.42,
        wspace=0.32,
    )
    if image is not None:
        colorbar_ax = fig.add_axes([0.94, 0.36, 0.012, 0.28])
        fig.colorbar(image, cax=colorbar_ax, label="cosine similarity")
    fig.text(
        0.5,
        0.012,
        "Correct rollouts are grouped by within-question mode (M); wrong rollouts are marked W.",
        ha="center",
        fontsize=6.0,
        color=COLORS["muted"],
    )
    save_figure(fig, out_path)
    plt.close(fig)
    return {
        "shown_indices": [int(idx) for idx in shown_indices],
        "selection_rule": selection_rule,
        "signature_definition": "normalized mean/last/trend/delta of the cumulative implicit-residual path",
        "signature_representation": signature_representation,
        "signature_raw_mix": signature_raw_mix if signature_representation == "stage2_centered" else None,
        "ordering": "correct rollouts grouped by within-question mode, followed by wrong rollouts",
        "shared_color_limits": [-1.0, 1.0],
        "panels": panel_metadata,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", type=parse_record_arg, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument("--pca_fit_records", type=int, default=200)
    parser.add_argument(
        "--signature_representation",
        choices=("raw", "stage2_centered"),
        default="raw",
        help="Representation used for correct-mode coloring and rollout-similarity heatmaps.",
    )
    parser.add_argument("--signature_raw_mix", type=float, default=0.25)
    parser.add_argument("--force", action="store_true", help="Regenerate figure exports even when all formats already exist.")
    parser.add_argument(
        "--selection_names",
        nargs="+",
        choices=("fixed", "outcome_balanced", "objective"),
        default=("fixed", "outcome_balanced", "objective"),
        help="Selection variants to render.",
    )
    parser.add_argument(
        "--normalizations",
        nargs="+",
        choices=("normalized", "raw"),
        default=("normalized", "raw"),
        help="3D path representations to render.",
    )
    parser.add_argument(
        "--make_residualized_paths",
        action="store_true",
        help="Render an outcome-free LOO view-template-residualized 3D path comparison for the objective selection.",
    )
    parser.add_argument(
        "--make_exploded_paths",
        action="store_true",
        help="Render a display-only TRACE exploded-path view with fixed outcome-independent view lanes.",
    )
    parser.add_argument(
        "--show_indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional predeclared question indices to add as an explicit non-geometry selection.",
    )
    parser.add_argument(
        "--show_selection_rule",
        default="predeclared question indices; selection does not use geometry",
    )
    parser.add_argument(
        "--require_common_questions",
        type=int,
        default=0,
        help="Fail instead of silently rendering a smaller shared-question set when this minimum is unmet.",
    )
    args = parser.parse_args()

    record_sets = {label: load_records(path, args.max_records) for label, path in args.record}
    common = sorted(set.intersection(*(set(records) for records in record_sets.values())))
    if not common:
        raise RuntimeError("No common question indices across record files")
    if args.require_common_questions > 0 and len(common) < args.require_common_questions:
        raise RuntimeError(
            "Insufficient common question records for a controlled global PCA: "
            f"required={args.require_common_questions}, found={len(common)}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry_json = out_dir / "trace_bridge_same_question_global_pca_3d.json"
    similarity_json = out_dir / "trace_bridge_same_question_rollout_similarity.json"
    summaries = json.loads(geometry_json.read_text(encoding="utf-8")) if geometry_json.is_file() else {}
    selections = {
        "fixed": common[:3],
        "outcome_balanced": select_outcome_balanced_indices(record_sets, common, count=3),
    }
    if args.show_indices:
        missing = [idx for idx in args.show_indices if idx not in common]
        if missing:
            raise RuntimeError(f"Predeclared --show_indices are not common to every method: {missing}")
        selections["objective"] = list(dict.fromkeys(args.show_indices))
    selections = {name: selections[name] for name in args.selection_names if name in selections}
    missing_selections = [name for name in args.selection_names if name not in selections]
    if missing_selections:
        raise RuntimeError(f"Requested selections need additional arguments: {missing_selections}")
    normalization_options = {"normalized": True, "raw": False}
    for suffix in args.normalizations:
        normalize_for_plot = normalization_options[suffix]
        for selection_name, shown_indices in selections.items():
            file_suffix = suffix if selection_name == "fixed" else f"{suffix}_{selection_name}"
            out_path = out_dir / f"trace_bridge_same_question_global_pca_3d_{file_suffix}.png"
            if not args.force and figure_exports_complete(out_path) and file_suffix in summaries:
                continue
            summaries[file_suffix] = make_plot(
                record_sets,
                common,
                shown_indices,
                out_path,
                pca_fit_records=args.pca_fit_records,
                normalize_for_plot=normalize_for_plot,
                selection_rule=(
                    "first common question indices"
                    if selection_name == "fixed"
                    else args.show_selection_rule
                    if selection_name == "objective"
                    else "most outcome-balanced questions; selection uses correct/wrong counts only, not geometry"
                ),
                signature_representation=args.signature_representation,
                signature_raw_mix=args.signature_raw_mix,
            )
    similarity_summaries = json.loads(similarity_json.read_text(encoding="utf-8")) if similarity_json.is_file() else {}
    for selection_name, shown_indices in selections.items():
        selection_rule = (
            "first common question indices"
            if selection_name == "fixed"
            else args.show_selection_rule
            if selection_name == "objective"
            else "most outcome-balanced questions; selection uses correct/wrong counts only, not geometry"
        )
        out_path = out_dir / f"trace_bridge_same_question_rollout_similarity_{selection_name}.png"
        if not args.force and figure_exports_complete(out_path) and selection_name in similarity_summaries:
            continue
        similarity_summaries[selection_name] = make_comparative_similarity_plot(
            record_sets,
            shown_indices,
            out_path,
            selection_rule=selection_rule,
            signature_representation=args.signature_representation,
            signature_raw_mix=args.signature_raw_mix,
        )
    geometry_json.write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    similarity_json.write_text(
        json.dumps(similarity_summaries, indent=2),
        encoding="utf-8",
    )
    if args.make_residualized_paths:
        if "objective" not in selections:
            raise RuntimeError("--make_residualized_paths requires objective selection and --show_indices")
        residualized_path = out_dir / "trace_bridge_same_question_residualized_paths_3d_objective.png"
        residualized_summary = make_view_residualized_path_plot(
            record_sets,
            common,
            selections["objective"],
            residualized_path,
            selection_rule=args.show_selection_rule,
            signature_representation=args.signature_representation,
            signature_raw_mix=args.signature_raw_mix,
        )
        (out_dir / "trace_bridge_same_question_residualized_paths_3d_objective.json").write_text(
            json.dumps(residualized_summary, indent=2),
            encoding="utf-8",
        )
    if args.make_exploded_paths:
        if "objective" not in selections:
            raise RuntimeError("--make_exploded_paths requires objective selection and --show_indices")
        exploded_path = out_dir / "trace_exploded_paths_3d_objective.png"
        exploded_summary = make_trace_exploded_path_plot(
            record_sets,
            common,
            selections["objective"],
            exploded_path,
            selection_rule=args.show_selection_rule,
            signature_representation=args.signature_representation,
            signature_raw_mix=args.signature_raw_mix,
        )
        (out_dir / "trace_exploded_paths_3d_objective.json").write_text(
            json.dumps(exploded_summary, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
