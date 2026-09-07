#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trace_bridge_geometry_summary import build_modes


def load_records(paths):
    records = []
    for path in paths:
        records.extend(torch.load(path, map_location="cpu", weights_only=False))
    return records


def as_float(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def cosine_matrix(a, b):
    a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8, None)
    return a @ b.T


def pca_fit(points, n_components=3, oversamples=16, n_iter=2):
    mean = points.mean(axis=0, keepdims=True)
    centered = (points - mean).astype(np.float32, copy=False)
    n_samples, n_features = centered.shape
    rank = min(n_components + oversamples, n_samples, n_features)
    if rank <= n_components + 2 or n_samples <= 64:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh[:n_components].T
    else:
        rng = np.random.default_rng(0)
        omega = rng.normal(size=(n_features, rank)).astype(np.float32)
        y = centered @ omega
        for _ in range(max(0, n_iter)):
            y = centered @ (centered.T @ y)
        q, _ = np.linalg.qr(y, mode="reduced")
        b = q.T @ centered
        _, singular_values, vh = np.linalg.svd(b, full_matrices=False)
        components = vh[:n_components].T
    variances = singular_values**2
    total = float((centered * centered).sum())
    explained = variances[:n_components] / total if total > 0 else np.zeros(n_components, dtype=np.float32)
    return mean.astype(np.float32, copy=False), components.astype(np.float32, copy=False), explained


def pca_project(points, mean, components):
    return (points - mean) @ components


def path_from_residuals(residuals, normalize_for_plot=False):
    path = np.cumsum(residuals, axis=0)
    path = np.concatenate([np.zeros_like(path[:1]), path], axis=0)
    if normalize_for_plot:
        scale = np.linalg.norm(path[-1])
        if scale > 1e-8:
            path = path / scale
    return path


def record_paths(record, normalize_for_plot=False):
    target = path_from_residuals(as_float(record["aggregated_explicit_residuals"]), normalize_for_plot)
    paths = [("target", target)]
    if "multiview_implicit_residuals" in record:
        residuals = as_float(record["multiview_implicit_residuals"])
        for view_idx in range(residuals.shape[0]):
            paths.append((f"view{view_idx}", path_from_residuals(residuals[view_idx], normalize_for_plot)))
    else:
        paths.append(("view0", path_from_residuals(as_float(record["implicit_residuals"]), normalize_for_plot)))
    return paths


def path_signature(path):
    mean_state = path.mean(axis=0)
    first = path[0]
    last = path[-1]
    trend = last - first
    deltas = path[1:] - path[:-1] if path.shape[0] > 1 else np.zeros_like(path[:1])
    delta = deltas.mean(axis=0)
    parts = []
    for weight, item in zip((0.5, 0.5, 1.0, 1.0), (mean_state, last, trend, delta)):
        norm = np.linalg.norm(item)
        parts.append(weight * item / max(norm, 1e-8))
    sig = np.concatenate(parts)
    return sig / max(np.linalg.norm(sig), 1e-8)


def record_view_distance(record):
    if "multiview_implicit_residuals" in record:
        residual_views = as_float(record["multiview_implicit_residuals"])
    else:
        residual_views = as_float(record["implicit_residuals"])[None, ...]
    if residual_views.shape[0] <= 1:
        return 0.0
    sig = np.stack([path_signature(np.cumsum(residuals, axis=0)) for residuals in residual_views], axis=0)
    sim = sig @ sig.T
    mask = ~np.eye(sig.shape[0], dtype=bool)
    return float((1.0 - sim[mask]).mean())


def record_rollout_outcomes(record):
    if "multiview_acc" not in record:
        return None
    return as_float(record["multiview_acc"]).reshape(-1) > 0.5


def plot_paths(records, out_path, max_records=9, pca_fit_records=200, normalize_for_plot=True):
    selected = records[:max_records]
    pca_records = records[:pca_fit_records]
    all_points = []
    for record in pca_records:
        for _, path in record_paths(record, normalize_for_plot=normalize_for_plot):
            all_points.append(path)
    if not all_points:
        return
    stacked = np.concatenate(all_points, axis=0)
    mean, components, explained = pca_fit(stacked, n_components=3)

    cols = min(3, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    fig = plt.figure(figsize=(5.4 * cols, 4.6 * rows))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for plot_idx, record in enumerate(selected, start=1):
        ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")
        outcomes = record_rollout_outcomes(record)
        correct_labeled = False
        wrong_labeled = False
        view_local_idx = 0
        for path_idx, (name, path) in enumerate(record_paths(record, normalize_for_plot=normalize_for_plot)):
            proj = pca_project(path, mean, components)
            if name == "target":
                ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color="black", linestyle="--", linewidth=2.0, label="target")
                ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], color="black", s=7, alpha=0.65)
            else:
                is_correct = None
                if outcomes is not None and view_local_idx < len(outcomes):
                    is_correct = bool(outcomes[view_local_idx])
                    color = "#15803d" if is_correct else "#dc2626"
                    label = None
                    if is_correct and not correct_labeled:
                        label, correct_labeled = "correct rollout", True
                    elif not is_correct and not wrong_labeled:
                        label, wrong_labeled = "wrong rollout", True
                else:
                    color = colors[path_idx % len(colors)]
                    label = name
                ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color=color, linewidth=1.4, alpha=0.82, label=label)
                ax.scatter(
                    proj[:, 0],
                    proj[:, 1],
                    proj[:, 2],
                    color=color,
                    s=np.linspace(5, 16, len(proj)),
                    alpha=0.55,
                )
                if len(proj) > 1:
                    movement = np.diff(proj, axis=0)
                    ax.quiver(
                        proj[:-1, 0],
                        proj[:-1, 1],
                        proj[:-1, 2],
                        movement[:, 0],
                        movement[:, 1],
                        movement[:, 2],
                        color=color,
                        alpha=0.45,
                        linewidth=0.45,
                        arrow_length_ratio=0.16,
                        normalize=False,
                    )
                endpoint_marker = "o" if is_correct is not False else "X"
                ax.scatter(proj[-1:, 0], proj[-1:, 1], proj[-1:, 2], color=color, s=24, marker=endpoint_marker)
                outcome_label = "C" if is_correct is True else "W" if is_correct is False else "V"
                ax.text(
                    proj[-1, 0],
                    proj[-1, 1],
                    proj[-1, 2],
                    f"{outcome_label}{view_local_idx}",
                    fontsize=6,
                    color=color,
                )
                view_local_idx += 1
        if outcomes is not None:
            title = (
                f"q{record.get('idx')} rollout-acc={outcomes.mean():.2f} "
                f"viewD={record_view_distance(record):.3f}"
            )
        else:
            title = f"q{record.get('idx')} acc={record.get('acc', 0):.0f} viewD={record_view_distance(record):.3f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        if plot_idx == 1:
            ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    meta = {
        "n_records": len(records),
        "n_plotted": len(selected),
        "n_pca_fit_records": len(pca_records),
        "n_pca_fit_points": int(stacked.shape[0]),
        "pca_components_shape": list(components.shape),
        "explained_variance_ratio": [float(x) for x in explained],
        "mean_multiview_signature_distance": float(np.mean([record_view_distance(record) for record in records])),
        "has_multiview": any("multiview_implicit_residuals" in record for record in records),
        "path_plot_mode": "origin_start_final_norm" if normalize_for_plot else "raw_cumsum",
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def plot_rollout_similarity_heatmaps(records, out_path, max_records=9, mixed_only=False):
    eligible = [record for record in records if "multiview_acc" in record]
    if mixed_only:
        eligible = [
            record
            for record in eligible
            if (lambda outcomes: bool(outcomes.any() and (~outcomes).any()))(record_rollout_outcomes(record))
        ]
        eligible.sort(
            key=lambda record: (
                min(
                    int(record_rollout_outcomes(record).sum()),
                    int((~record_rollout_outcomes(record)).sum()),
                ),
                -int(record.get("idx", -1)),
            ),
            reverse=True,
        )
    eligible = eligible[:max_records]
    if not eligible:
        return
    cols = min(3, len(eligible))
    rows = int(np.ceil(len(eligible) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.0 * rows), squeeze=False)
    pair_values = {"correct_correct": [], "wrong_wrong": [], "correct_wrong": []}
    mixed_count = 0
    image = None
    for plot_idx, record in enumerate(eligible):
        ax = axes.flat[plot_idx]
        outcomes = record_rollout_outcomes(record)
        residual_views = as_float(record["multiview_implicit_residuals"])
        signatures = np.stack([path_signature(np.cumsum(item, axis=0)) for item in residual_views], axis=0)
        mode_ids = np.full(len(outcomes), -1, dtype=np.int64)
        if outcomes.any():
            _, positive_modes, _, _ = build_modes(signatures[outcomes], max_modes=3, merge_threshold=0.65)
            mode_ids[outcomes] = positive_modes
        original_indices = np.arange(len(outcomes))
        order = np.lexsort((original_indices, mode_ids, ~outcomes))
        outcomes = outcomes[order]
        mode_ids = mode_ids[order]
        original_indices = original_indices[order]
        similarity = signatures[order] @ signatures[order].T
        image = ax.imshow(similarity, vmin=-1, vmax=1, cmap="coolwarm")
        labels = [
            f"M{mode_ids[idx] + 1}:v{original_indices[idx]}" if value else f"W:v{original_indices[idx]}"
            for idx, value in enumerate(outcomes)
        ]
        ax.set_xticks(range(len(labels)), labels=labels, fontsize=7, rotation=45)
        ax.set_yticks(range(len(labels)), labels=labels, fontsize=7)
        n_modes = len(set(mode_ids[mode_ids >= 0].tolist()))
        ax.set_title(
            f"q{record.get('idx')} correct={int(outcomes.sum())}/{len(outcomes)} modes={n_modes}",
            fontsize=9,
        )

        off_diag = ~np.eye(len(outcomes), dtype=bool)
        cc = outcomes[:, None] & outcomes[None, :] & off_diag
        ww = (~outcomes[:, None]) & (~outcomes[None, :]) & off_diag
        cw = outcomes[:, None] != outcomes[None, :]
        if cc.any():
            pair_values["correct_correct"].extend(similarity[cc].tolist())
        if ww.any():
            pair_values["wrong_wrong"].extend(similarity[ww].tolist())
        if cw.any():
            pair_values["correct_wrong"].extend(similarity[cw].tolist())
            mixed_count += 1
    for plot_idx in range(len(eligible), rows * cols):
        axes.flat[plot_idx].axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    title_suffix = " (outcome-balanced mixed questions)" if mixed_only else " (fixed question order)"
    fig.suptitle(f"Same-question rollout signature similarity{title_suffix}", fontsize=12)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "n_records": len(eligible),
        "mixed_count": mixed_count,
        "selection_rule": (
            "most outcome-balanced mixed questions; no geometry used for selection"
            if mixed_only
            else "first records in fixed dataset order"
        ),
        "mean_pair_similarity": {
            key: float(np.mean(values)) if values else None for key, values in pair_values.items()
        },
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def plot_heatmap(records, out_path, max_records=200):
    sims = []
    for record in records[:max_records]:
        target = as_float(record["aggregated_explicit_residuals"])
        if "multiview_implicit_residuals" in record:
            residuals = as_float(record["multiview_implicit_residuals"])
            for view_idx in range(residuals.shape[0]):
                sims.append(cosine_matrix(residuals[view_idx], target))
        else:
            sims.append(cosine_matrix(as_float(record["implicit_residuals"]), target))
    if not sims:
        return
    mean_sim = np.mean(np.stack(sims, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(mean_sim, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("TRACE-BRIDGE residual-to-target cosine")
    ax.set_xlabel("explicit compressed target slot")
    ax.set_ylabel("implicit latent slot")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    diag = np.diag(mean_sim)
    offdiag = mean_sim[~np.eye(mean_sim.shape[0], dtype=bool)] if mean_sim.shape[0] == mean_sim.shape[1] else np.array([])
    meta = {
        "n_records": min(len(records), max_records),
        "n_matrices": len(sims),
        "mean_diag_cosine": float(diag.mean()) if diag.size else None,
        "mean_offdiag_cosine": float(offdiag.mean()) if offdiag.size else None,
        "diag_minus_offdiag": float(diag.mean() - offdiag.mean()) if diag.size and offdiag.size else None,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def resample_assignment_progress(assignment, bins=32):
    if assignment.shape[1] == bins:
        return assignment
    source_x = np.linspace(0.0, 1.0, assignment.shape[1])
    target_x = np.linspace(0.0, 1.0, bins)
    rows = [np.interp(target_x, source_x, row) for row in assignment]
    out = np.stack(rows, axis=0)
    return out / np.clip(out.sum(axis=-1, keepdims=True), 1e-8, None)


def plot_assignment_heatmap(records, out_path, max_records=200, progress_bins=32):
    assignments = []
    for record in records[:max_records]:
        if "assignment" in record:
            assignment = as_float(record["assignment"])
            assignment = assignment / np.clip(assignment.sum(axis=-1, keepdims=True), 1e-8, None)
            assignments.append(resample_assignment_progress(assignment, bins=progress_bins))
    if not assignments:
        return
    mean_assignment = np.mean(np.stack(assignments, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(mean_assignment, vmin=0, vmax=max(0.05, float(mean_assignment.max())), cmap="viridis")
    ax.set_title("TRACE-BRIDGE latent-to-CoT-step assignment")
    ax.set_xlabel("normalized CoT progress")
    ax.set_ylabel("latent slot")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    entropy = -(mean_assignment * np.log(np.clip(mean_assignment, 1e-8, None))).sum(axis=-1)
    entropy = entropy / max(np.log(max(mean_assignment.shape[1], 2)), 1e-8)
    meta = {
        "n_records": len(assignments),
        "progress_bins": progress_bins,
        "mean_assignment_entropy": float(entropy.mean()),
        "max_assignment_probability": float(mean_assignment.max()),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=9)
    parser.add_argument("--pca_fit_records", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records([Path(p) for p in args.records])
    plot_paths(
        records,
        out_dir / "trace_bridge_paths_3d.png",
        max_records=args.max_records,
        pca_fit_records=args.pca_fit_records,
    )
    plot_heatmap(records, out_dir / "trace_bridge_similarity_heatmap.png")
    plot_rollout_similarity_heatmaps(records, out_dir / "trace_bridge_rollout_similarity_heatmap.png")
    plot_rollout_similarity_heatmaps(
        records,
        out_dir / "trace_bridge_rollout_similarity_heatmap_mixed.png",
        mixed_only=True,
    )
    plot_assignment_heatmap(records, out_dir / "trace_bridge_assignment_heatmap.png")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
