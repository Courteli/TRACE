#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_record(value):
    label, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("--record must be LABEL=/absolute/record.pt")
    return label, Path(path)


def load_records(path, max_records):
    records = torch.load(path, map_location="cpu", weights_only=False)[:max_records]
    return {int(record["idx"]): record for record in records}


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.float().numpy()
    return np.asarray(value, dtype=np.float32)


def assignment_metrics(assignment):
    probs = assignment / np.clip(assignment.sum(axis=-1, keepdims=True), 1e-8, None)
    step_position = np.linspace(0.0, 1.0, assignment.shape[1], dtype=np.float32)
    centers = (probs * step_position[None, :]).sum(axis=-1)
    diffs = np.diff(centers)
    entropy = -(probs * np.log(np.clip(probs, 1e-8, None))).sum(axis=-1)
    entropy = entropy / max(np.log(max(assignment.shape[1], 2)), 1e-8)
    return {
        "centers": centers,
        "span": float(centers.max() - centers.min()),
        "inversion_fraction": float((diffs < -1e-4).mean()) if len(diffs) else 0.0,
        "entropy": float(entropy.mean()),
    }


def plot(record_sets, indices, out_path, selection_rule):
    rows = len(record_sets)
    cols = len(indices)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.8 * rows), squeeze=False)
    metadata = []
    image = None
    for row_idx, (label, records) in enumerate(record_sets.items()):
        for col_idx, idx in enumerate(indices):
            ax = axes[row_idx, col_idx]
            assignment = as_numpy(records[idx]["assignment"])
            probs = assignment / np.clip(assignment.sum(axis=-1, keepdims=True), 1e-8, None)
            metrics = assignment_metrics(assignment)
            image = ax.imshow(probs, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
            y = np.arange(assignment.shape[0])
            x = metrics["centers"] * max(assignment.shape[1] - 1, 1)
            ax.plot(x, y, color="white", marker="o", markersize=3.5, linewidth=1.2, label="step centroid")
            ax.set_xlabel("Human CoT step")
            ax.set_ylabel("Compressed latent slot")
            ax.set_title(
                f"{label} | q{idx}\nspan={metrics['span']:.2f}, inversion={100.0 * metrics['inversion_fraction']:.1f}%",
                fontsize=9,
            )
            metadata.append(
                {
                    "method": label,
                    "idx": idx,
                    "n_latent_slots": int(assignment.shape[0]),
                    "n_cot_steps": int(assignment.shape[1]),
                    "span": metrics["span"],
                    "inversion_fraction": metrics["inversion_fraction"],
                    "entropy": metrics["entropy"],
                    "step_centers": metrics["centers"].tolist(),
                }
            )
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left", fontsize=7)
    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.065, right=0.91, hspace=0.45, wspace=0.28)
    if image is not None:
        colorbar_ax = fig.add_axes([0.93, 0.32, 0.014, 0.36])
        fig.colorbar(image, cax=colorbar_ax, label="row-normalized assignment")
    fig.suptitle(
        "CoT-step weak-anchor assignment (evaluation diagnostic; CoT is absent at inference)",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "indices": indices,
        "selection_rule": selection_rule,
        "interpretation_boundary": (
            "The assignment is computed against human CoT only for evaluation. "
            "It diagnoses Stage1 compression anchors and is not an inference input."
        ),
        "panels": metadata,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument(
        "--selection_rule",
        default="predeclared indices; selection does not use assignment or latent geometry",
    )
    args = parser.parse_args()

    record_sets = {label: load_records(path, args.max_records) for label, path in args.record}
    common = set.intersection(*(set(records) for records in record_sets.values()))
    missing = [idx for idx in args.indices if idx not in common]
    if missing:
        raise RuntimeError(f"Requested indices are not present for every method: {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = plot(
        record_sets,
        args.indices,
        args.out_dir / "assignment_comparison.png",
        args.selection_rule,
    )
    (args.out_dir / "assignment_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
