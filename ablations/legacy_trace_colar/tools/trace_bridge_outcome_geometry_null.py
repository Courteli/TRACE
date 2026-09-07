#!/usr/bin/env python
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trace_bridge_geometry_summary import (
    build_modes,
    normalize,
    pairwise_distance,
    prepare_group_signatures,
    record_signatures,
)
from trace_bridge_stage_geometry_delta import bootstrap_mean_ci


METRICS = (
    ("correct_within_mode_distance", "Correct compactness", -1.0),
    ("wrong_nearest_correct_mode_distance", "Wrong-to-correct distance", 1.0),
    ("cross_minus_correct_within_mode", "Correct/wrong margin", 1.0),
    ("wrong_minus_correct_within_mode", "Wrong dispersion gap", 1.0),
)


def parse_record(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--record must be LABEL=/absolute/trace_bridge_visual_test.pt")
    label, path = value.split("=", 1)
    return label, Path(path)


def as_bool_outcomes(record):
    value = record["multiview_acc"]
    if isinstance(value, torch.Tensor):
        value = value.float().numpy()
    return np.asarray(value).reshape(-1) > 0.5


def metric_bundle(signatures, outcomes, max_modes, merge_threshold):
    positive = signatures[outcomes]
    negative = signatures[~outcomes]
    if len(positive) == 0 or len(negative) == 0:
        return {}
    prototypes, assignments, _, _ = build_modes(
        positive,
        max_modes=max_modes,
        merge_threshold=merge_threshold,
    )
    within_mode = []
    for mode_idx in range(len(prototypes)):
        value = pairwise_distance(positive[assignments == mode_idx], offdiag=True)
        if value is not None:
            within_mode.append(value)
    compactness = float(np.mean(within_mode)) if within_mode else None
    wrong_nearest = float(
        (1.0 - (normalize(negative) @ normalize(prototypes).T).max(axis=1)).mean()
    )
    cross = pairwise_distance(positive, negative)
    wrong_within = pairwise_distance(negative, offdiag=True)
    return {
        "correct_within_mode_distance": compactness,
        "wrong_nearest_correct_mode_distance": wrong_nearest,
        "cross_minus_correct_within_mode": (
            None if compactness is None or cross is None else float(cross - compactness)
        ),
        "wrong_minus_correct_within_mode": (
            None if compactness is None or wrong_within is None else float(wrong_within - compactness)
        ),
    }


def canonical_assignment(assignments):
    mapping = {}
    output = []
    for value in assignments:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        output.append(mapping[value])
    return tuple(output)


def view_structure(signatures, outcomes, question_ids, max_modes, merge_threshold):
    n_views = signatures.shape[1]
    mean_cosine = np.einsum("nvd,nwd->vw", signatures, signatures) / len(signatures)
    coassign = np.zeros((n_views, n_views), dtype=np.float64)
    eligible = np.zeros((n_views, n_views), dtype=np.float64)
    templates = Counter()
    all_correct_count = 0

    for group, labels in zip(signatures, outcomes):
        correct_ids = np.flatnonzero(labels)
        if len(correct_ids) == 0:
            continue
        _, assignments, _, _ = build_modes(
            group[labels],
            max_modes=max_modes,
            merge_threshold=merge_threshold,
        )
        for left_pos, left_view in enumerate(correct_ids):
            for right_pos, right_view in enumerate(correct_ids):
                eligible[left_view, right_view] += 1.0
                coassign[left_view, right_view] += float(assignments[left_pos] == assignments[right_pos])
        if labels.all():
            all_correct_count += 1
            templates[canonical_assignment(assignments)] += 1

    coassign = np.divide(
        coassign,
        eligible,
        out=np.full_like(coassign, np.nan),
        where=eligible > 0,
    )

    fold_accuracy = []
    folds = np.asarray(question_ids, dtype=np.int64) % 2
    for fold in (0, 1):
        train = folds != fold
        test = folds == fold
        if not train.any() or not test.any():
            continue
        view_templates = normalize(signatures[train].mean(axis=0))
        scores = np.einsum("nvd,wd->nvw", normalize(signatures[test]), view_templates)
        predictions = scores.argmax(axis=-1)
        targets = np.arange(n_views, dtype=np.int64)[None, :]
        fold_accuracy.append(float((predictions == targets).mean()))

    dominant_template, dominant_count = (templates.most_common(1)[0] if templates else ((), 0))
    return {
        "n_views": n_views,
        "view_outcome_accuracy": outcomes.mean(axis=0).astype(float).tolist(),
        "view_outcome_accuracy_std": float(outcomes.mean(axis=0).std()),
        "mean_signature_cosine_by_view": mean_cosine.astype(float).tolist(),
        "correct_mode_coassignment_by_view": coassign.astype(float).tolist(),
        "crossfit_view_id_accuracy": float(np.mean(fold_accuracy)) if fold_accuracy else None,
        "chance_view_id_accuracy": 1.0 / n_views,
        "all_correct_question_count": all_correct_count,
        "dominant_all_correct_template": list(dominant_template),
        "dominant_all_correct_template_count": dominant_count,
        "dominant_all_correct_template_fraction": (
            float(dominant_count / all_correct_count) if all_correct_count else None
        ),
        "all_correct_templates": [
            {"template": list(template), "count": count}
            for template, count in templates.most_common()
        ],
    }


def summarize_rows(rows, rng, bootstrap_trials):
    summaries = []
    for key, label, direction in METRICS:
        selected = [row for row in rows if row["metric"] == key]
        observed = np.asarray([row["observed"] for row in selected], dtype=np.float64)
        null_mean = np.asarray([row["null_mean"] for row in selected], dtype=np.float64)
        excess = np.asarray([row["oriented_excess_over_null"] for row in selected], dtype=np.float64)
        if len(selected):
            low, high = bootstrap_mean_ci(excess, rng, bootstrap_trials)
        else:
            low = high = None
        summaries.append(
            {
                "metric": key,
                "label": label,
                "direction": "lower_is_better" if direction < 0 else "higher_is_better",
                "n": len(selected),
                "observed_mean": float(observed.mean()) if len(observed) else None,
                "within_question_null_mean": float(null_mean.mean()) if len(null_mean) else None,
                "oriented_excess_over_null": float(excess.mean()) if len(excess) else None,
                "oriented_excess_bootstrap_ci95_low": low,
                "oriented_excess_bootstrap_ci95_high": high,
                "positive_excess_fraction": float((excess > 0).mean()) if len(excess) else None,
                "pass_ci_above_zero": bool(low is not None and low > 0),
            }
        )
    return summaries


def analyze_method(label, path, max_records, max_modes, merge_threshold, null_trials, rng, bootstrap_trials):
    records = torch.load(path, map_location="cpu", weights_only=False)[:max_records]
    records = [record for record in records if "multiview_acc" in record]
    signatures = np.stack(
        [
            prepare_group_signatures(
                record_signatures(record),
                representation="stage2_centered",
                raw_mix=0.25,
            )
            for record in records
        ],
        axis=0,
    )
    outcomes = np.stack([as_bool_outcomes(record) for record in records], axis=0)
    question_ids = [int(record.get("idx", position)) for position, record in enumerate(records)]
    rows = []
    directions = {key: direction for key, _, direction in METRICS}
    for record_idx, (question_id, group, labels) in enumerate(zip(question_ids, signatures, outcomes)):
        observed = metric_bundle(group, labels, max_modes, merge_threshold)
        if not observed:
            continue
        null_values = {key: [] for key, _, _ in METRICS}
        question_rng = np.random.default_rng(1_000_003 * (record_idx + 1) + int(question_id))
        for _ in range(null_trials):
            permuted = metric_bundle(
                group,
                question_rng.permutation(labels),
                max_modes,
                merge_threshold,
            )
            for key, value in permuted.items():
                if value is not None and np.isfinite(value):
                    null_values[key].append(float(value))
        for key, value in observed.items():
            if value is None or not np.isfinite(value) or not null_values[key]:
                continue
            null_array = np.asarray(null_values[key], dtype=np.float64)
            null_mean = float(null_array.mean())
            direction = directions[key]
            rows.append(
                {
                    "method": label,
                    "idx": int(question_id),
                    "n_correct": int(labels.sum()),
                    "n_wrong": int((~labels).sum()),
                    "metric": key,
                    "observed": float(value),
                    "null_mean": null_mean,
                    "null_std": float(null_array.std(ddof=1)) if len(null_array) > 1 else 0.0,
                    "oriented_excess_over_null": float(direction * (float(value) - null_mean)),
                    "question_empirical_p_one_sided": float(
                        (1 + (direction * null_array >= direction * float(value)).sum())
                        / (1 + len(null_array))
                    ),
                }
            )
    return {
        "label": label,
        "record_path": str(path),
        "question_count": len(records),
        "path_count": int(outcomes.size),
        "null_trials_per_question": null_trials,
        "view_structure": view_structure(
            signatures,
            outcomes,
            question_ids,
            max_modes,
            merge_threshold,
        ),
        "outcome_geometry_null": summarize_rows(rows, rng, bootstrap_trials),
    }, rows


def plot_dashboard(payload, out_path):
    methods = payload["methods"]
    labels = list(methods)
    target_label = payload["target_label"]
    target = methods[target_label]
    colors = ("#475569", "#d97706", "#15803d", "#2563eb")
    color_map = {label: colors[idx % len(colors)] for idx, label in enumerate(labels)}
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.0))

    view_cosine = np.asarray(target["view_structure"]["mean_signature_cosine_by_view"])
    image = axes[0, 0].imshow(view_cosine, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[0, 0].set_xticks(np.arange(view_cosine.shape[0]), labels=[f"v{i}" for i in range(view_cosine.shape[0])])
    axes[0, 0].set_yticks(np.arange(view_cosine.shape[0]), labels=[f"v{i}" for i in range(view_cosine.shape[0])])
    axes[0, 0].set_title(f"{target_label}: mean signature cosine by fixed view ID")
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)

    coassign = np.asarray(target["view_structure"]["correct_mode_coassignment_by_view"])
    image = axes[0, 1].imshow(coassign, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0, 1].set_xticks(np.arange(coassign.shape[0]), labels=[f"v{i}" for i in range(coassign.shape[0])])
    axes[0, 1].set_yticks(np.arange(coassign.shape[0]), labels=[f"v{i}" for i in range(coassign.shape[0])])
    axes[0, 1].set_title(f"{target_label}: correct-path mode co-assignment")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.34
    view_accuracy = np.asarray([methods[label]["view_structure"]["crossfit_view_id_accuracy"] for label in labels])
    dominant = np.asarray([
        methods[label]["view_structure"]["dominant_all_correct_template_fraction"] or 0.0
        for label in labels
    ])
    axes[1, 0].bar(positions - width / 2, view_accuracy, width, color="#2563eb", label="cross-fit view-ID accuracy")
    axes[1, 0].bar(positions + width / 2, dominant, width, color="#d97706", label="dominant all-correct template")
    axes[1, 0].axhline(1.0 / target["view_structure"]["n_views"], color="#111827", linestyle="--", linewidth=1, label="view-ID chance")
    axes[1, 0].set_xticks(positions, labels=labels)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Fraction")
    axes[1, 0].set_title("Fixed-view predictability and template concentration")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.2)

    y = np.arange(len(METRICS), dtype=np.float64)
    offsets = np.linspace(-0.22, 0.22, max(1, len(labels)))
    for method_idx, label in enumerate(labels):
        rows = {row["metric"]: row for row in methods[label]["outcome_geometry_null"]}
        means = np.asarray([rows[key]["oriented_excess_over_null"] for key, _, _ in METRICS])
        low = np.asarray([rows[key]["oriented_excess_bootstrap_ci95_low"] for key, _, _ in METRICS])
        high = np.asarray([rows[key]["oriented_excess_bootstrap_ci95_high"] for key, _, _ in METRICS])
        axes[1, 1].errorbar(
            means,
            y + offsets[method_idx],
            xerr=np.vstack([means - low, high - means]),
            fmt="o",
            capsize=3,
            color=color_map[label],
            label=label,
        )
    axes[1, 1].axvline(0.0, color="#111827", linewidth=1)
    axes[1, 1].set_yticks(y, labels=[label for _, label, _ in METRICS])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Oriented excess over within-question label null (95% CI)")
    axes[1, 1].set_title("Does outcome geometry exceed fixed-view structure?")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(axis="x", alpha=0.2)

    fig.suptitle("TRACE outcome-geometry and fixed-view confound audit", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path, payload):
    lines = [
        "# TRACE Outcome-Geometry / Fixed-View Confound Audit",
        "",
        "Each question keeps its eight latent signatures fixed. The null independently permutes correct/wrong labels within that question while preserving the number correct. Positive oriented excess means the observed outcome geometry is better than the fixed-view null; a claim passes only when its question-bootstrap 95% CI is above zero.",
        "",
        "## Fixed-view structure",
        "",
        "| Method | Questions | Cross-fit view-ID accuracy | Chance | All-correct questions | Dominant mode template | Template fraction |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for label, method in payload["methods"].items():
        view = method["view_structure"]
        template = " ".join(str(value) for value in view["dominant_all_correct_template"])
        fraction = view["dominant_all_correct_template_fraction"]
        lines.append(
            f"| {label} | {method['question_count']} | {100.0 * view['crossfit_view_id_accuracy']:.1f}% | "
            f"{100.0 * view['chance_view_id_accuracy']:.1f}% | {view['all_correct_question_count']} | "
            f"`{template}` | {100.0 * fraction:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Outcome-label permutation null",
            "",
            "| Method | Metric | n | Observed | Label-null | Oriented excess (95% CI) | Pass |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for label, method in payload["methods"].items():
        for row in method["outcome_geometry_null"]:
            lines.append(
                f"| {label} | {row['label']} ({row['direction']}) | {row['n']} | "
                f"{row['observed_mean']:.4f} | {row['within_question_null_mean']:.4f} | "
                f"{row['oriented_excess_over_null']:+.4f} "
                f"[{row['oriented_excess_bootstrap_ci95_low']:+.4f}, {row['oriented_excess_bootstrap_ci95_high']:+.4f}] | "
                f"{'yes' if row['pass_ci_above_zero'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Absolute compactness, margin, or dispersion can be produced by fixed view-conditioned branches even when outcome labels are unrelated to those branches. Therefore the label-null excess, not the uncorrected absolute value, is the appropriate evidence for outcome-aware path geometry. This audit does not test answer accuracy or path existence; it tests whether path geometry specifically tracks correctness beyond fixed view identity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--target_label", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_records", type=int, default=200)
    parser.add_argument("--max_modes", type=int, default=3)
    parser.add_argument("--mode_merge_threshold", type=float, default=0.65)
    parser.add_argument("--null_trials", type=int, default=1024)
    parser.add_argument("--bootstrap_trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    record_paths = dict(args.record)
    if args.target_label not in record_paths:
        raise ValueError(f"target label {args.target_label!r} is not among {list(record_paths)}")
    rng = np.random.default_rng(args.seed)
    methods = {}
    all_rows = []
    for label, record_path in record_paths.items():
        method, rows = analyze_method(
            label,
            record_path,
            args.max_records,
            args.max_modes,
            args.mode_merge_threshold,
            args.null_trials,
            rng,
            args.bootstrap_trials,
        )
        methods[label] = method
        all_rows.extend(rows)
    payload = {
        "target_label": args.target_label,
        "signature_representation": "stage2_centered",
        "signature_raw_mix": 0.25,
        "max_modes": args.max_modes,
        "mode_merge_threshold": args.mode_merge_threshold,
        "null_trials_per_question": args.null_trials,
        "bootstrap_trials": args.bootstrap_trials,
        "seed": args.seed,
        "methods": methods,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "outcome_geometry_null.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if all_rows:
        with (args.out_dir / "outcome_geometry_null_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    write_markdown(args.out_dir / "outcome_geometry_null.md", payload)
    plot_dashboard(payload, args.out_dir / "outcome_geometry_null_dashboard.png")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
