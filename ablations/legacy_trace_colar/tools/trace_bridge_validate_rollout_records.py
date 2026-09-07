#!/usr/bin/env python
"""Validate the outcome-labeled rollout records used by TRACE geometry figures."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_record(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--record must be LABEL=/absolute/path.pt")
    label, path = value.split("=", 1)
    return label, Path(path)


def as_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_record_set(label, path, expected_records, expected_views, require_rollout_text):
    records = torch.load(path, map_location="cpu", weights_only=False)
    if len(records) != expected_records:
        raise ValueError(f"{label}: expected {expected_records} records, found {len(records)} in {path}")

    ids = []
    mixed_count = 0
    total_correct = 0
    required = {
        "idx",
        "multiview_implicit_residuals",
        "multiview_acc",
    }
    if require_rollout_text:
        required.update({"multiview_output_strings", "multiview_output_lengths"})
    for record_idx, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"{label}: record {record_idx} is missing required rollout fields {missing}")
        idx = int(record["idx"])
        ids.append(idx)
        residuals = as_array(record["multiview_implicit_residuals"])
        outcomes = as_array(record["multiview_acc"]).reshape(-1)
        outputs = list(record["multiview_output_strings"]) if "multiview_output_strings" in record else None
        lengths = (
            as_array(record["multiview_output_lengths"]).reshape(-1)
            if "multiview_output_lengths" in record
            else None
        )
        if residuals.ndim != 3 or residuals.shape[0] != expected_views:
            raise ValueError(
                f"{label}: record idx={idx} has residual shape {tuple(residuals.shape)}, "
                f"expected ({expected_views}, n_latents, hidden_size)"
            )
        if len(outcomes) != expected_views:
            raise ValueError(
                f"{label}: record idx={idx} has {len(outcomes)} outcome labels, expected={expected_views}"
            )
        if outputs is not None and len(outputs) != expected_views:
            raise ValueError(f"{label}: record idx={idx} has {len(outputs)} rollout outputs, expected={expected_views}")
        if lengths is not None and len(lengths) != expected_views:
            raise ValueError(f"{label}: record idx={idx} has {len(lengths)} rollout lengths, expected={expected_views}")
        if not np.isfinite(residuals).all() or not np.isfinite(outcomes).all() or (
            lengths is not None and not np.isfinite(lengths).all()
        ):
            raise ValueError(f"{label}: record idx={idx} contains non-finite rollout data")
        if not np.all(np.isin(outcomes, (0.0, 1.0))):
            raise ValueError(f"{label}: record idx={idx} has non-binary rollout outcomes")
        n_correct = int((outcomes > 0.5).sum())
        total_correct += n_correct
        mixed_count += int(0 < n_correct < expected_views)

    if len(set(ids)) != len(ids):
        raise ValueError(f"{label}: duplicate question IDs in rollout record")
    return {
        "path": str(path),
        "record_count": len(records),
        "unique_question_count": len(set(ids)),
        "views_per_question": expected_views,
        "mixed_question_count": mixed_count,
        "mixed_question_fraction": float(mixed_count / max(len(records), 1)),
        "rollout_accuracy": float(total_correct / max(len(records) * expected_views, 1)),
        "has_rollout_outputs": all("multiview_output_strings" in record for record in records),
        "has_rollout_lengths": all("multiview_output_lengths" in record for record in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", type=parse_record, required=True)
    parser.add_argument("--expected_records", type=int, default=200)
    parser.add_argument("--expected_views", type=int, default=8)
    parser.add_argument("--require_rollout_text", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = {
        "contract": {
            "expected_records": int(args.expected_records),
            "expected_views": int(args.expected_views),
            "require_rollout_text": bool(args.require_rollout_text),
            "required_fields": [
                "idx",
                "multiview_implicit_residuals",
                "multiview_acc",
                "multiview_output_strings",
                "multiview_output_lengths",
            ],
        },
        "records": {
            label: validate_record_set(
                label,
                path,
                args.expected_records,
                args.expected_views,
                args.require_rollout_text,
            )
            for label, path in args.record
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
