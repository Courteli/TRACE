#!/usr/bin/env python3
"""Audit TRACE Stage 1 and Final checkpoint architecture parity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-hparams", type=Path, required=True)
    parser.add_argument("--final-hparams", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def state_manifest(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    tensors = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
        }
        for key, value in state.items()
    }
    total = int(sum(item["numel"] for item in tensors.values()))
    del payload, state
    return {"tensors": tensors, "key_count": len(tensors), "numel": total}


def model_budget(path: Path) -> dict:
    config = OmegaConf.load(path)
    all_config = config.all_config if "all_config" in config else config
    kwargs = all_config.model.model_kwargs
    return {
        "n_latents": int(kwargs.readcot_config.n_latents),
        "max_answer_tokens": int(kwargs.hybrid_generation_config.max_new_tokens),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1 = state_manifest(args.stage1_checkpoint)
    final = state_manifest(args.final_checkpoint)
    stage1_budget = model_budget(args.stage1_hparams)
    final_budget = model_budget(args.final_hparams)
    stage1_tensors = stage1.pop("tensors")
    final_tensors = final.pop("tensors")
    stage1_keys = set(stage1_tensors)
    final_keys = set(final_tensors)
    common = stage1_keys & final_keys
    mismatches = [
        key
        for key in common
        if stage1_tensors[key]["shape"] != final_tensors[key]["shape"]
        or stage1_tensors[key]["dtype"] != final_tensors[key]["dtype"]
    ]
    added = sorted(final_keys - stage1_keys)
    removed = sorted(stage1_keys - final_keys)
    comparison = {
        "same_state_keys": stage1_keys == final_keys,
        "same_shapes_and_dtypes": not mismatches,
        "same_saved_elements": stage1["numel"] == final["numel"],
        "same_latent_slots": stage1_budget["n_latents"] == final_budget["n_latents"],
        "same_answer_budget": stage1_budget["max_answer_tokens"] == final_budget["max_answer_tokens"],
        "added_key_count": len(added),
        "removed_key_count": len(removed),
        "shape_or_dtype_mismatch_count": len(mismatches),
    }
    comparison["inference_architecture_parity"] = all(
        (
            comparison["same_state_keys"],
            comparison["same_shapes_and_dtypes"],
            comparison["same_saved_elements"],
            comparison["same_latent_slots"],
            comparison["same_answer_budget"],
        )
    )
    rows = [
        {"metric": "Latent slots", "TRACE Stage 1": stage1_budget["n_latents"], "TRACE Final": final_budget["n_latents"]},
        {"metric": "Maximum answer tokens", "TRACE Stage 1": stage1_budget["max_answer_tokens"], "TRACE Final": final_budget["max_answer_tokens"]},
        {"metric": "State tensor keys", "TRACE Stage 1": stage1["key_count"], "TRACE Final": final["key_count"]},
        {"metric": "Saved state elements", "TRACE Stage 1": stage1["numel"], "TRACE Final": final["numel"]},
        {"metric": "Added inference keys", "TRACE Stage 1": "--", "TRACE Final": len(added)},
        {"metric": "Removed inference keys", "TRACE Stage 1": "--", "TRACE Final": len(removed)},
        {"metric": "Shape/dtype mismatches", "TRACE Stage 1": "--", "TRACE Final": len(mismatches)},
    ]
    write_csv(args.output_dir / "source_data" / "inference_architecture_parity.csv", rows)
    md = [
        "| Metric | TRACE Stage 1 | TRACE Final |",
        "|---|---:|---:|",
    ]
    tex_rows = []
    for row in rows:
        before = f'{row["TRACE Stage 1"]:,}' if isinstance(row["TRACE Stage 1"], int) else row["TRACE Stage 1"]
        after = f'{row["TRACE Final"]:,}' if isinstance(row["TRACE Final"], int) else row["TRACE Final"]
        md.append(f'| {row["metric"]} | {before} | {after} |')
        tex_rows.append(f'{row["metric"]} & {before} & {after} ' + r"\\")
    md.extend(
        [
            "",
            f'Inference architecture parity: **{comparison["inference_architecture_parity"]}**.',
            "",
            "Outcome refinement changes existing parameter values; it does not add inference tensors, latent slots, or answer-token budget.",
        ]
    )
    (args.output_dir / "table_inference_architecture_parity.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    latex = "\n".join(
        [
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Metric & TRACE Stage 1 & TRACE Final \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (args.output_dir / "table_inference_architecture_parity.tex").write_text(
        latex + "\n", encoding="utf-8"
    )
    payload = {
        "stage1": {**stage1, **stage1_budget},
        "final": {**final, **final_budget},
        "comparison": comparison,
        "private_source_paths_omitted": True,
    }
    (args.output_dir / "inference_architecture_parity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
