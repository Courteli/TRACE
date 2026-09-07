#!/usr/bin/env python
import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"]
    tensors = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
        }
        for key, value in state_dict.items()
    }
    del payload, state_dict
    hparams_path = path.parent.parent / "hparams.yaml"
    hparams = OmegaConf.load(hparams_path).all_config
    model_kwargs = hparams.model.model_kwargs
    return {
        "path": str(path),
        "sha256": sha256(path),
        "file_size_bytes": path.stat().st_size,
        "hparams_path": str(hparams_path),
        "model_target": str(hparams.model.target),
        "n_latents": int(model_kwargs.readcot_config.n_latents),
        "hybrid_max_new_tokens": int(model_kwargs.hybrid_generation_config.max_new_tokens),
        "state_dict_key_count": len(tensors),
        "state_dict_numel": int(sum(item["numel"] for item in tensors.values())),
        "tensors": tensors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--stage1_label", default="Stage1")
    parser.add_argument("--stage2_label", default="TRACE epoch7")
    parser.add_argument("--title", default="Stage1 to TRACE Epoch7 Checkpoint Parity")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stage1 = checkpoint_manifest(args.stage1)
    stage2 = checkpoint_manifest(args.stage2)
    stage1_tensors = stage1.pop("tensors")
    stage2_tensors = stage2.pop("tensors")
    stage1_keys = set(stage1_tensors)
    stage2_keys = set(stage2_tensors)
    common = sorted(stage1_keys & stage2_keys)
    shape_mismatches = [
        {
            "key": key,
            "stage1": stage1_tensors[key],
            "stage2": stage2_tensors[key],
        }
        for key in common
        if stage1_tensors[key] != stage2_tensors[key]
    ]
    comparison = {
        "same_model_target": stage1["model_target"] == stage2["model_target"],
        "same_n_latents": stage1["n_latents"] == stage2["n_latents"],
        "same_hybrid_max_new_tokens": (
            stage1["hybrid_max_new_tokens"] == stage2["hybrid_max_new_tokens"]
        ),
        "same_state_dict_keys": stage1_keys == stage2_keys,
        "same_state_dict_shapes_and_dtypes": not shape_mismatches,
        "added_keys": sorted(stage2_keys - stage1_keys),
        "removed_keys": sorted(stage1_keys - stage2_keys),
        "added_numel": int(sum(stage2_tensors[key]["numel"] for key in stage2_keys - stage1_keys)),
        "removed_numel": int(sum(stage1_tensors[key]["numel"] for key in stage1_keys - stage2_keys)),
        "shape_or_dtype_mismatches": shape_mismatches,
        "inference_architecture_parity": (
            stage1["model_target"] == stage2["model_target"]
            and stage1["n_latents"] == stage2["n_latents"]
            and stage1["hybrid_max_new_tokens"] == stage2["hybrid_max_new_tokens"]
            and stage1_keys == stage2_keys
            and not shape_mismatches
        ),
    }
    payload = {"stage1": stage1, "stage2": stage2, "comparison": comparison}
    (args.out_dir / "checkpoint_parity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {args.title}",
        "",
        f"- Inference architecture parity: **{comparison['inference_architecture_parity']}**.",
        f"- Model targets: `{stage1['model_target']}` ({args.stage1_label}) and `{stage2['model_target']}` ({args.stage2_label}).",
        f"- Latent slots: **{stage1['n_latents']}** ({args.stage1_label}) and **{stage2['n_latents']}** ({args.stage2_label}).",
        f"- Hybrid answer budget: **{stage1['hybrid_max_new_tokens']}** and **{stage2['hybrid_max_new_tokens']}** tokens.",
        f"- State-dict keys: **{stage1['state_dict_key_count']}** and **{stage2['state_dict_key_count']}**.",
        f"- Saved state-dict elements: **{stage1['state_dict_numel']:,}** and **{stage2['state_dict_numel']:,}**.",
        f"- Added inference keys: **{len(comparison['added_keys'])}** ({comparison['added_numel']:,} elements).",
        f"- Removed inference keys: **{len(comparison['removed_keys'])}** ({comparison['removed_numel']:,} elements).",
        f"- Shape/dtype mismatches: **{len(comparison['shape_or_dtype_mismatches'])}**.",
    ]
    if comparison["inference_architecture_parity"]:
        lines.extend(
            [
                "",
                f"{args.stage2_label} changes the values of existing model parameters; it does not add a larger inference network or a longer latent budget. Training-only reward, clustering, and gradient-guard computations are absent from normal inference.",
            ]
        )
    else:
        relative = 100.0 * comparison["added_numel"] / max(1, stage1["state_dict_numel"])
        added = ", ".join(f"`{key}`" for key in comparison["added_keys"]) or "none"
        lines.extend(
            [
                "",
                f"Added keys in {args.stage2_label}: {added}.",
                f"They add **{comparison['added_numel']:,}** saved elements (**{relative:.4f}%** relative to {args.stage1_label}) while the latent and answer-token budgets remain unchanged.",
            ]
        )
    (args.out_dir / "checkpoint_parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
