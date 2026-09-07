#!/usr/bin/env python3
"""Verify that the final model is a fresh Stage0 -> Stage1 -> Stage2 lineage."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


STAGE0_TARGET = "src.models.cot.LitCot"
NATIVE_TARGET = "src.models.trace_role_native.LitTRACERoleNative"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path, expected_target, expected_rl, root):
    path = path.expanduser().resolve(strict=True)
    path.relative_to(root)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]["all_config"]
    target = config.model.target
    do_rl = bool(config.model.model_kwargs.get("do_trace_rl", False))
    if target != expected_target or do_rl != expected_rl:
        raise RuntimeError(
            f"invalid lineage node {path}: target={target}, do_trace_rl={do_rl}"
        )
    keys = set(checkpoint["state_dict"])
    has_role_policy = any(key.startswith("role_policy.") for key in keys)
    if has_role_policy != (expected_target == NATIVE_TARGET):
        raise RuntimeError(f"role-policy state mismatch in {path}")
    if any(key.startswith("stage1_role_reference.") for key in keys):
        raise RuntimeError(f"frozen Stage-1 reference must not be serialized: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "model_target": target,
        "do_trace_rl": do_rl,
        "state_tensor_count": len(keys),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve(strict=True)
    manifest = {
        "lineage": "base-Qwen3-4B -> native-run Stage0 CoT -> native-run Stage1 roles -> native-run Stage2 joint latent/answer RL",
        "historical_checkpoint_loaded": False,
        "stage0": inspect(args.stage0, STAGE0_TARGET, False, root),
        "stage1": inspect(args.stage1, NATIVE_TARGET, False, root),
        "stage2": inspect(args.stage2, NATIVE_TARGET, True, root),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
