#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def checkpoint_delta(base_path, trained_path):
    base = load_state(base_path)
    trained = load_state(trained_path)
    delta_sq = 0.0
    base_sq = 0.0
    changed = 0
    common = 0
    max_abs = 0.0
    for key, base_value in base.items():
        trained_value = trained.get(key)
        if trained_value is None or not torch.is_tensor(base_value) or not torch.is_floating_point(base_value):
            continue
        delta = trained_value.float() - base_value.float()
        delta_norm = float(delta.norm())
        base_norm = float(base_value.float().norm())
        delta_sq += delta_norm * delta_norm
        base_sq += base_norm * base_norm
        max_abs = max(max_abs, float(delta.abs().max()))
        changed += int(delta_norm > 0)
        common += 1
    return {
        "common_float_tensors": common,
        "changed_float_tensors": changed,
        "relative_l2_delta": math.sqrt(delta_sq) / max(math.sqrt(base_sq), 1e-30),
        "max_abs_delta": max_abs,
    }


def scalar_values(event_dir, tag):
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [event.value for event in accumulator.Scalars(tag)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--trained_ckpt", required=True)
    parser.add_argument("--event_dir", required=True)
    parser.add_argument("--min_relative_delta", type=float, default=1e-12)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    delta = checkpoint_delta(Path(args.base_ckpt), Path(args.trained_ckpt))
    effective_lr = scalar_values(Path(args.event_dir), "train/effective_lr")
    logged_lr = scalar_values(Path(args.event_dir), "lr-AdamW")
    did_step = scalar_values(Path(args.event_dir), "train/optimizer_did_step")
    max_lr = max(effective_lr + logged_lr, default=0.0)
    result = {
        "checkpoint_delta": delta,
        "max_logged_lr": max_lr,
        "optimizer_step_fraction": float(sum(did_step) / len(did_step)) if did_step else None,
        "passed": bool(
            max_lr > 0
            and delta["changed_float_tensors"] > 0
            and delta["relative_l2_delta"] > args.min_relative_delta
            and (not did_step or max(did_step) > 0)
        ),
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
