#!/usr/bin/env python3
"""Select and validate the best monitored checkpoint from one native stage."""

import argparse
import re
from pathlib import Path

import torch


MONITOR_RE = re.compile(r"monitor(-?\d+(?:\.\d+)?)")


def model_target(checkpoint):
    all_config = checkpoint.get("hyper_parameters", {}).get("all_config")
    try:
        return all_config.model.target
    except AttributeError:
        return all_config["model"]["target"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_root", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--do-trace-rl", choices=("true", "false"))
    args = parser.parse_args()

    root = args.stage_root.expanduser().resolve(strict=True)
    ranked = []
    for path in root.rglob("*.ckpt"):
        match = MONITOR_RE.search(path.name)
        if match:
            ranked.append((float(match.group(1)), path.resolve()))
    if not ranked:
        raise SystemExit(f"no monitored checkpoint found under {root}")
    _, best = max(ranked, key=lambda item: (item[0], str(item[1])))
    best.relative_to(root)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    target = model_target(checkpoint)
    if target != args.target:
        raise SystemExit(f"expected {args.target}, found {target} in {best}")
    if args.do_trace_rl is not None:
        expected = args.do_trace_rl == "true"
        config = checkpoint["hyper_parameters"]["all_config"]
        actual = bool(config.model.model_kwargs.get("do_trace_rl", False))
        if actual != expected:
            raise SystemExit(
                f"expected do_trace_rl={expected}, found {actual} in {best}"
            )
    print(best)


if __name__ == "__main__":
    main()
