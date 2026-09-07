#!/usr/bin/env python3
"""Select the newest exact-recovery checkpoint for one native stage."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_root", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--do-trace-rl", choices=("true", "false"), required=True)
    args = parser.parse_args()

    root = args.stage_root.expanduser().resolve(strict=True)
    candidates = sorted(
        root.rglob("last.ckpt"), key=lambda path: (path.stat().st_mtime_ns, str(path))
    )
    if not candidates:
        raise SystemExit(f"no last.ckpt found under {root}")
    path = candidates[-1].resolve()
    path.relative_to(root)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]["all_config"]
    target = str(config.model.target)
    do_rl = bool(config.model.model_kwargs.get("do_trace_rl", False))
    expected_rl = args.do_trace_rl == "true"
    if target != args.target or do_rl != expected_rl:
        raise SystemExit(
            f"checkpoint mismatch: target={target}, do_trace_rl={do_rl}: {path}"
        )
    if not checkpoint.get("optimizer_states"):
        raise SystemExit(f"weights-only checkpoint is not recoverable: {path}")
    print(path)


if __name__ == "__main__":
    main()
