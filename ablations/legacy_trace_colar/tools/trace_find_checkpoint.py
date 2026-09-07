#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def monitor_score(path):
    match = re.search(r"monitor(-?\d+(?:\.\d+)?)", path.name)
    return float(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_root",
        default="logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl",
    )
    parser.add_argument("--run_contains", required=True)
    parser.add_argument("--prefer", choices=["best", "last"], default="best")
    args = parser.parse_args()

    root = Path(args.log_root)
    run_dirs = [p for p in root.glob("*") if p.is_dir() and args.run_contains in p.name]
    if not run_dirs:
        raise SystemExit(f"no run dir matching {args.run_contains!r} under {root}")

    scored = []
    lasts = []
    for run_dir in run_dirs:
        for ckpt in run_dir.glob("checkpoints/*.ckpt"):
            if ckpt.name == "last.ckpt":
                lasts.append(ckpt)
                continue
            score = monitor_score(ckpt)
            if score is not None:
                scored.append((score, ckpt.stat().st_mtime, ckpt))

    if args.prefer == "best" and scored:
        print(max(scored, key=lambda item: (item[0], item[1]))[2])
        return
    if lasts:
        print(max(lasts, key=lambda p: p.stat().st_mtime))
        return
    if scored:
        print(max(scored, key=lambda item: item[1])[2])
        return
    raise SystemExit(f"no checkpoint matching {args.run_contains!r} under {root}")


if __name__ == "__main__":
    main()
