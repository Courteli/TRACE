#!/usr/bin/env python3
"""Reserve one managed CUDA device until the supervising worker is stopped."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve-mib", type=int, default=16384)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1024 <= args.reserve_mib <= 20000:
        raise SystemExit("--reserve-mib must be between 1024 and 20000")
    if not 5 <= args.poll_seconds <= 300:
        raise SystemExit("--poll-seconds must be between 5 and 300")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("CUDA lease requires exactly one visible GPU")

    torch.cuda.set_device(0)
    lease = torch.empty(
        args.reserve_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda:0",
    )
    lease.zero_()
    torch.cuda.synchronize()

    stopped = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)

    payload = {
        "status": "READY",
        "pid": os.getpid(),
        "reserve_mib": args.reserve_mib,
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready_file.with_suffix(args.ready_file.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.ready_file)
    print(json.dumps(payload), flush=True)

    while not stopped:
        time.sleep(args.poll_seconds)

    del lease
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
