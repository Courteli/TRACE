#!/usr/bin/env python3
"""Expose exactly one physical GPU to each externally launched DDP rank."""

import os
import sys


def main() -> None:
    command = sys.argv[1:]
    if not command:
        raise SystemExit("Usage: isolated_gpu_ddp_entry.py <python-script> [args...]")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    visible = [device.strip() for device in visible if device.strip()]
    original_local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", len(visible)))
    if len(visible) != local_world_size:
        raise RuntimeError(
            "Expected one listed physical GPU per local rank: "
            f"visible={visible}, local_world_size={local_world_size}"
        )
    if not 0 <= original_local_rank < len(visible):
        raise RuntimeError(
            f"LOCAL_RANK={original_local_rank} is outside {visible}"
        )

    physical_gpu = visible[original_local_rank]
    os.environ["TRACE_TORCHRUN_LOCAL_RANK"] = str(original_local_rank)
    os.environ["TRACE_PHYSICAL_GPU"] = physical_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    # Every process now sees only its own card as cuda:0. Global RANK and
    # WORLD_SIZE remain owned by torchrun/TorchElastic.
    os.environ["LOCAL_RANK"] = "0"
    os.execv(sys.executable, [sys.executable, *command])


if __name__ == "__main__":
    main()
