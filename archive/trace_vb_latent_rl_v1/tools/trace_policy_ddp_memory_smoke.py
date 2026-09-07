#!/usr/bin/env python3
"""Four-rank role-semantic memory/gradient gate for formal TRACE Stage 1."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate

ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(ROOT))

from src.utils.utils import instantiate_from_config
from tools.trace_policy_mechanism_smoke import (
    build_config,
    load_stage0,
    select_stress_indices,
)


MIB = 1024**2

# The smoke workload is deliberately identical in sample/target construction to
# Stage 1, but it cannot cover every allocator high-water mark that a long
# Lightning epoch can encounter.  The failed 2026-08-08 run showed a ~3.2 GiB
# gap between this short smoke's allocator peak and the later training peak.
# Require 4 GiB of *device-wide* room after the smoke peak.  Crucially, this is
# not ``total - this_rank_reserved``: it also charges background processes,
# CUDA/NCCL contexts, and memory left on the device by the other ranks.
MIN_EFFECTIVE_PEAK_HEADROOM_MIB = 4096.0
MIN_OBSERVED_DEVICE_FREE_MIB = 4096.0


def _physical_device_token(local_rank):
    """Return the CUDA_VISIBLE_DEVICES token backing ``local_rank``."""
    visible = [
        token.strip()
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    ]
    if len(visible) > local_rank:
        return visible[local_rank]
    return f"logical:{local_rank}"


def capture_device_memory(device, phase, step_index=None):
    """Capture allocator and device-wide memory from the same CUDA device.

    ``torch.cuda.mem_get_info`` includes every process using the device, while
    the allocator counters include only this rank.  Their difference is a
    conservative charge for external processes plus non-allocator CUDA/NCCL
    contexts.  Failure to obtain either side is intentionally fatal: silently
    reverting to per-process accounting would recreate the original bug.
    """
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    system_used_bytes = total_bytes - free_bytes
    non_allocator_bytes = max(0, system_used_bytes - reserved_bytes)
    snapshot = {
        "phase": phase,
        "device_free_mib": float(free_bytes / MIB),
        "device_total_mib": float(total_bytes / MIB),
        "device_used_mib": float(system_used_bytes / MIB),
        "this_rank_allocated_mib": float(allocated_bytes / MIB),
        "this_rank_reserved_mib": float(reserved_bytes / MIB),
        "external_or_nonallocator_mib": float(non_allocator_bytes / MIB),
    }
    if step_index is not None:
        snapshot["optimizer_step"] = int(step_index)
    return snapshot


def add_device_wide_headroom(report, snapshots):
    """Attach fail-closed, device-wide peak/headroom fields to a rank report."""
    if not snapshots:
        raise RuntimeError("Memory gate requires at least one device snapshot")
    totals = {round(item["device_total_mib"], 3) for item in snapshots}
    if len(totals) != 1:
        raise RuntimeError(
            f"Inconsistent CUDA totals in memory snapshots: {totals}"
        )
    total_mib = snapshots[0]["device_total_mib"]
    peak_reserved_mib = report["peak_cuda_memory_reserved_mib"]
    max_external_mib = max(
        item["external_or_nonallocator_mib"] for item in snapshots
    )
    minimum_free_mib = min(item["device_free_mib"] for item in snapshots)
    effective_peak_headroom_mib = (
        total_mib - peak_reserved_mib - max_external_mib
    )
    report.update(
        {
            "memory_accounting": (
                "device_total_minus_this_rank_peak_reserved_minus_"
                "max_external_or_nonallocator"
            ),
            "memory_snapshots": snapshots,
            "max_external_or_nonallocator_memory_mib": float(max_external_mib),
            "minimum_observed_device_free_mib": float(minimum_free_mib),
            "effective_peak_device_headroom_mib": float(
                effective_peak_headroom_mib
            ),
            "required_effective_peak_headroom_mib": (
                MIN_EFFECTIVE_PEAK_HEADROOM_MIB
            ),
            "required_minimum_observed_device_free_mib": (
                MIN_OBSERVED_DEVICE_FREE_MIB
            ),
        }
    )
    return report


def memory_gate_failures(report):
    """Return explicit reasons; missing measurements fail rather than pass."""
    failures = []
    required = {
        "effective_peak_device_headroom_mib": (
            MIN_EFFECTIVE_PEAK_HEADROOM_MIB
        ),
        "minimum_observed_device_free_mib": MIN_OBSERVED_DEVICE_FREE_MIB,
    }
    for key, threshold in required.items():
        value = report.get(key)
        if value is None:
            failures.append(f"missing:{key}")
        elif float(value) < threshold:
            failures.append(f"{key}={float(value):.1f}<{threshold:.1f}")
    if report.get("missing_trainable_gradients"):
        failures.append("missing_trainable_gradients")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=3)
    args = parser.parse_args()
    if args.optimizer_steps < 2:
        raise SystemExit("DDP stress requires at least two optimizer steps")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("Formal TRACE DDP stress requires exactly four ranks")

    torch.manual_seed(20260721 + rank)
    config = build_config()
    config.model.model_kwargs.trace_policy_config[
        "stage1_offload_cache_release_interval"
    ] = 1
    data_module = instantiate_from_config(
        config.data_module,
        extra_kwargs={"all_config": config},
    )
    data_module.setup("fit")
    model = instantiate_from_config(
        config.model,
        extra_kwargs={"all_config": config},
    )
    load_stage0(model, args.stage0_checkpoint)
    stress_cases = select_stress_indices(
        model,
        data_module.train_set,
        world_size,
    )
    case = stress_cases[rank]
    row = data_module.train_set[case["dataset_index"]]
    batch = default_collate([row])

    model = model.cuda(local_rank)
    model.train()
    wrapped = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.model.training_kwargs.optimizer.lr),
        weight_decay=float(
            config.model.training_kwargs.optimizer.weight_decay
        ),
        foreach=False,
    )
    # Wait until all four ranks have created their DDP/NCCL contexts.  In this
    # codebase ranks 1--3 also leave small contexts on logical device 0 during
    # model construction, so sampling before this barrier undercounts the most
    # constrained physical GPU.
    dist.barrier()
    memory_snapshots = [
        capture_device_memory(local_rank, "post_ddp_initialization")
    ]
    torch.cuda.reset_peak_memory_stats(local_rank)

    missing_after_final_backward = []
    losses = []
    for step_index in range(args.optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = wrapped(batch)
        memory_snapshots.append(
            capture_device_memory(local_rank, "post_forward", step_index)
        )
        loss = output["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("DDP stress produced a non-finite loss")
        for contract_key in (
            "trace_stage1_role_contract",
            "trace_stage1_commit_is_deterministic",
            "trace_stage1_commit_only_readout",
        ):
            if int(output[contract_key].item()) != 1:
                raise RuntimeError(
                    f"DDP stress violated {contract_key} on rank {rank}"
                )
        loss.backward()
        memory_snapshots.append(
            capture_device_memory(local_rank, "post_backward", step_index)
        )
        missing_after_final_backward = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.3)
        optimizer.step()
        memory_snapshots.append(
            capture_device_memory(local_rank, "post_optimizer", step_index)
        )
        model.on_train_batch_end(None, batch, step_index)
        memory_snapshots.append(
            capture_device_memory(local_rank, "post_batch_cleanup", step_index)
        )
        losses.append(float(loss.detach().float().cpu()))

    local_report = {
        "rank": rank,
        "local_rank": local_rank,
        "physical_device_token": _physical_device_token(local_rank),
        **case,
        "source_id": int(row["source_id"]),
        "losses": losses,
        "missing_trainable_gradients": missing_after_final_backward,
        "peak_cuda_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated(local_rank) / (1024**2)
        ),
        "peak_cuda_memory_reserved_mib": float(
            torch.cuda.max_memory_reserved(local_rank) / (1024**2)
        ),
        "current_cuda_memory_reserved_mib": float(
            torch.cuda.memory_reserved(local_rank) / (1024**2)
        ),
        "cuda_device_total_memory_mib": float(
            torch.cuda.get_device_properties(local_rank).total_memory / MIB
        ),
    }
    add_device_wide_headroom(local_report, memory_snapshots)
    local_report["memory_gate_failures"] = memory_gate_failures(local_report)
    # Kept as an audit diagnostic only.  This per-rank number was the old,
    # unsafe gate because it ignores every other consumer on the device.
    local_report["allocator_only_peak_headroom_mib_not_used_for_gate"] = (
        local_report["cuda_device_total_memory_mib"]
        - local_report["peak_cuda_memory_reserved_mib"]
    )
    local_report["allocator_only_current_headroom_mib_not_used_for_gate"] = (
        local_report["cuda_device_total_memory_mib"]
        - local_report["current_cuda_memory_reserved_mib"]
    )
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_report)
    passed = all(not item["memory_gate_failures"] for item in gathered)
    if rank == 0:
        report = {
            "status": "PASS" if passed else "FAIL",
            "scope": "real_four_rank_stage1_ddp_preflight_not_experiment_result",
            "world_size": world_size,
            "optimizer_steps": args.optimizer_steps,
            "role_schema": [
                "PLAN",
                "SOLVE1",
                "SOLVE2",
                "SOLVE3",
                "SOLVE4",
                "SOLVE5",
                "CHECK",
                "COMMIT",
            ],
            "posterior_paths_per_question": int(
                config.model.model_kwargs.trace_policy_config
                .stage1_posterior_samples
            ),
            "memory_gate": {
                "fail_closed": True,
                "accounts_for_external_processes": True,
                "accounts_for_other_rank_cuda_contexts": True,
                "minimum_effective_peak_headroom_mib": (
                    MIN_EFFECTIVE_PEAK_HEADROOM_MIB
                ),
                "minimum_observed_device_free_mib": (
                    MIN_OBSERVED_DEVICE_FREE_MIB
                ),
                "does_not_reduce_samples_or_targets": True,
            },
            "ranks": gathered,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()
    if not passed:
        raise RuntimeError(
            "Stage-1 DDP preflight lacks gradient coverage or 4 GiB "
            "device-wide effective headroom after external/context memory"
        )


if __name__ == "__main__":
    main()
