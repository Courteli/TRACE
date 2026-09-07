#!/usr/bin/env python3
"""Four-rank real-objective memory/gradient gate for TRACE-VB Stage 1."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate


ROOT = Path(os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))

from src.utils.utils import instantiate_from_config  # noqa: E402
from tools.trace_vb_stage1_smoke import build_config, load_stage0, select_stress_indices  # noqa: E402


MIB = 1024**2
MIN_HEADROOM_MIB = float(
    os.environ.get("TRACE_VB_DDP_MIN_HEADROOM_MIB", "4096")
)
if MIN_HEADROOM_MIB < 3800.0:
    raise RuntimeError(
        "TRACE_VB_DDP_MIN_HEADROOM_MIB may not be lower than 3800 MiB"
    )


def snapshot(device: torch.device, phase: str, step: int | None = None) -> dict:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    row = {
        "phase": phase,
        "free_mib": float(free / MIB),
        "total_mib": float(total / MIB),
        "reserved_mib": float(reserved / MIB),
        "external_or_nonallocator_mib": float(max(0, total - free - reserved) / MIB),
    }
    if step is not None:
        row["optimizer_step"] = int(step)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.optimizer_steps < 2:
        raise SystemExit("DDP memory smoke requires at least two optimizer steps")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("formal TRACE-VB DDP smoke requires exactly four ranks")

    torch.manual_seed(20260809 + rank)
    config = build_config()
    data = instantiate_from_config(config.data_module, extra_kwargs={"all_config": config})
    data.setup("fit")
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    load_stage0(model, args.stage0_checkpoint)
    if getattr(model, "trajectory_posterior", None) is not None:
        raise RuntimeError("TRACE-VB DDP smoke found a forbidden posterior")
    stress_cases = select_stress_indices(model, data.train_set, world_size)
    case = stress_cases[rank]
    row = data.train_set[case["dataset_index"]]
    batch = default_collate([row])

    model = model.to(device).train()
    wrapped = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.model.training_kwargs.optimizer.lr),
        weight_decay=float(config.model.training_kwargs.optimizer.weight_decay),
        foreach=False,
    )
    dist.barrier()
    snapshots = [snapshot(device, "post_ddp_initialization")]
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    missing = []
    for step in range(args.optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = wrapped(batch)
        snapshots.append(snapshot(device, "post_forward", step))
        loss = output["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"rank {rank} produced non-finite Stage-1 loss")
        for key in (
            "trace_vb_mean_path_answer_only",
            "trace_vb_question_only_paths",
            "trace_vb_path_only_commit_readout",
            "trace_vb_stochastic_path_count",
        ):
            if key not in output or int(output[key].item()) != 1:
                raise RuntimeError(f"rank {rank} violated {key}")
        if int(output.get("trace_answer_question_access", loss.new_ones(())).item()) != 0:
            raise RuntimeError(f"rank {rank} answer readout accessed the question")
        loss.backward()
        snapshots.append(snapshot(device, "post_backward", step))
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        torch.nn.utils.clip_grad_norm_(trainable, 0.3)
        optimizer.step()
        snapshots.append(snapshot(device, "post_optimizer", step))
        if hasattr(model, "on_train_batch_end"):
            model.on_train_batch_end(None, batch, step)
        snapshots.append(snapshot(device, "post_cleanup", step))
        losses.append(float(loss.detach().float().cpu()))

    total_mib = snapshots[0]["total_mib"]
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / MIB)
    max_external = max(item["external_or_nonallocator_mib"] for item in snapshots)
    effective_headroom = total_mib - peak_reserved - max_external
    minimum_free = min(item["free_mib"] for item in snapshots)
    failures = []
    if missing:
        failures.append("missing_trainable_gradients")
    if effective_headroom < MIN_HEADROOM_MIB:
        failures.append(f"effective_headroom={effective_headroom:.1f}<{MIN_HEADROOM_MIB:.1f}")
    if minimum_free < MIN_HEADROOM_MIB:
        failures.append(f"minimum_free={minimum_free:.1f}<{MIN_HEADROOM_MIB:.1f}")
    local = {
        "rank": rank,
        "local_rank": local_rank,
        **case,
        "source_id": int(row["source_id"]),
        "losses": losses,
        "missing_trainable_gradients": missing,
        "peak_reserved_mib": peak_reserved,
        "max_external_or_nonallocator_mib": max_external,
        "effective_peak_headroom_mib": effective_headroom,
        "minimum_observed_free_mib": minimum_free,
        "snapshots": snapshots,
        "failures": failures,
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    passed = all(not item["failures"] for item in gathered)
    if rank == 0:
        report = {
            "status": "PASS" if passed else "FAIL",
            "scope": "real_four_rank_trace_vb_stage1_smoke_not_an_experiment_result",
            "world_size": world_size,
            "optimizer_steps": args.optimizer_steps,
            "role_schema": ["PLAN", "SOLVE1", "SOLVE2", "SOLVE3", "SOLVE4", "SOLVE5", "REFINE", "COMMIT"],
            "posterior_paths": 0,
            "minimum_effective_and_observed_headroom_mib": MIN_HEADROOM_MIB,
            "ranks": gathered,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()
    if not passed:
        raise RuntimeError("TRACE-VB Stage-1 four-rank memory/gradient gate failed")


if __name__ == "__main__":
    main()
