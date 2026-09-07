#!/usr/bin/env python3
"""Real-model memory and gradient gate for formal TRACE Stage 1."""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate

ROOT = Path("/disk1/dingxukai/TRACE")
sys.path.insert(0, str(ROOT))

from src.utils.utils import instantiate_from_config
from tools.trace_policy_mechanism_smoke import (
    build_config,
    load_stage0,
    select_stress_indices,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=3)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    if args.optimizer_steps < 2:
        raise SystemExit("DDP stress requires at least two optimizer steps")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    initial_cuda_free_bytes, _ = torch.cuda.mem_get_info(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(
            "Formal TRACE DDP stress expected "
            f"{args.expected_world_size} ranks, got {world_size}"
        )
    if world_size * args.accumulation_steps != 4:
        raise RuntimeError(
            "Formal Stage-1 DDP stress must preserve effective batch 4"
        )

    torch.manual_seed(20260721 + rank)
    config = build_config()
    config.model.model_kwargs.trace_policy_config[
        "stage1_posterior_activation_offload"
    ] = False
    config.model.model_kwargs.trace_policy_config[
        "stage1_offload_cache_release_interval"
    ] = 0
    # The formal schedule intentionally warms this term from zero. The smoke
    # removes only that warmup so it can audit gradient connectivity for the
    # dynamic compressor before a long run starts.
    config.model.model_kwargs.readcot_config.dep_warmup_steps = 0
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
        world_size * args.accumulation_steps,
    )
    local_cases = [
        stress_cases[micro_index * world_size + rank]
        for micro_index in range(args.accumulation_steps)
    ]
    local_rows = [
        data_module.train_set[case["dataset_index"]]
        for case in local_cases
    ]
    local_batches = [default_collate([row]) for row in local_rows]

    model = model.cuda(local_rank)
    model.train()
    wrapped = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
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
    torch.cuda.reset_peak_memory_stats(local_rank)

    missing_after_final_backward = []
    losses = []
    deployment_compact_losses = []
    sampled_compact_losses = []
    for step_index in range(args.optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        micro_deployment_losses = []
        micro_sampled_losses = []
        for micro_index, batch in enumerate(local_batches):
            synchronize = micro_index + 1 == args.accumulation_steps
            sync_context = (
                torch.enable_grad() if synchronize else wrapped.no_sync()
            )
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = wrapped(batch)
                loss = output["total_loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError("DDP stress produced a non-finite loss")
                deployment_loss = output[
                    "trace_stage1_deployment_compact_loss"
                ]
                sampled_loss = output[
                    "trace_stage1_exchangeable_compact_loss"
                ]
                if not deployment_loss.requires_grad or not torch.isfinite(
                    deployment_loss
                ):
                    raise RuntimeError(
                        "DDP deployment risk is disconnected or non-finite"
                    )
                if (
                    not sampled_loss.requires_grad
                    or not torch.isfinite(sampled_loss)
                ):
                    raise RuntimeError(
                        "DDP sampled-path risk is disconnected or non-finite"
                    )
                (loss / args.accumulation_steps).backward()
            model.on_train_batch_end(
                None,
                batch,
                step_index * args.accumulation_steps + micro_index,
            )
            micro_losses.append(loss.detach().float())
            micro_deployment_losses.append(deployment_loss.detach().float())
            micro_sampled_losses.append(sampled_loss.detach().float())
        missing_after_final_backward = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.3)
        optimizer.step()
        losses.append(float(torch.stack(micro_losses).mean().cpu()))
        deployment_compact_losses.append(
            float(torch.stack(micro_deployment_losses).mean().cpu())
        )
        sampled_compact_losses.append(
            float(torch.stack(micro_sampled_losses).mean().cpu())
        )

    local_report = {
        "rank": rank,
        "local_rank": local_rank,
        "stress_cases": local_cases,
        "source_ids": [int(row["source_id"]) for row in local_rows],
        "losses": losses,
        "sampled_compact_losses": sampled_compact_losses,
        "deployment_compact_losses": deployment_compact_losses,
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
            torch.cuda.get_device_properties(local_rank).total_memory
            / (1024**2)
        ),
        "initial_cuda_free_mib": float(
            initial_cuda_free_bytes / (1024**2)
        ),
    }
    local_report["reserved_memory_headroom_mib"] = (
        local_report["cuda_device_total_memory_mib"]
        - local_report["peak_cuda_memory_reserved_mib"]
    )
    local_report["current_reserved_memory_headroom_mib"] = (
        local_report["cuda_device_total_memory_mib"]
        - local_report["current_cuda_memory_reserved_mib"]
    )
    local_report["external_adjusted_reserved_headroom_mib"] = (
        local_report["initial_cuda_free_mib"]
        - local_report["peak_cuda_memory_reserved_mib"]
    )
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_report)
    passed = all(
        not item["missing_trainable_gradients"]
        and item["external_adjusted_reserved_headroom_mib"] >= 1024.0
        for item in gathered
    )
    if rank == 0:
        report = {
            "status": "PASS" if passed else "FAIL",
            "scope": "real_four_rank_gpu_only_stage1_ddp_preflight_not_experiment_result",
            "cpu_activation_offload": False,
            "world_size": world_size,
            "accumulation_steps": args.accumulation_steps,
            "effective_batch_size": world_size * args.accumulation_steps,
            "optimizer_steps": args.optimizer_steps,
            "posterior_paths_per_question": int(
                config.model.model_kwargs.trace_policy_config
                .stage1_posterior_samples
            ),
            "privileged_sampled_paths_per_question": 0,
            "canonical_deployment_paths_per_question": 1,
            "canonical_deployment_path_is_rollout_member": False,
            "stage1_deployment_risk_mix": float(
                config.model.model_kwargs.trace_policy_config
                .stage1_deployment_risk_mix
            ),
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
            "Stage-1 DDP preflight lacks gradient coverage or 1 GiB "
            "external-adjusted GPU headroom"
        )


if __name__ == "__main__":
    main()
