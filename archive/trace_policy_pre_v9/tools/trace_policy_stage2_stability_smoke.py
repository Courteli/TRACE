#!/usr/bin/env python3
"""Real-checkpoint Stage-2 trust-region and dense-credit preflight."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf


ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(ROOT))

from src.utils.utils import instantiate_from_config  # noqa: E402


def build_config():
    trainer = OmegaConf.load(ROOT / "src/configs/trainer/default.yaml")
    model = OmegaConf.load(
        ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml"
    )
    dataset = OmegaConf.load(
        ROOT / "src/configs/datasets/gsm8k_aug_nl.yaml"
    )
    config = OmegaConf.merge(trainer, model, dataset)
    config.model.model_kwargs.do_trace_rl = True
    config.model.model_kwargs.trace_rl_config.rollout_micro_batch_size = 2
    config.model.model_kwargs.trace_rl_config.exp_batch_size = 2
    config.dataloader.batch_size = 1
    config.dataloader.val_batch_size = 1
    config.dataloader.num_workers = 0
    config.dataloader.persistent_workers = False
    config.args = OmegaConf.create(
        {"workspace_path": "/disk1/dingxukai", "no_log": True}
    )
    return config


def load_stage1(model, checkpoint_path: Path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
        raise RuntimeError("stability smoke requires a Stage-1 checkpoint")
    incompatible = model.load_state_dict(
        checkpoint["state_dict"],
        strict=False,
    )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "unexpected checkpoint keys: "
            + ", ".join(incompatible.unexpected_keys[:10])
        )
    if not model._cot_encoder_adapter_loaded:
        raise RuntimeError("Stage-1 checkpoint lacks the frozen CoT encoder")
    model._copy_path_adapter_to_answer_adapter()
    model._snapshot_stage1_policy()
    model._stage2_initialized = True
    del checkpoint
    gc.collect()


def answer_adapter_relative_drift(model) -> float:
    parameters = dict(model.llm.named_parameters())
    numerator = torch.zeros((), device=model.device)
    denominator = torch.zeros((), device=model.device)
    for name, value in parameters.items():
        if f".{model.answer_adapter_name}." not in name:
            continue
        reference_name = name.replace(
            f".{model.answer_adapter_name}.",
            f".{model.path_adapter_name}.",
        )
        reference = parameters.get(reference_name)
        if reference is None or reference.shape != value.shape:
            continue
        numerator += (
            value.detach().float() - reference.detach().float()
        ).square().sum()
        denominator += reference.detach().float().square().sum()
    return float(
        torch.sqrt(numerator / denominator.clamp_min(1e-30)).cpu()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if not args.stage1_checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.stage1_checkpoint}")
    if not torch.cuda.is_available():
        raise SystemExit("Stage-2 stability smoke requires one GPU")

    config = build_config()
    update_count = int(
        config.model.model_kwargs.trace_rl_config.policy_update_epochs
    )
    if update_count < 1:
        raise RuntimeError("stability smoke requires at least one policy update")
    data_module = instantiate_from_config(
        config.data_module,
        extra_kwargs={"all_config": config},
    )
    data_module.setup("fit")
    row = data_module.train_set[int(args.dataset_index)]
    model = instantiate_from_config(
        config.model,
        extra_kwargs={"all_config": config},
    )
    load_stage1(model, args.stage1_checkpoint)
    model = model.cuda().train()
    model.__dict__["manual_backward"] = lambda loss: loss.backward()
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=8.0e-7,
        weight_decay=0.01,
        foreach=False,
    )

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rollout = model.trace_policy_rollout(
            [row["question"]],
            [row["answer"]],
        )
        repeated_score_left = model._score_fixed_action_paths(
            [row["question"]],
            [row["answer"]],
            rollout["actions"][:1],
        )
        repeated_score_right = model._score_fixed_action_paths(
            [row["question"]],
            [row["answer"]],
            rollout["actions"][:1],
        )
    repeated_score_delta = float(
        (repeated_score_left - repeated_score_right).abs().max().cpu()
    )
    score_std_floor = float(
        config.model.model_kwargs.trace_rl_config.minimum_gold_score_std
    )
    initial_reference_delta = float(
        (
            rollout["old_answer_log_probs"]
            - rollout["stage1_answer_log_probs"]
        )
        .abs()
        .masked_select(
            rollout["answer_attention_mask"].to(torch.bool)
        )
        .max()
        .float()
        .cpu()
    )
    active_fraction = float(
        model._last_trace_metrics[
            "trace_policy/trajectory_advantage_active_fraction"
        ]
        .float()
        .cpu()
    )
    observed_answer_advantage_abs = float(
        rollout["answer_advantages"].abs().mean().float().cpu()
    )
    rollout["answer_advantages"] = torch.linspace(
        -1.0,
        1.0,
        steps=rollout["answer_advantages"].shape[0],
        device=model.device,
        dtype=rollout["answer_advantages"].dtype,
    ).unsqueeze(-1)

    update_reports = []
    for update_index in range(update_count):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            trajectory = model._trajectory_policy_update(rollout)
            answer = model._answer_policy_update(rollout)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite Stage-2 smoke gradient")
        optimizer.step()
        update_reports.append(
            {
                "update": update_index,
                "trajectory_policy_loss": float(
                    trajectory[0].detach().float().cpu()
                ),
                "trajectory_reference_kl": float(
                    trajectory[1].detach().float().cpu()
                ),
                "answer_policy_loss": float(
                    answer[0].detach().float().cpu()
                ),
                "answer_reference_kl": float(
                    answer[1].detach().float().cpu()
                ),
                "answer_ratio_deviation": float(
                    answer[2].detach().float().cpu()
                ),
                "answer_clip_fraction": float(
                    answer[3].detach().float().cpu()
                ),
                "answer_objective": float(
                    answer[4].detach().float().cpu()
                ),
                "grad_norm": float(grad_norm.detach().float().cpu()),
            }
        )

    drift = answer_adapter_relative_drift(model)
    passed = (
        initial_reference_delta <= 1e-4
        and active_fraction > 0.0
        and drift > 0.0
        and drift <= 0.01
        and score_std_floor >= max(1e-7, 10.0 * repeated_score_delta)
        and (
            update_count == 1
            or update_reports[-1]["answer_reference_kl"] > 0.0
        )
        and all(
            math.isfinite(np_value)
            for report in update_reports
            for np_value in report.values()
            if isinstance(np_value, float)
        )
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "scope": "stage2_stability_preflight_not_experiment_result",
        "project_root": str(ROOT),
        "checkpoint": str(args.stage1_checkpoint),
        "dataset_index": int(args.dataset_index),
        "group_size": int(rollout["rollout_group_size"]),
        "policy_update_epochs": update_count,
        "trace_steps": int(model.n_trace_steps),
        "initial_answer_reference_logprob_max_abs": (
            initial_reference_delta
        ),
        "trajectory_advantage_active_fraction": active_fraction,
        "repeat_score_max_abs_delta": repeated_score_delta,
        "registered_score_std_floor": score_std_floor,
        "observed_answer_advantage_abs": observed_answer_advantage_abs,
        "answer_smoke_advantage": (
            "deterministic_nonzero_connectivity_stress_only"
        ),
        "answer_adapter_relative_drift_after_updates": drift,
        "updates": update_reports,
        "peak_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated() / (1024**2)
        ),
        "peak_memory_reserved_mib": float(
            torch.cuda.max_memory_reserved() / (1024**2)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Stage-2 stability preflight failed")


if __name__ == "__main__":
    main()
