#!/usr/bin/env python3
"""Real-checkpoint Stage-2 role-credit and trust-region preflight."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate


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
        {"workspace_path": str(ROOT), "no_log": True}
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
    model._snapshot_stage1_policy()
    model._stage2_initialized = True
    del checkpoint
    gc.collect()


def policy_relative_drift(model) -> float:
    current = dict(model.trajectory_policy.named_parameters())
    reference = dict(model.stage1_policy_reference.named_parameters())
    numerator = torch.zeros((), device=model.device)
    denominator = torch.zeros((), device=model.device)
    for name, value in current.items():
        reference_value = reference[name]
        numerator += (
            value.detach().float() - reference_value.detach().float()
        ).square().sum()
        denominator += reference_value.detach().float().square().sum()
    return float(
        torch.sqrt(numerator / denominator.clamp_min(1e-30)).cpu()
    )


def decoder_fingerprint(model) -> tuple:
    """Compact exact fingerprint for all frozen answer-path parameters."""
    marker = f".{model.path_adapter_name}."
    count = 0
    total = torch.zeros((), device=model.device, dtype=torch.float64)
    absolute = torch.zeros((), device=model.device, dtype=torch.float64)
    squared = torch.zeros((), device=model.device, dtype=torch.float64)
    for name, parameter in model.llm.named_parameters():
        if marker not in name:
            continue
        value = parameter.detach().double()
        count += value.numel()
        total += value.sum()
        absolute += value.abs().sum()
        squared += value.square().sum()
    if count == 0:
        raise RuntimeError("no answer/path adapter parameters were found")
    return (
        int(count),
        float(total.cpu()),
        float(absolute.cpu()),
        float(squared.cpu()),
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
    formal_update_count = int(
        config.model.model_kwargs.trace_rl_config.policy_update_epochs
    )
    if formal_update_count < 1:
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
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or any(
        not name.startswith("trajectory_policy.")
        for name in trainable_names
    ):
        raise RuntimeError(
            "Stage 2 must train only trajectory_policy parameters; got "
            + ", ".join(trainable_names[:10])
        )
    if any("mean_heads.commit" in name for name in trainable_names):
        raise RuntimeError("deterministic COMMIT head must remain frozen")
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
    batch = default_collate([row])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gold_cots = model._decode_single_gold_cots(batch)
        explicit_features, _ = model._collect_single_cot_features(
            [row["question"]],
            gold_cots,
        )
        raw_role_targets = model._build_role_semantic_targets(
            explicit_features
        )
        role_targets = {
            key: value
            for key, value in raw_role_targets.items()
            if isinstance(value, torch.Tensor)
        }
        rollout = model.trace_policy_rollout(
            [row["question"]],
            [row["answer"]],
            role_targets,
        )
    expected_shape = (
        int(config.model.model_kwargs.trace_rl_config.group_size),
        8,
    )
    for key in (
        "semantic_step_rewards",
        "semantic_reward_mask",
        "trajectory_returns",
        "trajectory_advantages",
    ):
        if tuple(rollout[key].shape) != expected_shape:
            raise RuntimeError(
                f"{key} has shape {tuple(rollout[key].shape)}, "
                f"expected {expected_shape}"
            )
    commit_reward_zero = bool(
        torch.equal(
            rollout["semantic_step_rewards"][:, 7],
            torch.zeros_like(rollout["semantic_step_rewards"][:, 7]),
        )
    )
    commit_mask_zero = bool(
        torch.equal(
            rollout["semantic_reward_mask"][:, 7],
            torch.zeros_like(rollout["semantic_reward_mask"][:, 7]),
        )
    )
    commit_advantage_zero = bool(
        torch.equal(
            rollout["trajectory_advantages"][:, 7],
            torch.zeros_like(rollout["trajectory_advantages"][:, 7]),
        )
    )
    role_reward_abs = float(
        (
            rollout["semantic_step_rewards"]
            * rollout["semantic_reward_mask"]
        ).abs().sum().float().cpu()
    )
    active_fraction = float(
        model._last_trace_metrics[
            "trace_policy/trajectory_advantage_active_fraction"
        ]
        .float()
        .cpu()
    )
    initial_policy_drift = policy_relative_drift(model)
    decoder_before = decoder_fingerprint(model)

    update_reports = []
    stress_update_count = max(2, formal_update_count)
    for update_index in range(stress_update_count):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            trajectory = model._trajectory_policy_update(rollout)
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
                "action_ratio_deviation": float(
                    trajectory[2].detach().float().cpu()
                ),
                "action_clip_fraction": float(
                    trajectory[3].detach().float().cpu()
                ),
                "role_entropy": float(
                    trajectory[4].detach().float().cpu()
                ),
                "grad_norm": float(grad_norm.detach().float().cpu()),
            }
        )

    drift = policy_relative_drift(model)
    decoder_after = decoder_fingerprint(model)
    frozen_decoder_unchanged = decoder_before == decoder_after
    passed = (
        initial_policy_drift <= 1e-8
        and active_fraction > 0.0
        and role_reward_abs > 0.0
        and commit_reward_zero
        and commit_mask_zero
        and commit_advantage_zero
        and frozen_decoder_unchanged
        and drift > 0.0
        and drift <= 0.01
        and update_reports[-1]["trajectory_reference_kl"] > 0.0
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
        "group_size": int(
            config.model.model_kwargs.trace_rl_config.group_size
        ),
        "formal_policy_update_epochs": formal_update_count,
        "stress_policy_updates": stress_update_count,
        "trace_steps": int(model.n_trace_steps),
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
        "train_time_teacher_usage": "reward_scoring_only",
        "policy_conditioning": "question_only",
        "initial_policy_reference_relative_drift": initial_policy_drift,
        "trajectory_advantage_active_fraction": active_fraction,
        "semantic_role_reward_abs_sum": role_reward_abs,
        "commit_reward_exact_zero": commit_reward_zero,
        "commit_reward_mask_exact_zero": commit_mask_zero,
        "commit_advantage_exact_zero": commit_advantage_zero,
        "answer_decoder_frozen": True,
        "answer_decoder_unchanged": frozen_decoder_unchanged,
        "trajectory_policy_relative_drift_after_updates": drift,
        "trainable_parameter_names": trainable_names,
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
