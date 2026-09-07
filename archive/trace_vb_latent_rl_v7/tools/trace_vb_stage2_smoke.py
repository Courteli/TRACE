#!/usr/bin/env python3
"""Real-checkpoint Stage-2 value bridge, GAE, head-only PPO and memory gate."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils.data._utils.collate import default_collate
from omegaconf import OmegaConf


ROOT = Path(os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))

from src.utils.utils import instantiate_from_config  # noqa: E402
from src.modules.trace_vb import action_efficacy_hinge  # noqa: E402


MIB = 1024**2
MIN_HEADROOM_MIB = 4096.0


def build_config():
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "src/configs/trainer/default.yaml"),
        OmegaConf.load(ROOT / "src/configs/models/trace_vb_policy_qwen3_instruct.yaml"),
        OmegaConf.load(ROOT / "src/configs/datasets/gsm8k_aug_nl.yaml"),
    )
    config.model.model_kwargs.do_trace_rl = True
    config.model.training_kwargs.scheduler.warmup_steps = 300
    config.model.training_kwargs.scheduler.num_training_steps = 20480
    config.dataloader.batch_size = 1
    config.dataloader.val_batch_size = 1
    config.dataloader.num_workers = 0
    config.dataloader.persistent_workers = False
    config.args = OmegaConf.create({"workspace_path": str(ROOT), "no_log": True})
    return config


def load_stage1(model, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("trace_policy_training_stage", -1)) != 1:
        raise RuntimeError("Stage-2 smoke requires a TRACE-VB Stage-1 checkpoint")
    model.on_load_checkpoint(checkpoint)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("unexpected Stage-1 keys: " + ", ".join(incompatible.unexpected_keys[:10]))
    if not model._cot_encoder_adapter_loaded:
        raise RuntimeError("Stage-1 checkpoint lacks its frozen CoT encoder")
    if not model._solve_text_decoder_loaded:
        raise RuntimeError("Stage-1 checkpoint lacks its trained text decoder")
    model._snapshot_stage1_policy()
    model.value_critic.copy_from(model.sufficiency_head)
    model._value_bridge_initialized = True
    model._stage2_initialized = True
    del checkpoint
    gc.collect()


def exact_module_copy(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state, right_state = left.state_dict(), right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def trainable_fingerprint(module: torch.nn.Module) -> torch.Tensor:
    values = [p.detach().float().reshape(-1) for p in module.parameters() if p.requires_grad]
    if not values:
        raise RuntimeError("module has no trainable parameters")
    flat = torch.cat(values)
    return torch.stack((flat.sum(), flat.abs().sum(), flat.square().sum())).cpu()


def grad_norm(module: torch.nn.Module) -> float:
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).norm().cpu()) if values else 0.0


def snapshot(device: torch.device, phase: str) -> dict:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    return {
        "phase": phase,
        "free_mib": float(free / MIB),
        "total_mib": float(total / MIB),
        "reserved_mib": float(reserved / MIB),
        "external_or_nonallocator_mib": float(max(0, total - free - reserved) / MIB),
    }


def unwrap_optimizer(configured):
    if isinstance(configured, dict):
        return configured["optimizer"]
    if isinstance(configured, (tuple, list)):
        return configured[0]
    return configured


def unwrap_scheduler(configured):
    if not isinstance(configured, dict):
        return None
    scheduler = configured.get("lr_scheduler")
    if isinstance(scheduler, dict):
        scheduler = scheduler.get("scheduler")
    return scheduler


@torch.no_grad()
def measure_stage1_action_efficacy(model, question: str, *, paths: int = 8) -> dict:
    """Measure whether Stage-1 actions still causally move latent transitions."""
    if int(paths) != 8:
        raise ValueError("formal pre-RL efficacy audit requires exactly eight paths")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        map_path = model._trajectory_latents([question], deterministic=True)
    map_actions = map_path["actions"].detach().float()
    map_transitions = map_path["implicit_residuals"].detach().float()
    sampled_actions = []
    sampled_transitions = []
    action_masks = []
    del map_path
    for _ in range(int(paths)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sampled_path = model._trajectory_latents(
                [question], deterministic=False
            )
        sampled_actions.append(sampled_path["actions"].detach().float())
        sampled_transitions.append(
            sampled_path["implicit_residuals"].detach().float()
        )
        action_masks.append(
            sampled_path["stochastic_action_mask"].detach().bool()
        )
        del sampled_path
    sampled_actions = torch.cat(sampled_actions, dim=0)
    sampled_transitions = torch.cat(sampled_transitions, dim=0)
    action_mask = torch.cat(action_masks, dim=0)
    map_actions = map_actions.expand_as(sampled_actions)
    map_transitions = map_transitions.expand_as(sampled_transitions)
    formal_target = float(
        model.trace_config.get("stage1_minimum_action_efficacy_ratio", 0.02)
    )
    efficacy = action_efficacy_hinge(
        sampled_actions,
        map_actions,
        sampled_transitions,
        map_transitions,
        action_mask,
        minimum_ratio=formal_target,
    )
    action_distance = (
        (sampled_actions - map_actions).norm(dim=-1)
        / float(sampled_actions.shape[-1]) ** 0.5
    )
    transition_distance = (
        (sampled_transitions - map_transitions).norm(dim=-1)
        / float(sampled_transitions.shape[-1]) ** 0.5
    )
    active_ratios = (
        transition_distance
        / action_distance.clamp_min(torch.finfo(torch.float32).eps)
    )[action_mask]
    if active_ratios.numel() != 8 * 7:
        raise RuntimeError("pre-RL efficacy audit did not cover 8x7 actions")
    helper_ratio = float(efficacy.active_efficacy_ratio.cpu())
    direct_ratio = float(active_ratios.mean().cpu())
    if abs(helper_ratio - direct_ratio) > 1e-6:
        raise RuntimeError("action-efficacy helper and direct audit disagree")
    return {
        "paths": 8,
        "active_actions": int(active_ratios.numel()),
        "role_schema": "PLAN,SOLVE1,SOLVE2,SOLVE3,SOLVE4,SOLVE5,REFINE",
        "distance": "L2/sqrt(feature_dim)",
        "mean_ratio": helper_ratio,
        "minimum_ratio": float(active_ratios.min().cpu()),
        "maximum_ratio": float(active_ratios.max().cpu()),
        "formal_stage1_target": formal_target,
        "smoke_gate": 0.01,
        "hinge_at_formal_target": float(efficacy.loss.cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("TRACE-VB Stage-2 smoke requires CUDA")
    config = build_config()
    rl = config.model.model_kwargs.trace_rl_config
    if not (
        int(rl.group_size) == 8
        and int(rl.rollout_micro_batch_size) == 1
        and bool(rl.use_semantic_anchor)
        and int(rl.policy_update_epochs) == 4
        and int(rl.critic_warmup_batches) == 256
        and int(config.model.training_kwargs.scheduler.num_training_steps) == 20480
    ):
        raise RuntimeError("Stage-2 smoke did not load the formal budget")
    if any(float(rl[name]) != 0.0 for name in ("dense_outcome_weight", "step_reward_weight", "trajectory_length_weight")):
        raise RuntimeError("legacy reward shaping is active")

    data = instantiate_from_config(config.data_module, extra_kwargs={"all_config": config})
    data.setup("fit")
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    load_stage1(model, args.stage1_checkpoint)
    if not exact_module_copy(model.sufficiency_head, model.value_critic):
        raise RuntimeError("S-to-V copy is not exact at the Stage-2 boundary")
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    allowed_trainable_prefixes = tuple(
        f"trajectory_policy.{head}.{role}."
        for head in ("mean_heads", "log_std_heads")
        for role in ("plan", "solve", "check")
    ) + ("value_critic.",)
    escaped_trainable = [
        name for name in trainable_names
        if not name.startswith(allowed_trainable_prefixes)
    ]
    if not trainable_names or escaped_trainable:
        raise RuntimeError(
            "Stage-2 trainability escaped stochastic heads/value critic: "
            + ", ".join(escaped_trainable[:20])
        )
    required_actor_prefixes = allowed_trainable_prefixes[:-1]
    missing_actor_groups = [
        prefix
        for prefix in required_actor_prefixes
        if not any(name.startswith(prefix) for name in trainable_names)
    ]
    if missing_actor_groups:
        raise RuntimeError(
            "Stage-2 is missing trainable stochastic head groups: "
            + ", ".join(missing_actor_groups)
        )
    if not any(name.startswith("value_critic.") for name in trainable_names):
        raise RuntimeError("Stage-2 value critic has no trainable parameters")
    if any(parameter.requires_grad for parameter in model.solve_text_decoder.parameters()):
        raise RuntimeError(
            "Stage-2 semantic decoder must be frozen and optimizer-external"
        )
    forbidden_trainable_fragments = (
        "policy_trunk",
        "policy_step_embedding",
        "dynamics_step_embedding",
        "base_projector",
        "action_projector",
        "mean_heads.commit",
    )
    if any(
        fragment in name
        for name in trainable_names
        for fragment in forbidden_trainable_fragments
    ):
        raise RuntimeError(
            "Stage-2 shared dynamics, embeddings, or COMMIT are trainable"
        )

    device = torch.device("cuda:0")
    model = model.to(device).train()
    model.__dict__["manual_backward"] = lambda loss: loss.backward()
    configured_optimizers = model.configure_optimizers()
    optimizer = unwrap_optimizer(configured_optimizers)
    scheduler = unwrap_scheduler(configured_optimizers)
    if len(optimizer.param_groups) != 2:
        raise RuntimeError("Stage 2 requires separate actor and critic parameter groups")
    if scheduler is None:
        raise RuntimeError("Stage 2 smoke requires the formal step scheduler")
    # Creating the warmup scheduler immediately changes the live group LR to
    # the step-zero value (zero).  Validate the configured/base LR retained by
    # PyTorch instead of mistaking scheduler initialization for a mismatch.
    actor_base_lr = optimizer.param_groups[0].get(
        "initial_lr", optimizer.param_groups[0]["lr"]
    )
    critic_base_lr = optimizer.param_groups[1].get(
        "initial_lr", optimizer.param_groups[1]["lr"]
    )
    if float(actor_base_lr) != float(rl.actor_lr):
        raise RuntimeError("actor learning rate mismatch")
    if float(critic_base_lr) != float(rl.critic_lr):
        raise RuntimeError("critic learning rate mismatch")

    row = data.train_set[int(args.dataset_index)]
    torch.manual_seed(20260809)
    torch.cuda.manual_seed(20260809)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    snapshots = [snapshot(device, "post_initialization")]
    action_efficacy = measure_stage1_action_efficacy(
        model,
        str(row["question"]),
    )
    snapshots.append(snapshot(device, "post_action_efficacy_audit"))
    if action_efficacy["mean_ratio"] < action_efficacy["smoke_gate"]:
        raise RuntimeError(
            "Stage-1 action efficacy is too weak for RL: "
            f"{action_efficacy['mean_ratio']:.6f}"
            f"<{action_efficacy['smoke_gate']:.6f}"
        )
    batch = default_collate([row])
    model.vb_rollout_batches_seen.fill_(int(rl.critic_warmup_batches))
    semantic_anchor_weight = model._semantic_anchor_weight()
    if not semantic_anchor_weight > 0.0:
        raise RuntimeError("semantic anchor did not activate after critic warm-up")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        semantic_anchor = model._stage2_role_semantic_anchor(batch)
    if (
        not semantic_anchor["total"].requires_grad
        or not bool(torch.isfinite(semantic_anchor["total"]).item())
    ):
        raise RuntimeError("Stage-2 semantic anchor is detached or non-finite")
    if float(semantic_anchor["truncated_fraction"]) != 0.0:
        raise RuntimeError("Stage-2 semantic anchor detected CoT truncation")
    (semantic_anchor_weight * semantic_anchor["total"]).backward()
    semantic_anchor_actor_grad = grad_norm(model.trajectory_policy)
    semantic_anchor_decoder_grad = grad_norm(model.solve_text_decoder)
    if not semantic_anchor_actor_grad > 0.0:
        raise RuntimeError("semantic anchor has no actor-mean gradient")
    if semantic_anchor_decoder_grad != 0.0:
        raise RuntimeError("semantic anchor updated the frozen decoder")
    semantic_anchor_report = {
        "weight": semantic_anchor_weight,
        "total": float(semantic_anchor["total"].detach().float().cpu()),
        "plan": float(semantic_anchor["plan"].float().cpu()),
        "solve_text": float(semantic_anchor["solve_text"].float().cpu()),
        "refine": float(semantic_anchor["refine"].float().cpu()),
        "truncated_fraction": float(
            semantic_anchor["truncated_fraction"].float().cpu()
        ),
        "actor_grad_norm": semantic_anchor_actor_grad,
        "decoder_grad_norm": semantic_anchor_decoder_grad,
        "terminal_reward_unchanged": True,
    }
    optimizer.zero_grad(set_to_none=True)
    del batch, semantic_anchor
    snapshots.append(snapshot(device, "post_semantic_anchor_backward"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rollout = model.trace_policy_rollout([row["question"]], [row["answer"]])
    snapshots.append(snapshot(device, "post_rollout"))
    required = {
        "pre_action_states",
        "role_ids",
        "action_mask",
        "values",
        "old_action_log_probs",
        "actions",
        "terminal_rewards",
        "advantages",
        "returns",
        "greedy_accuracy",
        "greedy_lengths",
    }
    missing_keys = sorted(required - set(rollout))
    if missing_keys:
        raise RuntimeError("rollout is missing: " + ", ".join(missing_keys))
    float32_fields = (
        "actions",
        "old_action_log_probs",
        "values",
        "advantages",
        "returns",
    )
    wrong_dtypes = {
        name: str(rollout[name].dtype)
        for name in float32_fields
        if rollout[name].dtype != torch.float32
    }
    if wrong_dtypes:
        raise RuntimeError(
            "PPO rollout tensors must be float32: "
            + ", ".join(
                f"{name}={dtype}" for name, dtype in wrong_dtypes.items()
            )
        )
    forbidden = {"semantic_step_rewards", "dense_outcome", "trajectory_rewards", "answer_rewards"}
    if forbidden & set(rollout):
        raise RuntimeError("terminal-only rollout exposed legacy shaped rewards")
    if tuple(rollout["pre_action_states"].shape[:2]) != (8, 8):
        raise RuntimeError("pre-action states must have [group=8, role=8, hidden]")
    if rollout["pre_action_states"].requires_grad:
        raise RuntimeError("saved pre-action states must be detached")
    expected_roles = torch.arange(8, device=device).view(1, -1).expand(8, -1)
    if not torch.equal(rollout["role_ids"], expected_roles):
        raise RuntimeError("role IDs are not the ordered eight-state program")
    mask = rollout["action_mask"].bool()
    if not bool(mask[:, :7].all()) or bool(mask[:, 7].any()):
        raise RuntimeError("action mask must include PLAN-through-REFINE and exclude COMMIT")
    if not torch.equal(rollout["advantages"][:, 7], torch.zeros_like(rollout["advantages"][:, 7])):
        raise RuntimeError("COMMIT received a GAE advantage")
    if not torch.equal(rollout["returns"][:, 7], torch.zeros_like(rollout["returns"][:, 7])):
        raise RuntimeError("COMMIT received a value target")
    if not all(torch.isfinite(rollout[name]).all() for name in ("values", "advantages", "returns")):
        raise RuntimeError("critic or GAE produced non-finite values")
    if not bool(((rollout["terminal_rewards"] == 0) | (rollout["terminal_rewards"] == 1)).all()):
        raise RuntimeError("terminal reward is not exact correctness")

    # Critic-only warm-up update.  A raising LM forward proves the update uses
    # only cached states; it cannot silently replay Qwen.
    original_llm_forward = model.llm.forward
    llm_calls = 0

    def forbidden_llm_forward(*_args, **_kwargs):
        nonlocal llm_calls
        llm_calls += 1
        raise RuntimeError("head-only PPO attempted a Qwen forward")

    model.llm.forward = forbidden_llm_forward
    optimizer.zero_grad(set_to_none=True)
    warmup_metrics = model._trajectory_policy_update(rollout, actor_active=False)
    warmup_actor_grad = grad_norm(model.trajectory_policy)
    warmup_critic_grad = grad_norm(model.value_critic)
    if warmup_actor_grad != 0.0 or not warmup_critic_grad > 0.0:
        raise RuntimeError("critic warm-up did not freeze actor and train critic")
    optimizer.step()
    scheduler.step()

    actor_before = trainable_fingerprint(model.trajectory_policy)
    critic_before = trainable_fingerprint(model.value_critic)
    updates = []
    for update_index in range(4):
        optimizer.zero_grad(set_to_none=True)
        metrics = model._trajectory_policy_update(rollout, actor_active=True)
        actor_grad = grad_norm(model.trajectory_policy)
        critic_grad = grad_norm(model.value_critic)
        if not actor_grad > 0.0 or not critic_grad > 0.0:
            raise RuntimeError("active PPO epoch lacks actor or critic gradient")
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(rl.clip_grad_norm),
        )
        optimizer.step()
        scheduler.step()
        updates.append(
            {
                "index": update_index,
                "actor_loss": float(metrics["actor_loss"].cpu()),
                "value_loss": float(metrics["value_loss"].cpu()),
                "ratio_deviation": float(metrics["ratio_deviation"].cpu()),
                "clip_fraction": float(metrics["clip_fraction"].cpu()),
                "actor_grad_norm": actor_grad,
                "critic_grad_norm": critic_grad,
            }
        )
    model.llm.forward = original_llm_forward
    snapshots.append(snapshot(device, "post_four_head_updates"))
    actor_after = trainable_fingerprint(model.trajectory_policy)
    critic_after = trainable_fingerprint(model.value_critic)
    if torch.equal(actor_before, actor_after) or torch.equal(critic_before, critic_after):
        raise RuntimeError("four PPO epochs did not change both actor and critic")
    if llm_calls != 0:
        raise RuntimeError("head-only PPO called Qwen")
    if not updates[-1]["ratio_deviation"] > 0.0:
        raise RuntimeError("later PPO epochs never departed from the rollout policy")

    peak_reserved = float(torch.cuda.max_memory_reserved(device) / MIB)
    total_mib = snapshots[0]["total_mib"]
    max_external = max(item["external_or_nonallocator_mib"] for item in snapshots)
    effective_headroom = total_mib - peak_reserved - max_external
    minimum_free = min(item["free_mib"] for item in snapshots)
    failures = []
    if effective_headroom < MIN_HEADROOM_MIB:
        failures.append(f"effective_headroom={effective_headroom:.1f}<{MIN_HEADROOM_MIB:.1f}")
    if minimum_free < MIN_HEADROOM_MIB:
        failures.append(f"minimum_free={minimum_free:.1f}<{MIN_HEADROOM_MIB:.1f}")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "scope": "real_trace_vb_stage2_smoke_not_an_experiment_result",
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "value_bridge_exact_copy": True,
        "pre_rl_action_efficacy": action_efficacy,
        "critic_warmup_batches": int(rl.critic_warmup_batches),
        "warmup_actor_grad_norm": warmup_actor_grad,
        "semantic_anchor": semantic_anchor_report,
        "solve_text_decoder_frozen": True,
        "warmup_critic_grad_norm": warmup_critic_grad,
        "group_size": 8,
        "rollout_micro_batch_size": 1,
        "rollout_keys": sorted(required),
        "rollout_float32_fields": list(float32_fields),
        "commit_action_mask_zero": True,
        "terminal_exact_reward_only": True,
        "ppo_epochs": updates,
        "head_only_qwen_forward_calls": llm_calls,
        "actor_changed": True,
        "critic_changed": True,
        "trainable_parameter_names": trainable_names,
        "allowed_trainable_prefixes": list(allowed_trainable_prefixes),
        "shared_trunk_step_embeddings_and_commit_frozen": True,
        "peak_reserved_mib": peak_reserved,
        "max_external_or_nonallocator_mib": max_external,
        "effective_peak_headroom_mib": effective_headroom,
        "minimum_observed_free_mib": minimum_free,
        "required_headroom_mib": MIN_HEADROOM_MIB,
        "memory_snapshots": snapshots,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("TRACE-VB Stage-2 memory gate failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
