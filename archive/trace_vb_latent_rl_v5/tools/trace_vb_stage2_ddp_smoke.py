#!/usr/bin/env python3
"""Four-rank real-checkpoint DDP gate for TRACE-VB Stage 2.

This is deliberately a preflight, not a shortened training run.  Every rank
collects one real group-8 terminal-reward rollout with the Stage-1 checkpoint.
The expensive LM is then excluded from the update path, while a small DDP
proxy executes the same FP32 actor/critic objective over the cached states.
That makes the smoke both faithful to head-only PPO and able to exercise the
actual torch.distributed gradient reducer without replaying Qwen.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(ROOT))

from src.modules.trace_policy import (  # noqa: E402
    diagonal_gaussian_kl,
    gaussian_log_prob,
)
from src.modules.trace_vb import (  # noqa: E402
    masked_ppo_actor_loss,
    masked_value_loss,
)
from src.utils.utils import instantiate_from_config  # noqa: E402
from tools.trace_vb_stage2_smoke import (  # noqa: E402
    build_config,
    exact_module_copy,
    load_stage1,
    unwrap_optimizer,
    unwrap_scheduler,
)


MIB = 1024**2
MIN_HEADROOM_MIB = float(
    os.environ.get("TRACE_VB_DDP_MIN_HEADROOM_MIB", "4096")
)
if MIN_HEADROOM_MIB < 0.0 or 0.0 < MIN_HEADROOM_MIB < 3800.0:
    raise RuntimeError(
        "TRACE_VB_DDP_MIN_HEADROOM_MIB must be 0 (disabled) or at least "
        "3800 MiB"
    )
FLOAT32_ROLLOUT_FIELDS = (
    "actions",
    "pre_action_states",
    "old_action_log_probs",
    "terminal_rewards",
    "values",
    "advantages",
    "raw_advantages",
    "returns",
    "greedy_accuracy",
    "greedy_lengths",
)
STOCHASTIC_ROLES = ("plan", "solve", "check")


class Stage2HeadDDPProxy(nn.Module):
    """Exact full-group form of the cached-state TRACE-VB PPO objective.

    ``LitTRACEVB._trajectory_policy_update`` performs the same reductions in
    micro-batches and invokes Lightning's manual backward internally.  A
    backward call inside ``DistributedDataParallel.forward`` is not a valid
    way to arm DDP's reducer.  This proxy therefore returns the exact scalar
    objective and lets the caller perform a normal DDP backward.  It owns only
    the real actor, critic, and frozen Stage-1 reference modules.
    """

    def __init__(self, model):
        super().__init__()
        self.actor = model.trajectory_policy
        self.critic = model.value_critic
        self.reference = model.stage1_policy_reference
        self.n_steps = int(model.n_trace_steps)
        self.clip_epsilon = float(
            model.trace_rl_config.get("trajectory_clip_epsilon", 0.12)
        )
        self.kl_weight = float(
            model.trace_rl_config.get("stage1_policy_kl_weight", 0.02)
        )
        self.entropy_weight = float(
            model.trace_rl_config.get("entropy_coefficient", 0.001)
        )
        self.value_weight = float(
            model.trace_rl_config.get("value_loss_coefficient", 0.5)
        )
        self.value_loss_type = str(
            model.trace_rl_config.get("value_loss_type", "huber")
        )
        role_entropy_weights = list(
            model.trace_rl_config.get(
                "role_entropy_weights", [1.0] * 7 + [0.0]
            )
        )
        if len(role_entropy_weights) != self.n_steps:
            raise ValueError("role_entropy_weights must contain 8 entries")
        self.register_buffer(
            "role_entropy_weights",
            torch.tensor(role_entropy_weights, dtype=torch.float32),
            persistent=False,
        )

    def _parameters_on_states(
        self,
        policy: nn.Module,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means = []
        log_stds = []
        for step_index in range(self.n_steps):
            mean, log_std = policy.distribution_parameters(
                states[:, step_index, :].float(), step_index
            )
            means.append(mean.float())
            log_stds.append(log_std.float())
        return torch.stack(means, dim=1), torch.stack(log_stds, dim=1)

    @staticmethod
    def _masked_kl(
        means: torch.Tensor,
        log_stds: torch.Tensor,
        reference_means: torch.Tensor,
        reference_log_stds: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        per_step = diagonal_gaussian_kl(
            means,
            log_stds,
            reference_means,
            reference_log_stds,
        )
        weights = mask.to(device=per_step.device, dtype=per_step.dtype)
        return (per_step * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(
        self,
        rollout: dict[str, torch.Tensor],
        actor_active: bool,
    ) -> dict[str, torch.Tensor]:
        states = rollout["pre_action_states"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_action_log_probs"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]
        action_mask = rollout["action_mask"].bool()
        role_ids = rollout["role_ids"]
        with torch.autocast(device_type=states.device.type, enabled=False):
            if actor_active:
                current_means, current_log_stds = self._parameters_on_states(
                    self.actor, states
                )
            else:
                with torch.no_grad():
                    current_means, current_log_stds = self._parameters_on_states(
                        self.actor, states
                    )
            with torch.no_grad():
                reference_means, reference_log_stds = self._parameters_on_states(
                    self.reference, states
                )
            current_log_probs = gaussian_log_prob(
                actions.float(),
                current_means.float(),
                current_log_stds.float(),
            )
            actor_loss = masked_ppo_actor_loss(
                current_log_probs.float(),
                old_log_probs.float(),
                advantages.float(),
                action_mask,
                clip_epsilon=self.clip_epsilon,
            )
            stage1_kl = self._masked_kl(
                current_means.float(),
                current_log_stds.float(),
                reference_means.float(),
                reference_log_stds.float(),
                action_mask,
            )
            per_role_entropy = (
                current_log_stds.float()
                + 0.5 * math.log(2.0 * math.pi * math.e)
            ).sum(dim=-1)
            entropy_mask = (
                action_mask.float() * self.role_entropy_weights.view(1, -1)
            )
            entropy = (
                per_role_entropy * entropy_mask
            ).sum() / entropy_mask.sum().clamp_min(1.0)
            predicted_values = torch.sigmoid(
                self.critic(states.float(), role_ids).float()
            )
            value_loss = masked_value_loss(
                predicted_values,
                returns.float(),
                action_mask,
                loss_type=self.value_loss_type,
            )
            log_ratio = (current_log_probs - old_log_probs.float()).clamp(
                min=-20.0, max=20.0
            )
            ratio = torch.exp(log_ratio)
            active_count = action_mask.float().sum().clamp_min(1.0)
            ratio_deviation = (
                (ratio - 1.0).abs() * action_mask.float()
            ).sum() / active_count
            clip_fraction = (
                (
                    (ratio < 1.0 - self.clip_epsilon)
                    | (ratio > 1.0 + self.clip_epsilon)
                ).float()
                * action_mask.float()
            ).sum() / active_count
            objective = self.value_weight * value_loss
            if actor_active:
                objective = objective + (
                    actor_loss
                    + self.kl_weight * stage1_kl
                    - self.entropy_weight * entropy
                )
        for name, value in {
            "objective": objective,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "stage1_kl": stage1_kl,
            "entropy": entropy,
            "ratio_deviation": ratio_deviation,
            "clip_fraction": clip_fraction,
        }.items():
            if value.dtype != torch.float32 or not torch.isfinite(value):
                raise RuntimeError(
                    f"Stage-2 DDP {name} must be finite FP32, got "
                    f"{value.dtype}"
                )
        return {
            "objective": objective,
            "actor_loss": actor_loss.detach(),
            "value_loss": value_loss.detach(),
            "stage1_kl": stage1_kl.detach(),
            "entropy": entropy.detach(),
            "ratio_deviation": ratio_deviation.detach(),
            "clip_fraction": clip_fraction.detach(),
        }


def snapshot(device: torch.device, phase: str) -> dict:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    return {
        "phase": phase,
        "free_mib": float(free / MIB),
        "total_mib": float(total / MIB),
        "reserved_mib": float(reserved / MIB),
        "external_or_nonallocator_mib": float(
            max(0, total - free - reserved) / MIB
        ),
    }


def trainable_names(model: nn.Module) -> list[str]:
    return [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def validate_trainability(model: nn.Module) -> tuple[list[str], tuple[str, ...]]:
    names = trainable_names(model)
    allowed = tuple(
        f"trajectory_policy.{head}.{role}."
        for head in ("mean_heads", "log_std_heads")
        for role in STOCHASTIC_ROLES
    ) + ("value_critic.",)
    escaped = [name for name in names if not name.startswith(allowed)]
    missing = [
        prefix for prefix in allowed
        if not any(name.startswith(prefix) for name in names)
    ]
    forbidden = (
        "policy_trunk",
        "policy_step_embedding",
        "dynamics_step_embedding",
        "base_projector",
        "action_projector",
        "mean_heads.commit",
    )
    if not names or escaped or missing or any(
        fragment in name for name in names for fragment in forbidden
    ):
        raise RuntimeError(
            "invalid Stage-2 trainability whitelist: "
            f"escaped={escaped[:10]}, missing={missing[:10]}"
        )
    return names, allowed


def validate_rollout(rollout: dict[str, torch.Tensor], device: torch.device) -> None:
    required = set(FLOAT32_ROLLOUT_FIELDS) | {
        "role_ids",
        "action_mask",
    }
    missing = sorted(required - set(rollout))
    if missing:
        raise RuntimeError("Stage-2 DDP rollout is missing: " + ", ".join(missing))
    wrong_dtypes = {
        name: str(rollout[name].dtype)
        for name in FLOAT32_ROLLOUT_FIELDS
        if rollout[name].dtype != torch.float32
    }
    if wrong_dtypes:
        raise RuntimeError(f"Stage-2 DDP rollout is not FP32: {wrong_dtypes}")
    if tuple(rollout["pre_action_states"].shape[:2]) != (8, 8):
        raise RuntimeError("each rank must collect one [group=8, role=8] rollout")
    if rollout["pre_action_states"].requires_grad:
        raise RuntimeError("cached pre-action states must be detached")
    expected_roles = torch.arange(8, device=device).view(1, -1).expand(8, -1)
    if rollout["role_ids"].dtype != torch.long or not torch.equal(
        rollout["role_ids"], expected_roles
    ):
        raise RuntimeError("rollout role IDs are not the ordered eight-state program")
    mask = rollout["action_mask"]
    if mask.dtype != torch.bool or not bool(mask[:, :7].all()) or bool(mask[:, 7].any()):
        raise RuntimeError("action mask must include roles 0:7 and exclude COMMIT")
    rewards = rollout["terminal_rewards"]
    if not torch.equal(rewards, rollout["greedy_accuracy"]):
        raise RuntimeError("terminal reward must equal greedy exact-answer accuracy")
    if not bool(((rewards == 0.0) | (rewards == 1.0)).all()):
        raise RuntimeError("terminal reward must be binary exact correctness")
    if not torch.equal(
        rollout["advantages"][:, 7],
        torch.zeros_like(rollout["advantages"][:, 7]),
    ) or not torch.equal(
        rollout["returns"][:, 7],
        torch.zeros_like(rollout["returns"][:, 7]),
    ):
        raise RuntimeError("COMMIT must not receive GAE or a value target")
    forbidden = {
        "semantic_step_rewards",
        "dense_outcome",
        "trajectory_rewards",
        "answer_rewards",
    }
    if forbidden & set(rollout):
        raise RuntimeError("legacy shaped rewards escaped into the DDP rollout")
    if not all(
        bool(torch.isfinite(rollout[name]).all())
        for name in FLOAT32_ROLLOUT_FIELDS
    ):
        raise RuntimeError("Stage-2 DDP rollout contains non-finite tensors")


def module_parameters(module: nn.Module) -> list[torch.Tensor]:
    return [p for p in module.parameters() if p.requires_grad]


def clone_parameters(parameters: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def maximum_parameter_change(
    before: list[torch.Tensor],
    parameters: list[torch.Tensor],
) -> float:
    if len(before) != len(parameters):
        raise RuntimeError("parameter snapshot cardinality changed")
    return max(
        float((parameter.detach() - reference).abs().max().cpu())
        for reference, parameter in zip(before, parameters)
    )


def gradient_norm(parameters: Iterable[torch.Tensor]) -> float:
    norms = [
        parameter.grad.detach().float().norm()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(norms).norm().cpu()) if norms else 0.0


def cross_rank_max_difference(
    tensors: Iterable[torch.Tensor],
    *,
    rank: int,
    device: torch.device,
) -> float:
    """Compare every tensor with rank 0 and return the global max error."""
    maximum = torch.zeros((), device=device, dtype=torch.float32)
    for tensor in tensors:
        reference = tensor.detach().float().clone()
        dist.broadcast(reference, src=0)
        if rank != 0:
            maximum = torch.maximum(
                maximum,
                (tensor.detach().float() - reference).abs().max(),
            )
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.cpu())


def synchronized_gradients(
    parameters: list[torch.Tensor],
    *,
    rank: int,
    device: torch.device,
) -> float:
    presence = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        device=device,
        dtype=torch.int32,
    )
    minimum = presence.clone()
    maximum = presence.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise RuntimeError("gradient presence differs across DDP ranks")
    gradients = [
        parameter.grad for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return cross_rank_max_difference(
        gradients, rank=rank, device=device
    )


def reduced_mean(value: torch.Tensor) -> float:
    result = value.detach().float().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result.div_(dist.get_world_size())
    return float(result.cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--actor-updates", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.actor_updates < 2:
        raise SystemExit("Stage-2 DDP smoke requires at least two actor updates")
    if not torch.cuda.is_available():
        raise SystemExit("TRACE-VB Stage-2 DDP smoke requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("formal TRACE-VB Stage-2 DDP smoke requires four ranks")

    torch.manual_seed(20260809 + rank)
    torch.cuda.manual_seed(20260809 + rank)
    config = build_config()
    rl = config.model.model_kwargs.trace_rl_config
    if not (
        int(rl.group_size) == 8
        and int(rl.rollout_micro_batch_size) == 1
        and int(rl.critic_warmup_batches) == 256
        and bool(rl.use_semantic_anchor)
        and int(rl.policy_update_epochs) == 4
    ):
        raise RuntimeError("Stage-2 DDP smoke did not load the formal RL contract")
    if any(
        float(rl[name]) != 0.0
        for name in (
            "dense_outcome_weight",
            "step_reward_weight",
            "trajectory_length_weight",
        )
    ):
        raise RuntimeError("Stage-2 DDP smoke found active reward shaping")

    data = instantiate_from_config(
        config.data_module, extra_kwargs={"all_config": config}
    )
    data.setup("fit")
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    load_stage1(model, args.stage1_checkpoint)
    if not exact_module_copy(model.sufficiency_head, model.value_critic):
        raise RuntimeError("S-to-V copy is not exact at the DDP Stage-2 boundary")
    names, allowed_prefixes = validate_trainability(model)
    if any(parameter.requires_grad for parameter in model.solve_text_decoder.parameters()):
        raise RuntimeError(
            "Stage-2 DDP smoke requires a frozen semantic decoder"
        )
    model = model.to(device).train()

    dataset_index = (int(args.dataset_index) + rank) % len(data.train_set)
    row = data.train_set[dataset_index]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    snapshots = [snapshot(device, "post_initialization")]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rollout = model.trace_policy_rollout(
            [str(row["question"])], [str(row["answer"])]
        )
    validate_rollout(rollout, device)
    snapshots.append(snapshot(device, "post_group8_rollout"))

    proxy = Stage2HeadDDPProxy(model).to(device)
    wrapped = DistributedDataParallel(
        proxy,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=True,
    )
    configured_optimizers = model.configure_optimizers()
    optimizer = unwrap_optimizer(configured_optimizers)
    scheduler = unwrap_scheduler(configured_optimizers)
    if len(optimizer.param_groups) != 2:
        raise RuntimeError("Stage-2 DDP smoke requires actor/critic optimizer groups")
    if scheduler is None:
        raise RuntimeError("Stage-2 DDP smoke requires the formal step scheduler")
    # Scheduler construction applies the step-zero warmup multiplier to the
    # live LR.  The optimizer's initial_lr is the configured value that this
    # preflight is intended to validate.
    actor_base_lr = optimizer.param_groups[0].get(
        "initial_lr", optimizer.param_groups[0]["lr"]
    )
    critic_base_lr = optimizer.param_groups[1].get(
        "initial_lr", optimizer.param_groups[1]["lr"]
    )
    if float(actor_base_lr) != float(rl.actor_lr):
        raise RuntimeError("Stage-2 DDP actor LR mismatch")
    if float(critic_base_lr) != float(rl.critic_lr):
        raise RuntimeError("Stage-2 DDP critic LR mismatch")

    actor_parameters = module_parameters(proxy.actor)
    critic_parameters = module_parameters(proxy.critic)
    all_parameters = actor_parameters + critic_parameters
    if set(map(id, all_parameters)) != {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }:
        raise RuntimeError("optimizer parameters differ from DDP trainable heads")

    initial_parameter_sync = cross_rank_max_difference(
        all_parameters, rank=rank, device=device
    )
    if initial_parameter_sync != 0.0:
        raise RuntimeError(
            f"DDP parameters differ after initialization: {initial_parameter_sync}"
        )

    original_llm_forward = model.llm.forward
    llm_calls = 0

    def forbidden_llm_forward(*_args, **_kwargs):
        nonlocal llm_calls
        llm_calls += 1
        raise RuntimeError("Stage-2 DDP cached-state update replayed Qwen")

    model.llm.forward = forbidden_llm_forward
    actor_before_warmup = clone_parameters(actor_parameters)
    critic_before_warmup = clone_parameters(critic_parameters)
    optimizer.zero_grad(set_to_none=True)
    warmup_output = wrapped(rollout, False)
    warmup_output["objective"].backward()
    warmup_actor_grad = gradient_norm(actor_parameters)
    warmup_critic_grad = gradient_norm(critic_parameters)
    warmup_gradient_sync = synchronized_gradients(
        all_parameters, rank=rank, device=device
    )
    if warmup_actor_grad != 0.0 or not warmup_critic_grad > 0.0:
        raise RuntimeError("each DDP rank must freeze actor and train critic in warmup")
    if warmup_gradient_sync != 0.0:
        raise RuntimeError(
            f"warmup DDP gradients are not synchronized: {warmup_gradient_sync}"
        )
    torch.nn.utils.clip_grad_norm_(all_parameters, float(rl.clip_grad_norm))
    optimizer.step()
    scheduler.step()
    warmup_parameter_sync = cross_rank_max_difference(
        all_parameters, rank=rank, device=device
    )
    warmup_actor_change = maximum_parameter_change(
        actor_before_warmup, actor_parameters
    )
    warmup_critic_change = maximum_parameter_change(
        critic_before_warmup, critic_parameters
    )
    if warmup_parameter_sync != 0.0:
        raise RuntimeError("warmup optimizer step desynchronized DDP parameters")
    # The formal scheduler starts at LR=0, so the first finite optimizer step
    # intentionally changes no parameter and advances warmup to a positive LR.
    # Gradients above verify critic-only routing; active steps below verify the
    # resulting actor and critic parameter updates.
    if warmup_actor_change != 0.0 or warmup_critic_change != 0.0:
        raise RuntimeError("step-zero warmup unexpectedly changed parameters")
    if not all(float(group["lr"]) > 0.0 for group in optimizer.param_groups):
        raise RuntimeError("warmup scheduler did not advance to a positive LR")
    snapshots.append(snapshot(device, "post_critic_warmup_step"))

    active_updates = []
    for update_index in range(int(args.actor_updates)):
        actor_before = clone_parameters(actor_parameters)
        critic_before = clone_parameters(critic_parameters)
        optimizer.zero_grad(set_to_none=True)
        output = wrapped(rollout, True)
        output["objective"].backward()
        actor_grad = gradient_norm(actor_parameters)
        critic_grad = gradient_norm(critic_parameters)
        gradient_sync = synchronized_gradients(
            all_parameters, rank=rank, device=device
        )
        if not actor_grad > 0.0 or not critic_grad > 0.0:
            raise RuntimeError(
                "each active DDP update must train both actor and critic"
            )
        if gradient_sync != 0.0:
            raise RuntimeError(
                f"active DDP gradients are not synchronized: {gradient_sync}"
            )
        torch.nn.utils.clip_grad_norm_(all_parameters, float(rl.clip_grad_norm))
        optimizer.step()
        scheduler.step()
        parameter_sync = cross_rank_max_difference(
            all_parameters, rank=rank, device=device
        )
        actor_change = maximum_parameter_change(actor_before, actor_parameters)
        critic_change = maximum_parameter_change(critic_before, critic_parameters)
        if parameter_sync != 0.0:
            raise RuntimeError("active optimizer step desynchronized DDP parameters")
        if not actor_change > 0.0 or not critic_change > 0.0:
            raise RuntimeError("active DDP step failed to change actor and critic")
        active_updates.append(
            {
                "index": update_index,
                "global_mean_actor_loss": reduced_mean(output["actor_loss"]),
                "global_mean_value_loss": reduced_mean(output["value_loss"]),
                "global_mean_ratio_deviation": reduced_mean(
                    output["ratio_deviation"]
                ),
                "global_mean_clip_fraction": reduced_mean(
                    output["clip_fraction"]
                ),
                "actor_grad_norm": actor_grad,
                "critic_grad_norm": critic_grad,
                "gradient_sync_max_abs_diff": gradient_sync,
                "parameter_sync_max_abs_diff": parameter_sync,
                "actor_max_abs_change": actor_change,
                "critic_max_abs_change": critic_change,
            }
        )
        snapshots.append(snapshot(device, f"post_actor_critic_step_{update_index}"))

    model.llm.forward = original_llm_forward
    if llm_calls != 0:
        raise RuntimeError("Stage-2 DDP head updates called Qwen")
    if not active_updates[-1]["global_mean_ratio_deviation"] > 0.0:
        raise RuntimeError("second active DDP update did not observe a nonunit ratio")

    peak_reserved = float(torch.cuda.max_memory_reserved(device) / MIB)
    total_mib = snapshots[0]["total_mib"]
    max_external = max(
        item["external_or_nonallocator_mib"] for item in snapshots
    )
    effective_headroom = total_mib - peak_reserved - max_external
    minimum_free = min(item["free_mib"] for item in snapshots)
    failures = []
    if MIN_HEADROOM_MIB > 0.0 and effective_headroom < MIN_HEADROOM_MIB:
        failures.append(
            f"effective_headroom={effective_headroom:.1f}<{MIN_HEADROOM_MIB:.1f}"
        )
    if MIN_HEADROOM_MIB > 0.0 and minimum_free < MIN_HEADROOM_MIB:
        failures.append(
            f"minimum_free={minimum_free:.1f}<{MIN_HEADROOM_MIB:.1f}"
        )
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "dataset_index": dataset_index,
        "source_id": int(row["source_id"]),
        "terminal_rewards": rollout["terminal_rewards"].cpu().tolist(),
        "warmup_actor_grad_norm": warmup_actor_grad,
        "warmup_critic_grad_norm": warmup_critic_grad,
        "warmup_gradient_sync_max_abs_diff": warmup_gradient_sync,
        "warmup_parameter_sync_max_abs_diff": warmup_parameter_sync,
        "warmup_actor_max_abs_change": warmup_actor_change,
        "warmup_critic_max_abs_change": warmup_critic_change,
        "active_updates": active_updates,
        "head_only_qwen_forward_calls": llm_calls,
        "peak_reserved_mib": peak_reserved,
        "max_external_or_nonallocator_mib": max_external,
        "effective_peak_headroom_mib": effective_headroom,
        "minimum_observed_free_mib": minimum_free,
        "memory_snapshots": snapshots,
        "failures": failures,
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    passed = all(not item["failures"] for item in gathered)
    if rank == 0:
        report = {
            "status": "PASS" if passed else "FAIL",
            "scope": (
                "real_four_rank_trace_vb_stage2_ddp_smoke_"
                "not_an_experiment_result"
            ),
            "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
            "world_size": world_size,
            "group_size_per_rank": 8,
            "global_rollout_paths": 8 * world_size,
            "terminal_exact_reward_only": True,
            "rollout_float32_fields": list(FLOAT32_ROLLOUT_FIELDS),
            "value_bridge_exact_copy": True,
            "critic_warmup_backward_steps_per_rank": 1,
            "actor_critic_backward_steps_per_rank": int(args.actor_updates),
            "formal_critic_warmup_rollout_batches": int(
                rl.critic_warmup_batches
            ),
            "semantic_anchor_configured": True,
            "solve_text_decoder_frozen": True,
            "nonunit_ratio_observed_on_later_update": True,
            "ddp_gradient_and_parameter_sync_exact": True,
            "head_only_qwen_forward_calls": 0,
            "trainable_parameter_names": names,
            "allowed_trainable_prefixes": list(allowed_prefixes),
            "minimum_effective_and_observed_headroom_mib": MIN_HEADROOM_MIB,
            "ranks": gathered,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    del rollout, wrapped, proxy, model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()
    if not passed:
        raise RuntimeError("TRACE-VB Stage-2 four-rank DDP gate failed")


if __name__ == "__main__":
    main()
