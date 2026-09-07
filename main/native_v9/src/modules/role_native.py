"""Minimal role-native latent policy and process-credit utilities.

The module is deliberately independent from every historical TRACE
checkpoint.  It defines the eight roles used by the native model, constructs
label-preserving targets from the gold textual CoT, and supplies the Gaussian
policy quantities required by Stage-2 latent PPO.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_NAMES: Tuple[str, ...] = (
    "PLAN",
    "SOLVE1",
    "SOLVE2",
    "SOLVE3",
    "SOLVE4",
    "SOLVE5",
    "REFINE",
    "COMMIT",
)
N_ROLES = len(ROLE_NAMES)
COMMIT_INDEX = 7


def contiguous_spans(length: int, chunks: int = 5) -> Tuple[Tuple[int, int], ...]:
    """Partition an observed sequence once, in order, without inferred labels."""
    if length <= 0:
        raise ValueError("a gold CoT must contain at least one step")
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    base, remainder = divmod(length, chunks)
    spans: List[Tuple[int, int]] = []
    start = 0
    for index in range(chunks):
        width = base + int(index < remainder)
        end = start + width
        spans.append((start, end))
        start = end
    if start != length:
        raise RuntimeError("contiguous partition did not preserve the CoT")
    return tuple(spans)


@dataclass
class RoleTargets:
    solve: torch.Tensor
    solve_mask: torch.Tensor
    refine: torch.Tensor
    commit: torch.Tensor


def build_role_targets(
    explicit_features: Sequence[Dict[str, torch.Tensor]],
) -> RoleTargets:
    """Build PLAN/SOLVE/REFINE/COMMIT supervision from existing gold CoTs.

    No step type, hard negative, counterfactual edit, or extra annotation is
    inferred.  Five SOLVE targets are balanced contiguous sums of the text-CoT
    state transitions. REFINE reuses the actual final textual transition and
    COMMIT uses the final textual-CoT state.
    """
    solve_rows = []
    mask_rows = []
    refine_rows = []
    commit_rows = []
    for item in explicit_features:
        residuals = item["step_residuals"].float().detach()
        states = item["step_states"].float().detach()
        if residuals.ndim != 2 or states.ndim != 2:
            raise ValueError("gold CoT states must have [step, hidden] shape")
        if residuals.shape != states.shape or residuals.shape[0] <= 0:
            raise ValueError("gold CoT states and residuals must align")
        targets = residuals.new_zeros(5, residuals.shape[-1])
        mask = torch.zeros(5, device=residuals.device, dtype=torch.bool)
        for index, (start, end) in enumerate(contiguous_spans(residuals.shape[0])):
            if end > start:
                targets[index] = residuals[start:end].sum(dim=0)
                mask[index] = True
        solve_rows.append(targets)
        mask_rows.append(mask)
        refine_rows.append(residuals[-1])
        commit_rows.append(states[-1])
    return RoleTargets(
        solve=torch.stack(solve_rows, dim=0),
        solve_mask=torch.stack(mask_rows, dim=0),
        refine=torch.stack(refine_rows, dim=0),
        commit=torch.stack(commit_rows, dim=0),
    )


def masked_cosine_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    if mask.shape != predictions.shape[:-1]:
        raise ValueError("mask must match all non-hidden dimensions")
    scores = F.cosine_similarity(predictions.float(), targets.float(), dim=-1)
    weights = mask.to(scores.dtype)
    return ((1.0 - scores) * weights).sum() / weights.sum().clamp_min(1.0)


def gaussian_log_prob(
    actions: torch.Tensor,
    means: torch.Tensor,
    log_stds: torch.Tensor,
) -> torch.Tensor:
    if actions.shape != means.shape or means.shape != log_stds.shape:
        raise ValueError("Gaussian tensors must have identical shapes")
    values = -0.5 * (
        ((actions - means) * torch.exp(-log_stds)).square()
        + 2.0 * log_stds
        + math.log(2.0 * math.pi)
    )
    return values.sum(dim=-1)


def diagonal_gaussian_kl(
    means: torch.Tensor,
    log_stds: torch.Tensor,
    reference_means: torch.Tensor,
    reference_log_stds: torch.Tensor,
) -> torch.Tensor:
    variance_ratio = torch.exp(2.0 * (log_stds - reference_log_stds))
    mean_term = (
        (means - reference_means).square()
        * torch.exp(-2.0 * reference_log_stds)
    )
    return (
        reference_log_stds
        - log_stds
        + 0.5 * (variance_ratio + mean_term)
        - 0.5
    ).mean(dim=-1)


def clipped_policy_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    if not (
        current_log_probs.shape
        == old_log_probs.shape
        == advantages.shape
        == mask.shape
    ):
        raise ValueError("policy tensors must have identical shapes")
    ratio = torch.exp((current_log_probs - old_log_probs).clamp(-20.0, 20.0))
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    weights = mask.to(unclipped.dtype)
    return -(
        torch.minimum(unclipped, clipped) * weights
    ).sum() / weights.sum().clamp_min(1.0)


def group_standardize(values: torch.Tensor, group_size: int) -> torch.Tensor:
    """Standardize independent rollout groups without cross-question leakage."""
    if values.shape[0] % group_size != 0:
        raise ValueError("rollout count must be divisible by group_size")
    grouped = values.reshape(-1, group_size, *values.shape[1:])
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    return ((grouped - mean) / std).reshape_as(values)


def discounted_role_returns(step_scores: torch.Tensor, gamma: float) -> torch.Tensor:
    """Return-to-go for seven stochastic roles; COMMIT remains deterministic."""
    if step_scores.ndim != 2 or step_scores.shape[1] != N_ROLES:
        raise ValueError("step scores must have shape [rollout, 8]")
    returns = torch.zeros_like(step_scores)
    running = torch.zeros_like(step_scores[:, 0])
    for index in range(COMMIT_INDEX - 1, -1, -1):
        running = step_scores[:, index] + float(gamma) * running
        returns[:, index] = running
    return returns


class RoleLatentPolicy(nn.Module):
    """Low-dimensional role policy added inside the proven BRIDGE path."""

    def __init__(
        self,
        hidden_size: int,
        action_dim: int = 16,
        policy_hidden_size: int = 512,
        role_embedding_size: int = 64,
        initial_log_std: float = -0.7,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.role_embedding = nn.Embedding(N_ROLES, int(role_embedding_size))
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.hidden_size + int(role_embedding_size)),
            nn.Linear(self.hidden_size + int(role_embedding_size), policy_hidden_size),
            nn.GELU(),
            nn.Linear(policy_hidden_size, policy_hidden_size),
            nn.LayerNorm(policy_hidden_size),
        )
        # Five SOLVE positions share semantics but retain different role embeds.
        self.mean_heads = nn.ModuleDict(
            {
                key: nn.Linear(policy_hidden_size, self.action_dim)
                for key in ("plan", "solve", "refine", "commit")
            }
        )
        self.log_std_heads = nn.ModuleDict(
            {
                key: nn.Linear(policy_hidden_size, self.action_dim)
                for key in ("plan", "solve", "refine")
            }
        )
        self.action_projector = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_size),
            nn.Tanh(),
        )
        for head in self.mean_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.log_std_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, float(initial_log_std))
        nn.init.normal_(
            self.action_projector[0].weight,
            std=1.0 / math.sqrt(self.action_dim),
        )
        nn.init.zeros_(self.action_projector[0].bias)

    @staticmethod
    def head_key(role_index: int) -> str:
        if role_index == 0:
            return "plan"
        if 1 <= role_index <= 5:
            return "solve"
        if role_index == 6:
            return "refine"
        if role_index == COMMIT_INDEX:
            return "commit"
        raise ValueError("role index must be in [0, 7]")

    def distribution(
        self,
        states: torch.Tensor,
        role_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        role_ids = torch.full(
            (states.shape[0],),
            int(role_index),
            device=states.device,
            dtype=torch.long,
        )
        features = self.trunk(
            torch.cat([states.float(), self.role_embedding(role_ids).float()], dim=-1)
        )
        key = self.head_key(role_index)
        mean = self.mean_heads[key](features).clamp(-20.0, 20.0)
        if role_index == COMMIT_INDEX:
            log_std = torch.zeros_like(mean)
        else:
            log_std = self.log_std_heads[key](features).clamp(-2.5, 0.5)
        return mean, log_std

    def realize(
        self,
        states: torch.Tensor,
        role_index: int,
        *,
        deterministic: bool,
        forced_action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(states, role_index)
        if forced_action is not None:
            action = forced_action.float()
        elif role_index == COMMIT_INDEX or deterministic:
            action = mean
        else:
            action = mean + torch.exp(log_std) * torch.randn_like(mean)
        log_prob = (
            torch.zeros(action.shape[0], device=action.device, dtype=torch.float32)
            if role_index == COMMIT_INDEX
            else gaussian_log_prob(action.float(), mean.float(), log_std.float())
        )
        return action, log_prob, mean, log_std

