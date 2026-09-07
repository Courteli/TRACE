import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# TRACE uses one fixed, auditable latent program. SOLVE occupies five ordered
# slots but shares one functional policy head; its step embedding carries only
# within-role progress. COMMIT is a conditional-mean transition and therefore
# is deliberately absent from stochastic-policy objectives.
TRACE_N_ROLE_STEPS = 8
TRACE_PLAN_INDEX = 0
TRACE_SOLVE_START = 1
TRACE_SOLVE_END = 6
TRACE_N_SOLVE_STEPS = TRACE_SOLVE_END - TRACE_SOLVE_START
TRACE_CHECK_INDEX = 6
TRACE_COMMIT_INDEX = 7
TRACE_ROLE_NAMES: Tuple[str, ...] = (
    "PLAN",
    "SOLVE",
    "SOLVE",
    "SOLVE",
    "SOLVE",
    "SOLVE",
    "CHECK",
    "COMMIT",
)
TRACE_ROLE_HEAD_KEYS: Tuple[str, ...] = tuple(
    role.lower() for role in TRACE_ROLE_NAMES
)
TRACE_STOCHASTIC_ROLE_HEAD_KEYS: Tuple[str, ...] = (
    "plan",
    "solve",
    "check",
)
TRACE_MEAN_ROLE_HEAD_KEYS: Tuple[str, ...] = (
    *TRACE_STOCHASTIC_ROLE_HEAD_KEYS,
    "commit",
)


def validate_trace_role_schema(n_steps: int) -> None:
    """Reject configurations that silently change the fixed latent program."""
    if int(n_steps) != TRACE_N_ROLE_STEPS:
        raise ValueError(
            "role-semantic TRACE requires exactly eight transitions: "
            "PLAN, five SOLVE, CHECK, COMMIT; got "
            f"n_steps={n_steps}"
        )


def trace_role_name(step_index: int) -> str:
    """Return the public upper-case role name for one trajectory position."""
    if not 0 <= int(step_index) < TRACE_N_ROLE_STEPS:
        raise ValueError(
            f"step_index={step_index} is outside the eight-step role schema"
        )
    return TRACE_ROLE_NAMES[int(step_index)]


def trace_role_head_key(step_index: int) -> str:
    """Map five ordered SOLVE positions to their one shared policy head."""
    trace_role_name(step_index)
    return TRACE_ROLE_HEAD_KEYS[int(step_index)]


def trace_stochastic_action_mask(
    batch_size: Optional[int] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Return the seven stochastic slots and deterministic COMMIT mask."""
    mask = torch.ones(
        TRACE_N_ROLE_STEPS,
        device=device,
        dtype=dtype,
    )
    mask[TRACE_COMMIT_INDEX] = 0
    if batch_size is None:
        return mask
    if int(batch_size) < 0:
        raise ValueError("batch_size must be non-negative")
    return mask.unsqueeze(0).expand(int(batch_size), -1)


@dataclass(frozen=True)
class ContiguousCoTTargets:
    """Five monotone CoT residual targets and their non-padding mask."""

    targets: torch.Tensor
    mask: torch.Tensor
    spans: Tuple[Tuple[int, int], ...]
    solve_slot_indices: Tuple[int, ...]


def solve_slot_indices(
    n_cot_steps: int,
    *,
    n_solve_slots: int = TRACE_N_SOLVE_STEPS,
) -> Tuple[int, ...]:
    """Place short CoTs across SOLVE while anchoring the final step at the end."""
    if int(n_cot_steps) <= 0:
        raise ValueError("a CoT must contain at least one observed step")
    if int(n_solve_slots) <= 0:
        raise ValueError("n_solve_slots must be positive")
    active = min(int(n_cot_steps), int(n_solve_slots))
    if active == int(n_solve_slots):
        return tuple(range(int(n_solve_slots)))
    if active == 1:
        return (int(n_solve_slots) - 1,)
    denominator = active - 1
    final_slot = int(n_solve_slots) - 1
    return tuple(
        (index * final_slot + denominator // 2) // denominator
        for index in range(active)
    )


def contiguous_cot_chunk_spans(
    n_cot_steps: int,
    *,
    n_chunks: int = TRACE_N_SOLVE_STEPS,
) -> Tuple[Tuple[int, int], ...]:
    """Split CoT steps into balanced, contiguous chunks without relabeling.

    When the CoT is shorter than the requested chunk count, observed steps are
    spread monotonically over the available slots, with the final observed
    step anchored to the final SOLVE slot. Empty slots are masked rather than
    filled with invented supervision.
    """
    if int(n_cot_steps) <= 0:
        raise ValueError("a CoT must contain at least one observed step")
    if int(n_chunks) <= 0:
        raise ValueError("n_chunks must be positive")
    slots = solve_slot_indices(
        int(n_cot_steps),
        n_solve_slots=int(n_chunks),
    )
    spans = []
    start = 0
    if int(n_cot_steps) < int(n_chunks):
        active_slots = set(slots)
        for chunk_index in range(int(n_chunks)):
            width = int(chunk_index in active_slots)
            end = start + width
            spans.append((start, end))
            start = end
    else:
        base, remainder = divmod(int(n_cot_steps), int(n_chunks))
        for chunk_index in range(int(n_chunks)):
            width = base + int(chunk_index < remainder)
            end = start + width
            spans.append((start, end))
            start = end
    if start != int(n_cot_steps):
        raise RuntimeError("contiguous CoT partition did not cover every step")
    return tuple(spans)


def build_contiguous_cot_targets(
    step_residuals: torch.Tensor,
    *,
    n_chunks: int = TRACE_N_SOLVE_STEPS,
) -> ContiguousCoTTargets:
    """Aggregate a [CoT-step, hidden] sequence into five SOLVE targets.

    Summing adjacent state residuals gives the exact endpoint displacement of
    each observed chunk. It preserves order and needs neither learned
    alignment, dependency graph, nor synthetic role annotation.
    """
    if step_residuals.ndim != 2:
        raise ValueError(
            "step_residuals must have shape [CoT-step, hidden], got "
            f"{tuple(step_residuals.shape)}"
        )
    if step_residuals.shape[0] <= 0 or step_residuals.shape[1] <= 0:
        raise ValueError("step_residuals must have non-empty step and hidden axes")
    spans = contiguous_cot_chunk_spans(
        int(step_residuals.shape[0]),
        n_chunks=int(n_chunks),
    )
    targets = step_residuals.new_zeros(
        int(n_chunks),
        step_residuals.shape[-1],
    )
    mask = torch.zeros(
        int(n_chunks),
        device=step_residuals.device,
        dtype=torch.bool,
    )
    for chunk_index, (start, end) in enumerate(spans):
        if end <= start:
            continue
        targets[chunk_index] = step_residuals[start:end].sum(dim=0)
        mask[chunk_index] = True
    return ContiguousCoTTargets(
        targets=targets,
        mask=mask,
        spans=spans,
        solve_slot_indices=solve_slot_indices(
            int(step_residuals.shape[0]),
            n_solve_slots=int(n_chunks),
        ),
    )


def masked_cosine_similarity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return per-position cosine scores with inactive positions set to zero."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    if predictions.ndim < 2:
        raise ValueError("predictions must include position and hidden axes")
    expected_mask_shape = predictions.shape[:-1]
    if tuple(mask.shape) != tuple(expected_mask_shape):
        raise ValueError(
            "mask must match predictions without the hidden axis: expected "
            f"{tuple(expected_mask_shape)}, got {tuple(mask.shape)}"
        )
    scores = F.cosine_similarity(
        predictions.float(),
        targets.float(),
        dim=-1,
        eps=float(eps),
    )
    return scores * mask.to(device=scores.device, dtype=scores.dtype)


def masked_cosine_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Average one-minus-cosine only over observed semantic targets."""
    scores = masked_cosine_similarity(
        predictions,
        targets,
        mask,
        eps=eps,
    )
    weights = mask.to(device=scores.device, dtype=scores.dtype)
    return ((1.0 - scores) * weights).sum() / weights.sum().clamp_min(1.0)


def gaussian_log_prob(
    actions: torch.Tensor,
    means: torch.Tensor,
    log_stds: torch.Tensor,
) -> torch.Tensor:
    """Joint diagonal-Gaussian log probability for one latent action."""
    if actions.shape != means.shape or actions.shape != log_stds.shape:
        raise ValueError("actions, means, and log_stds must have identical shapes")
    inverse_variance = torch.exp(-2.0 * log_stds)
    elementwise = -0.5 * (
        (actions - means).square() * inverse_variance
        + 2.0 * log_stds
        + math.log(2.0 * math.pi)
    )
    return elementwise.sum(dim=-1)


def diagonal_gaussian_kl(
    means: torch.Tensor,
    log_stds: torch.Tensor,
    reference_means: torch.Tensor,
    reference_log_stds: torch.Tensor,
) -> torch.Tensor:
    """KL[current || Stage-1 reference], averaged over action dimensions."""
    if not (
        means.shape
        == log_stds.shape
        == reference_means.shape
        == reference_log_stds.shape
    ):
        raise ValueError("all Gaussian parameter tensors must have identical shapes")
    variance_ratio = torch.exp(
        2.0 * (log_stds - reference_log_stds)
    )
    mean_term = (
        (means - reference_means).square()
        * torch.exp(-2.0 * reference_log_stds)
    )
    kl = (
        reference_log_stds
        - log_stds
        + 0.5 * (variance_ratio + mean_term)
        - 0.5
    )
    return kl.mean(dim=-1)


def sampled_forward_kl(
    current_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    max_log_ratio: float = 10.0,
) -> torch.Tensor:
    """Estimate KL[current || reference] on tokens sampled from current."""
    if current_log_probs.shape != reference_log_probs.shape:
        raise ValueError(
            "current and reference token log probabilities must match"
        )
    if mask is not None and mask.shape != current_log_probs.shape:
        raise ValueError("mask must match token log probabilities")
    log_reference_ratio = (
        reference_log_probs - current_log_probs
    ).clamp(
        min=-float(max_log_ratio),
        max=float(max_log_ratio),
    )
    per_token = torch.expm1(log_reference_ratio) - log_reference_ratio
    if mask is None:
        return per_token.mean()
    weights = mask.to(dtype=per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


class GaussianTrajectoryPolicy(nn.Module):
    """Role-conditioned low-dimensional policy over one latent program.

    PLAN, SOLVE, and CHECK have distinct Gaussian heads. All five ordered
    SOLVE transitions share one head, while their step embeddings retain
    causal progress. COMMIT uses its conditional mean and is never a sampled
    policy action.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        n_steps: int,
        policy_hidden_size: int,
        step_embedding_size: int = 64,
        initial_log_std: float = -0.7,
        min_log_std: float = -2.5,
        max_log_std: float = 0.5,
    ):
        super().__init__()
        if hidden_size <= 0 or action_dim <= 0:
            raise ValueError("hidden_size and action_dim must be positive")
        validate_trace_role_schema(n_steps)
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.n_steps = int(n_steps)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.register_buffer(
            "stochastic_action_mask_template",
            trace_stochastic_action_mask(),
            persistent=False,
        )

        self.policy_step_embedding = nn.Embedding(
            self.n_steps,
            int(step_embedding_size),
        )
        self.policy_trunk = nn.Sequential(
            nn.LayerNorm(self.hidden_size + int(step_embedding_size)),
            nn.Linear(
                self.hidden_size + int(step_embedding_size),
                int(policy_hidden_size),
            ),
            nn.GELU(),
            nn.Linear(int(policy_hidden_size), int(policy_hidden_size)),
            nn.LayerNorm(int(policy_hidden_size)),
        )
        self.mean_heads = nn.ModuleDict(
            {
                role: nn.Linear(int(policy_hidden_size), self.action_dim)
                for role in TRACE_MEAN_ROLE_HEAD_KEYS
            }
        )
        self.log_std_heads = nn.ModuleDict(
            {
                role: nn.Linear(int(policy_hidden_size), self.action_dim)
                for role in TRACE_STOCHASTIC_ROLE_HEAD_KEYS
            }
        )

        self.dynamics_step_embedding = nn.Embedding(
            self.n_steps,
            self.hidden_size,
        )
        self.base_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.action_projector = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_size),
            nn.Tanh(),
        )

        for role in TRACE_MEAN_ROLE_HEAD_KEYS:
            nn.init.zeros_(self.mean_heads[role].weight)
            nn.init.zeros_(self.mean_heads[role].bias)
        for role in TRACE_STOCHASTIC_ROLE_HEAD_KEYS:
            nn.init.zeros_(self.log_std_heads[role].weight)
            nn.init.constant_(
                self.log_std_heads[role].bias,
                float(initial_log_std),
            )
        nn.init.normal_(
            self.dynamics_step_embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.hidden_size),
        )
        nn.init.zeros_(self.base_projector[-1].weight)
        nn.init.zeros_(self.base_projector[-1].bias)
        final = self.action_projector[0]
        nn.init.normal_(
            final.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.action_dim),
        )
        nn.init.zeros_(final.bias)

    def distribution_parameters(
        self,
        states: torch.Tensor,
        step_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 2 or states.shape[-1] != self.hidden_size:
            raise ValueError(
                "states must have shape [batch, hidden_size], got "
                f"{tuple(states.shape)}"
            )
        if not 0 <= int(step_index) < self.n_steps:
            raise ValueError(f"step_index={step_index} is outside the policy")
        step_ids = torch.full(
            (states.shape[0],),
            fill_value=int(step_index),
            device=states.device,
            dtype=torch.long,
        )
        policy_states = states.float()
        step_features = self.policy_step_embedding(step_ids).float()
        hidden = self.policy_trunk(
            torch.cat([policy_states, step_features], dim=-1)
        )
        role = trace_role_head_key(step_index)
        means = torch.nan_to_num(
            self.mean_heads[role](hidden),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(min=-20.0, max=20.0)
        if role == "commit":
            # Shape-compatible placeholder; COMMIT never samples from it and
            # is excluded from entropy, KL, and policy-ratio objectives.
            log_stds = torch.zeros_like(means)
        else:
            log_stds = torch.nan_to_num(
                self.log_std_heads[role](hidden),
                nan=self.min_log_std,
                posinf=self.max_log_std,
                neginf=self.min_log_std,
            ).clamp(self.min_log_std, self.max_log_std)
        return means, log_stds

    @staticmethod
    def is_stochastic_step(step_index: int) -> bool:
        """Return false only for the conditional-mean COMMIT transition."""
        trace_role_name(step_index)
        return int(step_index) != TRACE_COMMIT_INDEX

    def stochastic_action_mask(
        self,
        batch_size: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bool,
    ) -> torch.Tensor:
        """Expose the action mask on the caller's device and dtype."""
        target_device = (
            self.stochastic_action_mask_template.device
            if device is None
            else device
        )
        return trace_stochastic_action_mask(
            batch_size,
            device=target_device,
            dtype=dtype,
        )

    def realize_action(
        self,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        step_index: int,
        *,
        deterministic: bool = False,
        innovation: Optional[torch.Tensor] = None,
        forced_actions: Optional[torch.Tensor] = None,
        forced_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Realize one action, making COMMIT conditionally deterministic.

        Returns action, realized innovation, and joint action log probability.
        A forced COMMIT is intentionally ignored so suffix interventions cannot
        turn the summary transition back into a stochastic policy action.
        """
        if means.ndim != 2:
            raise ValueError(
                "means must have shape "
                f"[batch, {self.action_dim}], got {tuple(means.shape)}"
            )
        expected = (means.shape[0], self.action_dim)
        if tuple(means.shape) != expected:
            raise ValueError(
                "means must have shape "
                f"[batch, {self.action_dim}], got {tuple(means.shape)}"
            )
        if log_stds.shape != means.shape:
            raise ValueError("means and log_stds must have identical shapes")
        if not self.is_stochastic_step(step_index):
            zeros = torch.zeros_like(means)
            return means, zeros, means.new_zeros(means.shape[0])

        if innovation is None:
            epsilon = (
                torch.zeros_like(means)
                if deterministic
                else torch.randn_like(means)
            )
        else:
            if innovation.shape != means.shape:
                raise ValueError("innovation must match action parameters")
            epsilon = innovation.to(device=means.device, dtype=means.dtype)
        std = torch.exp(log_stds)
        sampled_action = means + std * epsilon

        if forced_actions is None:
            action = sampled_action
            realized_epsilon = epsilon
        else:
            if forced_actions.shape != means.shape:
                raise ValueError("forced_actions must match action parameters")
            if forced_mask is None:
                mask = torch.ones(
                    means.shape[0],
                    1,
                    device=means.device,
                    dtype=torch.bool,
                )
            else:
                if forced_mask.ndim == 1:
                    forced_mask = forced_mask.unsqueeze(-1)
                if tuple(forced_mask.shape) != (means.shape[0], 1):
                    raise ValueError(
                        "forced_mask must have shape [batch] or [batch, 1]"
                    )
                mask = forced_mask.to(device=means.device, dtype=torch.bool)
            action = torch.where(
                mask,
                forced_actions.to(device=means.device, dtype=means.dtype).detach(),
                sampled_action,
            )
            realized_epsilon = torch.where(
                mask,
                (action - means) / std.clamp_min(1e-8),
                epsilon,
            )
        return (
            action,
            realized_epsilon,
            gaussian_log_prob(action, means, log_stds),
        )

    def latent_input(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        step_index: int,
        *,
        action_scale: float,
        step_scale: float,
    ) -> torch.Tensor:
        if actions.shape != (states.shape[0], self.action_dim):
            raise ValueError(
                "actions must have shape "
                f"[batch, {self.action_dim}], got {tuple(actions.shape)}"
            )
        step_ids = torch.full(
            (states.shape[0],),
            fill_value=int(step_index),
            device=states.device,
            dtype=torch.long,
        )
        # Start from a stable hidden-state recurrence. The learned projector is
        # a residual correction, so a fresh TRACE policy does not destroy the
        # CoT-SFT representation before Stage 1 has seen any latent targets.
        base = states.float() + self.base_projector(states.float())
        step = self.dynamics_step_embedding(step_ids).float()
        action = self.action_projector(actions.float())
        return base + float(step_scale) * step + float(action_scale) * action

    def set_stage2_trainability(self):
        """Refine role features and stochastic heads, but freeze dynamics.

        V7 uses role-local rewards, so the shared role trunk and its step
        embedding must be able to separate PLAN/SOLVE/REFINE.  The latent
        transition projectors and the Stage-1 step prior remain frozen to
        protect the capability anchor.  During Stage 2 the model routes
        COMMIT through the immutable Stage-1 reference policy, so opening the
        shared stochastic trunk cannot move the deployment COMMIT action.
        """
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.policy_step_embedding.parameters():
            parameter.requires_grad_(True)
        for parameter in self.policy_trunk.parameters():
            parameter.requires_grad_(True)
        for role in TRACE_STOCHASTIC_ROLE_HEAD_KEYS:
            for parameter in self.mean_heads[role].parameters():
                parameter.requires_grad_(True)
            for parameter in self.log_std_heads[role].parameters():
                parameter.requires_grad_(True)


class CoTConditionedTrajectoryPosterior(nn.Module):
    """Training-only role posterior conditioned on one observed gold CoT.

    It predicts role-specific residuals over the question-only prior in the
    same action space. The five SOLVE transitions share one residual head.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int,
        n_steps: int,
        posterior_hidden_size: int,
        step_embedding_size: int = 64,
        min_log_std: float = -1.5,
        max_log_std: float = 0.5,
    ):
        super().__init__()
        if hidden_size <= 0 or action_dim <= 0:
            raise ValueError("hidden_size and action_dim must be positive")
        validate_trace_role_schema(n_steps)
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.n_steps = int(n_steps)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.step_embedding = nn.Embedding(
            self.n_steps,
            int(step_embedding_size),
        )
        input_size = (
            2 * self.hidden_size + int(step_embedding_size)
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, int(posterior_hidden_size)),
            nn.GELU(),
            nn.Linear(
                int(posterior_hidden_size),
                int(posterior_hidden_size),
            ),
            nn.LayerNorm(int(posterior_hidden_size)),
        )
        self.mean_delta_heads = nn.ModuleDict(
            {
                role: nn.Linear(int(posterior_hidden_size), self.action_dim)
                for role in TRACE_MEAN_ROLE_HEAD_KEYS
            }
        )
        self.log_std_delta_heads = nn.ModuleDict(
            {
                role: nn.Linear(int(posterior_hidden_size), self.action_dim)
                for role in TRACE_STOCHASTIC_ROLE_HEAD_KEYS
            }
        )
        for role in TRACE_MEAN_ROLE_HEAD_KEYS:
            nn.init.zeros_(self.mean_delta_heads[role].weight)
            nn.init.zeros_(self.mean_delta_heads[role].bias)
        for role in TRACE_STOCHASTIC_ROLE_HEAD_KEYS:
            nn.init.zeros_(self.log_std_delta_heads[role].weight)
            nn.init.zeros_(self.log_std_delta_heads[role].bias)

    def distribution_parameters(
        self,
        states: torch.Tensor,
        cot_context: torch.Tensor,
        step_index: int,
        prior_means: torch.Tensor,
        prior_log_stds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states.shape != cot_context.shape:
            raise ValueError("states and CoT context must have equal shapes")
        if states.ndim != 2 or states.shape[-1] != self.hidden_size:
            raise ValueError(
                "posterior states must have shape [batch, hidden_size]"
            )
        if prior_means.shape != (states.shape[0], self.action_dim):
            raise ValueError("prior action parameters have invalid shapes")
        if prior_log_stds.shape != prior_means.shape:
            raise ValueError("prior means and log stds must align")
        trace_role_name(step_index)
        step_ids = torch.full(
            (states.shape[0],),
            int(step_index),
            device=states.device,
            dtype=torch.long,
        )
        hidden = self.trunk(
            torch.cat(
                [
                    states.float(),
                    cot_context.float(),
                    self.step_embedding(step_ids).float(),
                ],
                dim=-1,
            )
        )
        role = trace_role_head_key(step_index)
        means = (
            prior_means.detach() + self.mean_delta_heads[role](hidden)
        )
        if role == "commit":
            log_stds = torch.zeros_like(means)
        else:
            log_stds = (
                prior_log_stds.detach()
                + self.log_std_delta_heads[role](hidden)
            ).clamp(self.min_log_std, self.max_log_std)
        return means, log_stds


def monotone_progress_centers(
    progress_logits: torch.Tensor,
    noise: torch.Tensor,
    *,
    noise_scale: float,
    minimum_increment: float = 1e-4,
) -> torch.Tensor:
    """Sample strictly ordered progress centers with a reparameterized noise."""
    if progress_logits.shape != noise.shape:
        raise ValueError("progress_logits and noise must have identical shapes")
    increments = F.softplus(
        progress_logits + float(noise_scale) * noise
    ) + float(minimum_increment)
    cumulative = increments.cumsum(dim=-1)
    return cumulative / cumulative[..., -1:].clamp_min(1e-8)


def action_conditioned_progress_centers(
    actions: torch.Tensor,
    *,
    progress_dim: int,
    action_scale: float,
    minimum_increment: float = 1e-4,
) -> torch.Tensor:
    """Convert each sampled action path into its own ordered progress schedule."""
    if actions.ndim < 2:
        raise ValueError(
            "actions must end with [trajectory_step, action_dimension]"
        )
    action_dim = actions.shape[-1]
    if not 0 <= int(progress_dim) < action_dim:
        raise ValueError(
            f"progress_dim={progress_dim} is invalid for action_dim={action_dim}"
        )
    if not math.isfinite(float(action_scale)) or float(action_scale) == 0.0:
        raise ValueError("action_scale must be finite and nonzero")
    if not math.isfinite(float(minimum_increment)) or minimum_increment <= 0:
        raise ValueError("minimum_increment must be finite and positive")
    progress_logits = (
        actions[..., int(progress_dim)].float() * float(action_scale)
    )
    increments = F.softplus(progress_logits) + float(minimum_increment)
    cumulative = increments.cumsum(dim=-1)
    return cumulative / cumulative[..., -1:].clamp_min(1e-8)


def stochastic_monotone_assignment(
    semantic_scores: torch.Tensor,
    progress_centers: torch.Tensor,
    *,
    sigma: float,
    progress_strength: float,
) -> torch.Tensor:
    """Marginalize semantic scores over globally monotone alignments."""
    if semantic_scores.ndim != 2:
        raise ValueError("semantic_scores must have shape [latent, CoT-step]")
    if progress_centers.shape != semantic_scores.shape[:1]:
        raise ValueError(
            "progress_centers must have one value per latent transition"
        )
    n_steps = semantic_scores.shape[1]
    if n_steps == 1:
        step_progress = semantic_scores.new_ones(1)
    else:
        step_progress = torch.linspace(
            0.0,
            1.0,
            steps=n_steps,
            device=semantic_scores.device,
            dtype=semantic_scores.dtype,
        )
    sigma = max(float(sigma), 1e-4)
    progress_bias = -(
        progress_centers.unsqueeze(-1) - step_progress.unsqueeze(0)
    ).square() / (2.0 * sigma * sigma)
    logits = semantic_scores + float(progress_strength) * progress_bias
    return monotonic_path_marginals(logits)


def monotonic_path_marginals(logits: torch.Tensor) -> torch.Tensor:
    """Forward-backward marginals over nondecreasing step assignments.

    A valid alignment chooses one CoT step per latent transition under
    ``k_1 <= ... <= k_M``. Repeated assignments are allowed, which is required
    when the latent path has more transitions than the explicit rationale.
    """
    if logits.ndim != 2 or min(logits.shape) <= 0:
        raise ValueError("logits must have shape [latent, CoT-step]")
    n_latents = logits.shape[0]

    forward = [logits[0]]
    for latent_index in range(1, n_latents):
        forward.append(
            logits[latent_index]
            + torch.logcumsumexp(forward[-1], dim=-1)
        )
    forward = torch.stack(forward, dim=0)

    backward = [torch.zeros_like(logits[-1])]
    for latent_index in range(n_latents - 2, -1, -1):
        continuation = logits[latent_index + 1] + backward[-1]
        suffix_logsumexp = torch.flip(
            torch.logcumsumexp(
                torch.flip(continuation, dims=[-1]),
                dim=-1,
            ),
            dims=[-1],
        )
        backward.append(suffix_logsumexp)
    backward = torch.stack(list(reversed(backward)), dim=0)

    log_partition = torch.logsumexp(forward[-1], dim=-1)
    marginals = torch.exp(forward + backward - log_partition)
    return marginals / marginals.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _safe_cosine_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    distance = 1.0 - F.cosine_similarity(
        left,
        right,
        dim=-1,
        eps=eps,
    )
    both_zero = (left_norm <= eps) & (right_norm <= eps)
    one_zero = (left_norm <= eps) ^ (right_norm <= eps)
    distance = torch.where(both_zero, torch.zeros_like(distance), distance)
    distance = torch.where(one_zero, torch.ones_like(distance), distance)
    return torch.nan_to_num(
        distance,
        nan=0.0,
        posinf=2.0,
        neginf=0.0,
    ).clamp(min=0.0, max=2.0)


def trajectory_distance_components(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    anchor_count: int = 3,
    position_weight: float = 0.45,
    direction_weight: float = 0.35,
    step_weight: float = 0.15,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Position, direction, and step distance over complete residual paths."""
    if left.ndim < 2 or right.ndim < 2:
        raise ValueError("paths must end in [transition, hidden]")
    if left.shape[-2:] != right.shape[-2:]:
        raise ValueError("path transition and hidden dimensions must match")
    left = left.float()
    right = right.float()
    n_steps = left.shape[-2]
    anchor_count = max(1, min(int(anchor_count), n_steps))
    fractions = torch.linspace(
        1.0 / anchor_count,
        1.0,
        steps=anchor_count,
        device=left.device,
    )
    anchors = torch.round((n_steps - 1) * fractions).long().unique(
        sorted=True
    )
    left_positions = left.cumsum(dim=-2).index_select(-2, anchors)
    right_positions = right.cumsum(dim=-2).index_select(-2, anchors)
    position = _safe_cosine_distance(
        left_positions,
        right_positions,
        eps=eps,
    ).mean(dim=-1)
    direction = _safe_cosine_distance(left, right, eps=eps).mean(dim=-1)
    scale = math.sqrt(left.shape[-1])
    left_steps = left.norm(dim=-1) / scale
    right_steps = right.norm(dim=-1) / scale
    step = (
        (left_steps - right_steps).abs()
        / (left_steps + right_steps + eps)
    ).mean(dim=-1)
    total = (
        float(position_weight) * position
        + float(direction_weight) * direction
        + float(step_weight) * step
    )
    return {
        "total": total,
        "position": position,
        "direction": direction,
        "step": step,
    }


def trajectory_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return trajectory_distance_components(left, right, **kwargs)["total"]


def path_noncollapse_loss(
    paths: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if paths.ndim != 4:
        raise ValueError("paths must have shape [batch, path, step, hidden]")
    normalized_norm = paths.float().norm(dim=-1) / math.sqrt(paths.shape[-1])
    return F.relu(float(margin) - normalized_norm).mean()


def minimum_action_entropy_loss(
    log_stds: torch.Tensor,
    *,
    minimum_std: float,
) -> torch.Tensor:
    """Keep every learned action distribution non-degenerate.

    This constrains distributional support, not pairwise path distance. It
    therefore leaves the model free to discover the number and geometry of
    valid modes while preventing a deterministic posterior/prior collapse.
    """
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    minimum_log_std = math.log(float(minimum_std))
    return F.relu(minimum_log_std - log_stds.float()).mean()


def standardized_within_question_actions(
    actions: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Standardize sampled actions across exchangeable paths of one question."""
    if actions.ndim != 4:
        raise ValueError(
            "actions must have shape [batch, path, transition, action]"
        )
    centered = actions.float() - actions.float().mean(
        dim=1,
        keepdim=True,
    )
    scale = centered.square().mean(dim=1, keepdim=True).sqrt()
    return centered / scale.clamp_min(float(eps))


def action_transition_identifiability_loss(
    predicted_actions: torch.Tensor,
    sampled_actions: torch.Tensor,
) -> torch.Tensor:
    """Require sampled policy choices to be recoverable from transitions.

    Targets are standardized only across IID paths of the same question, so
    the criterion tests whether path-to-path action differences survive in
    the latent transitions. It introduces no route identity or target gap.
    """
    if predicted_actions.shape != sampled_actions.shape:
        raise ValueError(
            "predicted and sampled actions must have identical shapes"
        )
    targets = standardized_within_question_actions(
        sampled_actions.detach()
    )
    return F.smooth_l1_loss(predicted_actions.float(), targets)


def pairwise_action_path_correlation(
    actions: torch.Tensor,
    paths: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Diagnostic correlation between action and realized path distances."""
    if actions.ndim != 4 or paths.ndim != 4:
        raise ValueError(
            "actions and paths must have [batch, path, step, feature] shapes"
        )
    if actions.shape[:3] != paths.shape[:3]:
        raise ValueError("actions and paths must share batch/path/step axes")
    if actions.shape[1] < 2:
        return paths.sum() * 0.0
    action_signatures = actions.float().flatten(start_dim=2)
    path_signatures = paths.float().flatten(start_dim=2)
    left, right = torch.triu_indices(
        actions.shape[1],
        actions.shape[1],
        offset=1,
        device=actions.device,
    )
    action_distances = (
        action_signatures[:, left] - action_signatures[:, right]
    ).norm(dim=-1)
    path_distances = (
        path_signatures[:, left] - path_signatures[:, right]
    ).norm(dim=-1)
    action_centered = action_distances - action_distances.mean(
        dim=1,
        keepdim=True,
    )
    path_centered = path_distances - path_distances.mean(
        dim=1,
        keepdim=True,
    )
    numerator = (action_centered * path_centered).sum(dim=1)
    denominator = (
        action_centered.square().sum(dim=1).sqrt()
        * path_centered.square().sum(dim=1).sqrt()
    )
    correlation = numerator / denominator.clamp_min(float(eps))
    valid = denominator > float(eps)
    return torch.where(valid, correlation, torch.zeros_like(correlation)).mean()


@dataclass(frozen=True)
class HardPathPair:
    group_index: int
    correct_index: int
    correct_peer_index: int
    wrong_index: int
    correct_radius: float
    wrong_distance: float
    hinge: float
    has_correct_peer: bool = True
    source: str = "exact_outcome"
    outcome_gap: float = 1.0


def mine_question_local_hard_pairs(
    residual_paths: torch.Tensor,
    correctness: torch.Tensor,
    *,
    group_size: int,
    margin: float,
    max_pairs_per_group: int = 1,
    distance_kwargs: Optional[dict] = None,
    outcome_scores: Optional[torch.Tensor] = None,
    minimum_score_gap: float = 0.0,
) -> List[HardPathPair]:
    """Select exact or score-ranked local trajectory relations.

    Exact mixed groups retain the original correct/wrong hard-negative rule.
    A homogeneous group may provide a continuous-outcome bootstrap pair only
    when its frozen gold-answer score gap clears an explicit numerical floor.
    Such a pair is never reported as an exact correct/wrong pair.
    """
    if residual_paths.ndim != 3:
        raise ValueError("residual_paths must have shape [rollout, step, hidden]")
    if correctness.numel() != residual_paths.shape[0]:
        raise ValueError("correctness must match rollout count")
    if residual_paths.shape[0] % int(group_size):
        raise ValueError("rollout count must be divisible by group_size")
    if outcome_scores is not None:
        if outcome_scores.numel() != residual_paths.shape[0]:
            raise ValueError("outcome_scores must match rollout count")
        outcome_scores = outcome_scores.detach().view(-1).float()
    if float(minimum_score_gap) < 0.0:
        raise ValueError("minimum_score_gap must be non-negative")
    distance_kwargs = distance_kwargs or {}
    labels = correctness.detach().view(-1).to(residual_paths.device) > 0.5
    paths = residual_paths.detach()
    output: List[HardPathPair] = []
    for group_index, start in enumerate(
        range(0, paths.shape[0], int(group_size))
    ):
        end = start + int(group_size)
        local_labels = labels[start:end]
        correct = torch.nonzero(local_labels, as_tuple=False).flatten()
        wrong = torch.nonzero(~local_labels, as_tuple=False).flatten()
        source = "exact_outcome"
        local_scores = (
            None if outcome_scores is None else outcome_scores[start:end]
        )
        if correct.numel() < 1 or wrong.numel() < 1:
            # An all-correct group contains no failed trajectory to rescue.
            # Ranking its valid paths as positive/negative would erode the
            # correct-mode diversity protected by the Stage-1 prior.
            if correct.numel() == int(group_size):
                continue
            if local_scores is None or not torch.isfinite(local_scores).all():
                continue
            ranking = torch.argsort(local_scores, descending=True)
            score_gap = float(
                (local_scores[ranking[0]] - local_scores[ranking[-1]]).item()
            )
            if score_gap < float(minimum_score_gap):
                continue
            positive_count = max(2, int(group_size) // 2)
            correct = ranking[:positive_count]
            wrong = ranking[positive_count:]
            if wrong.numel() == 0:
                wrong = ranking[-1:]
                correct = ranking[:-1]
            correct = correct[:1]
            source = "continuous_outcome"
        correct_paths = paths[start:end].index_select(0, correct)
        wrong_paths = paths[start:end].index_select(0, wrong)
        if source == "continuous_outcome":
            peer_candidates = torch.argsort(
                local_scores,
                descending=True,
            )[1 : max(2, int(group_size) // 2)]
            if peer_candidates.numel() > 0:
                peer_paths = paths[start:end].index_select(
                    0,
                    peer_candidates,
                )
                peer_distances = trajectory_distance(
                    correct_paths[:, None],
                    peer_paths[None, :],
                    **distance_kwargs,
                )
                peer_rank = int(peer_distances[0].argmin().item())
                continuous_peer = int(peer_candidates[peer_rank].item())
                continuous_radius = float(
                    peer_distances[0, peer_rank].item()
                )
            else:
                continuous_peer = int(correct[0].item())
                continuous_radius = 0.0
        elif correct.numel() >= 2:
            correct_distances = trajectory_distance(
                correct_paths[:, None],
                correct_paths[None, :],
                **distance_kwargs,
            )
            correct_distances.fill_diagonal_(float("inf"))
        else:
            correct_distances = None
        correct_wrong = trajectory_distance(
            correct_paths[:, None],
            wrong_paths[None, :],
            **distance_kwargs,
        )
        candidates = []
        for correct_rank in range(correct.numel()):
            if source == "continuous_outcome":
                has_correct_peer = continuous_peer != int(
                    correct[correct_rank].item()
                )
                peer_index = continuous_peer
                radius = continuous_radius
            else:
                has_correct_peer = correct_distances is not None
                peer_rank = (
                    int(correct_distances[correct_rank].argmin().item())
                    if has_correct_peer
                    else correct_rank
                )
                peer_index = int(correct[peer_rank].item())
                radius = (
                    float(correct_distances[correct_rank, peer_rank].item())
                    if has_correct_peer
                    else 0.0
                )
            wrong_rank = int(
                correct_wrong[correct_rank].argmin().item()
            )
            wrong_distance = float(
                correct_wrong[correct_rank, wrong_rank].item()
            )
            selected_wrong = int(wrong[wrong_rank].item())
            outcome_gap = (
                1.0
                if source == "exact_outcome"
                else float(
                    (
                        local_scores[int(correct[correct_rank].item())]
                        - local_scores[selected_wrong]
                    ).item()
                )
            )
            if (
                source == "continuous_outcome"
                and outcome_gap < float(minimum_score_gap)
            ):
                continue
            candidates.append(
                HardPathPair(
                    group_index=group_index,
                    correct_index=start + int(correct[correct_rank].item()),
                    correct_peer_index=start + peer_index,
                    wrong_index=start + selected_wrong,
                    correct_radius=radius,
                    wrong_distance=wrong_distance,
                    hinge=max(
                        0.0,
                        float(margin) + radius - wrong_distance,
                    ),
                    has_correct_peer=has_correct_peer,
                    source=source,
                    outcome_gap=outcome_gap,
                )
            )
        candidates.sort(
            key=lambda item: (
                item.hinge,
                -item.wrong_distance,
            ),
            reverse=True,
        )
        output.extend(candidates[: max(1, int(max_pairs_per_group))])
    return output


def counterfactual_action_batch(
    actions: torch.Tensor,
    innovations: torch.Tensor,
    pairs: Sequence[HardPathPair],
    *,
    step_indices: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Build causal transition swaps with recipient suffix innovations."""
    if actions.shape != innovations.shape or actions.ndim != 3:
        raise ValueError(
            "actions and innovations must have shape [rollout, step, action]"
        )
    n_steps = actions.shape[1]
    if step_indices is None:
        selected_steps = list(range(n_steps))
    else:
        selected_steps = sorted({int(index) for index in step_indices})
        if not selected_steps or any(
            index < 0 or index >= n_steps for index in selected_steps
        ):
            raise ValueError("step_indices must select valid transitions")
    forced_actions = []
    forced_masks = []
    suffix_innovations = []
    pair_indices = []
    step_indices = []
    directions = []
    for pair_index, pair in enumerate(pairs):
        for step_index in selected_steps:
            for direction, recipient, donor in (
                (0, pair.correct_index, pair.wrong_index),
                (1, pair.wrong_index, pair.correct_index),
            ):
                current_actions = actions[recipient].detach().clone()
                current_actions[step_index] = actions[donor, step_index]
                mask = torch.zeros(
                    n_steps,
                    device=actions.device,
                    dtype=torch.bool,
                )
                mask[: step_index + 1] = True
                forced_actions.append(current_actions)
                forced_masks.append(mask)
                suffix_innovations.append(
                    innovations[recipient].detach().clone()
                )
                pair_indices.append(pair_index)
                step_indices.append(step_index)
                directions.append(direction)
    if not forced_actions:
        empty_actions = actions.new_empty(
            (0, actions.shape[1], actions.shape[2])
        )
        return {
            "forced_actions": empty_actions,
            "forced_mask": torch.empty(
                0,
                actions.shape[1],
                device=actions.device,
                dtype=torch.bool,
            ),
            "innovations": empty_actions.clone(),
            "pair_indices": torch.empty(
                0,
                device=actions.device,
                dtype=torch.long,
            ),
            "step_indices": torch.empty(
                0,
                device=actions.device,
                dtype=torch.long,
            ),
            "directions": torch.empty(
                0,
                device=actions.device,
                dtype=torch.long,
            ),
        }
    return {
        "forced_actions": torch.stack(forced_actions, dim=0),
        "forced_mask": torch.stack(forced_masks, dim=0),
        "innovations": torch.stack(suffix_innovations, dim=0),
        "pair_indices": torch.tensor(
            pair_indices,
            device=actions.device,
            dtype=torch.long,
        ),
        "step_indices": torch.tensor(
            step_indices,
            device=actions.device,
            dtype=torch.long,
        ),
        "directions": torch.tensor(
            directions,
            device=actions.device,
            dtype=torch.long,
        ),
    }


def counterfactual_transition_credits(
    base_scores: torch.Tensor,
    counterfactual_scores: torch.Tensor,
    metadata: Dict[str, torch.Tensor],
    pairs: Sequence[HardPathPair],
    *,
    n_steps: int,
) -> torch.Tensor:
    """Compute bidirectional causal credit for every transition in each pair."""
    credits = base_scores.new_zeros((len(pairs), int(n_steps)))
    counts = base_scores.new_zeros((len(pairs), int(n_steps)))
    for row in range(counterfactual_scores.shape[0]):
        pair_index = int(metadata["pair_indices"][row].item())
        step_index = int(metadata["step_indices"][row].item())
        direction = int(metadata["directions"][row].item())
        pair = pairs[pair_index]
        if direction == 0:
            effect = (
                base_scores[pair.correct_index]
                - counterfactual_scores[row]
            )
        else:
            effect = (
                counterfactual_scores[row]
                - base_scores[pair.wrong_index]
            )
        credits[pair_index, step_index] += 0.5 * effect
        counts[pair_index, step_index] += 0.5
    expected = torch.zeros_like(counts)
    if counterfactual_scores.numel() > 0:
        expected[
            metadata["pair_indices"],
            metadata["step_indices"],
        ] = 1.0
    if pairs and not torch.allclose(counts, expected):
        raise ValueError(
            "counterfactual metadata must contain both directions exactly "
            "once for every selected pair-transition"
        )
    return credits


def group_standardize(
    values: torch.Tensor,
    *,
    group_size: int,
    preserve_trailing_positions: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    if values.shape[0] % int(group_size):
        raise ValueError("first dimension must be divisible by group_size")
    grouped = values.view(
        values.shape[0] // int(group_size),
        int(group_size),
        *values.shape[1:],
    )
    reduce_dims = (
        (1,)
        if preserve_trailing_positions and grouped.ndim > 2
        else tuple(range(1, grouped.ndim))
    )
    mean = grouped.mean(dim=reduce_dims, keepdim=True)
    std = grouped.std(dim=reduce_dims, keepdim=True, unbiased=False)
    standardized = (grouped - mean) / std.clamp_min(float(eps))
    return standardized.reshape_as(values)


def discounted_step_returns(
    step_rewards: torch.Tensor,
    *,
    gamma: float,
    reward_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute future-looking local returns over PLAN through CHECK.

    Step rewards are accepted at all eight tensor positions for API stability,
    but COMMIT is masked by schema and therefore has zero local return.
    """
    if step_rewards.ndim != 2:
        raise ValueError("step_rewards must have shape [batch, role-step]")
    expected = (step_rewards.shape[0], TRACE_N_ROLE_STEPS)
    if tuple(step_rewards.shape) != expected:
        raise ValueError(
            f"step_rewards must have shape {expected}, got "
            f"{tuple(step_rewards.shape)}"
        )
    if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be finite and lie in [0, 1]")
    schema_mask = trace_stochastic_action_mask(
        step_rewards.shape[0],
        device=step_rewards.device,
        dtype=step_rewards.dtype,
    )
    if reward_mask is None:
        active = schema_mask
    else:
        if tuple(reward_mask.shape) != expected:
            raise ValueError(
                f"reward_mask must have shape {expected}, got "
                f"{tuple(reward_mask.shape)}"
            )
        active = reward_mask.to(
            device=step_rewards.device,
            dtype=step_rewards.dtype,
        ) * schema_mask
    local_rewards = step_rewards.float() * active.float()
    returns = local_rewards.new_empty(expected)
    running = local_rewards.new_zeros(step_rewards.shape[0])
    for step_index in range(TRACE_N_ROLE_STEPS - 1, -1, -1):
        running = (
            local_rewards[:, step_index] + float(gamma) * running
        )
        returns[:, step_index] = running
    return returns


def build_discounted_role_returns(
    terminal_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    *,
    gamma: float,
    step_reward_weight: float,
    reward_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine terminal outcome with discounted future role rewards."""
    if terminal_rewards.ndim != 1:
        raise ValueError("terminal_rewards must have shape [batch]")
    if terminal_rewards.shape[0] != step_rewards.shape[0]:
        raise ValueError("terminal_rewards and step_rewards must align")
    if (
        not math.isfinite(float(step_reward_weight))
        or float(step_reward_weight) < 0.0
    ):
        raise ValueError("step_reward_weight must be finite and non-negative")
    local_returns = discounted_step_returns(
        step_rewards,
        gamma=gamma,
        reward_mask=reward_mask,
    )
    terminal = terminal_rewards.to(
        device=local_returns.device,
        dtype=local_returns.dtype,
    )
    return (
        terminal.unsqueeze(-1)
        + float(step_reward_weight) * local_returns
    )


def build_discounted_role_advantages(
    terminal_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    *,
    group_size: int,
    gamma: float,
    step_reward_weight: float,
    reward_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build per-position group advantages and exclude deterministic COMMIT."""
    returns = build_discounted_role_returns(
        terminal_rewards,
        step_rewards,
        gamma=gamma,
        step_reward_weight=step_reward_weight,
        reward_mask=reward_mask,
    )
    advantages = group_standardize(
        returns,
        group_size=group_size,
        preserve_trailing_positions=True,
    )
    action_mask = trace_stochastic_action_mask(
        terminal_rewards.shape[0],
        device=advantages.device,
        dtype=advantages.dtype,
    )
    return advantages * action_mask


def build_transition_advantages(
    terminal_rewards: torch.Tensor,
    *,
    n_steps: int,
    group_size: int,
    pairs: Sequence[HardPathPair],
    counterfactual_credits: Optional[torch.Tensor],
    counterfactual_weight: float,
    local_weight: float,
    credit_temperature: float,
    local_temperature: float,
) -> torch.Tensor:
    """Combine terminal outcome, hard-pair relation, and per-step credit."""
    terminal = terminal_rewards.view(-1)
    terminal_advantage = group_standardize(
        terminal.unsqueeze(-1),
        group_size=group_size,
    ).squeeze(-1)
    advantages = terminal_advantage.unsqueeze(-1).expand(
        -1,
        int(n_steps),
    ).clone()
    if pairs and counterfactual_credits is None:
        raise ValueError("counterfactual credits are required for hard pairs")
    for pair_index, pair in enumerate(pairs):
        credits = torch.tanh(
            counterfactual_credits[pair_index]
            / max(float(credit_temperature), 1e-6)
        )
        advantages[pair.correct_index] += (
            float(counterfactual_weight) * credits
        )
        advantages[pair.wrong_index] -= (
            float(counterfactual_weight) * credits
        )
        local_signal = math.tanh(
            pair.hinge / max(float(local_temperature), 1e-6)
        )
        if pair.has_correct_peer:
            advantages[pair.correct_index] += (
                0.5 * float(local_weight) * local_signal
            )
            advantages[pair.correct_peer_index] += (
                0.5 * float(local_weight) * local_signal
            )
        else:
            advantages[pair.correct_index] += (
                float(local_weight) * local_signal
            )
        advantages[pair.wrong_index] -= (
            float(local_weight) * local_signal
        )
    return group_standardize(
        advantages,
        group_size=group_size,
        preserve_trailing_positions=True,
    )


def clipped_policy_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not (
        current_log_probs.shape
        == old_log_probs.shape
        == advantages.shape
    ):
        raise ValueError("policy tensors must have identical shapes")
    ratio = torch.exp(current_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(
        1.0 - float(clip_epsilon),
        1.0 + float(clip_epsilon),
    ) * advantages
    losses = -torch.minimum(unclipped, clipped)
    if mask is None:
        return losses.mean()
    mask = mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)
