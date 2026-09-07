"""Small, auditable mathematical components for TRACE-VB.

This module deliberately has no dependency on the language-model wrapper.  It
contains the training-only PLAN forecast head, the shared architecture used by
the SFT sufficiency head and RL critic, and masked actor-critic objectives.  A
caller supplies the stochastic-action mask, so deterministic COMMIT states can
be retained in rollout tensors while being excluded from every policy/value
objective.
"""

import math
from typing import Callable, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GAEOutput(NamedTuple):
    """Detached advantages and lambda-return targets for active actions."""

    advantages: torch.Tensor
    returns: torch.Tensor


class ActionEfficacyOutput(NamedTuple):
    """Action-to-transition hinge loss and mean active efficacy ratio."""

    loss: torch.Tensor
    active_efficacy_ratio: torch.Tensor


def _validate_positive_integer(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _broadcast_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask broadcast to ``reference`` without copying."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if tuple(mask.shape) == tuple(reference.shape):
        return mask.to(device=reference.device, dtype=torch.bool)
    if mask.ndim == 1 and mask.shape[0] == reference.shape[-1]:
        shape = (1,) * (reference.ndim - 1) + (mask.shape[0],)
        return mask.to(
            device=reference.device,
            dtype=torch.bool,
        ).reshape(shape).expand_as(reference)
    raise ValueError(
        "mask must match the reference tensor or its final time axis: "
        f"got mask={tuple(mask.shape)}, reference={tuple(reference.shape)}"
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = _broadcast_mask(mask, values).to(dtype=values.dtype)
    # Avoid a device synchronization in every PPO minibatch.  The fixed TRACE
    # schema validates its seven-action mask once at construction time; the
    # clamp only makes this generic helper well-defined for an empty mask.
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class PlanForecastHead(nn.Module):
    """Predict ordered low-dimensional CoT targets from one PLAN state.

    The leading dimensions of ``plan_states`` are preserved.  For example,
    ``[batch, hidden]`` inputs produce ``[batch, 5, target_dim]`` outputs.  The
    module is training-only and therefore adds no answer-generation cost.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        target_dim: int,
        n_targets: int = 5,
        head_hidden_size: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = _validate_positive_integer(hidden_size, "hidden_size")
        self.target_dim = _validate_positive_integer(target_dim, "target_dim")
        self.n_targets = _validate_positive_integer(n_targets, "n_targets")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if head_hidden_size is None:
            head_hidden_size = min(self.hidden_size, 512)
        self.head_hidden_size = _validate_positive_integer(
            head_hidden_size,
            "head_hidden_size",
        )

        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.network = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                self.head_hidden_size,
                self.n_targets * self.target_dim,
            ),
        )

    def forward(self, plan_states: torch.Tensor) -> torch.Tensor:
        if plan_states.ndim < 1 or plan_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "plan_states must end in hidden_size="
                f"{self.hidden_size}, got {tuple(plan_states.shape)}"
            )
        head_dtype = self.input_norm.weight.dtype
        states = plan_states.to(dtype=head_dtype)
        predictions = self.network(self.input_norm(states))
        return predictions.reshape(
            *plan_states.shape[:-1],
            self.n_targets,
            self.target_dim,
        )


class LatentStepTextDecoder(nn.Module):
    """Training-only autoregressive decoder for one SOLVE residual.

    The decoder receives no question tokens, answer tokens, other latent
    states, or teacher hidden states. Its only sample-specific conditioning
    signal is the action-induced SOLVE residual. Previous gold-step tokens are
    supplied solely for ordinary teacher forcing. Returned states live in the
    backbone hidden space so callers can use a frozen tied LM head instead of
    allocating a second vocabulary matrix.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        decoder_hidden_size: int = 512,
        n_solve_roles: int = 5,
    ) -> None:
        super().__init__()
        self.hidden_size = _validate_positive_integer(hidden_size, "hidden_size")
        self.decoder_hidden_size = _validate_positive_integer(
            decoder_hidden_size,
            "decoder_hidden_size",
        )
        self.n_solve_roles = _validate_positive_integer(
            n_solve_roles,
            "n_solve_roles",
        )
        self.latent_norm = nn.LayerNorm(self.hidden_size)
        self.context_projection = nn.Linear(
            self.hidden_size,
            self.decoder_hidden_size,
        )
        self.token_projection = nn.Linear(
            self.hidden_size,
            self.decoder_hidden_size,
            bias=False,
        )
        self.role_embedding = nn.Embedding(
            self.n_solve_roles,
            self.decoder_hidden_size,
        )
        self.recurrent = nn.GRU(
            input_size=2 * self.decoder_hidden_size,
            hidden_size=self.decoder_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_projection = nn.Linear(
            self.decoder_hidden_size,
            self.hidden_size,
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        solve_residuals: torch.Tensor,
        previous_token_embeddings: torch.Tensor,
        solve_role_ids: torch.Tensor,
    ) -> torch.Tensor:
        if (
            solve_residuals.ndim != 2
            or solve_residuals.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                "solve_residuals must have shape [batch, hidden_size]"
            )
        if (
            previous_token_embeddings.ndim != 3
            or previous_token_embeddings.shape[0] != solve_residuals.shape[0]
            or previous_token_embeddings.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                "previous_token_embeddings must have shape "
                "[batch, tokens, hidden_size]"
            )
        if tuple(solve_role_ids.shape) != (solve_residuals.shape[0],):
            raise ValueError("solve_role_ids must have shape [batch]")
        if solve_role_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("solve_role_ids must use an integer dtype")
        if previous_token_embeddings.shape[1] <= 0:
            raise ValueError("the decoder requires at least one target token")

        module_dtype = self.latent_norm.weight.dtype
        residuals = solve_residuals.to(dtype=module_dtype)
        token_embeddings = previous_token_embeddings.to(dtype=module_dtype)
        role_ids = solve_role_ids.to(
            device=solve_residuals.device,
            dtype=torch.long,
        )
        if bool((role_ids < 0).any()) or bool(
            (role_ids >= self.n_solve_roles).any()
        ):
            raise ValueError("solve_role_ids contain an out-of-range role")

        context = torch.tanh(
            self.context_projection(self.latent_norm(residuals))
            + self.role_embedding(role_ids).to(dtype=module_dtype)
        )
        token_features = self.token_projection(token_embeddings)
        repeated_context = context[:, None, :].expand(
            -1,
            token_features.shape[1],
            -1,
        )
        decoder_inputs = torch.cat(
            [token_features, repeated_context],
            dim=-1,
        )
        decoded, _ = self.recurrent(
            decoder_inputs,
            context.unsqueeze(0).contiguous(),
        )
        return self.output_norm(self.output_projection(decoded))


def per_sequence_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    token_chunk_size: int = 8,
) -> torch.Tensor:
    """Return one normalized token loss per decoded CoT step chunk."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocabulary]")
    if labels.ndim != 2 or mask.ndim != 2:
        raise ValueError("labels and mask must have shape [batch, tokens]")
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("logits and labels must share batch/token axes")
    if tuple(mask.shape) != tuple(labels.shape):
        raise ValueError("mask must match labels")
    if labels.dtype != torch.long:
        raise TypeError("labels must use torch.long")
    token_chunk_size = _validate_positive_integer(
        token_chunk_size,
        "token_chunk_size",
    )
    batch_size, token_count = labels.shape
    numerators = logits[:, :0, :].sum(dim=(1, 2), dtype=torch.float32)
    weights = mask.to(device=logits.device, dtype=torch.float32)
    for start in range(0, token_count, token_chunk_size):
        end = min(start + token_chunk_size, token_count)
        local_loss = F.cross_entropy(
            logits[:, start:end, :].float().reshape(
                -1,
                logits.shape[-1],
            ),
            labels[:, start:end].reshape(-1),
            reduction="none",
        ).reshape(batch_size, end - start)
        numerators = numerators + (
            local_loss * weights[:, start:end]
        ).sum(dim=-1)
    return numerators / weights.sum(dim=-1).clamp_min(1.0)


def checkpointed_projected_token_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    project_logits: Callable[[torch.Tensor], torch.Tensor],
    *,
    token_chunk_size: int = 8,
) -> torch.Tensor:
    """Project and score token chunks without retaining full-vocabulary logits.

    The vocabulary projection lives inside a non-reentrant activation
    checkpoint. Backward therefore recomputes one small token chunk at a time
    instead of preserving a ``[batch, tokens, vocabulary]`` tensor for the
    entire decoded CoT. The returned objective is mathematically identical to
    :func:`per_sequence_token_cross_entropy`.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape [batch, tokens, hidden]"
        )
    if labels.ndim != 2 or mask.ndim != 2:
        raise ValueError("labels and mask must have shape [batch, tokens]")
    if tuple(hidden_states.shape[:2]) != tuple(labels.shape):
        raise ValueError("hidden_states and labels must share batch/token axes")
    if tuple(mask.shape) != tuple(labels.shape):
        raise ValueError("mask must match labels")
    if labels.dtype != torch.long:
        raise TypeError("labels must use torch.long")
    if labels.device != hidden_states.device or mask.device != hidden_states.device:
        raise ValueError("hidden_states, labels, and mask must share a device")
    if not callable(project_logits):
        raise TypeError("project_logits must be callable")
    token_chunk_size = _validate_positive_integer(
        token_chunk_size,
        "token_chunk_size",
    )
    batch_size, token_count = labels.shape
    if token_count <= 0:
        raise ValueError("at least one target token is required")

    def projected_token_losses(
        state_chunk: torch.Tensor,
        label_chunk: torch.Tensor,
    ) -> torch.Tensor:
        logits = project_logits(state_chunk)
        if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(
            label_chunk.shape
        ):
            raise ValueError(
                "project_logits must return [batch, tokens, vocabulary]"
            )
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            label_chunk.reshape(-1),
            reduction="none",
        ).reshape(label_chunk.shape)

    numerators = hidden_states[:, :0, :].sum(
        dim=(1, 2),
        dtype=torch.float32,
    )
    weights = mask.to(dtype=torch.float32)
    for start in range(0, token_count, token_chunk_size):
        end = min(start + token_chunk_size, token_count)
        state_chunk = hidden_states[:, start:end, :]
        label_chunk = labels[:, start:end]
        if torch.is_grad_enabled() and state_chunk.requires_grad:
            local_loss = checkpoint(
                projected_token_losses,
                state_chunk,
                label_chunk,
                use_reentrant=False,
            )
        else:
            local_loss = projected_token_losses(state_chunk, label_chunk)
        numerators = numerators + (
            local_loss * weights[:, start:end]
        ).sum(dim=-1)
    return numerators / weights.sum(dim=-1).clamp_min(1.0)


class RoleConditionedScalarHead(nn.Module):
    """A role-conditioned scalar estimator shared by sufficiency and value.

    Stage 1 may instantiate this class as ``S_phi`` and Stage 2 as ``V_psi``.
    ``V_psi.copy_from(S_phi)`` is an exact initialization bridge: all layer
    normalization, role embeddings, trunk, and output parameters are copied.
    The returned values are unconstrained scalar logits/predictions; callers
    choose BCE-with-logits, regression, or calibration appropriate to targets.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        n_roles: int,
        role_embedding_dim: int = 32,
        head_hidden_size: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = _validate_positive_integer(hidden_size, "hidden_size")
        self.n_roles = _validate_positive_integer(n_roles, "n_roles")
        self.role_embedding_dim = _validate_positive_integer(
            role_embedding_dim,
            "role_embedding_dim",
        )
        self.head_hidden_size = _validate_positive_integer(
            head_hidden_size,
            "head_hidden_size",
        )
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.dropout = float(dropout)

        self.state_norm = nn.LayerNorm(self.hidden_size)
        self.role_embedding = nn.Embedding(
            self.n_roles,
            self.role_embedding_dim,
        )
        self.network = nn.Sequential(
            nn.Linear(
                self.hidden_size + self.role_embedding_dim,
                self.head_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_size, 1),
        )

    @property
    def architecture_signature(self) -> Tuple[int, int, int, int, float]:
        """Shape signature required for a lossless S-to-V parameter copy."""
        return (
            self.hidden_size,
            self.n_roles,
            self.role_embedding_dim,
            self.head_hidden_size,
            self.dropout,
        )

    def copy_from(
        self,
        source: "RoleConditionedScalarHead",
    ) -> "RoleConditionedScalarHead":
        """Copy an isomorphic sufficiency/value head without sharing storage."""
        if not isinstance(source, RoleConditionedScalarHead):
            raise TypeError("source must be a RoleConditionedScalarHead")
        if self.architecture_signature != source.architecture_signature:
            raise ValueError(
                "source and destination head architectures must match: "
                f"source={source.architecture_signature}, "
                f"destination={self.architecture_signature}"
            )
        self.load_state_dict(source.state_dict(), strict=True)
        return self

    def forward(
        self,
        states: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> torch.Tensor:
        if states.ndim < 1 or states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"states must end in hidden_size={self.hidden_size}, got "
                f"{tuple(states.shape)}"
            )
        expected_role_shape = states.shape[:-1]
        if tuple(role_ids.shape) != tuple(expected_role_shape):
            raise ValueError(
                "role_ids must match every non-hidden state axis: expected "
                f"{tuple(expected_role_shape)}, got {tuple(role_ids.shape)}"
            )
        if role_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("role_ids must use an integer dtype")
        role_ids = role_ids.to(device=states.device, dtype=torch.long)
        head_dtype = self.state_norm.weight.dtype
        normalized = self.state_norm(states.to(dtype=head_dtype))
        role_features = self.role_embedding(role_ids).to(dtype=head_dtype)
        features = torch.cat([normalized, role_features], dim=-1)
        return self.network(features).squeeze(-1)


def action_efficacy_hinge(
    sampled_actions: torch.Tensor,
    map_actions: torch.Tensor,
    sampled_transitions: torch.Tensor,
    map_transitions: torch.Tensor,
    action_mask: torch.Tensor,
    minimum_ratio: float,
) -> ActionEfficacyOutput:
    """Require sampled actions to cause a proportional transition change.

    Feature distances are root-mean-square Euclidean distances, implemented
    as an L2 norm divided by the square root of the feature dimension.  This
    makes action and hidden-state distances comparable despite their different
    dimensionalities.  The action distance is detached: it defines a fixed
    minimum effect for this update and cannot be reduced by moving the policy
    action back toward its MAP counterpart.  ``action_mask`` must exactly
    match the leading action axes, so deterministic COMMIT is excluded
    explicitly rather than by a broadcasting accident.
    """
    tensors = {
        "sampled_actions": sampled_actions,
        "map_actions": map_actions,
        "sampled_transitions": sampled_transitions,
        "map_transitions": map_transitions,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim < 2 or tensor.shape[-1] <= 0:
            raise ValueError(
                f"{name} must have non-empty [..., time, feature] axes"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
    if sampled_actions.shape != map_actions.shape:
        raise ValueError("sampled_actions and map_actions must match exactly")
    if sampled_transitions.shape != map_transitions.shape:
        raise ValueError(
            "sampled_transitions and map_transitions must match exactly"
        )
    if sampled_actions.shape[:-1] != sampled_transitions.shape[:-1]:
        raise ValueError(
            "actions and transitions must share every leading axis"
        )
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError("all action and transition tensors must share a device")
    if not isinstance(action_mask, torch.Tensor):
        raise TypeError("action_mask must be a torch.Tensor")
    if action_mask.dtype != torch.bool:
        raise TypeError("action_mask must use torch.bool")
    if tuple(action_mask.shape) != tuple(sampled_actions.shape[:-1]):
        raise ValueError(
            "action_mask must exactly match the leading action axes: "
            f"expected {tuple(sampled_actions.shape[:-1])}, "
            f"got {tuple(action_mask.shape)}"
        )
    if action_mask.device != sampled_actions.device:
        raise ValueError("action_mask must share the action tensor device")
    if (
        isinstance(minimum_ratio, bool)
        or not math.isfinite(float(minimum_ratio))
        or float(minimum_ratio) <= 0.0
    ):
        raise ValueError("minimum_ratio must be finite and strictly positive")

    action_dimension = int(sampled_actions.shape[-1])
    transition_dimension = int(sampled_transitions.shape[-1])
    action_distance = (
        (sampled_actions.float() - map_actions.float()).norm(dim=-1)
        / math.sqrt(action_dimension)
    ).detach()
    transition_distance = (
        (
            sampled_transitions.float()
            - map_transitions.float()
        ).norm(dim=-1)
        / math.sqrt(transition_dimension)
    )
    required_transition_distance = float(minimum_ratio) * action_distance
    per_action_hinge = F.relu(
        required_transition_distance - transition_distance
    )
    loss = _masked_mean(per_action_hinge, action_mask)

    # A zero sampled-vs-MAP action has a zero hinge target; the clamped
    # denominator keeps its logging statistic finite without adding a tunable
    # epsilon to the public method.
    epsilon = torch.finfo(action_distance.dtype).eps
    efficacy_ratio = transition_distance / action_distance.clamp_min(epsilon)
    active_efficacy_ratio = _masked_mean(efficacy_ratio, action_mask)
    return ActionEfficacyOutput(
        loss=loss,
        active_efficacy_ratio=active_efficacy_ratio,
    )


def masked_terminal_reward_gae(
    terminal_rewards: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> GAEOutput:
    """Compute GAE when the only environment reward is terminal correctness.

    ``values`` and ``action_mask`` have shape ``[..., time]`` while
    ``terminal_rewards`` has shape ``[...]``.  The terminal reward is attached
    to the final *active* stochastic action in each trajectory.  Masked slots
    are skipped rather than treated as artificial episode boundaries, which
    naturally excludes a trailing deterministic COMMIT slot.  Both outputs
    are detached targets, preventing accidental critic gradients through GAE.
    """
    if values.ndim < 1:
        raise ValueError("values must include a time axis")
    if tuple(terminal_rewards.shape) != tuple(values.shape[:-1]):
        raise ValueError(
            "terminal_rewards must match values without its time axis: "
            f"got rewards={tuple(terminal_rewards.shape)}, "
            f"values={tuple(values.shape)}"
        )
    if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if (
        not math.isfinite(float(gae_lambda))
        or not 0.0 <= float(gae_lambda) <= 1.0
    ):
        raise ValueError("gae_lambda must be finite and in [0, 1]")

    mask = _broadcast_mask(action_mask, values)
    time_steps = int(values.shape[-1])
    flat_values = values.detach().reshape(-1, time_steps)
    flat_mask = mask.reshape(-1, time_steps)
    flat_rewards = terminal_rewards.detach().to(
        device=values.device,
        dtype=values.dtype,
    ).reshape(-1)
    advantages = torch.zeros_like(flat_values)
    next_value = torch.zeros_like(flat_rewards)
    next_advantage = torch.zeros_like(flat_rewards)
    has_later_action = torch.zeros_like(flat_rewards, dtype=torch.bool)

    for step_index in range(time_steps - 1, -1, -1):
        active = flat_mask[:, step_index]
        is_final_action = active & ~has_later_action
        reward = torch.where(
            is_final_action,
            flat_rewards,
            torch.zeros_like(flat_rewards),
        )
        delta = (
            reward
            + float(gamma) * next_value
            - flat_values[:, step_index]
        )
        candidate = (
            delta
            + float(gamma) * float(gae_lambda) * next_advantage
        )
        advantages[:, step_index] = torch.where(
            active,
            candidate,
            torch.zeros_like(candidate),
        )
        next_value = torch.where(
            active,
            flat_values[:, step_index],
            next_value,
        )
        next_advantage = torch.where(active, candidate, next_advantage)
        has_later_action = has_later_action | active

    advantages = advantages.reshape_as(values)
    returns = (advantages + values.detach()) * mask.to(dtype=values.dtype)
    return GAEOutput(advantages=advantages, returns=returns)


def masked_ppo_actor_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.12,
    max_abs_log_ratio: float = 20.0,
) -> torch.Tensor:
    """Clipped PPO actor objective averaged only over stochastic actions."""
    if not (
        current_log_probs.shape == old_log_probs.shape == advantages.shape
    ):
        raise ValueError(
            "current_log_probs, old_log_probs, and advantages must match"
        )
    if (
        not math.isfinite(float(clip_epsilon))
        or not 0.0 <= float(clip_epsilon) < 1.0
    ):
        raise ValueError("clip_epsilon must be finite and in [0, 1)")
    if (
        not math.isfinite(float(max_abs_log_ratio))
        or float(max_abs_log_ratio) <= 0.0
    ):
        raise ValueError("max_abs_log_ratio must be finite and positive")

    log_ratio = (
        current_log_probs - old_log_probs.detach()
    ).clamp(
        min=-float(max_abs_log_ratio),
        max=float(max_abs_log_ratio),
    )
    ratio = torch.exp(log_ratio)
    detached_advantages = advantages.detach().to(dtype=ratio.dtype)
    unclipped = ratio * detached_advantages
    clipped = ratio.clamp(
        min=1.0 - float(clip_epsilon),
        max=1.0 + float(clip_epsilon),
    ) * detached_advantages
    per_action_loss = -torch.minimum(unclipped, clipped)
    return _masked_mean(per_action_loss, action_mask)


def masked_value_loss(
    predicted_values: torch.Tensor,
    target_returns: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Huber or MSE critic loss averaged only over active action states."""
    if predicted_values.shape != target_returns.shape:
        raise ValueError("predicted_values and target_returns must match")
    normalized_loss_type = str(loss_type).strip().lower()
    targets = target_returns.detach().to(
        device=predicted_values.device,
        dtype=predicted_values.dtype,
    )
    if normalized_loss_type == "huber":
        if (
            not math.isfinite(float(huber_delta))
            or float(huber_delta) <= 0.0
        ):
            raise ValueError("huber_delta must be finite and positive")
        per_action_loss = F.huber_loss(
            predicted_values,
            targets,
            reduction="none",
            delta=float(huber_delta),
        )
    elif normalized_loss_type == "mse":
        per_action_loss = (predicted_values - targets).square()
    else:
        raise ValueError("loss_type must be either 'huber' or 'mse'")
    return _masked_mean(per_action_loss, action_mask)


__all__ = [
    "ActionEfficacyOutput",
    "GAEOutput",
    "LatentStepTextDecoder",
    "PlanForecastHead",
    "RoleConditionedScalarHead",
    "action_efficacy_hinge",
    "masked_ppo_actor_loss",
    "masked_terminal_reward_gae",
    "masked_value_loss",
    "per_sequence_token_cross_entropy",
]
