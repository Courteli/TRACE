import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Autoregressive low-dimensional policy over latent transitions.

    The policy has no route table or categorical view identity. Every sampled
    path uses the same conditional Gaussian heads. Dynamics modules are kept
    separate from policy heads so Stage 2 can freeze the Stage-1 transition
    semantics while refining only action probabilities.
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
        minimum_action_gate: float = 0.08,
        maximum_action_gate: float = 0.40,
        initial_action_gate: float = 0.20,
    ):
        super().__init__()
        if hidden_size <= 0 or action_dim <= 0 or n_steps <= 0:
            raise ValueError("hidden_size, action_dim, and n_steps must be positive")
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.n_steps = int(n_steps)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.minimum_action_gate = float(minimum_action_gate)
        self.maximum_action_gate = float(maximum_action_gate)
        if not 0.0 < self.minimum_action_gate < self.maximum_action_gate:
            raise ValueError(
                "action gates must satisfy 0 < minimum < maximum"
            )
        if not (
            self.minimum_action_gate
            <= float(initial_action_gate)
            <= self.maximum_action_gate
        ):
            raise ValueError("initial_action_gate must lie inside gate bounds")

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
        self.mean_head = nn.Linear(int(policy_hidden_size), self.action_dim)
        self.log_std_head = nn.Linear(
            int(policy_hidden_size),
            self.action_dim,
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
        self.action_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 1),
        )

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, float(initial_log_std))
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
        nn.init.zeros_(self.action_gate[-1].weight)
        gate_fraction = (
            float(initial_action_gate) - self.minimum_action_gate
        ) / (self.maximum_action_gate - self.minimum_action_gate)
        gate_fraction = min(max(gate_fraction, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(
            self.action_gate[-1].bias,
            math.log(gate_fraction / (1.0 - gate_fraction)),
        )

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
        means = torch.nan_to_num(
            self.mean_head(hidden),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(min=-20.0, max=20.0)
        log_stds = torch.nan_to_num(
            self.log_std_head(hidden),
            nan=self.min_log_std,
            posinf=self.max_log_std,
            neginf=self.min_log_std,
        ).clamp(min=self.min_log_std, max=self.max_log_std)
        return means, log_stds

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
        action = action / action.square().mean(
            dim=-1,
            keepdim=True,
        ).add(1e-6).sqrt()
        base_rms = base.detach().square().mean(
            dim=-1,
            keepdim=True,
        ).add(1e-6).sqrt()
        gate = self.minimum_action_gate + (
            self.maximum_action_gate - self.minimum_action_gate
        ) * torch.sigmoid(self.action_gate(base))
        return (
            base
            + float(step_scale) * step
            + float(action_scale) * gate * base_rms * action
        )

    def action_gate_values(self, states: torch.Tensor) -> torch.Tensor:
        """Return the bounded dynamics gate for mechanism diagnostics."""
        return self.minimum_action_gate + (
            self.maximum_action_gate - self.minimum_action_gate
        ) * torch.sigmoid(self.action_gate(states.float()))

    def set_stage2_trainability(self):
        """Freeze action semantics; refine only conditional action density."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (
            self.policy_step_embedding,
            self.policy_trunk,
            self.mean_head,
            self.log_std_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


class CoTConditionedTrajectoryPosterior(nn.Module):
    """Training-only posterior over actions conditioned on one gold CoT.

    It predicts a residual over the question-only prior in the same action
    space. There is no route table, view embedding, or persistent path ID.
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
        self.mean_delta = nn.Linear(
            int(posterior_hidden_size),
            self.action_dim,
        )
        self.log_std_delta = nn.Linear(
            int(posterior_hidden_size),
            self.action_dim,
        )
        nn.init.zeros_(self.mean_delta.weight)
        nn.init.zeros_(self.mean_delta.bias)
        nn.init.zeros_(self.log_std_delta.weight)
        nn.init.zeros_(self.log_std_delta.bias)

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
        # The question-only prior is shared by every exchangeable posterior
        # sample. Let task and formation gradients reach that shared prior;
        # the residual posterior remains training-only and is regularized by
        # KL, rather than making a separate MAP branch carry prior learning.
        means = prior_means + self.mean_delta(hidden)
        log_stds = (
            prior_log_stds + self.log_std_delta(hidden)
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
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Identify each sampled action from its realized latent transition.

    Positives are matched action/transition pairs. Negatives are only the
    other IID paths of the same question at the same transition position.
    Consequently this criterion has no route table and remains invariant to
    any permutation of the exchangeable path axis.
    """
    if predicted_actions.shape != sampled_actions.shape:
        raise ValueError(
            "predicted and sampled actions must have identical shapes"
        )
    if predicted_actions.ndim != 4 or predicted_actions.shape[1] < 2:
        raise ValueError(
            "identifiability requires [batch, path>=2, step, action]"
        )
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    predictions = F.normalize(
        predicted_actions.float().permute(0, 2, 1, 3),
        dim=-1,
    )
    targets = F.normalize(
        standardized_within_question_actions(
            sampled_actions.detach()
        ).permute(0, 2, 1, 3),
        dim=-1,
    )
    logits = torch.einsum(
        "bmia,bmja->bmij",
        predictions,
        targets,
    ) / float(temperature)
    labels = torch.arange(
        logits.shape[-1],
        device=logits.device,
    ).expand(logits.shape[0], logits.shape[1], -1)
    forward = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )
    backward = F.cross_entropy(
        logits.transpose(-1, -2).reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )
    return 0.5 * (forward + backward)


def action_transition_retrieval_accuracy(
    predicted_actions: torch.Tensor,
    sampled_actions: torch.Tensor,
) -> torch.Tensor:
    """Top-1 within-question retrieval used only as a diagnostic."""
    predictions = F.normalize(
        predicted_actions.float().permute(0, 2, 1, 3),
        dim=-1,
    )
    targets = F.normalize(
        standardized_within_question_actions(
            sampled_actions.detach()
        ).permute(0, 2, 1, 3),
        dim=-1,
    )
    logits = torch.einsum("bmia,bmja->bmij", predictions, targets)
    labels = torch.arange(
        logits.shape[-1],
        device=logits.device,
    ).view(1, 1, -1)
    return (logits.argmax(dim=-1) == labels).float().mean()


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

    Mixed groups use exact answer correctness. Homogeneous groups may still
    yield a pair when the frozen gold-answer score has a gap above the stated
    numerical floor. Such pairs are marked ``continuous_outcome`` and are not
    reported as exact correct/wrong pairs.
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
            # Only the highest-scoring path anchors a continuous pair. The
            # remaining high-score paths define its local mode radius.
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
                    float(
                        correct_distances[correct_rank, peer_rank].item()
                    )
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


def group_standardize_with_floor(
    values: torch.Tensor,
    *,
    group_size: int,
    minimum_std: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize only groups whose variation clears a numerical floor."""
    if values.shape[0] % int(group_size):
        raise ValueError("first dimension must be divisible by group_size")
    if float(minimum_std) <= 0.0:
        raise ValueError("minimum_std must be positive")
    grouped = values.float().view(
        values.shape[0] // int(group_size),
        int(group_size),
        *values.shape[1:],
    )
    reduce_dims = tuple(range(1, grouped.ndim))
    mean = grouped.mean(dim=reduce_dims, keepdim=True)
    std = grouped.std(dim=reduce_dims, keepdim=True, unbiased=False)
    active = std >= float(minimum_std)
    standardized = torch.where(
        active,
        (grouped - mean) / std.clamp_min(float(minimum_std)),
        torch.zeros_like(grouped),
    )
    return (
        standardized.reshape_as(values),
        std.reshape(std.shape[0], -1).mean(dim=-1),
        active.reshape(active.shape[0], -1).all(dim=-1),
    )


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
