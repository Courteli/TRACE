import itertools
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
    ):
        super().__init__()
        if hidden_size <= 0 or action_dim <= 0 or n_steps <= 0:
            raise ValueError("hidden_size, action_dim, and n_steps must be positive")
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.n_steps = int(n_steps)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

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

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, float(initial_log_std))
        nn.init.normal_(
            self.dynamics_step_embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.hidden_size),
        )
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
        base = self.base_projector(states.float())
        step = self.dynamics_step_embedding(step_ids).float()
        action = self.action_projector(actions.float())
        return base + float(step_scale) * step + float(action_scale) * action

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


def permutation_invariant_set_matching(
    model_paths: torch.Tensor,
    teacher_paths: torch.Tensor,
    *,
    distance_kwargs: Optional[dict] = None,
    max_exact_set_size: int = 6,
) -> Dict[str, torch.Tensor]:
    """Exact set matching with no persistent model or teacher route identity."""
    if model_paths.ndim != 4 or teacher_paths.ndim != 4:
        raise ValueError("path sets must have shape [batch, path, step, hidden]")
    if model_paths.shape != teacher_paths.shape:
        raise ValueError("model and teacher path sets must have identical shapes")
    set_size = model_paths.shape[1]
    if set_size > int(max_exact_set_size):
        raise ValueError(
            f"exact matching supports at most {max_exact_set_size} paths"
        )
    distance_kwargs = distance_kwargs or {}
    component_matrices = {
        name: model_paths.new_zeros(
            model_paths.shape[0],
            set_size,
            set_size,
            dtype=torch.float32,
        )
        for name in ("total", "position", "direction", "step")
    }
    for model_index in range(set_size):
        for teacher_index in range(set_size):
            components = trajectory_distance_components(
                model_paths[:, model_index],
                teacher_paths[:, teacher_index].detach(),
                **distance_kwargs,
            )
            for name, value in components.items():
                component_matrices[name][
                    :,
                    model_index,
                    teacher_index,
                ] = value

    permutations = torch.tensor(
        list(itertools.permutations(range(set_size))),
        device=model_paths.device,
        dtype=torch.long,
    )
    model_indices = torch.arange(set_size, device=model_paths.device)
    totals = []
    for permutation in permutations:
        totals.append(
            component_matrices["total"][
                :,
                model_indices,
                permutation,
            ].mean(dim=-1)
        )
    permutation_costs = torch.stack(totals, dim=1)
    best_indices = permutation_costs.argmin(dim=1)
    assignments = permutations.index_select(0, best_indices)

    output = {
        "loss": permutation_costs.gather(
            1,
            best_indices.unsqueeze(1),
        ).mean(),
        "assignments": assignments.detach(),
        "permutation_costs": permutation_costs.detach(),
    }
    batch_indices = torch.arange(model_paths.shape[0], device=model_paths.device)
    for name in ("position", "direction", "step"):
        rows = []
        for model_index in range(set_size):
            rows.append(
                component_matrices[name][
                    batch_indices,
                    model_index,
                    assignments[:, model_index],
                ]
            )
        output[name] = torch.stack(rows, dim=1).mean()
    return output


def reorder_teacher_set(
    teacher_paths: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    if teacher_paths.ndim != 4 or assignments.ndim != 2:
        raise ValueError("teacher paths and assignments have invalid ranks")
    gather_indices = assignments[:, :, None, None].expand(
        -1,
        -1,
        teacher_paths.shape[2],
        teacher_paths.shape[3],
    )
    return torch.gather(teacher_paths, dim=1, index=gather_indices)


def relation_geometry_loss(
    model_paths: torch.Tensor,
    matched_teacher_paths: torch.Tensor,
    *,
    distance_kwargs: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    """Preserve the teacher set's pairwise geometry after set matching."""
    if model_paths.shape != matched_teacher_paths.shape:
        raise ValueError("matched model and teacher sets must have equal shapes")
    distance_kwargs = distance_kwargs or {}
    model_distances = []
    teacher_distances = []
    for left in range(model_paths.shape[1]):
        for right in range(left + 1, model_paths.shape[1]):
            model_distances.append(
                trajectory_distance(
                    model_paths[:, left],
                    model_paths[:, right],
                    **distance_kwargs,
                )
            )
            teacher_distances.append(
                trajectory_distance(
                    matched_teacher_paths[:, left].detach(),
                    matched_teacher_paths[:, right].detach(),
                    **distance_kwargs,
                )
            )
    if not model_distances:
        zero = model_paths.sum() * 0.0
        return {
            "loss": zero,
            "model_distance": zero.detach(),
            "teacher_distance": zero.detach(),
        }
    model_values = torch.stack(model_distances, dim=1)
    teacher_values = torch.stack(teacher_distances, dim=1)
    return {
        "loss": (model_values - teacher_values).abs().mean(),
        "model_distance": model_values.mean().detach(),
        "teacher_distance": teacher_values.mean().detach(),
    }


def path_noncollapse_loss(
    paths: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if paths.ndim != 4:
        raise ValueError("paths must have shape [batch, path, step, hidden]")
    normalized_norm = paths.float().norm(dim=-1) / math.sqrt(paths.shape[-1])
    return F.relu(float(margin) - normalized_norm).mean()


def build_teacher_rationale_schedule(
    *,
    n_available: int,
    set_size: int,
    max_semantic_modes: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[int], List[int]]:
    """Sample rationales uniformly, then allocate an equal-size teacher set."""
    if n_available <= 0 or set_size <= 0 or max_semantic_modes <= 0:
        raise ValueError("rationale schedule arguments must be positive")
    mode_count = min(n_available, max_semantic_modes, set_size)
    selected = torch.randperm(
        n_available,
        generator=generator,
        device="cpu",
    )[:mode_count].tolist()
    rationale_indices = [
        selected[position % mode_count] for position in range(set_size)
    ]
    semantic_modes = [
        position % mode_count for position in range(set_size)
    ]
    order = torch.randperm(
        set_size,
        generator=generator,
        device="cpu",
    ).tolist()
    return (
        [rationale_indices[index] for index in order],
        [semantic_modes[index] for index in order],
    )


@dataclass(frozen=True)
class HardPathPair:
    group_index: int
    correct_index: int
    correct_peer_index: int
    wrong_index: int
    correct_radius: float
    wrong_distance: float
    hinge: float


def mine_question_local_hard_pairs(
    residual_paths: torch.Tensor,
    correctness: torch.Tensor,
    *,
    group_size: int,
    margin: float,
    max_pairs_per_group: int = 1,
    distance_kwargs: Optional[dict] = None,
) -> List[HardPathPair]:
    """Select outcome-local positive/positive/nearest-negative relations."""
    if residual_paths.ndim != 3:
        raise ValueError("residual_paths must have shape [rollout, step, hidden]")
    if correctness.numel() != residual_paths.shape[0]:
        raise ValueError("correctness must match rollout count")
    if residual_paths.shape[0] % int(group_size):
        raise ValueError("rollout count must be divisible by group_size")
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
        if correct.numel() < 2 or wrong.numel() < 1:
            continue
        correct_paths = paths[start:end].index_select(0, correct)
        wrong_paths = paths[start:end].index_select(0, wrong)
        correct_distances = trajectory_distance(
            correct_paths[:, None],
            correct_paths[None, :],
            **distance_kwargs,
        )
        correct_distances.fill_diagonal_(float("inf"))
        correct_wrong = trajectory_distance(
            correct_paths[:, None],
            wrong_paths[None, :],
            **distance_kwargs,
        )
        candidates = []
        for correct_rank in range(correct.numel()):
            peer_rank = int(
                correct_distances[correct_rank].argmin().item()
            )
            wrong_rank = int(
                correct_wrong[correct_rank].argmin().item()
            )
            radius = float(
                correct_distances[correct_rank, peer_rank].item()
            )
            wrong_distance = float(
                correct_wrong[correct_rank, wrong_rank].item()
            )
            candidates.append(
                HardPathPair(
                    group_index=group_index,
                    correct_index=start + int(correct[correct_rank].item()),
                    correct_peer_index=start + int(correct[peer_rank].item()),
                    wrong_index=start + int(wrong[wrong_rank].item()),
                    correct_radius=radius,
                    wrong_distance=wrong_distance,
                    hinge=max(
                        0.0,
                        float(margin) + radius - wrong_distance,
                    ),
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
) -> Dict[str, torch.Tensor]:
    """Build causal transition swaps with recipient suffix innovations."""
    if actions.shape != innovations.shape or actions.ndim != 3:
        raise ValueError(
            "actions and innovations must have shape [rollout, step, action]"
        )
    n_steps = actions.shape[1]
    forced_actions = []
    forced_masks = []
    suffix_innovations = []
    pair_indices = []
    step_indices = []
    directions = []
    for pair_index, pair in enumerate(pairs):
        for step_index in range(n_steps):
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
    return credits / counts.clamp_min(1.0)


def group_standardize(
    values: torch.Tensor,
    *,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    if values.shape[0] % int(group_size):
        raise ValueError("first dimension must be divisible by group_size")
    grouped = values.view(
        values.shape[0] // int(group_size),
        int(group_size),
        *values.shape[1:],
    )
    reduce_dims = tuple(range(1, grouped.ndim))
    mean = grouped.mean(dim=reduce_dims, keepdim=True)
    std = grouped.std(dim=reduce_dims, keepdim=True, unbiased=False)
    standardized = (grouped - mean) / std.clamp_min(float(eps))
    return standardized.reshape_as(values)


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
        advantages[pair.correct_index] += (
            0.5 * float(local_weight) * local_signal
        )
        advantages[pair.correct_peer_index] += (
            0.5 * float(local_weight) * local_signal
        )
        advantages[pair.wrong_index] -= (
            float(local_weight) * local_signal
        )
    return group_standardize(advantages, group_size=group_size)


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
