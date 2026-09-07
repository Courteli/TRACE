import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .read_stable_efficient import LitREADCoTStableEfficient
from .trace_bridge import LitTRACEBridge
from ..modules import grpo
from ..modules.readcot import aggregate_step_residuals


def _safe_cosine_distance(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine distance with zero/zero treated as identical."""
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    cosine = F.cosine_similarity(left, right, dim=-1, eps=eps)
    distance = 1.0 - cosine
    both_zero = (left_norm <= eps) & (right_norm <= eps)
    exactly_one_zero = (left_norm <= eps) ^ (right_norm <= eps)
    distance = torch.where(both_zero, torch.zeros_like(distance), distance)
    distance = torch.where(exactly_one_zero, torch.ones_like(distance), distance)
    return torch.nan_to_num(distance, nan=0.0, posinf=2.0, neginf=0.0)


def _path_anchor_indices(n_steps: int, anchor_count: int, device: torch.device) -> torch.Tensor:
    if n_steps <= 0:
        raise ValueError("A trajectory must contain at least one transition")
    anchor_count = max(1, min(int(anchor_count), n_steps))
    fractions = torch.linspace(
        1.0 / anchor_count,
        1.0,
        steps=anchor_count,
        device=device,
        dtype=torch.float32,
    )
    return torch.round((n_steps - 1) * fractions).long().unique(sorted=True)


def trace_path_distance_components(
    left_residuals: torch.Tensor,
    right_residuals: torch.Tensor,
    *,
    anchor_count: int = 3,
    position_weight: float = 0.45,
    direction_weight: float = 0.35,
    step_weight: float = 0.15,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compare complete residual trajectories without a compressed signature.

    Leading dimensions may broadcast. The final two dimensions must be
    ``[transition, hidden]``.
    """
    if left_residuals.ndim < 2 or right_residuals.ndim < 2:
        raise ValueError("Path tensors must end in [transition, hidden]")
    if left_residuals.shape[-2:] != right_residuals.shape[-2:]:
        raise ValueError(
            "Path shapes must agree on [transition, hidden], got "
            f"{tuple(left_residuals.shape[-2:])} and {tuple(right_residuals.shape[-2:])}"
        )

    left = left_residuals.float()
    right = right_residuals.float()
    n_steps = left.shape[-2]
    anchors = _path_anchor_indices(n_steps, anchor_count, left.device)

    left_positions = left.cumsum(dim=-2).index_select(-2, anchors)
    right_positions = right.cumsum(dim=-2).index_select(-2, anchors)
    position = _safe_cosine_distance(left_positions, right_positions, eps=eps).mean(dim=-1)

    # Residual t is the transition from z_{t-1} to z_t, including origin -> z_1.
    direction = _safe_cosine_distance(left, right, eps=eps).mean(dim=-1)

    norm_scale = math.sqrt(left.shape[-1])
    left_steps = left.norm(dim=-1) / norm_scale
    right_steps = right.norm(dim=-1) / norm_scale
    step = ((left_steps - right_steps).abs() / (left_steps + right_steps + eps)).mean(dim=-1)

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


def trace_path_distance(
    left_residuals: torch.Tensor,
    right_residuals: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return trace_path_distance_components(left_residuals, right_residuals, **kwargs)["total"]


class ExchangeablePathSeedMap(nn.Module):
    """Fixed continuous seed map with no trainable or persistent identity table."""

    def __init__(
        self,
        seed_dim: int,
        n_steps: int,
        hidden_size: int,
        *,
        basis_seed: int = 1729,
        n_controls: int = 3,
    ):
        super().__init__()
        if seed_dim <= 0 or n_steps <= 0 or hidden_size <= 0:
            raise ValueError("seed_dim, n_steps, and hidden_size must be positive")
        self.seed_dim = int(seed_dim)
        self.n_steps = int(n_steps)
        self.hidden_size = int(hidden_size)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(basis_seed))
        latent_basis = torch.randn(
            self.seed_dim,
            self.n_steps,
            self.hidden_size,
            generator=generator,
            dtype=torch.float32,
        )
        latent_basis = latent_basis / latent_basis.std(dim=-1, keepdim=True).clamp_min(1e-6)
        control_basis = torch.randn(
            self.seed_dim,
            int(n_controls),
            generator=generator,
            dtype=torch.float32,
        )
        # The bases are deterministic implementation constants, not checkpoint state.
        self.register_buffer("latent_basis", latent_basis, persistent=False)
        self.register_buffer("control_basis", control_basis, persistent=False)

    def latent_noise(self, seeds: torch.Tensor) -> torch.Tensor:
        self._validate(seeds)
        return torch.einsum(
            "bd,dth->bth",
            seeds.float(),
            self.latent_basis,
        ) / math.sqrt(self.seed_dim)

    def controls(self, seeds: torch.Tensor) -> torch.Tensor:
        self._validate(seeds)
        controls = torch.matmul(seeds.float(), self.control_basis) / math.sqrt(self.seed_dim)
        return torch.tanh(controls)

    def _validate(self, seeds: torch.Tensor):
        if seeds.ndim != 2 or seeds.shape[-1] != self.seed_dim:
            raise ValueError(
                f"Expected path seeds [batch, {self.seed_dim}], got {tuple(seeds.shape)}"
            )


@dataclass(frozen=True)
class LocalRankingTriplet:
    group_index: int
    correct_index: int
    peer_index: int
    wrong_index: int
    correct_radius: float
    wrong_distance: float
    hinge: float
    active: bool


def build_local_ranking_triplets(
    residual_paths: torch.Tensor,
    correctness: torch.Tensor,
    *,
    group_size: int,
    margin: float,
    distance_kwargs: Optional[dict] = None,
) -> List[LocalRankingTriplet]:
    """Mine question-local nearest-correct triplets from detached rollouts."""
    if residual_paths.ndim != 3:
        raise ValueError(f"Expected [rollout, transition, hidden], got {tuple(residual_paths.shape)}")
    if correctness.numel() != residual_paths.shape[0]:
        raise ValueError("Correctness labels must match the rollout count")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    distance_kwargs = distance_kwargs or {}
    paths = residual_paths.detach()
    labels = correctness.detach().view(-1).to(paths.device) > 0.5
    triplets: List[LocalRankingTriplet] = []
    group_index = 0
    for start in range(0, paths.shape[0], group_size):
        end = min(start + group_size, paths.shape[0])
        local_labels = labels[start:end]
        correct_local = torch.nonzero(local_labels, as_tuple=False).flatten()
        wrong_local = torch.nonzero(~local_labels, as_tuple=False).flatten()
        if correct_local.numel() < 2 or wrong_local.numel() < 1:
            group_index += 1
            continue

        correct_paths = paths[start:end].index_select(0, correct_local)
        wrong_paths = paths[start:end].index_select(0, wrong_local)
        correct_pairwise = trace_path_distance(
            correct_paths[:, None, :, :],
            correct_paths[None, :, :, :],
            **distance_kwargs,
        )
        correct_pairwise.fill_diagonal_(float("inf"))
        correct_wrong = trace_path_distance(
            correct_paths[:, None, :, :],
            wrong_paths[None, :, :, :],
            **distance_kwargs,
        )

        nearest_correct_for_wrong = correct_wrong.argmin(dim=0)
        for wrong_rank, correct_rank_tensor in enumerate(nearest_correct_for_wrong):
            correct_rank = int(correct_rank_tensor.item())
            peer_rank = int(correct_pairwise[correct_rank].argmin().item())
            correct_radius = float(correct_pairwise[correct_rank, peer_rank].item())
            wrong_distance = float(correct_wrong[correct_rank, wrong_rank].item())
            hinge = max(0.0, float(margin) + correct_radius - wrong_distance)
            triplets.append(
                LocalRankingTriplet(
                    group_index=group_index,
                    correct_index=start + int(correct_local[correct_rank].item()),
                    peer_index=start + int(correct_local[peer_rank].item()),
                    wrong_index=start + int(wrong_local[wrong_rank].item()),
                    correct_radius=correct_radius,
                    wrong_distance=wrong_distance,
                    hinge=hinge,
                    active=hinge > 0.0,
                )
            )
        group_index += 1
    return triplets


def local_ranking_surrogate(
    current_paths: torch.Tensor,
    reference_paths: torch.Tensor,
    triplets: Sequence[LocalRankingTriplet],
    *,
    distance_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """Frozen-mining surrogate whose gradient equals the active hinge gradient."""
    distance_kwargs = distance_kwargs or {}
    denominator = float(max(1, len(triplets)))
    loss = current_paths.sum() * 0.0
    for triplet in triplets:
        if not triplet.active:
            continue
        c = triplet.correct_index
        p = triplet.peer_index
        w = triplet.wrong_index
        loss = loss + trace_path_distance(
            current_paths[c],
            reference_paths[p].detach(),
            **distance_kwargs,
        )
        loss = loss - trace_path_distance(
            current_paths[c],
            reference_paths[w].detach(),
            **distance_kwargs,
        )
        loss = loss + trace_path_distance(
            current_paths[p],
            reference_paths[c].detach(),
            **distance_kwargs,
        )
        loss = loss - trace_path_distance(
            current_paths[w],
            reference_paths[c].detach(),
            **distance_kwargs,
        )
    return loss / denominator


def merge_accuracy_guarded_gradients(
    task_gradients: Sequence[Optional[torch.Tensor]],
    ranking_gradients: Sequence[Optional[torch.Tensor]],
    *,
    max_ratio: float,
) -> Tuple[List[Optional[torch.Tensor]], Dict[str, torch.Tensor]]:
    """Project conflicting ranking gradients and cap their global norm."""
    if len(task_gradients) != len(ranking_gradients):
        raise ValueError("Task and ranking gradient lists must have the same length")
    reference = next(
        (
            gradient
            for gradient in list(task_gradients) + list(ranking_gradients)
            if gradient is not None
        ),
        None,
    )
    if reference is None:
        zero = torch.zeros((), dtype=torch.float32)
        return [None] * len(task_gradients), {
            "task_grad_norm": zero,
            "ranking_grad_norm": zero,
            "projected_ranking_grad_norm": zero,
            "ranking_scale": zero,
            "task_ranking_cosine": zero,
            "conflict": zero,
        }

    device = reference.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    task_norm_sq = zero.clone()
    ranking_norm_sq = zero.clone()
    task_ranking_dot = zero.clone()
    for task_gradient, ranking_gradient in zip(task_gradients, ranking_gradients):
        if task_gradient is not None:
            task_norm_sq += task_gradient.float().square().sum()
        if ranking_gradient is not None:
            ranking_norm_sq += ranking_gradient.float().square().sum()
            if task_gradient is not None:
                task_ranking_dot += (task_gradient.float() * ranking_gradient.float()).sum()

    eps = torch.tensor(1e-12, device=device, dtype=torch.float32)
    task_norm = task_norm_sq.sqrt()
    ranking_norm = ranking_norm_sq.sqrt()
    conflict_dot = torch.minimum(task_ranking_dot, zero)
    projection_coefficient = conflict_dot / task_norm_sq.clamp_min(eps)
    projected_norm_sq = (
        ranking_norm_sq - conflict_dot.square() / task_norm_sq.clamp_min(eps)
    ).clamp_min(0.0)
    projected_norm = projected_norm_sq.sqrt()
    ranking_scale = torch.minimum(
        torch.ones((), device=device, dtype=torch.float32),
        float(max_ratio) * task_norm / projected_norm.clamp_min(eps),
    )
    if task_norm.item() == 0.0:
        ranking_scale.zero_()

    merged: List[Optional[torch.Tensor]] = []
    for task_gradient, ranking_gradient in zip(task_gradients, ranking_gradients):
        if task_gradient is None and ranking_gradient is None:
            merged.append(None)
            continue
        projected = None
        if ranking_gradient is not None:
            projected = ranking_gradient.float()
            if task_gradient is not None and conflict_dot.item() < 0.0:
                projected = projected - projection_coefficient * task_gradient.float()
            projected = ranking_scale * projected
        if task_gradient is None:
            merged.append(projected)
        elif projected is None:
            merged.append(task_gradient.float())
        else:
            merged.append(task_gradient.float() + projected)

    cosine = task_ranking_dot / (task_norm * ranking_norm).clamp_min(eps)
    return merged, {
        "task_grad_norm": task_norm,
        "ranking_grad_norm": ranking_norm,
        "projected_ranking_grad_norm": projected_norm,
        "ranking_scale": ranking_scale,
        "task_ranking_cosine": cosine,
        "conflict": (task_ranking_dot < 0).float(),
    }


class LitTRACEExchangeable(LitTRACEBridge):
    """TRACE with exchangeable paths and outcome-conditioned local refinement."""

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            all_config=all_config,
        )
        if self.trace_config.get("use_trace_view_embeddings", False):
            raise ValueError("Exchangeable TRACE forbids categorical trace-view embeddings")
        if self.trace_config.get("use_trace_step_view_embeddings", False):
            raise ValueError("Exchangeable TRACE forbids categorical trace-step-view embeddings")
        if hasattr(self, "trace_view_embeddings") or hasattr(self, "trace_step_view_embeddings"):
            raise RuntimeError("A fixed view-identity table was unexpectedly constructed")

        seed_dim = int(self.trace_config.get("path_seed_dim", 16))
        self.path_seed_map = ExchangeablePathSeedMap(
            seed_dim=seed_dim,
            n_steps=self.max_trace_latents,
            hidden_size=self.hidden_size,
            basis_seed=int(self.trace_config.get("path_seed_basis_seed", 1729)),
        )
        self.path_seed_dim = seed_dim

    def _distance_kwargs(self) -> dict:
        return {
            "anchor_count": int(self.trace_config.get("path_anchor_count", 3)),
            "position_weight": float(self.trace_config.get("path_position_mix", 0.45)),
            "direction_weight": float(self.trace_config.get("path_direction_mix", 0.35)),
            "step_weight": float(self.trace_config.get("path_step_mix", 0.15)),
        }

    def _sample_path_seeds(self, count: int) -> torch.Tensor:
        return torch.randn(count, self.path_seed_dim, device=self.device, dtype=torch.float32)

    def _path_seed_noise(self, seeds: torch.Tensor) -> torch.Tensor:
        return self.path_seed_map.latent_noise(seeds).to(self.device)

    def _question_only_latents(
        self,
        questions: Sequence[str],
        trace_view_ids: Optional[torch.Tensor] = None,
        trace_noise_std: Optional[float] = None,
        trace_latent_noise: Optional[torch.Tensor] = None,
        trace_latent_noise_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if trace_view_ids is not None and torch.any(trace_view_ids != 0):
            raise ValueError("Categorical view IDs are not valid in exchangeable TRACE")
        if trace_latent_noise is not None and trace_latent_noise_scale is None:
            trace_latent_noise_scale = float(
                self.trace_config.get("path_seed_latent_scale", 0.012)
            )
        return super()._question_only_latents(
            questions=questions,
            trace_view_ids=None,
            trace_noise_std=trace_noise_std,
            trace_latent_noise=trace_latent_noise,
            trace_latent_noise_scale=trace_latent_noise_scale,
        )

    def _make_seeded_assignment(
        self,
        assignment: torch.Tensor,
        path_seed: torch.Tensor,
    ) -> torch.Tensor:
        controls = self.path_seed_map.controls(path_seed.view(1, -1))[0]
        logits = torch.log(assignment.detach().float().clamp_min(1e-6))
        temperature = torch.exp(
            float(self.trace_config.get("seed_teacher_log_temperature_scale", 0.08))
            * controls[0]
        ).clamp(min=0.5, max=2.0)
        scores = logits / temperature

        n_slots, n_steps = assignment.shape
        if n_steps > 1:
            slot_pos = torch.linspace(
                0.0,
                1.0,
                n_slots,
                device=assignment.device,
                dtype=torch.float32,
            ).unsqueeze(1)
            step_pos = torch.linspace(
                0.0,
                1.0,
                n_steps,
                device=assignment.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            shift = (
                float(self.trace_config.get("seed_teacher_max_progress_shift", 0.05))
                * controls[1]
            )
            sigma = max(
                float(self.trace_config.get("seed_teacher_progress_sigma", 0.18)),
                1e-3,
            )
            base_bias = -((step_pos - slot_pos) ** 2) / (2.0 * sigma * sigma)
            shifted_center = (slot_pos + shift).clamp(0.0, 1.0)
            shifted_bias = -((step_pos - shifted_center) ** 2) / (2.0 * sigma * sigma)
            scores = scores + float(
                self.trace_config.get("seed_teacher_progress_strength", 0.4)
            ) * (shifted_bias - base_bias)

            slot_index = torch.arange(
                1,
                n_slots + 1,
                device=assignment.device,
                dtype=torch.float32,
            ).unsqueeze(1)
            step_index = torch.arange(
                1,
                n_steps + 1,
                device=assignment.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            pattern = torch.sin(
                math.pi * slot_index * step_index / float(max(n_slots, n_steps))
            )
            scores = scores + float(
                self.trace_config.get("seed_teacher_pattern_strength", 0.02)
            ) * controls[2] * pattern

        seeded = torch.softmax(scores, dim=-1)
        original_mix = float(self.trace_config.get("seed_teacher_original_mix", 0.20))
        if original_mix > 0:
            seeded = (1.0 - original_mix) * seeded + original_mix * assignment.detach().float()
            seeded = seeded / seeded.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return seeded.to(assignment.dtype)

    def _make_seeded_compression_outputs(
        self,
        explicit_features: List[Dict[str, torch.Tensor]],
        compression_outputs: Dict[str, torch.Tensor],
        path_seeds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        targets = []
        assignments = []
        for sample_idx, explicit_item in enumerate(explicit_features):
            assignment = self._make_seeded_assignment(
                compression_outputs["assignments"][sample_idx],
                path_seeds[sample_idx],
            )
            assignments.append(assignment)
            targets.append(
                aggregate_step_residuals(
                    assignment,
                    explicit_item["step_residuals"],
                )
            )
        output = dict(compression_outputs)
        output["assignments"] = torch.stack(assignments, dim=0)
        output["aggregated_explicit_residuals"] = torch.stack(targets, dim=0)
        return output

    def _exchangeable_path_alignment(
        self,
        predicted_residuals: torch.Tensor,
        target_residuals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        components = trace_path_distance_components(
            predicted_residuals,
            target_residuals.detach(),
            **self._distance_kwargs(),
        )
        norm_scale = math.sqrt(predicted_residuals.shape[-1])
        predicted_step = predicted_residuals.float().norm(dim=-1) / norm_scale
        target_step = target_residuals.detach().float().norm(dim=-1) / norm_scale
        noncollapse = F.relu(
            float(self.trace_config.get("path_noncollapse_margin", 0.02))
            - predicted_step
        ).mean(dim=-1)
        noncollapse_weight = float(self.trace_config.get("path_noncollapse_mix", 0.05))
        total = components["total"] + noncollapse_weight * noncollapse
        return {
            "trace_stage1_path_loss": total.mean(),
            "trace_stage1_position_loss": components["position"].mean(),
            "trace_stage1_direction_loss": components["direction"].mean(),
            "trace_stage1_step_loss": components["step"].mean(),
            "trace_stage1_noncollapse_loss": noncollapse.mean(),
            "trace_stage1_pred_step_norm": predicted_step.mean().detach(),
            "trace_stage1_target_step_norm": target_step.mean().detach(),
        }

    @staticmethod
    def _mean_scalar_dicts(items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = set.intersection(*(set(item.keys()) for item in items))
        return {
            key: torch.stack([item[key] for item in items]).mean()
            for key in sorted(keys)
        }

    def forward(self, batch):
        if not bool(self.trace_config.get("enable_trajectory_formation", True)):
            return LitREADCoTStableEfficient.forward(self, batch)

        questions = batch["question"]
        answers = batch["answer"]
        step_lists = self._decode_step_lists(batch)
        dependency_matrices = self._decode_cached_matrices(batch, "dependency_matrix")
        confidence_matrices = self._decode_cached_matrices(batch, "confidence_matrix")
        anchor_indices = self._decode_anchor_indices(batch)
        anchor_bonus_indices = self._decode_anchor_bonus_indices(batch)

        explicit_features = self._collect_explicit_batch_features(
            questions=questions,
            step_lists=step_lists,
            answers=answers,
            dependency_matrices=dependency_matrices,
            confidence_matrices=confidence_matrices,
        )
        base_compression = self._compress_explicit_reasoning(explicit_features)
        base_compression = self._apply_trace_progress_anchors(
            explicit_features,
            base_compression,
        )

        seed_pair = self._sample_path_seeds(2 * len(questions)).view(
            2,
            len(questions),
            self.path_seed_dim,
        )
        latent_branches = []
        teacher_branches = []
        residual_branches = []
        path_branches = []
        for branch_idx in range(2):
            teacher = self._make_seeded_compression_outputs(
                explicit_features,
                base_compression,
                seed_pair[branch_idx],
            )
            latent = self._question_only_latents(
                questions,
                trace_noise_std=0.0,
                trace_latent_noise=self._path_seed_noise(seed_pair[branch_idx]),
            )
            residual = self._compute_residual_alignment(
                implicit_residuals=latent["implicit_residuals"],
                aggregated_explicit_residuals=teacher["aggregated_explicit_residuals"],
            )
            path_logs = self._exchangeable_path_alignment(
                latent["implicit_residuals"],
                teacher["aggregated_explicit_residuals"],
            )
            latent_branches.append(latent)
            teacher_branches.append(teacher)
            residual_branches.append(residual)
            path_branches.append(path_logs)

        residual_outputs = self._mean_scalar_dicts(
            [
                {
                    key: value
                    for key, value in residual.items()
                    if isinstance(value, torch.Tensor) and value.ndim == 0
                }
                for residual in residual_branches
            ]
        )
        trace_logs = self._mean_scalar_dicts(path_branches)

        model_relation_components = trace_path_distance_components(
            latent_branches[0]["implicit_residuals"],
            latent_branches[1]["implicit_residuals"],
            **self._distance_kwargs(),
        )
        teacher_relation_components = trace_path_distance_components(
            teacher_branches[0]["aggregated_explicit_residuals"].detach(),
            teacher_branches[1]["aggregated_explicit_residuals"].detach(),
            **self._distance_kwargs(),
        )
        model_relation = model_relation_components["total"]
        teacher_relation = teacher_relation_components["total"]
        relation_error = (model_relation - teacher_relation).abs().mean()
        trace_logs.update(
            {
                "trace_stage1_relation_loss": relation_error,
                "trace_stage1_model_pair_distance": model_relation.mean().detach(),
                "trace_stage1_teacher_pair_distance": teacher_relation.mean().detach(),
                "trace_stage1_model_pair_position": model_relation_components[
                    "position"
                ].mean().detach(),
                "trace_stage1_model_pair_direction": model_relation_components[
                    "direction"
                ].mean().detach(),
                "trace_stage1_model_pair_step": model_relation_components[
                    "step"
                ].mean().detach(),
                "trace_stage1_teacher_pair_position": teacher_relation_components[
                    "position"
                ].mean().detach(),
                "trace_stage1_teacher_pair_direction": teacher_relation_components[
                    "direction"
                ].mean().detach(),
                "trace_stage1_teacher_pair_step": teacher_relation_components[
                    "step"
                ].mean().detach(),
            }
        )
        if "trace_progress_anchor_span" in base_compression:
            trace_logs["trace_stage1_progress_anchor_span"] = base_compression[
                "trace_progress_anchor_span"
            ]
            trace_logs["trace_stage1_progress_anchor_mean"] = base_compression[
                "trace_progress_anchor_mean"
            ]

        # The anchor-supervised branch is selected uniformly each step; neither
        # iid seed receives a persistent primary role.
        supervised_branch = int(torch.randint(0, 2, (1,), device=self.device).item())
        latent_outputs = latent_branches[supervised_branch]
        compression_outputs = teacher_branches[supervised_branch]

        answer_weight = self._scheduled_loss_weight(
            "answer",
            self.readcot_config.get("answer_loss_weight", 1.0),
        )
        if answer_weight > 0:
            answer_targets = [self.answer_template.format(answer) for answer in answers]
            answer_loss = self._teacher_force_target(
                past_key_values=latent_outputs["past_key_values"],
                context_attention_mask=latent_outputs["context_attention_mask"],
                target_texts=answer_targets,
            )
        else:
            answer_loss = latent_outputs["implicit_residuals"].new_zeros(())

        anchor_loss = answer_loss.new_zeros(())
        anchor_gate_loss = answer_loss.new_zeros(())
        use_anchor_supervision = self.readcot_config.get(
            "use_hybrid",
            False,
        ) or self.readcot_config.get("use_anchor_loss", False)
        if use_anchor_supervision:
            build_outputs = self._build_hybrid_targets(
                step_lists=step_lists,
                answers=answers,
                explicit_features=explicit_features,
                compression_outputs=compression_outputs,
                implicit_residuals=latent_outputs["implicit_residuals"],
                anchor_indices=anchor_indices,
                anchor_bonus_indices=anchor_bonus_indices,
                return_anchor_gate_targets=self.use_anchor_gate,
            )
            if self.use_anchor_gate:
                anchor_targets, _, anchor_gate_targets = build_outputs
            else:
                anchor_targets, _ = build_outputs
            prefix_texts = None
            if self.readcot_config.get("hybrid_seed_anchor_header", False):
                prefix = self.anchor_header + "\n"
                prefix_texts = [prefix] * len(anchor_targets)
                anchor_targets = [
                    target[len(prefix) :] if target.startswith(prefix) else target
                    for target in anchor_targets
                ]
            anchor_loss = self._teacher_force_target(
                past_key_values=latent_outputs["past_key_values"],
                context_attention_mask=latent_outputs["context_attention_mask"],
                target_texts=anchor_targets,
                prefix_texts=prefix_texts,
            )
            if self.use_anchor_gate and latent_outputs.get("anchor_gate") is not None:
                gate_target = torch.stack(anchor_gate_targets, dim=0).detach()
                anchor_gate_loss = F.smooth_l1_loss(
                    latent_outputs["anchor_gate"],
                    gate_target,
                )

        dep_weight = self._scheduled_loss_weight("dep", self.readcot_config.lambda_dep)
        res_weight = self._scheduled_loss_weight("res", self.readcot_config.lambda_res)
        anchor_weight = self._scheduled_loss_weight(
            "anchor",
            self.readcot_config.lambda_anchor,
        )
        anchor_gate_weight = self._scheduled_loss_weight(
            "anchor_gate",
            self.readcot_config.get("lambda_anchor_gate", 0.0),
        )
        path_weight = float(self.trace_config.get("stage1_path_weight", 0.14))
        relation_weight = float(
            self.trace_config.get("stage1_view_relation_weight", 0.08)
        )

        total_loss = answer_weight * answer_loss
        total_loss = total_loss + dep_weight * base_compression["dep_loss"]
        total_loss = total_loss + res_weight * residual_outputs["residual_loss"]
        if use_anchor_supervision:
            total_loss = total_loss + anchor_weight * anchor_loss
        if self.use_anchor_gate:
            total_loss = total_loss + anchor_gate_weight * anchor_gate_loss
        total_loss = total_loss + path_weight * trace_logs["trace_stage1_path_loss"]
        total_loss = total_loss + relation_weight * relation_error

        result = {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "dep_loss": base_compression["dep_loss"],
            "dep_f1": base_compression["dep_f1"],
            "residual_loss": residual_outputs["residual_loss"],
            "residual_mse_loss": residual_outputs["residual_mse_loss"],
            "residual_cosine_distance": residual_outputs["residual_cosine_distance"],
            "residual_similarity": residual_outputs["residual_similarity"],
            "anchor_loss": anchor_loss,
            "anchor_gate_loss": anchor_gate_loss,
            "lambda_answer_eff": answer_loss.new_tensor(answer_weight),
            "lambda_dep_eff": answer_loss.new_tensor(dep_weight),
            "lambda_res_eff": answer_loss.new_tensor(res_weight),
            "lambda_anchor_eff": answer_loss.new_tensor(anchor_weight),
            "lambda_anchor_gate_eff": answer_loss.new_tensor(anchor_gate_weight),
            "lambda_trace_path_eff": answer_loss.new_tensor(path_weight),
            "lambda_trace_relation_eff": answer_loss.new_tensor(relation_weight),
        }
        result.update(trace_logs)
        return result

    @torch.no_grad()
    def trace_rollout(self, questions: List[str], gt_answers):
        batch_size = len(questions)
        group_size = int(self.trace_rl_config.get("group_size", 8))
        group_questions = [
            question
            for question in questions
            for _ in range(group_size)
        ]
        path_seeds = self._sample_path_seeds(len(group_questions))
        latent_noise = self._path_seed_noise(path_seeds)

        pred_ids, n_latent_forward, latent_outputs = self._read_generate_with_latents_in_chunks(
            group_questions,
            trace_view_ids=None,
            trace_latent_noise=latent_noise,
            do_sample=True,
            temperature=float(self.trace_rl_config.get("temperature", 0.95)),
            top_p=float(self.trace_rl_config.get("top_p", 0.97)),
        )
        answer_attention_mask = pred_ids.ne(self.tokenizer.pad_token_id).long()
        pred_strings = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        output_lengths = answer_attention_mask.sum(dim=1).float()
        rollout_paths = latent_outputs["implicit_residuals"].float().detach()

        rewards = []
        accuracies = []
        advantages = []
        pos_counts = []
        mixed_flags = []
        eligible_flags = []
        length_weight = float(
            self.trace_rl_config.get("output_length_penalty_weight", 0.01)
        )
        target_length = float(self.trace_rl_config.get("target_output_length", 34.0))
        for sample_idx in range(batch_size):
            start = sample_idx * group_size
            end = start + group_size
            group_accuracy = torch.zeros(group_size, 1, device=self.device)
            for local_idx, prediction in enumerate(pred_strings[start:end]):
                predicted_answer = self.extract_answer_from_output(prediction)
                group_accuracy[local_idx] = self.verify_answer(
                    gt_answer=gt_answers[sample_idx],
                    pred_answer=predicted_answer,
                )
            group_reward = group_accuracy.view(-1).float()
            if length_weight > 0:
                group_reward = group_reward - length_weight * F.relu(
                    output_lengths[start:end] / max(target_length, 1.0) - 1.0
                )
            rewards.append(group_reward.view(-1, 1))
            accuracies.append(group_accuracy)
            advantages.append(grpo.group_advantages(group_reward).view(-1, 1))
            pos_count = int(group_accuracy.sum().item())
            pos_counts.append(float(pos_count))
            mixed_flags.append(float(0 < pos_count < group_size))
            eligible_flags.append(float(2 <= pos_count < group_size))

        rewards_tensor = torch.cat(rewards, dim=0)
        accuracies_tensor = torch.cat(accuracies, dim=0)
        advantages_tensor = torch.cat(advantages, dim=0)
        old_answer_logprobs = self._answer_logprobs_for_questions_in_chunks(
            questions=group_questions,
            answer_input_ids=pred_ids,
            answer_attention_mask=answer_attention_mask,
            trace_view_ids=None,
            trace_latent_noise=latent_noise,
        )
        latent_logprobs = torch.zeros_like(
            latent_outputs["latent_attention_mask"],
            dtype=old_answer_logprobs.dtype,
        )

        self._last_trace_metrics = {
            "trace_rl/pos_count": torch.tensor(
                float(np.mean(pos_counts)),
                device=self.device,
            ),
            "trace_rl/mixed_frac": torch.tensor(
                float(np.mean(mixed_flags)),
                device=self.device,
            ),
            "trace_rl/ranking_eligible_frac": torch.tensor(
                float(np.mean(eligible_flags)),
                device=self.device,
            ),
            "trace_rl/path_bonus": torch.zeros((), device=self.device),
        }
        experience = grpo.Experience(
            latent_logprobs=latent_logprobs,
            answer_logprobs=old_answer_logprobs,
            question_input_ids=latent_outputs["question_input_ids"],
            question_attention_mask=latent_outputs["question_attention_mask"],
            latent_inputs_embeds=latent_outputs["latent_inputs_embeds"],
            latent_attention_mask=latent_outputs["latent_attention_mask"],
            answer_input_ids=pred_ids,
            answer_attention_mask=answer_attention_mask,
            n_latent_forward=n_latent_forward,
            rewards=rewards_tensor,
            accuracies=accuracies_tensor,
            advantages=advantages_tensor,
        )
        return (
            experience,
            group_questions,
            path_seeds.detach(),
            latent_noise.detach(),
            rollout_paths,
        )

    def backward_stage2_local_ranking(
        self,
        group_questions: List[str],
        accuracies: torch.Tensor,
        rollout_paths: torch.Tensor,
        latent_noise: torch.Tensor,
        *,
        loss_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        group_size = int(self.trace_rl_config.get("group_size", 8))
        margin = float(self.trace_rl_config.get("stage2_local_ranking_margin", 0.08))
        triplets = build_local_ranking_triplets(
            residual_paths=rollout_paths,
            correctness=accuracies,
            group_size=group_size,
            margin=margin,
            distance_kwargs=self._distance_kwargs(),
        )
        active_triplets = [triplet for triplet in triplets if triplet.active]
        path_plans: Dict[int, List[Tuple[float, int]]] = {}
        for triplet in active_triplets:
            path_plans.setdefault(triplet.correct_index, []).extend(
                [
                    (1.0, triplet.peer_index),
                    (-1.0, triplet.wrong_index),
                ]
            )
            path_plans.setdefault(triplet.peer_index, []).append(
                (1.0, triplet.correct_index)
            )
            path_plans.setdefault(triplet.wrong_index, []).append(
                (-1.0, triplet.correct_index)
            )

        denominator = float(max(1, len(triplets)))
        selected_indices = sorted(path_plans)
        micro_batch = max(
            1,
            int(self.trace_rl_config.get("stage2_ranking_micro_batch_size", 1)),
        )
        local_chunk_count = math.ceil(len(selected_indices) / micro_batch)
        synchronized_chunk_count = local_chunk_count
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            chunk_count_tensor = torch.tensor(
                local_chunk_count,
                device=latent_noise.device,
                dtype=torch.long,
            )
            torch.distributed.all_reduce(
                chunk_count_tensor,
                op=torch.distributed.ReduceOp.MAX,
            )
            synchronized_chunk_count = int(chunk_count_tensor.item())

        dummy_chunk_count = 0
        for chunk_number in range(synchronized_chunk_count):
            chunk_start = chunk_number * micro_batch
            chunk_indices = selected_indices[chunk_start : chunk_start + micro_batch]
            is_dummy = not chunk_indices
            if is_dummy:
                # DDP ranks must execute the same number of backward passes. A
                # zero-valued path graph participates in synchronization without
                # inventing a ranking signal for an ineligible local group.
                chunk_indices = [0]
                dummy_chunk_count += 1
            index_tensor = torch.tensor(
                chunk_indices,
                device=latent_noise.device,
                dtype=torch.long,
            )
            current_outputs = self._question_only_latents(
                [group_questions[index] for index in chunk_indices],
                trace_noise_std=0.0,
                trace_latent_noise=latent_noise.index_select(0, index_tensor),
            )
            current_paths = current_outputs["implicit_residuals"].float()
            chunk_loss = current_paths.sum() * 0.0
            if not is_dummy:
                for local_index, global_index in enumerate(chunk_indices):
                    for coefficient, peer_index in path_plans[global_index]:
                        chunk_loss = chunk_loss + coefficient * trace_path_distance(
                            current_paths[local_index],
                            rollout_paths[peer_index].detach(),
                            **self._distance_kwargs(),
                        )
            self.manual_backward(float(loss_weight) * chunk_loss / denominator)
            del current_outputs, current_paths, chunk_loss

        zero = rollout_paths.new_zeros(())
        rank_loss = (
            rollout_paths.new_tensor([triplet.hinge for triplet in triplets]).mean()
            if triplets
            else zero
        )
        correct_radius = (
            rollout_paths.new_tensor(
                [triplet.correct_radius for triplet in triplets]
            ).mean()
            if triplets
            else zero
        )
        wrong_distance = (
            rollout_paths.new_tensor(
                [triplet.wrong_distance for triplet in triplets]
            ).mean()
            if triplets
            else zero
        )
        outcome_margin = wrong_distance - correct_radius
        eligible_groups = len({triplet.group_index for triplet in triplets})
        metrics = {
            "stage2_rank_loss": rank_loss,
            "stage2_rank_correct_local_radius": correct_radius,
            "stage2_rank_wrong_to_correct_distance": wrong_distance,
            "stage2_rank_outcome_margin": outcome_margin,
            "stage2_rank_triplet_count": rollout_paths.new_tensor(float(len(triplets))),
            "stage2_rank_active_triplet_count": rollout_paths.new_tensor(
                float(len(active_triplets))
            ),
            "stage2_rank_active_fraction": rollout_paths.new_tensor(
                float(len(active_triplets)) / float(max(1, len(triplets)))
            ),
            "stage2_rank_eligible_groups": rollout_paths.new_tensor(
                float(eligible_groups)
            ),
            "stage2_rank_local_backward_chunks": rollout_paths.new_tensor(
                float(local_chunk_count)
            ),
            "stage2_rank_synchronized_backward_chunks": rollout_paths.new_tensor(
                float(synchronized_chunk_count)
            ),
            "stage2_rank_dummy_backward_chunks": rollout_paths.new_tensor(
                float(dummy_chunk_count)
            ),
        }
        return rank_loss, metrics

    @torch.no_grad()
    def _merge_task_guarded_ranking_gradients(
        self,
        parameters: List[torch.nn.Parameter],
        task_gradients: List[Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        ranking_gradients = [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in parameters
        ]
        merged, metrics = merge_accuracy_guarded_gradients(
            task_gradients,
            ranking_gradients,
            max_ratio=float(
                self.trace_rl_config.get("stage2_ranking_grad_ratio", 0.25)
            ),
        )
        for parameter, gradient in zip(parameters, merged):
            if gradient is None:
                parameter.grad = None
            elif parameter.grad is None:
                parameter.grad = gradient.to(dtype=parameter.dtype)
            else:
                parameter.grad.copy_(gradient.to(dtype=parameter.grad.dtype))
        return {
            "stage2_guard_task_grad_norm": metrics["task_grad_norm"],
            "stage2_guard_ranking_grad_norm": metrics["ranking_grad_norm"],
            "stage2_guard_projected_ranking_grad_norm": metrics[
                "projected_ranking_grad_norm"
            ],
            "stage2_guard_ranking_scale": metrics["ranking_scale"],
            "stage2_guard_task_ranking_cosine": metrics["task_ranking_cosine"],
            "stage2_guard_conflict": metrics["conflict"],
        }

    def trace_rl_training_step(self, batch, batch_idx, dataloader_idx=0):
        optimizer = self.optimizers()
        questions = batch["question"]
        answers = batch["answer"]
        (
            experience,
            group_questions,
            _path_seeds,
            latent_noise,
            rollout_paths,
        ) = self.trace_rollout(
            questions=questions,
            gt_answers=answers,
        )

        optimizer.zero_grad()
        micro_batch = self._trace_rl_exp_batch_size()
        total_items = max(1, len(group_questions))
        answer_loss_sum = torch.zeros((), device=self.device)
        for start in range(0, len(group_questions), micro_batch):
            end = min(start + micro_batch, len(group_questions))
            current_answer_logprobs = self.answer_logprobs_for_questions(
                questions=group_questions[start:end],
                answer_input_ids=experience.answer_input_ids[start:end],
                answer_attention_mask=experience.answer_attention_mask[start:end],
                trace_view_ids=None,
                trace_latent_noise=latent_noise[start:end],
            )
            chunk_loss = self._ppo_answer_loss(
                answer_logprobs=current_answer_logprobs,
                old_answer_logprobs=experience.answer_logprobs[start:end].detach(),
                answer_attention_mask=experience.answer_attention_mask[start:end].float(),
                advantages=experience.advantages[start:end].detach(),
            )
            chunk_weight = float(end - start) / float(total_items)
            self.manual_backward(chunk_loss * chunk_weight)
            answer_loss_sum = answer_loss_sum + chunk_loss.detach() * float(end - start)
            del current_answer_logprobs, chunk_loss
        answer_loss = answer_loss_sum / float(total_items)

        replay_weight = float(
            self.trace_rl_config.get("stage2_sft_replay_weight", 0.05)
        )
        replay_raw = torch.zeros((), device=self.device)
        if replay_weight > 0:
            replay_dict = self.forward(batch=batch)
            replay_raw = replay_dict["total_loss"]
            self.manual_backward(replay_weight * replay_raw)

        ranking_weight = float(
            self.trace_rl_config.get("stage2_local_ranking_weight", 0.0)
        )
        ranking_raw = torch.zeros((), device=self.device)
        ranking_metrics: Dict[str, torch.Tensor] = {}
        guard_metrics: Dict[str, torch.Tensor] = {}
        use_guard = bool(
            self.trace_rl_config.get("stage2_accuracy_gradient_guard", True)
        )
        if ranking_weight > 0:
            parameters = [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            ]
            task_gradients = None
            if use_guard:
                task_gradients = [
                    parameter.grad.detach().clone()
                    if parameter.grad is not None
                    else None
                    for parameter in parameters
                ]
                optimizer.zero_grad(set_to_none=True)
            ranking_raw, ranking_metrics = self.backward_stage2_local_ranking(
                group_questions=group_questions,
                accuracies=experience.accuracies,
                rollout_paths=rollout_paths,
                latent_noise=latent_noise,
                loss_weight=ranking_weight,
            )
            if use_guard:
                guard_metrics = self._merge_task_guarded_ranking_gradients(
                    parameters,
                    task_gradients,
                )
                del task_gradients

        grad_norm = clip_grad_norm_(
            self.parameters(),
            max_norm=float(self.trace_rl_config.get("clip_grad_norm", 1.0)),
        )
        optimizer_did_step = False
        if torch.isfinite(grad_norm):
            optimizer.step()
            optimizer_did_step = True
            if bool(self.all_config.model.training_kwargs.get("use_scheduler", False)):
                scheduler = self.lr_schedulers()
                if isinstance(scheduler, (list, tuple)):
                    for item in scheduler:
                        item.step()
                elif scheduler is not None:
                    scheduler.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            self.log(
                "train/skipped_nonfinite",
                torch.tensor(1.0, device=self.device),
            )

        total_loss = (
            answer_loss
            + replay_weight * replay_raw.detach()
            + ranking_weight * ranking_raw.detach()
        )
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        effective_lr = (
            float(raw_optimizer.param_groups[0]["lr"])
            if raw_optimizer.param_groups
            else 0.0
        )
        logs = {
            "train/total_loss": total_loss.detach(),
            "train/answer_rl_loss": answer_loss.detach(),
            "train/stage2_local_ranking_loss": ranking_raw.detach(),
            "train/stage2_sft_replay_loss": replay_raw.detach(),
            "train/rewards": experience.rewards.mean().detach(),
            "train/accuracies": experience.accuracies.mean().detach(),
            "train/n_latent_forward": experience.n_latent_forward.float().mean().detach(),
            "train/output_length": experience.answer_attention_mask.float()
            .sum(dim=1)
            .mean()
            .detach(),
            "train/grad_norm": (
                grad_norm.detach()
                if isinstance(grad_norm, torch.Tensor)
                else torch.tensor(grad_norm)
            ),
            "train/effective_lr": torch.tensor(effective_lr, device=self.device),
            "train/optimizer_did_step": torch.tensor(
                float(optimizer_did_step),
                device=self.device,
            ),
        }
        if self.device.type == "cuda":
            gib = float(1024**3)
            peak_allocated = torch.tensor(
                torch.cuda.max_memory_allocated(self.device) / gib,
                device=self.device,
            )
            peak_reserved = torch.tensor(
                torch.cuda.max_memory_reserved(self.device) / gib,
                device=self.device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    peak_allocated,
                    op=torch.distributed.ReduceOp.MAX,
                )
                torch.distributed.all_reduce(
                    peak_reserved,
                    op=torch.distributed.ReduceOp.MAX,
                )
            logs.update(
                {
                    "train/cuda_peak_allocated_max_gib": peak_allocated,
                    "train/cuda_peak_reserved_max_gib": peak_reserved,
                }
            )
        logs.update(
            {
                f"train/{key}": value.detach()
                for key, value in self._last_trace_metrics.items()
            }
        )
        logs.update(
            {
                f"train/{key}": value.detach()
                for key, value in ranking_metrics.items()
            }
        )
        logs.update(
            {
                f"train/{key}": value.detach()
                for key, value in guard_metrics.items()
            }
        )
        self.log_dict(
            logs,
            sync_dist=True,
            prog_bar=True,
            batch_size=len(batch["idx"]),
        )
        return total_loss.detach()

    def _deterministic_visual_seeds(
        self,
        indices: Sequence[int],
        group_size: int,
    ) -> torch.Tensor:
        base_seed = int(self.trace_config.get("trace_visual_seed", 271828))
        rows = []
        for index in indices:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                (base_seed + int(index) * 1000003) % (2**63 - 1)
            )
            rows.append(
                torch.randn(
                    group_size,
                    self.path_seed_dim,
                    generator=generator,
                    dtype=torch.float32,
                )
            )
        return torch.stack(rows, dim=0).to(self.device)

    @torch.no_grad()
    def _build_trace_visual_records(
        self,
        batch,
        latent_outputs,
        output_strings: Sequence[str],
    ) -> List[dict]:
        questions = batch["question"]
        answers = batch["answer"]
        step_lists = self._decode_step_lists(batch)
        dependency_matrices = self._decode_cached_matrices(
            batch,
            "dependency_matrix",
        )
        confidence_matrices = self._decode_cached_matrices(
            batch,
            "confidence_matrix",
        )
        explicit_features = self._collect_explicit_batch_features(
            questions=questions,
            step_lists=step_lists,
            answers=answers,
            dependency_matrices=dependency_matrices,
            confidence_matrices=confidence_matrices,
        )
        compression_outputs = self._compress_explicit_reasoning(explicit_features)
        if bool(self.trace_config.get("enable_trajectory_formation", True)):
            compression_outputs = self._apply_trace_progress_anchors(
                explicit_features,
                compression_outputs,
            )

        group_size = int(self.trace_config.get("trace_visual_group_views", 1))
        visual_seeds = None
        multiview_outputs = None
        multiview_output_strings = None
        multiview_output_lengths = None
        multiview_acc = None
        if group_size > 1:
            visual_seeds = self._deterministic_visual_seeds(
                batch["idx"].tolist(),
                group_size,
            )
            flattened_seeds = visual_seeds.flatten(0, 1)
            repeated_questions = [
                question
                for question in questions
                for _ in range(group_size)
            ]
            batch_seed = int(self.trace_config.get("trace_visual_seed", 271828))
            batch_seed += sum(
                (local_idx + 1) * (int(index) + 1009)
                for local_idx, index in enumerate(batch["idx"].tolist())
            )
            fork_devices = (
                [torch.cuda.current_device()]
                if self.device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(batch_seed)
                output_ids, _, multiview_outputs = self._read_generate_with_latents_in_chunks(
                    repeated_questions,
                    trace_view_ids=None,
                    trace_latent_noise=self._path_seed_noise(flattened_seeds),
                    do_sample=bool(
                        self.trace_config.get("trace_visual_do_sample", True)
                    ),
                    temperature=float(
                        self.trace_config.get("trace_visual_temperature", 0.95)
                    ),
                    top_p=float(
                        self.trace_config.get("trace_visual_top_p", 0.97)
                    ),
                )
            multiview_output_strings = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
            multiview_output_lengths = output_ids.ne(
                self.tokenizer.pad_token_id
            ).sum(dim=1)
            multiview_acc = []
            for sample_idx, answer in enumerate(answers):
                start = sample_idx * group_size
                end = start + group_size
                for output_string in multiview_output_strings[start:end]:
                    predicted_answer = self.extract_answer_from_output(output_string)
                    multiview_acc.append(
                        float(
                            self.verify_answer(
                                gt_answer=answer,
                                pred_answer=predicted_answer,
                            )
                        )
                    )

        records = []
        for local_idx, index_value in enumerate(batch["idx"].tolist()):
            assignment = compression_outputs["assignments"][local_idx].detach().float()
            relation = compression_outputs["relation_probs"][local_idx].detach().float()
            target = compression_outputs["aggregated_explicit_residuals"][
                local_idx
            ].detach().float()
            top_weights, top_indices = assignment.topk(
                k=min(3, assignment.shape[1]),
                dim=1,
            )
            record = {
                "idx": int(index_value),
                "question": questions[local_idx],
                "answer": answers[local_idx],
                "steps": step_lists[local_idx],
                "output_string": output_strings[local_idx],
                "path_seed": torch.zeros(self.path_seed_dim, dtype=torch.float32),
                "latent_states": latent_outputs["latent_states"][local_idx]
                .detach()
                .to(torch.float16)
                .cpu(),
                "implicit_residuals": latent_outputs["implicit_residuals"][
                    local_idx
                ]
                .detach()
                .to(torch.float16)
                .cpu(),
                "aggregated_explicit_residuals": target.to(torch.float16).cpu(),
                "assignment": assignment.to(torch.float16).cpu(),
                "assignment_top_indices": top_indices.cpu(),
                "assignment_top_weights": top_weights.to(torch.float16).cpu(),
                "relation_probs": relation.to(torch.float16).cpu(),
                "dependency_probs": compression_outputs["dependency_probs"][
                    local_idx
                ]
                .detach()
                .to(torch.float16)
                .cpu(),
                "trace_seed_schema": "iid_continuous_exchangeable",
            }
            if multiview_outputs is not None:
                start = local_idx * group_size
                end = start + group_size
                seeded_assignments = []
                seeded_targets = []
                for path_seed in visual_seeds[local_idx]:
                    seeded_assignment = self._make_seeded_assignment(
                        assignment,
                        path_seed,
                    )
                    seeded_assignments.append(seeded_assignment)
                    seeded_targets.append(
                        aggregate_step_residuals(
                            seeded_assignment,
                            explicit_features[local_idx]["step_residuals"],
                        )
                    )
                record.update(
                    {
                        "multiview_path_seeds": visual_seeds[local_idx]
                        .detach()
                        .to(torch.float16)
                        .cpu(),
                        "multiview_latent_states": multiview_outputs[
                            "latent_states"
                        ][start:end]
                        .detach()
                        .to(torch.float16)
                        .cpu(),
                        "multiview_implicit_residuals": multiview_outputs[
                            "implicit_residuals"
                        ][start:end]
                        .detach()
                        .to(torch.float16)
                        .cpu(),
                        "multiview_teacher_assignments": torch.stack(
                            seeded_assignments
                        )
                        .to(torch.float16)
                        .cpu(),
                        "multiview_teacher_residuals": torch.stack(seeded_targets)
                        .to(torch.float16)
                        .cpu(),
                        "multiview_output_strings": multiview_output_strings[
                            start:end
                        ],
                        "multiview_acc": torch.tensor(
                            multiview_acc[start:end],
                            dtype=torch.float32,
                        ),
                        "multiview_output_lengths": multiview_output_lengths[
                            start:end
                        ]
                        .detach()
                        .cpu(),
                    }
                )
            records.append(record)
        return records

    def _save_trace_visual_cache(self, split: str):
        if not self._trace_visual_records:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not getattr(trainer, "is_global_zero", True):
            return
        try:
            log_dir = Path(self.logger.log_dir)
        except Exception:
            log_dir = Path(".")
        torch.save(
            self._trace_visual_records,
            log_dir / f"trace_exchangeable_visual_{split}.pt",
        )
