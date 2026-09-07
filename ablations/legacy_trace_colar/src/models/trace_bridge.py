import hashlib
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .read_stable_efficient import LitREADCoTStableEfficient
from ..modules import grpo
from ..modules.readcot import aggregate_step_residuals


class LitTRACEBridge(LitREADCoTStableEfficient):
    """TRACE on top of BRIDGE/READ-CoT.

    Stage 1 keeps BRIDGE's dependency-guided compression and compact anchors,
    then adds trajectory constraints on the implicit latent residual path.
    Optional Stage 2 uses reproducible multi-view latent rollouts, a GRPO
    answer-token objective, and outcome-conditioned direct path alignment.
    """

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)
        self.trace_config = model_kwargs.get("trace_bridge_config", model_kwargs.get("trace_config", {}))
        self.do_trace_rl = bool(model_kwargs.get("do_trace_rl", model_kwargs.get("do_rl", False)))
        self._last_trace_metrics: Dict[str, torch.Tensor] = {}
        self._trace_visual_records: List[dict] = []

        max_views = int(self.trace_config.get("max_trace_views", 32))
        self.max_trace_views = max(1, max_views)
        self.max_trace_latents = max(1, int(self.readcot_config.get("n_latents", 8)))
        if self.trace_config.get("use_trace_view_embeddings", True):
            self.trace_view_embeddings = nn.Embedding(self.max_trace_views, self.hidden_size)
            nn.init.normal_(self.trace_view_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
        if self.trace_config.get("use_trace_step_view_embeddings", False):
            self.trace_step_view_embeddings = nn.Embedding(
                self.max_trace_views * self.max_trace_latents,
                self.hidden_size,
            )
            nn.init.normal_(
                self.trace_step_view_embeddings.weight,
                mean=0.0,
                std=1.0 / math.sqrt(self.hidden_size),
            )

        if self.do_trace_rl:
            self.init_trace_rl()

    def init_trace_rl(self):
        self.automatic_optimization = False
        self.trace_rl_config = self.model_kwargs.trace_rl_config
        self.grpo_loss = grpo.GRPOLoss(rl_config=self.trace_rl_config)

    def training_step(self, batch, batch_idx=None, dataloader_idx=0):
        if self.do_trace_rl:
            return self.trace_rl_training_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        return super().training_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def on_fit_start(self):
        if self.do_trace_rl:
            self.limit_trace_rl_train_epoch_length()
        return super().on_fit_start()

    def on_train_epoch_start(self):
        if self.do_trace_rl:
            self.limit_trace_rl_train_epoch_length()
        return super().on_train_epoch_start()

    def on_test_start(self):
        self._trace_visual_records = []
        return super().on_test_start()

    def on_test_end(self):
        self._save_trace_visual_cache("test")
        return super().on_test_end()

    def on_validation_epoch_end(self):
        self._save_trace_visual_cache("val")
        return super().on_validation_epoch_end()

    def limit_trace_rl_train_epoch_length(self):
        target_count = int(self.trace_rl_config.get("n_train_samples_per_epoch", 512))
        all_indices = list(self.trainer.datamodule.get_all_train_indices())
        if not all_indices:
            return
        if target_count >= len(all_indices):
            selected = all_indices
        else:
            selected = random.choices(all_indices, k=target_count)
        self.trainer.datamodule.set_train_indices(selected)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _masked_cosine_distance(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        distance = 1.0 - F.cosine_similarity(pred, target, dim=-1)
        distance = torch.nan_to_num(distance, nan=0.0, posinf=2.0, neginf=0.0)
        return self._masked_mean(distance, mask)

    def _progress_anchor_mask(self, mask: torch.Tensor, anchor_count: int) -> torch.Tensor:
        if anchor_count <= 0:
            return mask
        anchor_mask = torch.zeros_like(mask)
        fractions = torch.linspace(
            1.0 / anchor_count,
            1.0,
            steps=anchor_count,
            device=mask.device,
            dtype=torch.float32,
        )
        lengths = mask.sum(dim=1).long()
        for batch_idx, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            valid = torch.nonzero(mask[batch_idx] > 0, as_tuple=False).flatten()
            positions = torch.round((length - 1) * fractions).long().clamp(min=0, max=length - 1)
            anchor_mask[batch_idx, valid[positions.unique()]] = 1.0
        return anchor_mask * mask

    def _path_signature_from_states(self, latent_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = latent_states.float()
        mean_state = states.mean(dim=1)
        first_state = states[:, 0, :]
        last_state = states[:, -1, :]
        trend_state = last_state - first_state
        if states.shape[1] <= 1:
            delta_state = torch.zeros_like(mean_state)
            step_coherence = torch.zeros(states.shape[0], device=states.device)
            noncollapse = torch.zeros(states.shape[0], device=states.device)
        else:
            deltas = states[:, 1:, :] - states[:, :-1, :]
            delta_state = deltas.mean(dim=1)
            delta_norm = deltas.norm(dim=-1) / math.sqrt(states.shape[-1])
            margin = float(self.trace_config.get("noncollapse_margin", 0.03))
            noncollapse = (delta_norm - margin).clamp_min(0.0).mean(dim=1)
            if deltas.shape[1] <= 1:
                step_coherence = torch.zeros(states.shape[0], device=states.device)
            else:
                step_dir = F.normalize(deltas, dim=-1)
                step_coherence = F.cosine_similarity(step_dir[:, 1:, :], step_dir[:, :-1, :], dim=-1).mean(dim=1)

        parts = [
            float(self.trace_config.get("signature_mean_weight", 0.5)) * F.normalize(mean_state, dim=-1),
            float(self.trace_config.get("signature_last_weight", 0.5)) * F.normalize(last_state, dim=-1),
            float(self.trace_config.get("signature_trend_weight", 1.0)) * F.normalize(trend_state, dim=-1),
            float(self.trace_config.get("signature_delta_weight", 1.0)) * F.normalize(delta_state, dim=-1),
        ]
        return F.normalize(torch.cat(parts, dim=-1), dim=-1), step_coherence, noncollapse

    def _path_signature_from_residuals(self, implicit_residuals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_path = implicit_residuals.float().cumsum(dim=1)
        return self._path_signature_from_states(residual_path)

    def _path_signature_for_outputs(
        self,
        latent_outputs: Dict[str, torch.Tensor],
        source: str = "states",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source in {"residual", "residuals", "implicit_residuals"}:
            return self._path_signature_from_residuals(latent_outputs["implicit_residuals"])
        return self._path_signature_from_states(latent_outputs["latent_states"])

    def _trace_stage1_losses(
        self,
        latent_outputs: Dict[str, torch.Tensor],
        compression_outputs: Dict[str, torch.Tensor],
        prefix: str = "trace_stage1",
    ) -> Dict[str, torch.Tensor]:
        cfg = self.trace_config
        pred_res = latent_outputs["implicit_residuals"].float()
        target_res = compression_outputs["aggregated_explicit_residuals"].detach().float()
        if pred_res.shape != target_res.shape:
            return {}

        mask = torch.ones(pred_res.shape[:2], device=pred_res.device, dtype=pred_res.dtype)
        pred_path = pred_res.cumsum(dim=1)
        target_path = target_res.cumsum(dim=1)
        anchor_count = int(cfg.get("path_anchor_count", 3))
        anchor_mask = self._progress_anchor_mask(mask, anchor_count)
        position_loss = self._masked_cosine_distance(pred_path, target_path, anchor_mask)

        pair_mask = mask[:, 1:] * mask[:, :-1]
        zero = torch.zeros((), device=pred_res.device, dtype=pred_res.dtype)
        if pair_mask.sum() > 0:
            pred_delta = pred_path[:, 1:, :] - pred_path[:, :-1, :]
            target_delta = target_path[:, 1:, :] - target_path[:, :-1, :]
            direction_loss = self._masked_cosine_distance(pred_delta, target_delta, pair_mask)
            norm_scale = math.sqrt(pred_res.shape[-1])
            pred_step = pred_delta.norm(dim=-1) / norm_scale
            target_step = target_delta.norm(dim=-1).detach() / norm_scale
            step_loss = self._masked_mean(F.smooth_l1_loss(pred_step, target_step, reduction="none"), pair_mask)
            noncollapse_margin = float(cfg.get("path_noncollapse_margin", 0.02))
            noncollapse_loss = self._masked_mean(F.relu(noncollapse_margin - pred_step), pair_mask)
            pred_step_mean = self._masked_mean(pred_step.detach(), pair_mask)
            target_step_mean = self._masked_mean(target_step.detach(), pair_mask)
        else:
            direction_loss = zero
            step_loss = zero
            noncollapse_loss = zero
            pred_step_mean = zero
            target_step_mean = zero

        position_mix = float(cfg.get("path_position_mix", 0.45))
        direction_mix = float(cfg.get("path_direction_mix", 0.35))
        step_mix = float(cfg.get("path_step_mix", 0.15))
        noncollapse_mix = float(cfg.get("path_noncollapse_mix", 0.05))
        total_mix = max(position_mix + direction_mix + step_mix + noncollapse_mix, 1e-6)
        path_loss = (
            position_mix * position_loss
            + direction_mix * direction_loss
            + step_mix * step_loss
            + noncollapse_mix * noncollapse_loss
        ) / total_mix

        return {
            f"{prefix}_path_loss": path_loss,
            f"{prefix}_position_loss": position_loss,
            f"{prefix}_direction_loss": direction_loss,
            f"{prefix}_step_loss": step_loss,
            f"{prefix}_noncollapse_loss": noncollapse_loss,
            f"{prefix}_pred_step_norm": pred_step_mean,
            f"{prefix}_target_step_norm": target_step_mean,
            f"{prefix}_anchor_count": anchor_mask.sum().detach(),
        }

    def _trace_multiview_losses(
        self,
        primary_outputs: Dict[str, torch.Tensor],
        view_outputs: Dict[str, torch.Tensor],
        compression_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.trace_config
        view_logs = self._trace_stage1_losses(
            latent_outputs=view_outputs,
            compression_outputs=compression_outputs,
            prefix="trace_stage1_view",
        )
        if not view_logs:
            return {}

        signature_source = str(cfg.get("multiview_signature_source", "states"))
        primary_sig, _, _ = self._path_signature_for_outputs(primary_outputs, signature_source)
        view_sig, _, _ = self._path_signature_for_outputs(view_outputs, signature_source)
        view_distance = 1.0 - F.cosine_similarity(primary_sig, view_sig, dim=-1)
        min_distance = float(cfg.get("multiview_min_signature_distance", 0.03))
        diversity_hinge = F.relu(min_distance - view_distance).mean()
        consistency = self._masked_cosine_distance(
            primary_outputs["implicit_residuals"].float(),
            view_outputs["implicit_residuals"].float(),
            torch.ones(
                primary_outputs["implicit_residuals"].shape[:2],
                device=primary_outputs["implicit_residuals"].device,
                dtype=torch.float32,
            ),
        )
        view_logs["trace_stage1_multiview_distance"] = view_distance.mean()
        view_logs["trace_stage1_multiview_diversity_hinge"] = diversity_hinge
        view_logs["trace_stage1_multiview_consistency"] = consistency
        path_weight = float(cfg.get("multiview_path_weight", 1.0))
        view_logs["trace_stage1_multiview_path_weight"] = view_distance.new_tensor(path_weight)
        view_logs["trace_stage1_multiview_loss"] = (
            path_weight * view_logs["trace_stage1_view_path_loss"]
            + float(cfg.get("multiview_diversity_weight", 0.10)) * diversity_hinge
            + float(cfg.get("multiview_consistency_weight", 0.02)) * consistency
        )
        return view_logs

    def _make_view_assignment(self, assignment: torch.Tensor, view_id: int) -> torch.Tensor:
        cfg = self.trace_config
        if assignment.shape[-1] <= 1:
            return assignment
        logits = torch.log(assignment.detach().float().clamp_min(1e-6))
        temperatures = cfg.get("multiview_teacher_temperatures", None)
        if temperatures:
            temp = float(temperatures[int(view_id) % len(temperatures)])
        else:
            temp = float(cfg.get("multiview_teacher_temperature", 1.0))
        temp = max(temp, 1e-3)
        scores = logits / temp

        progress_weight = float(cfg.get("multiview_teacher_progress_bias", 0.0))
        if progress_weight > 0:
            m, k = assignment.shape
            slot_pos = torch.linspace(0.0, 1.0, m, device=assignment.device, dtype=torch.float32).unsqueeze(1)
            step_pos = torch.linspace(0.0, 1.0, k, device=assignment.device, dtype=torch.float32).unsqueeze(0)
            shifts = cfg.get("multiview_teacher_progress_shifts", [-0.12, 0.0, 0.12, -0.06, 0.06])
            shift = float(shifts[int(view_id) % len(shifts)])
            sigma = max(float(cfg.get("multiview_teacher_progress_sigma", 0.22)), 1e-3)
            center = (slot_pos + shift).clamp(0.0, 1.0)
            progress_bias = -((step_pos - center) ** 2) / (2.0 * sigma * sigma)
            scores = scores + progress_weight * progress_bias.to(scores.dtype)

        pattern_weight = float(cfg.get("multiview_teacher_pattern_bias", 0.0))
        if pattern_weight > 0:
            m, k = assignment.shape
            slot = torch.arange(1, m + 1, device=assignment.device, dtype=torch.float32).unsqueeze(1)
            step = torch.arange(1, k + 1, device=assignment.device, dtype=torch.float32).unsqueeze(0)
            phase = float(int(view_id) + 1)
            pattern = torch.sin(slot * step * (phase * 1.61803398875))
            scores = scores + pattern_weight * pattern.to(scores.dtype)

        view_assignment = torch.softmax(scores, dim=-1)
        original_mix = float(cfg.get("multiview_teacher_original_mix", 0.0))
        if original_mix > 0:
            view_assignment = (1.0 - original_mix) * view_assignment + original_mix * assignment.detach().float()
            view_assignment = view_assignment / view_assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return view_assignment.to(assignment.dtype)

    def _make_multiview_compression_outputs(
        self,
        explicit_features: List[Dict[str, torch.Tensor]],
        compression_outputs: Dict[str, torch.Tensor],
        view_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.trace_config.get("use_multiview_teacher", False):
            return compression_outputs
        targets = []
        view_assignments = []
        base_assignments = compression_outputs["assignments"]
        for sample_idx, item in enumerate(explicit_features):
            view_id = int(view_ids[sample_idx].detach().item())
            view_assignment = self._make_view_assignment(base_assignments[sample_idx], view_id=view_id)
            targets.append(aggregate_step_residuals(view_assignment, item["step_residuals"]))
            view_assignments.append(view_assignment)
        out = dict(compression_outputs)
        out["aggregated_explicit_residuals"] = torch.stack(targets, dim=0)
        out["trace_view_assignments"] = view_assignments
        base_targets = compression_outputs["aggregated_explicit_residuals"].detach().float()
        view_targets = out["aggregated_explicit_residuals"].detach().float()
        out["trace_teacher_target_distance"] = (
            1.0 - F.cosine_similarity(base_targets.flatten(1), view_targets.flatten(1), dim=-1)
        ).mean()
        return out

    def _progress_anchor_assignment(self, assignment: torch.Tensor) -> torch.Tensor:
        if assignment.shape[-1] <= 1:
            return assignment.detach().float()
        m, k = assignment.shape
        slot_pos = torch.linspace(0.0, 1.0, m, device=assignment.device, dtype=torch.float32).unsqueeze(1)
        step_pos = torch.linspace(0.0, 1.0, k, device=assignment.device, dtype=torch.float32).unsqueeze(0)
        sigma = max(float(self.trace_config.get("stage1_progress_anchor_sigma", 0.22)), 1e-3)
        logits = -((step_pos - slot_pos) ** 2) / (2.0 * sigma * sigma)
        return torch.softmax(logits, dim=-1)

    def _apply_trace_progress_anchors(
        self,
        explicit_features: List[Dict[str, torch.Tensor]],
        compression_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mix = float(self.trace_config.get("stage1_progress_anchor_mix", 0.0))
        if mix <= 0:
            return compression_outputs
        mix = min(max(mix, 0.0), 1.0)
        assignments = []
        targets = []
        centers = []
        base_assignments = compression_outputs["assignments"]
        for sample_idx, item in enumerate(explicit_features):
            base = base_assignments[sample_idx].float()
            anchor = self._progress_anchor_assignment(base)
            assignment = (1.0 - mix) * base + mix * anchor
            assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            assignments.append(assignment.to(base_assignments[sample_idx].dtype))
            targets.append(aggregate_step_residuals(assignment.to(item["step_residuals"].dtype), item["step_residuals"]))

            if assignment.shape[-1] > 1:
                step_pos = torch.linspace(0.0, 1.0, assignment.shape[-1], device=assignment.device, dtype=torch.float32)
                centers.append((assignment.float() * step_pos.unsqueeze(0)).sum(dim=-1))

        out = dict(compression_outputs)
        out["assignments"] = torch.stack(assignments, dim=0)
        out["aggregated_explicit_residuals"] = torch.stack(targets, dim=0)
        if centers:
            center_tensor = torch.stack(centers, dim=0)
            out["trace_progress_anchor_span"] = (
                center_tensor.max(dim=-1).values - center_tensor.min(dim=-1).values
            ).mean()
            out["trace_progress_anchor_mean"] = center_tensor.mean()
        return out

    def _make_view_ids(self, batch_size: int, offset: int = 0, randomize: bool = False) -> torch.Tensor:
        if randomize:
            high = max(1, self.max_trace_views)
            ids = torch.randint(0, high, (batch_size,), device=self.device)
        else:
            ids = torch.arange(batch_size, device=self.device) + int(offset)
        return ids.remainder(self.max_trace_views).long()

    @staticmethod
    def _stable_question_seed(question: str, base_seed: int) -> int:
        digest = hashlib.sha256(question.encode("utf-8")).digest()
        question_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        return (question_seed + int(base_seed)) % (2**63 - 1)

    def _apply_trace_eval_intervention(
        self,
        latent_inputs_embeds: torch.Tensor,
        questions: Sequence[str],
    ) -> torch.Tensor:
        """Apply parameter-free trajectory controls only during evaluation."""
        mode = str(self.trace_config.get("trace_eval_intervention", "none")).strip().lower()
        if mode in {"", "none", "normal"}:
            return latent_inputs_embeds
        if self.training:
            raise RuntimeError("trace_eval_intervention is evaluation-only")
        if latent_inputs_embeds.shape[1] <= 1:
            return latent_inputs_embeds

        aliases = {
            "reversed": "reverse",
            "permute": "shuffle",
            "permuted": "shuffle",
            "collapse": "mean_repeat",
            "collapsed": "mean_repeat",
            "random": "random_direction",
        }
        mode = aliases.get(mode, mode)
        if mode in {"same_question_swap", "cross_question_swap"}:
            # These require replacing the generated latent K/V suffix and are
            # handled by LitTRACEFinal after the latent forward pass.
            return latent_inputs_embeds
        if mode == "reverse":
            return latent_inputs_embeds.flip(dims=[1])
        if mode == "mean_repeat":
            return latent_inputs_embeds.mean(dim=1, keepdim=True).expand_as(latent_inputs_embeds)
        if mode.startswith("replace_transition_"):
            try:
                transition_index = int(
                    mode.removeprefix("replace_transition_")
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid transition intervention {mode!r}"
                ) from exc
            if not 0 <= transition_index < latent_inputs_embeds.shape[1]:
                raise ValueError(
                    f"{mode!r} is outside the available latent transitions"
                )
            controlled = latent_inputs_embeds.clone()
            keep = [
                index
                for index in range(latent_inputs_embeds.shape[1])
                if index != transition_index
            ]
            replacement = latent_inputs_embeds[:, keep].float().mean(
                dim=1
            )
            replacement = F.normalize(replacement, dim=-1) * (
                latent_inputs_embeds[:, transition_index]
                .float()
                .norm(dim=-1, keepdim=True)
            )
            controlled[:, transition_index] = replacement.to(
                controlled.dtype
            )
            return controlled

        base_seed = int(self.trace_config.get("trace_eval_intervention_seed", 0))
        controlled = []
        for sample_idx, question in enumerate(questions):
            sample = latent_inputs_embeds[sample_idx]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._stable_question_seed(str(question), base_seed))
            if mode == "shuffle":
                permutation = torch.randperm(sample.shape[0], generator=generator, device="cpu").to(sample.device)
                controlled.append(sample.index_select(0, permutation))
                continue
            if mode == "random_direction":
                center = sample.float().mean(dim=0, keepdim=True)
                offsets = sample.float() - center
                offset_norms = offsets.norm(dim=-1, keepdim=True)
                random_offsets = torch.randn(offsets.shape, generator=generator, dtype=torch.float32)
                random_offsets = random_offsets.to(device=sample.device)
                random_offsets = F.normalize(random_offsets, dim=-1) * offset_norms
                controlled.append((center + random_offsets).to(dtype=sample.dtype))
                continue
            if mode == "same_norm_random_path":
                random_path = torch.randn(
                    sample.shape,
                    generator=generator,
                    dtype=torch.float32,
                ).to(device=sample.device)
                random_path = F.normalize(random_path, dim=-1)
                random_path = random_path * sample.float().norm(
                    dim=-1,
                    keepdim=True,
                )
                controlled.append(random_path.to(dtype=sample.dtype))
                continue
            raise ValueError(
                "Unsupported trace_eval_intervention="
                f"{mode!r}; expected none, reverse, shuffle, mean_repeat, "
                "random_direction, same_norm_random_path, "
                "same_question_swap, cross_question_swap, or "
                "replace_transition_0..7"
            )
        return torch.stack(controlled, dim=0)

    def _question_only_latents(
        self,
        questions: Sequence[str],
        trace_view_ids: Optional[torch.Tensor] = None,
        trace_noise_std: Optional[float] = None,
        trace_latent_noise: Optional[torch.Tensor] = None,
        trace_latent_noise_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        mode = self.readcot_config.get("implicit_latent_mode", "block")
        if mode != "block":
            return super()._question_only_latents(questions)

        batch_size = len(questions)
        n_latents = self.readcot_config.n_latents
        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.thinking_separator,
        )
        question_inputs_embeds = self.embedding(question_input_ids)
        question_position_ids = self.make_position_ids_for_current_input(
            question_attention_mask,
            question_inputs_embeds.shape[1],
        )
        question_outputs = self.llm.forward(
            inputs_embeds=question_inputs_embeds,
            attention_mask=question_attention_mask,
            position_ids=question_position_ids,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = question_outputs.past_key_values
        prev_state = self.state_norm(question_outputs.hidden_states[-1][:, -1, :])
        anchor_gate = None
        if self.use_anchor_gate:
            anchor_gate = torch.sigmoid(self.anchor_gate_predictor(prev_state))

        base_latent = self.latent_bridge(prev_state).unsqueeze(1)
        query_scale = float(self.readcot_config.get("block_latent_query_scale", 0.1))
        latent_queries = self.step_compressor.latent_queries[:n_latents].unsqueeze(0)
        latent_queries = latent_queries.expand(batch_size, -1, -1).to(base_latent.dtype)

        if trace_noise_std is None:
            trace_noise_std = float(self.trace_config.get("stage1_latent_query_noise_std", 0.0))
        if self.training and trace_noise_std and trace_noise_std > 0:
            latent_queries = latent_queries + torch.randn_like(latent_queries) * float(trace_noise_std)

        latent_inputs_embeds = base_latent + query_scale * latent_queries
        if trace_view_ids is not None and hasattr(self, "trace_view_embeddings"):
            view_scale = float(self.trace_config.get("trace_view_scale", 0.04))
            view_ids = trace_view_ids.to(device=self.device).long().remainder(self.max_trace_views)
            view_embeds = self.trace_view_embeddings(view_ids).unsqueeze(1).to(latent_inputs_embeds.dtype)
            latent_inputs_embeds = latent_inputs_embeds + view_scale * view_embeds
        if trace_view_ids is not None and hasattr(self, "trace_step_view_embeddings"):
            step_scale = float(self.trace_config.get("trace_step_view_scale", 0.02))
            if step_scale > 0:
                view_ids = trace_view_ids.to(device=self.device).long().remainder(self.max_trace_views)
                step_ids = torch.arange(n_latents, device=self.device).remainder(self.max_trace_latents)
                table_ids = view_ids.unsqueeze(1) * self.max_trace_latents + step_ids.unsqueeze(0)
                step_embeds = self.trace_step_view_embeddings(table_ids).to(latent_inputs_embeds.dtype)
                latent_inputs_embeds = latent_inputs_embeds + step_scale * step_embeds

        if trace_latent_noise is not None:
            expected_shape = (batch_size, n_latents, self.hidden_size)
            if tuple(trace_latent_noise.shape) != expected_shape:
                raise ValueError(
                    f"trace_latent_noise has shape {tuple(trace_latent_noise.shape)}, expected {expected_shape}"
                )
            if trace_latent_noise_scale is None:
                rl_cfg = self.model_kwargs.get("trace_rl_config", {})
                trace_latent_noise_scale = float(rl_cfg.get("stage2_latent_noise_scale", 0.0))
            latent_inputs_embeds = latent_inputs_embeds + float(trace_latent_noise_scale) * trace_latent_noise.to(
                device=latent_inputs_embeds.device,
                dtype=latent_inputs_embeds.dtype,
            )

        latent_inputs_embeds = latent_inputs_embeds.to(question_inputs_embeds.dtype)
        if anchor_gate is not None and self.readcot_config.get("anchor_gate_apply_to_latents", True):
            gate_scale = float(self.readcot_config.get("anchor_gate_scale", 0.5))
            gate_values = anchor_gate[:, :n_latents].unsqueeze(-1).to(latent_inputs_embeds.dtype)
            latent_inputs_embeds = latent_inputs_embeds * (1.0 + gate_scale * gate_values)

        latent_inputs_embeds = self._apply_trace_eval_intervention(latent_inputs_embeds, questions)

        latent_attention_mask = torch.ones(
            batch_size,
            n_latents,
            device=self.device,
            dtype=question_attention_mask.dtype,
        )
        context_attention_mask = torch.cat([question_attention_mask, latent_attention_mask], dim=1)
        latent_position_ids = self.make_position_ids_for_current_input(
            context_attention_mask,
            latent_inputs_embeds.shape[1],
        )
        latent_outputs = self.llm.forward(
            inputs_embeds=latent_inputs_embeds,
            attention_mask=context_attention_mask,
            position_ids=latent_position_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )

        latent_states = self.state_norm(latent_outputs.hidden_states[-1])
        previous_states = torch.cat([prev_state.unsqueeze(1), latent_states[:, :-1, :]], dim=1)
        implicit_residuals = latent_states - previous_states
        context_inputs_embeds = torch.cat([question_inputs_embeds, latent_inputs_embeds], dim=1)

        return {
            "question_input_ids": question_input_ids,
            "question_attention_mask": question_attention_mask,
            "latent_inputs_embeds": latent_inputs_embeds,
            "latent_attention_mask": latent_attention_mask,
            "context_inputs_embeds": context_inputs_embeds,
            "context_attention_mask": context_attention_mask,
            "past_key_values": latent_outputs.past_key_values,
            "implicit_residuals": implicit_residuals,
            "latent_states": latent_states,
            "anchor_gate": anchor_gate,
            "trace_view_ids": trace_view_ids,
        }

    def forward(self, batch):
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
        compression_outputs = self._compress_explicit_reasoning(explicit_features)
        compression_outputs = self._apply_trace_progress_anchors(explicit_features, compression_outputs)
        primary_view_ids = torch.zeros(len(questions), device=self.device, dtype=torch.long)
        latent_outputs = self._question_only_latents(questions, trace_view_ids=primary_view_ids)
        residual_targets = compression_outputs["aggregated_explicit_residuals"]
        residual_corruption = self.readcot_config.get("residual_target_corruption", "none")
        if residual_corruption and residual_corruption != "none":
            if residual_corruption == "shuffle":
                perm = torch.randperm(residual_targets.shape[1], device=residual_targets.device)
                residual_targets = residual_targets[:, perm, :]
            elif residual_corruption == "reverse":
                residual_targets = torch.flip(residual_targets, dims=[1])
            elif residual_corruption == "random":
                residual_targets = torch.randn_like(residual_targets)
            elif residual_corruption == "cross_sample":
                residual_targets = torch.roll(residual_targets, shifts=1, dims=0)
            else:
                raise ValueError(f"Unsupported residual_target_corruption: {residual_corruption}")
        residual_outputs = self._compute_residual_alignment(
            implicit_residuals=latent_outputs["implicit_residuals"],
            aggregated_explicit_residuals=residual_targets,
        )

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
        use_anchor_supervision = self.readcot_config.get("use_hybrid", False) or self.readcot_config.get(
            "use_anchor_loss", False
        )
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
                anchor_gate_loss = F.smooth_l1_loss(latent_outputs["anchor_gate"], gate_target)

        dep_weight = self._scheduled_loss_weight("dep", self.readcot_config.lambda_dep)
        res_weight = self._scheduled_loss_weight("res", self.readcot_config.lambda_res)
        anchor_weight = self._scheduled_loss_weight("anchor", self.readcot_config.lambda_anchor)
        anchor_gate_weight = self._scheduled_loss_weight(
            "anchor_gate",
            self.readcot_config.get("lambda_anchor_gate", 0.0),
        )

        trace_logs = {}
        trace_weight = float(self.trace_config.get("stage1_path_weight", 0.12))
        if trace_weight > 0:
            trace_logs.update(self._trace_stage1_losses(latent_outputs, compression_outputs))
        if "trace_progress_anchor_span" in compression_outputs:
            trace_logs["trace_stage1_progress_anchor_span"] = compression_outputs["trace_progress_anchor_span"]
            trace_logs["trace_stage1_progress_anchor_mean"] = compression_outputs["trace_progress_anchor_mean"]

        multiview_weight = float(self.trace_config.get("stage1_multiview_weight", 0.04))
        if multiview_weight > 0:
            view_ids = self._make_view_ids(len(questions), offset=1, randomize=True)
            view_outputs = self._question_only_latents(
                questions,
                trace_view_ids=view_ids,
                trace_noise_std=float(self.trace_config.get("stage1_multiview_noise_std", 0.0)),
            )
            view_compression_outputs = self._make_multiview_compression_outputs(
                explicit_features=explicit_features,
                compression_outputs=compression_outputs,
                view_ids=view_ids,
            )
            trace_logs.update(self._trace_multiview_losses(latent_outputs, view_outputs, view_compression_outputs))
            if "trace_teacher_target_distance" in view_compression_outputs:
                trace_logs["trace_stage1_multiview_teacher_distance"] = view_compression_outputs[
                    "trace_teacher_target_distance"
                ]

        total_loss = answer_weight * answer_loss
        total_loss = total_loss + dep_weight * compression_outputs["dep_loss"]
        total_loss = total_loss + res_weight * residual_outputs["residual_loss"]
        if use_anchor_supervision:
            total_loss = total_loss + anchor_weight * anchor_loss
        if self.use_anchor_gate:
            total_loss = total_loss + anchor_gate_weight * anchor_gate_loss
        if trace_weight > 0 and "trace_stage1_path_loss" in trace_logs:
            total_loss = total_loss + trace_weight * trace_logs["trace_stage1_path_loss"]
        if multiview_weight > 0 and "trace_stage1_multiview_loss" in trace_logs:
            total_loss = total_loss + multiview_weight * trace_logs["trace_stage1_multiview_loss"]

        result = {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "dep_loss": compression_outputs["dep_loss"],
            "dep_f1": compression_outputs["dep_f1"],
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
            "lambda_trace_path_eff": answer_loss.new_tensor(trace_weight),
            "lambda_trace_multiview_eff": answer_loss.new_tensor(multiview_weight),
        }
        result.update(trace_logs)
        return result

    @torch.no_grad()
    def read_generate_with_latents(
        self,
        questions: List[str],
        trace_view_ids: Optional[torch.Tensor] = None,
        trace_latent_noise: Optional[torch.Tensor] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        latent_outputs = self._question_only_latents(
            questions,
            trace_view_ids=trace_view_ids,
            trace_noise_std=0.0,
            trace_latent_noise=trace_latent_noise,
        )
        batch_size = len(questions)
        end_token_ids = torch.ones(batch_size, 1, device=self.device, dtype=torch.long) * self.thinking_separator_id
        end_token_embeds = self.embedding(end_token_ids)
        all_inputs_embeds = torch.cat([latent_outputs["context_inputs_embeds"], end_token_embeds], dim=1)
        all_attention_mask = torch.cat([latent_outputs["context_attention_mask"], torch.ones_like(end_token_ids)], dim=1)

        if self.readcot_config.get("hybrid_seed_anchor_header", False):
            prefix_ids, prefix_mask = self._prepare_raw_texts(
                [self.anchor_header + "\n"] * batch_size,
                padding_side="right",
            )
            prefix_embeds = self.embedding(prefix_ids)
            all_inputs_embeds = torch.cat([all_inputs_embeds, prefix_embeds], dim=1)
            all_attention_mask = torch.cat([all_attention_mask, prefix_mask], dim=1)

        generation_attention_mask = all_attention_mask
        generation_position_ids = None
        hidden_prefix_k = int(self.trace_config.get("trace_eval_hidden_prefix_k", -1))
        if hidden_prefix_k >= 0:
            if self.training:
                raise RuntimeError("trace_eval_hidden_prefix_k is an evaluation-only intervention")
            n_latents = int(self.readcot_config.n_latents)
            if hidden_prefix_k > n_latents:
                raise ValueError(
                    f"trace_eval_hidden_prefix_k={hidden_prefix_k} exceeds n_latents={n_latents}"
                )

            # Keep every latent slot and every downstream absolute position fixed,
            # but prevent answer generation from reading latent slots after k.
            question_length = latent_outputs["question_input_ids"].shape[1]
            latent_start = question_length
            latent_stop = latent_start + n_latents
            generation_attention_mask = all_attention_mask.clone()
            generation_attention_mask[:, latent_start + hidden_prefix_k : latent_stop] = 0
            generation_position_ids = all_attention_mask.long().cumsum(dim=-1) - 1
            generation_position_ids = generation_position_ids.masked_fill(all_attention_mask == 0, 0)

        generation_config = dict(self._get_generation_config())
        if do_sample is not None:
            generation_config["do_sample"] = bool(do_sample)
        if temperature is not None:
            generation_config["temperature"] = float(temperature)
        if top_p is not None:
            generation_config["top_p"] = float(top_p)

        generate_kwargs = {
            "inputs_embeds": all_inputs_embeds,
            "attention_mask": generation_attention_mask,
            **generation_config,
        }
        if generation_position_ids is not None:
            generate_kwargs["position_ids"] = generation_position_ids
        pred_ids = self.llm.generate(**generate_kwargs)
        n_latent_forward = torch.ones(batch_size, 1, device=self.device, dtype=torch.long) * self.readcot_config.n_latents
        return pred_ids, n_latent_forward, latent_outputs

    @torch.no_grad()
    def read_generate(self, questions: List[str]):
        pred_ids, n_latent_forward, _ = self.read_generate_with_latents(questions)
        return pred_ids, n_latent_forward

    def _prompt_ids_for_generated_answer(self, batch_size: int):
        end_token_ids = torch.ones(batch_size, 1, device=self.device, dtype=torch.long) * self.thinking_separator_id
        prompt_pieces = [end_token_ids]
        prompt_masks = [torch.ones_like(end_token_ids)]
        if self.readcot_config.get("hybrid_seed_anchor_header", False):
            prefix_ids, prefix_mask = self._prepare_raw_texts(
                [self.anchor_header + "\n"] * batch_size,
                padding_side="right",
            )
            prompt_pieces.append(prefix_ids)
            prompt_masks.append(prefix_mask)
        return torch.cat(prompt_pieces, dim=1), torch.cat(prompt_masks, dim=1)

    def answer_logprobs_for_questions(
        self,
        questions: List[str],
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        trace_view_ids: Optional[torch.Tensor] = None,
        trace_latent_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_outputs = self._question_only_latents(
            questions,
            trace_view_ids=trace_view_ids,
            trace_noise_std=0.0,
            trace_latent_noise=trace_latent_noise,
        )
        batch_size = len(questions)
        prompt_ids, prompt_mask = self._prompt_ids_for_generated_answer(batch_size)
        prompt_embeds = self.embedding(prompt_ids)
        answer_embeds = self.embedding(answer_input_ids)
        inputs_embeds = torch.cat(
            [latent_outputs["context_inputs_embeds"], prompt_embeds, answer_embeds],
            dim=1,
        )
        attention_mask = torch.cat(
            [latent_outputs["context_attention_mask"], prompt_mask, answer_attention_mask],
            dim=1,
        )
        position_ids = self.make_position_ids_for_current_input(attention_mask, inputs_embeds.shape[1])
        outputs = self.llm.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
        )
        context_len = latent_outputs["context_inputs_embeds"].shape[1]
        prompt_len = prompt_ids.shape[1]
        answer_len = answer_input_ids.shape[1]
        start = context_len + prompt_len - 1
        answer_logits = outputs.logits[:, start : start + answer_len, :]
        flat_logits = answer_logits.reshape(-1, answer_logits.shape[-1])
        flat_ids = answer_input_ids.reshape(-1)
        selected = []
        for chunk_start in range(0, flat_logits.shape[0], 16):
            chunk_end = min(chunk_start + 16, flat_logits.shape[0])
            logprobs = F.log_softmax(flat_logits[chunk_start:chunk_end], dim=-1)
            selected.append(logprobs.gather(dim=-1, index=flat_ids[chunk_start:chunk_end].unsqueeze(-1)).squeeze(-1))
        answer_logprobs = torch.cat(selected, dim=0).reshape_as(answer_input_ids)
        answer_logprobs = torch.nan_to_num(answer_logprobs, nan=-30.0, neginf=-30.0, posinf=30.0)
        return answer_logprobs.clamp(min=-30.0, max=30.0)

    def _trace_rl_exp_batch_size(self) -> int:
        trace_rl_config = getattr(self, "trace_rl_config", None)
        if trace_rl_config is None:
            trace_rl_config = self.model_kwargs.get("trace_rl_config", {})
        default_batch_size = self.trace_config.get("trace_visual_micro_batch_size", 1)
        return max(1, int(trace_rl_config.get("exp_batch_size", default_batch_size)))

    @staticmethod
    def _slice_tensor_dict(tensors: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
        return {key: value[start:end] if isinstance(value, torch.Tensor) else value for key, value in tensors.items()}

    @staticmethod
    def _cat_tensor_dict(chunks: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not chunks:
            return {}
        merged = {}
        for key, value in chunks[0].items():
            if isinstance(value, torch.Tensor):
                merged[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
            else:
                merged[key] = value
        return merged

    def _pad_generated_ids(self, chunks: Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(chunk.shape[1] for chunk in chunks)
        pad_id = self.tokenizer.pad_token_id
        padded = []
        for chunk in chunks:
            if chunk.shape[1] == max_len:
                padded.append(chunk)
                continue
            pad_width = max_len - chunk.shape[1]
            pad = torch.full(
                (chunk.shape[0], pad_width),
                fill_value=pad_id,
                device=chunk.device,
                dtype=chunk.dtype,
            )
            padded.append(torch.cat([chunk, pad], dim=1))
        return torch.cat(padded, dim=0)

    @torch.no_grad()
    def _read_generate_with_latents_in_chunks(
        self,
        questions: List[str],
        trace_view_ids: Optional[torch.Tensor] = None,
        trace_latent_noise: Optional[torch.Tensor] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        micro_batch = self._trace_rl_exp_batch_size()
        if len(questions) <= micro_batch:
            return self.read_generate_with_latents(
                questions,
                trace_view_ids=trace_view_ids,
                trace_latent_noise=trace_latent_noise,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        pred_chunks = []
        n_latent_chunks = []
        latent_chunks = []
        for start in range(0, len(questions), micro_batch):
            end = min(start + micro_batch, len(questions))
            view_chunk = trace_view_ids[start:end] if trace_view_ids is not None else None
            noise_chunk = trace_latent_noise[start:end] if trace_latent_noise is not None else None
            pred_ids, n_latent_forward, latent_outputs = self.read_generate_with_latents(
                questions[start:end],
                trace_view_ids=view_chunk,
                trace_latent_noise=noise_chunk,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            pred_chunks.append(pred_ids)
            n_latent_chunks.append(n_latent_forward)
            latent_chunks.append(latent_outputs)
        return (
            self._pad_generated_ids(pred_chunks),
            torch.cat(n_latent_chunks, dim=0),
            self._cat_tensor_dict(latent_chunks),
        )

    def _answer_logprobs_for_questions_in_chunks(
        self,
        questions: List[str],
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        trace_view_ids: Optional[torch.Tensor] = None,
        trace_latent_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        micro_batch = self._trace_rl_exp_batch_size()
        if len(questions) <= micro_batch:
            return self.answer_logprobs_for_questions(
                questions=questions,
                answer_input_ids=answer_input_ids,
                answer_attention_mask=answer_attention_mask,
                trace_view_ids=trace_view_ids,
                trace_latent_noise=trace_latent_noise,
            )

        logprob_chunks = []
        for start in range(0, len(questions), micro_batch):
            end = min(start + micro_batch, len(questions))
            view_chunk = trace_view_ids[start:end] if trace_view_ids is not None else None
            noise_chunk = trace_latent_noise[start:end] if trace_latent_noise is not None else None
            logprob_chunks.append(
                self.answer_logprobs_for_questions(
                    questions=questions[start:end],
                    answer_input_ids=answer_input_ids[start:end],
                    answer_attention_mask=answer_attention_mask[start:end],
                    trace_view_ids=view_chunk,
                    trace_latent_noise=noise_chunk,
                )
            )
        return torch.cat(logprob_chunks, dim=0)

    def _ppo_answer_loss(
        self,
        answer_logprobs: torch.Tensor,
        old_answer_logprobs: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        clip_eps = float(self.trace_rl_config.get("clip_eps", 0.08))
        token_advantages = advantages
        while token_advantages.dim() < answer_logprobs.dim():
            token_advantages = token_advantages.unsqueeze(-1)
        ratio = (answer_logprobs - old_answer_logprobs).exp()
        surr1 = ratio * token_advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * token_advantages
        loss = -torch.min(surr1, surr2)
        return (loss * answer_attention_mask).sum(dim=-1).div(answer_attention_mask.sum(dim=-1).clamp_min(1.0)).mean()

    @torch.no_grad()
    def trace_rollout(self, questions: List[str], gt_answers):
        batch_size = len(questions)
        group_size = int(self.trace_rl_config.get("group_size", 8))
        group_questions = []
        view_ids = []
        for q in questions:
            for view_idx in range(group_size):
                group_questions.append(q)
                view_ids.append(view_idx)
        view_ids_tensor = torch.tensor(view_ids, device=self.device, dtype=torch.long).remainder(self.max_trace_views)
        latent_noise_scale = float(self.trace_rl_config.get("stage2_latent_noise_scale", 0.0))
        latent_noise = None
        if latent_noise_scale > 0:
            latent_noise = torch.randn(
                len(group_questions),
                self.max_trace_latents,
                self.hidden_size,
                device=self.device,
                dtype=torch.float32,
            )

        pred_ids, n_latent_forward, latent_outputs = self._read_generate_with_latents_in_chunks(
            group_questions,
            trace_view_ids=view_ids_tensor,
            trace_latent_noise=latent_noise,
            do_sample=True,
            temperature=float(self.trace_rl_config.get("temperature", 0.8)),
            top_p=float(self.trace_rl_config.get("top_p", 0.95)),
        )
        answer_attention_mask = pred_ids.ne(self.tokenizer.pad_token_id).long()
        pred_strings = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        output_lengths = answer_attention_mask.sum(dim=1).float()

        signature_source = str(self.trace_rl_config.get(
            "rl_signature_source",
            self.trace_config.get("rl_signature_source", "states"),
        ))
        path, step_coherence, noncollapse = self._path_signature_for_outputs(latent_outputs, signature_source)
        rewards = []
        accuracies = []
        all_advantages = []
        metric_lists = {
            "trace_rl/pos_count": [],
            "trace_rl/mixed_frac": [],
            "trace_rl/mode_count": [],
            "trace_rl/effective_modes": [],
            "trace_rl/neg_proto_sim": [],
            "trace_rl/path_bonus": [],
        }
        for sample_idx in range(batch_size):
            start = sample_idx * group_size
            end = start + group_size
            group_pred = pred_strings[start:end]
            group_acc = torch.zeros(group_size, 1, device=self.device)
            for local_idx, pred in enumerate(group_pred):
                pred_a = self.extract_answer_from_output(pred)
                group_acc[local_idx] = self.verify_answer(gt_answer=gt_answers[sample_idx], pred_answer=pred_a)
            group_rewards, group_metrics = self.trace_group_rewards(
                group_acc=group_acc.view(-1),
                group_path=path[start:end],
                group_step=step_coherence[start:end],
                group_noncollapse=noncollapse[start:end],
                group_output_lengths=output_lengths[start:end],
            )
            group_advantages = grpo.group_advantages(group_rewards.view(-1)).view(-1, 1)
            rewards.append(group_rewards.view(-1, 1))
            accuracies.append(group_acc)
            all_advantages.append(group_advantages)
            for key, value in group_metrics.items():
                metric_lists[key].append(float(value))

        rewards = torch.cat(rewards, dim=0)
        accuracies = torch.cat(accuracies, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        old_answer_logprobs = self._answer_logprobs_for_questions_in_chunks(
            questions=group_questions,
            answer_input_ids=pred_ids,
            answer_attention_mask=answer_attention_mask,
            trace_view_ids=view_ids_tensor,
            trace_latent_noise=latent_noise,
        )
        latent_logprobs = torch.zeros_like(latent_outputs["latent_attention_mask"], dtype=old_answer_logprobs.dtype)

        self._last_trace_metrics = {
            key: torch.tensor(float(np.mean(values)) if values else 0.0, device=self.device)
            for key, values in metric_lists.items()
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
            rewards=rewards,
            accuracies=accuracies,
            advantages=advantages,
        )
        return experience, group_questions, view_ids_tensor, latent_noise, path.detach()

    def prepare_stage2_group_signatures(self, group_path: torch.Tensor) -> torch.Tensor:
        raw = F.normalize(group_path, dim=-1)
        if not bool(self.trace_rl_config.get("stage2_center_signatures", True)):
            return raw
        centered = F.normalize(raw - raw.mean(dim=0, keepdim=True), dim=-1)
        raw_mix = float(self.trace_rl_config.get("stage2_signature_raw_mix", 0.25))
        if raw_mix <= 0:
            return centered
        return F.normalize(torch.cat([centered, raw_mix * raw], dim=-1), dim=-1)

    def trace_group_rewards(
        self,
        group_acc: torch.Tensor,
        group_path: torch.Tensor,
        group_step: torch.Tensor,
        group_noncollapse: torch.Tensor,
        group_output_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.trace_rl_config
        group_path = self.prepare_stage2_group_signatures(group_path)
        group_size = group_acc.shape[0]
        rewards = group_acc.float().clone()
        length_weight = float(cfg.get("output_length_penalty_weight", 0.0))
        if length_weight > 0:
            target_len = float(cfg.get("target_output_length", 34.0))
            rewards = rewards - length_weight * F.relu(group_output_lengths / max(target_len, 1.0) - 1.0)

        bonus = torch.zeros_like(rewards)
        bonus += float(cfg.get("step_coherence_weight", 0.02)) * group_step.float()
        bonus += float(cfg.get("noncollapse_weight", 0.02)) * group_noncollapse.float()

        pos_mask = group_acc > 0.5
        neg_mask = ~pos_mask
        pos_count = int(pos_mask.sum().item())
        mode_count = 0
        effective_modes = torch.tensor(0.0, device=group_path.device)
        neg_proto_sim = torch.tensor(0.0, device=group_path.device)
        if pos_count > 0:
            pos_paths = group_path[pos_mask]
            prototypes, assignments, counts = self.build_positive_modes(pos_paths)
            mode_count = int(prototypes.shape[0])
            effective_modes = self.mode_effective_count(counts)
            pos_sims = pos_paths @ prototypes.T
            nearest_pos = pos_sims.max(dim=1).values
            bonus[pos_mask] += float(cfg.get("pos_mode_fit_weight", 0.12)) * nearest_pos
            if mode_count > 1:
                proto_sim = prototypes @ prototypes.T
                off_diag = ~torch.eye(mode_count, dtype=torch.bool, device=group_path.device)
                separation = (1.0 - proto_sim[off_diag].mean()).clamp_min(0.0)
                coverage = (effective_modes - 1.0) / max(float(min(mode_count, pos_count) - 1), 1.0)
                bonus[pos_mask] += float(cfg.get("mode_diversity_weight", 0.05)) * (
                    0.5 * separation + 0.5 * coverage.clamp(0.0, 1.0)
                )
            if neg_mask.any():
                neg_paths = group_path[neg_mask]
                nearest_neg = (neg_paths @ prototypes.T).max(dim=1).values
                neg_proto_sim = nearest_neg.mean()
                margin = float(cfg.get("neg_margin", 0.35))
                bonus[neg_mask] -= float(cfg.get("neg_repulsion_weight", 0.10)) * F.relu(nearest_neg - margin)

        trace_reward_weight = float(cfg.get("trace_reward_weight", 0.10))
        clip = float(cfg.get("trace_bonus_clip", 0.25))
        if clip > 0:
            bonus = bonus.clamp(min=-clip, max=clip)
        rewards = rewards + trace_reward_weight * bonus
        return rewards, {
            "trace_rl/pos_count": float(pos_count),
            "trace_rl/mixed_frac": float(0 < pos_count < group_size),
            "trace_rl/mode_count": float(mode_count),
            "trace_rl/effective_modes": float(effective_modes.item()),
            "trace_rl/neg_proto_sim": float(neg_proto_sim.item()),
            "trace_rl/path_bonus": float(bonus.mean().item()),
        }

    def build_positive_modes(self, pos_paths: torch.Tensor):
        max_modes = max(1, int(self.trace_rl_config.get("max_modes", 3)))
        merge_threshold = float(self.trace_rl_config.get("mode_merge_threshold", 0.82))
        n_pos = pos_paths.shape[0]
        target_modes = min(max_modes, n_pos)
        prototypes = [F.normalize(pos_paths[0], dim=-1)]
        selected = {0}
        while len(prototypes) < target_modes:
            proto_tensor = torch.stack(prototypes, dim=0)
            nearest = (pos_paths @ proto_tensor.T).max(dim=1).values
            if selected:
                nearest[torch.tensor(list(selected), device=pos_paths.device)] = 2.0
            local_idx = int(torch.argmin(nearest).item())
            if nearest[local_idx] > merge_threshold:
                break
            prototypes.append(F.normalize(pos_paths[local_idx], dim=-1))
            selected.add(local_idx)
        proto_tensor = F.normalize(torch.stack(prototypes, dim=0), dim=-1)
        assignments = torch.zeros(n_pos, device=pos_paths.device, dtype=torch.long)
        for _ in range(2):
            assignments = (pos_paths @ proto_tensor.T).argmax(dim=1)
            new_protos = []
            for mode_idx in range(proto_tensor.shape[0]):
                members = pos_paths[assignments == mode_idx]
                if members.numel() == 0:
                    new_protos.append(proto_tensor[mode_idx])
                else:
                    new_protos.append(F.normalize(members.mean(dim=0), dim=-1))
            proto_tensor = torch.stack(new_protos, dim=0)
        counts = torch.stack([(assignments == i).sum() for i in range(proto_tensor.shape[0])]).float()
        return proto_tensor, assignments, counts

    @staticmethod
    def mode_effective_count(counts: torch.Tensor) -> torch.Tensor:
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        return torch.exp(entropy)

    def _current_trace_signatures(
        self,
        questions: List[str],
        view_ids: torch.Tensor,
        latent_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        signature_source = str(self.trace_rl_config.get(
            "rl_signature_source",
            self.trace_config.get("rl_signature_source", "states"),
        ))
        latent_outputs = self._question_only_latents(
            questions,
            trace_view_ids=view_ids if view_ids is not None else None,
            trace_noise_std=0.0,
            trace_latent_noise=latent_noise if latent_noise is not None else None,
        )
        signatures, _, _ = self._path_signature_for_outputs(latent_outputs, signature_source)
        return signatures

    def _prepare_stage2_signature_with_reference(
        self,
        signature: torch.Tensor,
        raw_group_mean: torch.Tensor,
    ) -> torch.Tensor:
        raw = F.normalize(signature, dim=-1)
        if not bool(self.trace_rl_config.get("stage2_center_signatures", True)):
            return raw
        centered = F.normalize(raw - raw_group_mean, dim=-1)
        raw_mix = float(self.trace_rl_config.get("stage2_signature_raw_mix", 0.25))
        if raw_mix <= 0:
            return centered
        return F.normalize(torch.cat([centered, raw_mix * raw], dim=-1), dim=-1)

    def backward_stage2_direct_signature_loss(
        self,
        group_questions: List[str],
        view_ids: torch.Tensor,
        accuracies: torch.Tensor,
        rollout_signatures: torch.Tensor,
        latent_noise: Optional[torch.Tensor] = None,
        loss_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.trace_rl_config
        group_size = int(cfg.get("group_size", 8))
        detached_signatures = rollout_signatures.detach()
        acc = accuracies.detach().view(-1).to(detached_signatures.device)
        margin = float(cfg.get("stage2_direct_neg_margin", cfg.get("neg_margin", 0.25)))
        wrong_div_margin = float(cfg.get("stage2_direct_wrong_div_margin", 0.75))
        inter_mode_margin = float(cfg.get("stage2_direct_inter_mode_margin", 0.80))
        pos_weight = float(cfg.get("stage2_direct_pos_weight", 1.0))
        neg_weight = float(cfg.get("stage2_direct_neg_weight", 1.0))
        wrong_div_weight = float(cfg.get("stage2_direct_wrong_div_weight", 0.25))
        inter_mode_weight = float(cfg.get("stage2_direct_inter_mode_weight", 0.15))

        path_plans = []
        detached_losses = []
        pos_dist_values = []
        neg_sim_values = []
        wrong_within_values = []
        inter_mode_sim_values = []
        mode_count_values = []
        boundary_margin_values = []
        mixed_groups = 0
        n_groups = max(1, (detached_signatures.shape[0] + group_size - 1) // group_size)
        for start in range(0, detached_signatures.shape[0], group_size):
            end = min(start + group_size, detached_signatures.shape[0])
            group_pos_dist_values = []
            raw_group = F.normalize(detached_signatures[start:end], dim=-1)
            raw_group_mean = raw_group.mean(dim=0).detach()
            group_sig = self.prepare_stage2_group_signatures(detached_signatures[start:end]).detach()
            group_acc = acc[start:end]
            pos_mask = group_acc > 0.5
            neg_mask = ~pos_mask
            pos_indices = torch.nonzero(pos_mask, as_tuple=False).view(-1)
            neg_indices = torch.nonzero(neg_mask, as_tuple=False).view(-1)
            pos_prototypes = None
            assignments = None
            counts = None
            if pos_indices.numel() > 0:
                pos_sig = group_sig[pos_mask]
                pos_prototypes, assignments, counts = self.build_positive_modes(pos_sig.detach())
                mode_count_values.append(pos_sig.new_tensor(float(pos_prototypes.shape[0])))
                for mode_idx in range(pos_prototypes.shape[0]):
                    members = pos_sig[assignments == mode_idx]
                    if members.shape[0] <= 1:
                        continue
                    member_sum = members.sum(dim=0, keepdim=True)
                    leave_one_out = F.normalize((member_sum - members) / float(members.shape[0] - 1), dim=-1)
                    mode_pos_dist = list((1.0 - (members * leave_one_out).sum(dim=-1)).unbind())
                    group_pos_dist_values.extend(mode_pos_dist)
                    pos_dist_values.extend(mode_pos_dist)

            if pos_prototypes is not None and pos_prototypes.shape[0] > 1:
                mode_sim = pos_prototypes @ pos_prototypes.T
                off_diag = ~torch.eye(
                    pos_prototypes.shape[0],
                    dtype=torch.bool,
                    device=pos_prototypes.device,
                )
                inter_mode_sim_values.append(mode_sim[off_diag].mean())

            if pos_prototypes is not None and neg_indices.numel() > 0:
                mixed_groups += 1
                neg_sig = group_sig[neg_mask]
                nearest_neg_sim = (neg_sig @ pos_prototypes.detach().T).max(dim=1).values
                neg_sim_values.append(nearest_neg_sim.mean())
                reference_pos_dist = (
                    torch.stack(group_pos_dist_values).mean()
                    if group_pos_dist_values
                    else group_sig.new_tensor(0.0)
                )
                boundary_margin_values.append(nearest_neg_sim.mean() - (1.0 - reference_pos_dist))

                if neg_sig.shape[0] > 1 and wrong_div_weight > 0:
                    neg_pair_sim = neg_sig @ neg_sig.T
                    off_diag = ~torch.eye(neg_sig.shape[0], dtype=torch.bool, device=neg_sig.device)
                    wrong_within = neg_pair_sim[off_diag].mean()
                    wrong_within_values.append(wrong_within)

            positive_rank = {int(local_idx): rank for rank, local_idx in enumerate(pos_indices.tolist())}
            for local_idx in range(end - start):
                plan = {
                    "raw_group_mean": raw_group_mean,
                    "positive_target": None,
                    "other_mode_prototypes": None,
                    "positive_prototypes": None,
                    "other_wrong_signatures": None,
                    "positive_scale": 0.0,
                    "negative_scale": 0.0,
                    "inter_mode_scale": 0.0,
                    "wrong_div_scale": 0.0,
                }
                if local_idx in positive_rank and pos_prototypes is not None:
                    pos_rank = positive_rank[local_idx]
                    mode_idx = int(assignments[pos_rank].item())
                    same_mode = torch.nonzero(assignments == mode_idx, as_tuple=False).view(-1)
                    peers = same_mode[same_mode != pos_rank]
                    if peers.numel() > 0:
                        plan["positive_target"] = F.normalize(group_sig[pos_indices[peers]].mean(dim=0), dim=-1)
                        plan["positive_scale"] = pos_weight / float(max(1, pos_indices.numel()))
                    if pos_prototypes.shape[0] > 1 and inter_mode_weight > 0:
                        other_modes = torch.arange(pos_prototypes.shape[0], device=pos_prototypes.device) != mode_idx
                        plan["other_mode_prototypes"] = pos_prototypes[other_modes]
                        plan["inter_mode_scale"] = inter_mode_weight / float(max(1, pos_indices.numel()))
                elif pos_prototypes is not None:
                    plan["positive_prototypes"] = pos_prototypes
                    plan["negative_scale"] = neg_weight / float(max(1, neg_indices.numel()))
                    if neg_indices.numel() > 1 and wrong_div_weight > 0:
                        peers = neg_indices[neg_indices != local_idx]
                        plan["other_wrong_signatures"] = group_sig[peers]
                        plan["wrong_div_scale"] = wrong_div_weight / float(neg_indices.numel())
                path_plans.append(plan)

        geometry_micro_batch = max(1, int(cfg.get("stage2_geometry_micro_batch_size", 1)))
        for chunk_start in range(0, len(path_plans), geometry_micro_batch):
            chunk_end = min(chunk_start + geometry_micro_batch, len(path_plans))
            current_signatures = self._current_trace_signatures(
                group_questions[chunk_start:chunk_end],
                view_ids[chunk_start:chunk_end] if view_ids is not None else None,
                latent_noise[chunk_start:chunk_end] if latent_noise is not None else None,
            )
            chunk_loss = current_signatures.sum() * 0.0
            for chunk_offset, plan in enumerate(path_plans[chunk_start:chunk_end]):
                current = self._prepare_stage2_signature_with_reference(
                    current_signatures[chunk_offset],
                    plan["raw_group_mean"],
                )
                path_loss = current.sum() * 0.0
                if plan["positive_target"] is not None:
                    path_loss = path_loss + plan["positive_scale"] * (1.0 - current @ plan["positive_target"])
                if plan["other_mode_prototypes"] is not None:
                    other_sim = current @ plan["other_mode_prototypes"].T
                    path_loss = path_loss + plan["inter_mode_scale"] * F.relu(other_sim.mean() - inter_mode_margin)
                if plan["positive_prototypes"] is not None:
                    nearest_pos = (current @ plan["positive_prototypes"].T).max()
                    path_loss = path_loss + plan["negative_scale"] * F.relu(nearest_pos - margin)
                if plan["other_wrong_signatures"] is not None:
                    wrong_sim = current @ plan["other_wrong_signatures"].T
                    path_loss = path_loss + plan["wrong_div_scale"] * F.relu(wrong_sim.mean() - wrong_div_margin)
                chunk_loss = chunk_loss + float(loss_weight) * path_loss / float(n_groups)
                detached_losses.append(path_loss.detach() / float(n_groups))
            self.manual_backward(chunk_loss)
            del current_signatures, current, path_loss, chunk_loss

        zero = detached_signatures.new_tensor(0.0)
        loss = torch.stack(detached_losses).sum() if detached_losses else zero

        def mean_or_zero(values):
            return torch.stack(values).mean() if values else zero

        metrics = {
            "stage2_direct_pos_dist": mean_or_zero(pos_dist_values),
            "stage2_direct_neg_proto_sim": mean_or_zero(neg_sim_values),
            "stage2_direct_wrong_within_sim": mean_or_zero(wrong_within_values),
            "stage2_direct_inter_mode_sim": mean_or_zero(inter_mode_sim_values),
            "stage2_direct_mode_count": mean_or_zero(mode_count_values),
            "stage2_direct_boundary_sim_gap": mean_or_zero(boundary_margin_values),
            "stage2_direct_mixed_groups": torch.tensor(float(mixed_groups), device=detached_signatures.device),
            "stage2_geometry_micro_batch_size": torch.tensor(
                float(geometry_micro_batch),
                device=detached_signatures.device,
            ),
        }
        return loss, metrics

    @torch.no_grad()
    def _merge_accuracy_guarded_geometry_gradients(
        self,
        parameters: List[torch.nn.Parameter],
        task_gradients: List[Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        task_norm_sq = zero.clone()
        geometry_norm_sq = zero.clone()
        task_geometry_dot = zero.clone()
        for parameter, task_gradient in zip(parameters, task_gradients):
            geometry_gradient = parameter.grad
            if task_gradient is not None:
                task_float = task_gradient.float()
                task_norm_sq += task_float.square().sum()
            if geometry_gradient is not None:
                geometry_float = geometry_gradient.float()
                geometry_norm_sq += geometry_float.square().sum()
                if task_gradient is not None:
                    task_geometry_dot += (task_gradient.float() * geometry_float).sum()

        eps = torch.tensor(1e-12, device=self.device, dtype=torch.float32)
        task_norm = task_norm_sq.clamp_min(0.0).sqrt()
        geometry_norm = geometry_norm_sq.clamp_min(0.0).sqrt()
        conflict_dot = torch.minimum(task_geometry_dot, zero)
        projection_coefficient = conflict_dot / task_norm_sq.clamp_min(eps)
        projected_norm_sq = (geometry_norm_sq - conflict_dot.square() / task_norm_sq.clamp_min(eps)).clamp_min(0.0)
        projected_norm = projected_norm_sq.sqrt()
        max_ratio = float(self.trace_rl_config.get("stage2_geometry_grad_ratio", 0.25))
        geometry_scale = torch.minimum(
            torch.ones((), device=self.device, dtype=torch.float32),
            max_ratio * task_norm / projected_norm.clamp_min(eps),
        )
        if task_norm.item() == 0.0:
            geometry_scale.zero_()

        for parameter, task_gradient in zip(parameters, task_gradients):
            geometry_gradient = parameter.grad
            if task_gradient is None and geometry_gradient is None:
                continue
            if geometry_gradient is None:
                merged = task_gradient
            else:
                projected = geometry_gradient.float()
                if task_gradient is not None and conflict_dot.item() < 0.0:
                    projected = projected - projection_coefficient * task_gradient.float()
                merged = geometry_scale * projected
                if task_gradient is not None:
                    merged = task_gradient.float() + merged
            if parameter.grad is None:
                parameter.grad = merged.to(dtype=parameter.dtype)
            else:
                parameter.grad.copy_(merged.to(dtype=parameter.grad.dtype))

        cosine = task_geometry_dot / (task_norm * geometry_norm).clamp_min(eps)
        return {
            "stage2_guard_task_grad_norm": task_norm,
            "stage2_guard_geometry_grad_norm": geometry_norm,
            "stage2_guard_projected_geometry_grad_norm": projected_norm,
            "stage2_guard_geometry_scale": geometry_scale,
            "stage2_guard_task_geometry_cosine": cosine,
            "stage2_guard_conflict": (task_geometry_dot < 0).float(),
        }

    def trace_rl_training_step(self, batch, batch_idx, dataloader_idx=0):
        optimizer = self.optimizers()
        questions = batch["question"]
        answers = batch["answer"]
        experience, group_questions, view_ids, latent_noise, rollout_signatures = self.trace_rollout(
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
                trace_view_ids=view_ids[start:end] if view_ids is not None else None,
                trace_latent_noise=latent_noise[start:end] if latent_noise is not None else None,
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

        replay_weight = float(self.trace_rl_config.get("stage2_sft_replay_weight", 0.0))
        replay_raw = torch.zeros((), device=self.device)
        if replay_weight > 0:
            replay_dict = self.forward(batch=batch)
            replay_raw = replay_dict["total_loss"]
            self.manual_backward(replay_weight * replay_raw)

        direct_weight = float(self.trace_rl_config.get("stage2_direct_signature_weight", 0.0))
        direct_raw = torch.zeros((), device=self.device)
        direct_metrics: Dict[str, torch.Tensor] = {}
        guard_metrics: Dict[str, torch.Tensor] = {}
        use_gradient_guard = bool(self.trace_rl_config.get("stage2_accuracy_gradient_guard", False))
        if direct_weight > 0:
            parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
            task_gradients = None
            if use_gradient_guard:
                task_gradients = [
                    parameter.grad.detach().clone() if parameter.grad is not None else None
                    for parameter in parameters
                ]
                optimizer.zero_grad(set_to_none=True)
            direct_raw, direct_metrics = self.backward_stage2_direct_signature_loss(
                group_questions=group_questions,
                view_ids=view_ids,
                accuracies=experience.accuracies,
                rollout_signatures=rollout_signatures,
                latent_noise=latent_noise,
                loss_weight=direct_weight,
            )
            if use_gradient_guard:
                guard_metrics = self._merge_accuracy_guarded_geometry_gradients(parameters, task_gradients)
                del task_gradients

        grad_norm = clip_grad_norm_(self.parameters(), max_norm=float(self.trace_rl_config.get("clip_grad_norm", 1.0)))
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
            self.log("train/skipped_nonfinite", torch.tensor(1.0, device=self.device))
        total_loss = answer_loss + direct_weight * direct_raw.detach() + replay_weight * replay_raw.detach()
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        effective_lr = float(raw_optimizer.param_groups[0]["lr"]) if raw_optimizer.param_groups else 0.0

        logs = {
            "train/total_loss": total_loss.detach(),
            "train/answer_rl_loss": answer_loss.detach(),
            "train/stage2_direct_signature_loss": direct_raw.detach(),
            "train/stage2_sft_replay_loss": replay_raw.detach(),
            "train/rewards": experience.rewards.mean().detach(),
            "train/accuracies": experience.accuracies.mean().detach(),
            "train/n_latent_forward": experience.n_latent_forward.float().mean().detach(),
            "train/output_length": experience.answer_attention_mask.float().sum(dim=1).mean().detach(),
            "train/grad_norm": grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm),
            "train/effective_lr": torch.tensor(effective_lr, device=self.device),
            "train/optimizer_did_step": torch.tensor(float(optimizer_did_step), device=self.device),
        }
        logs.update({f"train/{k}": v.detach() for k, v in self._last_trace_metrics.items()})
        logs.update({f"train/{k}": v.detach() for k, v in direct_metrics.items()})
        logs.update({f"train/{k}": v.detach() for k, v in guard_metrics.items()})
        self.log_dict(logs, sync_dist=True, prog_bar=True, batch_size=len(batch["idx"]))
        return total_loss.detach()

    @torch.no_grad()
    def eval_generation(self, batch, split="val", batch_idx=None, dataloader_idx=0):
        indices = batch["idx"].tolist()
        questions = batch["question"]
        answers = batch["answer"]
        steps = batch["steps"]

        outputs_token_ids, n_latent_forward, latent_outputs = self.read_generate_with_latents(
            questions=list(questions),
            trace_view_ids=torch.zeros(len(questions), device=self.device, dtype=torch.long),
        )
        output_strings = self.tokenizer.batch_decode(outputs_token_ids, skip_special_tokens=True)

        visual_records = None
        visual_limit = int(self.trace_config.get("trace_visual_record_limit", 256))
        should_record_visuals = visual_limit > 0 and len(self._trace_visual_records) < visual_limit
        if self.trace_config.get("save_trace_visual_info", True) and should_record_visuals:
            visual_records = self._build_trace_visual_records(
                batch=batch,
                latent_outputs=latent_outputs,
                output_strings=output_strings,
            )

        all_acc = []
        all_output_length = []
        all_latent_forward = []
        for local_idx, (i, q, s, a, o_ids, o_str, nlf) in enumerate(
            zip(indices, questions, steps, answers, outputs_token_ids, output_strings, n_latent_forward)
        ):
            if i not in self.sample_logs:
                self.sample_logs[i]["question"] = q
                self.sample_logs[i]["steps"] = s
                self.sample_logs[i]["answer"] = a
                self.sample_logs[i]["pred_answer"] = []
                self.sample_logs[i]["pred_anchor_text"] = []
                self.sample_logs[i]["output_string"] = []
                self.sample_logs[i]["output_length"] = []
                self.sample_logs[i]["n_latent_forward"] = []
                self.sample_logs[i]["acc"] = []

            pred_a = self.extract_answer_from_output(o_str)
            pred_anchor_text = self._extract_predicted_anchor_text(o_str)
            acc = self.verify_answer(gt_answer=a, pred_answer=pred_a)
            o_length = (o_ids != self.tokenizer.pad_token_id).sum().item()

            self.sample_logs[i]["pred_answer"].append(pred_a)
            self.sample_logs[i]["pred_anchor_text"].append(pred_anchor_text)
            self.sample_logs[i]["output_string"].append(o_str)
            self.sample_logs[i]["output_length"].append(o_length)
            self.sample_logs[i]["n_latent_forward"].append(nlf.item())
            self.sample_logs[i]["acc"].append(acc)

            if visual_records is not None:
                visual_records[local_idx]["acc"] = float(acc)
                visual_records[local_idx]["pred_answer"] = pred_a
                visual_records[local_idx]["output_length"] = int(o_length)
                self._append_trace_visual_record(visual_records[local_idx])

            all_acc.append(acc)
            all_output_length.append(o_length)
            all_latent_forward.append(nlf.item())

        if bool(self.trace_config.get("trace_eval_skip_structure_metrics", False)):
            structure_metrics = {
                "dep_f1": torch.zeros((), device=self.device),
                "residual_similarity": torch.zeros((), device=self.device),
            }
        else:
            structure_metrics = self._compute_structure_metrics(batch)
        acc_count = sum(all_acc)
        acc_forward_count = sum([a * alf for a, alf in zip(all_acc, all_latent_forward)])
        mean_n_latent_forward_on_acc = np.mean(acc_forward_count / (acc_count + 1e-8))
        mean_acc = np.mean(all_acc)
        mean_n_latent_forward = np.mean(all_latent_forward)
        mean_output_length = np.mean(all_output_length)

        return {
            "monitor": mean_acc,
            f"{split}/acc": mean_acc,
            f"{split}/n_latent_forward": mean_n_latent_forward,
            f"{split}/n_latent_forward_on_acc": mean_n_latent_forward_on_acc,
            f"{split}/output_length": mean_output_length,
            f"{split}/dep_f1": structure_metrics["dep_f1"],
            f"{split}/residual_similarity": structure_metrics["residual_similarity"],
        }

    @torch.no_grad()
    def _build_trace_visual_records(self, batch, latent_outputs, output_strings: Sequence[str]) -> List[dict]:
        questions = batch["question"]
        answers = batch["answer"]
        step_lists = self._decode_step_lists(batch)
        dependency_matrices = self._decode_cached_matrices(batch, "dependency_matrix")
        confidence_matrices = self._decode_cached_matrices(batch, "confidence_matrix")
        explicit_features = self._collect_explicit_batch_features(
            questions=questions,
            step_lists=step_lists,
            answers=answers,
            dependency_matrices=dependency_matrices,
            confidence_matrices=confidence_matrices,
        )
        compression_outputs = self._compress_explicit_reasoning(explicit_features)
        compression_outputs = self._apply_trace_progress_anchors(explicit_features, compression_outputs)
        multiview_outputs = None
        multiview_output_strings = None
        multiview_acc = None
        multiview_output_lengths = None
        group_views = int(self.trace_config.get("trace_visual_group_views", 1))
        if group_views > 1:
            repeated_questions = []
            view_ids = []
            for question in questions:
                for view_idx in range(group_views):
                    repeated_questions.append(question)
                    view_ids.append(view_idx)
            view_ids_tensor = torch.tensor(view_ids, device=self.device, dtype=torch.long)
            visual_noise = self._make_trace_visual_noise(batch["idx"].tolist(), group_views)
            visual_do_sample = bool(self.trace_config.get("trace_visual_do_sample", False))
            base_seed = int(self.trace_config.get("trace_visual_noise_seed", 0))
            batch_seed = base_seed + sum(
                (local_idx + 1) * (int(idx) + 1009)
                for local_idx, idx in enumerate(batch["idx"].tolist())
            )
            fork_devices = [torch.cuda.current_device()] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(batch_seed)
                multiview_output_ids, _, multiview_outputs = self._read_generate_with_latents_in_chunks(
                    repeated_questions,
                    trace_view_ids=view_ids_tensor,
                    trace_latent_noise=visual_noise,
                    do_sample=visual_do_sample,
                    temperature=float(self.trace_config.get("trace_visual_temperature", 0.95)),
                    top_p=float(self.trace_config.get("trace_visual_top_p", 0.97)),
                )
            multiview_output_strings = self.tokenizer.batch_decode(
                multiview_output_ids,
                skip_special_tokens=True,
            )
            multiview_output_lengths = multiview_output_ids.ne(self.tokenizer.pad_token_id).sum(dim=1)
            multiview_acc = []
            for sample_idx, gt_answer in enumerate(answers):
                start = sample_idx * group_views
                end = start + group_views
                for output_string in multiview_output_strings[start:end]:
                    pred_answer = self.extract_answer_from_output(output_string)
                    multiview_acc.append(float(self.verify_answer(gt_answer=gt_answer, pred_answer=pred_answer)))
        records = []
        for local_idx, idx_value in enumerate(batch["idx"].tolist()):
            assignment = compression_outputs["assignments"][local_idx].detach().float().cpu()
            relation = compression_outputs["relation_probs"][local_idx].detach().float().cpu()
            target = compression_outputs["aggregated_explicit_residuals"][local_idx].detach().float().cpu()
            latent_states = latent_outputs["latent_states"][local_idx].detach().to(torch.float16).cpu()
            residuals = latent_outputs["implicit_residuals"][local_idx].detach().to(torch.float16).cpu()
            top_weights, top_indices = assignment.topk(k=min(3, assignment.shape[1]), dim=1)
            records.append(
                {
                    "idx": int(idx_value),
                    "question": questions[local_idx],
                    "answer": answers[local_idx],
                    "steps": step_lists[local_idx],
                    "output_string": output_strings[local_idx],
                    "latent_states": latent_states,
                    "implicit_residuals": residuals,
                    "aggregated_explicit_residuals": target.to(torch.float16),
                    "assignment": assignment.to(torch.float16),
                    "assignment_top_indices": top_indices.cpu(),
                    "assignment_top_weights": top_weights.to(torch.float16).cpu(),
                    "relation_probs": relation.to(torch.float16),
                    "dependency_probs": compression_outputs["dependency_probs"][local_idx].detach().to(torch.float16).cpu(),
                }
            )
            if multiview_outputs is not None:
                start = local_idx * group_views
                end = start + group_views
                records[-1]["multiview_latent_states"] = (
                    multiview_outputs["latent_states"][start:end].detach().to(torch.float16).cpu()
                )
                records[-1]["multiview_implicit_residuals"] = (
                    multiview_outputs["implicit_residuals"][start:end].detach().to(torch.float16).cpu()
                )
                records[-1]["multiview_view_ids"] = torch.arange(group_views, dtype=torch.long)
                records[-1]["multiview_output_strings"] = multiview_output_strings[start:end]
                records[-1]["multiview_acc"] = torch.tensor(multiview_acc[start:end], dtype=torch.float32)
                records[-1]["multiview_output_lengths"] = multiview_output_lengths[start:end].detach().cpu()
        return records

    def _make_trace_visual_noise(self, indices: Sequence[int], group_views: int) -> Optional[torch.Tensor]:
        noise_scale = float(self.trace_config.get("trace_visual_latent_noise_scale", 0.0))
        if noise_scale <= 0:
            return None
        base_seed = int(self.trace_config.get("trace_visual_noise_seed", 0))
        rows = []
        for idx in indices:
            for view_idx in range(group_views):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(base_seed + int(idx) * 1009 + int(view_idx) * 9176)
                rows.append(
                    torch.randn(
                        self.max_trace_latents,
                        self.hidden_size,
                        generator=generator,
                        dtype=torch.float32,
                    )
                )
        return torch.stack(rows, dim=0).to(self.device) if rows else None

    def _append_trace_visual_record(self, record: dict):
        limit = int(self.trace_config.get("trace_visual_record_limit", 256))
        if limit <= 0 or len(self._trace_visual_records) >= limit:
            return
        self._trace_visual_records.append(record)

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
        out_path = log_dir / f"trace_bridge_visual_{split}.pt"
        torch.save(self._trace_visual_records, out_path)
        if hasattr(self, "sample_logs"):
            self.sample_logs[f"trace_bridge_visual_{split}_pt"] = str(out_path)
