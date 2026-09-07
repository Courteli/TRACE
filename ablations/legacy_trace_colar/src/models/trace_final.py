import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .trace_exchangeable import (
    LitTRACEExchangeable,
    trace_path_distance,
    trace_path_distance_components,
)


def replace_cached_latent_suffix(
    target_cache,
    donor_cache,
    *,
    n_latents: int,
    donor_indices: Optional[torch.Tensor] = None,
):
    """Replace only latent-token K/V while preserving masked question K/V."""
    target_layers = getattr(target_cache, "layers", None)
    donor_layers = getattr(donor_cache, "layers", None)
    if target_layers is None or donor_layers is None:
        raise TypeError("TRACE path swaps require a Transformers Cache object")
    if len(target_layers) != len(donor_layers):
        raise ValueError("Target and donor caches have different layer counts")
    for target_layer, donor_layer in zip(target_layers, donor_layers):
        for attribute in ("keys", "values"):
            target = getattr(target_layer, attribute, None)
            donor = getattr(donor_layer, attribute, None)
            if target is None or donor is None:
                continue
            if target.shape[-2] < n_latents or donor.shape[-2] < n_latents:
                raise ValueError("Cache is shorter than the latent suffix")
            source = donor
            if donor_indices is not None:
                source = source.index_select(
                    0,
                    donor_indices.to(device=source.device),
                )
            replacement = source[..., -n_latents:, :].clone()
            target[..., -n_latents:, :].copy_(replacement)
    return target_cache


def build_path_bottleneck_attention_mask(
    question_attention_mask: torch.Tensor,
    latent_attention_mask: torch.Tensor,
    tail_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask every answer-stage query away from the original question K/V."""
    if question_attention_mask.ndim != 2:
        raise ValueError("question_attention_mask must be two-dimensional")
    if latent_attention_mask.ndim != 2 or tail_attention_mask.ndim != 2:
        raise ValueError("latent and tail masks must be two-dimensional")
    if not (
        question_attention_mask.shape[0]
        == latent_attention_mask.shape[0]
        == tail_attention_mask.shape[0]
    ):
        raise ValueError("all bottleneck masks must have the same batch size")
    return torch.cat(
        [
            torch.zeros_like(question_attention_mask),
            latent_attention_mask,
            tail_attention_mask,
        ],
        dim=1,
    )


def permutation_invariant_path_set_loss(
    model_paths: torch.Tensor,
    teacher_paths: torch.Tensor,
    *,
    distance_kwargs: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    """Match two sampled paths to two teachers without route identities."""
    if model_paths.ndim != 4 or teacher_paths.ndim != 4:
        raise ValueError("path sets must have shape [batch, 2, transition, hidden]")
    if model_paths.shape != teacher_paths.shape or model_paths.shape[1] != 2:
        raise ValueError(
            "model and teacher path sets must have identical two-path shapes"
        )
    distance_kwargs = distance_kwargs or {}
    pair_components = {}
    for model_index in range(2):
        for teacher_index in range(2):
            pair_components[(model_index, teacher_index)] = (
                trace_path_distance_components(
                    model_paths[:, model_index],
                    teacher_paths[:, teacher_index].detach(),
                    **distance_kwargs,
                )
            )

    direct = (
        pair_components[(0, 0)]["total"]
        + pair_components[(1, 1)]["total"]
    )
    swapped = (
        pair_components[(0, 1)]["total"]
        + pair_components[(1, 0)]["total"]
    )
    use_direct = direct <= swapped
    output = {
        "loss": 0.5 * torch.minimum(direct, swapped).mean(),
        "direct_fraction": use_direct.float().mean().detach(),
    }
    for component in ("position", "direction", "step"):
        direct_component = (
            pair_components[(0, 0)][component]
            + pair_components[(1, 1)][component]
        )
        swapped_component = (
            pair_components[(0, 1)][component]
            + pair_components[(1, 0)][component]
        )
        output[component] = (
            0.5
            * torch.where(
                use_direct,
                direct_component,
                swapped_component,
            ).mean()
        )
    return output


def path_noncollapse_loss(
    paths: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Unary anti-collapse regularizer, intentionally outside the path metric."""
    if paths.ndim != 4:
        raise ValueError("paths must have shape [batch, path, transition, hidden]")
    normalized_step = paths.float().norm(dim=-1) / math.sqrt(paths.shape[-1])
    return F.relu(float(margin) - normalized_step).mean()


class LitTRACEFinal(LitTRACEExchangeable):
    """Final TRACE: set-anchored formation and outcome-local refinement.

    The answer-stage attention mask implements q -> Z -> y: latent states may
    read the question, while answer tokens can read only the complete latent
    path and their own causal prefix.
    """

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            all_config=all_config,
        )
        if self.readcot_config.get("use_hybrid", False):
            raise ValueError("Final TRACE uses a latent-only answer readout")
        if self.readcot_config.get("use_anchor_loss", False):
            raise ValueError("Final TRACE does not decode explicit CoT anchors")
        if self.readcot_config.get("use_anchor_gate", False):
            raise ValueError("Final TRACE does not use a separate anchor gate")
        for parameter in self.residual_projector.parameters():
            parameter.requires_grad_(False)
        # PEFT checkpoints intentionally omit the frozen base model. Lightning
        # must therefore restore trainable TRACE/LoRA state non-strictly while
        # optimizer, scheduler, and loop state are restored from the same file.
        self.strict_loading = False
        self._sampling_stage2_paths = False

    def _decode_rationale_sets(self, batch) -> List[List[dict]]:
        primary_steps = self._decode_step_lists(batch)
        primary_dependencies = self._decode_cached_matrices(
            batch,
            "dependency_matrix",
        )
        primary_confidences = self._decode_cached_matrices(
            batch,
            "confidence_matrix",
        )
        raw_sets = batch.get("rationale_set_json")
        decoded = []
        for sample_index, steps in enumerate(primary_steps):
            fallback = {
                "steps": list(steps),
                "dependency_matrix": primary_dependencies[sample_index],
                "confidence_matrix": primary_confidences[sample_index],
                "fingerprint": "primary",
                "source": "gold",
                "verified": True,
            }
            if raw_sets is None:
                decoded.append([fallback])
                continue
            raw = raw_sets[sample_index]
            try:
                candidates = json.loads(raw) if raw else []
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid rationale_set_json at sample {sample_index}"
                ) from exc
            valid = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                candidate_steps = candidate.get("steps")
                if not isinstance(candidate_steps, list) or not candidate_steps:
                    continue
                if candidate.get("verified") is not True:
                    continue
                valid.append(
                    {
                        "steps": [str(step) for step in candidate_steps],
                        "dependency_matrix": candidate.get(
                            "dependency_matrix"
                        ),
                        "confidence_matrix": candidate.get(
                            "confidence_matrix"
                        ),
                        "fingerprint": str(
                            candidate.get("fingerprint", "unknown")
                        ),
                        "source": str(candidate.get("source", "unknown")),
                        "verified": True,
                    }
                )
            decoded.append(valid or [fallback])
        return decoded

    def _select_teacher_pair(self, rationale_sets: List[List[dict]]):
        first = []
        second = []
        counts = []
        for rationales in rationale_sets:
            first.append(rationales[0])
            counts.append(len(rationales))
            if len(rationales) == 1:
                second.append(rationales[0])
            else:
                alternative = int(
                    torch.randint(
                        1,
                        len(rationales),
                        (1,),
                        device=self.device,
                    ).item()
                )
                second.append(rationales[alternative])
        return first, second, counts

    def _teacher_paths_for_rationales(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        rationales: Sequence[dict],
    ):
        explicit_features = self._collect_explicit_batch_features(
            questions=questions,
            step_lists=[item["steps"] for item in rationales],
            answers=answers,
            dependency_matrices=[
                item.get("dependency_matrix") for item in rationales
            ],
            confidence_matrices=[
                item.get("confidence_matrix") for item in rationales
            ],
        )
        compression = self._compress_explicit_reasoning(explicit_features)
        compression = self._apply_trace_progress_anchors(
            explicit_features,
            compression,
        )
        return explicit_features, compression

    @staticmethod
    def _bottleneck_context_mask(latent_outputs) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros_like(latent_outputs["question_attention_mask"]),
                latent_outputs["latent_attention_mask"],
            ],
            dim=1,
        )

    def _teacher_force_bottleneck(
        self,
        latent_outputs,
        target_texts: Sequence[str],
    ) -> torch.Tensor:
        return self._teacher_force_target(
            past_key_values=latent_outputs["past_key_values"],
            context_attention_mask=self._bottleneck_context_mask(
                latent_outputs
            ),
            target_texts=target_texts,
        )

    def _same_question_donor_seeds(
        self,
        questions: Sequence[str],
    ) -> torch.Tensor:
        base_seed = int(
            self.trace_config.get("trace_eval_intervention_seed", 271828)
        )
        rows = []
        for question in questions:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                self._stable_question_seed(
                    str(question),
                    base_seed + 104729,
                )
            )
            rows.append(
                torch.randn(
                    self.path_seed_dim,
                    generator=generator,
                    dtype=torch.float32,
                )
            )
        return torch.stack(rows, dim=0).to(self.device)

    def _apply_answer_path_swap(
        self,
        latent_outputs: Dict[str, torch.Tensor],
        questions: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        mode = str(
            self.trace_config.get("trace_eval_intervention", "none")
        ).lower()
        if mode not in {"same_question_swap", "cross_question_swap"}:
            return latent_outputs
        if self.training:
            raise RuntimeError("Answer-path swaps are evaluation-only")

        n_latents = int(self.readcot_config.n_latents)
        donor_indices = None
        if mode == "same_question_swap":
            seeds = self._same_question_donor_seeds(questions)
            donor_outputs = self._question_only_latents(
                questions,
                trace_noise_std=0.0,
                trace_latent_noise=self._path_seed_noise(seeds),
            )
        else:
            batch_size = len(questions)
            if batch_size < 2:
                raise ValueError(
                    "cross_question_swap requires an evaluation batch "
                    "containing at least two questions"
                )
            donor_indices = torch.roll(
                torch.arange(batch_size, device=self.device),
                shifts=1,
            )
            donor_outputs = latent_outputs

        replace_cached_latent_suffix(
            latent_outputs["past_key_values"],
            donor_outputs["past_key_values"],
            n_latents=n_latents,
            donor_indices=donor_indices,
        )
        for key in (
            "latent_inputs_embeds",
            "latent_states",
            "implicit_residuals",
        ):
            donor = donor_outputs[key]
            if donor_indices is not None:
                donor = donor.index_select(0, donor_indices)
            latent_outputs[key] = donor.clone()
        donor_context = donor_outputs["context_inputs_embeds"][:, -n_latents:]
        if donor_indices is not None:
            donor_context = donor_context.index_select(0, donor_indices)
        latent_outputs["context_inputs_embeds"][
            :,
            -n_latents:,
        ] = donor_context
        latent_outputs["answer_path_intervention"] = mode
        if donor_indices is not None:
            latent_outputs["answer_path_donor_indices"] = (
                donor_indices.detach().cpu()
            )
        return latent_outputs

    def forward(self, batch):
        if not bool(self.trace_config.get("enable_trajectory_formation", True)):
            raise RuntimeError(
                "Final TRACE requires trajectory formation during Stage 1 replay"
            )

        questions = batch["question"]
        answers = batch["answer"]
        rationale_sets = self._decode_rationale_sets(batch)
        teacher_a, teacher_b, teacher_counts = self._select_teacher_pair(
            rationale_sets
        )
        _, compression_a = self._teacher_paths_for_rationales(
            questions,
            answers,
            teacher_a,
        )
        _, compression_b = self._teacher_paths_for_rationales(
            questions,
            answers,
            teacher_b,
        )
        teacher_paths = torch.stack(
            [
                compression_a["aggregated_explicit_residuals"],
                compression_b["aggregated_explicit_residuals"],
            ],
            dim=1,
        ).detach()

        batch_size = len(questions)
        center_seeds = torch.zeros(
            batch_size,
            self.path_seed_dim,
            device=self.device,
            dtype=torch.float32,
        )
        sampled_seeds = self._sample_path_seeds(batch_size)
        center_outputs = self._question_only_latents(
            questions,
            trace_noise_std=0.0,
            trace_latent_noise=self._path_seed_noise(center_seeds),
        )
        sampled_outputs = self._question_only_latents(
            questions,
            trace_noise_std=0.0,
            trace_latent_noise=self._path_seed_noise(sampled_seeds),
        )
        model_paths = torch.stack(
            [
                center_outputs["implicit_residuals"],
                sampled_outputs["implicit_residuals"],
            ],
            dim=1,
        )

        matching = permutation_invariant_path_set_loss(
            model_paths,
            teacher_paths,
            distance_kwargs=self._distance_kwargs(),
        )
        model_relation = trace_path_distance(
            model_paths[:, 0],
            model_paths[:, 1],
            **self._distance_kwargs(),
        )
        teacher_relation = trace_path_distance(
            teacher_paths[:, 0],
            teacher_paths[:, 1],
            **self._distance_kwargs(),
        )
        relation_loss = (model_relation - teacher_relation).abs().mean()
        noncollapse = path_noncollapse_loss(
            model_paths,
            margin=float(
                self.trace_config.get("path_noncollapse_margin", 0.02)
            ),
        )

        answer_weight = self._scheduled_loss_weight(
            "answer",
            self.readcot_config.get("answer_loss_weight", 1.0),
        )
        answer_targets = [
            self.answer_template.format(answer) for answer in answers
        ]
        if answer_weight > 0:
            center_answer_loss = self._teacher_force_bottleneck(
                center_outputs,
                answer_targets,
            )
            sampled_answer_loss = self._teacher_force_bottleneck(
                sampled_outputs,
                answer_targets,
            )
            answer_loss = 0.5 * (
                center_answer_loss + sampled_answer_loss
            )
        else:
            answer_loss = center_outputs["implicit_residuals"].new_zeros(())
            center_answer_loss = answer_loss
            sampled_answer_loss = answer_loss

        dependency_loss = 0.5 * (
            compression_a["dep_loss"] + compression_b["dep_loss"]
        )
        dependency_f1 = 0.5 * (
            compression_a["dep_f1"] + compression_b["dep_f1"]
        )
        dependency_weight = self._scheduled_loss_weight(
            "dep",
            self.readcot_config.get("lambda_dep", 0.04),
        )
        relation_mix = float(
            self.trace_config.get("stage1_relation_mix", 0.25)
        )
        noncollapse_mix = float(
            self.trace_config.get("path_noncollapse_mix", 0.05)
        )
        formation = (
            matching["loss"]
            + relation_mix * relation_loss
            + noncollapse_mix * noncollapse
        )
        formation_weight = float(
            self.trace_config.get("stage1_formation_weight", 0.14)
        )
        total_loss = (
            answer_weight * answer_loss
            + dependency_weight * dependency_loss
            + formation_weight * formation
        )
        return {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "trace_stage1_center_answer_loss": center_answer_loss,
            "trace_stage1_sampled_answer_loss": sampled_answer_loss,
            "dep_loss": dependency_loss,
            "dep_f1": dependency_f1,
            "trace_stage1_formation_loss": formation,
            "trace_stage1_set_matching_loss": matching["loss"],
            "trace_stage1_position_loss": matching["position"],
            "trace_stage1_direction_loss": matching["direction"],
            "trace_stage1_step_loss": matching["step"],
            "trace_stage1_relation_loss": relation_loss,
            "trace_stage1_noncollapse_loss": noncollapse,
            "trace_stage1_model_pair_distance": model_relation.mean().detach(),
            "trace_stage1_teacher_pair_distance": teacher_relation.mean()
            .detach(),
            "trace_stage1_set_direct_fraction": matching[
                "direct_fraction"
            ],
            "trace_stage1_mean_teacher_count": answer_loss.new_tensor(
                float(sum(teacher_counts)) / float(max(1, len(teacher_counts)))
            ),
            "trace_stage1_multi_teacher_fraction": answer_loss.new_tensor(
                float(sum(count >= 2 for count in teacher_counts))
                / float(max(1, len(teacher_counts)))
            ),
            "trace_bottleneck_question_access": answer_loss.new_zeros(()),
            "lambda_answer_eff": answer_loss.new_tensor(answer_weight),
            "lambda_dep_eff": answer_loss.new_tensor(dependency_weight),
            "lambda_trace_formation_eff": answer_loss.new_tensor(
                formation_weight
            ),
        }

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
        latent_outputs = self._apply_answer_path_swap(
            latent_outputs,
            questions,
        )
        batch_size = len(questions)
        prompt_ids, prompt_mask = self._prompt_ids_for_generated_answer(
            batch_size
        )
        prompt_embeds = self.embedding(prompt_ids)
        # Transformers 5.5 expects full-length bookkeeping embeddings when a
        # populated cache is supplied, then slices only the uncached prompt.
        # The cached answer mask below still blocks every question K/V.
        generation_inputs_embeds = torch.cat(
            [latent_outputs["context_inputs_embeds"], prompt_embeds],
            dim=1,
        )
        generation_attention_mask = build_path_bottleneck_attention_mask(
            latent_outputs["question_attention_mask"],
            latent_outputs["latent_attention_mask"],
            prompt_mask,
        )

        hidden_prefix_k = int(
            self.trace_config.get("trace_eval_hidden_prefix_k", -1)
        )
        if hidden_prefix_k >= 0:
            if self.training:
                raise RuntimeError(
                    "trace_eval_hidden_prefix_k is evaluation-only"
                )
            n_latents = int(self.readcot_config.n_latents)
            if hidden_prefix_k > n_latents:
                raise ValueError(
                    f"trace_eval_hidden_prefix_k={hidden_prefix_k} "
                    f"exceeds n_latents={n_latents}"
                )
            question_length = latent_outputs["question_attention_mask"].shape[1]
            generation_attention_mask[
                :,
                question_length + hidden_prefix_k : question_length + n_latents,
            ] = 0
        hidden_drop_index = int(
            self.trace_config.get("trace_eval_hidden_drop_index", -1)
        )
        if hidden_drop_index >= 0:
            if self.training:
                raise RuntimeError(
                    "trace_eval_hidden_drop_index is evaluation-only"
                )
            n_latents = int(self.readcot_config.n_latents)
            if hidden_drop_index >= n_latents:
                raise ValueError(
                    f"trace_eval_hidden_drop_index={hidden_drop_index} "
                    f"exceeds the last latent index {n_latents - 1}"
                )
            question_length = latent_outputs[
                "question_attention_mask"
            ].shape[1]
            generation_attention_mask[
                :,
                question_length + hidden_drop_index,
            ] = 0

        generation_config = dict(self._get_generation_config())
        if do_sample is not None:
            generation_config["do_sample"] = bool(do_sample)
        if temperature is not None:
            generation_config["temperature"] = float(temperature)
        if top_p is not None:
            generation_config["top_p"] = float(top_p)
        pred_ids = self.llm.generate(
            inputs_embeds=generation_inputs_embeds,
            attention_mask=generation_attention_mask,
            past_key_values=latent_outputs["past_key_values"],
            **generation_config,
        )
        latent_outputs["answer_context_attention_mask"] = (
            generation_attention_mask.detach()
        )
        n_latent_forward = torch.full(
            (batch_size, 1),
            fill_value=int(self.readcot_config.n_latents),
            device=self.device,
            dtype=torch.long,
        )
        return pred_ids, n_latent_forward, latent_outputs

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
        latent_outputs = self._apply_answer_path_swap(
            latent_outputs,
            questions,
        )
        batch_size = len(questions)
        prompt_ids, prompt_mask = self._prompt_ids_for_generated_answer(
            batch_size
        )
        current_ids = torch.cat([prompt_ids, answer_input_ids], dim=1)
        current_mask = torch.cat(
            [prompt_mask, answer_attention_mask],
            dim=1,
        )
        attention_mask = build_path_bottleneck_attention_mask(
            latent_outputs["question_attention_mask"],
            latent_outputs["latent_attention_mask"],
            current_mask,
        )
        outputs = self.llm.forward(
            input_ids=current_ids,
            attention_mask=attention_mask,
            past_key_values=latent_outputs["past_key_values"],
            output_hidden_states=False,
        )
        prompt_length = prompt_ids.shape[1]
        answer_length = answer_input_ids.shape[1]
        answer_logits = outputs.logits[
            :,
            prompt_length - 1 : prompt_length - 1 + answer_length,
            :,
        ]
        flat_logits = answer_logits.reshape(-1, answer_logits.shape[-1])
        flat_ids = answer_input_ids.reshape(-1)
        selected = []
        for start in range(0, flat_logits.shape[0], 16):
            end = min(start + 16, flat_logits.shape[0])
            logprobs = F.log_softmax(flat_logits[start:end], dim=-1)
            selected.append(
                logprobs.gather(
                    dim=-1,
                    index=flat_ids[start:end].unsqueeze(-1),
                ).squeeze(-1)
            )
        answer_logprobs = torch.cat(selected, dim=0).reshape_as(
            answer_input_ids
        )
        answer_logprobs = torch.nan_to_num(
            answer_logprobs,
            nan=-30.0,
            neginf=-30.0,
            posinf=30.0,
        )
        return answer_logprobs.clamp(min=-30.0, max=30.0)

    def _sample_path_seeds(self, count: int) -> torch.Tensor:
        seeds = super()._sample_path_seeds(count)
        if self._sampling_stage2_paths:
            group_size = int(self.trace_rl_config.get("group_size", 8))
            if count % group_size:
                raise ValueError(
                    "Stage 2 rollout count must be divisible by group size"
                )
            seeds.view(-1, group_size, self.path_seed_dim)[:, 0].zero_()
        return seeds

    @torch.no_grad()
    def trace_rollout(self, questions: List[str], gt_answers):
        self._sampling_stage2_paths = True
        try:
            result = super().trace_rollout(questions, gt_answers)
        finally:
            self._sampling_stage2_paths = False
        experience = result[0]
        group_size = int(self.trace_rl_config.get("group_size", 8))
        center_indices = torch.arange(
            0,
            experience.accuracies.shape[0],
            group_size,
            device=experience.accuracies.device,
        )
        self._last_trace_metrics["center_path_accuracy"] = (
            experience.accuracies.index_select(0, center_indices).float().mean()
        )
        return result

    def _deterministic_visual_seeds(
        self,
        indices: Sequence[int],
        group_size: int,
    ) -> torch.Tensor:
        seeds = super()._deterministic_visual_seeds(indices, group_size)
        seeds[:, 0].zero_()
        return seeds

    @torch.no_grad()
    def _build_trace_visual_records(
        self,
        batch,
        latent_outputs,
        output_strings: Sequence[str],
    ) -> List[dict]:
        records = super()._build_trace_visual_records(
            batch,
            latent_outputs,
            output_strings,
        )
        rationale_sets = self._decode_rationale_sets(batch)
        for record_index, (record, rationales) in enumerate(
            zip(records, rationale_sets)
        ):
            teacher_residuals = []
            teacher_assignments = []
            for rationale in rationales:
                _, rationale_compression = self._teacher_paths_for_rationales(
                    [batch["question"][record_index]],
                    [batch["answer"][record_index]],
                    [rationale],
                )
                teacher_residuals.append(
                    rationale_compression[
                        "aggregated_explicit_residuals"
                    ][0]
                )
                teacher_assignments.append(
                    rationale_compression["assignments"][0]
                )
            record["trace_seed_schema"] = (
                "deterministic_center_plus_iid_continuous"
            )
            record["path_bottleneck"] = {
                "question_answer_attention_access": 0,
                "latent_answer_attention_access": int(
                    self.readcot_config.n_latents
                ),
            }
            record["n_verified_rationales"] = len(rationales)
            record["rationale_fingerprints"] = [
                item["fingerprint"] for item in rationales
            ]
            record["rationale_teacher_residuals"] = (
                torch.stack(teacher_residuals)
                .detach()
                .to(torch.float16)
                .cpu()
            )
            record["rationale_teacher_assignments"] = [
                assignment.detach().to(torch.float16).cpu()
                for assignment in teacher_assignments
            ]
            # Synthetic assignment perturbations were a property of the prior
            # model and are not valid multi-rationale teachers for final TRACE.
            record.pop("multiview_teacher_assignments", None)
            record.pop("multiview_teacher_residuals", None)
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
            log_dir / f"trace_final_visual_{split}.pt",
        )
