import json
import re
from contextlib import nullcontext
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model_base import LitCoTModelBase
from ..modules.readcot import (
    LatentRelationHead,
    ReadStepCompressor,
    aggregate_step_residuals,
    cosine_similarity_mean,
    dependency_bce_loss,
    dependency_f1_score,
    induce_dependency_supervision,
    reconstruct_dependency_logits,
    sanitize_dependency_supervision,
    select_dependency_critical_anchors,
)


_COMPACT_EQUATION_PATTERN = re.compile(
    r"(?:\$?-?\d[\d,]*(?:\.\d+)?|[A-Za-z])"
    r"(?:\s*(?:[+\-*/xX×÷]|/)\s*(?:\$?-?\d[\d,]*(?:\.\d+)?|[A-Za-z]))+"
    r"\s*=\s*\$?-?\d[\d,]*(?:\.\d+)?"
)
_MATH_TOKEN_PATTERN = re.compile(r"\$?-?\d[\d,]*(?:\.\d+)?|[+\-*/xX×÷=]")


class LitREADCoT(LitCoTModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

        self.readcot_config = model_kwargs.readcot_config
        hidden_size = self.hidden_size
        residual_proj_size = self.readcot_config.get("residual_proj_size", hidden_size)
        bridge_hidden_size = self.readcot_config.get("bridge_hidden_size", hidden_size)
        self.anchor_header = "Anchors:"

        self.state_norm = nn.LayerNorm(hidden_size)
        self.step_compressor = ReadStepCompressor(
            hidden_size=hidden_size,
            n_latents=self.readcot_config.n_latents,
            dropout=self.readcot_config.get("dropout", 0.0),
        )
        self.latent_relation = LatentRelationHead(
            hidden_size=hidden_size,
            relation_hidden_size=self.readcot_config.get("relation_hidden_size", hidden_size),
        )
        self.latent_bridge = nn.Sequential(
            nn.Linear(hidden_size, bridge_hidden_size),
            nn.GELU(),
            nn.Linear(bridge_hidden_size, hidden_size),
        )
        self.residual_projector = nn.Sequential(
            nn.Linear(hidden_size, residual_proj_size),
            nn.GELU(),
            nn.Linear(residual_proj_size, residual_proj_size),
        )
        self.use_anchor_gate = bool(self.readcot_config.get("use_anchor_gate", False))
        if self.use_anchor_gate:
            self.anchor_gate_predictor = nn.Sequential(
                nn.Linear(hidden_size, bridge_hidden_size),
                nn.GELU(),
                nn.Linear(bridge_hidden_size, self.readcot_config.n_latents),
            )

    def _decode_step_lists(self, batch) -> List[List[str]]:
        if "step_list_json" in batch:
            return [json.loads(step_json) for step_json in batch["step_list_json"]]
        return [steps.split("\n") if steps else ["\n"] for steps in batch["steps"]]

    def _decode_cached_matrices(self, batch, key: str) -> List:
        json_key = f"{key}_json"
        if json_key not in batch:
            return [None] * len(batch["question"])

        matrices = []
        for matrix_json in batch[json_key]:
            if not matrix_json:
                matrices.append(None)
                continue
            try:
                matrices.append(json.loads(matrix_json))
            except json.JSONDecodeError:
                matrices.append(None)
        return matrices

    def _decode_cached_indices(self, batch, key: str) -> List:
        json_key = f"{key}_json"
        if json_key not in batch:
            return [None] * len(batch["question"])

        decoded = []
        for raw_json in batch[json_key]:
            if not raw_json:
                decoded.append(None)
                continue
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError:
                decoded.append(None)
                continue
            if not isinstance(parsed, list):
                decoded.append(None)
                continue
            indices = []
            for value in parsed:
                try:
                    indices.append(int(value))
                except (TypeError, ValueError):
                    continue
            decoded.append(indices if indices else None)
        return decoded

    def _decode_anchor_indices(self, batch) -> List:
        return self._decode_cached_indices(batch, "anchor_indices")

    def _decode_anchor_bonus_indices(self, batch) -> List:
        return self._decode_cached_indices(batch, "anchor_bonus_indices")

    def _prepare_raw_texts(self, text_list, padding_side="right", prefix="", suffix=""):
        if isinstance(text_list, str):
            text_list = [text_list]
        texts = [prefix + text + suffix for text in text_list]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side
        return inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)

    def _encode_single_segment(self, part: str, text: str, prefix: str = "", suffix: str = ""):
        base_text = getattr(self, f"{part}_template").format(text)
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        base_ids = self.tokenizer.encode(base_text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

        if len(base_ids) == 0:
            base_ids = self.tokenizer.encode("\n", add_special_tokens=False)

        input_ids = torch.tensor(
            [prefix_ids + base_ids + suffix_ids],
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        content_mask = torch.zeros_like(input_ids)
        start = len(prefix_ids)
        end = start + len(base_ids)
        content_mask[:, start:end] = 1
        return input_ids, attention_mask, content_mask

    def _pool_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden_states * weights).sum(dim=1) / denom

    def _last_content_hidden(self, hidden_states: torch.Tensor, content_mask: torch.Tensor) -> torch.Tensor:
        positions = content_mask[0].nonzero(as_tuple=False).flatten()
        if len(positions) == 0:
            return hidden_states[:, -1, :]
        return hidden_states[:, positions[-1], :]

    def _extract_explicit_sample_features(
        self,
        question: str,
        step_list: Sequence[str],
        answer: str,
        dependency_matrix=None,
        confidence_matrix=None,
    ) -> Dict[str, torch.Tensor]:
        if len(step_list) == 0:
            step_list = ["\n"]

        question_ids, question_mask, _ = self._encode_single_segment("question", question)
        outputs = self.llm.forward(
            input_ids=question_ids,
            attention_mask=question_mask,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        all_attention_mask = question_mask
        prev_state = self.state_norm(outputs.hidden_states[-1][:, -1, :])

        step_states = []
        step_residuals = []
        for idx, step in enumerate(step_list):
            prefix = self.thinking_separator if idx == 0 else ""
            suffix = "\n" if idx < len(step_list) - 1 else ""
            step_ids, step_mask, content_mask = self._encode_single_segment("steps", step, prefix=prefix, suffix=suffix)
            all_attention_mask = torch.cat([all_attention_mask, step_mask], dim=1)
            outputs = self.llm.forward(
                input_ids=step_ids,
                attention_mask=all_attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            step_hidden = outputs.hidden_states[-1][:, -step_ids.shape[1] :, :]
            pooled_state = self._pool_hidden(step_hidden, content_mask)
            current_state = self.state_norm(self._last_content_hidden(step_hidden, content_mask))
            step_states.append(pooled_state.squeeze(0))
            step_residuals.append((current_state - prev_state).squeeze(0))
            prev_state = current_state

        cached_dependency, cached_confidence = sanitize_dependency_supervision(
            dependency_matrix=dependency_matrix,
            confidence_matrix=confidence_matrix,
            n_steps=len(step_list),
            device=self.device,
            negative_confidence=self.readcot_config.get("negative_edge_confidence", 0.25),
        )
        if cached_dependency is None:
            cached_dependency, cached_confidence = induce_dependency_supervision(
                question=question,
                steps=step_list,
                answer=answer,
                negative_confidence=self.readcot_config.get("negative_edge_confidence", 0.25),
            )
            cached_dependency = cached_dependency.to(self.device)
            cached_confidence = cached_confidence.to(self.device)
        return {
            "step_states": torch.stack(step_states, dim=0),
            "step_residuals": torch.stack(step_residuals, dim=0),
            "dependency_matrix": cached_dependency,
            "confidence_matrix": cached_confidence,
        }

    def _collect_explicit_batch_features(
        self,
        questions: Sequence[str],
        step_lists: Sequence[Sequence[str]],
        answers: Sequence[str],
        dependency_matrices: Sequence = None,
        confidence_matrices: Sequence = None,
    ) -> List[Dict[str, torch.Tensor]]:
        dependency_matrices = dependency_matrices or [None] * len(questions)
        confidence_matrices = confidence_matrices or [None] * len(questions)
        detach_teacher = self.readcot_config.get("detach_explicit_teacher", True)
        teacher_was_training = self.llm.training
        if detach_teacher:
            self.llm.eval()
        ctx = torch.no_grad() if detach_teacher else nullcontext()
        with ctx:
            outputs = [
                self._extract_explicit_sample_features(
                    question=q,
                    step_list=s,
                    answer=a,
                    dependency_matrix=d,
                    confidence_matrix=c,
                )
                for q, s, a, d, c in zip(questions, step_lists, answers, dependency_matrices, confidence_matrices)
            ]
        if teacher_was_training:
            self.llm.train()
        return outputs

    def _question_only_latents(self, questions: Sequence[str]) -> Dict[str, torch.Tensor]:
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
        outputs = self.llm.forward(
            inputs_embeds=question_inputs_embeds,
            attention_mask=question_attention_mask,
            position_ids=question_position_ids,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        all_attention_mask = question_attention_mask
        prev_state = self.state_norm(outputs.hidden_states[-1][:, -1, :])
        anchor_gate = None
        if self.use_anchor_gate:
            anchor_gate = torch.sigmoid(self.anchor_gate_predictor(prev_state))

        latent_inputs_embeds = []
        implicit_residuals = []
        for latent_idx in range(n_latents):
            current_latent = self.latent_bridge(prev_state).to(question_inputs_embeds.dtype).unsqueeze(1)
            if anchor_gate is not None and self.readcot_config.get("anchor_gate_apply_to_latents", True):
                gate_scale = float(self.readcot_config.get("anchor_gate_scale", 0.5))
                gate_value = anchor_gate[:, latent_idx].view(batch_size, 1, 1).to(current_latent.dtype)
                current_latent = current_latent * (1.0 + gate_scale * gate_value)
            latent_inputs_embeds.append(current_latent)
            latent_mask = torch.ones(
                batch_size,
                1,
                device=self.device,
                dtype=all_attention_mask.dtype,
            )
            all_attention_mask = torch.cat([all_attention_mask, latent_mask], dim=1)
            latent_position_ids = self.make_position_ids_for_current_input(
                all_attention_mask,
                current_latent.shape[1],
            )
            outputs = self.llm.forward(
                inputs_embeds=current_latent,
                attention_mask=all_attention_mask,
                position_ids=latent_position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            current_state = self.state_norm(outputs.hidden_states[-1][:, -1, :])
            implicit_residuals.append(current_state - prev_state)
            prev_state = current_state

        latent_inputs_embeds = torch.cat(latent_inputs_embeds, dim=1)
        context_inputs_embeds = torch.cat([question_inputs_embeds, latent_inputs_embeds], dim=1)
        latent_attention_mask = torch.ones(
            batch_size,
            n_latents,
            device=self.device,
            dtype=question_attention_mask.dtype,
        )
        return {
            "question_input_ids": question_input_ids,
            "question_attention_mask": question_attention_mask,
            "latent_inputs_embeds": latent_inputs_embeds,
            "latent_attention_mask": latent_attention_mask,
            "context_inputs_embeds": context_inputs_embeds,
            "context_attention_mask": all_attention_mask,
            "past_key_values": past_key_values,
            "implicit_residuals": torch.stack(implicit_residuals, dim=1),
            "anchor_gate": anchor_gate,
        }

    def _masked_causal_ce(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Exact masked causal CE with bounded token-by-vocabulary workspace.

        Computing unreduced CE over every target token at once materializes a
        ``[batch * target_length, vocabulary]`` softmax workspace.  Qwen's
        vocabulary makes that transient allocation large enough to OOM on a
        long compact target even though the forward logits themselves fit.
        Token chunking changes only the reduction order: every token keeps the
        same full-vocabulary CE and all chunks share one global mask
        denominator.
        """
        if logits.ndim != 3:
            raise ValueError("logits must have shape [batch, sequence, vocab]")
        if input_ids.ndim != 2 or loss_mask.ndim != 2:
            raise ValueError("input_ids and loss_mask must be two-dimensional")
        if tuple(logits.shape[:2]) != tuple(input_ids.shape):
            raise ValueError("logits and input_ids must share batch/sequence axes")
        if tuple(loss_mask.shape) != tuple(input_ids.shape):
            raise ValueError("loss_mask must match input_ids")
        if input_ids.shape[1] <= 1:
            return logits[:, :0, :].sum(dtype=torch.float32)

        shift_labels = input_ids[:, 1:]
        shift_mask = loss_mask[:, 1:].float()
        denominator = shift_mask.sum().clamp_min(1.0)
        numerator = logits[:, :0, :].sum(dtype=torch.float32)
        token_chunk_size = 8
        token_count = int(shift_labels.shape[1])

        def chunk_weighted_ce(
            chunk_logits: torch.Tensor,
            chunk_labels: torch.Tensor,
            chunk_mask: torch.Tensor,
        ) -> torch.Tensor:
            chunk_losses = F.cross_entropy(
                chunk_logits,
                chunk_labels,
                reduction="none",
            )
            return (chunk_losses * chunk_mask).sum()

        for batch_index in range(int(logits.shape[0])):
            for start in range(0, token_count, token_chunk_size):
                end = min(start + token_chunk_size, token_count)
                chunk_mask = shift_mask[batch_index, start:end]
                if not bool(chunk_mask.ne(0).any()):
                    # Preserve a differentiable zero for all-masked chunks
                    # without allocating a full-vocabulary CE workspace.
                    numerator = numerator + (
                        logits[batch_index, start:end, :].sum(
                            dtype=torch.float32
                        )
                        * 0.0
                    )
                    continue
                chunk_logits = logits[batch_index, start:end, :]
                chunk_labels = shift_labels[batch_index, start:end]
                if torch.is_grad_enabled() and chunk_logits.requires_grad:
                    chunk_numerator = checkpoint(
                        chunk_weighted_ce,
                        chunk_logits,
                        chunk_labels,
                        chunk_mask,
                        use_reentrant=False,
                    )
                else:
                    chunk_numerator = chunk_weighted_ce(
                        chunk_logits,
                        chunk_labels,
                        chunk_mask,
                    )
                numerator = numerator + chunk_numerator
        return numerator / denominator

    def _teacher_force_target(
        self,
        past_key_values,
        context_attention_mask: torch.Tensor,
        target_texts: Sequence[str],
        prefix_texts: Sequence[str] = None,
    ) -> torch.Tensor:
        batch_size = len(target_texts)
        target_input_ids, target_attention_mask = self._prepare_raw_texts(
            target_texts,
            padding_side="right",
            suffix=self.tokenizer.eos_token,
        )
        separator_ids = torch.ones(
            batch_size,
            1,
            device=self.device,
            dtype=torch.long,
        ) * self.thinking_separator_id
        separator_mask = torch.ones_like(separator_ids)

        input_pieces = [separator_ids]
        loss_mask_pieces = [torch.zeros_like(separator_mask)]
        attention_mask_pieces = [separator_mask]

        if prefix_texts is not None:
            prefix_input_ids, prefix_attention_mask = self._prepare_raw_texts(
                prefix_texts,
                padding_side="right",
            )
            input_pieces.append(prefix_input_ids)
            loss_mask_pieces.append(torch.zeros_like(prefix_attention_mask))
            attention_mask_pieces.append(prefix_attention_mask)

        input_pieces.append(target_input_ids)
        loss_mask_pieces.append(target_attention_mask)
        attention_mask_pieces.append(target_attention_mask)

        final_input_ids = torch.cat(input_pieces, dim=1)
        final_loss_mask = torch.cat(loss_mask_pieces, dim=1)
        final_attention_mask = torch.cat([context_attention_mask] + attention_mask_pieces, dim=1)
        final_position_ids = self.make_position_ids_for_current_input(
            final_attention_mask,
            final_input_ids.shape[1],
        )
        outputs = self.llm.forward(
            input_ids=final_input_ids,
            attention_mask=final_attention_mask,
            position_ids=final_position_ids,
            past_key_values=past_key_values,
            output_hidden_states=False,
        )
        return self._masked_causal_ce(outputs.logits, final_input_ids, final_loss_mask)

    def _compute_residual_alignment(
        self,
        implicit_residuals: torch.Tensor,
        aggregated_explicit_residuals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        projected_implicit = self.residual_projector(implicit_residuals)
        projected_explicit_target = self.residual_projector(aggregated_explicit_residuals.detach()).detach()

        mse_implicit = projected_implicit
        mse_target = projected_explicit_target
        if self.readcot_config.get("normalize_residual_mse", False):
            mse_implicit = F.normalize(mse_implicit, p=2, dim=-1)
            mse_target = F.normalize(mse_target, p=2, dim=-1)

        residual_loss_type = self.readcot_config.get("residual_loss_type", "mse")
        if residual_loss_type == "mse":
            residual_mse_loss = F.mse_loss(mse_implicit, mse_target)
        elif residual_loss_type == "smooth_l1":
            residual_mse_loss = F.smooth_l1_loss(mse_implicit, mse_target)
        else:
            raise ValueError(f"Unsupported residual_loss_type: {residual_loss_type}")

        residual_mse_clamp = self.readcot_config.get("residual_mse_clamp", None)
        if residual_mse_clamp is not None:
            residual_mse_loss = residual_mse_loss.clamp(max=float(residual_mse_clamp))
        residual_cosine_distance = 1.0 - cosine_similarity_mean(projected_implicit, projected_explicit_target)
        residual_loss = residual_mse_loss + self.readcot_config.get("residual_cosine_weight", 0.0) * residual_cosine_distance

        return {
            "projected_implicit": projected_implicit,
            "projected_explicit_target": projected_explicit_target,
            "residual_loss": residual_loss,
            "residual_mse_loss": residual_mse_loss,
            "residual_cosine_distance": residual_cosine_distance,
            "residual_similarity": cosine_similarity_mean(projected_implicit.detach(), projected_explicit_target),
        }

    def _compress_explicit_reasoning(self, explicit_features: List[Dict[str, torch.Tensor]]):
        dependency_losses = []
        dependency_f1_scores = []
        aggregated_explicit_residuals = []
        assignments = []
        dependency_probs = []
        relation_probs = []
        relation_logits = []
        dependency_logits = []

        for item in explicit_features:
            compressed_latents, assignment = self.step_compressor(item["step_states"])
            rel_logits, rel_probs = self.latent_relation.forward_probs(compressed_latents)
            dep_logits, dep_probs = reconstruct_dependency_logits(
                assignment=assignment,
                relation_probs=rel_probs,
                center_relations=self.readcot_config.get("center_relation_probs", True),
            )

            dependency_losses.append(
                dependency_bce_loss(
                    dep_logits,
                    item["dependency_matrix"],
                    confidence=item.get("confidence_matrix"),
                    pos_weight_max=self.readcot_config.get("dep_pos_weight_max", None),
                    loss_clamp=self.readcot_config.get("dep_loss_clamp", None),
                )
            )
            dependency_f1_scores.append(
                dependency_f1_score(
                    dep_probs=dep_probs,
                    gold=item["dependency_matrix"],
                    threshold=self.readcot_config.get("dependency_threshold", 0.5),
                )
            )
            aggregated_explicit_residuals.append(aggregate_step_residuals(assignment, item["step_residuals"]))
            assignments.append(assignment)
            dependency_probs.append(dep_probs)
            relation_probs.append(rel_probs)
            relation_logits.append(rel_logits)
            dependency_logits.append(dep_logits)

        return {
            "dep_loss": torch.stack(dependency_losses).mean(),
            "dep_f1": torch.stack(dependency_f1_scores).mean(),
            "aggregated_explicit_residuals": torch.stack(aggregated_explicit_residuals, dim=0),
            "assignments": assignments,
            "dependency_probs": dependency_probs,
            "relation_probs": relation_probs,
            "relation_logits": relation_logits,
            "dependency_logits": dependency_logits,
        }

    def _normalize_compact_equation(self, equation: str) -> str:
        equation = equation.replace("×", "*").replace("÷", "/")
        equation = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", equation)
        equation = re.sub(r"\s*([+\-*/=])\s*", r"\1", equation)
        return equation.strip(" .,;")

    def _compact_anchor_step(self, step: str) -> str:
        step = re.sub(r"\s+", " ", step.strip())
        if not step:
            return step

        equations = [self._normalize_compact_equation(match.group(0)) for match in _COMPACT_EQUATION_PATTERN.finditer(step)]
        if equations:
            compact = "; ".join(dict.fromkeys(equations))
        else:
            math_tokens = _MATH_TOKEN_PATTERN.findall(step)
            if len(math_tokens) >= 3:
                compact = " ".join(math_tokens)
            else:
                compact = step

        max_chars = self.readcot_config.get("compact_anchor_max_chars", None)
        if max_chars is not None and len(compact) > int(max_chars):
            limit = int(max_chars)
            if equations:
                complete = []
                for equation in equations:
                    candidate = "; ".join(complete + [equation])
                    if complete and len(candidate) > limit:
                        break
                    complete.append(equation)
                    if len(candidate) > limit:
                        break
                compact = "; ".join(complete)
            else:
                words = compact.split()
                complete = []
                for word in words:
                    candidate = " ".join(complete + [word])
                    if complete and len(candidate) > limit:
                        break
                    complete.append(word)
                    if len(candidate) > limit:
                        break
                compact = " ".join(complete)
        return compact

    def _format_anchor_step(self, step: str) -> str:
        mode = self.readcot_config.get("anchor_text_mode", "raw")
        if mode == "compact_equation":
            return self._compact_anchor_step(step)
        if mode == "compact_text":
            compact = re.sub(r"\s+", " ", step.strip())
            max_chars = self.readcot_config.get("compact_anchor_max_chars", None)
            if max_chars is not None and len(compact) > int(max_chars):
                words = compact.split()
                complete = []
                for word in words:
                    candidate = " ".join(complete + [word])
                    if complete and len(candidate) > int(max_chars):
                        break
                    complete.append(word)
                    if len(candidate) > int(max_chars):
                        break
                compact = " ".join(complete)
            return compact
        if mode == "raw":
            return step.strip()
        raise ValueError(f"Unsupported anchor_text_mode: {mode}")

    def _anchor_gate_target_from_scores(
        self,
        assignment: torch.Tensor,
        anchor_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Project step-level anchor importance to latent slots with the shared assignment.

        This implements the paper-level signal g_m = sum_j A_mj u_j.  The target
        is used only for training an anchor-gate predictor; inference uses the
        predicted gate from question-only latent states.
        """
        if anchor_scores is None or anchor_scores.numel() == 0:
            return assignment.new_zeros(assignment.shape[0])
        step_scores = anchor_scores.to(device=assignment.device, dtype=assignment.dtype)
        step_scores = step_scores - step_scores.min()
        score_denom = step_scores.max().clamp_min(1e-6)
        step_scores = step_scores / score_denom
        gate_target = torch.matmul(assignment, step_scores)
        gate_target = gate_target / gate_target.max().clamp_min(1e-6)
        return gate_target.clamp(0.0, 1.0)

    def _build_hybrid_targets(
        self,
        step_lists: Sequence[Sequence[str]],
        answers: Sequence[str],
        explicit_features: List[Dict[str, torch.Tensor]],
        compression_outputs: Dict[str, torch.Tensor],
        implicit_residuals: torch.Tensor,
        anchor_indices: Sequence = None,
        anchor_bonus_indices: Sequence = None,
        return_anchor_gate_targets: bool = False,
    ) -> Tuple[List[str], List[List[int]]]:
        targets = []
        all_anchor_indices = []
        anchor_gate_targets = []
        strategy = self.readcot_config.get("anchor_strategy", "dependency_critical")
        anchor_indices = anchor_indices or [None] * len(step_lists)
        anchor_bonus_indices = anchor_bonus_indices or [None] * len(step_lists)

        projected_implicit = self.residual_projector(implicit_residuals.detach())
        projected_explicit = self.residual_projector(compression_outputs["aggregated_explicit_residuals"].detach())
        latent_residual_errors = (projected_implicit - projected_explicit).pow(2).mean(dim=-1)

        for idx, (
            steps,
            answer,
            explicit_item,
            assignment,
            dep_probs,
            latent_error,
            cached_anchor_indices,
            cached_anchor_bonus_indices,
        ) in enumerate(
            zip(
                step_lists,
                answers,
                explicit_features,
                compression_outputs["assignments"],
                compression_outputs["dependency_probs"],
                latent_residual_errors,
                anchor_indices,
                anchor_bonus_indices,
            )
        ):
            dep_error_matrix = (dep_probs.detach() - explicit_item["dependency_matrix"].detach()).abs()
            if explicit_item.get("confidence_matrix") is not None:
                dep_error_matrix = dep_error_matrix * explicit_item["confidence_matrix"].detach()
            dependency_errors = dep_error_matrix.sum(dim=0) + dep_error_matrix.sum(dim=1)
            residual_errors = torch.einsum("mk,m->k", assignment.detach(), latent_error.detach())
            anchor_bonus = torch.zeros(len(steps), dtype=torch.float32)
            if cached_anchor_bonus_indices is not None:
                for bonus_idx in cached_anchor_bonus_indices:
                    if 0 <= bonus_idx < len(steps):
                        anchor_bonus[bonus_idx] += 1.0
            computed_anchor_indices, anchor_scores = select_dependency_critical_anchors(
                steps=steps,
                dependency_matrix=explicit_item["dependency_matrix"].detach().cpu(),
                answer=answer,
                n_anchors=self.readcot_config.n_anchors,
                alpha=self.readcot_config.get("anchor_alpha", 1.0),
                beta=self.readcot_config.get("anchor_beta", 1.0),
                gamma=self.readcot_config.get("anchor_gamma", 1.0),
                dependency_errors=dependency_errors.detach().cpu(),
                residual_errors=residual_errors.detach().cpu(),
                anchor_bonus=anchor_bonus,
                anchor_bonus_weight=self.readcot_config.get("cached_anchor_bonus_weight", 2.0),
                strategy=strategy,
                length_penalty=self.readcot_config.get("anchor_length_penalty", 0.0),
            )
            n_select = min(len(steps), self.readcot_config.n_anchors)
            if cached_anchor_indices is not None and n_select > 0:
                selected_anchor_indices = sorted(
                    {idx for idx in cached_anchor_indices if 0 <= idx < len(steps)}
                )[:n_select]
                if len(selected_anchor_indices) < n_select:
                    fallback_anchor_indices, _ = select_dependency_critical_anchors(
                        steps=steps,
                        dependency_matrix=explicit_item["dependency_matrix"].detach().cpu(),
                        answer=answer,
                        n_anchors=self.readcot_config.n_anchors,
                        alpha=self.readcot_config.get("anchor_alpha", 1.0),
                        beta=self.readcot_config.get("anchor_beta", 1.0),
                        gamma=self.readcot_config.get("anchor_gamma", 1.0),
                        strategy=strategy,
                        length_penalty=self.readcot_config.get("anchor_length_penalty", 0.0),
                    )
                    for fallback_idx in fallback_anchor_indices:
                        if fallback_idx not in selected_anchor_indices:
                            selected_anchor_indices.append(fallback_idx)
                        if len(selected_anchor_indices) >= n_select:
                            break
                    selected_anchor_indices = sorted(selected_anchor_indices[:n_select])
            else:
                selected_anchor_indices = computed_anchor_indices
            all_anchor_indices.append(selected_anchor_indices)
            if return_anchor_gate_targets:
                anchor_gate_targets.append(self._anchor_gate_target_from_scores(assignment.detach(), anchor_scores))

            anchor_lines = []
            for i in selected_anchor_indices:
                if not steps[i].strip():
                    continue
                anchor_text = self._format_anchor_step(steps[i])
                if anchor_text:
                    anchor_lines.append(f"- {anchor_text}")
            if len(anchor_lines) == 0:
                anchor_lines = ["- no anchor selected"]
            anchor_block = self.anchor_header + "\n" + "\n".join(anchor_lines)
            targets.append(f"{anchor_block}\n{self.thinking_separator}{self.answer_template.format(answer)}")

        if return_anchor_gate_targets:
            return targets, all_anchor_indices, anchor_gate_targets
        return targets, all_anchor_indices

    def _extract_predicted_anchor_text(self, output_string: str) -> str:
        if self.anchor_header not in output_string:
            return ""
        if self.answer_template.format("") in output_string:
            anchor_part = output_string.split(self.answer_template.format(""), 1)[0]
        else:
            anchor_part = output_string
        anchor_part = anchor_part.split(self.anchor_header, 1)[-1]
        return anchor_part.strip("#\n ")

    def _get_generation_config(self):
        if self.readcot_config.get("use_hybrid", False) and self.model_kwargs.get("hybrid_generation_config") is not None:
            return self.model_kwargs.hybrid_generation_config
        return self.model_kwargs.answer_generation_config

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
        latent_outputs = self._question_only_latents(questions)
        residual_outputs = self._compute_residual_alignment(
            implicit_residuals=latent_outputs["implicit_residuals"],
            aggregated_explicit_residuals=compression_outputs["aggregated_explicit_residuals"],
        )

        answer_targets = [self.answer_template.format(answer) for answer in answers]
        answer_loss = self._teacher_force_target(
            past_key_values=latent_outputs["past_key_values"],
            context_attention_mask=latent_outputs["context_attention_mask"],
            target_texts=answer_targets,
        )

        anchor_loss = answer_loss.new_zeros(())
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
            anchor_loss = self._teacher_force_target(
                past_key_values=latent_outputs["past_key_values"],
                context_attention_mask=latent_outputs["context_attention_mask"],
                target_texts=anchor_targets,
            )
        anchor_gate_loss = answer_loss.new_zeros(())
        if self.use_anchor_gate and use_anchor_supervision and latent_outputs.get("anchor_gate") is not None:
            gate_target = torch.stack(anchor_gate_targets, dim=0).detach()
            anchor_gate_loss = F.smooth_l1_loss(latent_outputs["anchor_gate"], gate_target)

        total_loss = answer_loss
        total_loss = total_loss + self.readcot_config.lambda_dep * compression_outputs["dep_loss"]
        total_loss = total_loss + self.readcot_config.lambda_res * residual_outputs["residual_loss"]
        if use_anchor_supervision:
            total_loss = total_loss + self.readcot_config.lambda_anchor * anchor_loss
        if self.use_anchor_gate:
            total_loss = total_loss + self.readcot_config.get("lambda_anchor_gate", 0.0) * anchor_gate_loss

        return {
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
        }

    @torch.no_grad()
    def read_generate(self, questions: List[str]):
        latent_outputs = self._question_only_latents(questions)
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

        pred_ids = self.llm.generate(
            inputs_embeds=all_inputs_embeds,
            attention_mask=all_attention_mask,
            **self._get_generation_config(),
        )
        n_latent_forward = torch.ones(
            batch_size,
            1,
            device=self.device,
            dtype=torch.long,
        ) * self.readcot_config.n_latents
        return pred_ids, n_latent_forward

    @torch.no_grad()
    def _compute_structure_metrics(self, batch):
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
        latent_outputs = self._question_only_latents(questions)
        residual_outputs = self._compute_residual_alignment(
            implicit_residuals=latent_outputs["implicit_residuals"],
            aggregated_explicit_residuals=compression_outputs["aggregated_explicit_residuals"],
        )
        return {
            "dep_f1": compression_outputs["dep_f1"],
            "residual_similarity": residual_outputs["residual_similarity"],
        }

    @torch.no_grad()
    def eval_generation(self, batch, split="val", batch_idx=None, dataloader_idx=0):
        indices = batch["idx"].tolist()
        questions = batch["question"]
        answers = batch["answer"]
        steps = batch["steps"]

        outputs_token_ids, n_latent_forward = self.read_generate(questions=questions)
        output_strings = self.tokenizer.batch_decode(outputs_token_ids, skip_special_tokens=True)

        all_acc = []
        all_output_length = []
        all_latent_forward = []
        for i, q, s, a, o_ids, o_str, nlf in zip(
            indices, questions, steps, answers, outputs_token_ids, output_strings, n_latent_forward
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

            all_acc.append(acc)
            all_output_length.append(o_length)
            all_latent_forward.append(nlf.item())

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
