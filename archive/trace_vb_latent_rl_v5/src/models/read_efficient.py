from typing import Dict, Sequence

import torch

from .read import LitREADCoT
from ..modules.readcot import induce_dependency_supervision, sanitize_dependency_supervision


class LitREADCoTEfficient(LitREADCoT):
    """CSA/HCA-inspired READ-CoT variant with fewer LLM forward passes.

    This class intentionally lives beside ``LitREADCoT`` instead of replacing it.
    The original READ-CoT path remains available via ``src.models.read.LitREADCoT``.

    Efficiency changes:
    1. Packed explicit teacher: question + all explicit CoT steps are encoded in
       one causal forward, then step states/residuals are read from span positions.
       This is mathematically close to the previous cached step-by-step teacher
       path, but avoids K separate LLM calls for K steps.
    2. Block latent reasoning: all M latent slots are fed in one causal block
       forward. This follows the CSA/HCA intuition of operating over compressed
       block-level memory entries instead of repeatedly reading the full model M
       times, while preserving causal interaction among latent slots.
    """

    def _extract_explicit_sample_features(
        self,
        question: str,
        step_list: Sequence[str],
        answer: str,
        dependency_matrix=None,
        confidence_matrix=None,
    ) -> Dict[str, torch.Tensor]:
        mode = self.readcot_config.get("explicit_feature_mode", "packed")
        if mode != "packed":
            return super()._extract_explicit_sample_features(
                question=question,
                step_list=step_list,
                answer=answer,
                dependency_matrix=dependency_matrix,
                confidence_matrix=confidence_matrix,
            )
        return self._extract_explicit_sample_features_packed(
            question=question,
            step_list=step_list,
            answer=answer,
            dependency_matrix=dependency_matrix,
            confidence_matrix=confidence_matrix,
        )

    def _extract_explicit_sample_features_packed(
        self,
        question: str,
        step_list: Sequence[str],
        answer: str,
        dependency_matrix=None,
        confidence_matrix=None,
    ) -> Dict[str, torch.Tensor]:
        if len(step_list) == 0:
            step_list = ["\n"]

        input_ids, question_end_idx, step_spans = self._build_packed_explicit_ids(question, step_list)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]

        prev_state = self.state_norm(hidden[:, question_end_idx, :])
        step_states = []
        step_residuals = []
        for start, end in step_spans:
            step_hidden = hidden[:, start:end, :]
            pooled_state = step_hidden.mean(dim=1)
            current_state = self.state_norm(step_hidden[:, -1, :])
            step_states.append(pooled_state.squeeze(0))
            step_residuals.append((current_state - prev_state).squeeze(0))
            prev_state = current_state

        effective_steps = list(step_list[: len(step_spans)])
        if len(effective_steps) == 0:
            effective_steps = ["\n"]

        cached_dependency, cached_confidence = sanitize_dependency_supervision(
            dependency_matrix=dependency_matrix,
            confidence_matrix=confidence_matrix,
            n_steps=len(step_spans),
            device=self.device,
            negative_confidence=self.readcot_config.get("negative_edge_confidence", 0.25),
        )
        if cached_dependency is None:
            cached_dependency, cached_confidence = induce_dependency_supervision(
                question=question,
                steps=effective_steps,
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

    def _build_packed_explicit_ids(self, question: str, step_list: Sequence[str]):
        question_text = self.question_template.format(question)
        if self.readcot_config.get(
            "explicit_include_speed_prompt",
            False,
        ):
            question_text += self.speed_template.format(1)
        question_ids = self.tokenizer.encode(question_text, add_special_tokens=False)
        if len(question_ids) == 0:
            question_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        max_question_tokens = self.model_kwargs.get("max_question_tokens")
        if max_question_tokens is not None and len(question_ids) > int(max_question_tokens):
            question_ids = question_ids[-int(max_question_tokens):]

        token_ids = list(question_ids)
        question_end_idx = len(token_ids) - 1
        step_spans = []
        max_step_tokens = self.model_kwargs.get("max_step_tokens")
        max_total_tokens = self.model_kwargs.get("max_train_tokens")
        for idx, step in enumerate(step_list):
            prefix = self.thinking_separator if idx == 0 else ""
            suffix = "\n" if idx < len(step_list) - 1 else ""
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            step_ids = self.tokenizer.encode(self.steps_template.format(step), add_special_tokens=False)
            if max_step_tokens is not None and len(step_ids) > int(max_step_tokens):
                step_ids = step_ids[: int(max_step_tokens)]
            suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
            if len(step_ids) == 0:
                step_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            if max_total_tokens is not None:
                remaining = int(max_total_tokens) - len(token_ids) - len(prefix_ids) - len(suffix_ids)
                if remaining <= 0:
                    break
                step_ids = step_ids[:remaining]

            token_ids.extend(prefix_ids)
            start = len(token_ids)
            token_ids.extend(step_ids)
            end = len(token_ids)
            token_ids.extend(suffix_ids)
            step_spans.append((start, end))

        input_ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        return input_ids, question_end_idx, step_spans

    def _question_only_latents(self, questions: Sequence[str]) -> Dict[str, torch.Tensor]:
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
        latent_inputs_embeds = (base_latent + query_scale * latent_queries).to(question_inputs_embeds.dtype)
        if anchor_gate is not None and self.readcot_config.get("anchor_gate_apply_to_latents", True):
            gate_scale = float(self.readcot_config.get("anchor_gate_scale", 0.5))
            gate_values = anchor_gate[:, :n_latents].unsqueeze(-1).to(latent_inputs_embeds.dtype)
            latent_inputs_embeds = latent_inputs_embeds * (1.0 + gate_scale * gate_values)

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
            "anchor_gate": anchor_gate,
        }
