import math
import re
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .colar import LitCoLaR
from ..utils.utils import get_position_ids_from_attention_mask


class LitTRACETrajectoryCoLaR(LitCoLaR):
    """TRACE with hidden-space trajectory supervision.

    This is the clean TRACE core:

    1. extract a teacher hidden trajectory from explicit CoT;
    2. generate fixed question-only latent steps;
    3. align both node states and transition directions;
    4. train the answer readout from the generated latent path.

    Outcome contrast RL from `trace_colar.py` can still be used later as a
    refinement stage, but it is not part of this core supervised objective.
    """

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)
        self.trace_trajectory_config = model_kwargs.get("trace_trajectory_config", {})

    def trace_steps(self) -> int:
        return int(self.trace_trajectory_config.get("trace_steps", 8))

    def split_reasoning_steps(self, steps: str) -> List[str]:
        text = " ".join(str(steps).strip().split())
        if not text:
            return [""]
        pieces = re.split(r"(?<=[.!?])\s+", text)
        pieces = [piece.strip() for piece in pieces if piece.strip()]
        if len(pieces) <= 1:
            pieces = re.split(r"\s+(?:So|Then|Thus|Therefore|This means),?\s+", text)
            pieces = [piece.strip() for piece in pieces if piece.strip()]
        return pieces or [text]

    def token_ids_for_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def teacher_question_prefix(self, question: str, trace_steps: int) -> str:
        speed = self.trace_trajectory_config.get("teacher_speed", trace_steps)
        return self.question_template.format(question) + self.speed_template.format(speed) + self.thinking_separator

    def build_teacher_ids_and_positions(
        self, question: str, steps: str, answer: str, trace_steps: int
    ) -> Tuple[List[int], List[int]]:
        """Build one teacher sequence and return positions for h0...hK.

        h0 is the hidden state after the question prefix. h1...hK are hidden
        checkpoints along the explicit CoT. When enough sentence-like steps are
        available, checkpoints are sentence endpoints. Otherwise they are
        evenly spaced token-fraction checkpoints over the CoT text.
        """

        question_ids = self.token_ids_for_text(self.teacher_question_prefix(question, trace_steps))
        answer_ids = self.token_ids_for_text(
            self.thinking_separator + self.answer_template.format(str(answer)) + self.tokenizer.eos_token
        )
        if not question_ids:
            question_ids = [self.tokenizer.pad_token_id]

        h0_position = len(question_ids) - 1
        boundary_mode = self.trace_trajectory_config.get("teacher_boundary_mode", "sentence")
        step_pieces = self.split_reasoning_steps(steps)
        use_sentence_boundaries = boundary_mode == "sentence" and len(step_pieces) >= trace_steps

        ids = list(question_ids)
        h_positions = [h0_position]

        if use_sentence_boundaries:
            selected = {
                int(round(x))
                for x in torch.linspace(0, len(step_pieces) - 1, trace_steps).tolist()
            }
            # Make sure repeated rounding cannot reduce the number of selected endpoints.
            if len(selected) < trace_steps:
                selected = set(torch.linspace(0, len(step_pieces) - 1, trace_steps).long().tolist())
            endpoint_positions = []
            for step_idx, piece in enumerate(step_pieces):
                piece_ids = self.token_ids_for_text((" " if ids else "") + piece)
                if not piece_ids:
                    continue
                ids.extend(piece_ids)
                if step_idx in selected:
                    endpoint_positions.append(len(ids) - 1)
            if len(endpoint_positions) >= trace_steps:
                h_positions.extend(endpoint_positions[:trace_steps])
            else:
                use_sentence_boundaries = False

        if not use_sentence_boundaries:
            cot_ids = self.token_ids_for_text(" " + " ".join(str(steps).strip().split()))
            if not cot_ids:
                cot_ids = [self.tokenizer.pad_token_id]
            ids = list(question_ids) + cot_ids
            cot_start = len(question_ids)
            cot_len = len(cot_ids)
            for step_idx in range(trace_steps):
                frac_pos = math.ceil((step_idx + 1) * cot_len / trace_steps) - 1
                frac_pos = max(0, min(cot_len - 1, frac_pos))
                h_positions.append(cot_start + frac_pos)

        ids.extend(answer_ids)
        return ids, h_positions[: trace_steps + 1]

    @torch.no_grad()
    def extract_teacher_trajectory(self, questions, steps, answers, trace_steps: int) -> torch.Tensor:
        built = [
            self.build_teacher_ids_and_positions(q, s, a, trace_steps)
            for q, s, a in zip(questions, steps, answers)
        ]
        max_len = max(len(ids) for ids, _ in built)
        batch_size = len(built)
        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_len), device=self.device, dtype=torch.long)
        positions = torch.zeros((batch_size, trace_steps + 1), device=self.device, dtype=torch.long)
        for row_idx, (ids, h_positions) in enumerate(built):
            length = len(ids)
            input_ids[row_idx, :length] = torch.tensor(ids, device=self.device, dtype=torch.long)
            attention_mask[row_idx, :length] = 1
            positions[row_idx] = torch.tensor(h_positions, device=self.device, dtype=torch.long)

        outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=get_position_ids_from_attention_mask(attention_mask),
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        teacher_path = hidden[batch_indices, positions]
        return teacher_path.detach().float()

    def generate_trace_latent_inputs(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        trace_steps: int,
        use_mean: bool = False,
    ) -> torch.Tensor:
        latent_temperature = float(self.model_kwargs.latent_generation_config.get("latent_temperature", 1.0))
        question_embeds = self.embedding(question_input_ids)
        attention_mask = question_attention_mask
        outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=attention_mask,
            position_ids=get_position_ids_from_attention_mask(attention_mask),
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        current_position_ids = get_position_ids_from_attention_mask(attention_mask)[:, -1:]
        latent_inputs = []
        for _ in range(trace_steps):
            distributions = self.latent_policy.forward(
                outputs.hidden_states[-1][:, -1:, :],
                temperature=latent_temperature,
            )
            if use_mean:
                current_inputs_embeds = distributions.mean * self.embeds_std
            else:
                current_inputs_embeds = distributions.rsample() * self.embeds_std
            latent_inputs.append(current_inputs_embeds)
            step_mask = torch.ones(
                (question_input_ids.shape[0], 1),
                device=self.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([attention_mask, step_mask], dim=1)
            current_position_ids = current_position_ids + 1
            outputs = self.llm.forward(
                inputs_embeds=current_inputs_embeds,
                attention_mask=attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
        return torch.cat(latent_inputs, dim=1)

    def trace_question_inputs(self, questions, trace_steps: int):
        speed = self.trace_trajectory_config.get("student_speed", trace_steps)
        suffix = self.speed_template.format(speed) + self.thinking_separator
        return self.prepare_inputs(questions, padding_side="left", part="question", suffix=suffix)

    def forward(self, batch):
        cfg = self.trace_trajectory_config
        trace_steps = self.trace_steps()
        questions = batch["question"]
        steps = batch["steps"]
        answers = batch["answer"]

        question_input_ids, question_attention_mask = self.trace_question_inputs(questions, trace_steps)
        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answers,
            padding_side="right",
            part="answer",
            prefix=self.thinking_separator,
            suffix=self.tokenizer.eos_token,
        )
        use_mean_train = bool(cfg.get("use_mean_latents_train", False))
        latent_inputs = self.generate_trace_latent_inputs(
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            trace_steps=trace_steps,
            use_mean=use_mean_train,
        )

        question_embeds = self.embedding(question_input_ids)
        answer_embeds = self.embedding(answer_input_ids)
        batch_size = question_input_ids.shape[0]
        latent_attention_mask = torch.ones(
            (batch_size, trace_steps),
            device=self.device,
            dtype=question_attention_mask.dtype,
        )
        inputs_embeds = torch.cat([question_embeds, latent_inputs, answer_embeds], dim=1)
        attention_mask = torch.cat([question_attention_mask, latent_attention_mask, answer_attention_mask], dim=1)
        labels = torch.cat(
            [
                torch.full_like(question_input_ids, -100),
                torch.full((batch_size, trace_steps), -100, device=self.device, dtype=torch.long),
                answer_input_ids,
            ],
            dim=1,
        )
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.llm.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=get_position_ids_from_attention_mask(attention_mask),
            labels=labels,
            output_hidden_states=True,
        )
        answer_loss = outputs.loss
        hidden = outputs.hidden_states[-1].float()
        question_length = question_input_ids.shape[1]
        student_h0 = hidden[:, question_length - 1 : question_length, :]
        student_steps = hidden[:, question_length : question_length + trace_steps, :]
        student_path = torch.cat([student_h0, student_steps], dim=1)

        if "trace_teacher_path" in batch:
            teacher_path = batch["trace_teacher_path"].to(device=self.device, dtype=torch.float32)
            teacher_path = teacher_path[:, : trace_steps + 1, :]
        else:
            teacher_path = self.extract_teacher_trajectory(
                questions=questions,
                steps=steps,
                answers=answers,
                trace_steps=trace_steps,
            )

        norm_student_steps = F.normalize(student_steps, dim=-1)
        norm_teacher_steps = F.normalize(teacher_path[:, 1:, :], dim=-1)
        state_cos_per_step = F.cosine_similarity(norm_student_steps, norm_teacher_steps, dim=-1)
        state_loss = (1.0 - state_cos_per_step).mean()

        student_delta = student_path[:, 1:, :] - student_path[:, :-1, :]
        teacher_delta = teacher_path[:, 1:, :] - teacher_path[:, :-1, :]
        transition_cos = F.cosine_similarity(student_delta, teacher_delta, dim=-1)
        transition_loss = (1.0 - transition_cos).mean()

        total_loss = (
            float(cfg.get("answer_weight", 1.0)) * answer_loss
            + float(cfg.get("state_weight", 1.0)) * state_loss
            + float(cfg.get("transition_weight", 1.0)) * transition_loss
        )

        role_loss = torch.tensor(0.0, device=self.device)
        role_weight = float(cfg.get("role_weight", 0.0))
        if role_weight > 0 and trace_steps > 1:
            step_norm = F.normalize(student_steps, dim=-1)
            step_sim = torch.matmul(step_norm, step_norm.transpose(1, 2))
            off_diag = ~torch.eye(trace_steps, dtype=torch.bool, device=self.device).unsqueeze(0)
            role_margin = float(cfg.get("role_margin", 0.1))
            role_loss = F.relu(step_sim[off_diag.expand_as(step_sim)] - (1.0 - role_margin)).mean()
            total_loss = total_loss + role_weight * role_loss

        with torch.no_grad():
            state_cos = state_cos_per_step.mean()
            student_step_cos = F.cosine_similarity(student_steps[:, 1:, :], student_steps[:, :-1, :], dim=-1).mean()
            student_delta_norm = student_delta.norm(dim=-1).mean()
            teacher_delta_norm = teacher_delta.norm(dim=-1).mean()

        return {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "state_loss": state_loss,
            "transition_loss": transition_loss,
            "role_loss": role_loss,
            "trace/state_cos": state_cos,
            "trace/transition_cos": transition_cos.mean(),
            "trace/student_step_cos": student_step_cos,
            "trace/student_delta_norm": student_delta_norm,
            "trace/teacher_delta_norm": teacher_delta_norm,
            "trace_steps": torch.tensor(float(trace_steps), device=self.device),
        }

    @torch.no_grad()
    def extract_student_trajectory(self, questions: List[str], trace_steps: int = None):
        if trace_steps is None:
            trace_steps = self.trace_steps()
        question_input_ids, question_attention_mask = self.trace_question_inputs(questions, trace_steps)
        latent_inputs = self.generate_trace_latent_inputs(
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            trace_steps=trace_steps,
            use_mean=bool(self.trace_trajectory_config.get("use_mean_latents_eval", True)),
        )
        question_embeds = self.embedding(question_input_ids)
        inputs_embeds = torch.cat([question_embeds, latent_inputs], dim=1)
        attention_mask = torch.cat(
            [
                question_attention_mask,
                torch.ones((len(questions), trace_steps), device=self.device, dtype=question_attention_mask.dtype),
            ],
            dim=1,
        )
        outputs = self.llm.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=get_position_ids_from_attention_mask(attention_mask),
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1].float()
        question_length = question_input_ids.shape[1]
        student_h0 = hidden[:, question_length - 1 : question_length, :]
        student_steps = hidden[:, question_length : question_length + trace_steps, :]
        return torch.cat([student_h0, student_steps], dim=1)

    @torch.no_grad()
    def trace_latent_generate(self, questions: List[str]):
        trace_steps = self.trace_steps()
        question_input_ids, question_attention_mask = self.trace_question_inputs(questions, trace_steps)
        latent_inputs = self.generate_trace_latent_inputs(
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            trace_steps=trace_steps,
            use_mean=bool(self.trace_trajectory_config.get("use_mean_latents_eval", True)),
        )
        question_embeds = self.embedding(question_input_ids)
        end_ids = torch.full(
            (len(questions), 1),
            fill_value=self.thinking_separator_id,
            device=self.device,
            dtype=torch.long,
        )
        end_embeds = self.embedding(end_ids)
        inputs_embeds = torch.cat([question_embeds, latent_inputs, end_embeds], dim=1)
        attention_mask = torch.cat(
            [
                question_attention_mask,
                torch.ones((len(questions), trace_steps + 1), device=self.device, dtype=question_attention_mask.dtype),
            ],
            dim=1,
        )
        pred_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **self.model_kwargs.answer_generation_config,
        )
        n_latent_forward = torch.full((len(questions), 1), trace_steps, device=self.device, dtype=torch.long)
        return pred_ids, n_latent_forward

    @torch.no_grad()
    def latent_generate(self, questions, rl_mode=False, return_latent_hidden_states=False):
        trace_steps = self.trace_steps()
        question_input_ids, question_attention_mask = self.trace_question_inputs(questions, trace_steps)
        latent_inputs = self.generate_trace_latent_inputs(
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            trace_steps=trace_steps,
            use_mean=bool(self.trace_trajectory_config.get("use_mean_latents_eval", True)),
        )
        question_embeds = self.embedding(question_input_ids)
        end_ids = torch.full(
            (len(questions), 1),
            fill_value=self.thinking_separator_id,
            device=self.device,
            dtype=torch.long,
        )
        end_embeds = self.embedding(end_ids)
        inputs_embeds = torch.cat([question_embeds, latent_inputs, end_embeds], dim=1)
        attention_mask = torch.cat(
            [
                question_attention_mask,
                torch.ones((len(questions), trace_steps + 1), device=self.device, dtype=question_attention_mask.dtype),
            ],
            dim=1,
        )
        pred_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **self.model_kwargs.answer_generation_config,
        )
        latent_attention_mask = torch.ones((len(questions), trace_steps), device=self.device, dtype=torch.long)
        if rl_mode:
            return (
                question_input_ids,
                question_attention_mask,
                latent_inputs,
                latent_attention_mask,
                torch.cat([end_ids, pred_ids], dim=1),
            )
        n_latent_forward = torch.full((len(questions), 1), trace_steps, device=self.device, dtype=torch.long)
        return pred_ids, n_latent_forward
