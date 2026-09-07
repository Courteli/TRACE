import random
from typing import List

import torch
import torch.nn.functional as F

from .colar import LitCoLaR
from ..utils.utils import get_position_ids_from_attention_mask


class LitCoLaROneShotCompiler(LitCoLaR):
    """Train CoLaR with a one-shot latent protocol.

    This is a fair trained one-shot counterpart of CoLaR-r5.  It keeps the
    original CoLaR backbone, LoRA setup, latent policy, data pipeline, and
    evaluation code.  The only protocol change is that all compact latent
    states are generated in parallel from the question representation, and the
    SFT target is pooled into the same fixed one-shot slots.
    """

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)
        one_shot_config = model_kwargs.get("one_shot_config", {})
        self.num_one_shot_latents = int(one_shot_config.get("num_latents", 16))
        self.answer_loss_weight = float(one_shot_config.get("answer_loss_weight", 1.0))
        self.target_loss_weight = float(one_shot_config.get("target_loss_weight", 1.0))
        self.target_on_mean = bool(one_shot_config.get("target_on_mean", False))
        self.use_mean_at_eval = bool(one_shot_config.get("use_mean_at_eval", False))
        self.teacher_forced_answer = bool(one_shot_config.get("teacher_forced_answer", True))

        hidden_size = self.llm.config.hidden_size
        self.slot_embeddings = torch.nn.Parameter(torch.zeros(1, self.num_one_shot_latents, hidden_size))
        torch.nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)

    def _sample_compression_factor(self):
        max_compression_factor = self.model_kwargs.latent_cot_config.max_compression_factor
        if isinstance(max_compression_factor, int):
            return random.randint(1, max_compression_factor)
        if isinstance(max_compression_factor, str):
            factors = max_compression_factor.strip(",").split(",")
            return int(random.choice(factors))
        raise ValueError("max_compression_factor should be int or str")

    def _question_state(self, questions: List[str], speed):
        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format(speed) + self.thinking_separator,
        )
        question_position_ids = get_position_ids_from_attention_mask(question_attention_mask)
        question_inputs_embeds = self.embedding(question_input_ids)
        question_outputs = self.llm.forward(
            inputs_embeds=question_inputs_embeds,
            attention_mask=question_attention_mask,
            position_ids=question_position_ids,
            output_hidden_states=True,
        )
        question_state = question_outputs.hidden_states[-1][:, -1:, :]
        return question_input_ids, question_attention_mask, question_inputs_embeds, question_state

    def _one_shot_latents(self, question_state, temperature=1.0, use_mean=False):
        batch_size = question_state.shape[0]
        slot_states = question_state + self.slot_embeddings.expand(batch_size, -1, -1)
        distributions = self.latent_policy.forward(slot_states, temperature=temperature)
        latent_raw = distributions.mean if use_mean else distributions.rsample()
        latent_inputs_embeds = latent_raw.to(dtype=question_state.dtype) * self.embeds_std
        latent_attention_mask = torch.ones(
            size=(batch_size, self.num_one_shot_latents),
            device=self.device,
            dtype=torch.long,
        )
        return distributions, latent_raw, latent_inputs_embeds, latent_attention_mask

    def forward(self, batch):
        latent_cot_config = self.model_kwargs.latent_cot_config
        r = self._sample_compression_factor()

        questions = batch["question"]
        steps = batch["steps"]
        answers = batch["answer"]
        batch_size = len(questions)

        auto_prob = latent_cot_config.get("replace_r_with_auto_prob", 0)
        speed = "auto" if random.random() < auto_prob else r
        question_input_ids, question_attention_mask, question_embeds, question_state = self._question_state(
            questions=questions,
            speed=speed,
        )

        distributions, latent_raw, latent_inputs_embeds, latent_attention_mask = self._one_shot_latents(
            question_state=question_state,
            temperature=self.model_kwargs.latent_generation_config.get("latent_temperature", 1.0),
            use_mean=False,
        )

        target_embeds, target_mask = self.make_one_shot_targets(
            steps=steps,
            r=r,
            num_latents=self.num_one_shot_latents,
        )
        pred_for_target = distributions.mean if self.target_on_mean else latent_raw
        target_loss = F.mse_loss(
            pred_for_target,
            target_embeds.detach() / self.embeds_std,
            reduction="none",
        ).mean(dim=-1)
        target_loss = (target_loss * target_mask).sum() / target_mask.sum().clamp_min(1.0)

        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answers,
            padding_side="right",
            part="answer",
            prefix=self.thinking_separator,
            suffix=self.tokenizer.eos_token,
        )
        answer_inputs_embeds = self.embedding(answer_input_ids)

        if self.teacher_forced_answer:
            ce_latent_inputs_embeds = target_embeds.detach().to(dtype=question_embeds.dtype)
        else:
            ce_latent_inputs_embeds = latent_inputs_embeds

        all_inputs_embeds = torch.cat([question_embeds, ce_latent_inputs_embeds, answer_inputs_embeds], dim=1)
        all_attention_mask = torch.cat([question_attention_mask, latent_attention_mask, answer_attention_mask], dim=1)
        all_position_ids = get_position_ids_from_attention_mask(all_attention_mask)

        ignore_question = torch.full_like(question_input_ids, -100)
        ignore_latents = torch.full(
            size=(batch_size, self.num_one_shot_latents),
            device=self.device,
            dtype=answer_input_ids.dtype,
            fill_value=-100,
        )
        answer_labels = answer_input_ids.clone()
        answer_labels[answer_labels == self.tokenizer.pad_token_id] = -100
        labels = torch.cat([ignore_question, ignore_latents, answer_labels], dim=1)

        outputs = self.llm.forward(
            inputs_embeds=all_inputs_embeds,
            attention_mask=all_attention_mask,
            position_ids=all_position_ids,
            labels=labels,
        )
        answer_loss = outputs.loss
        total_loss = self.answer_loss_weight * answer_loss + self.target_loss_weight * target_loss

        zero = torch.zeros((), device=self.device, dtype=total_loss.dtype)
        return {
            "total_loss": total_loss,
            "ce_loss": answer_loss,
            "pred_embed_forward_loss": zero,
            "embed_modeling_loss": target_loss,
            "entropy": distributions.entropy().mean(),
        }

    def make_one_shot_targets(self, steps: List[str], r: int, num_latents: int):
        steps_input_ids, steps_attention_mask = self.prepare_inputs(
            steps,
            padding_side="left",
            part="steps",
        )
        if r == 1:
            steps_inputs_embeds = self.embedding(steps_input_ids)
        else:
            batch_size = len(steps)
            steps_pad_lengths = -(steps_attention_mask - 1).sum(dim=-1)
            n_extra_left_pad_length = r - 1 - steps_pad_lengths % r
            steps_length_left_padded = steps_attention_mask.shape[1] + n_extra_left_pad_length.max()
            min_right_pad_length = r - steps_length_left_padded % r

            all_steps_input_ids = []
            all_steps_attention_mask = []
            for batch_idx, left_length in enumerate(n_extra_left_pad_length):
                right_length = min_right_pad_length + (r - 1 - left_length)
                if right_length == r:
                    left_length += r
                    right_length = 0

                sample_ids = steps_input_ids[batch_idx]
                sample_mask = steps_attention_mask[batch_idx]
                if left_length > 0:
                    sample_ids = torch.cat(
                        [
                            torch.full(
                                size=(left_length,),
                                fill_value=self.tokenizer.pad_token_id,
                                device=sample_ids.device,
                                dtype=sample_ids.dtype,
                            ),
                            sample_ids,
                        ],
                    )
                    sample_mask = torch.cat(
                        [
                            torch.zeros(left_length, device=sample_mask.device, dtype=sample_mask.dtype),
                            sample_mask,
                        ],
                    )
                if right_length > 0:
                    sample_ids = torch.cat(
                        [
                            sample_ids,
                            torch.full(
                                size=(right_length,),
                                fill_value=self.tokenizer.pad_token_id,
                                device=sample_ids.device,
                                dtype=sample_ids.dtype,
                            ),
                        ],
                    )
                    sample_mask = torch.cat(
                        [
                            sample_mask,
                            torch.zeros(right_length, device=sample_mask.device, dtype=sample_mask.dtype),
                        ],
                    )
                all_steps_input_ids.append(sample_ids)
                all_steps_attention_mask.append(sample_mask)

            padded_steps_input_ids = torch.stack(all_steps_input_ids, dim=0)
            padded_steps_attention_mask = torch.stack(all_steps_attention_mask, dim=0)
            padded_steps_inputs_embeds = self.embedding(padded_steps_input_ids)
            padded_steps_inputs_embeds = padded_steps_inputs_embeds * padded_steps_attention_mask.unsqueeze(-1)

            padded_steps_length = padded_steps_inputs_embeds.shape[1]
            compressed_steps_length = padded_steps_length // r
            steps_inputs_embeds = padded_steps_inputs_embeds.reshape(
                batch_size,
                compressed_steps_length,
                r,
                padded_steps_inputs_embeds.shape[-1],
            ).sum(dim=2)
            steps_attention_mask = padded_steps_attention_mask.reshape(batch_size, compressed_steps_length, r).sum(
                dim=2,
            )
            if self.model_kwargs.latent_cot_config.get("sqrt_mean", False):
                steps_attention_mask = steps_attention_mask.sqrt()
            steps_inputs_embeds = steps_inputs_embeds / (steps_attention_mask.unsqueeze(-1) + 1e-5)
            steps_attention_mask = (steps_attention_mask != 0).long()

        return self.pool_sequence_to_slots(steps_inputs_embeds, steps_attention_mask, num_latents)

    def pool_sequence_to_slots(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, num_latents: int):
        batch_size, _, hidden_size = inputs_embeds.shape
        pooled = torch.zeros(
            size=(batch_size, num_latents, hidden_size),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        pooled_mask = torch.ones(size=(batch_size, num_latents), device=inputs_embeds.device, dtype=torch.float32)
        for batch_idx in range(batch_size):
            valid_embeds = inputs_embeds[batch_idx][attention_mask[batch_idx].bool()]
            valid_length = valid_embeds.shape[0]
            if valid_length == 0:
                pooled_mask[batch_idx] = 0
                continue
            for slot_idx in range(num_latents):
                start = int(slot_idx * valid_length / num_latents)
                end = int((slot_idx + 1) * valid_length / num_latents)
                if end <= start:
                    end = min(start + 1, valid_length)
                pooled[batch_idx, slot_idx] = valid_embeds[start:end].mean(dim=0)
        return pooled, pooled_mask

    @torch.no_grad()
    def latent_generate(
        self,
        questions: List[str],
        rl_mode=False,
        return_latent_hidden_states=False,
    ):
        latent_generation_config = self.model_kwargs.latent_generation_config
        answer_generation_config = self.model_kwargs.answer_generation_config
        speed = latent_generation_config["compression_factor"]

        question_input_ids, question_attention_mask, question_embeds, question_state = self._question_state(
            questions=questions,
            speed=speed,
        )
        _, _, latent_inputs_embeds, latent_attention_mask = self._one_shot_latents(
            question_state=question_state,
            temperature=latent_generation_config.get("latent_temperature", 1.0),
            use_mean=self.use_mean_at_eval,
        )
        batch_size = len(questions)
        n_latent_forward = latent_attention_mask.sum(dim=1, keepdim=True)

        end_of_thinking_ids = torch.full(
            size=(batch_size, 1),
            fill_value=self.thinking_separator_id,
            device=self.device,
            dtype=torch.long,
        )
        end_of_thinking_embeds = self.embedding(end_of_thinking_ids)

        all_inputs_embeds = torch.cat([question_embeds, latent_inputs_embeds, end_of_thinking_embeds], dim=1)
        all_attention_mask = torch.cat(
            [
                question_attention_mask,
                latent_attention_mask,
                torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long),
            ],
            dim=1,
        )
        pred_ids = self.llm.generate(
            inputs_embeds=all_inputs_embeds,
            attention_mask=all_attention_mask,
            **answer_generation_config,
        )

        if rl_mode:
            return (
                question_input_ids,
                question_attention_mask,
                latent_inputs_embeds,
                latent_attention_mask,
                torch.cat([end_of_thinking_ids, pred_ids], dim=1),
            )
        if return_latent_hidden_states:
            return pred_ids, n_latent_forward, []
        return pred_ids, n_latent_forward
