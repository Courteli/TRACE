import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .colar import LitCoLaR
from ..utils.utils import get_position_ids_from_attention_mask


class OneShotLatentHead(nn.Module):
    def __init__(self, hidden_size: int, num_latents: int, intermediate_size: int | None = None):
        super().__init__()
        self.num_latents = int(num_latents)
        intermediate_size = int(intermediate_size or hidden_size)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, self.num_latents * hidden_size),
        )
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, question_state: torch.Tensor) -> torch.Tensor:
        batch_size, hidden_size = question_state.shape
        return self.net(question_state).view(batch_size, self.num_latents, hidden_size)


class LitCoLaROneLatent(LitCoLaR):
    """CoLaR-r5 code path with one-shot latent generation.

    This class keeps CoLaR's original data module, Lightning training loop,
    LoRA setup, checkpointing, and evaluation. It replaces CoLaR's
    autoregressive latent rollout with a one-shot latent writer trained against
    the same compressed-CoT embedding targets plus answer supervision.
    """

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)
        one_latent_config = model_kwargs.get("one_latent_config", {})
        num_latents = int(
            one_latent_config.get(
                "num_latents",
                model_kwargs.latent_generation_config.get("compression_factor", 5),
            )
        )
        self.one_latent_head = OneShotLatentHead(
            hidden_size=self.llm.config.hidden_size,
            num_latents=num_latents,
            intermediate_size=one_latent_config.get("intermediate_size", self.llm.config.hidden_size),
        )
        self.one_latent_loss_weight = float(one_latent_config.get("answer_loss_weight", 1.0))
        self.one_latent_distill_weight = float(one_latent_config.get("distill_loss_weight", 1.0))
        for parameter in self.latent_policy.parameters():
            parameter.requires_grad_(False)

    def forward(self, batch):
        latent_cot_config = self.model_kwargs.latent_cot_config
        max_compression_factor = latent_cot_config.max_compression_factor
        if isinstance(max_compression_factor, int):
            r = random.randint(1, max_compression_factor)
        elif isinstance(max_compression_factor, str):
            factors = max_compression_factor.strip(",").split(",")
            r = int(random.choice(factors))
        else:
            raise ValueError("max_compression_factor should be int or str")

        questions = batch["question"]
        steps = batch["steps"]
        answers = batch["answer"]
        batch_size = len(questions)

        auto_prob = latent_cot_config.get("replace_r_with_auto_prob", 0)
        speed = "auto" if random.random() < auto_prob else r
        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format(speed) + self.thinking_separator,
        )
        question_position_ids = get_position_ids_from_attention_mask(question_attention_mask)
        question_embeds = self.embedding(question_input_ids)
        question_outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=question_attention_mask,
            position_ids=question_position_ids,
            output_hidden_states=True,
        )

        latent_raw = self.one_latent_head(question_outputs.hidden_states[-1][:, -1, :])
        latent_inputs_embeds = latent_raw.to(dtype=question_embeds.dtype) * self.embeds_std
        latent_attention_mask = torch.ones(
            size=(batch_size, latent_inputs_embeds.shape[1]),
            device=self.device,
            dtype=torch.long,
        )

        target_embeds, target_mask = self.make_one_shot_targets(
            steps=steps,
            r=r,
            num_latents=latent_inputs_embeds.shape[1],
        )
        target_loss = F.mse_loss(latent_raw, target_embeds.detach() / self.embeds_std, reduction="none").mean(dim=-1)
        target_loss = (target_loss * target_mask).sum() / target_mask.sum().clamp_min(1)

        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answers,
            padding_side="right",
            part="answer",
            prefix=self.thinking_separator,
            suffix=self.tokenizer.eos_token,
        )
        answer_inputs_embeds = self.embedding(answer_input_ids)

        all_inputs_embeds = torch.cat([question_embeds, latent_inputs_embeds, answer_inputs_embeds], dim=1)
        all_attention_mask = torch.cat([question_attention_mask, latent_attention_mask, answer_attention_mask], dim=1)
        all_position_ids = get_position_ids_from_attention_mask(all_attention_mask)

        ignore_question = torch.full_like(question_input_ids, -100)
        ignore_latent = torch.full(
            size=(batch_size, latent_inputs_embeds.shape[1]),
            device=self.device,
            dtype=answer_input_ids.dtype,
            fill_value=-100,
        )
        answer_labels = answer_input_ids.clone()
        answer_labels[answer_labels == self.tokenizer.pad_token_id] = -100
        labels = torch.cat([ignore_question, ignore_latent, answer_labels], dim=1)

        outputs = self.llm.forward(
            inputs_embeds=all_inputs_embeds,
            attention_mask=all_attention_mask,
            position_ids=all_position_ids,
            labels=labels,
        )
        answer_loss = outputs.loss
        total_loss = self.one_latent_distill_weight * target_loss + self.one_latent_loss_weight * answer_loss

        zero = torch.zeros((), device=self.device, dtype=total_loss.dtype)
        return {
            "total_loss": total_loss,
            "ce_loss": answer_loss,
            "pred_embed_forward_loss": zero,
            "embed_modeling_loss": target_loss,
            "entropy": zero,
            "one_latent_answer_loss": answer_loss,
        }

    def make_one_shot_targets(self, steps: List[str], r: int, num_latents: int):
        steps_input_ids, steps_attention_mask = self.prepare_inputs(
            steps,
            padding_side="left",
            part="steps",
            prefix=self.thinking_separator,
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
            for b, left_length in enumerate(n_extra_left_pad_length):
                right_length = min_right_pad_length + (r - 1 - left_length)
                if right_length == r:
                    left_length += r
                    right_length = 0
                sample_ids = steps_input_ids[b]
                sample_mask = steps_attention_mask[b]
                if left_length > 0:
                    sample_ids = torch.cat(
                        [
                            torch.ones(left_length, device=sample_ids.device, dtype=sample_ids.dtype)
                            * self.tokenizer.pad_token_id,
                            sample_ids,
                        ]
                    )
                    sample_mask = torch.cat(
                        [torch.zeros(left_length, device=sample_mask.device, dtype=sample_mask.dtype), sample_mask]
                    )
                if right_length > 0:
                    sample_ids = torch.cat(
                        [
                            sample_ids,
                            torch.ones(right_length, device=sample_ids.device, dtype=sample_ids.dtype)
                            * self.tokenizer.pad_token_id,
                        ]
                    )
                    sample_mask = torch.cat(
                        [sample_mask, torch.zeros(right_length, device=sample_mask.device, dtype=sample_mask.dtype)]
                    )
                all_steps_input_ids.append(sample_ids)
                all_steps_attention_mask.append(sample_mask)

            padded_steps_input_ids = torch.stack(all_steps_input_ids, dim=0)
            padded_steps_attention_mask = torch.stack(all_steps_attention_mask, dim=0)
            padded_steps_inputs_embeds = self.embedding(padded_steps_input_ids)
            padded_steps_inputs_embeds *= padded_steps_attention_mask.unsqueeze(-1)

            padded_steps_length = padded_steps_inputs_embeds.shape[1]
            compressed_steps_length = padded_steps_length // r
            steps_inputs_embeds = padded_steps_inputs_embeds.reshape(
                batch_size, compressed_steps_length, r, padded_steps_inputs_embeds.shape[-1]
            ).sum(dim=2)
            steps_attention_mask = padded_steps_attention_mask.reshape(batch_size, compressed_steps_length, r).sum(
                dim=2
            )
            if self.model_kwargs.latent_cot_config.get("sqrt_mean", False):
                steps_attention_mask = steps_attention_mask.sqrt()
            steps_inputs_embeds /= steps_attention_mask.unsqueeze(-1) + 1e-5
            steps_attention_mask = (steps_attention_mask != 0).long()

        return self.pool_sequence_to_slots(steps_inputs_embeds, steps_attention_mask, num_latents=num_latents)

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

    def one_latent_answer_loss(self, batch):
        return self.forward(batch)["one_latent_answer_loss"]

    @torch.no_grad()
    def latent_generate(
        self,
        questions: List[str],
        rl_mode=False,
        return_latent_hidden_states=False,
    ):
        latent_generation_config = self.model_kwargs.latent_generation_config
        answer_generation_config = self.model_kwargs.answer_generation_config
        batch_size = len(questions)

        speed = latent_generation_config["compression_factor"]
        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format(speed) + self.thinking_separator,
        )
        question_position_ids = get_position_ids_from_attention_mask(question_attention_mask)
        question_embeds = self.embedding(question_input_ids)
        question_outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=question_attention_mask,
            position_ids=question_position_ids,
            output_hidden_states=True,
        )

        question_state = question_outputs.hidden_states[-1][:, -1, :]
        latent_inputs_embeds = self.one_latent_head(question_state).to(dtype=question_embeds.dtype) * self.embeds_std
        latent_attention_mask = torch.ones(
            size=(batch_size, latent_inputs_embeds.shape[1]),
            device=self.device,
            dtype=torch.long,
        )
        n_latent_forward = latent_attention_mask.sum(dim=1, keepdim=True)

        end_of_thinking_ids = (
            torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long) * self.thinking_separator_id
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
