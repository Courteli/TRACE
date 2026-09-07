import torch

from .colar import LitCoLaR
from ..utils.utils import get_position_ids_from_attention_mask


class LitCoLaROneShotFair(LitCoLaR):
    """Inference-only one-shot CoLaR ablation.

    This class keeps the original CoLaR parameters and checkpoint format. It
    only replaces autoregressive latent rollout at inference with a single
    parallel latent draw from the original latent policy.
    """

    @torch.no_grad()
    def latent_generate(
        self,
        questions,
        rl_mode=False,
        return_latent_hidden_states=False,
    ):
        latent_generation_config = self.model_kwargs.latent_generation_config
        answer_generation_config = self.model_kwargs.answer_generation_config
        latent_temperature = latent_generation_config.get("latent_temperature", 1.0)
        num_latents = int(latent_generation_config.get("one_shot_num_latents", 16))

        batch_size = len(questions)

        speed = latent_generation_config["compression_factor"]
        suffix = self.speed_template.format(speed) + self.thinking_separator
        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=suffix,
        )
        question_position_ids = get_position_ids_from_attention_mask(question_attention_mask)
        question_embeds = self.embedding(question_input_ids)
        question_outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=question_attention_mask,
            position_ids=question_position_ids,
            output_hidden_states=True,
        )

        # Fair one-shot replacement: use the same CoLaR latent_policy, but do
        # not call Qwen between latent slots. No extra head or answer loss is
        # introduced, so the original CoLaR checkpoint can be evaluated as-is.
        question_state = question_outputs.hidden_states[-1][:, -1:, :]
        slot_states = question_state.expand(batch_size, num_latents, question_state.shape[-1]).contiguous()
        distributions = self.latent_policy.forward(slot_states, temperature=latent_temperature)
        latent_inputs_embeds = distributions.rsample() * self.embeds_std
        latent_attention_mask = torch.ones(
            size=(batch_size, num_latents),
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
