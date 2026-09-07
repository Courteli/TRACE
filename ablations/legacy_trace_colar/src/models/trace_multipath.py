import math
import random
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .colar import LitCoLaR
from ..modules import grpo


class LitTRACEMultiPathCoLaR(LitCoLaR):
    """TRACE proper: path-scaffolded latent training plus multi-path RL.

    Stage 1 keeps CoLaR's supervised compression interface, but adds a single
    path-scaffold objective over the predicted latent trajectory. Stage 2 uses
    the same trajectory representation on on-policy rollout groups: correct
    paths form several hidden-space modes, while wrong and hard-negative paths
    are pushed away from those modes without collapsing the latent steps.
    """

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)
        self.trace_config = model_kwargs.get("trace_multipath_config", model_kwargs.get("trace_config", {}))
        self._last_trace_metrics: Dict[str, torch.Tensor] = {}
        self._skip_next_epoch_filter = False
        self._trace_save_frozen_lora_parameter_names: List[str] = []
        if self.model_kwargs.do_rl and self.trace_config.get("freeze_llm_stage2", False):
            self.freeze_llm_for_stage2()

    def init_rl(self):
        self.grpo_loss = grpo.GRPOLoss(rl_config=self.model_kwargs.rl_config)
        self.replay_buffer = grpo.ReplayBuffer()
        self.automatic_optimization = False

    def freeze_llm_for_stage2(self):
        """Keep the answer decoder fixed while Stage 2 updates the latent policy."""
        for parameter in self.llm.parameters():
            parameter.requires_grad_(False)
        for parameter in self.latent_policy.parameters():
            parameter.requires_grad_(True)
        self._trace_save_frozen_lora_parameter_names = [
            name for name, _ in self.named_parameters() if name.startswith("llm.") and "lora_" in name
        ]

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        if not self._trace_save_frozen_lora_parameter_names:
            return
        state_dict = checkpoint["state_dict"]
        full_state_dict = self.state_dict()
        for name in self._trace_save_frozen_lora_parameter_names:
            if name in full_state_dict and name not in state_dict:
                state_dict[name] = full_state_dict[name]
        checkpoint["state_dict"] = state_dict

    def on_fit_start(self):
        output = super().on_fit_start()
        if self.model_kwargs.do_rl:
            self._skip_next_epoch_filter = True
        return output

    def on_train_epoch_start(self):
        if self.model_kwargs.do_rl:
            if self._skip_next_epoch_filter:
                self._skip_next_epoch_filter = False
            else:
                self.limit_rl_train_epoch_length()
            return super(LitCoLaR, self).on_train_epoch_start()
        return super().on_train_epoch_start()

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def _masked_cosine_distance(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        distance = 1.0 - F.cosine_similarity(pred, target, dim=-1)
        distance = torch.nan_to_num(distance, nan=0.0, posinf=2.0, neginf=0.0)
        return self._masked_mean(distance, mask)

    def _prefix_path(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted = states * mask.unsqueeze(-1)
        cumsum = weighted.cumsum(dim=1)
        denom = mask.cumsum(dim=1).clamp_min(1.0).unsqueeze(-1)
        return cumsum / denom

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
            valid_indices = torch.nonzero(mask[batch_idx] > 0, as_tuple=False).flatten()
            positions = torch.round((length - 1) * fractions).long().clamp(min=0, max=length - 1)
            anchor_mask[batch_idx, valid_indices[positions.unique()]] = 1.0
        return anchor_mask * mask

    def _split_cot_steps(self, cot: str) -> List[str]:
        parts = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", str(cot)) if part.strip()]
        return parts or [str(cot)]

    def _token_count(self, text: str) -> int:
        tokenized = self.tokenizer(str(text), add_special_tokens=False)
        return max(1, len(tokenized["input_ids"]))

    def _numeric_answer_value(self, text: str) -> Optional[float]:
        answer = self.extract_answer_from_output(str(text))
        matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(answer).replace(",", ""))
        if not matches:
            return None
        try:
            return float(matches[-1])
        except ValueError:
            return None

    def _cot_step_anchor_mask(
        self,
        cot_steps: List[str],
        mask: torch.Tensor,
        compression_factor: int,
        anchor_count: int,
        jitter: int = 0,
    ) -> torch.Tensor:
        anchor_mask = torch.zeros_like(mask)
        prefix_tokens = self._token_count(self.thinking_separator)
        fractions = torch.linspace(
            1.0 / max(anchor_count, 1),
            1.0,
            steps=max(anchor_count, 1),
            device=mask.device,
            dtype=torch.float32,
        )
        r = max(1, int(compression_factor))

        for batch_idx, cot in enumerate(cot_steps):
            valid_indices = torch.nonzero(mask[batch_idx] > 0, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            segments = self._split_cot_steps(cot)
            segment_lengths = [self._token_count(segment) for segment in segments]
            cumulative = []
            total = prefix_tokens
            for length in segment_lengths:
                total += length
                cumulative.append(total)
            if not cumulative:
                continue

            step_count = len(cumulative)
            selected = torch.ceil(fractions * step_count).long().clamp(min=1, max=step_count) - 1
            for step_idx in selected.unique().tolist():
                compressed_rank = math.ceil(cumulative[int(step_idx)] / r) - 1
                if jitter > 0:
                    compressed_rank += random.randint(-jitter, jitter)
                compressed_rank = max(0, min(int(compressed_rank), valid_indices.numel() - 1))
                anchor_mask[batch_idx, valid_indices[compressed_rank]] = 1.0

        return anchor_mask * mask

    def _merge_with_progress_fallback(
        self,
        anchor_mask: torch.Tensor,
        mask: torch.Tensor,
        anchor_count: int,
    ) -> torch.Tensor:
        if anchor_count <= 0:
            return anchor_mask
        fallback = self._progress_anchor_mask(mask, anchor_count)
        need_fallback = anchor_mask.sum(dim=1) < torch.minimum(
            mask.sum(dim=1),
            torch.ones_like(mask.sum(dim=1)) * min(anchor_count, 3),
        )
        if need_fallback.any():
            anchor_mask = anchor_mask.clone()
            anchor_mask[need_fallback] = torch.maximum(anchor_mask[need_fallback], fallback[need_fallback])
        return anchor_mask * mask

    def extra_sft_losses(self, **kwargs):
        cfg = self.trace_config
        if not cfg.get("enable_trace_path_pretraining", False):
            return {}

        if cfg.get("path_use_policy_mean", True):
            pred = kwargs["distributions"].mean.float()
        else:
            pred = kwargs["pred_embeds"].float()
        target = kwargs["gold_embeds_norm"].detach().float()
        mask = kwargs["steps_attention_mask"].float()
        if mask.sum() <= 0:
            return {}

        pred_path = self._prefix_path(pred, mask)
        target_path = self._prefix_path(target, mask)
        anchor_count = int(cfg.get("path_anchor_count", 3))
        raw_cot_steps = kwargs["batch"].get("steps", [])
        if isinstance(raw_cot_steps, str):
            cot_steps = [raw_cot_steps]
        else:
            cot_steps = list(raw_cot_steps)
        if not cot_steps:
            cot_steps = [""] * mask.shape[0]
        while len(cot_steps) < mask.shape[0]:
            cot_steps.append("")
        anchor_mask = self._cot_step_anchor_mask(
            cot_steps=cot_steps,
            mask=mask,
            compression_factor=int(kwargs["r"]),
            anchor_count=anchor_count,
            jitter=0,
        )
        anchor_mask = self._merge_with_progress_fallback(anchor_mask, mask, anchor_count)
        prefix_loss = self._masked_cosine_distance(pred_path, target_path, anchor_mask)

        bootstrap_mix = float(cfg.get("path_multiview_bootstrap_mix", 0.20))
        bootstrap_mix = min(max(bootstrap_mix, 0.0), 1.0)
        anchor_jitter = max(0, int(cfg.get("path_anchor_jitter", 1)))
        if bootstrap_mix > 0 and anchor_jitter > 0:
            bootstrap_mask = self._cot_step_anchor_mask(
                cot_steps=cot_steps,
                mask=mask,
                compression_factor=int(kwargs["r"]),
                anchor_count=anchor_count,
                jitter=anchor_jitter,
            )
            bootstrap_mask = self._merge_with_progress_fallback(bootstrap_mask, mask, anchor_count)
            bootstrap_loss = self._masked_cosine_distance(pred_path, target_path, bootstrap_mask)
            prefix_loss = (1.0 - bootstrap_mix) * prefix_loss + bootstrap_mix * bootstrap_loss
        else:
            bootstrap_mask = anchor_mask
            bootstrap_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

        pair_mask = mask[:, 1:] * mask[:, :-1]
        zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if pair_mask.sum() > 0:
            pred_delta = pred_path[:, 1:, :] - pred_path[:, :-1, :]
            target_delta = target_path[:, 1:, :] - target_path[:, :-1, :]
            direction_loss = self._masked_cosine_distance(pred_delta, target_delta, pair_mask)
            norm_scale = math.sqrt(pred.shape[-1])
            pred_step = pred_delta.norm(dim=-1) / norm_scale
            target_step = target_delta.norm(dim=-1).detach() / norm_scale
            step_values = F.smooth_l1_loss(pred_step, target_step, reduction="none")
            step_loss = self._masked_mean(step_values, pair_mask)
            pred_step_mean = self._masked_mean(pred_step.detach(), pair_mask)
            target_step_mean = self._masked_mean(target_step.detach(), pair_mask)
        else:
            direction_loss = zero
            step_loss = zero
            pred_step_mean = zero
            target_step_mean = zero

        direction_mix = float(cfg.get("path_direction_mix", 0.35))
        step_mix = float(cfg.get("path_step_mix", 0.15))
        direction_mix = min(max(direction_mix, 0.0), 1.0)
        step_mix = min(max(step_mix, 0.0), 1.0 - direction_mix)
        anchor_mix = 1.0 - direction_mix - step_mix
        path_loss = anchor_mix * prefix_loss + direction_mix * direction_loss + step_mix * step_loss
        weighted_path_loss = float(cfg.get("path_pretraining_weight", 0.15)) * path_loss

        return {
            "extra_sft_loss": weighted_path_loss,
            "trace_stage1_path_loss": path_loss,
            "trace_stage1_anchor_loss": prefix_loss,
            "trace_stage1_direction_loss": direction_loss,
            "trace_stage1_step_loss": step_loss,
            "trace_stage1_bootstrap_loss": bootstrap_loss,
            "trace_stage1_anchor_mix": torch.tensor(anchor_mix, device=pred.device, dtype=pred.dtype),
            "trace_stage1_direction_mix": torch.tensor(direction_mix, device=pred.device, dtype=pred.dtype),
            "trace_stage1_step_mix": torch.tensor(step_mix, device=pred.device, dtype=pred.dtype),
            "trace_stage1_bootstrap_mix": torch.tensor(bootstrap_mix, device=pred.device, dtype=pred.dtype),
            "trace_stage1_pred_step_norm": pred_step_mean,
            "trace_stage1_target_step_norm": target_step_mean,
            "trace_stage1_compression_view": torch.tensor(float(kwargs["r"]), device=pred.device, dtype=pred.dtype),
            "trace_stage1_anchor_count": anchor_mask.sum().detach(),
            "trace_stage1_bootstrap_anchor_count": bootstrap_mask.sum().detach(),
            "trace_stage1_valid_steps": mask.sum().detach(),
        }

    def origin_rollout_with_autocast(self, questions: List[str], gt_answers) -> grpo.Experience:
        device = getattr(self, "device", torch.device("cpu"))
        device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            return super().rollout(questions=questions, gt_answers=gt_answers)

    def limit_rl_train_epoch_length(self):
        if not self.trace_config.get("filter_informative_train_indices", True):
            return super().limit_rl_train_epoch_length()

        rl_config = self.model_kwargs.rl_config
        target_count = int(rl_config.n_train_samples_per_epoch)
        all_indices = list(self.trainer.datamodule.get_all_train_indices())
        if not all_indices:
            return super().limit_rl_train_epoch_length()

        candidate_count = int(self.trace_config.get("filter_candidate_count", 0) or 0)
        if candidate_count <= 0:
            factor = float(self.trace_config.get("filter_candidate_factor", 1.0))
            candidate_count = max(target_count, int(round(target_count * factor)))
        if candidate_count <= len(all_indices):
            candidate_indices = random.sample(all_indices, k=candidate_count)
        else:
            candidate_indices = random.choices(all_indices, k=candidate_count)
        self.trainer.datamodule.set_train_indices(candidate_indices)

        train_set = getattr(self.trainer.datamodule, "train_set", None)
        if train_set is None:
            dataloader = self.trainer.datamodule.get_dataloader_to_filter_indices()
        else:
            dataloader = DataLoader(
                train_set,
                batch_size=int(self.trace_config.get("filter_batch_size", 1)),
                shuffle=False,
            )

        mixed, positive, fallback, stats = self.filter_informative_indices(
            dataloader=dataloader,
            target_count=target_count,
        )
        selected = self.build_filtered_epoch_indices(
            mixed_indices=mixed,
            positive_indices=positive,
            fallback_indices=fallback,
            all_indices=all_indices,
            target_count=target_count,
        )
        random.shuffle(selected)
        self.trainer.datamodule.set_train_indices(selected)
        self.log_filter_stats(stats=stats, selected=selected, mixed_indices=mixed, positive_indices=positive)

    @torch.no_grad()
    def filter_informative_indices(self, dataloader, target_count: int):
        group_size = int(self.model_kwargs.rl_config.group_size)
        mixed, positive, fallback = [], [], []
        stats = {"seen": 0, "mixed": 0, "positive": 0, "all_wrong": 0, "all_correct": 0}
        stop_when_full = bool(self.trace_config.get("filter_stop_when_full", True))
        max_batches = int(self.trace_config.get("filter_max_batches", 0) or 0)

        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            questions = batch["question"]
            answers = batch["answer"]
            idx = batch["idx"]
            idx_list = idx.view(-1).cpu().tolist() if isinstance(idx, torch.Tensor) else list(idx)
            idx_list = [int(i) for i in idx_list]

            experience = self.origin_rollout_with_autocast(questions=questions, gt_answers=answers)
            acc = experience.accuracies.view(len(questions), group_size).float()
            pos_counts = acc.sum(dim=1).detach().cpu()
            for sample_idx, pos_count_tensor in zip(idx_list, pos_counts):
                pos_count = int(pos_count_tensor.item())
                stats["seen"] += 1
                if 0 < pos_count < group_size:
                    mixed.append(sample_idx)
                    positive.append(sample_idx)
                    fallback.append(sample_idx)
                    stats["mixed"] += 1
                    stats["positive"] += 1
                elif pos_count == group_size:
                    positive.append(sample_idx)
                    fallback.append(sample_idx)
                    stats["all_correct"] += 1
                    stats["positive"] += 1
                else:
                    fallback.append(sample_idx)
                    stats["all_wrong"] += 1
            torch.cuda.empty_cache()
            if stop_when_full and len(mixed) >= target_count:
                break
        return mixed, positive, fallback, stats

    def build_filtered_epoch_indices(
        self,
        mixed_indices: List[int],
        positive_indices: List[int],
        fallback_indices: List[int],
        all_indices: List[int],
        target_count: int,
    ) -> List[int]:
        mixed_fraction = float(self.trace_config.get("filter_mixed_fraction", 0.7))
        positive_fraction = float(self.trace_config.get("filter_positive_fraction", 0.9))
        selected: List[int] = []

        mixed_target = min(target_count, int(round(target_count * mixed_fraction)))
        if mixed_indices:
            if len(mixed_indices) >= mixed_target:
                selected.extend(random.sample(mixed_indices, k=mixed_target))
            else:
                selected.extend(mixed_indices)
                selected.extend(random.choices(mixed_indices, k=mixed_target - len(mixed_indices)))

        positive_target = min(target_count, int(round(target_count * positive_fraction)))
        positive_pool = [idx for idx in positive_indices if idx not in set(selected)]
        while len(selected) < positive_target and positive_pool:
            selected.append(random.choice(positive_pool))

        fallback_pool = [idx for idx in fallback_indices if idx not in set(selected)]
        while len(selected) < target_count and fallback_pool:
            selected.append(random.choice(fallback_pool))

        while len(selected) < target_count:
            selected.append(random.choice(all_indices))
        return selected[:target_count]

    def log_filter_stats(self, stats: dict, selected: List[int], mixed_indices: List[int], positive_indices: List[int]):
        mixed_set = set(mixed_indices)
        positive_set = set(positive_indices)
        mixed_frac = stats["mixed"] / max(stats["seen"], 1)
        positive_frac = stats["positive"] / max(stats["seen"], 1)
        selected_mixed = sum(1 for idx in selected if idx in mixed_set) / max(len(selected), 1)
        selected_positive = sum(1 for idx in selected if idx in positive_set) / max(len(selected), 1)
        message = (
            "TRACE MultiPath filter: "
            f"seen={stats['seen']} mixed={stats['mixed']} positive={stats['positive']} "
            f"all_wrong={stats['all_wrong']} all_correct={stats['all_correct']} "
            f"mixed_frac={mixed_frac:.4f} positive_frac={positive_frac:.4f} "
            f"selected_mixed={selected_mixed:.4f} selected_positive={selected_positive:.4f}"
        )
        text_logger = getattr(self, "text_logger", None)
        if text_logger is not None:
            text_logger.log(message)
        else:
            print(message)
        experiment = getattr(getattr(self, "logger", None), "experiment", None)
        if experiment is not None and hasattr(experiment, "add_scalar"):
            step = int(getattr(self, "global_step", 0) or 0)
            experiment.add_scalar("train/trace_filter/mixed_frac", mixed_frac, step)
            experiment.add_scalar("train/trace_filter/positive_frac", positive_frac, step)
            experiment.add_scalar("train/trace_filter/selected_mixed_frac", selected_mixed, step)
            experiment.add_scalar("train/trace_filter/selected_positive_frac", selected_positive, step)

    def rl_training_step(self, batch, batch_idx, dataloader_idx=0):
        rl_config = self.model_kwargs.rl_config
        questions = batch["question"]
        answers = batch["answer"]
        self.replay_buffer.clear()
        optimizer = self.optimizers()

        experience = self.rollout(questions=questions, gt_answers=answers)
        self.replay_buffer.append(experience.to("cpu"))
        group_size = int(rl_config.group_size)
        if len(questions) > 0 and group_size > 0:
            rollout_group_acc = experience.accuracies.detach().float().view(len(questions), group_size)
            all_wrong_frac_for_replay = (rollout_group_acc.sum(dim=1) <= 0.0).float().mean()
        else:
            all_wrong_frac_for_replay = torch.tensor(0.0, device=self.device)

        log_dict = {
            "train/rewards": experience.rewards.mean(),
            "train/accuracies": experience.accuracies.mean(),
            "train/n_latent_forward": experience.n_latent_forward.float().mean(),
        }
        if self._last_trace_metrics:
            log_dict.update({f"train/{k}": v for k, v in self._last_trace_metrics.items()})
        self.log_dict(log_dict, sync_dist=True, prog_bar=False, batch_size=len(batch["idx"]))

        torch.cuda.empty_cache()
        experience_dataloader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=rl_config.exp_batch_size,
            shuffle=True,
            collate_fn=grpo.join_experience_batch,
        )

        last_loss = None
        for experience in experience_dataloader:
            experience = experience.to(self.device)
            try:
                latent_logprobs, answer_logprobs = self.get_logprobs(e=experience)
            except ValueError:
                self.log("train/skipped_nonfinite", torch.tensor(1.0, device=self.device))
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            if (not torch.isfinite(latent_logprobs).all()) or (not torch.isfinite(answer_logprobs).all()):
                self.log("train/skipped_nonfinite", torch.tensor(1.0, device=self.device))
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            loss_dict = self.grpo_loss(
                latent_logprobs=latent_logprobs,
                answer_logprobs=answer_logprobs,
                experience=experience,
            )
            if not torch.isfinite(loss_dict["total_loss"]):
                self.log("train/skipped_nonfinite", torch.tensor(1.0, device=self.device))
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            optimizer.zero_grad()
            self.manual_backward(loss_dict["total_loss"])
            grad_norm = clip_grad_norm_(self.parameters(), max_norm=rl_config.get("clip_grad_norm", 1.0))
            if not torch.isfinite(grad_norm):
                self.log("train/skipped_nonfinite", torch.tensor(1.0, device=self.device))
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            optimizer.step()

            train_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            train_loss_dict["train/grad_norm"] = grad_norm
            self.log_dict(train_loss_dict, sync_dist=True, prog_bar=False)
            last_loss = loss_dict["total_loss"].detach()

        if last_loss is None:
            last_loss = torch.tensor(0.0, device=self.device)
        base_replay_weight = float(self.trace_config.get("stage2_sft_replay_weight", 0.0) or 0.0)
        all_wrong_multiplier = float(self.trace_config.get("stage2_all_wrong_replay_multiplier", 0.0) or 0.0)
        if base_replay_weight > 0:
            replay_loss_dict = self.forward(batch=batch)
            replay_weight = replay_loss_dict["total_loss"].new_tensor(base_replay_weight)
            if all_wrong_multiplier > 0:
                all_wrong_frac = all_wrong_frac_for_replay.to(device=replay_weight.device, dtype=replay_weight.dtype)
                replay_weight = replay_weight * (1.0 + all_wrong_multiplier * all_wrong_frac)
            replay_loss = replay_weight * replay_loss_dict["total_loss"]
            if torch.isfinite(replay_loss):
                optimizer.zero_grad()
                self.manual_backward(replay_loss)
                replay_grad_norm = clip_grad_norm_(self.parameters(), max_norm=rl_config.get("clip_grad_norm", 1.0))
                if torch.isfinite(replay_grad_norm):
                    optimizer.step()
                    replay_logs = {
                        "train/stage2_sft_replay_loss": replay_loss.detach(),
                        "train/stage2_sft_replay_raw_loss": replay_loss_dict["total_loss"].detach(),
                        "train/stage2_sft_replay_effective_weight": replay_weight.detach(),
                        "train/stage2_sft_replay_all_wrong_frac": all_wrong_frac_for_replay.detach(),
                        "train/stage2_sft_replay_grad_norm": replay_grad_norm,
                    }
                    for key in (
                        "trace_stage1_path_loss",
                        "trace_stage1_anchor_loss",
                        "trace_stage1_direction_loss",
                        "trace_stage1_step_loss",
                    ):
                        if key in replay_loss_dict and isinstance(replay_loss_dict[key], torch.Tensor):
                            replay_logs[f"train/stage2_replay_{key}"] = replay_loss_dict[key].detach()
                    self.log_dict(replay_logs, sync_dist=True, prog_bar=False, batch_size=len(batch["idx"]))
                    last_loss = last_loss + replay_loss.detach()
                else:
                    self.log("train/skipped_nonfinite", torch.tensor(1.0, device=self.device))
                    optimizer.zero_grad(set_to_none=True)
            else:
                self.log("train/skipped_nonfinite", torch.tensor(1.0, device=self.device))
                optimizer.zero_grad(set_to_none=True)
        return last_loss

    @torch.no_grad()
    def rollout(self, questions: List[str], gt_answers) -> grpo.Experience:
        self._last_trace_metrics = {}
        batch_size = len(questions)
        group_size = int(self.model_kwargs.rl_config.group_size)

        experience = self.origin_rollout_with_autocast(questions=questions, gt_answers=gt_answers)
        if self.trace_config.get("resample_informative_groups", False):
            experience = self.resample_informative_groups(
                questions=questions,
                gt_answers=gt_answers,
                initial_experience=experience,
                batch_size=batch_size,
                group_size=group_size,
            )
        if self.trace_config.get("enable_trace_multipath_reward", True):
            experience = self.apply_trace_multipath_rewards(
                experience=experience,
                batch_size=batch_size,
                group_size=group_size,
                gt_answers=gt_answers,
            )
        return experience

    @torch.no_grad()
    def resample_informative_groups(
        self,
        questions: List[str],
        gt_answers,
        initial_experience: grpo.Experience,
        batch_size: int,
        group_size: int,
    ) -> grpo.Experience:
        max_attempts = max(1, int(self.trace_config.get("resample_max_attempts", 3)))
        best = initial_experience
        best_score = self.informative_group_score(best, batch_size=batch_size, group_size=group_size)
        attempts = 1
        while attempts < max_attempts and best_score < float(self.trace_config.get("resample_target_score", 1.0)):
            candidate = self.origin_rollout_with_autocast(questions=questions, gt_answers=gt_answers)
            attempts += 1
            score = self.informative_group_score(candidate, batch_size=batch_size, group_size=group_size)
            if score > best_score:
                best, best_score = candidate, score
        self._last_trace_metrics["trace/resample_attempts"] = torch.tensor(float(attempts), device=self.device)
        self._last_trace_metrics["trace/resample_score"] = torch.tensor(float(best_score), device=self.device)
        return best

    def informative_group_score(self, experience: grpo.Experience, batch_size: int, group_size: int) -> float:
        group_acc = experience.accuracies.view(batch_size, group_size).float()
        pos_counts = group_acc.sum(dim=1)
        mixed = ((pos_counts > 0) & (pos_counts < group_size)).float().mean().item()
        nontrivial = (pos_counts > 0).float().mean().item()
        score = mixed + 0.25 * nontrivial

        if self.trace_config.get("filter_score_geometry", True):
            path, _, _, _ = self.latent_path_embeddings(experience)
            hard_scores = []
            diversity_scores = []
            for sample_idx in range(batch_size):
                start = sample_idx * group_size
                end = start + group_size
                group_path = self.prepare_group_paths(path[start:end])
                group_pos = group_acc[sample_idx] > 0.5
                if group_pos.any() and (~group_pos).any():
                    pos_paths = group_path[group_pos]
                    neg_paths = group_path[~group_pos]
                    prototypes, _, _ = self.build_positive_modes(
                        pos_paths,
                        torch.zeros(pos_paths.shape[0], device=pos_paths.device),
                    )
                    nearest_neg = (neg_paths @ prototypes.T).max(dim=1).values
                    hard_threshold = float(self.trace_config.get("hard_threshold", 0.25))
                    hard_scores.append((nearest_neg > hard_threshold).float().mean().item())
                if int(group_pos.sum().item()) > 1:
                    pos_paths = group_path[group_pos]
                    pos_sim = pos_paths @ pos_paths.T
                    off_diag = ~torch.eye(pos_paths.shape[0], dtype=torch.bool, device=pos_paths.device)
                    diversity_scores.append((1.0 - pos_sim[off_diag].mean()).clamp_min(0.0).item())
            if hard_scores:
                score += 0.20 * (sum(hard_scores) / len(hard_scores))
            if diversity_scores:
                score += 0.10 * (sum(diversity_scores) / len(diversity_scores))
        return score

    def latent_path_embeddings(self, experience: grpo.Experience):
        latents = experience.latent_inputs_embeds.float()
        mask = experience.latent_attention_mask.float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_state = (latents * mask.unsqueeze(-1)).sum(dim=1) / denom
        lengths = mask.sum(dim=1).long().clamp_min(1)
        gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, latents.shape[-1])
        first_state = latents[:, 0, :]
        last_state = latents.gather(dim=1, index=gather_idx).squeeze(1)
        trend_state = last_state - first_state

        if latents.shape[1] <= 1:
            zeros = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.float32)
            delta_state = torch.zeros_like(mean_state)
            path = self.build_path_signature(mean_state, last_state, trend_state, delta_state)
            return path, zeros, zeros, zeros

        deltas = latents[:, 1:, :] - latents[:, :-1, :]
        delta_mask = mask[:, 1:] * mask[:, :-1]
        delta_denom = delta_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        delta_state = (deltas * delta_mask.unsqueeze(-1)).sum(dim=1) / delta_denom
        delta_norm = deltas.norm(dim=-1)
        delta_norm_unit = delta_norm / (math.sqrt(latents.shape[-1]) * float(self.embeds_std))
        noncollapse_margin = float(self.trace_config.get("noncollapse_margin", 0.08))
        noncollapse = (delta_norm_unit - noncollapse_margin).clamp_min(0.0)
        noncollapse = (noncollapse * delta_mask).sum(dim=1) / delta_mask.sum(dim=1).clamp_min(1.0)
        mean_delta_norm = (delta_norm_unit * delta_mask).sum(dim=1) / delta_mask.sum(dim=1).clamp_min(1.0)

        if deltas.shape[1] <= 1:
            coherence = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.float32)
        else:
            delta_dir = F.normalize(deltas, dim=-1)
            pair_mask = delta_mask[:, 1:] * delta_mask[:, :-1]
            step_cos = F.cosine_similarity(delta_dir[:, 1:, :], delta_dir[:, :-1, :], dim=-1)
            coherence = (step_cos * pair_mask).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1.0)
        path = self.build_path_signature(mean_state, last_state, trend_state, delta_state)
        return path, coherence, noncollapse, mean_delta_norm

    def build_path_signature(
        self,
        mean_state: torch.Tensor,
        last_state: torch.Tensor,
        trend_state: torch.Tensor,
        delta_state: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.trace_config
        mean_weight = float(cfg.get("signature_mean_weight", 0.5))
        last_weight = float(cfg.get("signature_last_weight", 0.5))
        trend_weight = float(cfg.get("signature_trend_weight", 1.0))
        delta_weight = float(cfg.get("signature_delta_weight", 1.0))
        parts = [
            mean_weight * F.normalize(mean_state, dim=-1),
            last_weight * F.normalize(last_state, dim=-1),
            trend_weight * F.normalize(trend_state, dim=-1),
            delta_weight * F.normalize(delta_state, dim=-1),
        ]
        return F.normalize(torch.cat(parts, dim=-1), dim=-1)

    def prepare_group_paths(self, group_path: torch.Tensor) -> torch.Tensor:
        if not self.trace_config.get("center_path_signature", True):
            return F.normalize(group_path, dim=-1)
        centered = F.normalize(group_path - group_path.mean(dim=0, keepdim=True), dim=-1)
        raw_mix = float(self.trace_config.get("path_signature_raw_mix", 0.25))
        if raw_mix <= 0:
            return centered
        return F.normalize(torch.cat([centered, raw_mix * group_path], dim=-1), dim=-1)

    def build_positive_modes(
        self,
        pos_paths: torch.Tensor,
        pos_conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_modes = max(1, int(self.trace_config.get("max_modes", 3)))
        merge_threshold = float(self.trace_config.get("mode_merge_threshold", 0.82))
        n_pos = pos_paths.shape[0]
        min_modes = min(max_modes, n_pos, max(1, int(self.trace_config.get("min_positive_modes", 1))))
        target_modes = min(max_modes, n_pos, max(min_modes, int(self.trace_config.get("target_positive_modes", max_modes))))
        order = torch.argsort(pos_conf, descending=True)
        prototypes = []
        seed_indices = []
        assignments = torch.zeros(n_pos, device=pos_paths.device, dtype=torch.long)

        for local_idx in order.tolist():
            candidate = pos_paths[local_idx]
            if not prototypes:
                prototypes.append(candidate)
                seed_indices.append(local_idx)
                break

        selected = set(seed_indices)
        while len(prototypes) < min_modes:
            proto_tensor = torch.stack(prototypes, dim=0)
            nearest = (pos_paths @ proto_tensor.T).max(dim=1).values
            if selected:
                selected_tensor = torch.tensor(list(selected), device=pos_paths.device, dtype=torch.long)
                nearest[selected_tensor] = 2.0
            local_idx = int(torch.argmin(nearest).item())
            prototypes.append(pos_paths[local_idx])
            seed_indices.append(local_idx)
            selected.add(local_idx)

        for local_idx in order.tolist():
            if local_idx in selected:
                continue
            if len(prototypes) >= target_modes:
                break
            candidate = pos_paths[local_idx]
            proto_tensor = torch.stack(prototypes, dim=0)
            max_sim = (candidate.unsqueeze(0) @ proto_tensor.T).max()
            if max_sim < merge_threshold and len(prototypes) < max_modes:
                prototypes.append(candidate)
                seed_indices.append(local_idx)
                selected.add(local_idx)

        proto_tensor = F.normalize(torch.stack(prototypes, dim=0), dim=-1)
        for _ in range(2):
            sims = pos_paths @ proto_tensor.T
            assignments = sims.argmax(dim=1)
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

    def mode_entropy(self, counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        effective_modes = torch.exp(entropy)
        return entropy, effective_modes

    @torch.no_grad()
    def numeric_proximity_scores(
        self,
        experience: grpo.Experience,
        gt_answers,
        batch_size: int,
        group_size: int,
    ) -> Optional[torch.Tensor]:
        if gt_answers is None:
            return None
        pred_strings = self.tokenizer.batch_decode(
            experience.answer_input_ids.detach().cpu(),
            skip_special_tokens=True,
        )
        scores = []
        for sample_idx in range(batch_size):
            gold = self._numeric_answer_value(gt_answers[sample_idx])
            for pred_string in pred_strings[sample_idx * group_size : (sample_idx + 1) * group_size]:
                pred = self._numeric_answer_value(pred_string)
                if gold is None or pred is None:
                    scores.append(0.0)
                    continue
                scale = max(abs(gold), 1.0)
                rel_error = abs(pred - gold) / scale
                scores.append(1.0 / (1.0 + rel_error))
        return torch.tensor(scores, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def apply_trace_multipath_rewards(
        self,
        experience: grpo.Experience,
        batch_size: int,
        group_size: int,
        gt_answers=None,
    ) -> grpo.Experience:
        cfg = self.trace_config
        path, step_coherence, noncollapse, mean_delta_norm = self.latent_path_embeddings(experience)
        answer_mask = experience.answer_attention_mask.float()
        answer_conf = (experience.answer_logprobs.float() * answer_mask).sum(dim=1)
        answer_conf = answer_conf / answer_mask.sum(dim=1).clamp_min(1.0)
        numeric_weight = float(cfg.get("all_wrong_numeric_reward_weight", 0.0) or 0.0)
        numeric_clip = float(cfg.get("all_wrong_numeric_reward_clip", 2.0) or 2.0)
        dense_numeric_weight = float(cfg.get("numeric_dense_reward_weight", 0.0) or 0.0)
        dense_numeric_clip = float(cfg.get("numeric_dense_reward_clip", 2.0) or 2.0)
        near_miss_weight = float(cfg.get("near_miss_path_bootstrap_weight", 0.0) or 0.0)
        near_miss_clip = float(cfg.get("near_miss_path_bootstrap_clip", 2.0) or 2.0)
        near_miss_min_span = float(cfg.get("near_miss_path_min_span", 0.02) or 0.02)
        near_miss_min_score = float(cfg.get("near_miss_path_min_score", 0.0) or 0.0)
        near_miss_neg_margin = float(cfg.get("near_miss_path_neg_margin", 0.15) or 0.15)
        numeric_scores = None
        if numeric_weight > 0 or dense_numeric_weight > 0 or near_miss_weight > 0:
            numeric_scores = self.numeric_proximity_scores(
                experience=experience,
                gt_answers=gt_answers,
                batch_size=batch_size,
                group_size=group_size,
            )

        shaped_rewards = experience.rewards.float().clone()
        all_advantages = []
        metrics = {
            "trace/bonus": [],
            "trace/answer_reward": [],
            "trace/pos_count": [],
            "trace/neg_count": [],
            "trace/hard_count": [],
            "trace/mixed_frac": [],
            "trace/mode_count": [],
            "trace/effective_modes": [],
            "trace/mode_entropy": [],
            "trace/pos_intra_sim": [],
            "trace/pos_pair_sim": [],
            "trace/pos_duplicate_penalty": [],
            "trace/pos_hard_sim": [],
            "trace/proto_inter_sim": [],
            "trace/neg_proto_sim": [],
            "trace/hard_proto_sim": [],
            "trace/step_coherence": [],
            "trace/noncollapse": [],
            "trace/delta_norm": [],
            "trace/all_wrong_numeric_active": [],
            "trace/all_wrong_numeric_span": [],
            "trace/all_wrong_numeric_bonus": [],
            "trace/numeric_dense_active": [],
            "trace/numeric_dense_span": [],
            "trace/numeric_dense_abs_bonus": [],
            "trace/near_miss_path_active": [],
            "trace/near_miss_path_span": [],
            "trace/near_miss_path_best": [],
            "trace/near_miss_path_proto_sim": [],
            "trace/near_miss_path_abs_bonus": [],
        }

        reward_weight = float(cfg.get("trace_reward_weight", 0.25))
        pos_fit_weight = float(cfg.get("pos_mode_fit_weight", 0.5))
        mode_div_weight = float(cfg.get("mode_diversity_weight", 0.25))
        pos_pair_weight = float(cfg.get("pos_pair_repulsion_weight", 0.0))
        mode_balance_weight = float(cfg.get("mode_balance_weight", 0.0))
        hard_pos_weight = float(cfg.get("hard_pos_repulsion_weight", 0.0))
        neg_weight = float(cfg.get("neg_repulsion_weight", 0.6))
        hard_weight = float(cfg.get("hard_repulsion_weight", 1.0))
        step_weight = float(cfg.get("step_coherence_weight", 0.05))
        noncollapse_weight = float(cfg.get("noncollapse_weight", 0.05))
        neg_margin = float(cfg.get("neg_margin", 0.35))
        hard_margin = float(cfg.get("hard_margin", 0.15))
        hard_threshold = float(cfg.get("hard_threshold", 0.55))
        pos_pair_margin = float(cfg.get("pos_pair_margin", 0.75))
        bonus_clip = float(cfg.get("trace_bonus_clip", 0.0) or 0.0)
        max_modes = max(1, int(cfg.get("max_modes", 3)))

        for sample_idx in range(batch_size):
            start = sample_idx * group_size
            end = start + group_size
            group_path = self.prepare_group_paths(path[start:end])
            group_acc = experience.accuracies[start:end].view(-1).float()
            group_conf = answer_conf[start:end]
            group_step = step_coherence[start:end]
            group_noncollapse = noncollapse[start:end]
            group_delta_norm = mean_delta_norm[start:end]

            pos_mask = group_acc > 0.5
            neg_mask = ~pos_mask
            pos_count = int(pos_mask.sum().item())
            neg_count = int(neg_mask.sum().item())
            bonus = torch.zeros(group_size, device=group_path.device, dtype=torch.float32)
            numeric_bonus = torch.zeros_like(bonus)
            dense_numeric_bonus = torch.zeros_like(bonus)
            near_miss_bonus = torch.zeros_like(bonus)

            bonus += step_weight * group_step
            bonus += noncollapse_weight * group_noncollapse

            metrics["trace/answer_reward"].append(group_acc.mean().item())
            metrics["trace/pos_count"].append(float(pos_count))
            metrics["trace/neg_count"].append(float(neg_count))
            metrics["trace/mixed_frac"].append(float(0 < pos_count < group_size))
            metrics["trace/step_coherence"].append(group_step.mean().item())
            metrics["trace/noncollapse"].append(group_noncollapse.mean().item())
            metrics["trace/delta_norm"].append(group_delta_norm.mean().item())
            metrics["trace/all_wrong_numeric_active"].append(0.0)
            metrics["trace/all_wrong_numeric_span"].append(0.0)
            metrics["trace/all_wrong_numeric_bonus"].append(0.0)
            metrics["trace/numeric_dense_active"].append(0.0)
            metrics["trace/numeric_dense_span"].append(0.0)
            metrics["trace/numeric_dense_abs_bonus"].append(0.0)
            metrics["trace/near_miss_path_active"].append(0.0)
            metrics["trace/near_miss_path_span"].append(0.0)
            metrics["trace/near_miss_path_best"].append(0.0)
            metrics["trace/near_miss_path_proto_sim"].append(0.0)
            metrics["trace/near_miss_path_abs_bonus"].append(0.0)

            hard_count = 0
            mode_count = 0
            effective_modes = torch.tensor(0.0, device=group_path.device)
            mode_entropy = torch.tensor(0.0, device=group_path.device)
            pos_intra_sim = torch.tensor(0.0, device=group_path.device)
            pos_pair_sim = torch.tensor(0.0, device=group_path.device)
            pos_duplicate_penalty = torch.tensor(0.0, device=group_path.device)
            pos_hard_sim = torch.tensor(0.0, device=group_path.device)
            proto_inter_sim = torch.tensor(0.0, device=group_path.device)
            neg_proto_sim = torch.tensor(0.0, device=group_path.device)
            hard_proto_sim = torch.tensor(0.0, device=group_path.device)

            if pos_count > 0:
                pos_paths = group_path[pos_mask]
                pos_conf = group_conf[pos_mask]
                pos_indices = pos_mask.nonzero(as_tuple=False).view(-1)
                prototypes, assignments, counts = self.build_positive_modes(pos_paths, pos_conf)
                mode_count = int(prototypes.shape[0])
                mode_entropy, effective_modes = self.mode_entropy(counts)
                pos_proto_sim = pos_paths @ prototypes.T
                nearest_pos_sim, nearest_pos_mode = pos_proto_sim.max(dim=1)
                bonus[pos_mask] += pos_fit_weight * nearest_pos_sim
                pos_intra_sim = nearest_pos_sim.mean()

                if pos_count > 1 and pos_pair_weight > 0:
                    pair_sim = pos_paths @ pos_paths.T
                    eye = torch.eye(pos_count, dtype=torch.bool, device=group_path.device)
                    off_diag = ~eye
                    pos_pair_sim = pair_sim[off_diag].mean()
                    nearest_other = pair_sim.masked_fill(eye, -2.0).max(dim=1).values
                    duplicate_penalty = F.relu(nearest_other - pos_pair_margin)
                    pos_duplicate_penalty = duplicate_penalty.mean()
                    bonus[pos_indices] -= pos_pair_weight * duplicate_penalty

                if mode_count > 1:
                    proto_sim = prototypes @ prototypes.T
                    off_diag = ~torch.eye(mode_count, dtype=torch.bool, device=group_path.device)
                    proto_inter_sim = proto_sim[off_diag].mean()
                    separation = (1.0 - proto_inter_sim).clamp_min(0.0)
                    coverage = (effective_modes - 1.0) / max(float(min(max_modes, pos_count) - 1), 1.0)
                    bonus[pos_mask] += mode_div_weight * (0.5 * separation + 0.5 * coverage.clamp(0.0, 1.0))
                    if mode_balance_weight > 0:
                        assigned_counts = counts[assignments].to(device=group_path.device)
                        rarity = 1.0 - assigned_counts / max(float(pos_count), 1.0)
                        bonus[pos_indices] += mode_balance_weight * rarity

                if neg_count > 0:
                    neg_paths = group_path[neg_mask]
                    neg_proto = neg_paths @ prototypes.T
                    nearest_neg_sim, _ = neg_proto.max(dim=1)
                    neg_proto_sim = nearest_neg_sim.mean()
                    neg_penalty = F.relu(nearest_neg_sim - neg_margin)
                    neg_indices = neg_mask.nonzero(as_tuple=False).view(-1)
                    bonus[neg_indices] -= neg_weight * neg_penalty

                    hard_mask_local = nearest_neg_sim > hard_threshold
                    if hard_mask_local.any():
                        hard_indices = neg_indices[hard_mask_local]
                        hard_sims = nearest_neg_sim[hard_mask_local]
                        hard_proto_sim = hard_sims.mean()
                        hard_count = int(hard_mask_local.sum().item())
                        bonus[hard_indices] -= hard_weight * F.relu(hard_sims - hard_margin)
                        if hard_pos_weight > 0:
                            hard_paths = group_path[hard_indices]
                            pos_hard = (pos_paths @ hard_paths.T).max(dim=1).values
                            pos_hard_sim = pos_hard.mean()
                            bonus[pos_indices] -= hard_pos_weight * F.relu(pos_hard - hard_margin)
                    else:
                        hard_top_k = int(cfg.get("hard_top_k", 0) or 0)
                        if hard_top_k > 0:
                            k = min(hard_top_k, neg_count)
                            top_vals, top_local = torch.topk(nearest_neg_sim, k=k)
                            hard_indices = neg_indices[top_local]
                            hard_proto_sim = top_vals.mean()
                            hard_count = int(k)
                            bonus[hard_indices] -= hard_weight * F.relu(top_vals - hard_margin)
                            if hard_pos_weight > 0:
                                hard_paths = group_path[hard_indices]
                                pos_hard = (pos_paths @ hard_paths.T).max(dim=1).values
                                pos_hard_sim = pos_hard.mean()
                                bonus[pos_indices] -= hard_pos_weight * F.relu(pos_hard - hard_margin)

            if bonus_clip > 0:
                bonus = bonus.clamp(min=-bonus_clip, max=bonus_clip)
            scaled_bonus = reward_weight * bonus.unsqueeze(1)
            if pos_count == 0 and numeric_scores is not None and numeric_weight > 0:
                group_numeric = numeric_scores[start:end].to(device=group_path.device)
                numeric_span = group_numeric.max() - group_numeric.min()
                if torch.isfinite(numeric_span) and numeric_span > 1e-6:
                    centered = group_numeric - group_numeric.mean()
                    z = centered / group_numeric.std(unbiased=False).clamp_min(1e-6)
                    numeric_bonus = numeric_weight * z.clamp(min=-numeric_clip, max=numeric_clip)
                    metrics["trace/all_wrong_numeric_active"][-1] = 1.0
                    metrics["trace/all_wrong_numeric_span"][-1] = float(numeric_span.item())
                    metrics["trace/all_wrong_numeric_bonus"][-1] = float(numeric_bonus.mean().item())
            if numeric_scores is not None and dense_numeric_weight > 0:
                group_numeric = numeric_scores[start:end].to(device=group_path.device)
                numeric_span = group_numeric.max() - group_numeric.min()
                if torch.isfinite(numeric_span) and numeric_span > 1e-6:
                    centered = group_numeric - group_numeric.mean()
                    z = centered / group_numeric.std(unbiased=False).clamp_min(1e-6)
                    dense_numeric_bonus = dense_numeric_weight * z.clamp(
                        min=-dense_numeric_clip,
                        max=dense_numeric_clip,
                    )
                    metrics["trace/numeric_dense_active"][-1] = 1.0
                    metrics["trace/numeric_dense_span"][-1] = float(numeric_span.item())
                    metrics["trace/numeric_dense_abs_bonus"][-1] = float(dense_numeric_bonus.abs().mean().item())
            if pos_count == 0 and numeric_scores is not None and near_miss_weight > 0:
                group_numeric = numeric_scores[start:end].to(device=group_path.device)
                numeric_span = group_numeric.max() - group_numeric.min()
                if torch.isfinite(numeric_span) and numeric_span > near_miss_min_span:
                    best_idx = int(torch.argmax(group_numeric).item())
                    best_score = group_numeric[best_idx]
                    if best_score >= near_miss_min_score:
                        best_path = group_path[best_idx]
                        path_to_best = group_path @ best_path
                        normalized_numeric = (group_numeric - group_numeric.min()) / numeric_span.clamp_min(1e-6)
                        positive_anchor = normalized_numeric * path_to_best.clamp_min(0.0)
                        near_wrong_penalty = (1.0 - normalized_numeric) * F.relu(path_to_best - near_miss_neg_margin)
                        near_miss_raw = positive_anchor - near_wrong_penalty
                        near_miss_bonus = near_miss_weight * near_miss_raw.clamp(
                            min=-near_miss_clip,
                            max=near_miss_clip,
                        )
                        metrics["trace/near_miss_path_active"][-1] = 1.0
                        metrics["trace/near_miss_path_span"][-1] = float(numeric_span.item())
                        metrics["trace/near_miss_path_best"][-1] = float(best_score.item())
                        metrics["trace/near_miss_path_proto_sim"][-1] = float(path_to_best.mean().item())
                        metrics["trace/near_miss_path_abs_bonus"][-1] = float(near_miss_bonus.abs().mean().item())
            shaped_rewards[start:end] = (
                shaped_rewards[start:end]
                + scaled_bonus
                + numeric_bonus.unsqueeze(1)
                + dense_numeric_bonus.unsqueeze(1)
                + near_miss_bonus.unsqueeze(1)
            )
            all_advantages.append(grpo.group_advantages(shaped_rewards[start:end]))

            metrics["trace/bonus"].append(scaled_bonus.mean().item())
            metrics["trace/hard_count"].append(float(hard_count))
            metrics["trace/mode_count"].append(float(mode_count))
            metrics["trace/effective_modes"].append(float(effective_modes.item()))
            metrics["trace/mode_entropy"].append(float(mode_entropy.item()))
            metrics["trace/pos_intra_sim"].append(float(pos_intra_sim.item()))
            metrics["trace/pos_pair_sim"].append(float(pos_pair_sim.item()))
            metrics["trace/pos_duplicate_penalty"].append(float(pos_duplicate_penalty.item()))
            metrics["trace/pos_hard_sim"].append(float(pos_hard_sim.item()))
            metrics["trace/proto_inter_sim"].append(float(proto_inter_sim.item()))
            metrics["trace/neg_proto_sim"].append(float(neg_proto_sim.item()))
            metrics["trace/hard_proto_sim"].append(float(hard_proto_sim.item()))

        experience.rewards = shaped_rewards
        experience.advantages = torch.cat(all_advantages, dim=0)
        self._last_trace_metrics.update(
            {
                key: torch.tensor(sum(values) / max(len(values), 1), device=self.device, dtype=torch.float32)
                for key, values in metrics.items()
            }
        )
        return experience
