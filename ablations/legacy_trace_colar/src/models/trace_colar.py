import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from .colar import LitCoLaR
from ..modules import grpo


class LitTRACECoLaR(LitCoLaR):
    """CoLaR with outcome-supervised latent trajectory relation rewards.

    The SFT path is intentionally inherited from CoLaR: original CoT steps still
    initialize compressed latent reasoning. TRACE only changes the RL stage by
    turning group outcomes into trajectory-level relational rewards.
    """

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)
        self.trace_config = model_kwargs.get("trace_config", {})
        self._last_trace_metrics = {}

    def on_fit_start(self):
        if self.model_kwargs.do_rl and self.trace_config.get("trace_filter_mixed_train_indices", False):
            LitCoLaR.limit_rl_train_epoch_length(self)
            return super(LitCoLaR, self).on_fit_start()
        return super().on_fit_start()

    def limit_rl_train_epoch_length(self):
        if not self.trace_config.get("trace_filter_mixed_train_indices", False):
            return super().limit_rl_train_epoch_length()

        rl_config = self.model_kwargs.rl_config
        n_indices = int(rl_config.n_train_samples_per_epoch)
        all_indices = list(self.trainer.datamodule.get_all_train_indices())
        if not all_indices:
            return super().limit_rl_train_epoch_length()

        candidate_count = int(self.trace_config.get("trace_filter_candidate_count", 0) or 0)
        if candidate_count <= 0:
            candidate_factor = float(self.trace_config.get("trace_filter_candidate_factor", 1.0))
            candidate_count = max(n_indices, int(round(n_indices * candidate_factor)))
        candidate_count = max(1, candidate_count)
        if candidate_count <= len(all_indices):
            candidate_indices = random.sample(all_indices, k=candidate_count)
        else:
            candidate_indices = random.choices(all_indices, k=candidate_count)

        self.trainer.datamodule.set_train_indices(candidate_indices)
        filter_batch_size = int(self.trace_config.get("trace_filter_batch_size", 1))
        train_set = getattr(self.trainer.datamodule, "train_set", None)
        if train_set is not None:
            dataloader_to_filter_indices = DataLoader(train_set, batch_size=filter_batch_size, shuffle=False)
        else:
            dataloader_to_filter_indices = self.trainer.datamodule.get_dataloader_to_filter_indices()
        mixed_indices, fallback_indices, stats = self.filter_trace_train_indices(
            dataloader_to_filter_indices,
            target_count=n_indices,
        )

        selected = self.build_trace_filtered_epoch_indices(
            mixed_indices=mixed_indices,
            fallback_indices=fallback_indices,
            all_indices=all_indices,
            target_count=n_indices,
        )
        random.shuffle(selected)
        self.trainer.datamodule.set_train_indices(selected)
        self._log_trace_filter_stats(stats=stats, selected=selected, mixed_indices=mixed_indices)

    def rl_training_step(self, batch, batch_idx, dataloader_idx=0):
        output = super().rl_training_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        if self._last_trace_metrics:
            self.log_dict(
                {f"train/{k}": v for k, v in self._last_trace_metrics.items()},
                sync_dist=True,
                prog_bar=False,
                batch_size=len(batch["idx"]),
            )
        return output

    @torch.no_grad()
    def rollout_group_accuracies(self, questions: List[str], gt_answers) -> torch.Tensor:
        rl_config = self.model_kwargs.rl_config
        group_size = int(rl_config.group_size)
        batch_size = len(questions)
        group_questions = []
        for question in questions:
            group_questions.extend([question] * group_size)

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":")[0]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            _, _, _, latent_attention_mask, pred_ids = self.latent_generate(
                questions=group_questions,
                rl_mode=True,
            )
        pred_answer_strings = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        n_latent_forward = latent_attention_mask.sum(dim=1)

        all_accuracies = []
        for sample_idx in range(batch_size):
            start = sample_idx * group_size
            end = start + group_size
            _, accuracies = self.get_group_rewards_and_acc(
                pred_answers=pred_answer_strings[start:end],
                gt_answer=gt_answers[sample_idx],
                n_latent_forward=n_latent_forward[start:end],
            )
            all_accuracies.append(accuracies.view(1, group_size))
        return torch.cat(all_accuracies, dim=0)

    @torch.no_grad()
    def filter_trace_train_indices(self, dataloader_to_filter_indices, target_count: int) -> Tuple[List[int], List[int], dict]:
        group_size = int(self.model_kwargs.rl_config.group_size)
        mixed_indices = []
        fallback_indices = []
        stats = {"seen": 0, "mixed": 0, "all_wrong": 0, "all_correct": 0}
        stop_when_full = self.trace_config.get("trace_filter_stop_when_full", True)

        for batch in tqdm.tqdm(dataloader_to_filter_indices, desc="TRACE filtering mixed train indices"):
            questions = batch["question"]
            answers = batch["answer"]
            idx = batch["idx"]
            idx_list = idx.view(-1).cpu().tolist() if isinstance(idx, torch.Tensor) else list(idx)
            idx_list = [int(i) for i in idx_list]

            accuracies = self.rollout_group_accuracies(questions=questions, gt_answers=answers)
            pos_counts = accuracies.sum(dim=1).detach().cpu()
            for sample_idx, pos_count_tensor in zip(idx_list, pos_counts):
                pos_count = int(pos_count_tensor.item())
                stats["seen"] += 1
                if 0 < pos_count < group_size:
                    mixed_indices.append(sample_idx)
                    fallback_indices.append(sample_idx)
                    stats["mixed"] += 1
                elif pos_count == 0:
                    fallback_indices.append(sample_idx)
                    stats["all_wrong"] += 1
                else:
                    stats["all_correct"] += 1

            torch.cuda.empty_cache()
            if stop_when_full and len(mixed_indices) >= target_count:
                break

        return mixed_indices, fallback_indices, stats

    def build_trace_filtered_epoch_indices(
        self,
        mixed_indices: List[int],
        fallback_indices: List[int],
        all_indices: List[int],
        target_count: int,
    ) -> List[int]:
        selected = list(mixed_indices[:target_count])
        if len(selected) >= target_count:
            return selected

        mixed_fill_fraction = float(self.trace_config.get("trace_filter_mixed_fill_fraction", 0.5))
        mixed_fill_target = int(round(target_count * mixed_fill_fraction))
        if mixed_indices and len(selected) < mixed_fill_target:
            selected.extend(random.choices(mixed_indices, k=mixed_fill_target - len(selected)))

        remaining = target_count - len(selected)
        if remaining <= 0:
            return selected[:target_count]

        dedup_selected = set(selected)
        fallback_pool = [idx for idx in fallback_indices if idx not in dedup_selected]
        if fallback_pool:
            if remaining <= len(fallback_pool):
                selected.extend(random.sample(fallback_pool, k=remaining))
            else:
                selected.extend(fallback_pool)
                selected.extend(random.choices(fallback_pool, k=remaining - len(fallback_pool)))

        remaining = target_count - len(selected)
        if remaining > 0:
            selected.extend(random.choices(all_indices, k=remaining))
        return selected[:target_count]

    def _log_trace_filter_stats(self, stats: dict, selected: List[int], mixed_indices: List[int]):
        mixed_frac = stats["mixed"] / max(stats["seen"], 1)
        mixed_set = set(mixed_indices)
        selected_mixed_count = sum(1 for idx in selected if idx in mixed_set)
        selected_mixed_frac = selected_mixed_count / max(len(selected), 1)
        message = (
            "TRACE mixed filter: "
            f"seen={stats['seen']} mixed={stats['mixed']} all_wrong={stats['all_wrong']} "
            f"all_correct={stats['all_correct']} mixed_frac={mixed_frac:.4f} "
            f"selected={len(selected)} selected_mixed_frac={selected_mixed_frac:.4f}"
        )
        text_logger = getattr(self, "text_logger", None)
        if text_logger is not None:
            text_logger.log(message)
        else:
            print(message)
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None) if logger is not None else None
        if experiment is not None and hasattr(experiment, "add_scalar"):
            step = int(getattr(self, "global_step", 0) or 0)
            experiment.add_scalar("train/trace_filter/mixed_frac", mixed_frac, step)
            experiment.add_scalar("train/trace_filter/selected_mixed_frac", selected_mixed_frac, step)

    @torch.no_grad()
    def rollout(self, questions: List[str], gt_answers) -> grpo.Experience:
        self._last_trace_metrics = {}
        batch_size = len(questions)
        group_size = int(self.model_kwargs.rl_config.group_size)
        experience = super().rollout(questions=questions, gt_answers=gt_answers)
        if self.trace_config.get("trace_resample_mixed_rollout", False):
            experience = self.resample_mixed_rollout(
                questions=questions,
                gt_answers=gt_answers,
                initial_experience=experience,
                batch_size=batch_size,
                group_size=group_size,
            )
        if self.trace_config.get("enable_trace_reward", False):
            experience = self.apply_trace_rewards(
                experience=experience,
                batch_size=batch_size,
                group_size=group_size,
            )
        return experience

    def mixed_rollout_stats(self, experience: grpo.Experience, batch_size: int, group_size: int):
        group_acc = experience.accuracies.view(batch_size, group_size).float()
        pos_counts = group_acc.sum(dim=1)
        mixed = (pos_counts > 0) & (pos_counts < group_size)
        return {
            "mixed_frac": mixed.float().mean().item(),
            "pos_count": pos_counts.float().mean().item(),
        }

    @torch.no_grad()
    def resample_mixed_rollout(
        self,
        questions: List[str],
        gt_answers,
        initial_experience: grpo.Experience,
        batch_size: int,
        group_size: int,
    ) -> grpo.Experience:
        max_attempts = max(1, int(self.trace_config.get("trace_resample_mixed_max_attempts", 4)))
        target_frac = float(self.trace_config.get("trace_resample_mixed_target_frac", 1.0))

        best_experience = initial_experience
        best_stats = self.mixed_rollout_stats(initial_experience, batch_size=batch_size, group_size=group_size)
        attempts = 1
        while attempts < max_attempts and best_stats["mixed_frac"] < target_frac:
            candidate = super().rollout(questions=questions, gt_answers=gt_answers)
            attempts += 1
            candidate_stats = self.mixed_rollout_stats(candidate, batch_size=batch_size, group_size=group_size)
            if candidate_stats["mixed_frac"] > best_stats["mixed_frac"]:
                best_experience = candidate
                best_stats = candidate_stats

        self._last_trace_metrics.update(
            {
                "trace/resample_attempts": torch.tensor(float(attempts), device=self.device),
                "trace/resample_mixed_frac": torch.tensor(best_stats["mixed_frac"], device=self.device),
                "trace/resample_pos_count": torch.tensor(best_stats["pos_count"], device=self.device),
            }
        )
        return best_experience

    @torch.no_grad()
    def apply_trace_rewards(self, experience: grpo.Experience, batch_size: int, group_size: int) -> grpo.Experience:
        cfg = self.trace_config
        latent_mask = experience.latent_attention_mask.float()
        denom = latent_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (experience.latent_inputs_embeds * latent_mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled = F.normalize(pooled.float(), dim=-1)

        answer_mask = experience.answer_attention_mask.float()
        answer_conf = (experience.answer_logprobs.float() * answer_mask).sum(dim=1) / answer_mask.sum(dim=1).clamp_min(1.0)

        shaped_rewards = experience.rewards.float().clone()
        all_advantages = []
        metrics = {
            "trace/bonus": [],
            "trace/pos_count": [],
            "trace/neg_count": [],
            "trace/hard_count": [],
            "trace/pos_neg_sim": [],
            "trace/pos_pos_sim": [],
            "trace/hard_pos_sim": [],
            "trace/raw_pos_neg_sim": [],
            "trace/raw_pos_pos_sim": [],
        }

        reward_weight = float(cfg.get("trace_reward_weight", 0.1))
        pos_neg_weight = float(cfg.get("pos_neg_weight", 1.0))
        hard_neg_weight = float(cfg.get("hard_neg_weight", 1.0))
        pos_pos_diversity_weight = float(cfg.get("pos_pos_diversity_weight", 0.2))
        pos_agreement_weight = float(cfg.get("pos_agreement_weight", 0.1))
        hard_top_frac = float(cfg.get("hard_negative_top_frac", 0.5))
        center_trajectories = bool(cfg.get("trace_center_trajectories", False))

        for sample_idx in range(batch_size):
            start = sample_idx * group_size
            end = start + group_size
            raw_group_h = pooled[start:end]
            if center_trajectories:
                group_h = F.normalize(raw_group_h - raw_group_h.mean(dim=0, keepdim=True), dim=-1)
            else:
                group_h = raw_group_h
            group_acc = experience.accuracies[start:end].view(-1).float()
            group_conf = answer_conf[start:end]
            pos_mask = group_acc > 0.5
            neg_mask = ~pos_mask
            bonus = torch.zeros(group_size, device=group_h.device, dtype=torch.float32)

            pos_count = int(pos_mask.sum().item())
            neg_count = int(neg_mask.sum().item())
            metrics["trace/pos_count"].append(float(pos_count))
            metrics["trace/neg_count"].append(float(neg_count))

            if pos_count > 0 and neg_count > 0:
                pos_h = group_h[pos_mask]
                neg_h = group_h[neg_mask]
                pos_neg_sim = pos_h @ neg_h.T
                raw_pos_neg_sim = raw_group_h[pos_mask] @ raw_group_h[neg_mask].T
                pos_to_neg_dist = (1.0 - pos_neg_sim).mean(dim=1)
                neg_to_pos_sim = pos_neg_sim.mean(dim=0)
                bonus[pos_mask] += pos_neg_weight * pos_to_neg_dist
                bonus[neg_mask] -= pos_neg_weight * (neg_to_pos_sim + 1.0) * 0.5
                metrics["trace/pos_neg_sim"].append(pos_neg_sim.mean().item())
                metrics["trace/raw_pos_neg_sim"].append(raw_pos_neg_sim.mean().item())

                n_hard = max(1, int(round(neg_count * hard_top_frac)))
                neg_conf = group_conf[neg_mask]
                hard_local = torch.topk(neg_conf, k=min(n_hard, neg_count)).indices
                hard_global = neg_mask.nonzero(as_tuple=False).view(-1)[hard_local]
                hard_h = group_h[hard_global]
                hard_pos_sim = hard_h @ pos_h.T
                bonus[hard_global] -= hard_neg_weight * (hard_pos_sim.mean(dim=1) + 1.0) * 0.5
                bonus[pos_mask] += hard_neg_weight * (1.0 - hard_pos_sim).mean(dim=0)
                metrics["trace/hard_count"].append(float(len(hard_global)))
                metrics["trace/hard_pos_sim"].append(hard_pos_sim.mean().item())
            else:
                metrics["trace/pos_neg_sim"].append(0.0)
                metrics["trace/raw_pos_neg_sim"].append(0.0)
                metrics["trace/hard_count"].append(0.0)
                metrics["trace/hard_pos_sim"].append(0.0)

            if pos_count > 1:
                pos_h = group_h[pos_mask]
                pos_pos_sim = pos_h @ pos_h.T
                raw_pos_h = raw_group_h[pos_mask]
                raw_pos_pos_sim = raw_pos_h @ raw_pos_h.T
                off_diag = ~torch.eye(pos_count, dtype=torch.bool, device=pos_pos_sim.device)
                pos_div = (1.0 - pos_pos_sim).masked_fill(~off_diag, 0.0).sum(dim=1) / max(pos_count - 1, 1)
                bonus[pos_mask] += pos_pos_diversity_weight * pos_div
                bonus[pos_mask] += pos_agreement_weight * ((pos_count - 1) / max(group_size - 1, 1))
                metrics["trace/pos_pos_sim"].append(pos_pos_sim[off_diag].mean().item())
                metrics["trace/raw_pos_pos_sim"].append(raw_pos_pos_sim[off_diag].mean().item())
            else:
                metrics["trace/pos_pos_sim"].append(0.0)
                metrics["trace/raw_pos_pos_sim"].append(0.0)

            scaled_bonus = reward_weight * bonus.unsqueeze(1)
            shaped_rewards[start:end] = shaped_rewards[start:end] + scaled_bonus
            metrics["trace/bonus"].append(scaled_bonus.mean().item())
            all_advantages.append(grpo.group_advantages(shaped_rewards[start:end]))

        experience.rewards = shaped_rewards
        experience.advantages = torch.cat(all_advantages, dim=0)
        self._last_trace_metrics.update({
            key: torch.tensor(sum(values) / max(len(values), 1), device=self.device, dtype=torch.float32)
            for key, values in metrics.items()
        })
        return experience
