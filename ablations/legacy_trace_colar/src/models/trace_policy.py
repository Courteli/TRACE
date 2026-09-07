import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .read_stable_efficient import LitREADCoTStableEfficient
from ..modules.readcot import (
    aggregate_step_residuals,
    dependency_bce_loss,
    dependency_f1_score,
    reconstruct_dependency_logits,
)
from ..modules.trace_policy import (
    GaussianTrajectoryPolicy,
    HardPathPair,
    build_teacher_rationale_schedule,
    build_transition_advantages,
    clipped_policy_loss,
    counterfactual_action_batch,
    counterfactual_transition_credits,
    diagonal_gaussian_kl,
    gaussian_log_prob,
    group_standardize,
    mine_question_local_hard_pairs,
    monotone_progress_centers,
    path_noncollapse_loss,
    permutation_invariant_set_matching,
    relation_geometry_loss,
    reorder_teacher_set,
    stochastic_monotone_assignment,
    trajectory_distance,
)


def build_path_bottleneck_mask(
    question_attention_mask: torch.Tensor,
    latent_attention_mask: torch.Tensor,
    answer_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Answer queries can access latent K/V but never original question K/V."""
    if not (
        question_attention_mask.ndim
        == latent_attention_mask.ndim
        == answer_attention_mask.ndim
        == 2
    ):
        raise ValueError("all attention masks must be two-dimensional")
    if not (
        question_attention_mask.shape[0]
        == latent_attention_mask.shape[0]
        == answer_attention_mask.shape[0]
    ):
        raise ValueError("all attention masks must share a batch dimension")
    return torch.cat(
        [
            torch.zeros_like(question_attention_mask),
            latent_attention_mask,
            answer_attention_mask,
        ],
        dim=1,
    )


def _right_pad(
    tensors: Sequence[torch.Tensor],
    *,
    value: float,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("cannot pad an empty tensor sequence")
    max_length = max(tensor.shape[1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        if tensor.shape[1] == max_length:
            padded.append(tensor)
            continue
        shape = list(tensor.shape)
        shape[1] = max_length - tensor.shape[1]
        extension = torch.full(
            shape,
            fill_value=value,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        padded.append(torch.cat([tensor, extension], dim=1))
    return torch.cat(padded, dim=0)


def extract_stage2_policy_reference(checkpoint: dict):
    """Return the immutable prior only from a genuine Stage-2 checkpoint."""
    checkpoint_stage = int(
        checkpoint.get("trace_policy_training_stage", 1)
    )
    if checkpoint_stage != 2:
        return None
    reference = checkpoint.get("trace_stage1_policy_reference")
    if reference is None:
        raise RuntimeError(
            "Stage-2 checkpoint is missing its immutable Stage-1 "
            "policy reference"
        )
    return reference


def summarize_unique_validation_records(
    shards: Sequence[Sequence[Tuple[int, float, int]]],
    *,
    expected_count: int,
) -> Dict[str, float]:
    """Deduplicate DDP padding and require one deterministic result per item."""
    records: Dict[int, Tuple[float, int]] = {}
    for shard in shards:
        for index, accuracy, output_length in shard:
            value = (float(accuracy), int(output_length))
            if index in records and records[index] != value:
                raise RuntimeError(
                    f"validation result for index {index} is nondeterministic"
                )
            records[int(index)] = value
    if len(records) != int(expected_count):
        raise RuntimeError(
            "full validation contract failed: "
            f"found {len(records)} unique questions, "
            f"expected {expected_count}"
        )
    return {
        "accuracy": float(
            np.mean([value[0] for value in records.values()])
        ),
        "output_length": float(
            np.mean([value[1] for value in records.values()])
        ),
        "unique_questions": float(len(records)),
    }


class LitTRACEPolicy(LitREADCoTStableEfficient):
    """TRACE with stochastic trajectory formation and trajectory-policy RL.

    Stage 1 samples exchangeable paths from an autoregressive Gaussian policy
    and matches them to a stochastic multi-rationale teacher set. Stage 2
    treats each latent action as the optimized policy action and assigns
    transition-specific credit by recomputing causal counterfactual suffixes.
    """

    path_adapter_name = "default"
    teacher_adapter_name = "trace_teacher"
    answer_adapter_name = "trace_answer"

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            all_config=all_config,
        )
        self.trace_config = model_kwargs.get("trace_policy_config", {})
        self.trace_rl_config = model_kwargs.get("trace_rl_config", {})
        self.do_trace_rl = bool(model_kwargs.get("do_trace_rl", False))
        for unused_module in (
            self.latent_bridge,
            self.residual_projector,
        ):
            for parameter in unused_module.parameters():
                parameter.requires_grad_(False)
        if not self.model_kwargs.get("do_lora", False):
            raise ValueError(
                "TRACE requires LoRA to isolate teacher, path, and answer roles"
            )
        if self.teacher_adapter_name not in self.llm.peft_config:
            teacher_config = copy.deepcopy(
                self.llm.peft_config[self.path_adapter_name]
            )
            self.llm.add_adapter(
                self.teacher_adapter_name,
                teacher_config,
            )
        self._match_adapter_storage_to_path(self.teacher_adapter_name)
        self._set_adapter_parameter_trainability()
        self.n_trace_steps = int(self.readcot_config.n_latents)
        if self.n_trace_steps <= 0:
            raise ValueError("TRACE requires at least one latent transition")
        if self.readcot_config.get("implicit_latent_mode") == "block":
            # TRACE's action distribution is autoregressive even if the source
            # BRIDGE config used a block latent implementation.
            self.readcot_config.implicit_latent_mode = "autoregressive-policy"
        if self.readcot_config.get("use_hybrid", False):
            raise ValueError("TRACE policy uses a latent-only answer bottleneck")
        if self.readcot_config.get("use_anchor_loss", False):
            raise ValueError("TRACE policy does not decode explicit CoT anchors")
        if self.readcot_config.get("use_anchor_gate", False):
            raise ValueError("TRACE policy does not use a route gate")
        prohibited = (
            "use_trace_view_embeddings",
            "use_trace_step_view_embeddings",
            "max_trace_views",
        )
        for key in prohibited:
            if key in self.trace_config:
                raise ValueError(
                    f"{key} belongs to the rejected fixed-view design"
                )

        self.trajectory_policy = GaussianTrajectoryPolicy(
            hidden_size=self.hidden_size,
            action_dim=int(self.trace_config.get("action_dim", 16)),
            n_steps=self.n_trace_steps,
            policy_hidden_size=int(
                self.trace_config.get("policy_hidden_size", 512)
            ),
            step_embedding_size=int(
                self.trace_config.get("policy_step_embedding_size", 64)
            ),
            initial_log_std=float(
                self.trace_config.get("initial_log_std", -0.7)
            ),
            min_log_std=float(
                self.trace_config.get("min_log_std", -2.5)
            ),
            max_log_std=float(
                self.trace_config.get("max_log_std", 0.5)
            ),
        )
        self.stage1_policy_reference = copy.deepcopy(
            self.trajectory_policy
        )
        for parameter in self.stage1_policy_reference.parameters():
            parameter.requires_grad_(False)

        self.teacher_progress_logits = torch.nn.Parameter(
            torch.zeros(self.n_trace_steps)
        )
        self._reference_restored = False
        self._loaded_stage2_state = False
        self._teacher_adapter_loaded = False
        self._stage2_initialized = False
        self._last_trace_metrics: Dict[str, torch.Tensor] = {}
        self._trace_visual_records: List[dict] = []
        self._validation_question_records: List[
            Tuple[int, float, int]
        ] = []
        self.strict_loading = False

        if self.do_trace_rl:
            required_stage2_objectives = (
                "use_trajectory_policy_loss",
                "use_answer_policy_loss",
            )
            disabled = [
                key
                for key in required_stage2_objectives
                if not bool(self.trace_rl_config.get(key, False))
            ]
            if disabled:
                raise ValueError(
                    "Final TRACE Stage 2 requires both latent-action and "
                    f"answer-token policy objectives; disabled: {disabled}"
                )
            if int(
                self.trace_rl_config.get("policy_update_epochs", 2)
            ) < 2:
                raise ValueError(
                    "Final TRACE requires at least two policy updates per "
                    "rollout so PPO clipping is operational"
                )
            self._initialize_stage2_modules()
            self.automatic_optimization = False

    def _initialize_stage2_modules(self):
        if not self.model_kwargs.get("do_lora", False):
            raise ValueError(
                "Stage 2 answer-token GRPO requires phase-isolated LoRA"
            )
        if self.answer_adapter_name not in self.llm.peft_config:
            adapter_config = copy.deepcopy(
                self.llm.peft_config[self.path_adapter_name]
            )
            self.llm.add_adapter(
                self.answer_adapter_name,
                adapter_config,
            )
        self._match_adapter_storage_to_path(self.answer_adapter_name)

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.trajectory_policy.set_stage2_trainability()
        self._set_adapter_parameter_trainability()
        self._activate_answer_adapter()

    def _set_adapter_parameter_trainability(self):
        path_marker = f".{self.path_adapter_name}."
        teacher_marker = f".{self.teacher_adapter_name}."
        answer_marker = f".{self.answer_adapter_name}."
        for name, parameter in self.llm.named_parameters():
            if teacher_marker in name:
                parameter.requires_grad_(False)
            elif path_marker in name:
                parameter.requires_grad_(not self.do_trace_rl)
            elif answer_marker in name:
                parameter.requires_grad_(self.do_trace_rl)

    def _activate_path_adapter(self):
        self.llm.set_adapter(self.path_adapter_name)
        self._set_adapter_parameter_trainability()

    def _activate_teacher_adapter(self):
        self.llm.set_adapter(self.teacher_adapter_name)
        self._set_adapter_parameter_trainability()

    def _activate_answer_adapter(self):
        if not self.do_trace_rl:
            return
        self.llm.set_adapter(self.answer_adapter_name)
        self._set_adapter_parameter_trainability()

    def _copy_path_adapter(self, target_adapter_name: str):
        self._match_adapter_storage_to_path(target_adapter_name)
        parameters = dict(self.llm.named_parameters())
        path_marker = f".{self.path_adapter_name}."
        target_marker = f".{target_adapter_name}."
        copied = 0
        with torch.no_grad():
            for name, value in list(parameters.items()):
                if path_marker not in name:
                    continue
                target_name = name.replace(path_marker, target_marker)
                if (
                    target_name in parameters
                    and parameters[target_name].shape == value.shape
                ):
                    parameters[target_name].copy_(value.detach())
                    copied += 1
        if copied == 0:
            raise RuntimeError(
                f"could not map the path adapter to {target_adapter_name}"
            )
        return copied

    def _match_adapter_storage_to_path(self, target_adapter_name: str):
        """Match added PEFT adapters to the Stage-0 path adapter dtype."""
        parameters = dict(self.llm.named_parameters())
        path_marker = f".{self.path_adapter_name}."
        target_marker = f".{target_adapter_name}."
        matched = 0
        with torch.no_grad():
            for name, source in list(parameters.items()):
                if path_marker not in name:
                    continue
                target_name = name.replace(path_marker, target_marker)
                target = parameters.get(target_name)
                if target is None or target.shape != source.shape:
                    continue
                if (
                    target.dtype != source.dtype
                    or target.device != source.device
                ):
                    target.data = target.data.to(
                        device=source.device,
                        dtype=source.dtype,
                    )
                matched += 1
        if matched == 0:
            raise RuntimeError(
                f"could not align adapter storage for {target_adapter_name}"
            )
        return matched

    def _copy_path_adapter_to_answer_adapter(self):
        if self.do_trace_rl:
            self._copy_path_adapter(self.answer_adapter_name)

    def _copy_path_adapter_to_teacher_adapter(self):
        self._copy_path_adapter(self.teacher_adapter_name)
        self._teacher_adapter_loaded = True
        self._set_adapter_parameter_trainability()

    def _snapshot_stage1_policy(self):
        self.stage1_policy_reference.load_state_dict(
            self.trajectory_policy.state_dict(),
            strict=True,
        )
        self.stage1_policy_reference.eval()
        for parameter in self.stage1_policy_reference.parameters():
            parameter.requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        answer_marker = f".{self.answer_adapter_name}."
        teacher_marker = f".{self.teacher_adapter_name}."
        self._loaded_stage2_state = any(
            answer_marker in name for name in state_dict
        )
        self._teacher_adapter_loaded = any(
            teacher_marker in name for name in state_dict
        )
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    def on_load_checkpoint(self, checkpoint):
        checkpoint_stage = int(
            checkpoint.get("trace_policy_training_stage", 1)
        )
        reference = extract_stage2_policy_reference(checkpoint)
        self._reference_restored = False
        if reference is not None:
            self.stage1_policy_reference.load_state_dict(
                reference,
                strict=True,
            )
            self._reference_restored = True
        self._loaded_stage2_state = checkpoint_stage == 2
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        full_state = self.state_dict()
        preserve_prefixes = (
            "trajectory_policy.",
            "teacher_progress_logits",
            "step_compressor.",
            "latent_relation.",
            "state_norm.",
        )
        path_marker = f".{self.path_adapter_name}."
        teacher_marker = f".{self.teacher_adapter_name}."
        for name, value in full_state.items():
            if (
                name.startswith(preserve_prefixes)
                or path_marker in name
                or teacher_marker in name
            ):
                checkpoint["state_dict"][name] = value
        checkpoint_stage = 2 if self.do_trace_rl else 1
        checkpoint["trace_policy_training_stage"] = checkpoint_stage
        if checkpoint_stage == 2:
            checkpoint["trace_stage1_policy_reference"] = {
                name: value.detach().cpu()
                for name, value in (
                    self.stage1_policy_reference.state_dict().items()
                )
            }
        else:
            # A Stage-1 checkpoint must never publish the constructor-time
            # reference as if it were the trained Stage-1 policy.
            checkpoint.pop("trace_stage1_policy_reference", None)

    def on_fit_start(self):
        if not self._teacher_adapter_loaded:
            if self.do_trace_rl:
                raise RuntimeError(
                    "Stage 2 requires a Stage-1 checkpoint containing the "
                    "frozen CoT teacher adapter"
                )
            self._copy_path_adapter_to_teacher_adapter()
        if self.do_trace_rl:
            if self._loaded_stage2_state and not self._reference_restored:
                raise RuntimeError(
                    "A Stage-2 state was loaded without its immutable Stage-1 "
                    "policy reference. Resume from the full Lightning "
                    "checkpoint instead of loading weights only."
                )
            if not self._loaded_stage2_state:
                self._copy_path_adapter_to_answer_adapter()
            if not self._reference_restored:
                self._snapshot_stage1_policy()
            self._stage2_initialized = True
            self._activate_answer_adapter()
            self._validate_trace_rl_epoch_budget()
        return super().on_fit_start()

    def _validate_trace_rl_epoch_budget(self):
        """Require full-split four-rank training without dataset mutation."""
        target_unique = int(
            self.trace_rl_config.get(
                "n_train_samples_per_epoch",
                6726,
            )
        )
        limit_batches = self.trainer.limit_train_batches
        if isinstance(limit_batches, bool) or not isinstance(
            limit_batches,
            int,
        ):
            raise RuntimeError(
                "Stage 2 requires an integer trainer.limit_train_batches"
            )
        local_batch_size = int(self.all_config.dataloader.batch_size)
        world_size = int(self.trainer.world_size)
        synchronized_batch_size = local_batch_size * world_size
        expected_batches = math.ceil(
            target_unique / float(synchronized_batch_size)
        )
        if int(limit_batches) != expected_batches:
            raise RuntimeError(
                "Stage-2 full-split batch mismatch: "
                f"limit_train_batches={limit_batches}, expected "
                f"ceil({target_unique}/{synchronized_batch_size})="
                f"{expected_batches}"
            )
        realized = int(limit_batches) * local_batch_size * world_size
        padding = realized - target_unique
        if padding < 0 or padding >= synchronized_batch_size:
            raise RuntimeError(
                "Stage-2 DDP padding mismatch: "
                f"{limit_batches} batches x {local_batch_size} local batch "
                f"x {world_size} ranks = {realized}, target unique questions "
                f"={target_unique}"
            )
        if int(self.trainer.accumulate_grad_batches) != 1:
            raise RuntimeError(
                "Stage-2 manual policy updates require "
                "accumulate_grad_batches=1"
            )
        all_indices = list(
            self.trainer.datamodule.get_all_train_indices()
        )
        train_set = self.trainer.datamodule.train_set
        if len(all_indices) != target_unique:
            raise RuntimeError(
                "Stage-2 configured full-split size does not match the "
                f"dataset: config={target_unique}, dataset={len(all_indices)}"
            )
        if len(train_set) != len(all_indices):
            raise RuntimeError(
                "Stage 2 must sample from the full training split; "
                "do not mutate dataset indices to impose the epoch budget"
            )

    def training_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=0,
    ):
        if self.do_trace_rl:
            if not self._stage2_initialized:
                raise RuntimeError("Stage 2 policy reference was not initialized")
            return self.trace_rl_training_step(
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=dataloader_idx,
            )
        return super().training_step(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def _decode_rationale_sets(self, batch) -> List[List[dict]]:
        gold_steps = self._decode_step_lists(batch)
        gold_dependencies = self._decode_cached_matrices(
            batch,
            "dependency_matrix",
        )
        gold_confidences = self._decode_cached_matrices(
            batch,
            "confidence_matrix",
        )
        raw_sets = batch.get("rationale_set_json")
        decoded = []
        for sample_index, steps in enumerate(gold_steps):
            fallback = {
                "steps": list(steps),
                "dependency_matrix": gold_dependencies[sample_index],
                "confidence_matrix": gold_confidences[sample_index],
                "fingerprint": "gold_fallback",
                "source": "gold",
                "verified": True,
            }
            if raw_sets is None:
                decoded.append([fallback])
                continue
            try:
                candidates = json.loads(raw_sets[sample_index])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid rationale_set_json at sample {sample_index}"
                ) from exc
            valid = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                steps_value = candidate.get("steps")
                if (
                    candidate.get("verified") is not True
                    or not isinstance(steps_value, list)
                    or not steps_value
                ):
                    continue
                valid.append(
                    {
                        "steps": [str(step) for step in steps_value],
                        "dependency_matrix": candidate.get(
                            "dependency_matrix"
                        ),
                        "confidence_matrix": candidate.get(
                            "confidence_matrix"
                        ),
                        "fingerprint": str(
                            candidate.get("fingerprint", "unknown")
                        ),
                        "source": str(
                            candidate.get("source", "unknown")
                        ),
                        "verified": True,
                    }
                )
            decoded.append(valid or [fallback])
        return decoded

    def _distance_kwargs(self) -> dict:
        return {
            "anchor_count": int(
                self.trace_config.get("path_anchor_count", 3)
            ),
            "position_weight": float(
                self.trace_config.get("path_position_weight", 0.45)
            ),
            "direction_weight": float(
                self.trace_config.get("path_direction_weight", 0.35)
            ),
            "step_weight": float(
                self.trace_config.get("path_step_weight", 0.15)
            ),
        }

    def _trajectory_latents(
        self,
        questions: Sequence[str],
        *,
        deterministic: bool = False,
        innovations: Optional[torch.Tensor] = None,
        forced_actions: Optional[torch.Tensor] = None,
        forced_action_mask: Optional[torch.Tensor] = None,
        compute_reference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate one causal latent trajectory per question."""
        self._activate_path_adapter()
        batch_size = len(questions)
        action_dim = self.trajectory_policy.action_dim
        expected = (batch_size, self.n_trace_steps, action_dim)
        if innovations is not None and tuple(innovations.shape) != expected:
            raise ValueError(
                f"innovations have shape {tuple(innovations.shape)}, "
                f"expected {expected}"
            )
        if forced_actions is not None and tuple(forced_actions.shape) != expected:
            raise ValueError(
                f"forced_actions have shape {tuple(forced_actions.shape)}, "
                f"expected {expected}"
            )
        if forced_actions is not None and forced_action_mask is None:
            forced_action_mask = torch.ones(
                batch_size,
                self.n_trace_steps,
                device=self.device,
                dtype=torch.bool,
            )
        if forced_action_mask is not None and tuple(
            forced_action_mask.shape
        ) != (batch_size, self.n_trace_steps):
            raise ValueError("forced_action_mask has an invalid shape")

        question_ids, question_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.thinking_separator,
        )
        question_embeds = self.embedding(question_ids)
        question_outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=question_mask,
            position_ids=self.make_position_ids_for_current_input(
                question_mask,
                question_embeds.shape[1],
            ),
            output_hidden_states=True,
            use_cache=True,
        )
        cache = question_outputs.past_key_values
        context_mask = question_mask
        previous_state = self.state_norm(
            question_outputs.hidden_states[-1][:, -1, :]
        )
        latent_inputs = []
        latent_states = []
        residuals = []
        actions = []
        sampled_innovations = []
        log_probs = []
        means = []
        log_stds = []
        reference_means = []
        reference_log_stds = []
        action_scale = float(
            self.trace_config.get("action_embedding_scale", 0.10)
        )
        step_scale = float(
            self.trace_config.get("dynamics_step_scale", 0.10)
        )

        for step_index in range(self.n_trace_steps):
            mean, log_std = self.trajectory_policy.distribution_parameters(
                previous_state,
                step_index,
            )
            std = torch.exp(log_std)
            if innovations is None:
                epsilon = (
                    torch.zeros_like(mean)
                    if deterministic
                    else torch.randn_like(mean)
                )
            else:
                epsilon = innovations[:, step_index].to(mean.dtype)
            sampled_action = mean + std * epsilon
            if forced_actions is not None:
                mask = forced_action_mask[:, step_index].unsqueeze(-1)
                action = torch.where(
                    mask,
                    forced_actions[:, step_index].to(mean.dtype).detach(),
                    sampled_action,
                )
                realized_epsilon = torch.where(
                    mask,
                    (action - mean) / std.clamp_min(1e-8),
                    epsilon,
                )
            else:
                action = sampled_action
                realized_epsilon = epsilon
            current_input = self.trajectory_policy.latent_input(
                previous_state,
                action,
                step_index,
                action_scale=action_scale,
                step_scale=step_scale,
            ).to(question_embeds.dtype)
            current_mask = torch.ones(
                batch_size,
                1,
                device=self.device,
                dtype=context_mask.dtype,
            )
            context_mask = torch.cat([context_mask, current_mask], dim=1)
            outputs = self.llm.forward(
                inputs_embeds=current_input.unsqueeze(1),
                attention_mask=context_mask,
                position_ids=self.make_position_ids_for_current_input(
                    context_mask,
                    1,
                ),
                past_key_values=cache,
                output_hidden_states=True,
                use_cache=True,
            )
            cache = outputs.past_key_values
            current_state = self.state_norm(
                outputs.hidden_states[-1][:, -1, :]
            )
            latent_inputs.append(current_input)
            latent_states.append(current_state)
            residuals.append(current_state - previous_state)
            actions.append(action)
            sampled_innovations.append(realized_epsilon)
            log_probs.append(gaussian_log_prob(action, mean, log_std))
            means.append(mean)
            log_stds.append(log_std)
            if compute_reference:
                with torch.no_grad():
                    ref_mean, ref_log_std = (
                        self.stage1_policy_reference
                        .distribution_parameters(
                            previous_state.detach(),
                            step_index,
                        )
                    )
                reference_means.append(ref_mean)
                reference_log_stds.append(ref_log_std)
            previous_state = current_state

        latent_inputs_tensor = torch.stack(latent_inputs, dim=1)
        latent_mask = torch.ones(
            batch_size,
            self.n_trace_steps,
            device=self.device,
            dtype=question_mask.dtype,
        )
        result = {
            "question_input_ids": question_ids,
            "question_attention_mask": question_mask,
            "question_inputs_embeds": question_embeds,
            "latent_inputs_embeds": latent_inputs_tensor,
            "latent_attention_mask": latent_mask,
            "context_inputs_embeds": torch.cat(
                [question_embeds, latent_inputs_tensor],
                dim=1,
            ),
            "context_attention_mask": context_mask,
            "past_key_values": cache,
            "latent_states": torch.stack(latent_states, dim=1),
            "implicit_residuals": torch.stack(residuals, dim=1),
            "actions": torch.stack(actions, dim=1),
            "innovations": torch.stack(sampled_innovations, dim=1),
            "action_log_probs": torch.stack(log_probs, dim=1),
            "action_means": torch.stack(means, dim=1),
            "action_log_stds": torch.stack(log_stds, dim=1),
        }
        if compute_reference:
            result["reference_action_means"] = torch.stack(
                reference_means,
                dim=1,
            )
            result["reference_action_log_stds"] = torch.stack(
                reference_log_stds,
                dim=1,
            )
        return result

    @staticmethod
    def _bottleneck_context_mask(
        trajectory_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros_like(
                    trajectory_outputs["question_attention_mask"]
                ),
                trajectory_outputs["latent_attention_mask"],
            ],
            dim=1,
        )

    def _teacher_force_bottleneck(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        target_texts: Sequence[str],
    ) -> torch.Tensor:
        self._activate_answer_adapter()
        return self._teacher_force_target(
            past_key_values=trajectory_outputs["past_key_values"],
            context_attention_mask=self._bottleneck_context_mask(
                trajectory_outputs
            ),
            target_texts=target_texts,
        )

    def _prompt_ids_for_answer(self, batch_size: int):
        prompt_ids = torch.full(
            (batch_size, 1),
            fill_value=self.thinking_separator_id,
            device=self.device,
            dtype=torch.long,
        )
        return prompt_ids, torch.ones_like(prompt_ids)

    @staticmethod
    def _resolve_latent_read_mask(
        trajectory_outputs: Dict[str, torch.Tensor],
        latent_read_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        available = trajectory_outputs["latent_attention_mask"]
        if latent_read_mask is None:
            return available
        if tuple(latent_read_mask.shape) != tuple(available.shape):
            raise ValueError(
                "latent_read_mask must have shape "
                f"{tuple(available.shape)}, got "
                f"{tuple(latent_read_mask.shape)}"
            )
        return latent_read_mask.to(
            device=available.device,
            dtype=available.dtype,
        ) * available

    @staticmethod
    def _fork_past_key_values(past_key_values):
        """Create a mutable cache shell over the same read-only prefix K/V."""
        try:
            return type(past_key_values)(list(past_key_values))
        except (TypeError, ValueError):
            return copy.deepcopy(past_key_values)

    def _generate_answers_from_trajectory(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        *,
        do_sample: bool,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        latent_read_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._activate_answer_adapter()
        batch_size = trajectory_outputs["latent_states"].shape[0]
        prompt_ids, prompt_mask = self._prompt_ids_for_answer(batch_size)
        prompt_embeds = self.embedding(prompt_ids)
        full_embeddings = torch.cat(
            [
                trajectory_outputs["context_inputs_embeds"],
                prompt_embeds,
            ],
            dim=1,
        )
        attention_mask = build_path_bottleneck_mask(
            trajectory_outputs["question_attention_mask"],
            self._resolve_latent_read_mask(
                trajectory_outputs,
                latent_read_mask,
            ),
            prompt_mask,
        )
        generation_config = dict(self._get_generation_config())
        generation_config["do_sample"] = bool(do_sample)
        if do_sample:
            generation_config["temperature"] = float(
                temperature if temperature is not None else 0.95
            )
            generation_config["top_p"] = float(
                top_p if top_p is not None else 0.97
            )
        return self.llm.generate(
            inputs_embeds=full_embeddings,
            attention_mask=attention_mask,
            past_key_values=self._fork_past_key_values(
                trajectory_outputs["past_key_values"]
            ),
            **generation_config,
        )

    def _answer_token_log_probs(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        latent_read_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._activate_answer_adapter()
        batch_size = answer_input_ids.shape[0]
        prompt_ids, prompt_mask = self._prompt_ids_for_answer(batch_size)
        current_ids = torch.cat([prompt_ids, answer_input_ids], dim=1)
        current_mask = torch.cat(
            [prompt_mask, answer_attention_mask],
            dim=1,
        )
        attention_mask = build_path_bottleneck_mask(
            trajectory_outputs["question_attention_mask"],
            self._resolve_latent_read_mask(
                trajectory_outputs,
                latent_read_mask,
            ),
            current_mask,
        )
        outputs = self.llm.forward(
            input_ids=current_ids,
            attention_mask=attention_mask,
            past_key_values=self._fork_past_key_values(
                trajectory_outputs["past_key_values"]
            ),
            output_hidden_states=False,
        )
        answer_length = answer_input_ids.shape[1]
        logits = outputs.logits[
            :,
            prompt_ids.shape[1] - 1 : prompt_ids.shape[1] - 1 + answer_length,
            :,
        ]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = answer_input_ids.reshape(-1)
        selected = []
        for start in range(0, flat_logits.shape[0], 16):
            end = min(start + 16, flat_logits.shape[0])
            selected.append(
                F.log_softmax(flat_logits[start:end], dim=-1)
                .gather(
                    dim=-1,
                    index=flat_targets[start:end].unsqueeze(-1),
                )
                .squeeze(-1)
            )
        log_probs = torch.cat(selected, dim=0).reshape_as(
            answer_input_ids
        )
        return torch.nan_to_num(
            log_probs,
            nan=-30.0,
            neginf=-30.0,
            posinf=30.0,
        ).clamp(min=-30.0, max=30.0)

    def _gold_answer_scores(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        answers: Sequence[str],
        latent_read_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_ids, target_mask = self._prepare_raw_texts(
            [self.answer_template.format(answer) for answer in answers],
            padding_side="right",
            suffix=self.tokenizer.eos_token,
        )
        log_probs = self._answer_token_log_probs(
            trajectory_outputs,
            target_ids,
            target_mask,
            latent_read_mask=latent_read_mask,
        )
        return (
            log_probs * target_mask
        ).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)

    def _stochastic_teacher_view(
        self,
        explicit_item: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        states = explicit_item["step_states"].float()
        hidden_size = states.shape[-1]
        queries = self.step_compressor.query_proj(
            self.step_compressor.latent_queries[: self.n_trace_steps]
        )
        keys = self.step_compressor.key_proj(states)
        values = self.step_compressor.value_proj(states)
        semantic_scores = (
            queries @ keys.transpose(0, 1)
        ) / math.sqrt(hidden_size)
        noise = torch.randn(
            self.n_trace_steps,
            device=states.device,
            dtype=torch.float32,
        )
        centers = monotone_progress_centers(
            self.teacher_progress_logits.float(),
            noise,
            noise_scale=float(
                self.trace_config.get(
                    "teacher_progress_noise_scale",
                    0.55,
                )
            ),
        ).to(semantic_scores.dtype)
        assignment = stochastic_monotone_assignment(
            semantic_scores,
            centers,
            sigma=float(
                self.trace_config.get("teacher_progress_sigma", 0.20)
            ),
            progress_strength=float(
                self.trace_config.get(
                    "teacher_progress_strength",
                    1.0,
                )
            ),
        )
        compressed = assignment @ values
        relation_logits, relation_probs = self.latent_relation.forward_probs(
            compressed
        )
        dependency_logits, dependency_probs = reconstruct_dependency_logits(
            assignment=assignment,
            relation_probs=relation_probs,
            center_relations=self.readcot_config.get(
                "center_relation_probs",
                True,
            ),
        )
        dependency_loss = dependency_bce_loss(
            dependency_logits,
            explicit_item["dependency_matrix"],
            confidence=explicit_item.get("confidence_matrix"),
            pos_weight_max=self.readcot_config.get(
                "dep_pos_weight_max",
                None,
            ),
            loss_clamp=self.readcot_config.get("dep_loss_clamp", None),
        )
        dependency_f1 = dependency_f1_score(
            dep_probs=dependency_probs,
            gold=explicit_item["dependency_matrix"],
            threshold=self.readcot_config.get(
                "dependency_threshold",
                0.5,
            ),
        )
        return {
            "path": aggregate_step_residuals(
                assignment,
                explicit_item["step_residuals"].float(),
            ),
            "assignment": assignment,
            "progress_centers": centers,
            "dependency_loss": dependency_loss,
            "dependency_f1": dependency_f1,
            "relation_probs": relation_probs,
        }

    def _build_teacher_set(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        rationale_sets: Sequence[Sequence[dict]],
    ) -> Dict[str, torch.Tensor]:
        self._activate_teacher_adapter()
        set_size = int(
            self.trace_config.get("stage1_teacher_set_size", 4)
        )
        max_modes = int(
            self.trace_config.get("stage1_max_semantic_modes", 2)
        )
        schedules = []
        semantic_modes = []
        selected_keys = []
        for sample_index, rationales in enumerate(rationale_sets):
            schedule, local_modes = build_teacher_rationale_schedule(
                n_available=len(rationales),
                set_size=set_size,
                max_semantic_modes=max_modes,
            )
            schedules.append(schedule)
            semantic_modes.append(local_modes)
            for rationale_index in sorted(set(schedule)):
                selected_keys.append((sample_index, rationale_index))

        explicit_questions = []
        explicit_steps = []
        explicit_answers = []
        explicit_dependencies = []
        explicit_confidences = []
        for sample_index, rationale_index in selected_keys:
            rationale = rationale_sets[sample_index][rationale_index]
            explicit_questions.append(questions[sample_index])
            explicit_steps.append(rationale["steps"])
            explicit_answers.append(answers[sample_index])
            explicit_dependencies.append(
                rationale.get("dependency_matrix")
            )
            explicit_confidences.append(
                rationale.get("confidence_matrix")
            )
        explicit_features = self._collect_explicit_batch_features(
            questions=explicit_questions,
            step_lists=explicit_steps,
            answers=explicit_answers,
            dependency_matrices=explicit_dependencies,
            confidence_matrices=explicit_confidences,
        )
        self._activate_path_adapter()
        feature_map = {
            key: feature for key, feature in zip(selected_keys, explicit_features)
        }

        paths = []
        assignments = []
        centers = []
        relation_probs = []
        dependency_losses = []
        dependency_f1s = []
        for sample_index, schedule in enumerate(schedules):
            sample_paths = []
            sample_assignments = []
            sample_centers = []
            sample_relations = []
            for rationale_index in schedule:
                view = self._stochastic_teacher_view(
                    feature_map[(sample_index, rationale_index)]
                )
                sample_paths.append(view["path"])
                sample_assignments.append(view["assignment"])
                sample_centers.append(view["progress_centers"])
                sample_relations.append(view["relation_probs"])
                dependency_losses.append(view["dependency_loss"])
                dependency_f1s.append(view["dependency_f1"])
            paths.append(torch.stack(sample_paths, dim=0))
            assignments.append(sample_assignments)
            centers.append(torch.stack(sample_centers, dim=0))
            relation_probs.append(sample_relations)

        return {
            "paths": torch.stack(paths, dim=0),
            "assignments": assignments,
            "progress_centers": torch.stack(centers, dim=0),
            "relation_probs": relation_probs,
            "semantic_mode_ids": torch.tensor(
                semantic_modes,
                device=self.device,
                dtype=torch.long,
            ),
            "dependency_loss": torch.stack(dependency_losses).mean(),
            "dependency_f1": torch.stack(dependency_f1s).mean(),
            "multi_rationale_fraction": torch.tensor(
                float(
                    sum(len(rationales) >= 2 for rationales in rationale_sets)
                )
                / float(max(1, len(rationale_sets))),
                device=self.device,
            ),
        }

    def forward(self, batch):
        """Stage 1: IID path set matched to stochastic multi-rationale teachers."""
        if self.do_trace_rl:
            raise RuntimeError(
                "Stage-1 forward is not used as replay during Stage 2"
            )
        questions = list(batch["question"])
        answers = list(batch["answer"])
        rationale_sets = self._decode_rationale_sets(batch)
        teacher_set = self._build_teacher_set(
            questions,
            answers,
            rationale_sets,
        )
        set_size = teacher_set["paths"].shape[1]
        repeated_questions = [
            question
            for question in questions
            for _ in range(set_size)
        ]
        model_outputs = self._trajectory_latents(
            repeated_questions,
            deterministic=False,
        )
        model_paths = model_outputs["implicit_residuals"].view(
            len(questions),
            set_size,
            self.n_trace_steps,
            self.hidden_size,
        )
        teacher_paths = teacher_set["paths"].detach()
        matching = permutation_invariant_set_matching(
            model_paths,
            teacher_paths,
            distance_kwargs=self._distance_kwargs(),
        )
        matched_teachers = reorder_teacher_set(
            teacher_paths,
            matching["assignments"],
        )
        relation = relation_geometry_loss(
            model_paths,
            matched_teachers,
            distance_kwargs=self._distance_kwargs(),
        )
        noncollapse = path_noncollapse_loss(
            model_paths,
            margin=float(
                self.trace_config.get("path_noncollapse_margin", 0.02)
            ),
        )
        answer_targets = [
            self.answer_template.format(answer)
            for answer in answers
            for _ in range(set_size)
        ]
        answer_loss = self._teacher_force_bottleneck(
            model_outputs,
            answer_targets,
        )
        relation_weight = float(
            self.trace_config.get("stage1_relation_weight", 0.25)
        )
        noncollapse_weight = float(
            self.trace_config.get("stage1_noncollapse_weight", 0.05)
        )
        formation = (
            matching["loss"]
            + relation_weight * relation["loss"]
            + noncollapse_weight * noncollapse
        )
        formation_weight = float(
            self.trace_config.get("stage1_formation_weight", 0.14)
        )
        dependency_weight = self._scheduled_loss_weight(
            "dep",
            self.readcot_config.get("lambda_dep", 0.04),
        )
        total_loss = (
            answer_loss
            + formation_weight * formation
            + dependency_weight * teacher_set["dependency_loss"]
        )
        policy_std = torch.exp(
            model_outputs["action_log_stds"].float()
        )
        return {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "dep_loss": teacher_set["dependency_loss"],
            "dep_f1": teacher_set["dependency_f1"],
            "trace_stage1_formation_loss": formation,
            "trace_stage1_set_matching_loss": matching["loss"],
            "trace_stage1_position_loss": matching["position"],
            "trace_stage1_direction_loss": matching["direction"],
            "trace_stage1_step_loss": matching["step"],
            "trace_stage1_relation_loss": relation["loss"],
            "trace_stage1_noncollapse_loss": noncollapse,
            "trace_stage1_model_pair_distance": relation[
                "model_distance"
            ],
            "trace_stage1_teacher_pair_distance": relation[
                "teacher_distance"
            ],
            "trace_stage1_policy_std": policy_std.mean().detach(),
            "trace_stage1_policy_std_min": policy_std.min().detach(),
            "trace_stage1_multi_rationale_fraction": teacher_set[
                "multi_rationale_fraction"
            ],
            "trace_stage1_teacher_progress_span": (
                teacher_set["progress_centers"][..., -1]
                - teacher_set["progress_centers"][..., 0]
            ).mean().detach(),
            "trace_bottleneck_question_access": total_loss.new_zeros(()),
            "lambda_trace_formation_eff": total_loss.new_tensor(
                formation_weight
            ),
            "lambda_dep_eff": total_loss.new_tensor(dependency_weight),
        }

    def _rollout_micro_batch_size(self) -> int:
        return max(
            1,
            int(self.trace_rl_config.get("rollout_micro_batch_size", 1)),
        )

    def _optimization_micro_batch_size(self) -> int:
        return max(
            1,
            int(self.trace_rl_config.get("exp_batch_size", 1)),
        )

    def _answers_to_accuracy(
        self,
        output_ids: torch.Tensor,
        answers: Sequence[str],
    ) -> torch.Tensor:
        output_strings = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        accuracy = []
        for output, answer in zip(output_strings, answers):
            prediction = self.extract_answer_from_output(output)
            accuracy.append(
                self.verify_answer(
                    gt_answer=answer,
                    pred_answer=prediction,
                )
            )
        return torch.tensor(
            accuracy,
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def _score_fixed_action_paths(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        scores = []
        micro_batch = self._rollout_micro_batch_size()
        full_mask = torch.ones(
            actions.shape[:2],
            device=actions.device,
            dtype=torch.bool,
        )
        for start in range(0, len(questions), micro_batch):
            end = min(start + micro_batch, len(questions))
            trajectory = self._trajectory_latents(
                questions[start:end],
                forced_actions=actions[start:end],
                forced_action_mask=full_mask[start:end],
            )
            scores.append(
                self._gold_answer_scores(
                    trajectory,
                    answers[start:end],
                )
            )
            del trajectory
        return torch.cat(scores, dim=0)

    @torch.no_grad()
    def _score_counterfactual_paths(
        self,
        group_questions: Sequence[str],
        group_answers: Sequence[str],
        pairs: Sequence[HardPathPair],
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if metadata["forced_actions"].shape[0] == 0:
            return torch.empty(0, device=self.device)
        counterfactual_questions = []
        counterfactual_answers = []
        for row in range(metadata["forced_actions"].shape[0]):
            pair = pairs[int(metadata["pair_indices"][row].item())]
            direction = int(metadata["directions"][row].item())
            recipient = (
                pair.correct_index if direction == 0 else pair.wrong_index
            )
            counterfactual_questions.append(group_questions[recipient])
            counterfactual_answers.append(group_answers[recipient])

        scores = []
        micro_batch = self._rollout_micro_batch_size()
        for start in range(
            0,
            len(counterfactual_questions),
            micro_batch,
        ):
            end = min(
                start + micro_batch,
                len(counterfactual_questions),
            )
            trajectory = self._trajectory_latents(
                counterfactual_questions[start:end],
                innovations=metadata["innovations"][start:end],
                forced_actions=metadata["forced_actions"][start:end],
                forced_action_mask=metadata["forced_mask"][start:end],
            )
            scores.append(
                self._gold_answer_scores(
                    trajectory,
                    counterfactual_answers[start:end],
                )
            )
            del trajectory
        return torch.cat(scores, dim=0)

    @torch.no_grad()
    def trace_policy_rollout(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """Collect IID policy paths and separate path and token outcomes."""
        group_size = int(self.trace_rl_config.get("group_size", 8))
        if group_size < 3:
            raise ValueError("group_size must allow positive/positive/negative")
        group_questions = [
            question for question in questions for _ in range(group_size)
        ]
        group_answers = [
            answer for answer in answers for _ in range(group_size)
        ]
        micro_batch = self._rollout_micro_batch_size()
        action_chunks = []
        innovation_chunks = []
        action_log_prob_chunks = []
        path_chunks = []
        greedy_accuracy_chunks = []
        greedy_length_chunks = []
        sampled_accuracy_chunks = []
        sampled_id_chunks = []
        sampled_mask_chunks = []
        old_answer_log_prob_chunks = []

        for start in range(0, len(group_questions), micro_batch):
            end = min(start + micro_batch, len(group_questions))
            chunk_questions = group_questions[start:end]
            chunk_answers = group_answers[start:end]

            sampled_path = self._trajectory_latents(
                chunk_questions,
                deterministic=False,
            )
            actions = sampled_path["actions"].detach()
            innovations = sampled_path["innovations"].detach()
            greedy_ids = self._generate_answers_from_trajectory(
                sampled_path,
                do_sample=False,
            )
            greedy_accuracy = self._answers_to_accuracy(
                greedy_ids,
                chunk_answers,
            )
            greedy_lengths = greedy_ids.ne(
                self.tokenizer.pad_token_id
            ).float().sum(dim=-1)
            action_chunks.append(actions)
            innovation_chunks.append(innovations)
            action_log_prob_chunks.append(
                sampled_path["action_log_probs"].detach()
            )
            path_chunks.append(
                sampled_path["implicit_residuals"].float().detach()
            )
            greedy_accuracy_chunks.append(greedy_accuracy)
            greedy_length_chunks.append(greedy_lengths)
            del sampled_path, greedy_ids

            full_mask = torch.ones(
                actions.shape[:2],
                device=self.device,
                dtype=torch.bool,
            )
            answer_path = self._trajectory_latents(
                chunk_questions,
                forced_actions=actions,
                forced_action_mask=full_mask,
            )
            sampled_ids = self._generate_answers_from_trajectory(
                answer_path,
                do_sample=True,
                temperature=float(
                    self.trace_rl_config.get("answer_temperature", 0.95)
                ),
                top_p=float(
                    self.trace_rl_config.get("answer_top_p", 0.97)
                ),
            )
            sampled_mask = sampled_ids.ne(
                self.tokenizer.pad_token_id
            ).long()
            sampled_accuracy = self._answers_to_accuracy(
                sampled_ids,
                chunk_answers,
            )
            sampled_accuracy_chunks.append(sampled_accuracy)
            sampled_id_chunks.append(sampled_ids)
            sampled_mask_chunks.append(sampled_mask)
            del answer_path

            score_path = self._trajectory_latents(
                chunk_questions,
                forced_actions=actions,
                forced_action_mask=full_mask,
            )
            old_answer_log_prob_chunks.append(
                self._answer_token_log_probs(
                    score_path,
                    sampled_ids,
                    sampled_mask,
                ).detach()
            )
            del score_path

        actions = torch.cat(action_chunks, dim=0)
        innovations = torch.cat(innovation_chunks, dim=0)
        old_action_log_probs = torch.cat(
            action_log_prob_chunks,
            dim=0,
        )
        rollout_paths = torch.cat(path_chunks, dim=0)
        greedy_accuracy = torch.cat(greedy_accuracy_chunks, dim=0)
        greedy_lengths = torch.cat(greedy_length_chunks, dim=0)
        sampled_accuracy = torch.cat(sampled_accuracy_chunks, dim=0)
        sampled_ids = _right_pad(
            sampled_id_chunks,
            value=float(self.tokenizer.pad_token_id),
        )
        sampled_mask = _right_pad(sampled_mask_chunks, value=0.0)
        old_answer_log_probs = _right_pad(
            old_answer_log_prob_chunks,
            value=0.0,
        )
        output_lengths = sampled_mask.float().sum(dim=-1)
        length_weight = float(
            self.trace_rl_config.get(
                "output_length_penalty_weight",
                0.03,
            )
        )
        target_length = float(
            self.trace_rl_config.get("target_output_length", 33.5)
        )
        trajectory_length_penalty = length_weight * F.relu(
            greedy_lengths / max(target_length, 1.0) - 1.0
        )
        answer_length_penalty = length_weight * F.relu(
            output_lengths / max(target_length, 1.0) - 1.0
        )
        trajectory_rewards = greedy_accuracy - trajectory_length_penalty
        answer_rewards = sampled_accuracy - answer_length_penalty

        pairs = mine_question_local_hard_pairs(
            rollout_paths,
            greedy_accuracy,
            group_size=group_size,
            margin=float(
                self.trace_rl_config.get("local_ranking_margin", 0.08)
            ),
            max_pairs_per_group=int(
                self.trace_rl_config.get(
                    "max_hard_pairs_per_group",
                    1,
                )
            ),
            distance_kwargs=self._distance_kwargs(),
        )
        counterfactual_credits = rollout_paths.new_zeros(
            (0, self.n_trace_steps)
        )
        if pairs:
            base_scores = self._score_fixed_action_paths(
                group_questions,
                group_answers,
                actions,
            )
            counterfactual_metadata = counterfactual_action_batch(
                actions,
                innovations,
                pairs,
            )
            counterfactual_scores = self._score_counterfactual_paths(
                group_questions,
                group_answers,
                pairs,
                counterfactual_metadata,
            )
            counterfactual_credits = counterfactual_transition_credits(
                base_scores,
                counterfactual_scores,
                counterfactual_metadata,
                pairs,
                n_steps=self.n_trace_steps,
            )

        trajectory_advantages = build_transition_advantages(
            trajectory_rewards,
            n_steps=self.n_trace_steps,
            group_size=group_size,
            pairs=pairs,
            counterfactual_credits=(
                counterfactual_credits if pairs else None
            ),
            counterfactual_weight=float(
                self.trace_rl_config.get(
                    "counterfactual_credit_weight",
                    0.35,
                )
            ),
            local_weight=float(
                self.trace_rl_config.get("local_relation_weight", 0.10)
            ),
            credit_temperature=float(
                self.trace_rl_config.get(
                    "counterfactual_credit_temperature",
                    0.10,
                )
            ),
            local_temperature=float(
                self.trace_rl_config.get(
                    "local_relation_temperature",
                    0.08,
                )
            ),
        )
        answer_advantages = group_standardize(
            answer_rewards.unsqueeze(-1),
            group_size=group_size,
        )

        positive_counts = greedy_accuracy.view(
            -1,
            group_size,
        ).sum(dim=1)
        eligible = (positive_counts >= 2) & (
            positive_counts < group_size
        )
        pair_hinges = (
            torch.tensor(
                [pair.hinge for pair in pairs],
                device=self.device,
            )
            if pairs
            else torch.zeros(1, device=self.device)
        )
        credit_abs = (
            counterfactual_credits.abs().mean()
            if pairs
            else torch.zeros((), device=self.device)
        )
        self._last_trace_metrics = {
            "trace_policy/greedy_path_accuracy": greedy_accuracy.mean(),
            "trace_policy/sampled_answer_accuracy": sampled_accuracy.mean(),
            "trace_policy/positive_count": positive_counts.mean(),
            "trace_policy/mixed_group_fraction": (
                (positive_counts > 0) & (positive_counts < group_size)
            ).float().mean(),
            "trace_policy/counterfactual_eligible_fraction": (
                eligible.float().mean()
            ),
            "trace_policy/hard_pair_count": torch.tensor(
                float(len(pairs)),
                device=self.device,
            ),
            "trace_policy/hard_pair_hinge": pair_hinges.mean(),
            "trace_policy/counterfactual_credit_abs": credit_abs,
            "trace_policy/counterfactual_transition_coverage": torch.tensor(
                float(len(pairs) * self.n_trace_steps)
                / float(max(1, len(group_questions) * self.n_trace_steps)),
                device=self.device,
            ),
        }
        return {
            "group_questions": group_questions,
            "group_answers": group_answers,
            "actions": actions,
            "innovations": innovations,
            "rollout_paths": rollout_paths,
            "old_action_log_probs": old_action_log_probs,
            "trajectory_rewards": trajectory_rewards,
            "trajectory_advantages": trajectory_advantages,
            "greedy_accuracy": greedy_accuracy,
            "answer_input_ids": sampled_ids,
            "answer_attention_mask": sampled_mask,
            "old_answer_log_probs": old_answer_log_probs,
            "answer_rewards": answer_rewards,
            "answer_advantages": answer_advantages,
            "sampled_accuracy": sampled_accuracy,
            "counterfactual_credits": counterfactual_credits,
        }

    def _trajectory_policy_update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        questions = rollout["group_questions"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_action_log_probs"]
        advantages = rollout["trajectory_advantages"]
        micro_batch = self._optimization_micro_batch_size()
        total_items = len(questions)
        policy_loss_sum = torch.zeros((), device=self.device)
        prior_kl_sum = torch.zeros((), device=self.device)
        ratio_deviation_sum = torch.zeros((), device=self.device)
        clip_fraction_sum = torch.zeros((), device=self.device)
        prior_weight = float(
            self.trace_rl_config.get("stage1_policy_kl_weight", 0.02)
        )
        clip_epsilon = float(
            self.trace_rl_config.get("trajectory_clip_epsilon", 0.12)
        )
        full_mask = torch.ones(
            actions.shape[:2],
            device=self.device,
            dtype=torch.bool,
        )
        for start in range(0, total_items, micro_batch):
            end = min(start + micro_batch, total_items)
            current = self._trajectory_latents(
                questions[start:end],
                forced_actions=actions[start:end],
                forced_action_mask=full_mask[start:end],
                compute_reference=True,
            )
            policy_loss = clipped_policy_loss(
                current["action_log_probs"],
                old_log_probs[start:end].detach(),
                advantages[start:end].detach(),
                clip_epsilon=clip_epsilon,
            )
            ratio = torch.exp(
                current["action_log_probs"]
                - old_log_probs[start:end].detach()
            )
            ratio_deviation = (ratio - 1.0).abs().mean()
            clip_fraction = (
                (ratio < 1.0 - clip_epsilon)
                | (ratio > 1.0 + clip_epsilon)
            ).float().mean()
            prior_kl = diagonal_gaussian_kl(
                current["action_means"],
                current["action_log_stds"],
                current["reference_action_means"],
                current["reference_action_log_stds"],
            ).mean()
            chunk_weight = float(end - start) / float(total_items)
            self.manual_backward(
                chunk_weight * (policy_loss + prior_weight * prior_kl)
            )
            policy_loss_sum += (
                policy_loss.detach() * float(end - start)
            )
            prior_kl_sum += prior_kl.detach() * float(end - start)
            ratio_deviation_sum += (
                ratio_deviation.detach() * float(end - start)
            )
            clip_fraction_sum += (
                clip_fraction.detach() * float(end - start)
            )
            del (
                current,
                policy_loss,
                prior_kl,
                ratio,
                ratio_deviation,
                clip_fraction,
            )
        return (
            policy_loss_sum / float(total_items),
            prior_kl_sum / float(total_items),
            ratio_deviation_sum / float(total_items),
            clip_fraction_sum / float(total_items),
        )

    def _answer_policy_update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        questions = rollout["group_questions"]
        actions = rollout["actions"]
        answer_ids = rollout["answer_input_ids"]
        answer_mask = rollout["answer_attention_mask"]
        old_log_probs = rollout["old_answer_log_probs"]
        advantages = rollout["answer_advantages"]
        micro_batch = self._optimization_micro_batch_size()
        total_items = len(questions)
        loss_sum = torch.zeros((), device=self.device)
        clip_epsilon = float(
            self.trace_rl_config.get("answer_clip_epsilon", 0.12)
        )
        full_mask = torch.ones(
            actions.shape[:2],
            device=self.device,
            dtype=torch.bool,
        )
        for start in range(0, total_items, micro_batch):
            end = min(start + micro_batch, total_items)
            with torch.no_grad():
                trajectory = self._trajectory_latents(
                    questions[start:end],
                    forced_actions=actions[start:end],
                    forced_action_mask=full_mask[start:end],
                )
            current_log_probs = self._answer_token_log_probs(
                trajectory,
                answer_ids[start:end],
                answer_mask[start:end],
            )
            token_advantages = advantages[start:end].expand_as(
                current_log_probs
            )
            loss = clipped_policy_loss(
                current_log_probs,
                old_log_probs[start:end].detach(),
                token_advantages.detach(),
                clip_epsilon=clip_epsilon,
                mask=answer_mask[start:end],
            )
            chunk_weight = float(end - start) / float(total_items)
            self.manual_backward(chunk_weight * loss)
            loss_sum += loss.detach() * float(end - start)
            del trajectory, current_log_probs, loss
        return loss_sum / float(total_items)

    def trace_rl_training_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=0,
    ):
        optimizer = self.optimizers()
        rollout = self.trace_policy_rollout(
            questions=list(batch["question"]),
            answers=list(batch["answer"]),
        )
        update_epochs = int(
            self.trace_rl_config.get("policy_update_epochs", 2)
        )
        if update_epochs < 2:
            raise RuntimeError("invalid Stage-2 policy update contract")
        trajectory_losses = []
        answer_losses = []
        prior_kls = []
        ratio_deviations = []
        clip_fractions = []
        grad_norms = []
        optimizer_steps = 0
        for _ in range(update_epochs):
            optimizer.zero_grad(set_to_none=True)
            (
                trajectory_loss,
                prior_kl,
                ratio_deviation,
                clip_fraction,
            ) = self._trajectory_policy_update(rollout)
            answer_loss = self._answer_policy_update(rollout)

            grad_norm = clip_grad_norm_(
                [
                    parameter
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ],
                max_norm=float(
                    self.trace_rl_config.get("clip_grad_norm", 1.0)
                ),
            )
            optimizer_did_step = bool(torch.isfinite(grad_norm))
            if optimizer_did_step:
                optimizer.step()
                optimizer_steps += 1
                if bool(
                    self.all_config.model.training_kwargs.get(
                        "use_scheduler",
                        False,
                    )
                ):
                    scheduler = self.lr_schedulers()
                    if isinstance(scheduler, (list, tuple)):
                        for item in scheduler:
                            item.step()
                    elif scheduler is not None:
                        scheduler.step()
            else:
                optimizer.zero_grad(set_to_none=True)
            trajectory_losses.append(trajectory_loss)
            answer_losses.append(answer_loss)
            prior_kls.append(prior_kl)
            ratio_deviations.append(ratio_deviation)
            clip_fractions.append(clip_fraction)
            grad_norms.append(grad_norm.detach())

        trajectory_loss = torch.stack(trajectory_losses).mean()
        answer_loss = torch.stack(answer_losses).mean()
        prior_kl = torch.stack(prior_kls).mean()
        ratio_deviation = torch.stack(ratio_deviations).mean()
        final_ratio_deviation = ratio_deviations[-1]
        clip_fraction = torch.stack(clip_fractions).mean()
        final_clip_fraction = clip_fractions[-1]
        grad_norm = torch.stack(grad_norms).mean()

        prior_weight = float(
            self.trace_rl_config.get("stage1_policy_kl_weight", 0.02)
        )
        total_loss = (
            trajectory_loss
            + answer_loss
            + prior_weight * prior_kl
        )
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        learning_rate = float(raw_optimizer.param_groups[0]["lr"])
        logs = {
            "train/total_loss": total_loss.detach(),
            "train/trajectory_policy_loss": trajectory_loss.detach(),
            "train/answer_policy_loss": answer_loss.detach(),
            "train/stage1_policy_kl": prior_kl.detach(),
            "train/action_ratio_deviation": ratio_deviation.detach(),
            "train/action_ratio_deviation_final_update": (
                final_ratio_deviation.detach()
            ),
            "train/action_clip_fraction": clip_fraction.detach(),
            "train/action_clip_fraction_final_update": (
                final_clip_fraction.detach()
            ),
            "train/trajectory_reward": rollout[
                "trajectory_rewards"
            ].mean().detach(),
            "train/answer_reward": rollout[
                "answer_rewards"
            ].mean().detach(),
            "train/output_length": rollout[
                "answer_attention_mask"
            ].float().sum(dim=-1).mean().detach(),
            "train/n_latent_forward": torch.tensor(
                float(self.n_trace_steps),
                device=self.device,
            ),
            "train/grad_norm": grad_norm.detach(),
            "train/effective_lr": torch.tensor(
                learning_rate,
                device=self.device,
            ),
            "train/optimizer_did_step": torch.tensor(
                float(optimizer_steps) / float(update_epochs),
                device=self.device,
            ),
            "train/policy_update_epochs": torch.tensor(
                float(update_epochs),
                device=self.device,
            ),
        }
        logs.update(
            {
                f"train/{name}": value.detach()
                for name, value in self._last_trace_metrics.items()
            }
        )
        self.log_dict(
            logs,
            sync_dist=True,
            prog_bar=True,
            batch_size=len(batch["idx"]),
        )
        return total_loss.detach()

    @torch.no_grad()
    def read_generate_with_trajectory(
        self,
        questions: Sequence[str],
    ):
        trajectory = self._trajectory_latents(
            questions,
            deterministic=True,
        )
        output_ids = self._generate_answers_from_trajectory(
            trajectory,
            do_sample=False,
        )
        n_latent = torch.full(
            (len(questions), 1),
            fill_value=self.n_trace_steps,
            device=self.device,
            dtype=torch.long,
        )
        return output_ids, n_latent, trajectory

    @torch.no_grad()
    def read_generate(self, questions: List[str]):
        output_ids, n_latent, _ = self.read_generate_with_trajectory(
            questions
        )
        return output_ids, n_latent

    @staticmethod
    def _visual_seed(question: str, index: int, base_seed: int) -> int:
        digest = hashlib.sha256(
            f"{base_seed}|{index}|{question}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "little") % (2**63 - 1)

    def _pairwise_path_distances(
        self,
        paths: torch.Tensor,
    ) -> torch.Tensor:
        return trajectory_distance(
            paths[:, None],
            paths[None, :],
            **self._distance_kwargs(),
        )

    @torch.no_grad()
    def _build_policy_visual_record(
        self,
        *,
        index: int,
        question: str,
        answer: str,
        rationale_set: Sequence[dict],
        map_trajectory: Dict[str, torch.Tensor],
        map_local_index: int,
        map_prediction: str,
        map_accuracy: float,
        map_output_length: int,
    ) -> dict:
        group_size = int(
            self.trace_config.get("visual_group_size", 8)
        )
        seed = self._visual_seed(
            question,
            index,
            int(self.trace_config.get("visual_seed", 271828)),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        innovations = torch.randn(
            group_size,
            self.n_trace_steps,
            self.trajectory_policy.action_dim,
            generator=generator,
            dtype=torch.float32,
        ).to(self.device)
        questions = [question] * group_size
        answers = [answer] * group_size
        micro_batch = self._rollout_micro_batch_size()
        tensor_chunks: Dict[str, List[torch.Tensor]] = {
            "actions": [],
            "action_means": [],
            "action_log_stds": [],
            "latent_states": [],
            "implicit_residuals": [],
        }
        predictions = []
        correctness = []
        output_lengths = []
        for start in range(0, group_size, micro_batch):
            end = min(start + micro_batch, group_size)
            trajectory = self._trajectory_latents(
                questions[start:end],
                innovations=innovations[start:end],
            )
            output_ids = self._generate_answers_from_trajectory(
                trajectory,
                do_sample=False,
            )
            output_strings = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
            for output, target, token_ids in zip(
                output_strings,
                answers[start:end],
                output_ids,
            ):
                prediction = self.extract_answer_from_output(output)
                predictions.append(prediction)
                correctness.append(
                    int(self.verify_answer(target, prediction))
                )
                output_lengths.append(
                    int(
                        token_ids.ne(self.tokenizer.pad_token_id)
                        .sum()
                        .item()
                    )
                )
            for name in tensor_chunks:
                tensor_chunks[name].append(
                    trajectory[name].detach().float()
                )
            del trajectory, output_ids
        tensors = {
            name: torch.cat(chunks, dim=0)
            for name, chunks in tensor_chunks.items()
        }
        correctness_tensor = torch.tensor(
            correctness,
            device=self.device,
            dtype=torch.float32,
        )
        pairs = mine_question_local_hard_pairs(
            tensors["implicit_residuals"],
            correctness_tensor,
            group_size=group_size,
            margin=float(
                self.trace_rl_config.get("local_ranking_margin", 0.08)
            ),
            max_pairs_per_group=int(
                self.trace_rl_config.get(
                    "max_hard_pairs_per_group",
                    1,
                )
            ),
            distance_kwargs=self._distance_kwargs(),
        )

        fork_devices = (
            [torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed + 1)
            teacher_set = self._build_teacher_set(
                [question],
                [answer],
                [rationale_set],
            )
        teacher_paths = teacher_set["paths"][0]
        student_distances = self._pairwise_path_distances(
            tensors["implicit_residuals"]
        )
        teacher_distances = self._pairwise_path_distances(teacher_paths)
        return {
            "idx": int(index),
            "question": question,
            "answer": answer,
            "map_prediction": map_prediction,
            "map_correct": int(map_accuracy),
            "map_output_length": int(map_output_length),
            "map_path_type": "conditional_policy_mean",
            "map_actions": map_trajectory["actions"][map_local_index]
            .detach()
            .to(torch.float16)
            .cpu(),
            "map_action_means": map_trajectory["action_means"][
                map_local_index
            ]
            .detach()
            .to(torch.float16)
            .cpu(),
            "map_action_log_stds": map_trajectory["action_log_stds"][
                map_local_index
            ]
            .detach()
            .to(torch.float16)
            .cpu(),
            "map_latent_states": map_trajectory["latent_states"][
                map_local_index
            ]
            .detach()
            .to(torch.float16)
            .cpu(),
            "map_implicit_residuals": map_trajectory[
                "implicit_residuals"
            ][map_local_index]
            .detach()
            .to(torch.float16)
            .cpu(),
            "rollout_schema": "iid_conditional_gaussian",
            "rollout_seed": int(seed),
            "rollout_innovations": innovations.to(torch.float16).cpu(),
            "rollout_actions": tensors["actions"].to(torch.float16).cpu(),
            "rollout_action_means": tensors["action_means"]
            .to(torch.float16)
            .cpu(),
            "rollout_action_log_stds": tensors["action_log_stds"]
            .to(torch.float16)
            .cpu(),
            "rollout_latent_states": tensors["latent_states"]
            .to(torch.float16)
            .cpu(),
            "rollout_implicit_residuals": tensors[
                "implicit_residuals"
            ]
            .to(torch.float16)
            .cpu(),
            "rollout_predictions": predictions,
            "rollout_correctness": correctness,
            "rollout_output_lengths": output_lengths,
            "rollout_path_distance_matrix": student_distances.cpu(),
            "hard_pairs": [
                {
                    "correct_index": pair.correct_index,
                    "correct_peer_index": pair.correct_peer_index,
                    "wrong_index": pair.wrong_index,
                    "correct_radius": pair.correct_radius,
                    "wrong_distance": pair.wrong_distance,
                    "hinge": pair.hinge,
                }
                for pair in pairs
            ],
            "teacher_schema": (
                "verified_rationale_set_x_stochastic_monotone_compression"
            ),
            "teacher_paths": teacher_paths.detach()
            .to(torch.float16)
            .cpu(),
            "teacher_assignments": [
                assignment.detach().to(torch.float16).cpu()
                for assignment in teacher_set["assignments"][0]
            ],
            "teacher_progress_centers": teacher_set[
                "progress_centers"
            ][0]
            .detach()
            .cpu(),
            "teacher_semantic_mode_ids": teacher_set[
                "semantic_mode_ids"
            ][0]
            .detach()
            .cpu(),
            "teacher_relation_probs": [
                relation.detach().to(torch.float16).cpu()
                for relation in teacher_set["relation_probs"][0]
            ],
            "teacher_path_distance_matrix": teacher_distances.cpu(),
            "rationale_fingerprints": [
                rationale.get("fingerprint", "unknown")
                for rationale in rationale_set
            ],
            "answer_question_attention_access": 0,
            "answer_latent_attention_access": self.n_trace_steps,
            "visualization_contract": {
                "projection": "global_train_fit_pca_only",
                "manual_offsets": False,
                "per_path_rescaling": False,
                "outcome_used_for_projection": False,
            },
        }

    @torch.no_grad()
    def eval_generation(
        self,
        batch,
        split="val",
        batch_idx=None,
        dataloader_idx=0,
    ):
        indices = batch["idx"].tolist()
        questions = list(batch["question"])
        answers = list(batch["answer"])
        steps = batch["steps"]
        output_ids, n_latent, trajectory = (
            self.read_generate_with_trajectory(questions)
        )
        output_strings = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )
        rationale_sets = self._decode_rationale_sets(batch)
        accuracies = []
        output_lengths = []
        for local_index, (
            index,
            question,
            reasoning,
            answer,
            token_ids,
            output,
            latent_count,
        ) in enumerate(
            zip(
                indices,
                questions,
                steps,
                answers,
                output_ids,
                output_strings,
                n_latent,
            )
        ):
            prediction = self.extract_answer_from_output(output)
            accuracy = self.verify_answer(answer, prediction)
            output_length = int(
                token_ids.ne(self.tokenizer.pad_token_id).sum().item()
            )
            if index not in self.sample_logs:
                self.sample_logs[index]["question"] = question
                self.sample_logs[index]["steps"] = reasoning
                self.sample_logs[index]["answer"] = answer
                self.sample_logs[index]["pred_answer"] = []
                self.sample_logs[index]["output_string"] = []
                self.sample_logs[index]["output_length"] = []
                self.sample_logs[index]["n_latent_forward"] = []
                self.sample_logs[index]["acc"] = []
            self.sample_logs[index]["pred_answer"].append(prediction)
            self.sample_logs[index]["output_string"].append(output)
            self.sample_logs[index]["output_length"].append(output_length)
            self.sample_logs[index]["n_latent_forward"].append(
                int(latent_count.item())
            )
            self.sample_logs[index]["acc"].append(accuracy)
            accuracies.append(accuracy)
            output_lengths.append(output_length)

            record_limit = int(
                self.trace_config.get("visual_record_limit", 0)
            )
            if (
                record_limit > 0
                and len(self._trace_visual_records) < record_limit
            ):
                self._trace_visual_records.append(
                    self._build_policy_visual_record(
                        index=int(index),
                        question=question,
                        answer=answer,
                        rationale_set=rationale_sets[local_index],
                        map_trajectory=trajectory,
                        map_local_index=local_index,
                        map_prediction=prediction,
                        map_accuracy=accuracy,
                        map_output_length=output_length,
                    )
                )

        mean_accuracy = float(np.mean(accuracies))
        return {
            "monitor": mean_accuracy,
            f"{split}/acc": mean_accuracy,
            f"{split}/n_latent_forward": float(self.n_trace_steps),
            f"{split}/n_latent_forward_on_acc": float(
                self.n_trace_steps
            ),
            f"{split}/output_length": float(np.mean(output_lengths)),
        }

    def on_validation_epoch_start(self):
        self._validation_question_records = []
        return super().on_validation_epoch_start()

    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        metrics = self.eval_generation(
            batch=batch,
            split="val",
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        for index in batch["idx"].tolist():
            self._validation_question_records.append(
                (
                    int(index),
                    float(self.sample_logs[index]["acc"][-1]),
                    int(self.sample_logs[index]["output_length"][-1]),
                )
            )
        return metrics

    def on_test_start(self):
        self._trace_visual_records = []
        return super().on_test_start()

    def _save_trace_visual_records(self, split: str):
        if not self._trace_visual_records:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not getattr(
            trainer,
            "is_global_zero",
            True,
        ):
            return
        try:
            directory = Path(self.logger.log_dir)
        except Exception:
            directory = Path(".")
        torch.save(
            self._trace_visual_records,
            directory / f"trace_policy_visual_{split}.pt",
        )

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return super().on_validation_epoch_end()
        if dist.is_available() and dist.is_initialized():
            shards = [None] * dist.get_world_size()
            dist.all_gather_object(
                shards,
                self._validation_question_records,
            )
        else:
            shards = [self._validation_question_records]
        expected_count = len(self.trainer.datamodule.val_set)
        summary = summarize_unique_validation_records(
            shards,
            expected_count=expected_count,
        )
        self.log_dict(
            {
                "monitor": summary["accuracy"],
                "val/acc": summary["accuracy"],
                "val/output_length": summary["output_length"],
                "val/n_latent_forward": float(self.n_trace_steps),
                "val/unique_questions": summary["unique_questions"],
            },
            sync_dist=False,
            on_step=False,
            on_epoch=True,
            batch_size=expected_count,
        )
        self._save_trace_visual_records("val")
        return super().on_validation_epoch_end()

    def on_test_end(self):
        self._save_trace_visual_records("test")
        return super().on_test_end()
