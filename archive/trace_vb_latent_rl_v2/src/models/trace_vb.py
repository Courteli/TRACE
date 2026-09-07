import copy
import collections
import ctypes
import gc
import hashlib
import json
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_cosine_schedule_with_warmup

from .read_stable_efficient import LitREADCoTStableEfficient
from ..modules.readcot import (
    aggregate_step_residuals,
    dependency_bce_loss,
    dependency_f1_score,
    reconstruct_dependency_logits,
)
from ..modules.trace_policy import (
    CoTConditionedTrajectoryPosterior,
    GaussianTrajectoryPolicy,
    build_contiguous_cot_targets,
    build_discounted_role_advantages,
    build_discounted_role_returns,
    contiguous_cot_chunk_spans,
    HardPathPair,
    action_conditioned_progress_centers,
    action_transition_identifiability_loss,
    build_transition_advantages,
    clipped_policy_loss,
    counterfactual_action_batch,
    counterfactual_transition_credits,
    diagonal_gaussian_kl,
    gaussian_log_prob,
    group_standardize,
    masked_cosine_loss,
    masked_cosine_similarity,
    minimum_action_entropy_loss,
    mine_question_local_hard_pairs,
    pairwise_action_path_correlation,
    path_noncollapse_loss,
    sampled_forward_kl,
    stochastic_monotone_assignment,
    trajectory_distance,
    trajectory_distance_components,
)
from ..modules.trace_vb import (
    LatentStepTextDecoder,
    PlanForecastHead,
    RoleConditionedScalarHead,
    action_efficacy_hinge,
    checkpointed_projected_token_cross_entropy,
    masked_ppo_actor_loss,
    masked_terminal_reward_gae,
    masked_value_loss,
)


@contextmanager
def selective_saved_activation_offload(
    *,
    minimum_bytes: int,
    pin_memory: bool,
):
    """Offload large dense activations while leaving SDPA metadata in place."""

    def pack(tensor: torch.Tensor):
        nbytes = tensor.numel() * tensor.element_size()
        should_offload = (
            tensor.device.type == "cuda"
            and tensor.is_floating_point()
            and not tensor.is_leaf
            and tensor.ndim in (2, 3)
            and tensor.is_contiguous()
            and nbytes >= minimum_bytes
        )
        if not should_offload:
            return tensor
        cpu_tensor = tensor.detach().to(
            device="cpu",
            non_blocking=False,
            copy=True,
        )
        if pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
        return (
            "trace_saved_activation_cpu",
            tensor.device,
            cpu_tensor,
            pin_memory,
        )

    def unpack(packed):
        if (
            not isinstance(packed, tuple)
            or len(packed) != 4
            or packed[0] != "trace_saved_activation_cpu"
        ):
            return packed
        _, device, cpu_tensor, was_pinned = packed
        return cpu_tensor.to(
            device=device,
            non_blocking=was_pinned,
        )

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


class Stage1RecoveryCheckpoint(Callback):
    """Keep one resumable, optimizer-complete checkpoint within each epoch."""

    def __init__(self, every_n_train_steps: int) -> None:
        super().__init__()
        self.every_n_train_steps = int(every_n_train_steps)
        if self.every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be positive")
        self._last_saved_step = -1
        self._last_checkpoint_path: Optional[Path] = None

    @property
    def state_key(self) -> str:
        return (
            f"{self.__class__.__qualname__}"
            f"{{every_n_train_steps={self.every_n_train_steps}}}"
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if bool(getattr(pl_module, "do_trace_rl", False)):
            return
        step = int(trainer.global_step)
        if (
            step <= 0
            or step == self._last_saved_step
            or step % self.every_n_train_steps != 0
        ):
            return
        primary = trainer.checkpoint_callback
        directory = getattr(primary, "dirpath", None)
        if not directory:
            directory = Path(trainer.default_root_dir) / "checkpoints"
        directory = Path(str(directory))
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / (
            f"recovery-epoch{int(trainer.current_epoch):02d}"
            f"-step{step:06d}.ckpt"
        )
        trainer.save_checkpoint(str(checkpoint_path), weights_only=False)
        if (
            trainer.is_global_zero
            and self._last_checkpoint_path is not None
            and self._last_checkpoint_path != checkpoint_path
            and self._last_checkpoint_path.is_file()
        ):
            self._last_checkpoint_path.unlink()
        self._last_checkpoint_path = checkpoint_path
        self._last_saved_step = step


def _parse_linux_memory_gib(text: str, key: str) -> float:
    """Parse one Linux procfs memory field expressed in KiB."""
    prefix = f"{key}:"
    for line in str(text).splitlines():
        if not line.startswith(prefix):
            continue
        fields = line[len(prefix) :].strip().split()
        if not fields:
            break
        value_kib = float(fields[0])
        unit = fields[1].lower() if len(fields) > 1 else "kib"
        if unit not in ("kb", "kib"):
            raise RuntimeError(f"unsupported {key} unit: {unit}")
        return value_kib / (1024.0 * 1024.0)
    raise RuntimeError(f"missing Linux memory field: {key}")


def _host_memory_guard_reasons(
    *,
    maximum_rank_rss_gib: float,
    minimum_host_available_gib: float,
    maximum_allowed_rank_rss_gib: float,
    minimum_required_host_available_gib: float,
) -> List[str]:
    """Return deterministic fail-closed reasons for a host-memory snapshot."""
    values = (
        maximum_rank_rss_gib,
        minimum_host_available_gib,
        maximum_allowed_rank_rss_gib,
        minimum_required_host_available_gib,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return ["non_finite_memory_measurement"]
    reasons = []
    if maximum_rank_rss_gib >= maximum_allowed_rank_rss_gib:
        reasons.append(
            "maximum_rank_rss_gib="
            f"{maximum_rank_rss_gib:.3f}>="
            f"{maximum_allowed_rank_rss_gib:.3f}"
        )
    if minimum_host_available_gib <= minimum_required_host_available_gib:
        reasons.append(
            "minimum_host_available_gib="
            f"{minimum_host_available_gib:.3f}<="
            f"{minimum_required_host_available_gib:.3f}"
        )
    return reasons


class Stage1HostMemoryGuard(Callback):
    """Save and fail closed before this job can approach host OOM."""

    def __init__(
        self,
        *,
        every_n_train_steps: int,
        maximum_rank_rss_gib: float,
        minimum_host_available_gib: float,
    ) -> None:
        super().__init__()
        self.every_n_train_steps = int(every_n_train_steps)
        self.maximum_rank_rss_gib = float(maximum_rank_rss_gib)
        self.minimum_host_available_gib = float(minimum_host_available_gib)
        if self.every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be positive")
        if (
            not math.isfinite(self.maximum_rank_rss_gib)
            or self.maximum_rank_rss_gib <= 0.0
        ):
            raise ValueError("maximum_rank_rss_gib must be finite and positive")
        if (
            not math.isfinite(self.minimum_host_available_gib)
            or self.minimum_host_available_gib <= 0.0
        ):
            raise ValueError(
                "minimum_host_available_gib must be finite and positive"
            )
        self._last_checked_step = -1

    @property
    def state_key(self) -> str:
        return (
            f"{self.__class__.__qualname__}"
            f"{{every_n_train_steps={self.every_n_train_steps},"
            f"maximum_rank_rss_gib={self.maximum_rank_rss_gib},"
            f"minimum_host_available_gib={self.minimum_host_available_gib}}}"
        )

    @staticmethod
    def _release_allocator_caches() -> None:
        gc.collect()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except AttributeError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _local_memory_snapshot() -> Tuple[float, float]:
        process_status = Path("/proc/self/status").read_text(
            encoding="utf-8"
        )
        memory_info = Path("/proc/meminfo").read_text(encoding="utf-8")
        return (
            _parse_linux_memory_gib(process_status, "VmRSS"),
            _parse_linux_memory_gib(memory_info, "MemAvailable"),
        )

    @staticmethod
    def _checkpoint_directory(trainer) -> Path:
        primary = trainer.checkpoint_callback
        directory = getattr(primary, "dirpath", None)
        if not directory:
            directory = Path(trainer.default_root_dir) / "checkpoints"
        return Path(str(directory))

    def on_train_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx: int,
    ) -> None:
        if bool(getattr(pl_module, "do_trace_rl", False)):
            return
        step = int(trainer.global_step)
        if (
            step == self._last_checked_step
            or step % self.every_n_train_steps != 0
        ):
            return
        self._last_checked_step = step
        self._release_allocator_caches()
        local_rss_gib, local_available_gib = self._local_memory_snapshot()
        rss = torch.tensor(
            local_rss_gib,
            device=pl_module.device,
            dtype=torch.float64,
        )
        available = torch.tensor(
            local_available_gib,
            device=pl_module.device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(rss, op=dist.ReduceOp.MAX)
            dist.all_reduce(available, op=dist.ReduceOp.MIN)
        maximum_rss_gib = float(rss.item())
        minimum_available_gib = float(available.item())

        if trainer.is_global_zero and trainer.logger is not None:
            experiment = getattr(trainer.logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "add_scalar"):
                experiment.add_scalar(
                    "system/maximum_rank_rss_gib",
                    maximum_rss_gib,
                    step,
                )
                experiment.add_scalar(
                    "system/minimum_host_available_gib",
                    minimum_available_gib,
                    step,
                )

        reasons = _host_memory_guard_reasons(
            maximum_rank_rss_gib=maximum_rss_gib,
            minimum_host_available_gib=minimum_available_gib,
            maximum_allowed_rank_rss_gib=self.maximum_rank_rss_gib,
            minimum_required_host_available_gib=(
                self.minimum_host_available_gib
            ),
        )
        if not reasons:
            return

        directory = self._checkpoint_directory(trainer)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / (
            f"host-memory-guard-epoch{int(trainer.current_epoch):02d}"
            f"-step{step:06d}.ckpt"
        )
        trainer.save_checkpoint(str(checkpoint_path), weights_only=False)
        if trainer.is_global_zero:
            report = {
                "status": "CONTROLLED_STOP",
                "global_step": step,
                "maximum_rank_rss_gib": maximum_rss_gib,
                "minimum_host_available_gib": minimum_available_gib,
                "maximum_allowed_rank_rss_gib": self.maximum_rank_rss_gib,
                "minimum_required_host_available_gib": (
                    self.minimum_host_available_gib
                ),
                "reasons": reasons,
                "checkpoint": str(checkpoint_path),
            }
            checkpoint_path.with_suffix(".json").write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
        raise RuntimeError(
            "TRACE_VB_HOST_MEMORY_GUARD: "
            + "; ".join(reasons)
            + f"; recovery={checkpoint_path}"
        )


def build_path_bottleneck_mask(
    question_attention_mask: torch.Tensor,
    latent_attention_mask: torch.Tensor,
    answer_attention_mask: torch.Tensor,
    *,
    include_question: bool = False,
) -> torch.Tensor:
    """Build the answer decoder's source mask.

    The caller supplies the already-resolved latent mask.  The formal VB-v3
    model uses ``include_question=True`` together with a COMMIT-only latent
    mask: answer queries retain the reliable raw-question channel and read the
    final committed state, but never bypass COMMIT to inspect the seven private
    reasoning states directly.
    """
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
            (
                question_attention_mask
                if include_question
                else torch.zeros_like(question_attention_mask)
            ),
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
    shards: Sequence[
        Sequence[Tuple[int, float, int, str, bool, bool]]
    ],
    *,
    expected_count: int,
) -> Dict[str, float]:
    """Deduplicate DDP padding and summarize full-set generation behavior."""
    records: Dict[int, Tuple[float, int, str, bool, bool]] = {}
    for shard in shards:
        for raw_record in shard:
            if len(raw_record) != 6:
                raise RuntimeError(
                    "TRACE-VB-v3 validation records require index, accuracy, "
                    "length, prediction, validity, and nonempty-output fields"
                )
            (
                index,
                accuracy,
                output_length,
                prediction_key,
                valid_answer,
                nonempty_output,
            ) = raw_record
            value = (
                float(accuracy),
                int(output_length),
                str(prediction_key),
                bool(valid_answer),
                bool(nonempty_output),
            )
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
    prediction_counts = collections.Counter(
        value[2] for value in records.values()
    )
    return {
        "accuracy": float(
            np.mean([value[0] for value in records.values()])
        ),
        "output_length": float(
            np.mean([value[1] for value in records.values()])
        ),
        "unique_questions": float(len(records)),
        "unique_predictions": float(len(prediction_counts)),
        "unique_prediction_ratio": float(
            len(prediction_counts) / max(1, len(records))
        ),
        "top1_mode_fraction": float(
            prediction_counts.most_common(1)[0][1]
            / max(1, len(records))
        ),
        "valid_answer_fraction": float(
            np.mean([value[3] for value in records.values()])
        ),
        "nonempty_output_fraction": float(
            np.mean([value[4] for value in records.values()])
        ),
    }


class LitTRACEVB(LitREADCoTStableEfficient):
    """TRACE-VB role program with semantic-to-outcome value bridging.

    The eight recurrent states have a fixed computation contract:
    PLAN, five shared-dynamics SOLVE states, REFINE, and deterministic COMMIT.
    Stage 1 distils ordered textual-CoT semantics and answer sufficiency into
    the latent program. Stage 2 freezes the language model and uses a calibrated
    role critic plus GAE to assign exact-answer outcome credit to latent actions.
    """

    path_adapter_name = "default"
    cot_encoder_adapter_name = "trace_cot_encoder"
    answer_adapter_name = "trace_answer"

    @staticmethod
    def _trace_position_ids(
        attention_mask: torch.Tensor,
        current_length: int,
    ) -> torch.Tensor:
        """Position current tokens from the unmasked causal source sequence."""
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions = positions.masked_fill(attention_mask == 0, 0)
        return positions[:, -int(current_length) :]

    def __init__(self, model_kwargs, training_kwargs, all_config=None):
        super().__init__(
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            all_config=all_config,
        )
        self.trace_config = model_kwargs.get("trace_policy_config", {})
        self.trace_rl_config = model_kwargs.get("trace_rl_config", {})
        self.do_trace_rl = bool(model_kwargs.get("do_trace_rl", False))
        if bool(
            self.trace_config.get(
                "stage1_posterior_activation_offload",
                False,
            )
        ):
            raise ValueError(
                "TRACE-VB-v3 forbids saved-tensor CPU activation offload: "
                "the exact offload path has a measured per-step host-RSS "
                "leak. Use GPU-resident activations plus checkpointed "
                "solve-text projection instead."
            )
        # TRACE does not use the inherited deterministic bridge or its random
        # residual projection. Keeping those dormant parameters would make the
        # compact-target selector depend on an untrained legacy branch.
        self.latent_bridge = torch.nn.Identity()
        self.residual_projector = torch.nn.Identity()
        # The inherited READ compressor/relation modules belong to the former
        # corridor objective.  Role targets now come directly from contiguous
        # CoT residual chunks, so these legacy diagnostics must neither enter
        # the optimizer nor create DDP unused-parameter failures.
        for legacy_module in (self.step_compressor, self.latent_relation):
            for parameter in legacy_module.parameters():
                parameter.requires_grad_(False)
        if not self.model_kwargs.get("do_lora", False):
            raise ValueError(
                "TRACE requires LoRA to isolate teacher, path, and answer roles"
            )
        if self.cot_encoder_adapter_name not in self.llm.peft_config:
            cot_encoder_config = copy.deepcopy(
                self.llm.peft_config[self.path_adapter_name]
            )
            self.llm.add_adapter(
                self.cot_encoder_adapter_name,
                cot_encoder_config,
            )
        self._match_adapter_storage_to_path(
            self.cot_encoder_adapter_name
        )
        self._set_adapter_parameter_trainability()
        self.n_trace_steps = int(self.readcot_config.n_latents)
        if self.n_trace_steps != 8:
            raise ValueError(
                "TRACE-VB requires exactly 8 latent states: "
                "PLAN, SOLVE x5, REFINE, COMMIT"
            )
        if self.readcot_config.get("implicit_latent_mode") == "block":
            # TRACE's action distribution is autoregressive even if the source
            # BRIDGE config used a block latent implementation.
            self.readcot_config.implicit_latent_mode = "autoregressive-policy"
        if self.readcot_config.get("use_anchor_gate", False):
            raise ValueError("TRACE policy does not use a route gate")
        answer_context_mode = str(
            self.trace_config.get(
                "answer_context_mode",
                "question_and_commit",
            )
        )
        if answer_context_mode != "question_and_commit":
            raise ValueError(
                "TRACE-VB-v3 requires answer_context_mode="
                "question_and_commit: the answer must retain the raw "
                "question and may read only deterministic COMMIT among "
                "the latent states"
            )
        self.answer_context_mode = answer_context_mode
        self.answer_reads_question = True
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
        # TRACE-VB deliberately removes the CoT-conditioned posterior.  Both
        # SFT paths are question-only, matching deployment and eliminating the
        # old three-posterior-plus-MAP activation graph.
        self.trajectory_posterior = None
        self.posterior_context_norm = torch.nn.Identity()

        semantic_dim = int(
            self.trace_config.get("semantic_projection_size", 256)
        )
        if semantic_dim <= 0:
            raise ValueError("semantic_projection_size must be positive")
        projection_generator = torch.Generator(device="cpu")
        projection_generator.manual_seed(
            int(self.trace_config.get("semantic_projection_seed", 20260809))
        )
        semantic_projection = torch.randn(
            semantic_dim,
            self.hidden_size,
            generator=projection_generator,
            dtype=torch.float32,
        ) / math.sqrt(float(self.hidden_size))
        self.register_buffer(
            "semantic_projection",
            semantic_projection,
            persistent=True,
        )
        self.plan_forecaster = PlanForecastHead(
            hidden_size=self.hidden_size,
            target_dim=semantic_dim,
            n_targets=5,
            head_hidden_size=int(
                self.trace_config.get("plan_forecast_hidden_size", 512)
            ),
        )
        self.solve_text_decoder = LatentStepTextDecoder(
            hidden_size=self.hidden_size,
            decoder_hidden_size=int(
                self.trace_config.get("solve_text_decoder_hidden_size", 512)
            ),
            n_solve_roles=5,
        )
        scalar_head_kwargs = {
            "hidden_size": self.hidden_size,
            "n_roles": self.n_trace_steps,
            "role_embedding_dim": int(
                self.trace_rl_config.get(
                    "critic_role_embedding_size",
                    64,
                )
            ),
            "head_hidden_size": int(
                self.trace_rl_config.get("critic_hidden_size", 512)
            ),
        }
        self.sufficiency_head = RoleConditionedScalarHead(
            **scalar_head_kwargs
        )
        self.value_critic = RoleConditionedScalarHead(**scalar_head_kwargs)
        for parameter in self.value_critic.parameters():
            parameter.requires_grad_(False)

        self._sufficiency_cache = None
        self._sufficiency_cache_validated = False
        self._value_bridge_initialized = False
        self.register_buffer(
            "vb_rollout_batches_seen",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.stage1_policy_reference = copy.deepcopy(
            self.trajectory_policy
        )
        for parameter in self.stage1_policy_reference.parameters():
            parameter.requires_grad_(False)

        self._solve_text_decoder_loaded = False
        self._reference_restored = False
        self._loaded_stage2_state = False
        self._cot_encoder_adapter_loaded = False
        self._stage2_initialized = False
        self._last_trace_metrics: Dict[str, torch.Tensor] = {}
        self._trace_visual_records: List[dict] = []
        self._validation_question_records: List[
            Tuple[int, float, int, str, bool, bool]
        ] = []
        self.strict_loading = False

        if self.do_trace_rl:
            required_stage2_objectives = ("use_trajectory_policy_loss",)
            disabled = [
                key
                for key in required_stage2_objectives
                if not bool(self.trace_rl_config.get(key, False))
            ]
            if disabled:
                raise ValueError(
                    "Role-semantic Stage 2 requires latent-policy RL; "
                    f"disabled: {disabled}"
                )
            if int(
                self.trace_rl_config.get("policy_update_epochs", 4)
            ) < 2:
                raise ValueError(
                    "TRACE-VB requires at least two PPO epochs so clipping "
                    "operates on non-unit post-update ratios"
                )
            positive_stage2_weights = (
                "stage1_policy_kl_weight",
                "value_loss_coefficient",
            )
            invalid_weights = [
                key
                for key in positive_stage2_weights
                if float(self.trace_rl_config.get(key, 0.0)) <= 0.0
            ]
            if invalid_weights:
                raise ValueError(
                    "TRACE-VB requires value learning and a Stage-1 "
                    "trust-region; non-positive weights: "
                    f"{invalid_weights}"
                )
            self._initialize_stage2_modules()
            self.automatic_optimization = False

    def _initialize_stage2_modules(self):
        # The LM, transition semantics, COMMIT, PLAN forecast, and answer
        # channel are immutable.  Only stochastic Gaussian output heads and
        # the calibrated role critic are optimized.
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.trajectory_policy.set_stage2_trainability()
        for parameter in self.value_critic.parameters():
            parameter.requires_grad_(True)
        self._set_adapter_parameter_trainability()
        self._activate_path_adapter()

    def configure_optimizers(self):
        """Use separate actor/critic rates during head-only Stage-2 PPO."""
        if not self.do_trace_rl:
            return super().configure_optimizers()

        self.trainable_parameter_names = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        actor_parameters = [
            parameter
            for parameter in self.trajectory_policy.parameters()
            if parameter.requires_grad
        ]
        critic_parameters = [
            parameter
            for parameter in self.value_critic.parameters()
            if parameter.requires_grad
        ]
        if not actor_parameters or not critic_parameters:
            raise RuntimeError(
                "TRACE-VB Stage 2 requires trainable actor and critic heads"
            )
        allowed_ids = {
            id(parameter)
            for parameter in actor_parameters + critic_parameters
        }
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed_ids
        ]
        if unexpected:
            raise RuntimeError(
                "Stage-2 trainability escaped actor/critic heads: "
                + ", ".join(unexpected[:20])
            )

        optimizer_config = self.all_config.model.training_kwargs.optimizer
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": actor_parameters,
                    "lr": float(
                        self.trace_rl_config.get("actor_lr", 8.0e-7)
                    ),
                },
                {
                    "params": critic_parameters,
                    "lr": float(
                        self.trace_rl_config.get("critic_lr", 1.0e-4)
                    ),
                },
            ],
            weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
            foreach=bool(optimizer_config.get("foreach", False)),
        )
        if not bool(
            self.all_config.model.training_kwargs.get("use_scheduler", False)
        ):
            return {"optimizer": optimizer}
        scheduler_config = self.all_config.model.training_kwargs.scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(scheduler_config.warmup_steps),
            num_training_steps=int(scheduler_config.num_training_steps),
        )
        self.lr_scheduler = scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_callbacks(self):
        """Add Stage-1 recovery and host guards beside validation callbacks."""
        inherited = super().configure_callbacks()
        if inherited is None:
            callbacks = []
        elif isinstance(inherited, (list, tuple)):
            callbacks = list(inherited)
        else:
            callbacks = [inherited]
        recovery_interval = int(
            self.trace_config.get(
                "stage1_recovery_checkpoint_interval",
                0,
            )
        )
        guard_interval = int(
            self.trace_config.get(
                "stage1_host_memory_guard_interval",
                0,
            )
        )
        if not self.do_trace_rl and recovery_interval > 0:
            callbacks.append(
                Stage1RecoveryCheckpoint(recovery_interval)
            )
        if not self.do_trace_rl and guard_interval > 0:
            callbacks.append(
                Stage1HostMemoryGuard(
                    every_n_train_steps=guard_interval,
                    maximum_rank_rss_gib=float(
                        self.trace_config.get(
                            "stage1_maximum_rank_rss_gib",
                            20.0,
                        )
                    ),
                    minimum_host_available_gib=float(
                        self.trace_config.get(
                            "stage1_minimum_host_available_gib",
                            192.0,
                        )
                    ),
                )
            )
        return callbacks

    def _set_adapter_parameter_trainability(self):
        path_marker = f".{self.path_adapter_name}."
        cot_encoder_marker = f".{self.cot_encoder_adapter_name}."
        answer_marker = f".{self.answer_adapter_name}."
        for name, parameter in self.llm.named_parameters():
            if cot_encoder_marker in name:
                parameter.requires_grad_(False)
            elif path_marker in name:
                parameter.requires_grad_(not self.do_trace_rl)
            elif answer_marker in name:
                parameter.requires_grad_(False)

    def _activate_path_adapter(self):
        self.llm.set_adapter(self.path_adapter_name)
        self._set_adapter_parameter_trainability()

    def _activate_cot_encoder_adapter(self):
        self.llm.set_adapter(self.cot_encoder_adapter_name)
        self._set_adapter_parameter_trainability()

    def _activate_answer_adapter(self):
        self.llm.set_adapter(self.path_adapter_name)
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
        return 0

    def _copy_path_adapter_to_cot_encoder_adapter(self):
        self._copy_path_adapter(self.cot_encoder_adapter_name)
        self._cot_encoder_adapter_loaded = True
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
        cot_encoder_marker = f".{self.cot_encoder_adapter_name}."
        # Stage 2 has no trainable answer adapter, so checkpoint metadata set
        # by on_load_checkpoint is the authoritative stage marker.
        self._loaded_stage2_state = (
            self._loaded_stage2_state
            or any(answer_marker in name for name in state_dict)
        )
        self._cot_encoder_adapter_loaded = (
            self._cot_encoder_adapter_loaded
            or any(cot_encoder_marker in name for name in state_dict)
        )
        self._solve_text_decoder_loaded = (
            self._solve_text_decoder_loaded
            or any(name.startswith("solve_text_decoder.") for name in state_dict)
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
        self._value_bridge_initialized = bool(
            checkpoint.get("trace_vb_value_bridge_initialized", False)
        )
        return super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        full_state = self.state_dict()
        preserve_prefixes = (
            "trajectory_policy.",
            "plan_forecaster.",
            "solve_text_decoder.",
            "sufficiency_head.",
            "value_critic.",
            "semantic_projection",
            "vb_rollout_batches_seen",
            "state_norm.",
        )
        path_marker = f".{self.path_adapter_name}."
        cot_encoder_marker = f".{self.cot_encoder_adapter_name}."
        for name, value in full_state.items():
            if (
                name.startswith(preserve_prefixes)
                or path_marker in name
                or cot_encoder_marker in name
            ):
                checkpoint["state_dict"][name] = value
        checkpoint_stage = 2 if self.do_trace_rl else 1
        checkpoint["trace_policy_training_stage"] = checkpoint_stage
        checkpoint["trace_vb_schema_version"] = "trace_vb_v3"
        checkpoint["trace_vb_value_bridge_initialized"] = bool(
            self._value_bridge_initialized
        )
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
        if not self._cot_encoder_adapter_loaded:
            if self.do_trace_rl:
                raise RuntimeError(
                    "Stage 2 requires a Stage-1 checkpoint containing the "
                    "frozen single-CoT encoder adapter"
                )
            self._copy_path_adapter_to_cot_encoder_adapter()
        if self.do_trace_rl:
            if (
                bool(self.trace_rl_config.get("use_semantic_anchor", True))
                and not self._solve_text_decoder_loaded
            ):
                raise RuntimeError(
                    "Stage 2 semantic anchoring requires a Stage-1 checkpoint "
                    "containing solve_text_decoder parameters"
                )
            if self._loaded_stage2_state and not self._reference_restored:
                raise RuntimeError(
                    "A Stage-2 state was loaded without its immutable Stage-1 "
                    "policy reference. Resume from the full Lightning "
                    "checkpoint instead of loading weights only."
                )
            if not self._reference_restored:
                self._snapshot_stage1_policy()
            if not self._loaded_stage2_state:
                self.value_critic.copy_from(self.sufficiency_head)
                self._value_bridge_initialized = True
            self._stage2_initialized = True
            self._activate_path_adapter()
            self._validate_trace_rl_epoch_budget()
        else:
            self._load_and_validate_sufficiency_cache()
        return super().on_fit_start()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_and_validate_sufficiency_cache(self) -> None:
        """Load the immutable Stage-0 prefix signal and fail closed.

        The formal run never constructs labels online.  It accepts only the
        registered full training split, a cache tied to that file, and (when
        configured) the exact frozen Stage-0 checkpoint fingerprint.
        """
        path_value = self.trace_config.get("sufficiency_cache_path")
        if not path_value:
            raise RuntimeError("TRACE-VB Stage 1 requires sufficiency_cache_path")
        path = Path(str(path_value)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing TRACE-VB sufficiency cache: {path}")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict):
            raise TypeError("TRACE-VB sufficiency cache must be a dictionary")
        if cache.get("schema_version") != "trace_vb_prefix_sufficiency_v1":
            raise RuntimeError(
                "unsupported TRACE-VB sufficiency schema: "
                f"{cache.get('schema_version')!r}"
            )
        metadata = cache.get("metadata")
        rows = cache.get("rows")
        by_idx = cache.get("by_idx")
        if not isinstance(metadata, dict) or not isinstance(rows, list):
            raise TypeError("malformed TRACE-VB sufficiency cache metadata")
        if not isinstance(by_idx, dict):
            raise TypeError("malformed TRACE-VB sufficiency by_idx table")
        canonical_by_idx = {int(key): value for key, value in by_idx.items()}
        if len(canonical_by_idx) != len(rows):
            raise RuntimeError("sufficiency cache rows/by_idx cardinality mismatch")

        data_cfg = self.all_config.data_module
        train_path = Path(str(data_cfg.dataset_dir)) / str(data_cfg.train_file)
        train_path = train_path.resolve()
        fingerprints = metadata.get("fingerprints", {})
        expected_data_sha = str(fingerprints.get("data_sha256", ""))
        if not expected_data_sha or self._sha256_file(train_path) != expected_data_sha:
            raise RuntimeError(
                "sufficiency cache data fingerprint does not match the "
                "configured training split"
            )
        expected_teacher_sha = str(
            self.trace_config.get("sufficiency_teacher_checkpoint_sha256", "")
        )
        cached_teacher_sha = str(
            fingerprints.get("teacher_checkpoint_sha256", "")
        )
        if expected_teacher_sha and cached_teacher_sha != expected_teacher_sha:
            raise RuntimeError(
                "sufficiency cache was not built by the registered Stage-0 "
                "teacher"
            )
        fingerprint_contract = {
            "tokenizer_sha256": "sufficiency_tokenizer_sha256",
            "prompt_sha256": "sufficiency_prompt_sha256",
            "composite_sha256": "sufficiency_composite_sha256",
        }
        for cached_name, config_name in fingerprint_contract.items():
            expected = str(self.trace_config.get(config_name, ""))
            actual = str(fingerprints.get(cached_name, ""))
            if not expected or actual != expected:
                raise RuntimeError(
                    "sufficiency cache fingerprint mismatch for "
                    f"{cached_name}"
                )
        if len(rows) != 6726 or set(canonical_by_idx) != set(range(6726)):
            raise RuntimeError(
                "formal TRACE-VB cache must cover all 6726 registered "
                "training questions exactly once"
            )
        for index, row in canonical_by_idx.items():
            if int(row.get("idx", -1)) != index:
                raise RuntimeError(f"malformed sufficiency row index {index}")
            n_steps = int(row.get("n_steps", -1))
            if n_steps <= 0 or any(
                len(row.get(name, ())) != n_steps
                for name in ("scores", "valid_mask", "prefix_logps")
            ):
                raise RuntimeError(f"malformed sufficiency row {index}")
            if any(
                len(row.get(name, ())) != self.n_trace_steps
                for name in (
                    "role_scores",
                    "role_valid_mask",
                    "role_source_prefix_index",
                )
            ):
                raise RuntimeError(
                    f"malformed pre-action role alignment in row {index}"
                )
        self._sufficiency_cache = {
            "path": str(path),
            "sha256": self._sha256_file(path),
            "metadata": metadata,
            "by_idx": canonical_by_idx,
        }
        self._sufficiency_cache_validated = True

    def _batch_sufficiency_targets(
        self,
        batch,
        solve_spans: Sequence[Sequence[Tuple[int, int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align gold-prefix sufficiency to pre-action role states.

        s0 (question) and s1 (after PLAN) use the defined question-only
        baseline 0.  s2..s6 correspond to the endpoints after SOLVE1..5.
        COMMIT is deterministic and receives no actor/value target.
        """
        if not self._sufficiency_cache_validated:
            raise RuntimeError("sufficiency cache was not validated")
        raw_indices = batch["idx"]
        indices = (
            [int(value) for value in raw_indices.detach().cpu().tolist()]
            if isinstance(raw_indices, torch.Tensor)
            else [int(value) for value in raw_indices]
        )
        questions = list(batch["question"])
        batch_size = len(indices)
        targets = torch.zeros(
            batch_size, self.n_trace_steps, device=self.device, dtype=torch.float32
        )
        mask = torch.zeros_like(targets, dtype=torch.bool)
        solve_importance = torch.zeros(
            batch_size, 5, device=self.device, dtype=torch.float32
        )
        by_idx = self._sufficiency_cache["by_idx"]
        for row_index, (dataset_index, question, spans) in enumerate(
            zip(indices, questions, solve_spans)
        ):
            if dataset_index not in by_idx:
                raise RuntimeError(f"missing sufficiency row {dataset_index}")
            row = by_idx[dataset_index]
            question_sha = hashlib.sha256(
                str(question).encode("utf-8")
            ).hexdigest()
            if row.get("question_sha256") != question_sha:
                raise RuntimeError(
                    f"sufficiency question fingerprint mismatch at {dataset_index}"
                )
            slot_sources = [
                int(end) - 1 if int(end) > int(start) else None
                for start, end in spans
            ]
            expected_sources = [
                -1,
                -1,
                *slot_sources[:4],
                slot_sources[4],
                int(row["n_steps"]) - 1,
            ]
            if list(row["role_source_prefix_index"]) != expected_sources:
                raise RuntimeError(
                    "sufficiency/cache CoT chunk alignment diverged at "
                    f"row {dataset_index}"
                )
            role_scores = torch.tensor(
                row["role_scores"], device=self.device, dtype=torch.float32
            )
            role_mask = torch.tensor(
                row["role_valid_mask"], device=self.device, dtype=torch.bool
            )
            # COMMIT is retained in the cache for an auditable alignment but
            # is never an actor/value action in TRACE-VB.
            role_mask[-1] = False
            targets[row_index] = role_scores
            mask[row_index] = role_mask
            previous_score = 0.0
            previous_valid = bool(role_mask[1].item())
            for solve_index in range(5):
                result_role = solve_index + 2
                if not bool(role_mask[result_role].item()):
                    # Empty short-CoT slots carry no observation; retain the
                    # most recent valid endpoint so a later occupied slot gets
                    # its true incremental importance. Leakage is monotone, so
                    # it cannot create a later valid endpoint to bridge into.
                    continue
                score = float(role_scores[result_role].item())
                if previous_valid:
                    solve_importance[row_index, solve_index] = abs(
                        score - previous_score
                    )
                previous_score = score
                previous_valid = True
        return targets, mask, solve_importance

    def _validate_trace_rl_epoch_budget(self):
        """Validate a four-rank sampled budget over the intact source split."""
        target_questions = int(
            self.trace_rl_config.get(
                "n_train_samples_per_epoch",
                2048,
            )
        )
        if target_questions <= 0:
            raise RuntimeError(
                "Stage-2 n_train_samples_per_epoch must be positive"
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
            target_questions / float(synchronized_batch_size)
        )
        if int(limit_batches) != expected_batches:
            raise RuntimeError(
                "Stage-2 sampled-budget batch mismatch: "
                f"limit_train_batches={limit_batches}, expected "
                f"ceil({target_questions}/{synchronized_batch_size})="
                f"{expected_batches}"
            )
        realized = int(limit_batches) * local_batch_size * world_size
        padding = realized - target_questions
        if padding < 0 or padding >= synchronized_batch_size:
            raise RuntimeError(
                "Stage-2 DDP padding mismatch: "
                f"{limit_batches} batches x {local_batch_size} local batch "
                f"x {world_size} ranks = {realized}, target unique questions "
                f"={target_questions}"
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
        if target_questions > len(all_indices):
            raise RuntimeError(
                "Stage-2 sampled budget exceeds the registered source split: "
                f"budget={target_questions}, dataset={len(all_indices)}"
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

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Return released offload/cache blocks to the host and CUDA driver."""
        interval = int(
            self.trace_config.get(
                "stage1_offload_cache_release_interval",
                0,
            )
        )
        should_release = (
            not self.do_trace_rl
            and bool(
                self.trace_config.get(
                    "stage1_posterior_activation_offload",
                    False,
                )
            )
            and interval > 0
            and (batch_idx + 1) % interval == 0
        )
        if should_release:
            gc.collect()
            try:
                ctypes.CDLL(None).malloc_trim(0)
            except AttributeError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def _decode_single_gold_cots(self, batch) -> List[dict]:
        gold_steps = self._decode_step_lists(batch)
        gold_dependencies = self._decode_cached_matrices(
            batch,
            "dependency_matrix",
        )
        gold_confidences = self._decode_cached_matrices(
            batch,
            "confidence_matrix",
        )
        decoded = []
        for sample_index, steps in enumerate(gold_steps):
            decoded.append(
                {
                    "steps": list(steps),
                    "dependency_matrix": gold_dependencies[sample_index],
                    "confidence_matrix": gold_confidences[sample_index],
                    "source": "original_gsm8k_gold_cot",
                    "annotation_status": (
                        "dataset_gold_not_independently_runtime_verified"
                    ),
                }
            )
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

    def _stage1_posterior_activation_context(self):
        """Offload saved posterior activations without changing gradients."""
        enabled = bool(
            self.trace_config.get(
                "stage1_posterior_activation_offload",
                False,
            )
        )
        if (
            not enabled
            or self.do_trace_rl
            or not self.training
            or not torch.is_grad_enabled()
            or not torch.cuda.is_available()
        ):
            return nullcontext()
        return selective_saved_activation_offload(
            minimum_bytes=int(
                self.trace_config.get(
                    "stage1_posterior_activation_offload_min_bytes",
                    1_048_576,
                )
            ),
            pin_memory=bool(
                self.trace_config.get(
                    "stage1_posterior_activation_offload_pin_memory",
                    False,
                )
            )
        )

    def _trajectory_latents(
        self,
        questions: Sequence[str],
        *,
        deterministic: bool = False,
        posterior_context: Optional[torch.Tensor] = None,
        innovations: Optional[torch.Tensor] = None,
        forced_actions: Optional[torch.Tensor] = None,
        forced_action_mask: Optional[torch.Tensor] = None,
        compute_reference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate one causal latent trajectory per question."""
        if posterior_context is not None:
            raise ValueError(
                "TRACE-VB has no CoT-conditioned posterior; Stage 1 and "
                "deployment both use question-only trajectories"
            )
        self._activate_path_adapter()
        base_model = self.llm.get_base_model()
        backbone = getattr(base_model, "model", None)
        if backbone is None:
            raise RuntimeError(
                "TRACE requires a causal LM exposing its hidden-state backbone "
                "as get_base_model().model"
            )
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
            suffix=(
                self.speed_template.format(1)
                + self.thinking_separator
            ),
        )
        question_embeds = self.embedding(question_ids)
        question_outputs = backbone(
            inputs_embeds=question_embeds,
            attention_mask=question_mask,
            position_ids=self._trace_position_ids(
                question_mask,
                question_embeds.shape[1],
            ),
            output_hidden_states=False,
            use_cache=True,
            return_dict=True,
        )
        cache = question_outputs.past_key_values
        context_mask = question_mask
        previous_state = self.state_norm(
            question_outputs.last_hidden_state[:, -1, :]
        )
        question_state = previous_state
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
        prior_means = []
        prior_log_stds = []
        pre_action_states = []
        action_scale = float(
            self.trace_config.get("action_embedding_scale", 0.10)
        )
        step_scale = float(
            self.trace_config.get("dynamics_step_scale", 0.10)
        )

        for step_index in range(self.n_trace_steps):
            # PPO must evaluate an action at the exact state from which that
            # action was sampled.  Storing this tensor avoids replaying Qwen
            # during the four head-only update epochs.
            pre_action_states.append(previous_state)
            # Lightning wraps training_step in bf16 autocast.  PPO ratios at
            # actor_lr=8e-7 require substantially more mantissa than bf16;
            # keep every Gaussian quantity and stored old log-prob in FP32.
            with torch.autocast(
                device_type=previous_state.device.type,
                enabled=False,
            ):
                prior_mean, prior_log_std = (
                    self.trajectory_policy.distribution_parameters(
                        previous_state.float(),
                        step_index,
                    )
                )
                mean, log_std = prior_mean, prior_log_std
                action, realized_epsilon, action_log_prob = (
                    self.trajectory_policy.realize_action(
                        mean.float(),
                        log_std.float(),
                        step_index,
                        deterministic=deterministic,
                        innovation=(
                            None
                            if innovations is None
                            else innovations[:, step_index].float()
                        ),
                        forced_actions=(
                            None
                            if forced_actions is None
                            else forced_actions[:, step_index].float()
                        ),
                        forced_mask=(
                            None
                            if forced_action_mask is None
                            else forced_action_mask[:, step_index]
                        ),
                    )
                )
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
            outputs = backbone(
                inputs_embeds=current_input.unsqueeze(1),
                attention_mask=context_mask,
                position_ids=self._trace_position_ids(
                    context_mask,
                    1,
                ),
                past_key_values=cache,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            current_state = self.state_norm(
                outputs.last_hidden_state[:, -1, :]
            )
            latent_inputs.append(current_input)
            latent_states.append(current_state)
            residuals.append(current_state - previous_state)
            actions.append(action)
            sampled_innovations.append(realized_epsilon)
            log_probs.append(action_log_prob)
            means.append(mean)
            log_stds.append(log_std)
            prior_means.append(prior_mean)
            prior_log_stds.append(prior_log_std)
            if compute_reference:
                with torch.no_grad():
                    with torch.autocast(
                        device_type=previous_state.device.type,
                        enabled=False,
                    ):
                        ref_mean, ref_log_std = (
                            self.stage1_policy_reference
                            .distribution_parameters(
                                previous_state.detach().float(),
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
            "question_attention_mask": question_mask,
            "question_state": question_state,
            "latent_attention_mask": latent_mask,
            "stochastic_action_mask": (
                self.trajectory_policy.stochastic_action_mask(
                    batch_size,
                    device=latent_mask.device,
                    dtype=latent_mask.dtype,
                )
            ),
            "context_attention_mask": context_mask,
            "past_key_values": cache,
            "latent_states": torch.stack(latent_states, dim=1),
            "pre_action_states": torch.stack(pre_action_states, dim=1),
            "implicit_residuals": torch.stack(residuals, dim=1),
            "actions": torch.stack(actions, dim=1),
            "innovations": torch.stack(sampled_innovations, dim=1),
            "action_log_probs": torch.stack(log_probs, dim=1),
            "action_means": torch.stack(means, dim=1),
            "action_log_stds": torch.stack(log_stds, dim=1),
            "prior_action_means": torch.stack(prior_means, dim=1),
            "prior_action_log_stds": torch.stack(
                prior_log_stds,
                dim=1,
            ),
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

    def _teacher_force_bottleneck(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        target_texts: Sequence[str],
        *,
        include_hybrid_header: Optional[bool] = None,
    ) -> torch.Tensor:
        self._activate_answer_adapter()
        target_ids, target_mask = self._prepare_raw_texts(
            target_texts,
            padding_side="right",
            suffix=self.tokenizer.eos_token,
        )
        separator_ids, separator_mask = self._prompt_ids_for_answer(
            len(target_texts),
            include_hybrid_header=include_hybrid_header,
        )
        current_ids = torch.cat([separator_ids, target_ids], dim=1)
        current_mask = torch.cat([separator_mask, target_mask], dim=1)
        loss_mask = torch.cat(
            [torch.zeros_like(separator_mask), target_mask],
            dim=1,
        )
        attention_mask = build_path_bottleneck_mask(
            trajectory_outputs["question_attention_mask"],
            self._resolve_latent_read_mask(
                trajectory_outputs,
                None,
            ),
            current_mask,
            include_question=self.answer_reads_question,
        )
        source_position_mask = torch.cat(
            [
                trajectory_outputs["context_attention_mask"],
                current_mask,
            ],
            dim=1,
        )
        outputs = self.llm.forward(
            input_ids=current_ids,
            attention_mask=attention_mask,
            position_ids=self._trace_position_ids(
                source_position_mask,
                current_ids.shape[1],
            ),
            past_key_values=self._fork_past_key_values(
                trajectory_outputs["past_key_values"]
            ),
            output_hidden_states=False,
        )
        return self._masked_causal_ce(
            outputs.logits,
            current_ids,
            loss_mask,
        )

    def _prompt_ids_for_answer(
        self,
        batch_size: int,
        *,
        include_hybrid_header: Optional[bool] = None,
    ):
        prompt_ids = torch.full(
            (batch_size, 1),
            fill_value=self.thinking_separator_id,
            device=self.device,
            dtype=torch.long,
        )
        prompt_mask = torch.ones_like(prompt_ids)
        if include_hybrid_header is None:
            include_hybrid_header = bool(
                self.readcot_config.get("use_hybrid", False)
                and self.readcot_config.get(
                    "hybrid_seed_anchor_header",
                    False,
                )
            )
        if include_hybrid_header:
            header_ids, header_mask = self._prepare_raw_texts(
                [self.anchor_header + "\n"] * batch_size,
                padding_side="right",
            )
            prompt_ids = torch.cat([prompt_ids, header_ids], dim=1)
            prompt_mask = torch.cat([prompt_mask, header_mask], dim=1)
        return prompt_ids, prompt_mask

    @staticmethod
    def _commit_only_latent_mask(
        latent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if latent_attention_mask.ndim != 2:
            raise ValueError("latent attention mask must be two-dimensional")
        commit_mask = torch.zeros_like(latent_attention_mask)
        commit_mask[:, -1] = latent_attention_mask[:, -1]
        return commit_mask

    @staticmethod
    def _resolve_latent_read_mask(
        trajectory_outputs: Dict[str, torch.Tensor],
        latent_read_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        available = trajectory_outputs["latent_attention_mask"]
        commit_only = LitTRACEVB._commit_only_latent_mask(available)
        if latent_read_mask is None:
            return commit_only
        if tuple(latent_read_mask.shape) != tuple(available.shape):
            raise ValueError(
                "latent_read_mask must have shape "
                f"{tuple(available.shape)}, got "
                f"{tuple(latent_read_mask.shape)}"
            )
        requested = latent_read_mask.to(
            device=available.device,
            dtype=available.dtype,
        )
        return requested * commit_only

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
        include_hybrid_header: Optional[bool] = None,
    ) -> torch.Tensor:
        self._activate_answer_adapter()
        batch_size = trajectory_outputs["latent_states"].shape[0]
        prompt_ids, prompt_mask = self._prompt_ids_for_answer(
            batch_size,
            include_hybrid_header=include_hybrid_header,
        )
        attention_mask = build_path_bottleneck_mask(
            trajectory_outputs["question_attention_mask"],
            self._resolve_latent_read_mask(
                trajectory_outputs,
                latent_read_mask,
            ),
            prompt_mask,
            include_question=self.answer_reads_question,
        )
        source_position_mask = torch.cat(
            [
                trajectory_outputs["context_attention_mask"],
                prompt_mask,
            ],
            dim=1,
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
        generated = self.llm.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            position_ids=self._trace_position_ids(
                source_position_mask,
                prompt_ids.shape[1],
            ),
            past_key_values=self._fork_past_key_values(
                trajectory_outputs["past_key_values"]
            ),
            **generation_config,
        )
        if (
            generated.shape[1] >= prompt_ids.shape[1]
            and torch.equal(
                generated[:, : prompt_ids.shape[1]],
                prompt_ids,
            )
        ):
            generated = generated[:, prompt_ids.shape[1] :]
        return generated

    def _answer_token_log_probs(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        latent_read_mask: Optional[torch.Tensor] = None,
        include_hybrid_header: Optional[bool] = None,
        decoder_role: str = "deployed",
    ) -> torch.Tensor:
        if decoder_role == "deployed":
            self._activate_answer_adapter()
        elif decoder_role == "stage1_path_value":
            self._activate_path_adapter()
        else:
            raise ValueError(f"unknown decoder_role: {decoder_role}")
        batch_size = answer_input_ids.shape[0]
        prompt_ids, prompt_mask = self._prompt_ids_for_answer(
            batch_size,
            include_hybrid_header=include_hybrid_header,
        )
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
            include_question=self.answer_reads_question,
        )
        source_position_mask = torch.cat(
            [
                trajectory_outputs["context_attention_mask"],
                current_mask,
            ],
            dim=1,
        )
        outputs = self.llm.forward(
            input_ids=current_ids,
            attention_mask=attention_mask,
            position_ids=self._trace_position_ids(
                source_position_mask,
                current_ids.shape[1],
            ),
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
        """Score path interventions with the frozen Stage-1 answer channel.

        Stage 2 deploys a compact-equation decoder. Directly teacher-forcing an
        answer behind that prompt would score a different sequence protocol.
        The Stage-1 answer adapter was trained on answer-only targets and stays
        frozen during Stage 2. It is evaluated with the deployment
        question+COMMIT mask: private role states can affect this score only
        through the recomputed COMMIT state.
        """
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
            include_hybrid_header=False,
            decoder_role="stage1_path_value",
        )
        return (
            log_probs * target_mask
        ).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)

    def _target_token_count(self, target: str) -> int:
        return len(
            self.tokenizer.encode(
                target + self.tokenizer.eos_token,
                add_special_tokens=False,
            )
        )

    def _fit_compact_target_to_generation_budget(
        self,
        target: str,
        answer: str,
    ) -> str:
        """Keep whole compact equations within the deployed token budget."""
        budget = int(
            self.trace_config.get(
                "compact_target_max_new_tokens",
                self.model_kwargs.hybrid_generation_config.max_new_tokens,
            )
        )
        if budget <= 0:
            raise ValueError("compact target token budget must be positive")
        if target.startswith(self.anchor_header + "\n"):
            target = target[len(self.anchor_header) + 1 :]
        answer_suffix = (
            self.thinking_separator + self.answer_template.format(answer)
        )
        marker_index = target.rfind(answer_suffix)
        if marker_index < 0:
            raise ValueError("compact target is missing its protected answer")
        if self._target_token_count(answer_suffix) > budget:
            raise ValueError(
                "answer suffix alone exceeds the deployed generation budget"
            )
        if self._target_token_count(target) <= budget:
            return target

        raw_lines = [
            line.strip()
            for line in target[:marker_index].splitlines()
            if line.strip().startswith("- ")
        ]
        candidates = []
        for line in raw_lines:
            text = line[2:].strip()
            clauses = [
                clause.strip()
                for clause in text.split(";")
                if clause.strip()
            ]
            candidates.extend(f"- {clause}" for clause in clauses)

        kept = []
        for line in candidates:
            candidate = "\n".join(kept + [line, answer_suffix])
            if self._target_token_count(candidate) <= budget:
                kept.append(line)
        fitted = "\n".join(kept + [answer_suffix])
        fitted_length = self._target_token_count(fitted)
        if fitted_length > budget:
            raise RuntimeError(
                f"compact target has {fitted_length} tokens, budget={budget}"
            )
        return fitted

    def _cot_posterior_context(
        self,
        explicit_item: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        states = explicit_item["step_states"].float().detach()
        direction = states[-1] - states[0]
        summary = states.mean(dim=0) + 0.5 * direction
        return self.posterior_context_norm(summary)

    def _single_cot_corridor(
        self,
        explicit_item: Dict[str, torch.Tensor],
        student_path: torch.Tensor,
        student_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Align one sampled path to the single observed CoT corridor.

        The CoT is not resampled and does not define a synthetic target set.
        Each sampled action path defines its own strictly ordered compression
        schedule, then selects a nondecreasing alignment to the same observed
        step sequence using detached path semantics.
        """
        states = explicit_item["step_states"].float()
        queries = self.step_compressor.query_proj(
            student_path.detach().float()
        )
        keys = self.step_compressor.key_proj(states)
        values = self.step_compressor.value_proj(states)
        semantic_scores = (
            F.normalize(queries, dim=-1)
            @ F.normalize(keys, dim=-1).transpose(0, 1)
        )
        centers = action_conditioned_progress_centers(
            student_actions.detach().float(),
            progress_dim=int(
                self.trace_config.get("corridor_progress_action_dim", 0)
            ),
            action_scale=float(
                self.trace_config.get("corridor_progress_action_scale", 1.0)
            ),
        ).to(semantic_scores.dtype)
        assignment = stochastic_monotone_assignment(
            semantic_scores,
            centers,
            sigma=float(
                self.trace_config.get("corridor_progress_sigma", 0.20)
            ),
            progress_strength=float(
                self.trace_config.get(
                    "corridor_progress_strength",
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
            "dependency_probs": dependency_probs,
            "relation_probs": relation_probs,
        }

    def _collect_single_cot_features(
        self,
        questions: Sequence[str],
        gold_cots: Sequence[dict],
    ) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
        self._activate_cot_encoder_adapter()
        explicit_features = self._collect_explicit_batch_features(
            questions=list(questions),
            step_lists=[cot["steps"] for cot in gold_cots],
            # Do not append a separate answer field to the posterior encoder.
            # The original annotated CoT itself may state the final answer.
            answers=[""] * len(gold_cots),
            dependency_matrices=[
                cot.get("dependency_matrix") for cot in gold_cots
            ],
            confidence_matrices=[
                cot.get("confidence_matrix") for cot in gold_cots
            ],
        )
        self._activate_path_adapter()
        contexts = torch.stack(
            [
                self._cot_posterior_context(item)
                for item in explicit_features
            ],
            dim=0,
        )
        return explicit_features, contexts

    def _build_single_cot_corridors(
        self,
        explicit_features: Sequence[Dict[str, torch.Tensor]],
        model_paths: torch.Tensor,
        model_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if model_paths.ndim != 4:
            raise ValueError(
                "Stage-1 paths must have shape [batch, sample, step, hidden]"
            )
        if len(explicit_features) != model_paths.shape[0]:
            raise ValueError("explicit CoTs and sampled path groups must align")
        if model_actions.ndim != 4:
            raise ValueError(
                "Stage-1 actions must have shape [batch, sample, step, action]"
            )
        if model_actions.shape[:3] != model_paths.shape[:3]:
            raise ValueError("sampled paths and actions must align")
        paths = []
        assignments = []
        centers = []
        relation_probs = []
        dependency_probs = []
        dependency_losses = []
        dependency_f1s = []
        assignment_diversities = []
        for sample_index, explicit_item in enumerate(explicit_features):
            sample_paths = []
            sample_assignments = []
            sample_centers = []
            sample_relations = []
            sample_dependency_probs = []
            for student_path, student_actions in zip(
                model_paths[sample_index],
                model_actions[sample_index],
            ):
                corridor = self._single_cot_corridor(
                    explicit_item,
                    student_path,
                    student_actions,
                )
                sample_paths.append(corridor["path"])
                sample_assignments.append(corridor["assignment"])
                sample_centers.append(corridor["progress_centers"])
                sample_relations.append(corridor["relation_probs"])
                sample_dependency_probs.append(
                    corridor["dependency_probs"]
                )
                dependency_losses.append(corridor["dependency_loss"])
                dependency_f1s.append(corridor["dependency_f1"])
            paths.append(torch.stack(sample_paths, dim=0))
            assignments.append(sample_assignments)
            centers.append(torch.stack(sample_centers, dim=0))
            relation_probs.append(sample_relations)
            dependency_probs.append(sample_dependency_probs)
            assignment_diversities.append(
                torch.stack(sample_assignments, dim=0)
                .float()
                .std(dim=0, unbiased=False)
                .mean()
            )

        stacked_centers = torch.stack(centers, dim=0)
        return {
            "paths": torch.stack(paths, dim=0),
            "assignments": assignments,
            "progress_centers": stacked_centers,
            "relation_probs": relation_probs,
            "dependency_probs": dependency_probs,
            "dependency_loss": torch.stack(dependency_losses).mean(),
            "dependency_f1": torch.stack(dependency_f1s).mean(),
            "assignment_diversity": torch.stack(
                assignment_diversities
            ).mean(),
            "progress_schedule_diversity": stacked_centers.float()
            .std(dim=1, unbiased=False)
            .mean(),
        }

    def _build_role_semantic_targets(
        self,
        explicit_features: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, object]:
        """Build label-preserving SOLVE/REFINE targets from one gold CoT.

        PLAN is intentionally *not* aligned to a retrospective summary.  Its
        supervision is the ordered forecast of these five SOLVE targets, which
        makes PLAN's functional meaning testable without role annotations.
        """
        diagnostic_plan_targets = []
        solve_targets = []
        solve_masks = []
        teacher_summaries = []
        solve_spans = []
        for explicit_item in explicit_features:
            states = explicit_item["step_states"].float().detach()
            residuals = explicit_item["step_residuals"].float().detach()
            if states.ndim != 2 or residuals.shape != states.shape:
                raise ValueError(
                    "CoT teacher states/residuals must have [step, hidden] shape"
                )
            # Retained only for the frozen post-hoc geometry protocol used to
            # compare both checkpoints.  The active Stage-1 objective never
            # aligns PLAN to this retrospective summary; it uses forecast loss.
            diagnostic_plan_targets.append(
                states.mean(dim=0) + 0.5 * (states[-1] - states[0])
            )
            chunked = build_contiguous_cot_targets(residuals)
            solve_targets.append(chunked.targets)
            solve_masks.append(chunked.mask)
            teacher_summaries.append(states[-1])
            solve_spans.append(chunked.spans)
        return {
            "plan": torch.stack(diagnostic_plan_targets, dim=0),
            "solve": torch.stack(solve_targets, dim=0),
            "solve_mask": torch.stack(solve_masks, dim=0),
            "summary": torch.stack(teacher_summaries, dim=0),
            "solve_spans": tuple(solve_spans),
        }

    @staticmethod
    def _solve_text_chunk_records(
        gold_cots: Sequence[dict],
        solve_spans: Sequence[Sequence[Tuple[int, int]]],
    ) -> List[Tuple[int, int, str]]:
        """Build sample-local text targets with the exact cached span mapping."""
        if len(gold_cots) != len(solve_spans):
            raise ValueError("gold CoTs and solve spans must align exactly")
        records: List[Tuple[int, int, str]] = []
        for sample_index, (gold_cot, spans) in enumerate(
            zip(gold_cots, solve_spans)
        ):
            steps = list(gold_cot.get("steps", ()))
            if not steps:
                raise ValueError("every training sample must contain a gold CoT")
            if len(spans) != 5:
                raise ValueError("every sample must define five SOLVE spans")
            previous_end = 0
            for solve_index, raw_span in enumerate(spans):
                if len(raw_span) != 2:
                    raise ValueError("a SOLVE span must contain start and end")
                start, end = (int(raw_span[0]), int(raw_span[1]))
                if start != previous_end or not 0 <= start <= end <= len(steps):
                    raise ValueError(
                        "SOLVE spans must be monotone, contiguous, and sample-local"
                    )
                previous_end = end
                if end <= start:
                    continue
                pieces = [str(step).strip() for step in steps[start:end]]
                if any(not piece for piece in pieces):
                    raise ValueError("active CoT steps must contain non-empty text")
                records.append(
                    (sample_index, solve_index, "\n".join(pieces))
                )
            if previous_end != len(steps):
                raise ValueError("SOLVE spans must cover the full gold CoT")
        if not records:
            raise ValueError("no active SOLVE text targets were constructed")
        return records

    def _solve_text_token_records(
        self,
        gold_cots: Sequence[dict],
        solve_spans: Sequence[Sequence[Tuple[int, int]]],
    ) -> List[Tuple[int, int, Tuple[int, ...]]]:
        """Partition every sample-local CoT token exactly once over SOLVE1--5.

        The solve_spans mapping is still validated against the frozen
        teacher's step mapping, but text supervision does not rely on
        punctuation quality. The complete ordered CoT is tokenized
        independently for each sample and split into five balanced,
        contiguous spans. No question, answer, other sample, generated
        rationale, or evaluation record is consulted.
        """
        self._solve_text_chunk_records(gold_cots, solve_spans)
        records: List[Tuple[int, int, Tuple[int, ...]]] = []
        for sample_index, gold_cot in enumerate(gold_cots):
            pieces = [
                str(step).strip()
                for step in gold_cot.get("steps", ())
            ]
            if not pieces or any(not piece for piece in pieces):
                raise ValueError(
                    "every training sample must contain non-empty gold CoT steps"
                )
            token_ids = [
                int(value)
                for value in self.tokenizer.encode(
                    "\n".join(pieces),
                    add_special_tokens=False,
                )
            ]
            if not token_ids:
                raise ValueError("a gold CoT produced no supervision tokens")
            token_spans = contiguous_cot_chunk_spans(
                len(token_ids),
                n_chunks=5,
            )
            reconstructed: List[int] = []
            for solve_index, (start, end) in enumerate(token_spans):
                if end <= start:
                    continue
                chunk = tuple(token_ids[start:end])
                records.append((sample_index, solve_index, chunk))
                reconstructed.extend(chunk)
            if reconstructed != token_ids:
                raise RuntimeError(
                    "SOLVE token spans did not preserve the complete ordered CoT"
                )
        if not records:
            raise ValueError("no active SOLVE token targets were constructed")
        return records

    def _tokenize_solve_text_targets(
        self,
        target_token_ids: Sequence[Sequence[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build teacher-forcing tensors with EOS and fail-closed bounds."""
        if not target_token_ids:
            raise ValueError("at least one SOLVE text target is required")
        max_tokens = int(
            self.trace_config.get("solve_text_decoder_max_tokens", 96)
        )
        if max_tokens < 2:
            raise ValueError("solve_text_decoder_max_tokens must be at least 2")
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise RuntimeError("the tokenizer must define eos_token_id")
        start_id = self.tokenizer.bos_token_id
        if start_id is None:
            start_id = eos_id
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id

        label_rows: List[List[int]] = []
        truncated = 0
        for raw_token_ids in target_token_ids:
            token_ids = [int(value) for value in raw_token_ids]
            if not token_ids:
                raise ValueError("active SOLVE token targets cannot be empty")
            if len(token_ids) > max_tokens - 1:
                token_ids = token_ids[: max_tokens - 1]
                truncated += 1
            label_rows.append([int(value) for value in token_ids] + [int(eos_id)])
        if truncated and bool(
            self.trace_config.get(
                "solve_text_decoder_fail_on_truncation",
                True,
            )
        ):
            raise RuntimeError(
                f"{truncated} SOLVE text targets exceed the configured "
                f"{max_tokens}-token bound; refusing silent CoT truncation"
            )
        token_count = max(len(row) for row in label_rows)
        labels = torch.full(
            (len(label_rows), token_count),
            fill_value=int(pad_id),
            device=self.device,
            dtype=torch.long,
        )
        previous_ids = torch.full_like(labels, fill_value=int(pad_id))
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for row_index, row in enumerate(label_rows):
            row_tensor = torch.tensor(
                row,
                device=self.device,
                dtype=torch.long,
            )
            labels[row_index, : len(row)] = row_tensor
            previous_ids[row_index, 0] = int(start_id)
            if len(row) > 1:
                previous_ids[row_index, 1 : len(row)] = row_tensor[:-1]
            mask[row_index, : len(row)] = True
        truncated_fraction = labels.new_tensor(
            float(truncated) / float(len(label_rows)),
            dtype=torch.float32,
        )
        return previous_ids, labels, mask, truncated_fraction

    def _frozen_lm_head_logits(
        self,
        decoder_states: torch.Tensor,
    ) -> torch.Tensor:
        """Project decoder states without allowing the LM head to learn."""
        output_head = self.llm.get_output_embeddings()
        if output_head is None or not hasattr(output_head, "weight"):
            raise RuntimeError("the language model must expose an output head")
        weight = output_head.weight.detach()
        bias = getattr(output_head, "bias", None)
        if bias is not None:
            bias = bias.detach()
        return F.linear(
            decoder_states.to(dtype=weight.dtype),
            weight,
            bias,
        )

    def _solve_text_decoder_loss(
        self,
        implicit_residuals: torch.Tensor,
        gold_cots: Sequence[dict],
        solve_spans: Sequence[Sequence[Tuple[int, int]]],
        solve_importance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode each active CoT chunk from only its SOLVE transition."""
        if implicit_residuals.ndim != 4:
            raise ValueError(
                "text decoder residuals must have [batch, path, 8, hidden] shape"
            )
        if (
            implicit_residuals.shape[2] != self.n_trace_steps
            or implicit_residuals.shape[3] != self.hidden_size
        ):
            raise ValueError("text decoder received an invalid TRACE path shape")
        records = self._solve_text_token_records(gold_cots, solve_spans)
        batch_size, path_count = implicit_residuals.shape[:2]
        if len(gold_cots) != batch_size:
            raise ValueError("text targets and residual batch must align")
        if solve_importance is not None and tuple(solve_importance.shape) != (
            batch_size,
            5,
        ):
            raise ValueError("solve_importance must have shape [batch, 5]")

        selected_residuals = torch.stack(
            [
                implicit_residuals[sample_index, :, solve_index + 1, :]
                for sample_index, solve_index, _ in records
            ],
            dim=0,
        ).reshape(-1, self.hidden_size)
        target_token_ids = [
            token_ids
            for _, _, token_ids in records
            for _ in range(path_count)
        ]
        role_ids = torch.tensor(
            [
                solve_index
                for _, solve_index, _ in records
                for _ in range(path_count)
            ],
            device=self.device,
            dtype=torch.long,
        )
        record_weights = []
        alpha = float(self.trace_config.get("sufficiency_gain_alpha", 1.0))
        clip = float(self.trace_config.get("sufficiency_gain_clip", 1.0))
        for sample_index, solve_index, _ in records:
            weight = 1.0
            if solve_importance is not None:
                weight += alpha * float(
                    solve_importance[sample_index, solve_index]
                    .detach()
                    .float()
                    .clamp(min=0.0, max=clip)
                    .item()
                )
            record_weights.extend([weight] * path_count)
        weights = selected_residuals.new_tensor(
            record_weights,
            dtype=torch.float32,
        )

        previous_ids, labels, token_mask, truncated_fraction = (
            self._tokenize_solve_text_targets(target_token_ids)
        )
        with torch.no_grad():
            previous_embeddings = self.embedding(previous_ids).detach()
        decoder_states = self.solve_text_decoder(
            selected_residuals,
            previous_embeddings,
            role_ids,
        )
        per_record_loss = checkpointed_projected_token_cross_entropy(
            decoder_states,
            labels,
            token_mask,
            self._frozen_lm_head_logits,
            token_chunk_size=int(
                self.trace_config.get("solve_text_ce_chunk_size", 8)
            ),
        )
        normalized_weights = weights / weights.sum().clamp_min(1.0)
        loss = (per_record_loss * normalized_weights).sum()
        return {
            "loss": loss,
            "active_chunks": loss.new_tensor(float(len(records))),
            "tokens_per_chunk": token_mask.float().sum(dim=-1).mean().detach(),
            "truncated_fraction": truncated_fraction.detach(),
        }

    def _role_semantic_losses(
        self,
        latent_states: torch.Tensor,
        implicit_residuals: torch.Tensor,
        role_targets: Dict[str, object],
        solve_importance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute PLAN forecast, SOLVE, and REFINE formation losses."""
        if latent_states.ndim != 4 or implicit_residuals.shape != latent_states.shape:
            raise ValueError(
                "role semantic paths must have [batch, path, 8, hidden] shape"
            )
        if latent_states.shape[2] != self.n_trace_steps:
            raise ValueError("role semantic loss received a non-eight-step path")
        batch_size, path_count = latent_states.shape[:2]
        projected_solve = F.linear(
            role_targets["solve"].float(),
            self.semantic_projection.float(),
        )
        projected_solve = projected_solve[:, None, :, :].expand(
            batch_size, path_count, -1, -1
        )
        forecast = self.plan_forecaster(latent_states[:, :, 0, :])
        forecast_mask = role_targets["solve_mask"][:, None, :].expand(
            batch_size, path_count, -1
        )
        plan_forecast_loss = masked_cosine_loss(
            forecast,
            projected_solve,
            forecast_mask,
        )

        solve_target = role_targets["solve"][:, None, :, :].expand(
            batch_size,
            path_count,
            -1,
            -1,
        )
        solve_mask = role_targets["solve_mask"][:, None, :].expand(
            batch_size,
            path_count,
            -1,
        )
        solve_scores = masked_cosine_similarity(
            implicit_residuals[:, :, 1:6, :], solve_target, solve_mask
        )
        solve_weights = solve_mask.to(dtype=solve_scores.dtype)
        if solve_importance is not None:
            if tuple(solve_importance.shape) != (batch_size, 5):
                raise ValueError("solve_importance must have shape [batch, 5]")
            alpha = float(
                self.trace_config.get("sufficiency_gain_alpha", 1.0)
            )
            clip = float(
                self.trace_config.get("sufficiency_gain_clip", 1.0)
            )
            importance = 1.0 + alpha * solve_importance.float().clamp(
                min=0.0, max=clip
            )
            solve_weights = solve_weights * importance[:, None, :]
        solve_loss = (
            (1.0 - solve_scores) * solve_weights
        ).sum() / solve_weights.sum().clamp_min(1.0)

        correction_target = (
            role_targets["summary"][:, None, :]
            - latent_states[:, :, 5, :].detach()
        )
        refine_mask = torch.ones(
            batch_size,
            path_count,
            device=latent_states.device,
            dtype=torch.bool,
        )
        refine_loss = masked_cosine_loss(
            implicit_residuals[:, :, 6, :],
            correction_target,
            refine_mask,
        )
        return {
            "plan_forecast": plan_forecast_loss,
            "solve": solve_loss,
            "refine": refine_loss,
        }

    def _semantic_anchor_weight(self) -> float:
        """Cosine-decayed Stage-2 semantic regularizer after critic warm-up."""
        if not bool(self.trace_rl_config.get("use_semantic_anchor", True)):
            return 0.0
        initial = float(
            self.trace_rl_config.get("semantic_anchor_initial_weight", 0.02)
        )
        minimum = float(
            self.trace_rl_config.get("semantic_anchor_minimum_weight", 0.002)
        )
        decay_batches = int(
            self.trace_rl_config.get("semantic_anchor_decay_batches", 1536)
        )
        if (
            not math.isfinite(initial)
            or not math.isfinite(minimum)
            or initial < minimum
            or minimum < 0.0
            or decay_batches <= 0
        ):
            raise ValueError(
                "semantic anchor weights must be finite with "
                "initial >= minimum >= 0 and positive decay_batches"
            )
        warmup_batches = int(
            self.trace_rl_config.get("critic_warmup_batches", 256)
        )
        batches_seen = int(self.vb_rollout_batches_seen.item())
        if batches_seen < warmup_batches:
            return 0.0
        progress = min(
            1.0,
            max(
                0.0,
                float(batches_seen - warmup_batches)
                / float(decay_batches),
            ),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum + (initial - minimum) * cosine

    def _stage2_role_semantic_anchor(
        self,
        batch,
    ) -> Dict[str, torch.Tensor]:
        """Re-anchor the deterministic mean path without shaping RL reward."""
        if not self.do_trace_rl or not self.training:
            raise RuntimeError("semantic anchoring is training-only Stage 2")
        if not torch.is_grad_enabled():
            raise RuntimeError("semantic anchoring requires actor gradients")
        questions = list(batch["question"])
        gold_cots = self._decode_single_gold_cots(batch)
        with torch.no_grad():
            explicit_features, _ = self._collect_single_cot_features(
                questions,
                gold_cots,
            )
            role_targets = self._build_role_semantic_targets(
                explicit_features
            )
        mean_outputs = self._trajectory_latents(
            questions,
            deterministic=True,
        )
        mean_states = mean_outputs["latent_states"].unsqueeze(1)
        mean_residuals = mean_outputs["implicit_residuals"].unsqueeze(1)
        semantic = self._role_semantic_losses(
            mean_states,
            mean_residuals,
            role_targets,
        )
        solve_text = self._solve_text_decoder_loss(
            mean_residuals,
            gold_cots,
            role_targets["solve_spans"],
        )

        component_weights = {
            "plan": float(
                self.trace_rl_config.get("semantic_anchor_plan_weight", 0.25)
            ),
            "solve_text": float(
                self.trace_rl_config.get(
                    "semantic_anchor_solve_text_weight",
                    0.50,
                )
            ),
            "refine": float(
                self.trace_rl_config.get("semantic_anchor_refine_weight", 0.25)
            ),
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in component_weights.values()
        ):
            raise ValueError("semantic anchor component weights must be finite")
        weight_sum = sum(component_weights.values())
        if weight_sum <= 0.0:
            raise ValueError("semantic anchor requires a positive component")
        total = (
            component_weights["plan"] * semantic["plan_forecast"]
            + component_weights["solve_text"] * solve_text["loss"]
            + component_weights["refine"] * semantic["refine"]
        ) / weight_sum
        return {
            "total": total,
            "plan": semantic["plan_forecast"].detach(),
            "solve_text": solve_text["loss"].detach(),
            "refine": semantic["refine"].detach(),
            "truncated_fraction": solve_text[
                "truncated_fraction"
            ].detach(),
        }

    def _role_step_rewards(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        role_targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return post-hoc formation scores and zero for COMMIT.

        These values are never RL rewards. PLAN is scored by its actual
        five-target forecast objective; SOLVE/REFINE use the same frozen-CoT
        semantic targets as Stage 1.
        """
        states = trajectory_outputs["latent_states"]
        residuals = trajectory_outputs["implicit_residuals"]
        if states.ndim != 3 or states.shape[1] != self.n_trace_steps:
            raise ValueError("role rewards require [rollout, 8, hidden] states")
        rewards = states.new_zeros(states.shape[0], self.n_trace_steps).float()
        active = torch.ones(
            states.shape[0],
            device=states.device,
            dtype=torch.bool,
        )
        plan_forecasts = self.plan_forecaster(states[:, 0, :])
        plan_targets = F.linear(
            role_targets["solve"].float(),
            self.semantic_projection.float(),
        )
        plan_scores = masked_cosine_similarity(
            plan_forecasts,
            plan_targets,
            role_targets["solve_mask"],
        )
        plan_weights = role_targets["solve_mask"].float()
        rewards[:, 0] = (
            plan_scores * plan_weights
        ).sum(dim=-1) / plan_weights.sum(dim=-1).clamp_min(1.0)
        rewards[:, 1:6] = masked_cosine_similarity(
            residuals[:, 1:6, :],
            role_targets["solve"],
            role_targets["solve_mask"],
        )
        correction_target = (
            role_targets["summary"] - states[:, 5, :].detach()
        )
        rewards[:, 6] = masked_cosine_similarity(
            residuals[:, 6, :],
            correction_target,
            active,
        )
        return rewards

    def _build_role_compact_targets(
        self,
        gold_cots: Sequence[dict],
        answers: Sequence[str],
        solve_spans: Sequence[Sequence[Tuple[int, int]]],
    ) -> List[str]:
        """Create compact textual SFT targets from the same five CoT chunks."""
        targets = []
        for cot, answer, spans in zip(gold_cots, answers, solve_spans):
            steps = cot["steps"]
            anchor_lines = []
            for start, end in spans:
                if end <= start:
                    continue
                anchor_text = self._format_anchor_step(steps[end - 1])
                if anchor_text:
                    anchor_lines.append(f"- {anchor_text}")
            if not anchor_lines:
                anchor_lines = ["- no compact equation"]
            target = (
                self.anchor_header
                + "\n"
                + "\n".join(anchor_lines)
                + "\n"
                + self.thinking_separator
                + self.answer_template.format(answer)
            )
            targets.append(
                self._fit_compact_target_to_generation_budget(target, answer)
            )
        return targets

    @staticmethod
    def _masked_role_kl(
        means: torch.Tensor,
        log_stds: torch.Tensor,
        reference_means: torch.Tensor,
        reference_log_stds: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        per_step = diagonal_gaussian_kl(
            means,
            log_stds,
            reference_means,
            reference_log_stds,
        )
        weights = mask.to(device=per_step.device, dtype=per_step.dtype)
        return (per_step * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(self, batch):
        """Run Stage 1 with offload restricted to bounded heavy branches."""
        return self._stage1_forward_impl(batch)

    def _stage1_forward_impl(self, batch):
        """Stage 1: form a deployable role program from one observed CoT.

        The two paths are question-only and share all parameters: one sampled
        path supplies robust formation/variance gradients and one conditional
        mean path exactly matches inference.  Only the mean COMMIT path is
        teacher-forced to the gold answer.
        """
        if self.do_trace_rl:
            raise RuntimeError("Stage-1 forward is not used during latent RL")
        questions = list(batch["question"])
        answers = list(batch["answer"])
        gold_cots = self._decode_single_gold_cots(batch)
        explicit_features, _ = self._collect_single_cot_features(
            questions, gold_cots
        )
        role_targets = self._build_role_semantic_targets(explicit_features)
        sufficiency_targets, sufficiency_mask, solve_importance = (
            self._batch_sufficiency_targets(
                batch, role_targets["solve_spans"]
            )
        )

        stochastic_paths = int(
            self.trace_config.get("stage1_stochastic_paths", 1)
        )
        if stochastic_paths != 1:
            raise ValueError(
                "TRACE-VB requires exactly one stochastic Stage-1 path"
            )
        # All Stage-1 activations stay on GPU. The former saved-tensor CPU
        # offload leaked roughly 30 MiB of host RSS per optimizer step in the
        # real four-rank workload; the matched GPU-resident run was flat.
        sampled_outputs = self._trajectory_latents(
            questions, deterministic=False
        )
        map_outputs = self._trajectory_latents(
            questions, deterministic=True
        )
        formation_states = torch.stack(
            [
                sampled_outputs["latent_states"],
                map_outputs["latent_states"],
            ],
            dim=1,
        )
        formation_residuals = torch.stack(
            [
                sampled_outputs["implicit_residuals"],
                map_outputs["implicit_residuals"],
            ],
            dim=1,
        )
        semantic_losses = self._role_semantic_losses(
            formation_states,
            formation_residuals,
            role_targets,
            solve_importance=solve_importance,
        )
        solve_text = self._solve_text_decoder_loss(
            formation_residuals,
            gold_cots,
            role_targets["solve_spans"],
            solve_importance=solve_importance,
        )
        efficacy = action_efficacy_hinge(
            sampled_outputs["actions"],
            map_outputs["actions"],
            sampled_outputs["implicit_residuals"],
            map_outputs["implicit_residuals"],
            sampled_outputs["stochastic_action_mask"].bool(),
            minimum_ratio=float(
                self.trace_config.get(
                    "stage1_minimum_action_efficacy_ratio", 0.02
                )
            ),
        )

        answer_targets = [
            self.answer_template.format(answer) for answer in answers
        ]
        answer_loss = self._teacher_force_bottleneck(
            map_outputs,
            answer_targets,
            include_hybrid_header=False,
        )

        formation_pre_states = torch.stack(
            [
                sampled_outputs["pre_action_states"],
                map_outputs["pre_action_states"],
            ],
            dim=1,
        )
        batch_size, path_count = formation_pre_states.shape[:2]
        role_ids = torch.arange(
            self.n_trace_steps, device=self.device, dtype=torch.long
        ).view(1, 1, -1).expand(batch_size, path_count, -1)
        sufficiency_logits = self.sufficiency_head(
            formation_pre_states, role_ids
        )
        expanded_targets = sufficiency_targets[:, None, :].expand_as(
            sufficiency_logits
        )
        expanded_mask = sufficiency_mask[:, None, :].expand_as(
            sufficiency_logits
        )
        sufficiency_terms = F.binary_cross_entropy_with_logits(
            sufficiency_logits.float(),
            expanded_targets.float(),
            reduction="none",
        )
        sufficiency_weights = expanded_mask.to(sufficiency_terms.dtype)
        sufficiency_loss = (
            sufficiency_terms * sufficiency_weights
        ).sum() / sufficiency_weights.sum().clamp_min(1.0)

        minimum_std = float(
            self.trace_config.get("stage1_minimum_action_std", 0.15)
        )
        variance_floor_loss = minimum_action_entropy_loss(
            sampled_outputs["action_log_stds"][:, :7, :],
            minimum_std=minimum_std,
        )
        answer_weight = float(
            self.trace_config.get("stage1_answer_weight", 1.0)
        )
        plan_weight = float(
            self.trace_config.get("stage1_plan_forecast_weight", 0.10)
        )
        solve_weight = float(
            self.trace_config.get("stage1_solve_weight", 0.20)
        )
        solve_text_weight = float(
            self.trace_config.get("stage1_solve_text_weight", 0.20)
        )
        refine_weight = float(
            self.trace_config.get("stage1_refine_weight", 0.10)
        )
        sufficiency_weight = float(
            self.trace_config.get("stage1_sufficiency_weight", 0.10)
        )
        variance_weight = float(
            self.trace_config.get("stage1_variance_weight", 0.005)
        )
        efficacy_weight = float(
            self.trace_config.get("stage1_action_efficacy_weight", 0.02)
        )
        total_loss = (
            answer_weight * answer_loss
            + plan_weight * semantic_losses["plan_forecast"]
            + solve_weight * semantic_losses["solve"]
            + solve_text_weight * solve_text["loss"]
            + refine_weight * semantic_losses["refine"]
            + sufficiency_weight * sufficiency_loss
            + variance_weight * variance_floor_loss
            + efficacy_weight * efficacy.loss
        )
        active_sufficiency = sufficiency_weights.sum().detach()
        sufficiency_probabilities = torch.sigmoid(
            sufficiency_logits.float()
        )
        sufficiency_mae = (
            (sufficiency_probabilities - expanded_targets).abs()
            * sufficiency_weights
        ).sum() / active_sufficiency.clamp_min(1.0)
        sampled_std = torch.exp(
            sampled_outputs["action_log_stds"][:, :7, :].float()
        )
        return {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "trace_vb_plan_forecast_loss": semantic_losses[
                "plan_forecast"
            ],
            "trace_vb_solve_loss": semantic_losses["solve"],
            "trace_vb_refine_loss": semantic_losses["refine"],
            "trace_vb_solve_text_loss": solve_text["loss"],
            "trace_vb_solve_text_active_chunks": solve_text["active_chunks"],
            "trace_vb_solve_text_tokens_per_chunk": solve_text["tokens_per_chunk"],
            "trace_vb_solve_text_truncated_fraction": solve_text[
                "truncated_fraction"
            ],
            "trace_vb_sufficiency_loss": sufficiency_loss,
            "trace_vb_sufficiency_mae": sufficiency_mae.detach(),
            "trace_vb_sufficiency_active": active_sufficiency,
            "trace_vb_variance_floor_loss": variance_floor_loss,
            "trace_vb_action_efficacy_loss": efficacy.loss,
            "trace_vb_action_efficacy_ratio": (
                efficacy.active_efficacy_ratio.detach()
            ),
            "trace_vb_action_std_mean": sampled_std.mean().detach(),
            "trace_vb_action_std_min": sampled_std.min().detach(),
            "trace_vb_mean_path_answer_only": total_loss.new_ones(()),
            "trace_vb_question_only_paths": total_loss.new_ones(()),
            "trace_vb_question_commit_readout": total_loss.new_ones(()),
            "trace_vb_private_latent_answer_access": total_loss.new_zeros(()),
            "trace_vb_stochastic_path_count": total_loss.new_ones(()),
            "trace_answer_question_access": total_loss.new_ones(()),
            "lambda_answer_eff": total_loss.new_tensor(answer_weight),
            "lambda_plan_forecast_eff": total_loss.new_tensor(plan_weight),
            "lambda_solve_eff": total_loss.new_tensor(solve_weight),
            "lambda_refine_eff": total_loss.new_tensor(refine_weight),
            "lambda_solve_text_eff": total_loss.new_tensor(solve_text_weight),
            "lambda_sufficiency_eff": total_loss.new_tensor(
                sufficiency_weight
            ),
            "lambda_variance_eff": total_loss.new_tensor(variance_weight),
            "lambda_action_efficacy_eff": total_loss.new_tensor(
                efficacy_weight
            ),
        }

    def _legacy_role_forward(self, batch):
        """Deprecated predecessor retained only for checkpoint forensics."""
        if self.do_trace_rl:
            raise RuntimeError("Stage-1 forward is not used during latent RL")
        questions = list(batch["question"])
        answers = list(batch["answer"])
        gold_cots = self._decode_single_gold_cots(batch)
        explicit_features, cot_contexts = self._collect_single_cot_features(
            questions,
            gold_cots,
        )
        role_targets = self._build_role_semantic_targets(explicit_features)
        sample_count = int(
            self.trace_config.get("stage1_posterior_samples", 3)
        )
        if sample_count < 2:
            raise ValueError("Stage 1 requires at least two posterior paths")
        repeated_questions = [
            question
            for question in questions
            for _ in range(sample_count)
        ]
        repeated_contexts = cot_contexts.repeat_interleave(
            sample_count,
            dim=0,
        )
        with self._stage1_posterior_activation_context():
            sampled_outputs = self._trajectory_latents(
                repeated_questions,
                deterministic=False,
                posterior_context=repeated_contexts,
            )
        map_outputs = self._trajectory_latents(
            questions,
            deterministic=True,
        )

        batch_size = len(questions)
        sampled_states = sampled_outputs["latent_states"].view(
            batch_size,
            sample_count,
            self.n_trace_steps,
            self.hidden_size,
        )
        sampled_residuals = sampled_outputs["implicit_residuals"].view(
            batch_size,
            sample_count,
            self.n_trace_steps,
            self.hidden_size,
        )
        formation_states = torch.cat(
            [sampled_states, map_outputs["latent_states"].unsqueeze(1)],
            dim=1,
        )
        formation_residuals = torch.cat(
            [sampled_residuals, map_outputs["implicit_residuals"].unsqueeze(1)],
            dim=1,
        )
        semantic_losses = self._role_semantic_losses(
            formation_states,
            formation_residuals,
            role_targets,
        )

        sampled_answer_targets = [
            self.answer_template.format(answer)
            for answer in answers
            for _ in range(sample_count)
        ]
        with self._stage1_posterior_activation_context():
            sampled_answer_loss = self._teacher_force_bottleneck(
                sampled_outputs,
                sampled_answer_targets,
                include_hybrid_header=False,
            )
        compact_targets = self._build_role_compact_targets(
            gold_cots,
            answers,
            role_targets["solve_spans"],
        )
        compact_target_lengths = torch.tensor(
            [self._target_token_count(target) for target in compact_targets],
            device=self.device,
            dtype=torch.float32,
        )
        map_compact_loss = self._teacher_force_bottleneck(
            map_outputs,
            compact_targets,
            include_hybrid_header=True,
        )
        sampled_answer_weight = float(
            self.trace_config.get("stage1_sampled_answer_weight", 0.35)
        )
        map_compact_weight = float(
            self.trace_config.get("stage1_map_compact_weight", 1.0)
        )
        answer_loss = (
            sampled_answer_weight * sampled_answer_loss
            + map_compact_weight * map_compact_loss
        )

        posterior_kl = self._masked_role_kl(
            sampled_outputs["action_means"],
            sampled_outputs["action_log_stds"],
            sampled_outputs["prior_action_means"],
            sampled_outputs["prior_action_log_stds"],
            sampled_outputs["stochastic_action_mask"],
        )
        minimum_stds = tuple(
            float(value)
            for value in self.trace_config.get(
                "stage1_role_minimum_stds",
                (0.30, 0.25, 0.22, 0.19, 0.16, 0.13, 0.08),
            )
        )
        if len(minimum_stds) != self.n_trace_steps - 1:
            raise ValueError("stage1_role_minimum_stds must contain 7 values")
        role_entropy_terms = []
        for step_index, minimum_std in enumerate(minimum_stds):
            role_entropy_terms.append(
                0.5
                * (
                    minimum_action_entropy_loss(
                        sampled_outputs["action_log_stds"][:, step_index],
                        minimum_std=minimum_std,
                    )
                    + minimum_action_entropy_loss(
                        sampled_outputs["prior_action_log_stds"][:, step_index],
                        minimum_std=minimum_std,
                    )
                )
            )
        role_entropy_loss = torch.stack(role_entropy_terms).mean()

        plan_weight = float(
            self.trace_config.get("stage1_plan_weight", 0.08)
        )
        solve_weight = float(
            self.trace_config.get("stage1_solve_weight", 0.20)
        )
        check_weight = float(
            self.trace_config.get("stage1_check_weight", 0.10)
        )
        posterior_kl_weight = float(
            self.trace_config.get("stage1_posterior_kl_weight", 0.05)
        )
        role_entropy_weight = float(
            self.trace_config.get("stage1_role_entropy_weight", 0.005)
        )
        total_loss = (
            answer_loss
            + plan_weight * semantic_losses["plan"]
            + solve_weight * semantic_losses["solve"]
            + check_weight * semantic_losses["check"]
            + posterior_kl_weight * posterior_kl
            + role_entropy_weight * role_entropy_loss
        )
        posterior_std = torch.exp(
            sampled_outputs["action_log_stds"][:, :7].float()
        )
        prior_std = torch.exp(
            sampled_outputs["prior_action_log_stds"][:, :7].float()
        )
        return {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "trace_stage1_sampled_answer_loss": sampled_answer_loss,
            "trace_stage1_map_compact_loss": map_compact_loss,
            "trace_stage1_plan_loss": semantic_losses["plan"],
            "trace_stage1_solve_loss": semantic_losses["solve"],
            "trace_stage1_check_loss": semantic_losses["check"],
            "trace_stage1_posterior_prior_kl": posterior_kl,
            "trace_stage1_role_entropy_loss": role_entropy_loss,
            "trace_stage1_compact_target_tokens": (
                compact_target_lengths.mean().detach()
            ),
            "trace_stage1_compact_target_tokens_max": (
                compact_target_lengths.max().detach()
            ),
            "trace_stage1_posterior_std": posterior_std.mean().detach(),
            "trace_stage1_prior_std": prior_std.mean().detach(),
            "trace_stage1_role_contract": total_loss.new_ones(()),
            "trace_stage1_commit_is_deterministic": total_loss.new_ones(()),
            "trace_stage1_commit_only_readout": total_loss.new_ones(()),
            "trace_stage1_iid_posterior_samples": total_loss.new_tensor(
                float(sample_count)
            ),
            "trace_answer_question_access": total_loss.new_tensor(
                float(self.answer_reads_question)
            ),
            "lambda_plan_eff": total_loss.new_tensor(plan_weight),
            "lambda_solve_eff": total_loss.new_tensor(solve_weight),
            "lambda_check_eff": total_loss.new_tensor(check_weight),
            "lambda_posterior_kl_eff": total_loss.new_tensor(
                posterior_kl_weight
            ),
            "lambda_role_entropy_eff": total_loss.new_tensor(
                role_entropy_weight
            ),
            "lambda_sampled_answer_eff": total_loss.new_tensor(
                sampled_answer_weight
            ),
            "lambda_map_compact_eff": total_loss.new_tensor(
                map_compact_weight
            ),
        }

    def _legacy_corridor_forward(self, batch):
        """Stage 1: form a question-only prior from one-CoT path posterior."""
        if self.do_trace_rl:
            raise RuntimeError(
                "Stage-1 forward is not used as replay during Stage 2"
            )
        questions = list(batch["question"])
        answers = list(batch["answer"])
        gold_cots = self._decode_single_gold_cots(batch)
        explicit_features, cot_contexts = self._collect_single_cot_features(
            questions,
            gold_cots,
        )
        sample_count = int(
            self.trace_config.get("stage1_posterior_samples", 3)
        )
        if sample_count < 2:
            raise ValueError(
                "Stage 1 requires at least two IID posterior paths"
            )
        repeated_questions = [
            question
            for question in questions
            for _ in range(sample_count)
        ]
        repeated_contexts = cot_contexts.repeat_interleave(
            sample_count,
            dim=0,
        )
        with self._stage1_posterior_activation_context():
            model_outputs = self._trajectory_latents(
                repeated_questions,
                deterministic=False,
                posterior_context=repeated_contexts,
            )
        map_outputs = self._trajectory_latents(
            questions,
            deterministic=True,
        )
        model_paths = model_outputs["implicit_residuals"].view(
            len(questions),
            sample_count,
            self.n_trace_steps,
            self.hidden_size,
        )
        action_groups = model_outputs["actions"].view(
            len(questions),
            sample_count,
            self.n_trace_steps,
            self.trajectory_policy.action_dim,
        )
        map_paths = map_outputs["implicit_residuals"].unsqueeze(1)
        map_actions = map_outputs["actions"].unsqueeze(1)
        formation_paths = torch.cat([model_paths, map_paths], dim=1)
        formation_actions = torch.cat(
            [action_groups, map_actions],
            dim=1,
        )
        corridors = self._build_single_cot_corridors(
            explicit_features,
            formation_paths,
            formation_actions,
        )
        corridor_paths = corridors["paths"]
        distance = trajectory_distance_components(
            formation_paths,
            corridor_paths.detach(),
            **self._distance_kwargs(),
        )
        noncollapse = path_noncollapse_loss(
            formation_paths,
            margin=float(
                self.trace_config.get("path_noncollapse_margin", 0.02)
            ),
        )
        predicted_actions = self.transition_action_decoder(
            model_paths.float()
        )
        action_identifiability = action_transition_identifiability_loss(
            predicted_actions,
            action_groups,
        )
        sampled_answer_targets = [
            self.answer_template.format(answer)
            for answer in answers
            for _ in range(sample_count)
        ]
        with self._stage1_posterior_activation_context():
            sampled_answer_loss = self._teacher_force_bottleneck(
                model_outputs,
                sampled_answer_targets,
                include_hybrid_header=False,
            )
        # The compact deployment target is selected from the same dynamic,
        # action-conditioned corridor as the conditional-mean deployment path.
        # It therefore contains only equations from the original CoT without
        # invoking the inherited fixed latent-query compressor.
        compression_outputs = {
            "aggregated_explicit_residuals": corridor_paths[:, -1],
            "assignments": [
                sample_assignments[-1]
                for sample_assignments in corridors["assignments"]
            ],
            "dependency_probs": [
                sample_dependencies[-1]
                for sample_dependencies in corridors["dependency_probs"]
            ],
        }
        compact_targets, _ = self._build_hybrid_targets(
            step_lists=[cot["steps"] for cot in gold_cots],
            answers=answers,
            explicit_features=explicit_features,
            compression_outputs=compression_outputs,
            implicit_residuals=map_outputs["implicit_residuals"],
        )
        if self.readcot_config.get(
            "hybrid_seed_anchor_header",
            False,
        ):
            prefix = self.anchor_header + "\n"
            compact_targets = [
                target[len(prefix) :]
                if target.startswith(prefix)
                else target
                for target in compact_targets
            ]
        compact_targets = [
            self._fit_compact_target_to_generation_budget(target, answer)
            for target, answer in zip(compact_targets, answers)
        ]
        compact_target_lengths = torch.tensor(
            [
                self._target_token_count(target)
                for target in compact_targets
            ],
            device=self.device,
            dtype=torch.float32,
        )
        map_compact_loss = self._teacher_force_bottleneck(
            map_outputs,
            compact_targets,
            include_hybrid_header=True,
        )
        sampled_answer_weight = float(
            self.trace_config.get(
                "stage1_sampled_answer_weight",
                0.35,
            )
        )
        map_compact_weight = self._scheduled_loss_weight(
            "anchor",
            float(
                self.trace_config.get(
                    "stage1_map_compact_weight",
                    1.0,
                )
            ),
        )
        answer_loss = (
            sampled_answer_weight * sampled_answer_loss
            + map_compact_weight * map_compact_loss
        )
        noncollapse_weight = float(
            self.trace_config.get("stage1_noncollapse_weight", 0.05)
        )
        action_identifiability_weight = float(
            self.trace_config.get(
                "stage1_action_identifiability_weight",
                0.05,
            )
        )
        formation = (
            distance["total"].mean()
            + noncollapse_weight * noncollapse
            + action_identifiability_weight * action_identifiability
        )
        formation_weight = float(
            self.trace_config.get("stage1_formation_weight", 0.14)
        )
        dependency_weight = self._scheduled_loss_weight(
            "dep",
            self.readcot_config.get("lambda_dep", 0.04),
        )
        posterior_kl = diagonal_gaussian_kl(
            model_outputs["action_means"],
            model_outputs["action_log_stds"],
            model_outputs["prior_action_means"],
            model_outputs["prior_action_log_stds"],
        ).mean()
        posterior_kl_weight = float(
            self.trace_config.get("stage1_posterior_kl_weight", 0.05)
        )
        minimum_action_std = float(
            self.trace_config.get("stage1_minimum_action_std", 0.20)
        )
        action_entropy_floor = 0.5 * (
            minimum_action_entropy_loss(
                model_outputs["action_log_stds"],
                minimum_std=minimum_action_std,
            )
            + minimum_action_entropy_loss(
                model_outputs["prior_action_log_stds"],
                minimum_std=minimum_action_std,
            )
        )
        entropy_weight = float(
            self.trace_config.get(
                "stage1_entropy_floor_weight",
                0.02,
            )
        )
        total_loss = (
            answer_loss
            + formation_weight * formation
            + posterior_kl_weight * posterior_kl
            + entropy_weight * action_entropy_floor
            + dependency_weight * corridors["dependency_loss"]
        )
        posterior_std = torch.exp(
            model_outputs["action_log_stds"].float()
        )
        prior_std = torch.exp(
            model_outputs["prior_action_log_stds"].float()
        )
        return {
            "total_loss": total_loss,
            "answer_loss": answer_loss,
            "trace_stage1_sampled_answer_loss": sampled_answer_loss,
            "trace_stage1_map_compact_loss": map_compact_loss,
            "trace_stage1_compact_target_tokens": (
                compact_target_lengths.mean().detach()
            ),
            "trace_stage1_compact_target_tokens_max": (
                compact_target_lengths.max().detach()
            ),
            "dep_loss": corridors["dependency_loss"],
            "dep_f1": corridors["dependency_f1"],
            "trace_stage1_formation_loss": formation,
            "trace_stage1_corridor_loss": distance["total"].mean(),
            "trace_stage1_sampled_corridor_loss": (
                distance["total"][:, :sample_count].mean()
            ),
            "trace_stage1_map_corridor_loss": (
                distance["total"][:, -1].mean()
            ),
            "trace_stage1_position_loss": distance["position"].mean(),
            "trace_stage1_direction_loss": distance["direction"].mean(),
            "trace_stage1_step_loss": distance["step"].mean(),
            "trace_stage1_noncollapse_loss": noncollapse,
            "trace_stage1_action_identifiability_loss": (
                action_identifiability
            ),
            "trace_stage1_action_entropy_floor_loss": (
                action_entropy_floor
            ),
            "trace_stage1_posterior_prior_kl": posterior_kl,
            "trace_stage1_action_path_correlation": (
                pairwise_action_path_correlation(
                    action_groups.detach(),
                    model_paths.detach(),
                )
            ),
            "trace_stage1_posterior_std": posterior_std.mean().detach(),
            "trace_stage1_posterior_std_min": posterior_std.min().detach(),
            "trace_stage1_prior_std": prior_std.mean().detach(),
            "trace_stage1_single_cot_contract": total_loss.new_ones(()),
            "trace_stage1_iid_posterior_samples": total_loss.new_tensor(
                float(sample_count)
            ),
            "trace_stage1_map_supervised_paths": total_loss.new_ones(()),
            "trace_stage1_map_structure_supervised": (
                total_loss.new_ones(())
            ),
            "trace_stage1_structure_supervised_paths": (
                total_loss.new_tensor(float(sample_count + 1))
            ),
            "trace_stage1_corridor_progress_span": (
                corridors["progress_centers"][..., -1]
                - corridors["progress_centers"][..., 0]
            ).mean().detach(),
            "trace_stage1_alignment_diversity": corridors[
                "assignment_diversity"
            ].detach(),
            "trace_stage1_progress_schedule_diversity": corridors[
                "progress_schedule_diversity"
            ].detach(),
            "trace_answer_question_access": total_loss.new_tensor(
                float(self.answer_reads_question)
            ),
            "lambda_trace_formation_eff": total_loss.new_tensor(
                formation_weight
            ),
            "lambda_dep_eff": total_loss.new_tensor(dependency_weight),
            "lambda_posterior_kl_eff": total_loss.new_tensor(
                posterior_kl_weight
            ),
            "lambda_entropy_floor_eff": total_loss.new_tensor(
                entropy_weight
            ),
            "lambda_sampled_answer_eff": total_loss.new_tensor(
                sampled_answer_weight
            ),
            "lambda_map_compact_eff": total_loss.new_tensor(
                map_compact_weight
            ),
            "minimum_action_std": total_loss.new_tensor(
                minimum_action_std
            ),
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

    @staticmethod
    def _hard_pair_path_indices(
        pairs: Sequence[HardPathPair],
    ) -> List[int]:
        return sorted(
            {
                path_index
                for pair in pairs
                for path_index in (
                    pair.correct_index,
                    pair.wrong_index,
                )
            }
        )

    @torch.no_grad()
    def _score_hard_pair_base_paths(
        self,
        group_questions: Sequence[str],
        group_answers: Sequence[str],
        actions: torch.Tensor,
        pairs: Sequence[HardPathPair],
    ) -> torch.Tensor:
        """Score only original paths used by the selected interventions."""
        selected = self._hard_pair_path_indices(pairs)
        if not selected:
            return actions.new_zeros(actions.shape[0])
        selected_tensor = torch.tensor(
            selected,
            device=actions.device,
            dtype=torch.long,
        )
        selected_scores = self._score_fixed_action_paths(
            [group_questions[index] for index in selected],
            [group_answers[index] for index in selected],
            actions.index_select(0, selected_tensor),
        )
        base_scores = selected_scores.new_zeros(actions.shape[0])
        base_scores.index_copy_(0, selected_tensor, selected_scores)
        return base_scores

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
        """Collect group-8 paths with exact-answer terminal reward only."""
        forbidden_shaping = (
            "dense_outcome_weight",
            "step_reward_weight",
            "trajectory_length_weight",
        )
        active_shaping = [
            name
            for name in forbidden_shaping
            if float(self.trace_rl_config.get(name, 0.0)) != 0.0
        ]
        if active_shaping:
            raise RuntimeError(
                "TRACE-VB forbids outcome/semantic/length reward shaping: "
                + ", ".join(active_shaping)
            )
        group_size = int(self.trace_rl_config.get("group_size", 8))
        if group_size != 8:
            raise ValueError("formal TRACE-VB Stage 2 requires group_size=8")
        group_questions = [
            question for question in questions for _ in range(group_size)
        ]
        group_answers = [
            answer for answer in answers for _ in range(group_size)
        ]
        micro_batch = self._rollout_micro_batch_size()
        action_chunks = []
        action_log_prob_chunks = []
        pre_action_state_chunks = []
        action_mask_chunks = []
        greedy_accuracy_chunks = []
        greedy_length_chunks = []

        for start in range(0, len(group_questions), micro_batch):
            end = min(start + micro_batch, len(group_questions))
            chunk_questions = group_questions[start:end]
            chunk_answers = group_answers[start:end]
            sampled_path = self._trajectory_latents(
                chunk_questions,
                deterministic=False,
            )
            greedy_ids = self._generate_answers_from_trajectory(
                sampled_path,
                do_sample=False,
            )
            greedy_accuracy_chunks.append(
                self._answers_to_accuracy(greedy_ids, chunk_answers)
            )
            greedy_length_chunks.append(
                greedy_ids.ne(self.tokenizer.pad_token_id)
                .float()
                .sum(dim=-1)
            )
            action_chunks.append(sampled_path["actions"].detach())
            action_log_prob_chunks.append(
                sampled_path["action_log_probs"].detach()
            )
            pre_action_state_chunks.append(
                sampled_path["pre_action_states"].float().detach()
            )
            action_mask_chunks.append(
                sampled_path["stochastic_action_mask"].bool().detach()
            )
            del sampled_path, greedy_ids

        # PPO ratios are especially sensitive to BF16 quantisation around
        # ratio == 1.  Keep the rollout boundary explicitly FP32 even though
        # trajectory sampling already disables autocast for these tensors.
        actions = torch.cat(action_chunks, dim=0).float()
        old_action_log_probs = torch.cat(
            action_log_prob_chunks,
            dim=0,
        ).float()
        pre_action_states = torch.cat(pre_action_state_chunks, dim=0)
        action_mask = torch.cat(action_mask_chunks, dim=0)
        greedy_accuracy = torch.cat(greedy_accuracy_chunks, dim=0)
        greedy_lengths = torch.cat(greedy_length_chunks, dim=0)
        terminal_rewards = greedy_accuracy.detach()
        role_ids = torch.arange(
            self.n_trace_steps, device=self.device, dtype=torch.long
        ).view(1, -1).expand(pre_action_states.shape[0], -1)
        with torch.autocast(
            device_type=pre_action_states.device.type,
            enabled=False,
        ):
            values = torch.sigmoid(
                self.value_critic(
                    pre_action_states.float(), role_ids
                ).float()
            )
            gae = masked_terminal_reward_gae(
                terminal_rewards.float(),
                gamma=float(
                    self.trace_rl_config.get("gamma", 1.0)
                ),
                gae_lambda=float(
                    self.trace_rl_config.get("gae_lambda", 0.95)
                ),
                values=values.float(),
                action_mask=action_mask,
            )
        active_advantages = gae.advantages[action_mask]
        advantage_mean = active_advantages.mean()
        advantage_std = active_advantages.std(unbiased=False).clamp_min(1e-6)
        advantages = torch.where(
            action_mask,
            (gae.advantages - advantage_mean) / advantage_std,
            torch.zeros_like(gae.advantages),
        )
        grouped_accuracy = greedy_accuracy.view(-1, group_size)
        positive_counts = grouped_accuracy.sum(dim=1)
        self._last_trace_metrics = {
            "trace_vb/greedy_path_accuracy": greedy_accuracy.mean(),
            "trace_policy/positive_count": positive_counts.mean(),
            "trace_policy/mixed_group_fraction": (
                (positive_counts > 0) & (positive_counts < group_size)
            ).float().mean(),
            "trace_vb/raw_advantage_mean": active_advantages.mean(),
            "trace_vb/raw_advantage_std": active_advantages.std(
                unbiased=False
            ),
            "trace_vb/value_mean": values[action_mask].mean(),
            "trace_vb/exact_terminal_only": values.new_ones(()),
        }
        return {
            "group_questions": group_questions,
            "group_answers": group_answers,
            "actions": actions,
            "pre_action_states": pre_action_states,
            "role_ids": role_ids,
            "action_mask": action_mask,
            "old_action_log_probs": old_action_log_probs,
            "terminal_rewards": terminal_rewards,
            "values": values.detach(),
            "advantages": advantages.detach(),
            "raw_advantages": gae.advantages.detach(),
            "returns": gae.returns.detach(),
            "greedy_accuracy": greedy_accuracy,
            "greedy_lengths": greedy_lengths,
        }

    @torch.no_grad()
    def _legacy_semantic_rollout(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        role_targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Deprecated shaped-reward rollout retained for forensics only."""
        group_size = int(self.trace_rl_config.get("group_size", 8))
        if group_size < 2:
            raise ValueError("Stage-2 group_size must be at least two")
        group_questions = [
            question for question in questions for _ in range(group_size)
        ]
        group_answers = [
            answer for answer in answers for _ in range(group_size)
        ]
        grouped_targets = {
            key: value.repeat_interleave(group_size, dim=0)
            for key, value in role_targets.items()
            if isinstance(value, torch.Tensor)
        }
        micro_batch = self._rollout_micro_batch_size()
        action_chunks = []
        innovation_chunks = []
        action_log_prob_chunks = []
        residual_chunks = []
        state_chunks = []
        step_reward_chunks = []
        greedy_accuracy_chunks = []
        greedy_length_chunks = []
        frozen_gold_score_chunks = []

    @torch.no_grad()
    def _legacy_counterfactual_rollout(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """Collect IID policy paths and separate path and token outcomes."""
        group_size = int(self.trace_rl_config.get("group_size", 8))
        if group_size < 2:
            raise ValueError(
                "group_size must allow a mixed correct/wrong rollout group"
            )
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
        stage1_answer_log_prob_chunks = []
        frozen_gold_score_chunks = []

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
            sampled_ids = self._generate_answers_from_trajectory(
                sampled_path,
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
            old_answer_log_prob_chunks.append(
                self._answer_token_log_probs(
                    sampled_path,
                    sampled_ids,
                    sampled_mask,
                ).detach()
            )
            stage1_answer_log_prob_chunks.append(
                self._answer_token_log_probs(
                    sampled_path,
                    sampled_ids,
                    sampled_mask,
                    decoder_role="stage1_path_value",
                ).detach()
            )
            frozen_gold_score_chunks.append(
                self._gold_answer_scores(
                    sampled_path,
                    chunk_answers,
                ).detach()
            )
            del sampled_path, greedy_ids

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
        stage1_answer_log_probs = _right_pad(
            stage1_answer_log_prob_chunks,
            value=0.0,
        )
        frozen_gold_scores = torch.cat(
            frozen_gold_score_chunks,
            dim=0,
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
        dense_outcome_advantages = group_standardize(
            frozen_gold_scores.unsqueeze(-1),
            group_size=group_size,
        ).squeeze(-1)
        frozen_score_stds = frozen_gold_scores.view(
            -1,
            group_size,
        ).std(dim=1, unbiased=False)
        trajectory_rewards = (
            greedy_accuracy
            - trajectory_length_penalty
            + float(
                self.trace_rl_config.get("dense_outcome_weight", 0.25)
            )
            * dense_outcome_advantages
        )
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
            outcome_scores=frozen_gold_scores,
            minimum_score_gap=float(
                self.trace_rl_config.get(
                    "minimum_gold_score_gap",
                    2.0e-3,
                )
            ),
        )
        counterfactual_credits = rollout_paths.new_zeros(
            (len(pairs), self.n_trace_steps)
        )
        exact_pair_indices = [
            index
            for index, pair in enumerate(pairs)
            if pair.source == "exact_outcome"
        ]
        continuous_pair_indices = [
            index
            for index, pair in enumerate(pairs)
            if pair.source == "continuous_outcome"
        ]
        continuous_steps: List[int] = []

        def assign_counterfactual_credits(
            pair_indices: Sequence[int],
            *,
            step_indices: Optional[Sequence[int]] = None,
        ) -> None:
            if not pair_indices:
                return
            selected_pairs = [pairs[index] for index in pair_indices]
            counterfactual_metadata = counterfactual_action_batch(
                actions,
                innovations,
                selected_pairs,
                step_indices=step_indices,
            )
            counterfactual_scores = self._score_counterfactual_paths(
                group_questions,
                group_answers,
                selected_pairs,
                counterfactual_metadata,
            )
            selected_credits = counterfactual_transition_credits(
                frozen_gold_scores,
                counterfactual_scores,
                counterfactual_metadata,
                selected_pairs,
                n_steps=self.n_trace_steps,
            )
            index_tensor = torch.tensor(
                pair_indices,
                device=self.device,
                dtype=torch.long,
            )
            counterfactual_credits.index_copy_(
                0,
                index_tensor,
                selected_credits,
            )

        # Preserve the canonical behavior for genuine correct/wrong pairs:
        # every transition is intervened on in both directions.
        assign_counterfactual_credits(exact_pair_indices)
        if continuous_pair_indices:
            steps_per_pair = min(
                self.n_trace_steps,
                max(
                    1,
                    int(
                        self.trace_rl_config.get(
                            "continuous_counterfactual_steps_per_pair",
                            2,
                        )
                    ),
                ),
            )
            step_offset = int(self.global_step) % self.n_trace_steps
            continuous_steps = sorted(
                {
                    (
                        step_offset
                        + index * self.n_trace_steps // steps_per_pair
                    )
                    % self.n_trace_steps
                    for index in range(steps_per_pair)
                }
            )
            assign_counterfactual_credits(
                continuous_pair_indices,
                step_indices=continuous_steps,
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
        counterfactual_eligible = (positive_counts >= 1) & (
            positive_counts < group_size
        )
        positive_pair_eligible = (positive_counts >= 2) & (
            positive_counts < group_size
        )
        exact_pair_hinges = (
            torch.tensor(
                [pairs[index].hinge for index in exact_pair_indices],
                device=self.device,
            )
            if exact_pair_indices
            else torch.zeros(1, device=self.device)
        )
        continuous_pair_hinges = (
            torch.tensor(
                [pairs[index].hinge for index in continuous_pair_indices],
                device=self.device,
            )
            if continuous_pair_indices
            else torch.zeros(1, device=self.device)
        )
        scored_pair_steps = (
            len(exact_pair_indices) * self.n_trace_steps
            + len(continuous_pair_indices) * len(continuous_steps)
        )
        credit_abs = (
            counterfactual_credits.abs().sum()
            / float(max(1, scored_pair_steps))
        )
        continuous_gaps = (
            torch.tensor(
                [pairs[index].outcome_gap for index in continuous_pair_indices],
                device=self.device,
            )
            if continuous_pair_indices
            else torch.zeros(1, device=self.device)
        )
        self._last_trace_metrics = {
            "trace_policy/greedy_path_accuracy": greedy_accuracy.mean(),
            "trace_policy/sampled_answer_accuracy": sampled_accuracy.mean(),
            "trace_policy/positive_count": positive_counts.mean(),
            "trace_policy/mixed_group_fraction": (
                (positive_counts > 0) & (positive_counts < group_size)
            ).float().mean(),
            "trace_policy/counterfactual_eligible_fraction": (
                counterfactual_eligible.float().mean()
            ),
            "trace_policy/positive_pair_eligible_fraction": (
                positive_pair_eligible.float().mean()
            ),
            "trace_policy/hard_pair_count": torch.tensor(
                float(len(exact_pair_indices)),
                device=self.device,
            ),
            "trace_policy/continuous_pair_count": torch.tensor(
                float(len(continuous_pair_indices)),
                device=self.device,
            ),
            "trace_policy/trajectory_pair_count": torch.tensor(
                float(len(pairs)),
                device=self.device,
            ),
            "trace_policy/continuous_outcome_gap": continuous_gaps.mean(),
            "trace_policy/hard_pair_hinge": exact_pair_hinges.mean(),
            "trace_policy/continuous_pair_hinge": (
                continuous_pair_hinges.mean()
            ),
            "trace_policy/counterfactual_credit_abs": credit_abs,
            "trace_policy/exact_counterfactual_question_coverage": torch.tensor(
                float(len(exact_pair_indices))
                / float(max(1, len(questions))),
                device=self.device,
            ),
            "trace_policy/continuous_counterfactual_question_coverage": (
                torch.tensor(
                    float(len(continuous_pair_indices))
                    / float(max(1, len(questions))),
                    device=self.device,
                )
            ),
            "trace_policy/counterfactual_question_coverage": torch.tensor(
                float(len(pairs))
                / float(max(1, len(questions))),
                device=self.device,
            ),
            "trace_policy/counterfactual_path_slot_coverage": torch.tensor(
                float(2 * scored_pair_steps)
                / float(max(1, len(group_questions) * self.n_trace_steps)),
                device=self.device,
            ),
            "trace_policy/frozen_gold_score": frozen_gold_scores.mean(),
            "trace_policy/frozen_gold_score_group_std": (
                frozen_score_stds.mean()
            ),
            "trace_policy/dense_outcome_abs": (
                dense_outcome_advantages.abs().mean()
            ),
            "trace_policy/trajectory_advantage_active_fraction": (
                trajectory_advantages.abs() > 1e-6
            ).float().mean(),
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
            "stage1_answer_log_probs": stage1_answer_log_probs,
            "answer_rewards": answer_rewards,
            "answer_advantages": answer_advantages,
            "sampled_accuracy": sampled_accuracy,
            "counterfactual_credits": counterfactual_credits,
        }

    def _policy_parameters_on_stored_states(
        self,
        policy: GaussianTrajectoryPolicy,
        pre_action_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one actor on rollout states without replaying the LM."""
        if pre_action_states.ndim != 3:
            raise ValueError(
                "pre_action_states must have shape [rollout, 8, hidden]"
            )
        means = []
        log_stds = []
        with torch.autocast(
            device_type=pre_action_states.device.type,
            enabled=False,
        ):
            for step_index in range(self.n_trace_steps):
                mean, log_std = policy.distribution_parameters(
                    pre_action_states[:, step_index, :].float(),
                    step_index,
                )
                means.append(mean.float())
                log_stds.append(log_std.float())
        return torch.stack(means, dim=1), torch.stack(log_stds, dim=1)

    def _trajectory_policy_update(
        self,
        rollout: Dict[str, torch.Tensor],
        *,
        actor_active: bool,
    ) -> Dict[str, torch.Tensor]:
        """One PPO epoch over cached states; Qwen is never rerun here."""
        states = rollout["pre_action_states"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_action_log_probs"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]
        action_mask = rollout["action_mask"].bool()
        role_ids = rollout["role_ids"]
        total_items = int(actions.shape[0])
        micro_batch = self._optimization_micro_batch_size()
        if total_items <= 0:
            raise RuntimeError("empty TRACE-VB PPO rollout")
        clip_epsilon = float(
            self.trace_rl_config.get("trajectory_clip_epsilon", 0.12)
        )
        kl_weight = float(
            self.trace_rl_config.get("stage1_policy_kl_weight", 0.02)
        )
        entropy_weight = float(
            self.trace_rl_config.get("entropy_coefficient", 0.001)
        )
        value_weight = float(
            self.trace_rl_config.get("value_loss_coefficient", 0.5)
        )
        role_entropy_weights = torch.tensor(
            list(
                self.trace_rl_config.get(
                    "role_entropy_weights", [1.0] * 7 + [0.0]
                )
            ),
            device=self.device,
            dtype=torch.float32,
        )
        if role_entropy_weights.numel() != self.n_trace_steps:
            raise ValueError("role_entropy_weights must contain 8 entries")
        totals = {
            name: states.new_zeros((), dtype=torch.float32)
            for name in (
                "actor_loss",
                "value_loss",
                "stage1_kl",
                "entropy",
                "ratio_deviation",
                "clip_fraction",
                "objective",
                "value_mae",
            )
        }
        for start in range(0, total_items, micro_batch):
            end = min(start + micro_batch, total_items)
            chunk_states = states[start:end]
            chunk_mask = action_mask[start:end]
            if actor_active:
                current_means, current_log_stds = (
                    self._policy_parameters_on_stored_states(
                        self.trajectory_policy, chunk_states
                    )
                )
            else:
                with torch.no_grad():
                    current_means, current_log_stds = (
                        self._policy_parameters_on_stored_states(
                            self.trajectory_policy, chunk_states
                        )
                    )
            with torch.autocast(
                device_type=chunk_states.device.type,
                enabled=False,
            ):
                current_log_probs = gaussian_log_prob(
                    actions[start:end].float(),
                    current_means.float(),
                    current_log_stds.float(),
                )
            with torch.no_grad():
                reference_means, reference_log_stds = (
                    self._policy_parameters_on_stored_states(
                        self.stage1_policy_reference, chunk_states
                    )
                )
            with torch.autocast(
                device_type=chunk_states.device.type,
                enabled=False,
            ):
                actor_loss = masked_ppo_actor_loss(
                    current_log_probs.float(),
                    old_log_probs[start:end].float(),
                    advantages[start:end].float(),
                    chunk_mask,
                    clip_epsilon=clip_epsilon,
                )
                stage1_kl = self._masked_role_kl(
                    current_means.float(),
                    current_log_stds.float(),
                    reference_means.float(),
                    reference_log_stds.float(),
                    chunk_mask,
                )
                per_role_entropy = (
                    current_log_stds.float()
                    + 0.5 * math.log(2.0 * math.pi * math.e)
                ).sum(dim=-1)
            entropy_mask = (
                chunk_mask.float() * role_entropy_weights.view(1, -1)
            )
            entropy = (
                per_role_entropy * entropy_mask
            ).sum() / entropy_mask.sum().clamp_min(1.0)
            with torch.autocast(
                device_type=chunk_states.device.type,
                enabled=False,
            ):
                predicted_values = torch.sigmoid(
                    self.value_critic(
                        chunk_states.float(), role_ids[start:end]
                    ).float()
                )
                value_loss = masked_value_loss(
                    predicted_values,
                    returns[start:end].float(),
                    chunk_mask,
                    loss_type=str(
                        self.trace_rl_config.get("value_loss_type", "huber")
                    ),
                )
            value_mae = (
                (predicted_values - returns[start:end]).abs()
                * chunk_mask.float()
            ).sum() / chunk_mask.float().sum().clamp_min(1.0)
            log_ratio = (
                current_log_probs - old_log_probs[start:end]
            ).clamp(min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            active_count = chunk_mask.float().sum().clamp_min(1.0)
            ratio_deviation = (
                (ratio - 1.0).abs() * chunk_mask.float()
            ).sum() / active_count
            clip_fraction = (
                (
                    (ratio < 1.0 - clip_epsilon)
                    | (ratio > 1.0 + clip_epsilon)
                ).float()
                * chunk_mask.float()
            ).sum() / active_count
            actor_objective = (
                actor_loss + kl_weight * stage1_kl - entropy_weight * entropy
            )
            objective = value_weight * value_loss
            if actor_active:
                objective = objective + actor_objective
            chunk_weight = float(end - start) / float(total_items)
            self.manual_backward(chunk_weight * objective)
            measurements = {
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "stage1_kl": stage1_kl,
                "entropy": entropy,
                "ratio_deviation": ratio_deviation,
                "clip_fraction": clip_fraction,
                "objective": objective,
                "value_mae": value_mae,
            }
            for name, value in measurements.items():
                totals[name] = totals[name] + (
                    chunk_weight * value.detach().float()
                )
        totals["actor_active"] = states.new_tensor(float(actor_active))
        return totals

    def _legacy_trajectory_policy_update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
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
        role_entropy_sum = torch.zeros((), device=self.device)
        prior_weight = float(
            self.trace_rl_config.get("stage1_policy_kl_weight", 0.02)
        )
        clip_epsilon = float(
            self.trace_rl_config.get("trajectory_clip_epsilon", 0.12)
        )
        full_mask = self.trajectory_policy.stochastic_action_mask(
            actions.shape[0],
            device=self.device,
            dtype=torch.bool,
        )
        entropy_weights = torch.tensor(
            self.trace_rl_config.get(
                "role_entropy_weights",
                (1.0, 0.75, 0.60, 0.45, 0.30, 0.20, 0.10, 0.0),
            ),
            device=self.device,
            dtype=torch.float32,
        )
        if tuple(entropy_weights.shape) != (self.n_trace_steps,):
            raise ValueError("role_entropy_weights must contain 8 values")
        entropy_coefficient = float(
            self.trace_rl_config.get("role_entropy_coefficient", 0.001)
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
                mask=full_mask[start:end],
            )
            ratio = torch.exp(
                current["action_log_probs"]
                - old_log_probs[start:end].detach()
            )
            active = full_mask[start:end].to(ratio.dtype)
            active_count = active.sum().clamp_min(1.0)
            ratio_deviation = (
                (ratio - 1.0).abs() * active
            ).sum() / active_count
            clip_fraction = (
                (
                    (ratio < 1.0 - clip_epsilon)
                    | (ratio > 1.0 + clip_epsilon)
                ).to(active.dtype)
                * active
            ).sum() / active_count
            prior_kl = self._masked_role_kl(
                current["action_means"],
                current["action_log_stds"],
                current["reference_action_means"],
                current["reference_action_log_stds"],
                full_mask[start:end],
            )
            per_step_entropy = (
                current["action_log_stds"].float()
                + 0.5 * math.log(2.0 * math.pi * math.e)
            ).mean(dim=-1)
            local_entropy_weights = (
                entropy_weights.unsqueeze(0)
                * active
            )
            role_entropy = (
                per_step_entropy * local_entropy_weights
            ).sum() / local_entropy_weights.sum().clamp_min(1.0)
            chunk_weight = float(end - start) / float(total_items)
            self.manual_backward(
                chunk_weight
                * (
                    policy_loss
                    + prior_weight * prior_kl
                    - entropy_coefficient * role_entropy
                )
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
            role_entropy_sum += (
                role_entropy.detach() * float(end - start)
            )
            del (
                current,
                policy_loss,
                prior_kl,
                ratio,
                active,
                ratio_deviation,
                clip_fraction,
                role_entropy,
            )
        return (
            policy_loss_sum / float(total_items),
            prior_kl_sum / float(total_items),
            ratio_deviation_sum / float(total_items),
            clip_fraction_sum / float(total_items),
            role_entropy_sum / float(total_items),
        )

    def _answer_policy_update(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        questions = rollout["group_questions"]
        actions = rollout["actions"]
        answer_ids = rollout["answer_input_ids"]
        answer_mask = rollout["answer_attention_mask"]
        old_log_probs = rollout["old_answer_log_probs"]
        stage1_log_probs = rollout["stage1_answer_log_probs"]
        advantages = rollout["answer_advantages"]
        micro_batch = self._optimization_micro_batch_size()
        total_items = len(questions)
        policy_loss_sum = torch.zeros((), device=self.device)
        reference_kl_sum = torch.zeros((), device=self.device)
        ratio_deviation_sum = torch.zeros((), device=self.device)
        clip_fraction_sum = torch.zeros((), device=self.device)
        objective_sum = torch.zeros((), device=self.device)
        clip_epsilon = float(
            self.trace_rl_config.get("answer_clip_epsilon", 0.12)
        )
        policy_weight = float(
            self.trace_rl_config.get("answer_policy_weight", 0.10)
        )
        reference_weight = float(
            self.trace_rl_config.get(
                "stage1_answer_kl_weight",
                0.10,
            )
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
            ratio = torch.exp(
                current_log_probs
                - old_log_probs[start:end].detach()
            )
            active = answer_mask[start:end].to(ratio.dtype)
            active_count = active.sum().clamp_min(1.0)
            ratio_deviation = (
                (ratio - 1.0).abs() * active
            ).sum() / active_count
            clip_fraction = (
                (
                    (ratio < 1.0 - clip_epsilon)
                    | (ratio > 1.0 + clip_epsilon)
                ).to(active.dtype)
                * active
            ).sum() / active_count
            reference_kl = sampled_forward_kl(
                current_log_probs,
                stage1_log_probs[start:end].detach(),
                mask=answer_mask[start:end],
            )
            objective = (
                policy_weight * loss
                + reference_weight * reference_kl
            )
            chunk_weight = float(end - start) / float(total_items)
            self.manual_backward(chunk_weight * objective)
            policy_loss_sum += loss.detach() * float(end - start)
            reference_kl_sum += (
                reference_kl.detach() * float(end - start)
            )
            ratio_deviation_sum += (
                ratio_deviation.detach() * float(end - start)
            )
            clip_fraction_sum += (
                clip_fraction.detach() * float(end - start)
            )
            objective_sum += objective.detach() * float(end - start)
            del (
                trajectory,
                current_log_probs,
                loss,
                ratio,
                active,
                ratio_deviation,
                clip_fraction,
                reference_kl,
                objective,
            )
        denominator = float(total_items)
        return (
            policy_loss_sum / denominator,
            reference_kl_sum / denominator,
            ratio_deviation_sum / denominator,
            clip_fraction_sum / denominator,
            objective_sum / denominator,
        )

    def trace_rl_training_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=0,
    ):
        """Stage 2: calibrated terminal-reward GAE and head-only PPO."""
        questions = list(batch["question"])
        answers = list(batch["answer"])
        optimizer = self.optimizers()
        rollout = self.trace_policy_rollout(
            questions=questions,
            answers=answers,
        )
        update_epochs = int(
            self.trace_rl_config.get("policy_update_epochs", 4)
        )
        if update_epochs != 4:
            raise RuntimeError(
                "formal TRACE-VB requires exactly four head-only PPO epochs"
            )
        warmup_batches = int(
            self.trace_rl_config.get("critic_warmup_batches", 256)
        )
        actor_active = int(self.vb_rollout_batches_seen.item()) >= warmup_batches
        semantic_anchor_weight = self._semantic_anchor_weight()
        update_metrics = []
        grad_norms = []
        optimizer_steps = 0
        for update_index in range(update_epochs):
            optimizer.zero_grad(set_to_none=True)
            metrics = self._trajectory_policy_update(
                rollout,
                actor_active=actor_active,
            )
            zero = rollout["actions"].new_zeros(())
            metrics.update(
                {
                    "semantic_anchor": zero,
                    "semantic_anchor_plan": zero,
                    "semantic_anchor_solve_text": zero,
                    "semantic_anchor_refine": zero,
                    "semantic_anchor_weight": zero,
                    "semantic_anchor_applied": zero,
                    "semantic_anchor_truncated_fraction": zero,
                }
            )
            if (
                actor_active
                and update_index == 0
                and semantic_anchor_weight > 0.0
            ):
                anchor = self._stage2_role_semantic_anchor(batch)
                if not bool(torch.isfinite(anchor["total"]).item()):
                    raise RuntimeError("non-finite Stage-2 semantic anchor")
                weighted_anchor = semantic_anchor_weight * anchor["total"]
                self.manual_backward(weighted_anchor)
                metrics["objective"] += weighted_anchor.detach()
                metrics["semantic_anchor"] = anchor["total"].detach()
                metrics["semantic_anchor_plan"] = anchor["plan"]
                metrics["semantic_anchor_solve_text"] = anchor["solve_text"]
                metrics["semantic_anchor_refine"] = anchor["refine"]
                metrics["semantic_anchor_weight"] = zero.new_tensor(
                    semantic_anchor_weight
                )
                metrics["semantic_anchor_applied"] = zero.new_ones(())
                metrics["semantic_anchor_truncated_fraction"] = anchor[
                    "truncated_fraction"
                ]
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
            update_metrics.append(metrics)
            grad_norms.append(grad_norm.detach())
        self.vb_rollout_batches_seen.add_(1)

        reduced = {
            name: torch.stack([item[name] for item in update_metrics]).mean()
            for name in update_metrics[0]
        }
        grad_norm = torch.stack(grad_norms).mean()
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        actor_lr = float(raw_optimizer.param_groups[0]["lr"])
        critic_lr = float(raw_optimizer.param_groups[1]["lr"])
        mask = rollout["action_mask"].float()
        reward_targets = rollout["terminal_rewards"][:, None].expand_as(mask)
        brier = (
            (rollout["values"] - reward_targets).square() * mask
        ).sum() / mask.sum().clamp_min(1.0)
        final_values = rollout["values"][:, 6]
        correct = rollout["terminal_rewards"] > 0.5
        wrong = ~correct
        value_gap = final_values.new_zeros(())
        if bool(correct.any()) and bool(wrong.any()):
            value_gap = final_values[correct].mean() - final_values[wrong].mean()
        logs = {
            "train/total_loss": reduced["objective"].detach(),
            "train/trajectory_policy_loss": reduced[
                "actor_loss"
            ].detach(),
            "train/value_loss": reduced["value_loss"].detach(),
            "train/value_mae": reduced["value_mae"].detach(),
            "train/value_brier": brier.detach(),
            "train/value_correct_wrong_gap": value_gap.detach(),
            "train/stage1_policy_kl": reduced["stage1_kl"].detach(),
            "train/semantic_anchor": update_metrics[0][
                "semantic_anchor"
            ].detach(),
            "train/semantic_anchor_plan": update_metrics[0][
                "semantic_anchor_plan"
            ].detach(),
            "train/semantic_anchor_solve_text": update_metrics[0][
                "semantic_anchor_solve_text"
            ].detach(),
            "train/semantic_anchor_refine": update_metrics[0][
                "semantic_anchor_refine"
            ].detach(),
            "train/semantic_anchor_weight": update_metrics[0][
                "semantic_anchor_weight"
            ].detach(),
            "train/semantic_anchor_applied": update_metrics[0][
                "semantic_anchor_applied"
            ].detach(),
            "train/semantic_anchor_truncated_fraction": update_metrics[0][
                "semantic_anchor_truncated_fraction"
            ].detach(),
            "train/role_entropy": reduced["entropy"].detach(),
            "train/action_ratio_deviation": reduced[
                "ratio_deviation"
            ].detach(),
            "train/action_ratio_deviation_final_update": (
                update_metrics[-1]["ratio_deviation"].detach()
            ),
            "train/action_clip_fraction": reduced[
                "clip_fraction"
            ].detach(),
            "train/action_clip_fraction_final_update": (
                update_metrics[-1]["clip_fraction"].detach()
            ),
            "train/exact_terminal_reward": rollout[
                "terminal_rewards"
            ].mean().detach(),
            "train/raw_gae_advantage": rollout[
                "raw_advantages"
            ][rollout["action_mask"]].mean().detach(),
            "train/output_length": rollout[
                "greedy_lengths"
            ].mean().detach(),
            "train/n_latent_forward": torch.tensor(
                float(self.n_trace_steps),
                device=self.device,
            ),
            "train/grad_norm": grad_norm.detach(),
            "train/actor_lr": torch.tensor(
                actor_lr,
                device=self.device,
            ),
            "train/critic_lr": torch.tensor(
                critic_lr,
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
            "train/answer_decoder_frozen": torch.ones(
                (),
                device=self.device,
            ),
            "train/head_only_ppo": torch.ones((), device=self.device),
            "train/actor_active": torch.tensor(
                float(actor_active), device=self.device
            ),
            "train/critic_warmup_remaining_batches": torch.tensor(
                float(
                    max(
                        0,
                        warmup_batches
                        - int(self.vb_rollout_batches_seen.item()),
                    )
                ),
                device=self.device,
            ),
            "train/rollout_batches_seen": self.vb_rollout_batches_seen.float(),
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
        return reduced["objective"].detach()

    def _legacy_trace_rl_training_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=0,
    ):
        """Deprecated semantic-reward Stage-2 loop; never called by VB."""
        questions = list(batch["question"])
        answers = list(batch["answer"])
        gold_cots = self._decode_single_gold_cots(batch)
        explicit_features, _ = self._collect_single_cot_features(
            questions,
            gold_cots,
        )
        role_targets = self._build_role_semantic_targets(explicit_features)
        tensor_role_targets = {
            key: value
            for key, value in role_targets.items()
            if isinstance(value, torch.Tensor)
        }
        optimizer = self.optimizers()
        rollout = self._legacy_semantic_rollout(
            questions=questions,
            answers=answers,
            role_targets=tensor_role_targets,
        )
        update_epochs = int(
            self.trace_rl_config.get("policy_update_epochs", 1)
        )
        if update_epochs < 1:
            raise RuntimeError("invalid Stage-2 policy update contract")
        trajectory_losses = []
        prior_kls = []
        ratio_deviations = []
        clip_fractions = []
        role_entropies = []
        grad_norms = []
        optimizer_steps = 0

    def _legacy_answer_joint_training_step(
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
            self.trace_rl_config.get("policy_update_epochs", 1)
        )
        if update_epochs < 1:
            raise RuntimeError("invalid Stage-2 policy update contract")
        trajectory_losses = []
        answer_losses = []
        answer_reference_kls = []
        answer_ratio_deviations = []
        answer_clip_fractions = []
        answer_objectives = []
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
            (
                answer_loss,
                answer_reference_kl,
                answer_ratio_deviation,
                answer_clip_fraction,
                answer_objective,
            ) = self._answer_policy_update(rollout)

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
            answer_reference_kls.append(answer_reference_kl)
            answer_ratio_deviations.append(answer_ratio_deviation)
            answer_clip_fractions.append(answer_clip_fraction)
            answer_objectives.append(answer_objective)
            prior_kls.append(prior_kl)
            ratio_deviations.append(ratio_deviation)
            clip_fractions.append(clip_fraction)
            grad_norms.append(grad_norm.detach())

        trajectory_loss = torch.stack(trajectory_losses).mean()
        answer_loss = torch.stack(answer_losses).mean()
        answer_reference_kl = torch.stack(
            answer_reference_kls
        ).mean()
        answer_ratio_deviation = torch.stack(
            answer_ratio_deviations
        ).mean()
        answer_ratio_deviation_final = answer_ratio_deviations[-1]
        answer_clip_fraction = torch.stack(
            answer_clip_fractions
        ).mean()
        answer_clip_fraction_final = answer_clip_fractions[-1]
        answer_objective = torch.stack(answer_objectives).mean()
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
            + answer_objective
            + prior_weight * prior_kl
        )
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        learning_rate = float(raw_optimizer.param_groups[0]["lr"])
        logs = {
            "train/total_loss": total_loss.detach(),
            "train/trajectory_policy_loss": trajectory_loss.detach(),
            "train/answer_policy_loss": answer_loss.detach(),
            "train/answer_objective": answer_objective.detach(),
            "train/stage1_answer_kl": answer_reference_kl.detach(),
            "train/answer_ratio_deviation": (
                answer_ratio_deviation.detach()
            ),
            "train/answer_ratio_deviation_final_update": (
                answer_ratio_deviation_final.detach()
            ),
            "train/answer_clip_fraction": (
                answer_clip_fraction.detach()
            ),
            "train/answer_clip_fraction_final_update": (
                answer_clip_fraction_final.detach()
            ),
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
        gold_cot: dict,
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
            "pre_action_states": [],
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
        value_head = (
            self.value_critic if self.do_trace_rl else self.sufficiency_head
        )
        value_head_type = (
            "outcome_critic" if self.do_trace_rl else "sufficiency"
        )
        rollout_role_ids = torch.arange(
            self.n_trace_steps,
            device=self.device,
            dtype=torch.long,
        ).view(1, -1).expand(group_size, -1)
        with torch.autocast(device_type=self.device.type, enabled=False):
            rollout_value_predictions = torch.sigmoid(
                value_head(
                    tensors["pre_action_states"].float(),
                    rollout_role_ids,
                ).float()
            )
            map_value_predictions = torch.sigmoid(
                value_head(
                    map_trajectory["pre_action_states"][
                        map_local_index : map_local_index + 1
                    ].float(),
                    rollout_role_ids[:1],
                ).float()
            )[0]
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
            explicit_features, _ = self._collect_single_cot_features(
                [question],
                [gold_cot],
            )
            role_targets = self._build_role_semantic_targets(
                explicit_features
            )
            rollout_role_targets = {
                key: value.repeat_interleave(group_size, dim=0)
                for key, value in role_targets.items()
                if isinstance(value, torch.Tensor)
            }
            rollout_role_rewards = self._role_step_rewards(
                {
                    "latent_states": tensors["latent_states"],
                    "implicit_residuals": tensors["implicit_residuals"],
                },
                rollout_role_targets,
            )
            map_role_rewards = self._role_step_rewards(
                {
                    "latent_states": map_trajectory["latent_states"][
                        map_local_index : map_local_index + 1
                    ],
                    "implicit_residuals": map_trajectory[
                        "implicit_residuals"
                    ][map_local_index : map_local_index + 1],
                },
                {
                    key: value
                    for key, value in role_targets.items()
                    if isinstance(value, torch.Tensor)
                },
            )[0]
        student_distances = self._pairwise_path_distances(
            tensors["implicit_residuals"]
        )
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
            "rollout_schema": (
                "iid_role_conditioned_gaussian_with_deterministic_commit"
            ),
            "rollout_seed": int(seed),
            "rollout_innovations": innovations.to(torch.float16).cpu(),
            "rollout_actions": tensors["actions"].to(torch.float16).cpu(),
            "rollout_action_means": tensors["action_means"]
            .to(torch.float16)
            .cpu(),
            "rollout_action_log_stds": tensors["action_log_stds"]
            .to(torch.float16)
            .cpu(),
            "rollout_value_predictions": rollout_value_predictions
            .to(torch.float32)
            .cpu(),
            "map_value_predictions": map_value_predictions
            .to(torch.float32)
            .cpu(),
            "value_head_type": value_head_type,
            "value_action_mask": self.trajectory_policy
            .stochastic_action_mask(dtype=torch.bool)
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
            "role_schema": (
                "PLAN,SOLVE1,SOLVE2,SOLVE3,SOLVE4,SOLVE5,REFINE,COMMIT"
            ),
            "role_teacher_plan": role_targets["plan"][0]
            .to(torch.float16)
            .cpu(),
            "role_teacher_solve": role_targets["solve"][0]
            .to(torch.float16)
            .cpu(),
            "role_teacher_solve_mask": role_targets["solve_mask"][0].cpu(),
            "role_teacher_summary": role_targets["summary"][0]
            .to(torch.float16)
            .cpu(),
            "map_role_rewards": map_role_rewards.to(torch.float16).cpu(),
            "rollout_role_rewards": rollout_role_rewards
            .to(torch.float16)
            .cpu(),
            "rollout_path_distance_matrix": student_distances.cpu(),
            "hard_pairs": [
                {
                    "correct_index": pair.correct_index,
                    "correct_peer_index": pair.correct_peer_index,
                    "wrong_index": pair.wrong_index,
                    "correct_radius": pair.correct_radius,
                    "wrong_distance": pair.wrong_distance,
                    "hinge": pair.hinge,
                    "has_correct_peer": pair.has_correct_peer,
                }
                for pair in pairs
            ],
            "gold_cot_source": gold_cot["source"],
            "gold_cot_steps": list(gold_cot["steps"]),
            "answer_question_attention_access": int(
                self.answer_reads_question
            ),
            "answer_latent_attention_access": 1,
            "answer_latent_attention_role": "COMMIT",
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
        gold_cots = self._decode_single_gold_cots(batch)
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
                        gold_cot=gold_cots[local_index],
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
            prediction = self.sample_logs[index]["pred_answer"][-1]
            output_string = self.sample_logs[index]["output_string"][-1]
            self._validation_question_records.append(
                (
                    int(index),
                    float(self.sample_logs[index]["acc"][-1]),
                    int(self.sample_logs[index]["output_length"][-1]),
                    json.dumps(
                        prediction,
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    ),
                    bool(prediction),
                    bool(str(output_string).strip()),
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
                "val/unique_predictions": summary["unique_predictions"],
                "val/unique_prediction_ratio": summary[
                    "unique_prediction_ratio"
                ],
                "val/top1_mode_fraction": summary["top1_mode_fraction"],
                "val/valid_answer_fraction": summary[
                    "valid_answer_fraction"
                ],
                "val/nonempty_output_fraction": summary[
                    "nonempty_output_fraction"
                ],
            },
            sync_dist=False,
            on_step=False,
            on_epoch=True,
            batch_size=expected_count,
        )
        if getattr(self.trainer, "is_global_zero", True):
            try:
                directory = Path(self.logger.log_dir)
            except Exception:
                directory = Path(".")
            directory.mkdir(parents=True, exist_ok=True)
            report = {
                "schema_version": "trace_vb_v3_validation_behavior_v1",
                "epoch_index": int(self.current_epoch),
                "world_size": int(
                    dist.get_world_size()
                    if dist.is_available() and dist.is_initialized()
                    else 1
                ),
                **summary,
            }
            (directory / f"validation_epoch_{int(self.current_epoch):03d}.json").write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
        self._save_trace_visual_records("val")
        return super().on_validation_epoch_end()

    def on_test_end(self):
        self._save_trace_visual_records("test")
        return super().on_test_end()
