import copy
import collections
import ctypes
import gc
import hashlib
import json
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint
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
    evidence_gated_group_advantages,
    masked_ppo_actor_loss,
)
from ..utils.safe_checkpoint import safe_load_checkpoint
from ..utils.log import JsonLogger


TRACE_VB_ZERO_ACTION_RESET_SCHEMA = "trace_vb_v8_zero_action_reset_v1"
TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES: Tuple[str, ...] = (
    "trajectory_policy.mean_heads.plan.weight",
    "trajectory_policy.mean_heads.plan.bias",
    "trajectory_policy.mean_heads.solve.weight",
    "trajectory_policy.mean_heads.solve.bias",
    # The implementation name remains ``check`` for v7 checkpoint
    # compatibility; this is the paper-facing REFINE role.
    "trajectory_policy.mean_heads.check.weight",
    "trajectory_policy.mean_heads.check.bias",
    "trajectory_policy.mean_heads.commit.weight",
    "trajectory_policy.mean_heads.commit.bias",
    "trajectory_policy.action_projector.0.bias",
)
TRACE_VB_ZERO_ACTION_RESET_ROLES: Tuple[str, ...] = (
    "PLAN",
    "SOLVE",
    "REFINE",
    "COMMIT",
)


def reset_v8_zero_action_policy_tensors(
    trajectory_policy: GaussianTrajectoryPolicy,
) -> dict:
    """Reset exactly the deployment zero-action prior after a v7 load.

    This operation is deliberately narrower than the capability-spine rewind.
    It clears the four role means and the additive projector bias while
    preserving the projector weight, policy features, log-variance heads, and
    every teacher/semantic module.  Callers own the one-time state transition;
    this helper provides exact structural and numerical validation.
    """
    if not isinstance(trajectory_policy, GaussianTrajectoryPolicy):
        raise TypeError(
            "zero-action reset requires a GaussianTrajectoryPolicy"
        )
    expected_roles = ("plan", "solve", "check", "commit")
    observed_roles = tuple(trajectory_policy.mean_heads.keys())
    if observed_roles != expected_roles:
        raise RuntimeError(
            "zero-action mean-head coverage mismatch: expected "
            f"{expected_roles}, found {observed_roles}"
        )
    action_dim = int(trajectory_policy.action_dim)
    hidden_size = int(trajectory_policy.hidden_size)
    if action_dim <= 0 or hidden_size <= 0:
        raise RuntimeError("zero-action policy dimensions must be positive")

    targets: List[Tuple[str, torch.nn.Parameter]] = []
    feature_width = None
    for role in expected_roles:
        head = trajectory_policy.mean_heads[role]
        if not isinstance(head, torch.nn.Linear) or head.bias is None:
            raise TypeError(
                f"zero-action mean head {role} must be a biased Linear"
            )
        if head.weight.ndim != 2 or head.weight.shape[0] != action_dim:
            raise RuntimeError(
                f"zero-action mean head {role} has invalid weight shape "
                f"{tuple(head.weight.shape)}"
            )
        if tuple(head.bias.shape) != (action_dim,):
            raise RuntimeError(
                f"zero-action mean head {role} has invalid bias shape "
                f"{tuple(head.bias.shape)}"
            )
        if feature_width is None:
            feature_width = int(head.weight.shape[1])
        elif int(head.weight.shape[1]) != feature_width:
            raise RuntimeError(
                "zero-action mean heads do not share one feature width"
            )
        for suffix, parameter in (
            ("weight", head.weight),
            ("bias", head.bias),
        ):
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(
                    f"zero-action mean_heads.{role}.{suffix} is not a Parameter"
                )
            if not parameter.is_floating_point():
                raise TypeError(
                    f"zero-action mean_heads.{role}.{suffix} is not floating"
                )
            if not torch.isfinite(parameter.detach()).all().item():
                raise RuntimeError(
                    f"zero-action mean_heads.{role}.{suffix} is non-finite"
                )
            targets.append(
                (
                    f"trajectory_policy.mean_heads.{role}.{suffix}",
                    parameter,
                )
            )

    projector = trajectory_policy.action_projector
    if (
        not isinstance(projector, torch.nn.Sequential)
        or len(projector) < 1
        or not isinstance(projector[0], torch.nn.Linear)
        or projector[0].bias is None
    ):
        raise TypeError(
            "zero-action action_projector[0] must be a biased Linear"
        )
    projector_linear = projector[0]
    if tuple(projector_linear.weight.shape) != (hidden_size, action_dim):
        raise RuntimeError(
            "zero-action action projector has invalid weight shape: "
            f"{tuple(projector_linear.weight.shape)}"
        )
    if tuple(projector_linear.bias.shape) != (hidden_size,):
        raise RuntimeError(
            "zero-action action projector has invalid bias shape: "
            f"{tuple(projector_linear.bias.shape)}"
        )
    for label, parameter in (
        ("action_projector.0.weight", projector_linear.weight),
        ("action_projector.0.bias", projector_linear.bias),
    ):
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(f"zero-action {label} is not a Parameter")
        if not parameter.is_floating_point():
            raise TypeError(f"zero-action {label} is not floating")
        if not torch.isfinite(parameter.detach()).all().item():
            raise RuntimeError(f"zero-action {label} is non-finite")
    targets.append(
        (
            "trajectory_policy.action_projector.0.bias",
            projector_linear.bias,
        )
    )

    names = tuple(name for name, _ in targets)
    if names != TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES:
        raise RuntimeError(
            "zero-action reset target coverage is not the registered nine "
            f"tensors: {names}"
        )
    if len({parameter.data_ptr() for _, parameter in targets}) != len(targets):
        raise RuntimeError("zero-action reset targets contain aliased tensors")
    with torch.no_grad():
        for _, parameter in targets:
            parameter.zero_()
    not_zero = [
        name
        for name, parameter in targets
        if torch.count_nonzero(parameter.detach()).item() != 0
        or not torch.isfinite(parameter.detach()).all().item()
    ]
    if not_zero:
        raise RuntimeError(
            "zero-action reset did not produce exact finite zeros: "
            + ", ".join(not_zero)
        )
    return {
        "schema_version": TRACE_VB_ZERO_ACTION_RESET_SCHEMA,
        "applied": True,
        "operation_count": 1,
        "tensor_count": len(targets),
        "tensor_names": list(names),
        "semantic_roles": list(TRACE_VB_ZERO_ACTION_RESET_ROLES),
        "action_projector_weight_preserved": True,
    }


def resolve_v8_metric_provenance(trace_config) -> dict:
    """Resolve the two immutable validation artifacts from model config."""
    contracts = {}
    for label in (
        "metric_safe_baseline",
        "registered_capability_validation",
    ):
        raw_path = str(trace_config.get(f"{label}_path", "")).strip()
        if not raw_path or not Path(raw_path).is_absolute():
            raise RuntimeError(
                f"TRACE-VB-v8 requires an absolute {label} path"
            )
        sha256 = str(
            trace_config.get(f"{label}_sha256", "")
        ).strip().lower()
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise RuntimeError(
                f"TRACE-VB-v8 requires a valid {label} SHA256"
            )
        correct_count = trace_config.get(f"{label}_correct_count", -1)
        questions = trace_config.get(f"{label}_questions", -1)
        if isinstance(correct_count, bool) or isinstance(questions, bool):
            raise RuntimeError(f"TRACE-VB-v8 {label} counts cannot be boolean")
        correct_count = int(correct_count)
        questions = int(questions)
        if questions <= 0 or not 0 <= correct_count <= questions:
            raise RuntimeError(f"TRACE-VB-v8 {label} counts are invalid")
        contracts[label] = {
            "path": str(Path(raw_path).resolve()),
            "sha256": sha256,
            "correct_count": correct_count,
            "questions": questions,
        }
    return contracts


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


class Stage2RecoveryCheckpoint(Callback):
    """Keep one full-state checkpoint keyed by persistent rollout count."""

    filename_glob = "stage2-recovery-rollout*.ckpt"

    def __init__(self, every_n_rollout_batches: int = 64) -> None:
        super().__init__()
        self.every_n_rollout_batches = int(every_n_rollout_batches)
        if self.every_n_rollout_batches <= 0:
            raise ValueError("every_n_rollout_batches must be positive")
        self._last_saved_rollout = -1

    @property
    def state_key(self) -> str:
        return (
            f"{self.__class__.__qualname__}"
            "{every_n_rollout_batches="
            f"{self.every_n_rollout_batches}"
            "}"
        )

    @staticmethod
    def _checkpoint_directory(trainer) -> Path:
        primary = trainer.checkpoint_callback
        directory = getattr(primary, "dirpath", None)
        if not directory:
            directory = Path(trainer.default_root_dir) / "checkpoints"
        return Path(str(directory))

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not bool(getattr(pl_module, "do_trace_rl", False)):
            return
        counter = getattr(pl_module, "vb_rollout_batches_seen", None)
        if counter is None or not isinstance(counter, torch.Tensor):
            raise RuntimeError(
                "Stage 2 recovery requires persistent "
                "vb_rollout_batches_seen"
            )
        if counter.numel() != 1:
            raise RuntimeError(
                "vb_rollout_batches_seen must be a scalar tensor"
            )
        rollout = int(counter.detach().item())
        if (
            rollout <= 0
            or rollout == self._last_saved_rollout
            or rollout % self.every_n_rollout_batches != 0
        ):
            return

        step = int(trainer.global_step)
        directory = self._checkpoint_directory(trainer)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / (
            f"stage2-recovery-rollout{rollout:06d}"
            f"-globalstep{step:06d}.ckpt"
        )
        trainer.save_checkpoint(str(checkpoint_path), weights_only=False)
        if trainer.is_global_zero:
            # The glob is scoped to the active logger's checkpoint directory.
            # A resumed process therefore also removes an older recovery file
            # from that same logger after the new full-state save succeeds.
            for stale_path in directory.glob(self.filename_glob):
                if stale_path != checkpoint_path and stale_path.is_file():
                    stale_path.unlink()
        self._last_saved_rollout = rollout


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

    The caller supplies the already-resolved latent mask.  The formal VB-v5
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
                    "TRACE-VB-v8 validation records require index, accuracy, "
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
    correct_count = int(
        sum(int(value[0] > 0.5) for value in records.values())
    )
    return {
        "accuracy": float(correct_count / max(1, len(records))),
        "correct_count": correct_count,
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


def select_capability_kl_mask(
    loss_mask: torch.Tensor,
    protected_loss_mask: torch.Tensor,
    *,
    scope: str,
) -> torch.Tensor:
    """Select the auditable token support for capability distillation."""
    if tuple(loss_mask.shape) != tuple(protected_loss_mask.shape):
        raise ValueError("capability KL masks must have identical shapes")
    normalized_scope = str(scope).strip().lower()
    if normalized_scope == "full_target":
        return loss_mask
    if normalized_scope in {"answer_suffix", "protected_suffix"}:
        return protected_loss_mask
    raise ValueError(
        "stage1_capability_kl_scope must be full_target or answer_suffix"
    )


def resolve_capability_lora_pairs(
    named_parameters: Dict[str, torch.nn.Parameter],
    *,
    path_adapter_name: str,
    capability_adapter_name: str,
    expected_tensors: int,
) -> List[Tuple[str, torch.nn.Parameter, str, torch.nn.Parameter]]:
    """Resolve an exact capability-to-deployment LoRA tensor bijection."""
    path_marker = f".{path_adapter_name}."
    capability_marker = f".{capability_adapter_name}."
    capability_names = sorted(
        name
        for name in named_parameters
        if capability_marker in name
        and (".lora_A." in name or ".lora_B." in name)
        and name.endswith(".weight")
    )
    if len(capability_names) != int(expected_tensors):
        raise RuntimeError(
            "capability rewind coverage mismatch: expected "
            f"{int(expected_tensors)}, found {len(capability_names)}"
        )
    pairs = []
    seen_targets = set()
    for capability_name in capability_names:
        path_name = capability_name.replace(
            capability_marker,
            path_marker,
            1,
        )
        if path_name not in named_parameters:
            raise RuntimeError(
                f"deployment adapter is missing rewind tensor {path_name}"
            )
        capability_parameter = named_parameters[capability_name]
        path_parameter = named_parameters[path_name]
        if capability_parameter.shape != path_parameter.shape:
            raise RuntimeError(
                "capability rewind shape mismatch: "
                f"{capability_name} -> {path_name}"
            )
        if path_name in seen_targets:
            raise RuntimeError(
                f"duplicate deployment rewind target {path_name}"
            )
        seen_targets.add(path_name)
        pairs.append(
            (
                capability_name,
                capability_parameter,
                path_name,
                path_parameter,
            )
        )
    return pairs


def rewind_capability_spine_tensors(
    named_parameters: Dict[str, torch.nn.Parameter],
    capability_queries: torch.Tensor,
    dynamics_step_embedding: torch.Tensor,
    *,
    path_adapter_name: str,
    capability_adapter_name: str,
    expected_tensors: int,
) -> int:
    """Copy the immutable capability spine into its deployment mirrors."""
    pairs = resolve_capability_lora_pairs(
        named_parameters,
        path_adapter_name=path_adapter_name,
        capability_adapter_name=capability_adapter_name,
        expected_tensors=expected_tensors,
    )
    if capability_queries.shape != dynamics_step_embedding.shape:
        raise RuntimeError(
            "capability queries and dynamics step prior have different shapes"
        )
    with torch.no_grad():
        for _, capability_parameter, _, path_parameter in pairs:
            path_parameter.copy_(capability_parameter.detach())
        dynamics_step_embedding.copy_(capability_queries.detach())
    return len(pairs)


def assert_capability_spine_parity(
    named_parameters: Dict[str, torch.nn.Parameter],
    capability_queries: torch.Tensor,
    dynamics_step_embedding: torch.Tensor,
    *,
    path_adapter_name: str,
    capability_adapter_name: str,
    expected_tensors: int,
) -> int:
    """Fail closed unless every deployment spine tensor is exactly equal."""
    pairs = resolve_capability_lora_pairs(
        named_parameters,
        path_adapter_name=path_adapter_name,
        capability_adapter_name=capability_adapter_name,
        expected_tensors=expected_tensors,
    )
    mismatched = [
        f"{capability_name} -> {path_name}"
        for capability_name, capability_parameter, path_name, path_parameter
        in pairs
        if not torch.equal(
            capability_parameter.detach(), path_parameter.detach()
        )
    ]
    if mismatched:
        raise RuntimeError(
            "deployment LoRA is not at exact capability parity: "
            + ", ".join(mismatched[:8])
        )
    if capability_queries.shape != dynamics_step_embedding.shape:
        raise RuntimeError(
            "capability queries and dynamics step prior have different shapes"
        )
    if not torch.equal(
        capability_queries.detach(), dynamics_step_embedding.detach()
    ):
        raise RuntimeError(
            "dynamics step prior is not at exact capability-query parity"
        )
    return len(pairs)


def validate_v8_checkpoint_provenance(
    checkpoint: dict,
    *,
    expected_lora_tensors: int,
    expected_cot_sha256: str,
    expected_registered_capability_checkpoint_sha256: str,
    expected_registered_capability_payload_tensors: int,
    expected_registered_capability_payload_sha256: str,
    expected_metric_safe_baseline_path: str,
    expected_metric_safe_baseline_sha256: str,
    expected_metric_safe_baseline_correct_count: int,
    expected_metric_safe_baseline_questions: int,
    expected_registered_capability_validation_path: str,
    expected_registered_capability_validation_sha256: str,
    expected_registered_capability_validation_correct_count: int,
    expected_registered_capability_validation_questions: int,
) -> dict:
    """Parse fail-closed v8 rewind/provenance metadata for any load path."""
    if str(checkpoint.get("trace_vb_schema_version", "")) != "trace_vb_v8":
        raise RuntimeError("checkpoint is not TRACE-VB-v8")
    rewound = checkpoint.get("trace_vb_capability_spine_rewound") is True
    raw_rewind_count = checkpoint.get(
        "trace_vb_capability_spine_rewind_count"
    )
    rewind_count = (
        raw_rewind_count if type(raw_rewind_count) is int else -1
    )
    expected_count = int(expected_lora_tensors)
    if not rewound or rewind_count != expected_count:
        raise RuntimeError(
            "v8 checkpoint lacks complete capability rewind metadata: "
            f"rewound={rewound}, count={rewind_count}, "
            f"expected={expected_count}"
        )
    expected_sha = str(expected_cot_sha256).strip().lower()
    source_sha = str(
        checkpoint.get("trace_vb_cot_encoder_checkpoint_sha256", "")
    ).strip().lower()
    if (
        len(expected_sha) != 64
        or source_sha != expected_sha
    ):
        raise RuntimeError(
            "v8 checkpoint has wrong CoT encoder provenance: expected "
            f"{expected_sha or '<missing>'}, "
            f"found {source_sha or '<missing>'}"
        )
    expected_registered_sha = str(
        expected_registered_capability_checkpoint_sha256
    ).strip().lower()
    registered_sha = str(
        checkpoint.get(
            "trace_vb_registered_capability_checkpoint_sha256", ""
        )
    ).strip().lower()
    if (
        len(expected_registered_sha) != 64
        or registered_sha != expected_registered_sha
    ):
        raise RuntimeError(
            "v8 checkpoint has wrong registered capability checkpoint "
            f"provenance: expected {expected_registered_sha or '<missing>'}, "
            f"found {registered_sha or '<missing>'}"
        )
    expected_payload_sha = str(
        expected_registered_capability_payload_sha256
    ).strip().lower()
    payload_sha = str(
        checkpoint.get(
            "trace_vb_registered_capability_payload_sha256", ""
        )
    ).strip().lower()
    if (
        len(expected_payload_sha) != 64
        or payload_sha != expected_payload_sha
    ):
        raise RuntimeError(
            "v8 checkpoint has wrong canonical capability payload "
            f"provenance: expected {expected_payload_sha or '<missing>'}, "
            f"found {payload_sha or '<missing>'}"
        )
    payload_tensors = int(
        checkpoint.get(
            "trace_vb_registered_capability_payload_tensors", -1
        )
    )
    if payload_tensors != int(
        expected_registered_capability_payload_tensors
    ):
        raise RuntimeError(
            "v8 checkpoint canonical capability payload coverage mismatch: "
            f"expected {int(expected_registered_capability_payload_tensors)}, "
            f"found {payload_tensors}"
        )
    v7_source_sha = str(
        checkpoint.get("trace_vb_v7_source_checkpoint_sha256", "")
    ).strip().lower()
    if len(v7_source_sha) != 64 or any(
        character not in "0123456789abcdef"
        for character in v7_source_sha
    ):
        raise RuntimeError(
            "v8 checkpoint is missing a valid v7 source checkpoint SHA256"
        )
    reset_schema = str(
        checkpoint.get("trace_vb_zero_action_reset_schema_version", "")
    )
    reset_applied = checkpoint.get("trace_vb_zero_action_reset_applied")
    reset_operation_count = checkpoint.get(
        "trace_vb_zero_action_reset_operation_count"
    )
    reset_tensor_count = checkpoint.get(
        "trace_vb_zero_action_reset_tensor_count"
    )
    reset_tensor_names = checkpoint.get(
        "trace_vb_zero_action_reset_target_names"
    )
    reset_source_schema = str(
        checkpoint.get("trace_vb_zero_action_reset_source_schema", "")
    )
    if reset_schema != TRACE_VB_ZERO_ACTION_RESET_SCHEMA:
        raise RuntimeError("v8 checkpoint has the wrong zero-action reset schema")
    if reset_applied is not True:
        raise RuntimeError("v8 checkpoint does not attest the zero-action reset")
    if type(reset_operation_count) is not int or reset_operation_count != 1:
        raise RuntimeError(
            "v8 checkpoint zero-action reset operation count is not exactly one"
        )
    if (
        type(reset_tensor_count) is not int
        or reset_tensor_count != len(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES)
    ):
        raise RuntimeError(
            "v8 checkpoint zero-action reset tensor count is not exactly nine"
        )
    if reset_tensor_names != list(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES):
        raise RuntimeError(
            "v8 checkpoint zero-action reset tensor coverage is not exact"
        )
    if reset_source_schema != "trace_vb_v7":
        raise RuntimeError(
            "v8 checkpoint zero-action reset did not originate from v7"
        )
    reset_source_sha = str(
        checkpoint.get(
            "trace_vb_zero_action_reset_source_checkpoint_sha256", ""
        )
    ).strip().lower()
    if reset_source_sha != v7_source_sha:
        raise RuntimeError(
            "v8 checkpoint zero-action reset source SHA256 differs from v7"
        )

    def validate_metric_artifact(
        prefix: str,
        *,
        expected_path: str,
        expected_sha256: str,
        expected_correct_count: int,
        expected_questions: int,
    ) -> dict:
        observed_path = str(checkpoint.get(f"{prefix}_path", ""))
        canonical_expected_path = str(Path(str(expected_path)).resolve())
        if observed_path != canonical_expected_path:
            raise RuntimeError(
                f"v8 checkpoint has wrong {prefix} path: "
                f"expected {canonical_expected_path}, found "
                f"{observed_path or '<missing>'}"
            )
        observed_sha = str(
            checkpoint.get(f"{prefix}_sha256", "")
        ).strip().lower()
        normalized_expected_sha = str(expected_sha256).strip().lower()
        if (
            len(normalized_expected_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in normalized_expected_sha
            )
            or observed_sha != normalized_expected_sha
        ):
            raise RuntimeError(f"v8 checkpoint has wrong {prefix} SHA256")
        observed_correct = checkpoint.get(f"{prefix}_correct_count")
        observed_questions = checkpoint.get(f"{prefix}_questions")
        if (
            type(observed_correct) is not int
            or observed_correct != int(expected_correct_count)
        ):
            raise RuntimeError(f"v8 checkpoint has wrong {prefix} correct count")
        if (
            type(observed_questions) is not int
            or observed_questions != int(expected_questions)
        ):
            raise RuntimeError(f"v8 checkpoint has wrong {prefix} question count")
        return {
            "path": observed_path,
            "sha256": observed_sha,
            "correct_count": observed_correct,
            "questions": observed_questions,
        }

    metric_safe_baseline = validate_metric_artifact(
        "trace_vb_metric_safe_baseline",
        expected_path=expected_metric_safe_baseline_path,
        expected_sha256=expected_metric_safe_baseline_sha256,
        expected_correct_count=expected_metric_safe_baseline_correct_count,
        expected_questions=expected_metric_safe_baseline_questions,
    )
    registered_capability_validation = validate_metric_artifact(
        "trace_vb_registered_capability_validation",
        expected_path=expected_registered_capability_validation_path,
        expected_sha256=expected_registered_capability_validation_sha256,
        expected_correct_count=(
            expected_registered_capability_validation_correct_count
        ),
        expected_questions=expected_registered_capability_validation_questions,
    )
    return {
        "rewind_count": rewind_count,
        "rewind_source": str(
            checkpoint.get(
                "trace_vb_capability_spine_rewind_source",
                "v8_checkpoint",
            )
        ),
        "cot_encoder_sha256": source_sha,
        "registered_capability_checkpoint_sha256": registered_sha,
        "registered_capability_payload_tensors": payload_tensors,
        "registered_capability_payload_sha256": payload_sha,
        "v7_source_checkpoint_sha256": v7_source_sha,
        "zero_action_reset_schema_version": reset_schema,
        "zero_action_reset_applied": True,
        "zero_action_reset_operation_count": reset_operation_count,
        "zero_action_reset_tensor_count": reset_tensor_count,
        "zero_action_reset_target_names": list(reset_tensor_names),
        "zero_action_reset_source_schema": reset_source_schema,
        "zero_action_reset_source_checkpoint_sha256": reset_source_sha,
        "metric_safe_baseline": metric_safe_baseline,
        "registered_capability_validation": (
            registered_capability_validation
        ),
    }


def write_v8_checkpoint_provenance(
    checkpoint: dict,
    *,
    rewound: bool,
    rewind_count: int,
    rewind_source: str,
    cot_encoder_sha256: str,
    registered_capability_checkpoint_sha256: str,
    registered_capability_payload_tensors: int,
    registered_capability_payload_sha256: str,
    v7_source_checkpoint_sha256: str,
    zero_action_reset_schema_version: str,
    zero_action_reset_applied: bool,
    zero_action_reset_operation_count: int,
    zero_action_reset_tensor_count: int,
    zero_action_reset_target_names: Sequence[str],
    zero_action_reset_source_schema: str,
    metric_safe_baseline_path: str,
    metric_safe_baseline_sha256: str,
    metric_safe_baseline_correct_count: int,
    metric_safe_baseline_questions: int,
    registered_capability_validation_path: str,
    registered_capability_validation_sha256: str,
    registered_capability_validation_correct_count: int,
    registered_capability_validation_questions: int,
) -> None:
    """Write the metadata required by resume and weights-only loading."""
    checkpoint["trace_vb_schema_version"] = "trace_vb_v8"
    checkpoint["trace_vb_capability_spine_rewound"] = bool(rewound)
    checkpoint["trace_vb_capability_spine_rewind_count"] = int(
        rewind_count
    )
    checkpoint["trace_vb_capability_spine_rewind_source"] = str(
        rewind_source
    )
    checkpoint["trace_vb_cot_encoder_checkpoint_sha256"] = str(
        cot_encoder_sha256
    ).strip().lower()
    checkpoint[
        "trace_vb_registered_capability_checkpoint_sha256"
    ] = str(registered_capability_checkpoint_sha256).strip().lower()
    checkpoint[
        "trace_vb_registered_capability_payload_sha256"
    ] = str(registered_capability_payload_sha256).strip().lower()
    checkpoint[
        "trace_vb_registered_capability_payload_tensors"
    ] = int(registered_capability_payload_tensors)
    checkpoint["trace_vb_v7_source_checkpoint_sha256"] = str(
        v7_source_checkpoint_sha256
    ).strip().lower()
    if (
        str(zero_action_reset_schema_version)
        != TRACE_VB_ZERO_ACTION_RESET_SCHEMA
    ):
        raise RuntimeError("cannot save an invalid zero-action reset schema")
    if zero_action_reset_applied is not True:
        raise RuntimeError("cannot save before the zero-action reset is applied")
    if (
        type(zero_action_reset_operation_count) is not int
        or zero_action_reset_operation_count != 1
    ):
        raise RuntimeError("zero-action reset operation count must equal one")
    if (
        type(zero_action_reset_tensor_count) is not int
        or zero_action_reset_tensor_count
        != len(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES)
    ):
        raise RuntimeError("zero-action reset tensor count must equal nine")
    if list(zero_action_reset_target_names) != list(
        TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES
    ):
        raise RuntimeError("zero-action reset tensor names are not exact")
    checkpoint["trace_vb_zero_action_reset_schema_version"] = str(
        zero_action_reset_schema_version
    )
    checkpoint["trace_vb_zero_action_reset_applied"] = True
    checkpoint["trace_vb_zero_action_reset_operation_count"] = 1
    checkpoint["trace_vb_zero_action_reset_tensor_count"] = len(
        TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES
    )
    checkpoint["trace_vb_zero_action_reset_target_names"] = list(
        TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES
    )
    if str(zero_action_reset_source_schema) != "trace_vb_v7":
        raise RuntimeError("zero-action reset source schema must be trace_vb_v7")
    checkpoint["trace_vb_zero_action_reset_source_schema"] = "trace_vb_v7"
    checkpoint[
        "trace_vb_zero_action_reset_source_checkpoint_sha256"
    ] = str(v7_source_checkpoint_sha256).strip().lower()

    metric_fields = (
        (
            "trace_vb_metric_safe_baseline",
            metric_safe_baseline_path,
            metric_safe_baseline_sha256,
            metric_safe_baseline_correct_count,
            metric_safe_baseline_questions,
        ),
        (
            "trace_vb_registered_capability_validation",
            registered_capability_validation_path,
            registered_capability_validation_sha256,
            registered_capability_validation_correct_count,
            registered_capability_validation_questions,
        ),
    )
    for prefix, path, sha256, correct_count, questions in metric_fields:
        canonical_path = str(Path(str(path)).resolve())
        normalized_sha = str(sha256).strip().lower()
        if not canonical_path or canonical_path == ".":
            raise RuntimeError(f"cannot save an invalid {prefix} path")
        if len(normalized_sha) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_sha
        ):
            raise RuntimeError(f"cannot save an invalid {prefix} SHA256")
        if (
            type(correct_count) is not int
            or type(questions) is not int
            or not 0 <= correct_count <= questions
            or questions <= 0
        ):
            raise RuntimeError(f"cannot save invalid {prefix} metric counts")
        checkpoint[f"{prefix}_path"] = canonical_path
        checkpoint[f"{prefix}_sha256"] = normalized_sha
        checkpoint[f"{prefix}_correct_count"] = correct_count
        checkpoint[f"{prefix}_questions"] = questions


def collect_stage1_role_only_parameters(
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
    *,
    path_adapter_name: str,
) -> Tuple[List[str], List[torch.nn.Parameter]]:
    """Collect Stage-1 parameters while rejecting capability-spine escape."""
    path_marker = f".{path_adapter_name}."
    names = []
    parameters = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if path_marker in name:
            raise RuntimeError(
                "Stage 1 deployment LoRA escaped the frozen capability "
                f"spine: {name}"
            )
        if name.startswith("llm."):
            raise RuntimeError(
                "Stage 1 language-model parameter escaped the frozen "
                f"capability spine: {name}"
            )
        if name.startswith("capability_"):
            raise RuntimeError(
                "Stage 1 capability teacher parameter escaped freezing: "
                f"{name}"
            )
        if name == "trajectory_policy.dynamics_step_embedding.weight":
            raise RuntimeError(
                "Stage 1 capability query prior escaped freezing"
            )
        names.append(name)
        parameters.append(parameter)
    if not parameters:
        raise RuntimeError("Stage 1 requires trainable role parameters")
    return names, parameters


def validate_required_state_coverage(
    expected_state: Dict[str, torch.Tensor],
    source_state: Dict[str, torch.Tensor],
    *,
    required_prefixes: Sequence[str],
) -> List[str]:
    """Require complete shape-compatible warm-start state for each module."""
    required_names = sorted(
        name
        for name in expected_state
        if name.startswith(tuple(required_prefixes))
    )
    if not required_names:
        raise RuntimeError("warm-start coverage prefixes matched no tensors")
    missing = [name for name in required_names if name not in source_state]
    if missing:
        raise RuntimeError(
            "v8 warm-start is missing retained role tensors: "
            + ", ".join(missing[:12])
        )
    wrong_shapes = [
        name
        for name in required_names
        if expected_state[name].shape != source_state[name].shape
    ]
    if wrong_shapes:
        raise RuntimeError(
            "v8 warm-start role tensor shape mismatch: "
            + ", ".join(wrong_shapes[:12])
        )
    return required_names


_REGISTERED_CAPABILITY_LEGACY_MAPPING = {
    "state_norm.weight": "capability_state_norm.weight",
    "state_norm.bias": "capability_state_norm.bias",
    "latent_bridge.0.weight": "capability_latent_bridge.0.weight",
    "latent_bridge.0.bias": "capability_latent_bridge.0.bias",
    "latent_bridge.2.weight": "capability_latent_bridge.2.weight",
    "latent_bridge.2.bias": "capability_latent_bridge.2.bias",
    "step_compressor.latent_queries": "capability_latent_queries",
    "anchor_gate_predictor.0.weight": (
        "capability_anchor_gate_predictor.0.weight"
    ),
    "anchor_gate_predictor.0.bias": (
        "capability_anchor_gate_predictor.0.bias"
    ),
    "anchor_gate_predictor.2.weight": (
        "capability_anchor_gate_predictor.2.weight"
    ),
    "anchor_gate_predictor.2.bias": (
        "capability_anchor_gate_predictor.2.bias"
    ),
}


def canonical_registered_capability_payload(
    registered_state: Dict[str, torch.Tensor],
    *,
    path_adapter_name: str = "default",
    capability_adapter_name: str = "trace_capability",
    expected_lora_tensors: int = 504,
    n_trace_steps: int = 8,
) -> collections.OrderedDict:
    """Map the registered legacy checkpoint to v8 canonical payload names."""
    path_marker = f".{path_adapter_name}."
    capability_marker = f".{capability_adapter_name}."
    lora_names = sorted(
        name
        for name in registered_state
        if name.startswith("llm.")
        and path_marker in name
        and (".lora_A." in name or ".lora_B." in name)
        and name.endswith(".weight")
    )
    if len(lora_names) != int(expected_lora_tensors):
        raise RuntimeError(
            "registered capability LoRA coverage mismatch: expected "
            f"{int(expected_lora_tensors)}, found {len(lora_names)}"
        )
    payload = collections.OrderedDict()
    for source_name in lora_names:
        target_name = source_name.replace(
            path_marker, capability_marker, 1
        )
        payload[target_name] = registered_state[source_name].detach()
    for source_name, target_name in (
        _REGISTERED_CAPABILITY_LEGACY_MAPPING.items()
    ):
        if source_name not in registered_state:
            raise RuntimeError(
                "registered capability is missing legacy tensor "
                f"{source_name}"
            )
        payload[target_name] = registered_state[source_name].detach()
    trace_view_name = "trace_view_embeddings.weight"
    trace_step_view_name = "trace_step_view_embeddings.weight"
    if trace_view_name not in registered_state:
        raise RuntimeError("registered capability is missing trace views")
    if trace_step_view_name not in registered_state:
        raise RuntimeError(
            "registered capability is missing trace step views"
        )
    trace_views = registered_state[trace_view_name]
    trace_step_views = registered_state[trace_step_view_name]
    if trace_views.ndim != 2 or trace_views.shape[0] < 1:
        raise RuntimeError("registered trace-view table has wrong shape")
    if (
        trace_step_views.ndim != 2
        or trace_step_views.shape[0] < int(n_trace_steps)
        or trace_step_views.shape[1] != trace_views.shape[1]
    ):
        raise RuntimeError(
            "registered trace-step-view table has wrong shape"
        )
    # The registered 540/747 behavior used trace view id 0 and the first
    # eight flattened step-view rows. This exactly matches v7's migration.
    payload["capability_trace_view"] = trace_views[0].detach()
    payload["capability_trace_step_views"] = trace_step_views[
        : int(n_trace_steps)
    ].detach()
    return collections.OrderedDict(sorted(payload.items()))


def canonical_tensor_payload_sha256(
    payload: Dict[str, torch.Tensor],
) -> str:
    """Hash names, dtypes, shapes, and exact tensor bytes deterministically."""
    digest = hashlib.sha256()
    for name in sorted(payload):
        tensor = payload[name].detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": str(name),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def verify_v7_registered_capability_payload(
    v7_state: Dict[str, torch.Tensor],
    registered_state: Dict[str, torch.Tensor],
    *,
    path_adapter_name: str = "default",
    capability_adapter_name: str = "trace_capability",
    expected_lora_tensors: int = 504,
    n_trace_steps: int = 8,
) -> dict:
    """Prove every v7 capability tensor equals the registered payload."""
    payload = canonical_registered_capability_payload(
        registered_state,
        path_adapter_name=path_adapter_name,
        capability_adapter_name=capability_adapter_name,
        expected_lora_tensors=expected_lora_tensors,
        n_trace_steps=n_trace_steps,
    )
    missing = [name for name in payload if name not in v7_state]
    if missing:
        raise RuntimeError(
            "v7 source is missing registered capability tensors: "
            + ", ".join(missing[:12])
        )
    mismatched = []
    cast_count = 0
    deployed_payload = collections.OrderedDict()
    for name, registered_value in payload.items():
        v7_value = v7_state[name].detach().cpu().contiguous()
        registered_value = registered_value.detach().cpu().contiguous()
        if registered_value.dtype != v7_value.dtype:
            cast_count += 1
        # Canonical deployment dtype is the explicit v7/v8 target tensor
        # dtype. This mirrors load_state_dict and prevents a harmless storage
        # dtype difference in the legacy artifact from becoming a false
        # provenance failure; equality is still exact after the audited cast.
        canonical_value = registered_value.to(dtype=v7_value.dtype)
        deployed_payload[name] = canonical_value
        if (
            v7_value.shape != canonical_value.shape
            or not torch.equal(v7_value, canonical_value)
        ):
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(
            "v7 capability payload differs from registered checkpoint: "
            + ", ".join(mismatched[:12])
        )
    return {
        "tensor_count": len(payload),
        "canonical_cast_count": cast_count,
        "canonical_payload_sha256": canonical_tensor_payload_sha256(
            deployed_payload
        ),
    }


def ensure_validation_json_logger(module) -> bool:
    """Create validate-only JSON logging once without replacing fit logging."""
    if getattr(module, "json_logger", None) is not None:
        return False
    args = getattr(getattr(module, "all_config", None), "args", None)
    no_log = bool(getattr(args, "no_log", False))
    module.json_logger = JsonLogger(
        module,
        log_file_name="validation",
        tmp_log=no_log,
    )
    return True


class LitTRACEVB(LitREADCoTStableEfficient):
    """TRACE-VB-v8 capability-preserving role program.

    The eight recurrent states have a fixed computation contract:
    PLAN, five shared-dynamics SOLVE states, REFINE, and deterministic COMMIT.
    Stage 1 distils ordered textual-CoT semantics and answer sufficiency into
    the latent program. Stage 2 freezes the language model and applies
    role-local latent policy optimization, combining exact-answer evidence
    with discounted step supervision from the sample's existing gold CoT.
    """

    path_adapter_name = "default"
    cot_encoder_adapter_name = "trace_cot_encoder"
    capability_adapter_name = "trace_capability"
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
        self.use_capability_anchor = bool(
            self.trace_config.get("use_capability_anchor", True)
        )
        if not self.use_capability_anchor:
            raise ValueError(
                "TRACE-VB-v8 requires the registered capability anchor"
            )
        spine_switches = {
            "stage1_freeze_capability_spine": bool(
                self.trace_config.get(
                    "stage1_freeze_capability_spine", True
                )
            ),
            "stage1_freeze_path_lora": bool(
                self.trace_config.get("stage1_freeze_path_lora", False)
            ),
            "stage1_freeze_capability_query_prior": bool(
                self.trace_config.get(
                    "stage1_freeze_capability_query_prior", False
                )
            ),
            "stage1_require_capability_spine_parity": bool(
                self.trace_config.get(
                    "stage1_require_capability_spine_parity", False
                )
            ),
        }
        self.stage1_freeze_capability_spine = all(
            spine_switches.values()
        )
        if not self.stage1_freeze_capability_spine:
            disabled = sorted(
                name for name, enabled in spine_switches.items() if not enabled
            )
            raise ValueError(
                "TRACE-VB-v8 requires a frozen deployment LoRA and "
                "capability query prior with exact parity in Stage 1; "
                "disabled: " + ", ".join(disabled)
            )
        if str(
            self.trace_config.get(
                "stage1_capability_kl_scope", "full_target"
            )
        ).strip().lower() != "full_target":
            raise ValueError(
                "TRACE-VB-v8 requires full-target capability KL"
            )
        if bool(
            self.trace_config.get(
                "stage1_posterior_activation_offload",
                False,
            )
        ):
            raise ValueError(
                "TRACE-VB-v8 forbids saved-tensor CPU activation offload: "
                "the exact offload path has a measured per-step host-RSS "
                "leak. Use GPU-resident activations plus checkpointed "
                "solve-text projection instead."
            )
        # The student recurrence uses a frozen copy of the proven block bridge
        # as its zero-action reference.  The inherited bridge remains dormant;
        # explicit capability modules below are fail-closed checkpoint targets.
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
        if self.capability_adapter_name not in self.llm.peft_config:
            capability_config = copy.deepcopy(
                self.llm.peft_config[self.path_adapter_name]
            )
            self.llm.add_adapter(
                self.capability_adapter_name,
                capability_config,
            )
        self._match_adapter_storage_to_path(
            self.cot_encoder_adapter_name
        )
        self._match_adapter_storage_to_path(
            self.capability_adapter_name
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
                "TRACE-VB-v8 requires answer_context_mode="
                "question_and_commit: the answer must retain the raw "
                "question and may read only deterministic COMMIT among "
                "the latent states"
            )
        self.answer_context_mode = answer_context_mode
        self.answer_reads_question = True
        if not bool(
            self.trace_config.get("commit_causal_summary", True)
        ):
            raise ValueError(
                "TRACE-VB-v8 requires the deterministic causal COMMIT "
                "summary transition"
            )
        capability_bridge_hidden = int(
            self.trace_config.get("capability_bridge_hidden_size", 1024)
        )
        self.capability_state_norm = torch.nn.LayerNorm(self.hidden_size)
        self.capability_latent_bridge = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, capability_bridge_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(capability_bridge_hidden, self.hidden_size),
        )
        self.capability_anchor_gate_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, capability_bridge_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(capability_bridge_hidden, self.n_trace_steps),
        )
        self.capability_latent_queries = torch.nn.Parameter(
            torch.empty(self.n_trace_steps, self.hidden_size),
            requires_grad=False,
        )
        # Registered validation always used trace view id 0.  Preserve only
        # the actually exercised rows instead of carrying the unused 32-view
        # tables into the new method.
        self.capability_trace_view = torch.nn.Parameter(
            torch.empty(self.hidden_size), requires_grad=False
        )
        self.capability_trace_step_views = torch.nn.Parameter(
            torch.empty(self.n_trace_steps, self.hidden_size),
            requires_grad=False,
        )
        torch.nn.init.normal_(
            self.capability_latent_queries,
            mean=0.0,
            std=1.0 / math.sqrt(self.hidden_size),
        )
        torch.nn.init.zeros_(self.capability_trace_view)
        torch.nn.init.zeros_(self.capability_trace_step_views)
        for module in (
            self.capability_state_norm,
            self.capability_latent_bridge,
            self.capability_anchor_gate_predictor,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
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
        # The registered block bridge supplies the stable zero-action base.
        # Keeping the unused learned base projector trainable would both move
        # the anchor and create an unused-parameter DDP failure.
        for parameter in self.trajectory_policy.base_projector.parameters():
            parameter.requires_grad_(False)
        # This table is the deployment mirror of capability_latent_queries.
        # V8 rewinds it once and then treats it as part of the immutable
        # capability spine; role specialization remains in policy_step_embedding,
        # the policy trunk/heads, posterior, and action projector.
        for parameter in (
            self.trajectory_policy.dynamics_step_embedding.parameters()
        ):
            parameter.requires_grad_(False)
        # V5 keeps exactly one training-only CoT teacher path and one
        # question-only deployment path.  This restores the proven semantic
        # distillation bridge without recreating the old three-posterior
        # activation graph or any CPU saved-tensor offload.
        self.trajectory_posterior = CoTConditionedTrajectoryPosterior(
            hidden_size=self.hidden_size,
            action_dim=self.trajectory_policy.action_dim,
            n_steps=self.n_trace_steps,
            posterior_hidden_size=int(
                self.trace_config.get("posterior_hidden_size", 512)
            ),
            step_embedding_size=int(
                self.trace_config.get("posterior_step_embedding_size", 64)
            ),
            min_log_std=float(
                self.trace_config.get("posterior_min_log_std", -1.5)
            ),
            max_log_std=float(
                self.trace_config.get("posterior_max_log_std", 0.5)
            ),
        )
        self.posterior_context_norm = torch.nn.LayerNorm(self.hidden_size)

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
        self.register_buffer(
            "vb_score_pair_credit",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "vb_score_pair_count",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "vb_score_proxy_decided",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "vb_score_proxy_enabled",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "vb_last_stage1_kl",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )
        self.stage1_policy_reference = copy.deepcopy(
            self.trajectory_policy
        )
        for parameter in self.stage1_policy_reference.parameters():
            parameter.requires_grad_(False)

        self._solve_text_decoder_loaded = False
        self._capability_anchor_loaded = False
        self._capability_spine_rewound = False
        self._capability_spine_rewind_count = 0
        self._capability_spine_rewind_source = None
        self._zero_action_reset_schema_version = None
        self._zero_action_reset_applied = False
        self._zero_action_reset_operation_count = 0
        self._zero_action_reset_tensor_count = 0
        self._zero_action_reset_target_names: Tuple[str, ...] = ()
        self._zero_action_reset_source_schema = None
        self._metric_safe_baseline_path = None
        self._metric_safe_baseline_sha256 = None
        self._metric_safe_baseline_correct_count = -1
        self._metric_safe_baseline_questions = -1
        self._registered_capability_validation_path = None
        self._registered_capability_validation_sha256 = None
        self._registered_capability_validation_correct_count = -1
        self._registered_capability_validation_questions = -1
        self._registered_capability_checkpoint_sha256 = None
        self._registered_capability_payload_tensor_count = 0
        self._registered_capability_payload_sha256 = None
        self._v7_source_checkpoint_sha256 = None
        self._reference_restored = False
        self._loaded_stage2_state = False
        self._cot_encoder_adapter_loaded = False
        self._cot_encoder_checkpoint_sha256 = None
        self._stage2_initialized = False
        self._last_trace_metrics: Dict[str, torch.Tensor] = {}
        self._trace_visual_records: List[dict] = []
        self._validation_question_records: List[
            Tuple[int, float, int, str, bool, bool]
        ] = []
        self.strict_loading = False

        if self.do_trace_rl:
            required_stage2_objectives = (
                "use_trajectory_policy_loss",
                "use_evidence_gated_group_rl",
                "use_terminal_exact_reward",
            )
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
            update_epochs = int(
                self.trace_rl_config.get("policy_update_epochs", 2)
            )
            if not 1 <= update_epochs <= 2:
                raise ValueError(
                    "TRACE-VB-v8 permits one or two evidence-gated PPO "
                    "updates per rollout"
                )
            if bool(self.trace_rl_config.get("use_gae", False)):
                raise ValueError(
                    "TRACE-VB-v8 forbids critic-GAE actor advantages"
                )
            positive_stage2_weights = (
                "stage1_policy_kl_weight",
                "step_reward_weight",
            )
            invalid_weights = [
                key
                for key in positive_stage2_weights
                if float(self.trace_rl_config.get(key, 0.0)) <= 0.0
            ]
            if invalid_weights:
                raise ValueError(
                    "TRACE-VB-v8 requires positive trust-region and "
                    "role-local step weights; "
                    "non-positive weights: "
                    f"{invalid_weights}"
                )
            calibration_batches = int(
                self.trace_rl_config.get("score_calibration_batches", 128)
            )
            minimum_pairs = int(
                self.trace_rl_config.get("score_proxy_minimum_pairs", 64)
            )
            minimum_auc = float(
                self.trace_rl_config.get("score_proxy_minimum_auc", 0.60)
            )
            minimum_gap = float(
                self.trace_rl_config.get("minimum_gold_score_gap", 2.0e-3)
            )
            target_kl = float(
                self.trace_rl_config.get("stage1_policy_target_kl", 0.01)
            )
            if calibration_batches <= 0 or minimum_pairs <= 0:
                raise ValueError(
                    "score calibration batches and minimum pairs must be positive"
                )
            if not math.isfinite(minimum_auc) or not 0.5 <= minimum_auc <= 1.0:
                raise ValueError("score_proxy_minimum_auc must be in [0.5, 1]")
            if not math.isfinite(minimum_gap) or minimum_gap <= 0.0:
                raise ValueError("minimum_gold_score_gap must be positive")
            if not math.isfinite(target_kl) or target_kl <= 0.0:
                raise ValueError("stage1_policy_target_kl must be positive")
            step_discount = float(
                self.trace_rl_config.get("step_reward_discount", 0.90)
            )
            if not math.isfinite(step_discount) or not 0.0 <= step_discount <= 1.0:
                raise ValueError("step_reward_discount must lie in [0, 1]")
            self._initialize_stage2_modules()
            self.automatic_optimization = False

    def _initialize_stage2_modules(self):
        # The LM, transition semantics, COMMIT, PLAN forecast, and answer
        # channel are immutable. The stochastic role trunk, role-step features,
        # and PLAN/SOLVE/REFINE Gaussian heads are optimized; dynamics and the
        # former value critic remain frozen and cannot manufacture preferences.
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.trajectory_policy.set_stage2_trainability()
        self._set_adapter_parameter_trainability()
        self._activate_path_adapter()

    def configure_optimizers(self):
        """Use capability-safe Stage-1 and role-local Stage-2 LR groups."""
        if not self.do_trace_rl:
            (
                self.trainable_parameter_names,
                role_parameters,
            ) = collect_stage1_role_only_parameters(
                list(self.named_parameters()),
                path_adapter_name=self.path_adapter_name,
            )
            optimizer_config = self.all_config.model.training_kwargs.optimizer
            optimizer = torch.optim.AdamW(
                [
                    {
                        "name": "role_program",
                        "params": role_parameters,
                        "lr": float(
                            self.trace_config.get(
                                "stage1_role_lr", 2.0e-6
                            )
                        ),
                    },
                ],
                weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
                foreach=bool(optimizer_config.get("foreach", False)),
            )
            if not bool(
                self.all_config.model.training_kwargs.get(
                    "use_scheduler", False
                )
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

        self.trainable_parameter_names = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        actor_named_parameters = [
            (name, parameter)
            for name, parameter in self.trajectory_policy.named_parameters()
            if parameter.requires_grad
        ]
        actor_parameters = [parameter for _, parameter in actor_named_parameters]
        if not actor_parameters:
            raise RuntimeError("TRACE-VB-v8 Stage 2 requires role actor parameters")
        allowed_ids = {id(parameter) for parameter in actor_parameters}
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed_ids
        ]
        if unexpected:
            raise RuntimeError(
                "Stage-2 trainability escaped stochastic actor heads: "
                + ", ".join(unexpected[:20])
            )

        optimizer_config = self.all_config.model.training_kwargs.optimizer
        head_parameters = [
            parameter
            for name, parameter in actor_named_parameters
            if name.startswith("mean_heads.")
            or name.startswith("log_std_heads.")
        ]
        feature_parameters = [
            parameter
            for name, parameter in actor_named_parameters
            if not (
                name.startswith("mean_heads.")
                or name.startswith("log_std_heads.")
            )
        ]
        if not head_parameters or not feature_parameters:
            raise RuntimeError("Stage 2 role feature/head groups are incomplete")
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": head_parameters,
                    "lr": float(
                        self.trace_rl_config.get("actor_head_lr", 2.0e-6)
                    ),
                },
                {
                    "params": feature_parameters,
                    "lr": float(
                        self.trace_rl_config.get("actor_feature_lr", 4.0e-7)
                    ),
                }
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
        """Add stage-local recovery and host guards beside validation."""
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
        if self.do_trace_rl:
            stage2_recovery_interval = int(
                self.trace_config.get(
                    "stage2_recovery_checkpoint_interval",
                    64,
                )
            )
            if stage2_recovery_interval <= 0:
                raise RuntimeError(
                    "Stage 2 requires a positive "
                    "stage2_recovery_checkpoint_interval"
                )
            callbacks.append(
                Stage2RecoveryCheckpoint(stage2_recovery_interval)
            )
        return callbacks

    def _set_adapter_parameter_trainability(self):
        path_marker = f".{self.path_adapter_name}."
        cot_encoder_marker = f".{self.cot_encoder_adapter_name}."
        capability_marker = f".{self.capability_adapter_name}."
        answer_marker = f".{self.answer_adapter_name}."
        for name, parameter in self.llm.named_parameters():
            if cot_encoder_marker in name:
                parameter.requires_grad_(False)
            elif capability_marker in name:
                parameter.requires_grad_(False)
            elif path_marker in name:
                # V8's deployment LoRA is rewound to the registered
                # capability adapter exactly once and is immutable in both
                # stages.  Role learning happens outside this adapter.
                parameter.requires_grad_(False)
            elif answer_marker in name:
                parameter.requires_grad_(False)

    def _activate_path_adapter(self):
        self.llm.set_adapter(self.path_adapter_name)
        self._set_adapter_parameter_trainability()

    def _activate_cot_encoder_adapter(self):
        self.llm.set_adapter(self.cot_encoder_adapter_name)
        self._set_adapter_parameter_trainability()

    def _activate_capability_adapter(self):
        self.llm.set_adapter(self.capability_adapter_name)
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

    def validate_v8_warm_start_coverage(self, source_state: dict) -> dict:
        """Fail closed if any retained role/posterior/semantic state is absent."""
        required_prefixes = (
            "state_norm.",
            "trajectory_policy.",
            "trajectory_posterior.",
            "posterior_context_norm.",
            "plan_forecaster.",
            "solve_text_decoder.",
            "sufficiency_head.",
            "semantic_projection",
        )
        required_names = validate_required_state_coverage(
            super().state_dict(),
            source_state,
            required_prefixes=required_prefixes,
        )
        return {
            "retained_tensor_count": len(required_names),
            "required_prefixes": list(required_prefixes),
        }

    def register_v8_capability_provenance(
        self,
        *,
        v7_source_state: dict,
        v7_source_checkpoint_sha256: str,
        registered_capability_state: dict,
        registered_capability_checkpoint_sha256: str,
    ) -> dict:
        """Register a byte-verified legacy capability before v7 rewind."""
        expected_checkpoint_sha = str(
            self.trace_config.get(
                "registered_capability_checkpoint_sha256", ""
            )
        ).strip().lower()
        actual_checkpoint_sha = str(
            registered_capability_checkpoint_sha256
        ).strip().lower()
        if (
            len(expected_checkpoint_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_checkpoint_sha
            )
            or actual_checkpoint_sha != expected_checkpoint_sha
        ):
            raise RuntimeError(
                "registered capability checkpoint SHA256 mismatch: "
                f"expected {expected_checkpoint_sha or '<missing>'}, "
                f"found {actual_checkpoint_sha or '<missing>'}"
            )
        v7_source_sha = str(
            v7_source_checkpoint_sha256
        ).strip().lower()
        if len(v7_source_sha) != 64 or any(
            character not in "0123456789abcdef"
            for character in v7_source_sha
        ):
            raise RuntimeError("v7 source checkpoint SHA256 is invalid")
        payload_report = verify_v7_registered_capability_payload(
            v7_source_state,
            registered_capability_state,
            path_adapter_name=self.path_adapter_name,
            capability_adapter_name=self.capability_adapter_name,
            expected_lora_tensors=int(
                self.trace_config.get(
                    "capability_expected_lora_tensors", 504
                )
            ),
            n_trace_steps=self.n_trace_steps,
        )
        expected_payload_tensors = int(
            self.trace_config.get(
                "registered_capability_payload_tensors", 517
            )
        )
        if int(payload_report["tensor_count"]) != expected_payload_tensors:
            raise RuntimeError(
                "registered capability canonical payload coverage mismatch: "
                f"expected {expected_payload_tensors}, found "
                f"{payload_report['tensor_count']}"
            )
        expected_payload_sha = str(
            self.trace_config.get(
                "registered_capability_payload_sha256", ""
            )
        ).strip().lower()
        actual_payload_sha = str(
            payload_report["canonical_payload_sha256"]
        ).strip().lower()
        if (
            len(expected_payload_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_payload_sha
            )
            or actual_payload_sha != expected_payload_sha
        ):
            raise RuntimeError(
                "registered capability canonical payload SHA256 mismatch: "
                f"expected {expected_payload_sha or '<missing>'}, "
                f"found {actual_payload_sha}"
            )
        self._registered_capability_checkpoint_sha256 = (
            actual_checkpoint_sha
        )
        self._registered_capability_payload_tensor_count = int(
            payload_report["tensor_count"]
        )
        self._registered_capability_payload_sha256 = actual_payload_sha
        self._v7_source_checkpoint_sha256 = v7_source_sha
        return {
            **payload_report,
            "registered_capability_checkpoint_sha256": (
                actual_checkpoint_sha
            ),
            "v7_source_checkpoint_sha256": v7_source_sha,
        }

    def _assert_registered_capability_provenance(self) -> None:
        expected_checkpoint_sha = str(
            self.trace_config.get(
                "registered_capability_checkpoint_sha256", ""
            )
        ).strip().lower()
        expected_payload_sha = str(
            self.trace_config.get(
                "registered_capability_payload_sha256", ""
            )
        ).strip().lower()
        expected_payload_tensors = int(
            self.trace_config.get(
                "registered_capability_payload_tensors", 517
            )
        )
        for label, value in (
            ("registered capability checkpoint", expected_checkpoint_sha),
            ("canonical capability payload", expected_payload_sha),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef"
                for character in value
            ):
                raise RuntimeError(f"{label} registry SHA256 is invalid")
        if (
            self._registered_capability_checkpoint_sha256
            != expected_checkpoint_sha
        ):
            raise RuntimeError(
                "registered capability checkpoint provenance is absent or "
                "does not match the v8 registry"
            )
        if (
            int(self._registered_capability_payload_tensor_count)
            != expected_payload_tensors
        ):
            raise RuntimeError(
                "canonical capability payload tensor count is absent or "
                "does not match the v8 registry"
            )
        if (
            self._registered_capability_payload_sha256
            != expected_payload_sha
        ):
            raise RuntimeError(
                "canonical capability payload provenance is absent or does "
                "not match the v8 registry"
            )
        v7_source_sha = str(
            self._v7_source_checkpoint_sha256 or ""
        ).strip().lower()
        if len(v7_source_sha) != 64 or any(
            character not in "0123456789abcdef"
            for character in v7_source_sha
        ):
            raise RuntimeError("v7 source checkpoint provenance is missing")

    def initialize_v8_capability_spine(
        self,
        source_checkpoint: dict,
        *,
        allow_v7_rewind: bool = False,
    ) -> dict:
        """Initialize or verify the one-time v8 capability rewind.

        A v7 weights-only initialization performs the only mutating rewind.
        Loading a v8 checkpoint never rewinds again: its saved parity metadata
        and actual tensors must already satisfy the same exact contract.
        """
        source_schema = str(
            source_checkpoint.get("trace_vb_schema_version", "")
        )
        expected = int(
            self.trace_config.get(
                "capability_expected_lora_tensors", 504
            )
        )
        if source_schema == "trace_vb_v8":
            if allow_v7_rewind:
                raise RuntimeError(
                    "a v8 checkpoint must never request another capability "
                    "rewind"
                )
            metric_contract = resolve_v8_metric_provenance(
                self.trace_config
            )
            student_baseline = metric_contract["metric_safe_baseline"]
            capability_validation = metric_contract[
                "registered_capability_validation"
            ]
            metadata = validate_v8_checkpoint_provenance(
                source_checkpoint,
                expected_lora_tensors=expected,
                expected_cot_sha256=str(
                    self.trace_config.get(
                        "stage0_cot_encoder_checkpoint_sha256", ""
                    )
                ),
                expected_registered_capability_checkpoint_sha256=str(
                    self.trace_config.get(
                        "registered_capability_checkpoint_sha256", ""
                    )
                ),
                expected_registered_capability_payload_tensors=int(
                    self.trace_config.get(
                        "registered_capability_payload_tensors", 517
                    )
                ),
                expected_registered_capability_payload_sha256=str(
                    self.trace_config.get(
                        "registered_capability_payload_sha256", ""
                    )
                ),
                expected_metric_safe_baseline_path=(
                    student_baseline["path"]
                ),
                expected_metric_safe_baseline_sha256=(
                    student_baseline["sha256"]
                ),
                expected_metric_safe_baseline_correct_count=(
                    student_baseline["correct_count"]
                ),
                expected_metric_safe_baseline_questions=(
                    student_baseline["questions"]
                ),
                expected_registered_capability_validation_path=(
                    capability_validation["path"]
                ),
                expected_registered_capability_validation_sha256=(
                    capability_validation["sha256"]
                ),
                expected_registered_capability_validation_correct_count=(
                    capability_validation["correct_count"]
                ),
                expected_registered_capability_validation_questions=(
                    capability_validation["questions"]
                ),
            )
            self._capability_spine_rewound = True
            self._capability_spine_rewind_count = metadata[
                "rewind_count"
            ]
            self._capability_spine_rewind_source = metadata[
                "rewind_source"
            ]
            self._cot_encoder_checkpoint_sha256 = metadata[
                "cot_encoder_sha256"
            ]
            self._registered_capability_checkpoint_sha256 = metadata[
                "registered_capability_checkpoint_sha256"
            ]
            self._registered_capability_payload_tensor_count = metadata[
                "registered_capability_payload_tensors"
            ]
            self._registered_capability_payload_sha256 = metadata[
                "registered_capability_payload_sha256"
            ]
            self._v7_source_checkpoint_sha256 = metadata[
                "v7_source_checkpoint_sha256"
            ]
            self._zero_action_reset_schema_version = metadata[
                "zero_action_reset_schema_version"
            ]
            self._zero_action_reset_applied = metadata[
                "zero_action_reset_applied"
            ]
            self._zero_action_reset_operation_count = metadata[
                "zero_action_reset_operation_count"
            ]
            self._zero_action_reset_tensor_count = metadata[
                "zero_action_reset_tensor_count"
            ]
            self._zero_action_reset_target_names = tuple(
                metadata["zero_action_reset_target_names"]
            )
            self._zero_action_reset_source_schema = metadata[
                "zero_action_reset_source_schema"
            ]
            self._metric_safe_baseline_path = student_baseline["path"]
            self._metric_safe_baseline_sha256 = student_baseline["sha256"]
            self._metric_safe_baseline_correct_count = student_baseline[
                "correct_count"
            ]
            self._metric_safe_baseline_questions = student_baseline[
                "questions"
            ]
            self._registered_capability_validation_path = (
                capability_validation["path"]
            )
            self._registered_capability_validation_sha256 = (
                capability_validation["sha256"]
            )
            self._registered_capability_validation_correct_count = (
                capability_validation["correct_count"]
            )
            self._registered_capability_validation_questions = (
                capability_validation["questions"]
            )
            return self._assert_capability_spine_contract()
        if source_schema != "trace_vb_v7":
            raise RuntimeError(
                "TRACE-VB-v8 weights-only initialization requires a "
                "trace_vb_v7 or trace_vb_v8 checkpoint, found "
                f"{source_schema!r}"
            )
        if not allow_v7_rewind:
            raise RuntimeError(
                "v7 weights-only initialization requires the explicit "
                "rewind authorization"
            )
        # Provenance must be established from the independent registered
        # checkpoint before any deployment tensor is mutated.
        self._assert_registered_capability_provenance()
        if self._capability_spine_rewound:
            raise RuntimeError(
                "capability spine rewind was requested more than once"
            )
        if self._zero_action_reset_applied:
            raise RuntimeError(
                "zero-action policy reset was requested more than once"
            )
        if not self._capability_anchor_loaded:
            raise RuntimeError(
                "cannot rewind before the complete capability anchor loads"
            )
        metric_contract = resolve_v8_metric_provenance(self.trace_config)
        student_baseline = metric_contract["metric_safe_baseline"]
        capability_validation = metric_contract[
            "registered_capability_validation"
        ]
        copied = rewind_capability_spine_tensors(
            dict(self.llm.named_parameters()),
            self.capability_latent_queries,
            self.trajectory_policy.dynamics_step_embedding.weight,
            path_adapter_name=self.path_adapter_name,
            capability_adapter_name=self.capability_adapter_name,
            expected_tensors=expected,
        )
        reset_report = reset_v8_zero_action_policy_tensors(
            self.trajectory_policy
        )
        self._capability_spine_rewound = True
        self._capability_spine_rewind_count = copied
        self._capability_spine_rewind_source = "trace_vb_v7_capability"
        self._zero_action_reset_schema_version = reset_report[
            "schema_version"
        ]
        self._zero_action_reset_applied = True
        self._zero_action_reset_operation_count = reset_report[
            "operation_count"
        ]
        self._zero_action_reset_tensor_count = reset_report["tensor_count"]
        self._zero_action_reset_target_names = tuple(
            reset_report["tensor_names"]
        )
        self._zero_action_reset_source_schema = "trace_vb_v7"
        self._metric_safe_baseline_path = student_baseline["path"]
        self._metric_safe_baseline_sha256 = student_baseline["sha256"]
        self._metric_safe_baseline_correct_count = student_baseline[
            "correct_count"
        ]
        self._metric_safe_baseline_questions = student_baseline["questions"]
        self._registered_capability_validation_path = capability_validation[
            "path"
        ]
        self._registered_capability_validation_sha256 = capability_validation[
            "sha256"
        ]
        self._registered_capability_validation_correct_count = (
            capability_validation["correct_count"]
        )
        self._registered_capability_validation_questions = (
            capability_validation["questions"]
        )
        self._set_adapter_parameter_trainability()
        self.trajectory_policy.dynamics_step_embedding.weight.requires_grad_(
            False
        )
        return self._assert_capability_spine_contract()

    def _assert_capability_spine_contract(self) -> dict:
        if not self._capability_anchor_loaded:
            raise RuntimeError(
                "TRACE-VB-v8 requires a fully mapped capability anchor"
            )
        if not self._capability_spine_rewound:
            raise RuntimeError(
                "TRACE-VB-v8 capability spine rewind is not registered"
            )
        expected = int(
            self.trace_config.get(
                "capability_expected_lora_tensors", 504
            )
        )
        if int(self._capability_spine_rewind_count) != expected:
            raise RuntimeError(
                "registered capability rewind count is incomplete: "
                f"{self._capability_spine_rewind_count} != {expected}"
            )
        copied = assert_capability_spine_parity(
            dict(self.llm.named_parameters()),
            self.capability_latent_queries,
            self.trajectory_policy.dynamics_step_embedding.weight,
            path_adapter_name=self.path_adapter_name,
            capability_adapter_name=self.capability_adapter_name,
            expected_tensors=expected,
        )
        path_marker = f".{self.path_adapter_name}."
        trainable_path = [
            name
            for name, parameter in self.llm.named_parameters()
            if path_marker in name and parameter.requires_grad
        ]
        if trainable_path:
            raise RuntimeError(
                "deployment LoRA is not frozen: "
                + ", ".join(trainable_path[:8])
            )
        capability_marker = f".{self.capability_adapter_name}."
        trainable_capability = [
            name
            for name, parameter in self.llm.named_parameters()
            if capability_marker in name and parameter.requires_grad
        ]
        if trainable_capability:
            raise RuntimeError(
                "capability adapter is not frozen: "
                + ", ".join(trainable_capability[:8])
            )
        if self.capability_latent_queries.requires_grad:
            raise RuntimeError("capability latent queries are not frozen")
        if (
            self.trajectory_policy.dynamics_step_embedding.weight
            .requires_grad
        ):
            raise RuntimeError("capability query prior is not frozen")
        return {
            "lora_tensors": copied,
            "query_prior_exact": True,
            "deployment_lora_frozen": True,
            "query_prior_frozen": True,
        }

    def _assert_cot_encoder_provenance(self) -> None:
        expected = str(
            self.trace_config.get(
                "stage0_cot_encoder_checkpoint_sha256", ""
            )
        ).strip().lower()
        actual = str(
            self._cot_encoder_checkpoint_sha256 or ""
        ).strip().lower()
        if not expected or len(expected) != 64:
            raise RuntimeError(
                "TRACE-VB-v8 requires a registered Stage-0 encoder SHA256"
            )
        if actual != expected:
            raise RuntimeError(
                "Stage-0 CoT encoder provenance mismatch: expected "
                f"{expected}, found {actual or '<missing>'}"
            )

    def _assert_zero_action_reset_contract(self) -> dict:
        """Validate one-time reset provenance without constraining trained heads."""
        if self._zero_action_reset_schema_version != (
            TRACE_VB_ZERO_ACTION_RESET_SCHEMA
        ):
            raise RuntimeError("zero-action reset schema is not registered")
        if self._zero_action_reset_applied is not True:
            raise RuntimeError("zero-action reset is not registered")
        if (
            type(self._zero_action_reset_operation_count) is not int
            or self._zero_action_reset_operation_count != 1
        ):
            raise RuntimeError(
                "zero-action reset operation count is not exactly one"
            )
        if (
            type(self._zero_action_reset_tensor_count) is not int
            or self._zero_action_reset_tensor_count
            != len(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES)
        ):
            raise RuntimeError("zero-action reset tensor count is not nine")
        if tuple(self._zero_action_reset_target_names) != (
            TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES
        ):
            raise RuntimeError("zero-action reset target coverage is not exact")
        if self._zero_action_reset_source_schema != "trace_vb_v7":
            raise RuntimeError("zero-action reset source schema is not v7")
        expected = resolve_v8_metric_provenance(self.trace_config)
        observed = {
            "metric_safe_baseline": {
                "path": self._metric_safe_baseline_path,
                "sha256": self._metric_safe_baseline_sha256,
                "correct_count": self._metric_safe_baseline_correct_count,
                "questions": self._metric_safe_baseline_questions,
            },
            "registered_capability_validation": {
                "path": self._registered_capability_validation_path,
                "sha256": self._registered_capability_validation_sha256,
                "correct_count": (
                    self._registered_capability_validation_correct_count
                ),
                "questions": self._registered_capability_validation_questions,
            },
        }
        if observed != expected:
            raise RuntimeError(
                "zero-action reset metric provenance does not match config"
            )
        return {
            "zero_action_reset_schema_version": (
                self._zero_action_reset_schema_version
            ),
            "zero_action_reset_operation_count": 1,
            "zero_action_reset_tensor_count": len(
                TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES
            ),
            "metric_safe_baseline_correct_count": observed[
                "metric_safe_baseline"
            ]["correct_count"],
            "registered_capability_validation_correct_count": observed[
                "registered_capability_validation"
            ]["correct_count"],
        }

    def assert_v8_initialization_contract(self) -> dict:
        """Verify the immutable spine and strong Stage-0 provenance."""
        report = self._assert_capability_spine_contract()
        report.update(self._assert_zero_action_reset_contract())
        self._assert_registered_capability_provenance()
        if not self._cot_encoder_adapter_loaded:
            raise RuntimeError(
                "TRACE-VB-v8 requires the registered CoT encoder adapter"
            )
        self._assert_cot_encoder_provenance()
        return report

    def _snapshot_stage1_policy(self):
        self.stage1_policy_reference.load_state_dict(
            self.trajectory_policy.state_dict(),
            strict=True,
        )
        self.stage1_policy_reference.eval()
        for parameter in self.stage1_policy_reference.parameters():
            parameter.requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = collections.OrderedDict(state_dict)
        legacy_bridge_keys = (
            "latent_bridge.0.weight",
            "latent_bridge.0.bias",
            "latent_bridge.2.weight",
            "latent_bridge.2.bias",
            "state_norm.weight",
            "state_norm.bias",
            "step_compressor.latent_queries",
            "anchor_gate_predictor.0.weight",
            "anchor_gate_predictor.0.bias",
            "anchor_gate_predictor.2.weight",
            "anchor_gate_predictor.2.bias",
            "trace_view_embeddings.weight",
            "trace_step_view_embeddings.weight",
        )
        is_legacy_capability_checkpoint = (
            all(name in state_dict for name in legacy_bridge_keys)
            and not any(
                name.startswith("trajectory_policy.") for name in state_dict
            )
        )
        if is_legacy_capability_checkpoint:
            legacy_lora_names = sorted(
                name
                for name in state_dict
                if name.startswith("llm.")
                and ".default." in name
                and (".lora_A." in name or ".lora_B." in name)
                and name.endswith(".weight")
            )
            expected_lora_tensors = int(
                self.trace_config.get(
                    "capability_expected_lora_tensors", 504
                )
            )
            if len(legacy_lora_names) != expected_lora_tensors:
                raise RuntimeError(
                    "capability checkpoint LoRA coverage mismatch: expected "
                    f"{expected_lora_tensors}, found {len(legacy_lora_names)}"
                )
            model_state = super().state_dict()
            for source_name in legacy_lora_names:
                target_name = source_name.replace(
                    f".{self.path_adapter_name}.",
                    f".{self.capability_adapter_name}.",
                )
                if target_name not in model_state:
                    raise RuntimeError(
                        "capability adapter is missing mapped tensor "
                        f"{target_name}"
                    )
                if model_state[target_name].shape != state_dict[source_name].shape:
                    raise RuntimeError(
                        f"capability LoRA shape mismatch for {source_name}"
                    )
                state_dict[target_name] = state_dict[source_name]

            bridge_mapping = {
                "state_norm.weight": "capability_state_norm.weight",
                "state_norm.bias": "capability_state_norm.bias",
                "latent_bridge.0.weight": "capability_latent_bridge.0.weight",
                "latent_bridge.0.bias": "capability_latent_bridge.0.bias",
                "latent_bridge.2.weight": "capability_latent_bridge.2.weight",
                "latent_bridge.2.bias": "capability_latent_bridge.2.bias",
                "step_compressor.latent_queries": "capability_latent_queries",
                "anchor_gate_predictor.0.weight": (
                    "capability_anchor_gate_predictor.0.weight"
                ),
                "anchor_gate_predictor.0.bias": (
                    "capability_anchor_gate_predictor.0.bias"
                ),
                "anchor_gate_predictor.2.weight": (
                    "capability_anchor_gate_predictor.2.weight"
                ),
                "anchor_gate_predictor.2.bias": (
                    "capability_anchor_gate_predictor.2.bias"
                ),
            }
            for source_name, target_name in bridge_mapping.items():
                if target_name not in model_state:
                    raise RuntimeError(
                        f"student is missing capability tensor {target_name}"
                    )
                if model_state[target_name].shape != state_dict[source_name].shape:
                    raise RuntimeError(
                        f"capability bridge shape mismatch for {source_name}"
                    )
                state_dict[target_name] = state_dict[source_name]
            trace_view_table = state_dict["trace_view_embeddings.weight"]
            trace_step_view_table = state_dict[
                "trace_step_view_embeddings.weight"
            ]
            if (
                trace_view_table.ndim != 2
                or trace_view_table.shape[0] < 1
                or trace_view_table.shape[1] != self.hidden_size
            ):
                raise RuntimeError("capability trace-view table has wrong shape")
            if (
                trace_step_view_table.ndim != 2
                or trace_step_view_table.shape[0] < self.n_trace_steps
                or trace_step_view_table.shape[1] != self.hidden_size
            ):
                raise RuntimeError(
                    "capability trace-step-view table has wrong shape"
                )
            state_dict["capability_trace_view"] = (
                trace_view_table[0].clone()
            )
            state_dict["capability_trace_step_views"] = (
                trace_step_view_table[: self.n_trace_steps].clone()
            )
            step_prior_name = (
                "trajectory_policy.dynamics_step_embedding.weight"
            )
            if (
                model_state[step_prior_name].shape
                != state_dict["step_compressor.latent_queries"].shape
            ):
                raise RuntimeError("legacy latent queries cannot seed role steps")
            state_dict[step_prior_name] = state_dict[
                "step_compressor.latent_queries"
            ]
            self._capability_anchor_loaded = True

        capability_marker = f".{self.capability_adapter_name}."
        capability_lora_names = sorted(
            name
            for name in state_dict
            if capability_marker in name
            and (".lora_A." in name or ".lora_B." in name)
            and name.endswith(".weight")
        )
        capability_required = (
            "capability_state_norm.weight",
            "capability_state_norm.bias",
            "capability_latent_bridge.0.weight",
            "capability_latent_bridge.0.bias",
            "capability_latent_bridge.2.weight",
            "capability_latent_bridge.2.bias",
            "capability_anchor_gate_predictor.0.weight",
            "capability_anchor_gate_predictor.0.bias",
            "capability_anchor_gate_predictor.2.weight",
            "capability_anchor_gate_predictor.2.bias",
            "capability_latent_queries",
            "capability_trace_view",
            "capability_trace_step_views",
        )
        capability_present = bool(capability_lora_names) or any(
            name in state_dict for name in capability_required
        )
        if capability_present:
            expected = int(
                self.trace_config.get(
                    "capability_expected_lora_tensors", 504
                )
            )
            if len(capability_lora_names) != expected:
                raise RuntimeError(
                    "capability adapter coverage mismatch: expected "
                    f"{expected}, found {len(capability_lora_names)}"
                )
            missing = [
                name for name in capability_required if name not in state_dict
            ]
            if missing:
                raise RuntimeError(
                    "capability checkpoint is incomplete: "
                    + ", ".join(missing)
                )
            expected_state = super().state_dict()
            for name in (*capability_lora_names, *capability_required):
                if name not in expected_state:
                    raise RuntimeError(
                        f"capability checkpoint has unknown tensor {name}"
                    )
                if expected_state[name].shape != state_dict[name].shape:
                    raise RuntimeError(
                        f"capability checkpoint shape mismatch for {name}"
                    )
            self._capability_anchor_loaded = True
        answer_marker = f".{self.answer_adapter_name}."
        cot_encoder_marker = f".{self.cot_encoder_adapter_name}."
        # Stage 2 has no trainable answer adapter, so checkpoint metadata set
        # by on_load_checkpoint is the authoritative stage marker.
        self._loaded_stage2_state = (
            self._loaded_stage2_state
            or any(answer_marker in name for name in state_dict)
        )
        cot_encoder_names = sorted(
            name
            for name in state_dict
            if cot_encoder_marker in name
            and (".lora_A." in name or ".lora_B." in name)
            and name.endswith(".weight")
        )
        if cot_encoder_names:
            expected_cot = int(
                self.trace_config.get("stage0_expected_lora_tensors", 504)
            )
            if len(cot_encoder_names) != expected_cot:
                raise RuntimeError(
                    "CoT encoder adapter coverage mismatch: expected "
                    f"{expected_cot}, found {len(cot_encoder_names)}"
                )
            expected_state = super().state_dict()
            for name in cot_encoder_names:
                if (
                    name not in expected_state
                    or expected_state[name].shape != state_dict[name].shape
                ):
                    raise RuntimeError(
                        f"CoT encoder checkpoint shape mismatch for {name}"
                    )
            self._cot_encoder_adapter_loaded = True
        self._solve_text_decoder_loaded = (
            self._solve_text_decoder_loaded
            or any(name.startswith("solve_text_decoder.") for name in state_dict)
        )
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    def load_cot_encoder_state_dict(
        self,
        state_dict,
        *,
        checkpoint_sha256: Optional[str] = None,
    ):
        """Load only the registered Stage-0 LoRA into the frozen CoT encoder.

        This deliberately cannot overwrite the deployment/path adapter or the
        frozen capability adapter.  Stage-0 is used solely to encode the
        already-observed gold CoT that supplies Stage-1 role targets.
        """
        source_names = sorted(
            name
            for name in state_dict
            if name.startswith("llm.")
            and f".{self.path_adapter_name}." in name
            and (".lora_A." in name or ".lora_B." in name)
            and name.endswith(".weight")
        )
        expected = int(
            self.trace_config.get("stage0_expected_lora_tensors", 504)
        )
        if len(source_names) != expected:
            raise RuntimeError(
                "Stage-0 CoT encoder LoRA coverage mismatch: expected "
                f"{expected}, found {len(source_names)}"
            )
        model_state = super().state_dict()
        mapped = collections.OrderedDict()
        for source_name in source_names:
            target_name = source_name.replace(
                f".{self.path_adapter_name}.",
                f".{self.cot_encoder_adapter_name}.",
            )
            if target_name not in model_state:
                raise RuntimeError(
                    f"CoT encoder adapter is missing {target_name}"
                )
            value = state_dict[source_name]
            if model_state[target_name].shape != value.shape:
                raise RuntimeError(
                    f"Stage-0 CoT encoder shape mismatch for {source_name}"
                )
            mapped[target_name] = value
        incompatible = super().load_state_dict(mapped, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected Stage-0 CoT encoder keys: "
                + ", ".join(incompatible.unexpected_keys[:20])
            )
        self._cot_encoder_adapter_loaded = True
        if checkpoint_sha256 is not None:
            normalized_sha = str(checkpoint_sha256).strip().lower()
            if len(normalized_sha) != 64 or any(
                character not in "0123456789abcdef"
                for character in normalized_sha
            ):
                raise ValueError("CoT encoder SHA256 must be 64 hex digits")
            self._cot_encoder_checkpoint_sha256 = normalized_sha
        self._set_adapter_parameter_trainability()
        return incompatible

    def on_load_checkpoint(self, checkpoint):
        schema_version = str(
            checkpoint.get("trace_vb_schema_version", "")
        )
        if schema_version == "trace_vb_v8":
            metric_contract = resolve_v8_metric_provenance(
                self.trace_config
            )
            student_baseline = metric_contract["metric_safe_baseline"]
            capability_validation = metric_contract[
                "registered_capability_validation"
            ]
            metadata = validate_v8_checkpoint_provenance(
                checkpoint,
                expected_lora_tensors=int(
                    self.trace_config.get(
                        "capability_expected_lora_tensors", 504
                    )
                ),
                expected_cot_sha256=str(
                    self.trace_config.get(
                        "stage0_cot_encoder_checkpoint_sha256", ""
                    )
                ),
                expected_registered_capability_checkpoint_sha256=str(
                    self.trace_config.get(
                        "registered_capability_checkpoint_sha256", ""
                    )
                ),
                expected_registered_capability_payload_tensors=int(
                    self.trace_config.get(
                        "registered_capability_payload_tensors", 517
                    )
                ),
                expected_registered_capability_payload_sha256=str(
                    self.trace_config.get(
                        "registered_capability_payload_sha256", ""
                    )
                ),
                expected_metric_safe_baseline_path=(
                    student_baseline["path"]
                ),
                expected_metric_safe_baseline_sha256=(
                    student_baseline["sha256"]
                ),
                expected_metric_safe_baseline_correct_count=(
                    student_baseline["correct_count"]
                ),
                expected_metric_safe_baseline_questions=(
                    student_baseline["questions"]
                ),
                expected_registered_capability_validation_path=(
                    capability_validation["path"]
                ),
                expected_registered_capability_validation_sha256=(
                    capability_validation["sha256"]
                ),
                expected_registered_capability_validation_correct_count=(
                    capability_validation["correct_count"]
                ),
                expected_registered_capability_validation_questions=(
                    capability_validation["questions"]
                ),
            )
            self._capability_spine_rewound = True
            self._capability_spine_rewind_count = metadata[
                "rewind_count"
            ]
            self._capability_spine_rewind_source = metadata[
                "rewind_source"
            ]
            self._cot_encoder_checkpoint_sha256 = metadata[
                "cot_encoder_sha256"
            ]
            self._registered_capability_checkpoint_sha256 = metadata[
                "registered_capability_checkpoint_sha256"
            ]
            self._registered_capability_payload_tensor_count = metadata[
                "registered_capability_payload_tensors"
            ]
            self._registered_capability_payload_sha256 = metadata[
                "registered_capability_payload_sha256"
            ]
            self._v7_source_checkpoint_sha256 = metadata[
                "v7_source_checkpoint_sha256"
            ]
            self._zero_action_reset_schema_version = metadata[
                "zero_action_reset_schema_version"
            ]
            self._zero_action_reset_applied = metadata[
                "zero_action_reset_applied"
            ]
            self._zero_action_reset_operation_count = metadata[
                "zero_action_reset_operation_count"
            ]
            self._zero_action_reset_tensor_count = metadata[
                "zero_action_reset_tensor_count"
            ]
            self._zero_action_reset_target_names = tuple(
                metadata["zero_action_reset_target_names"]
            )
            self._zero_action_reset_source_schema = metadata[
                "zero_action_reset_source_schema"
            ]
            self._metric_safe_baseline_path = student_baseline["path"]
            self._metric_safe_baseline_sha256 = student_baseline["sha256"]
            self._metric_safe_baseline_correct_count = student_baseline[
                "correct_count"
            ]
            self._metric_safe_baseline_questions = student_baseline[
                "questions"
            ]
            self._registered_capability_validation_path = (
                capability_validation["path"]
            )
            self._registered_capability_validation_sha256 = (
                capability_validation["sha256"]
            )
            self._registered_capability_validation_correct_count = (
                capability_validation["correct_count"]
            )
            self._registered_capability_validation_questions = (
                capability_validation["questions"]
            )
        else:
            self._capability_spine_rewound = False
            self._capability_spine_rewind_count = 0
            self._capability_spine_rewind_source = None
            self._cot_encoder_checkpoint_sha256 = None
            self._registered_capability_checkpoint_sha256 = None
            self._registered_capability_payload_tensor_count = 0
            self._registered_capability_payload_sha256 = None
            self._v7_source_checkpoint_sha256 = None
            self._zero_action_reset_schema_version = None
            self._zero_action_reset_applied = False
            self._zero_action_reset_operation_count = 0
            self._zero_action_reset_tensor_count = 0
            self._zero_action_reset_target_names = ()
            self._zero_action_reset_source_schema = None
            self._metric_safe_baseline_path = None
            self._metric_safe_baseline_sha256 = None
            self._metric_safe_baseline_correct_count = -1
            self._metric_safe_baseline_questions = -1
            self._registered_capability_validation_path = None
            self._registered_capability_validation_sha256 = None
            self._registered_capability_validation_correct_count = -1
            self._registered_capability_validation_questions = -1
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
        # validate-only never configures an optimizer, while the base class
        # filters checkpoints through this list. Populate it from the actual
        # trainability state so the step-0 candidate is self-contained.
        if not hasattr(self, "trainable_parameter_names"):
            self.trainable_parameter_names = [
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            ]
        super().on_save_checkpoint(checkpoint)
        full_state = self.state_dict()
        preserve_prefixes = (
            "trajectory_policy.",
            "trajectory_posterior.",
            "posterior_context_norm.",
            "plan_forecaster.",
            "solve_text_decoder.",
            "sufficiency_head.",
            "value_critic.",
            "semantic_projection",
            "vb_",
            "state_norm.",
            "capability_state_norm.",
            "capability_latent_bridge.",
            "capability_anchor_gate_predictor.",
            "capability_latent_queries",
            "capability_trace_view",
            "capability_trace_step_views",
        )
        path_marker = f".{self.path_adapter_name}."
        cot_encoder_marker = f".{self.cot_encoder_adapter_name}."
        capability_marker = f".{self.capability_adapter_name}."
        for name, value in full_state.items():
            if (
                name.startswith(preserve_prefixes)
                or path_marker in name
                or cot_encoder_marker in name
                or capability_marker in name
            ):
                checkpoint["state_dict"][name] = value
        checkpoint_stage = 2 if self.do_trace_rl else 1
        checkpoint["trace_policy_training_stage"] = checkpoint_stage
        write_v8_checkpoint_provenance(
            checkpoint,
            rewound=self._capability_spine_rewound,
            rewind_count=self._capability_spine_rewind_count,
            rewind_source=self._capability_spine_rewind_source or "",
            cot_encoder_sha256=(
                self._cot_encoder_checkpoint_sha256 or ""
            ),
            registered_capability_checkpoint_sha256=(
                self._registered_capability_checkpoint_sha256 or ""
            ),
            registered_capability_payload_tensors=(
                self._registered_capability_payload_tensor_count
            ),
            registered_capability_payload_sha256=(
                self._registered_capability_payload_sha256 or ""
            ),
            v7_source_checkpoint_sha256=(
                self._v7_source_checkpoint_sha256 or ""
            ),
            zero_action_reset_schema_version=(
                self._zero_action_reset_schema_version or ""
            ),
            zero_action_reset_applied=self._zero_action_reset_applied,
            zero_action_reset_operation_count=(
                self._zero_action_reset_operation_count
            ),
            zero_action_reset_tensor_count=(
                self._zero_action_reset_tensor_count
            ),
            zero_action_reset_target_names=(
                self._zero_action_reset_target_names
            ),
            zero_action_reset_source_schema=(
                self._zero_action_reset_source_schema or ""
            ),
            metric_safe_baseline_path=(
                self._metric_safe_baseline_path or ""
            ),
            metric_safe_baseline_sha256=(
                self._metric_safe_baseline_sha256 or ""
            ),
            metric_safe_baseline_correct_count=(
                self._metric_safe_baseline_correct_count
            ),
            metric_safe_baseline_questions=(
                self._metric_safe_baseline_questions
            ),
            registered_capability_validation_path=(
                self._registered_capability_validation_path or ""
            ),
            registered_capability_validation_sha256=(
                self._registered_capability_validation_sha256 or ""
            ),
            registered_capability_validation_correct_count=(
                self._registered_capability_validation_correct_count
            ),
            registered_capability_validation_questions=(
                self._registered_capability_validation_questions
            ),
        )
        checkpoint["trace_vb_value_bridge_initialized"] = False
        checkpoint["trace_vb_stage2_objective"] = (
            "capability_anchored_role_local_latent_rl"
            if checkpoint_stage == 2
            else "capability_preserving_role_semantic_cot_sft"
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
        self.assert_v8_initialization_contract()
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
            self._value_bridge_initialized = False
            self._stage2_initialized = True
            self._activate_path_adapter()
            self._validate_trace_rl_epoch_budget()
        else:
            self._load_and_validate_sufficiency_cache()
        return super().on_fit_start()

    def on_validation_start(self):
        # ``trainer.validate`` does not invoke on_fit_start.  The step-0
        # candidate therefore enforces exactly the same fail-closed rewind and
        # provenance contract here before evaluating or saving a checkpoint.
        # Normal fit already owns its train JsonLogger; this helper is
        # idempotent and never replaces it.
        ensure_validation_json_logger(self)
        self.assert_v8_initialization_contract()
        return super().on_validation_start()

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
        cache = safe_load_checkpoint(path, map_location="cpu")
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

    @torch.no_grad()
    def _capability_teacher_trajectory(
        self,
        questions: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """Reproduce the frozen registered block-latent teacher in one LM.

        The teacher owns only a frozen PEFT adapter and the small bridge/query
        tensors from the verified 72.29% checkpoint.  It shares the base 4B
        model with the student and is used only to provide Stage-1 answer
        suffix targets; deployment never calls this branch.
        """
        if not self._capability_anchor_loaded:
            raise RuntimeError("capability teacher weights were not mapped")
        self._activate_capability_adapter()
        try:
            base_model = self.llm.get_base_model()
            backbone = getattr(base_model, "model", None)
            if backbone is None:
                raise RuntimeError("capability teacher requires a causal backbone")
            question_ids, question_mask = self.prepare_inputs(
                list(questions),
                padding_side="left",
                part="question",
                suffix=self.thinking_separator,
            )
            question_embeds = self.embedding(question_ids)
            question_outputs = backbone(
                inputs_embeds=question_embeds,
                attention_mask=question_mask,
                position_ids=self._trace_position_ids(
                    question_mask, question_embeds.shape[1]
                ),
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            previous_state = self.capability_state_norm(
                question_outputs.last_hidden_state[:, -1, :]
            )
            query_scale = float(
                self.trace_config.get("capability_query_scale", 0.10)
            )
            view_scale = float(
                self.trace_config.get("capability_trace_view_scale", 0.30)
            )
            step_view_scale = float(
                self.trace_config.get(
                    "capability_trace_step_view_scale", 0.15
                )
            )
            gate_scale = float(
                self.trace_config.get(
                    "capability_anchor_gate_scale", 0.50
                )
            )
            # Preserve the registered 72.2892% model's bf16 arithmetic order.
            # Combining these terms in fp32 is mathematically equivalent but can
            # move a greedy-generation boundary example, so parity validation
            # intentionally mirrors the historical implementation term by term.
            anchor_gate = torch.sigmoid(
                self.capability_anchor_gate_predictor(previous_state)
            )
            base_latent = self.capability_latent_bridge(previous_state)[:, None, :]
            latent_queries = self.capability_latent_queries[None, :, :].expand(
                len(questions), -1, -1
            ).to(base_latent.dtype)
            latent_inputs = base_latent + query_scale * latent_queries
            view_embeds = self.capability_trace_view[None, None, :].expand(
                len(questions), -1, -1
            ).to(latent_inputs.dtype)
            latent_inputs = latent_inputs + view_scale * view_embeds
            step_view_embeds = self.capability_trace_step_views[
                None, :, :
            ].expand(len(questions), -1, -1).to(latent_inputs.dtype)
            latent_inputs = latent_inputs + step_view_scale * step_view_embeds
            latent_inputs = latent_inputs.to(question_embeds.dtype)
            gate_values = anchor_gate[:, : self.n_trace_steps, None].to(
                latent_inputs.dtype
            )
            latent_inputs = latent_inputs * (1.0 + gate_scale * gate_values)
            latent_mask = torch.ones(
                len(questions),
                self.n_trace_steps,
                device=self.device,
                dtype=question_mask.dtype,
            )
            context_mask = torch.cat([question_mask, latent_mask], dim=1)
            latent_outputs = backbone(
                inputs_embeds=latent_inputs,
                attention_mask=context_mask,
                position_ids=self._trace_position_ids(
                    context_mask, self.n_trace_steps
                ),
                past_key_values=question_outputs.past_key_values,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            latent_states = self.capability_state_norm(
                latent_outputs.last_hidden_state
            )
            context_inputs_embeds = torch.cat(
                [question_embeds, latent_inputs], dim=1
            )
            return {
                "question_input_ids": question_ids,
                "question_inputs_embeds": question_embeds,
                "question_attention_mask": question_mask,
                "latent_inputs_embeds": latent_inputs,
                "latent_attention_mask": latent_mask,
                "context_inputs_embeds": context_inputs_embeds,
                "context_attention_mask": context_mask,
                "past_key_values": latent_outputs.past_key_values,
                "latent_states": latent_states,
            }
        finally:
            self._activate_path_adapter()

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
        if posterior_context is not None and tuple(
            posterior_context.shape
        ) != (batch_size, self.hidden_size):
            raise ValueError(
                "posterior_context must have shape "
                f"{(batch_size, self.hidden_size)}"
            )
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
            # Match the registered 72.29% block teacher exactly. Role actions
            # are residuals around this prompt/latent convention.
            suffix=self.thinking_separator,
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
        raw_question_state = question_outputs.last_hidden_state[:, -1, :]
        previous_state = self.state_norm(raw_question_state)
        capability_question_state = self.capability_state_norm(
            raw_question_state
        )
        capability_base = self.capability_latent_bridge(
            capability_question_state
        )
        capability_anchor_gate = torch.sigmoid(
            self.capability_anchor_gate_predictor(
                capability_question_state
            )
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
        view_scale = float(
            self.trace_config.get("capability_trace_view_scale", 0.30)
        )
        step_view_scale = float(
            self.trace_config.get(
                "capability_trace_step_view_scale", 0.15
            )
        )
        gate_scale = float(
            self.trace_config.get("capability_anchor_gate_scale", 0.50)
        )

        for step_index in range(self.n_trace_steps):
            # PPO must evaluate an action at the exact state from which that
            # action was sampled.  Storing this tensor avoids replaying Qwen
            # during the at-most-two cached-state role-local updates.
            pre_action_states.append(previous_state)
            # Lightning wraps training_step in bf16 autocast.  PPO ratios at
            # actor_lr=8e-7 require substantially more mantissa than bf16;
            # keep every Gaussian quantity and stored old log-prob in FP32.
            with torch.autocast(
                device_type=previous_state.device.type,
                enabled=False,
            ):
                if (
                    self.do_trace_rl
                    and self._stage2_initialized
                    and step_index == self.n_trace_steps - 1
                ):
                    # A trainable shared role trunk is useful for local credit,
                    # but COMMIT must remain the immutable Stage-1 conditional
                    # mean. It is never sampled and never receives actor credit.
                    with torch.no_grad():
                        prior_mean, prior_log_std = (
                            self.stage1_policy_reference
                            .distribution_parameters(
                                previous_state.float(), step_index
                            )
                        )
                else:
                    prior_mean, prior_log_std = (
                        self.trajectory_policy.distribution_parameters(
                            previous_state.float(),
                            step_index,
                        )
                    )
                if posterior_context is None:
                    mean, log_std = prior_mean, prior_log_std
                else:
                    mean, log_std = (
                        self.trajectory_posterior.distribution_parameters(
                            previous_state.float(),
                            posterior_context.float(),
                            step_index,
                            prior_mean,
                            prior_log_std,
                        )
                    )
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
            step_ids = torch.full(
                (batch_size,),
                fill_value=step_index,
                device=self.device,
                dtype=torch.long,
            )
            # At zero action this is byte-for-byte the registered block
            # teacher's latent-input arithmetic: cast each learned prior to
            # the running bf16 dtype and add it in historical order.
            step_prior = self.trajectory_policy.dynamics_step_embedding(
                step_ids
            ).to(capability_base.dtype)
            anchored_input = capability_base + step_scale * step_prior
            view_prior = self.capability_trace_view[None, :].to(
                anchored_input.dtype
            )
            anchored_input = anchored_input + view_scale * view_prior
            step_view_prior = self.capability_trace_step_views[
                step_index
            ][None, :].to(anchored_input.dtype)
            anchored_input = (
                anchored_input + step_view_scale * step_view_prior
            ).to(question_embeds.dtype)
            gate_value = capability_anchor_gate[
                :, step_index : step_index + 1
            ].to(anchored_input.dtype)
            anchored_input = anchored_input * (
                1.0 + gate_scale * gate_value
            )
            # Role actions are residuals around the exact registered
            # validation-time latent input, not part of its frozen gate.
            with torch.autocast(
                device_type=previous_state.device.type,
                enabled=False,
            ):
                action_residual = self.trajectory_policy.action_projector(
                    action.float()
                )
            current_input = (
                anchored_input.float()
                + action_scale * action_residual.float()
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
            # The last causal token has already read PLAN through REFINE.  V7
            # therefore uses its normalized state directly as deterministic
            # COMMIT; an extra ad-hoc residual would break exact anchor init.
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

    @torch.no_grad()
    def _capability_teacher_logits(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        target_texts: Sequence[str],
        *,
        include_hybrid_header: bool,
    ) -> torch.Tensor:
        """Return frozen all-role teacher logits for the student's target."""
        self._activate_capability_adapter()
        try:
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
            attention_mask = build_path_bottleneck_mask(
                trajectory_outputs["question_attention_mask"],
                trajectory_outputs["latent_attention_mask"],
                current_mask,
                include_question=True,
            )
            source_position_mask = torch.cat(
                [trajectory_outputs["context_attention_mask"], current_mask],
                dim=1,
            )
            return self.llm.forward(
                input_ids=current_ids,
                attention_mask=attention_mask,
                position_ids=self._trace_position_ids(
                    source_position_mask, current_ids.shape[1]
                ),
                past_key_values=self._fork_past_key_values(
                    trajectory_outputs["past_key_values"]
                ),
                output_hidden_states=False,
            ).logits.detach()
        finally:
            self._activate_path_adapter()

    @staticmethod
    def _masked_teacher_kl(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        temperature: float,
        token_chunk_size: int = 4,
    ) -> torch.Tensor:
        """Causal suffix KL with a bounded token-by-vocabulary workspace."""
        if student_logits.shape != teacher_logits.shape:
            raise ValueError("student and capability logits must match")
        if tuple(token_mask.shape) != tuple(student_logits.shape[:2]):
            raise ValueError("capability KL mask must match token positions")
        if not math.isfinite(float(temperature)) or temperature <= 0.0:
            raise ValueError("capability KL temperature must be positive")
        shifted_mask = token_mask[:, 1:].float()
        denominator = shifted_mask.sum().clamp_min(1.0)
        numerator = student_logits[:, :0, :].sum(dtype=torch.float32)
        scale = float(temperature)
        for row in range(int(student_logits.shape[0])):
            for start in range(0, int(shifted_mask.shape[1]), token_chunk_size):
                end = min(start + token_chunk_size, shifted_mask.shape[1])
                weights = shifted_mask[row, start:end]
                if not bool(weights.ne(0).any()):
                    numerator = numerator + (
                        student_logits[row, start:end, :].sum(
                            dtype=torch.float32
                        )
                        * 0.0
                    )
                    continue
                student_log_probs = F.log_softmax(
                    student_logits[row, start:end, :].float() / scale,
                    dim=-1,
                )
                teacher_probs = F.softmax(
                    teacher_logits[row, start:end, :].float() / scale,
                    dim=-1,
                )
                per_token = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                ).sum(dim=-1) * (scale * scale)
                numerator = numerator + (per_token * weights).sum()
        return numerator / denominator

    def _teacher_force_bottleneck(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
        target_texts: Sequence[str],
        *,
        include_hybrid_header: Optional[bool] = None,
        protected_suffixes: Optional[Sequence[str]] = None,
        capability_teacher_logits: Optional[torch.Tensor] = None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
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
        protected_loss_mask = None
        if protected_suffixes is not None:
            if len(protected_suffixes) != len(target_texts):
                raise ValueError(
                    "protected suffixes and target texts must align"
                )
            protected_target_mask = torch.zeros_like(target_mask)
            for row_index, protected_suffix in enumerate(protected_suffixes):
                suffix_ids = self.tokenizer.encode(
                    str(protected_suffix) + self.tokenizer.eos_token,
                    add_special_tokens=False,
                )
                target_length = int(target_mask[row_index].sum().item())
                suffix_length = len(suffix_ids)
                if suffix_length <= 0 or suffix_length > target_length:
                    raise ValueError(
                        "protected answer suffix does not fit its target"
                    )
                expected = target_ids.new_tensor(suffix_ids)
                observed = target_ids[
                    row_index,
                    target_length - suffix_length : target_length,
                ]
                if not torch.equal(observed, expected):
                    raise ValueError(
                        "protected answer suffix is not an exact target suffix"
                    )
                protected_target_mask[
                    row_index,
                    target_length - suffix_length : target_length,
                ] = 1
            protected_loss_mask = torch.cat(
                [torch.zeros_like(separator_mask), protected_target_mask],
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
        position_ids = self._trace_position_ids(
            source_position_mask,
            current_ids.shape[1],
        )

        def answer_logits(
            token_ids: torch.Tensor,
            full_attention_mask: torch.Tensor,
            token_position_ids: torch.Tensor,
        ) -> torch.Tensor:
            # This closure is recomputed during backward. Adapter state is
            # mutable, so select the student *inside* the closure rather than
            # trusting whichever frozen teacher ran most recently.
            self._activate_path_adapter()
            # Qwen updates DynamicCache objects in place. Fork inside this
            # function so checkpoint recomputation always starts from the
            # immutable trajectory prefix rather than an already extended
            # answer cache.
            return self.llm.forward(
                input_ids=token_ids,
                attention_mask=full_attention_mask,
                position_ids=token_position_ids,
                past_key_values=self._fork_past_key_values(
                    trajectory_outputs["past_key_values"]
                ),
                output_hidden_states=False,
            ).logits

        checkpoint_answers = bool(
            self.trace_config.get(
                "stage1_answer_activation_checkpoint", True
            )
        ) and self.training and torch.is_grad_enabled()
        if checkpoint_answers:
            logits = checkpoint(
                answer_logits,
                current_ids,
                attention_mask,
                position_ids,
                use_reentrant=False,
            )
        else:
            logits = answer_logits(
                current_ids,
                attention_mask,
                position_ids,
            )
        full_loss = self._masked_causal_ce(
            logits,
            current_ids,
            loss_mask,
        )
        if protected_loss_mask is None:
            if capability_teacher_logits is not None:
                raise ValueError(
                    "capability distillation requires a protected suffix"
                )
            return full_loss
        protected_loss = self._masked_causal_ce(
            logits,
            current_ids,
            protected_loss_mask,
        )
        if capability_teacher_logits is None:
            return full_loss, protected_loss
        capability_kl = self._masked_teacher_kl(
            logits,
            capability_teacher_logits,
            select_capability_kl_mask(
                loss_mask,
                protected_loss_mask,
                scope=str(
                    self.trace_config.get(
                        "stage1_capability_kl_scope", "full_target"
                    )
                ),
            ),
            temperature=float(
                self.trace_config.get(
                    "stage1_capability_temperature", 2.0
                )
            ),
            token_chunk_size=int(
                self.trace_config.get(
                    "stage1_capability_kl_chunk_size", 4
                )
            ),
        )
        return full_loss, protected_loss, capability_kl

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
        if include_hybrid_header is None:
            include_hybrid_header = bool(
                self.trace_config.get(
                    "deployment_compact_reasoning",
                    False,
                )
            )
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

    @torch.no_grad()
    def _generate_capability_answers_from_trajectory(
        self,
        trajectory_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Exact registered-teacher readout used only by parity validation.

        This deliberately uses the historical single full-inputs-embeds prefill.
        The deployed student continues to use the causal question+COMMIT path.
        """
        self._activate_capability_adapter()
        try:
            batch_size = trajectory_outputs["latent_states"].shape[0]
            prompt_ids, prompt_mask = self._prompt_ids_for_answer(
                batch_size,
                include_hybrid_header=True,
            )
            prompt_embeds = self.embedding(prompt_ids)
            all_inputs_embeds = torch.cat(
                [
                    trajectory_outputs["context_inputs_embeds"],
                    prompt_embeds,
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [trajectory_outputs["context_attention_mask"], prompt_mask],
                dim=1,
            )
            generation_config = dict(
                self.model_kwargs.hybrid_generation_config
            )
            generation_config["do_sample"] = False
            generated = self.llm.generate(
                inputs_embeds=all_inputs_embeds,
                attention_mask=attention_mask,
                **generation_config,
            )
            return generated
        finally:
            self._activate_path_adapter()

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

        # Forward-greedy packing retained verbose early equations and often
        # discarded the final answer-producing transition. Protect the final
        # observed CoT equation first, then admit earlier clauses while keeping
        # the emitted text in its original causal order.
        priority = (
            [len(candidates) - 1] + list(range(len(candidates) - 1))
            if candidates
            else []
        )
        kept_indices = []
        for candidate_index in priority:
            proposed = sorted(kept_indices + [candidate_index])
            candidate = "\n".join(
                [candidates[index] for index in proposed] + [answer_suffix]
            )
            if self._target_token_count(candidate) <= budget:
                kept_indices.append(candidate_index)
        kept = [candidates[index] for index in sorted(kept_indices)]
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
        commit_target = role_targets["summary"][:, None, :].expand(
            batch_size,
            path_count,
            -1,
        )
        commit_loss = masked_cosine_loss(
            latent_states[:, :, 7, :],
            commit_target,
            refine_mask,
        )
        return {
            "plan_forecast": plan_forecast_loss,
            "solve": solve_loss,
            "refine": refine_loss,
            "commit": commit_loss,
        }

    def _semantic_anchor_weight(self) -> float:
        """Cosine-decayed semantic regularizer after score calibration."""
        if not bool(self.trace_rl_config.get("use_semantic_anchor", True)):
            return 0.0
        initial = float(
            self.trace_rl_config.get("semantic_anchor_initial_weight", 0.02)
        )
        minimum = float(
            self.trace_rl_config.get("semantic_anchor_minimum_weight", 0.01)
        )
        decay_batches = int(
            self.trace_rl_config.get("semantic_anchor_decay_batches", 4096)
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
        calibration_batches = int(
            self.trace_rl_config.get("score_calibration_batches", 128)
        )
        batches_seen = int(self.vb_rollout_batches_seen.item())
        if batches_seen < calibration_batches:
            return 0.0
        progress = min(
            1.0,
            max(
                0.0,
                float(batches_seen - calibration_batches)
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
        """Return role-local process rewards and zero for COMMIT.

        Stage 2 discounts and standardizes these auditable scores within each
        question group before adding them (weight 0.15) to terminal evidence.
        PLAN uses its five-target forecast objective; SOLVE/REFINE use the
        same frozen-CoT semantic targets as Stage 1.
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
        """Emit the last two complete equations plus the protected answer."""
        max_equations = int(
            self.trace_config.get("compact_target_max_equations", 2)
        )
        if max_equations != 2:
            raise ValueError(
                "TRACE-VB-v8 requires compact_target_max_equations=2"
            )
        targets = []
        for cot, answer, spans in zip(gold_cots, answers, solve_spans):
            steps = cot["steps"]
            causal_equations = []
            for start, end in spans:
                if end <= start:
                    continue
                causal_equations.extend(
                    self._complete_compact_equations(steps[end - 1])
                )
            # Keep the most recent occurrence of a repeated equation, then
            # restore causal order among the two selected atomic equations.
            selected_reversed = []
            seen = set()
            for equation in reversed(causal_equations):
                if equation in seen:
                    continue
                seen.add(equation)
                selected_reversed.append(equation)
                if len(selected_reversed) == max_equations:
                    break
            selected = list(reversed(selected_reversed))
            answer_suffix = (
                self.thinking_separator
                + self.answer_template.format(answer)
            )
            if selected:
                target = (
                    self.anchor_header
                    + "\n"
                    + "\n".join(f"- {equation}" for equation in selected)
                    + "\n"
                    + answer_suffix
                )
            else:
                # No fabricated prose/no-op anchor: answer directly.
                target = answer_suffix
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

        One training-only path is conditioned on this sample's gold CoT; one
        deterministic question-only path exactly matches deployment.  The
        teacher path transfers role semantics while the deployment path learns
        the same compact-reasoning-plus-answer protocol used at validation.
        """
        if self.do_trace_rl:
            raise RuntimeError("Stage-1 forward is not used during latent RL")
        questions = list(batch["question"])
        answers = list(batch["answer"])
        gold_cots = self._decode_single_gold_cots(batch)
        explicit_features, cot_contexts = self._collect_single_cot_features(
            questions, gold_cots
        )
        role_targets = self._build_role_semantic_targets(explicit_features)
        sufficiency_targets, sufficiency_mask, solve_importance = (
            self._batch_sufficiency_targets(
                batch, role_targets["solve_spans"]
            )
        )

        teacher_paths = int(
            self.trace_config.get("stage1_posterior_samples", 1)
        )
        if teacher_paths != 1:
            raise ValueError(
                "TRACE-VB-v8 requires exactly one CoT teacher path"
            )
        # All Stage-1 activations stay on GPU. The former saved-tensor CPU
        # offload leaked roughly 30 MiB of host RSS per optimizer step in the
        # real four-rank workload; the matched GPU-resident run was flat.
        sampled_outputs = self._trajectory_latents(
            questions,
            deterministic=False,
            posterior_context=cot_contexts,
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
        # The posterior path is supervised at every latent role below.  Its
        # former answer forward is replaced by a frozen, verified all-role
        # capability teacher, keeping the Stage-1 memory budget at two answer
        # forwards while transferring the old strong readout into q+COMMIT.
        sampled_answer_loss = map_outputs["latent_states"].new_zeros(())
        compact_targets = self._build_role_compact_targets(
            gold_cots,
            answers,
            role_targets["solve_spans"],
        )
        compact_target_lengths = torch.tensor(
            [
                self._target_token_count(target)
                for target in compact_targets
            ],
            device=self.device,
            dtype=torch.float32,
        )
        protected_answer_suffixes = [
            self.thinking_separator + self.answer_template.format(answer)
            for answer in answers
        ]
        capability_outputs = self._capability_teacher_trajectory(questions)
        capability_logits = self._capability_teacher_logits(
            capability_outputs,
            compact_targets,
            include_hybrid_header=True,
        )
        (
            map_compact_loss,
            map_answer_suffix_loss,
            capability_full_target_kl,
        ) = (
            self._teacher_force_bottleneck(
                map_outputs,
                compact_targets,
                include_hybrid_header=True,
                protected_suffixes=protected_answer_suffixes,
                capability_teacher_logits=capability_logits,
            )
        )
        sampled_answer_weight = float(
            self.trace_config.get("stage1_sampled_answer_weight", 0.35)
        )
        map_compact_weight = float(
            self.trace_config.get("stage1_map_compact_weight", 1.0)
        )
        map_answer_suffix_weight = float(
            self.trace_config.get("stage1_map_answer_suffix_weight", 0.50)
        )
        capability_weight = float(
            self.trace_config.get("stage1_capability_kl_weight", 0.25)
        )
        answer_loss = (
            sampled_answer_weight * sampled_answer_loss
            + map_compact_weight * map_compact_loss
            + map_answer_suffix_weight * map_answer_suffix_loss
            + capability_weight * capability_full_target_kl
        )
        posterior_kl = self._masked_role_kl(
            sampled_outputs["action_means"],
            sampled_outputs["action_log_stds"],
            sampled_outputs["prior_action_means"],
            sampled_outputs["prior_action_log_stds"],
            sampled_outputs["stochastic_action_mask"],
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
        posterior_kl_weight = float(
            self.trace_config.get("stage1_posterior_kl_weight", 0.05)
        )
        plan_weight = float(
            self.trace_config.get("stage1_plan_forecast_weight", 0.10)
        )
        solve_weight = float(
            self.trace_config.get("stage1_solve_weight", 0.20)
        )
        solve_text_weight = float(
            self.trace_config.get("stage1_solve_text_weight", 0.05)
        )
        refine_weight = float(
            self.trace_config.get("stage1_refine_weight", 0.10)
        )
        commit_weight = float(
            self.trace_config.get("stage1_commit_weight", 0.10)
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
            + posterior_kl_weight * posterior_kl
            + plan_weight * semantic_losses["plan_forecast"]
            + solve_weight * semantic_losses["solve"]
            + solve_text_weight * solve_text["loss"]
            + refine_weight * semantic_losses["refine"]
            + commit_weight * semantic_losses["commit"]
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
            "trace_stage1_sampled_answer_loss": sampled_answer_loss,
            "trace_stage1_map_compact_loss": map_compact_loss,
            "trace_stage1_map_answer_suffix_loss": map_answer_suffix_loss,
            "trace_stage1_capability_full_target_kl": (
                capability_full_target_kl
            ),
            # Compatibility alias for older metric collectors.  The v8 schema
            # and canonical key above make clear that the support is now the
            # entire compact target, not only the answer suffix.
            "trace_stage1_capability_suffix_kl": (
                capability_full_target_kl
            ),
            "trace_stage1_posterior_kl": posterior_kl,
            "trace_stage1_compact_target_tokens": (
                compact_target_lengths.mean().detach()
            ),
            "trace_vb_plan_forecast_loss": semantic_losses[
                "plan_forecast"
            ],
            "trace_vb_solve_loss": semantic_losses["solve"],
            "trace_vb_refine_loss": semantic_losses["refine"],
            "trace_vb_commit_loss": semantic_losses["commit"],
            "trace_vb_solve_text_loss": solve_text["loss"],
            "trace_vb_solve_text_active_chunks": solve_text["active_chunks"],
            "trace_vb_solve_text_tokens_per_chunk": solve_text["tokens_per_chunk"],
            "trace_vb_solve_text_truncated_fraction": solve_text[
                "truncated_fraction"
            ],
            "trace_vb_answer_activation_checkpoint": total_loss.new_tensor(
                float(
                    bool(
                        self.trace_config.get(
                            "stage1_answer_activation_checkpoint", True
                        )
                    )
                )
            ),
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
            "trace_vb_deployment_compact_reasoning": total_loss.new_ones(()),
            "trace_vb_question_only_deployment_path": total_loss.new_ones(()),
            "trace_vb_training_only_cot_teacher_path": total_loss.new_ones(()),
            "trace_vb_question_commit_readout": total_loss.new_ones(()),
            "trace_vb_commit_causal_summary": total_loss.new_ones(()),
            "trace_vb_private_latent_answer_access": total_loss.new_zeros(()),
            "trace_vb_teacher_posterior_count": total_loss.new_ones(()),
            "trace_vb_frozen_capability_teacher": total_loss.new_ones(()),
            "trace_answer_question_access": total_loss.new_ones(()),
            "lambda_answer_eff": total_loss.new_tensor(answer_weight),
            "lambda_sampled_answer_eff": total_loss.new_tensor(
                sampled_answer_weight
            ),
            "lambda_map_compact_eff": total_loss.new_tensor(map_compact_weight),
            "lambda_map_answer_suffix_eff": total_loss.new_tensor(
                map_answer_suffix_weight
            ),
            "lambda_capability_kl_eff": total_loss.new_tensor(
                capability_weight
            ),
            "lambda_posterior_kl_eff": total_loss.new_tensor(
                posterior_kl_weight
            ),
            "lambda_plan_forecast_eff": total_loss.new_tensor(plan_weight),
            "lambda_solve_eff": total_loss.new_tensor(solve_weight),
            "lambda_refine_eff": total_loss.new_tensor(refine_weight),
            "lambda_commit_eff": total_loss.new_tensor(commit_weight),
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
        role_targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Collect group-8 paths and combine outcome with role-local credit."""
        forbidden_shaping = (
            "dense_outcome_weight",
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
        grouped_targets = {
            key: value.repeat_interleave(group_size, dim=0)
            for key, value in role_targets.items()
            if isinstance(value, torch.Tensor)
        }
        micro_batch = self._rollout_micro_batch_size()
        action_chunks = []
        action_log_prob_chunks = []
        pre_action_state_chunks = []
        action_mask_chunks = []
        greedy_accuracy_chunks = []
        greedy_length_chunks = []
        gold_score_chunks = []
        step_reward_chunks = []

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
            gold_score_chunks.append(
                self._gold_answer_scores(
                    sampled_path,
                    chunk_answers,
                ).detach()
            )
            chunk_targets = {
                key: value[start:end]
                for key, value in grouped_targets.items()
            }
            step_reward_chunks.append(
                self._role_step_rewards(sampled_path, chunk_targets).detach()
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
        gold_answer_scores = torch.cat(gold_score_chunks, dim=0).float()
        step_rewards = torch.cat(step_reward_chunks, dim=0).float()
        terminal_rewards = greedy_accuracy.detach()
        evidence = evidence_gated_group_advantages(
            terminal_rewards,
            gold_answer_scores,
            group_size=group_size,
            enable_likelihood_fallback=bool(
                self.vb_score_proxy_enabled.item()
            ),
            minimum_gold_score_gap=float(
                self.trace_rl_config.get(
                    "minimum_gold_score_gap",
                    2.0e-3,
                )
            ),
        )
        local_advantages = build_discounted_role_advantages(
            torch.zeros_like(terminal_rewards),
            step_rewards,
            group_size=group_size,
            gamma=float(
                self.trace_rl_config.get("step_reward_discount", 0.90)
            ),
            step_reward_weight=1.0,
            reward_mask=action_mask,
        )
        step_reward_weight = float(
            self.trace_rl_config.get("step_reward_weight", 0.15)
        )
        advantages = (
            evidence.path_advantages[:, None]
            + step_reward_weight * local_advantages
        ) * action_mask.float()
        role_advantage_dispersion = (
            advantages[:, :7].std(dim=1, unbiased=False).mean()
        )
        grouped_accuracy = greedy_accuracy.view(-1, group_size)
        positive_counts = grouped_accuracy.sum(dim=1)
        all_wrong_weights = evidence.all_wrong_group_mask.float()
        mean_all_wrong_spread = (
            evidence.gold_score_spread * all_wrong_weights
        ).sum() / all_wrong_weights.sum().clamp_min(1.0)
        proxy_auc = evidence.proxy_pair_credit / (
            evidence.proxy_pair_count.clamp_min(1.0)
        )
        self._last_trace_metrics = {
            "trace_vb/greedy_path_accuracy": greedy_accuracy.mean(),
            "trace_policy/positive_count": positive_counts.mean(),
            "trace_policy/all_correct_group_fraction": (
                evidence.all_correct_group_mask.float().mean()
            ),
            "trace_policy/mixed_group_fraction": (
                evidence.mixed_group_mask.float().mean()
            ),
            "trace_policy/all_wrong_group_fraction": (
                evidence.all_wrong_group_mask.float().mean()
            ),
            "trace_policy/likelihood_group_fraction": (
                evidence.likelihood_group_mask.float().mean()
            ),
            "trace_policy/informative_group_fraction": (
                evidence.path_informative_mask.float().mean()
            ),
            "trace_policy/oracle_pass_at_8": (
                (positive_counts > 0).float().mean()
            ),
            "trace_policy/gold_score_spread_all_wrong": mean_all_wrong_spread,
            "trace_policy/proxy_pair_auc_batch": proxy_auc,
            "trace_policy/proxy_pair_count_batch": evidence.proxy_pair_count,
            "trace_policy/likelihood_proxy_enabled": (
                terminal_rewards.new_tensor(
                    float(bool(self.vb_score_proxy_enabled.item()))
                )
            ),
            "trace_vb/group_reward_mean": evidence.policy_rewards.mean(),
            "trace_vb/group_advantage_std": evidence.path_advantages.std(
                unbiased=False
            ),
            "trace_vb/role_step_reward_mean": step_rewards[:, :7].mean(),
            "trace_vb/role_local_advantage_abs": (
                local_advantages[:, :7].abs().mean()
            ),
            "trace_vb/role_advantage_dispersion": role_advantage_dispersion,
            "trace_vb/step_reward_weight": terminal_rewards.new_tensor(
                step_reward_weight
            ),
            "trace_vb/exact_terminal_primary": terminal_rewards.new_ones(()),
            "trace_vb/critic_gae_actor_signal": terminal_rewards.new_zeros(()),
        }
        return {
            "group_questions": group_questions,
            "group_answers": group_answers,
            "actions": actions,
            "pre_action_states": pre_action_states,
            "action_mask": action_mask,
            "old_action_log_probs": old_action_log_probs,
            "terminal_rewards": terminal_rewards,
            "advantages": advantages.detach(),
            "step_rewards": step_rewards.detach(),
            "local_advantages": local_advantages.detach(),
            "policy_rewards": evidence.policy_rewards.detach(),
            "gold_answer_scores": gold_answer_scores.detach(),
            "path_informative_mask": (
                evidence.path_informative_mask.detach()
            ),
            "failure_entropy_mask": evidence.failure_entropy_mask.detach(),
            "proxy_pair_credit": evidence.proxy_pair_credit.detach(),
            "proxy_pair_count": evidence.proxy_pair_count.detach(),
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

    @torch.no_grad()
    def _stage1_policy_kl_on_rollout(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Measure the current actor against Stage 1 on cached rollout states."""
        states = rollout["pre_action_states"]
        action_mask = rollout["action_mask"].bool()
        total_items = int(states.shape[0])
        micro_batch = self._optimization_micro_batch_size()
        if total_items <= 0:
            raise RuntimeError("empty TRACE-VB KL measurement rollout")
        total = states.new_zeros((), dtype=torch.float32)
        for start in range(0, total_items, micro_batch):
            end = min(start + micro_batch, total_items)
            current_means, current_log_stds = (
                self._policy_parameters_on_stored_states(
                    self.trajectory_policy,
                    states[start:end],
                )
            )
            reference_means, reference_log_stds = (
                self._policy_parameters_on_stored_states(
                    self.stage1_policy_reference,
                    states[start:end],
                )
            )
            chunk_kl = self._masked_role_kl(
                current_means.float(),
                current_log_stds.float(),
                reference_means.float(),
                reference_log_stds.float(),
                action_mask[start:end],
            )
            total = total + (
                float(end - start) / float(total_items)
            ) * chunk_kl.float()
        return total

    def _trajectory_policy_update(
        self,
        rollout: Dict[str, torch.Tensor],
        *,
        outcome_active: bool,
    ) -> Dict[str, torch.Tensor]:
        """One critic-free group-relative PPO epoch over cached states."""
        states = rollout["pre_action_states"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_action_log_probs"]
        advantages = rollout["advantages"]
        action_mask = rollout["action_mask"].bool()
        failure_entropy_mask = rollout["failure_entropy_mask"].bool()
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
                "stage1_kl",
                "entropy",
                "ratio_deviation",
                "clip_fraction",
                "objective",
            )
        }
        for start in range(0, total_items, micro_batch):
            end = min(start + micro_batch, total_items)
            chunk_states = states[start:end]
            chunk_mask = action_mask[start:end]
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
                chunk_mask.float()
                * role_entropy_weights.view(1, -1)
                * failure_entropy_mask[start:end, None].float()
            )
            entropy = (
                per_role_entropy * entropy_mask
            ).sum() / entropy_mask.sum().clamp_min(1.0)
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
            outcome_scale = float(bool(outcome_active))
            objective = (
                outcome_scale * actor_loss
                + kl_weight * stage1_kl
                - outcome_scale * entropy_weight * entropy
            )
            chunk_weight = float(end - start) / float(total_items)
            self.manual_backward(chunk_weight * objective)
            measurements = {
                "actor_loss": actor_loss,
                "stage1_kl": stage1_kl,
                "entropy": entropy,
                "ratio_deviation": ratio_deviation,
                "clip_fraction": clip_fraction,
                "objective": objective,
            }
            for name, value in measurements.items():
                totals[name] = totals[name] + (
                    chunk_weight * value.detach().float()
                )
        totals["outcome_active"] = states.new_tensor(float(outcome_active))
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

    @staticmethod
    def _distributed_sum_scalar(value: torch.Tensor) -> torch.Tensor:
        result = value.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    @staticmethod
    def _distributed_mean_scalar(value: torch.Tensor) -> torch.Tensor:
        result = LitTRACEVB._distributed_sum_scalar(value)
        if dist.is_available() and dist.is_initialized():
            result = result / float(dist.get_world_size())
        return result

    def trace_rl_training_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=0,
    ):
        """Stage 2: calibrated evidence-gated group latent PPO."""
        questions = list(batch["question"])
        answers = list(batch["answer"])
        optimizer = self.optimizers()
        gold_cots = self._decode_single_gold_cots(batch)
        explicit_features, _ = self._collect_single_cot_features(
            questions, gold_cots
        )
        role_targets = self._build_role_semantic_targets(explicit_features)
        rollout = self.trace_policy_rollout(
            questions=questions,
            answers=answers,
            role_targets=role_targets,
        )

        # Calibrate the frozen likelihood proxy only on mixed groups where
        # exact correctness supplies an independent label. All ranks maintain
        # identical persistent counters, so resuming cannot silently change
        # whether the all-wrong fallback is enabled.
        pair_credit = self._distributed_sum_scalar(
            rollout["proxy_pair_credit"]
        )
        pair_count = self._distributed_sum_scalar(
            rollout["proxy_pair_count"]
        )
        self.vb_score_pair_credit.add_(pair_credit)
        self.vb_score_pair_count.add_(pair_count)
        calibration_batches = int(
            self.trace_rl_config.get("score_calibration_batches", 128)
        )
        batches_seen_before = int(self.vb_rollout_batches_seen.item())
        # Calibration gates only the all-wrong likelihood fallback. Exact
        # mixed-group evidence and observed CoT role rewards are valid from the
        # first rollout, so withholding all updates would waste signal.
        actor_active = True
        if (
            not bool(self.vb_score_proxy_decided.item())
            and batches_seen_before + 1 >= calibration_batches
        ):
            minimum_pairs = float(
                self.trace_rl_config.get("score_proxy_minimum_pairs", 64)
            )
            minimum_auc = float(
                self.trace_rl_config.get("score_proxy_minimum_auc", 0.60)
            )
            cumulative_auc = self.vb_score_pair_credit / (
                self.vb_score_pair_count.clamp_min(1.0)
            )
            enabled = (
                bool(
                    self.trace_rl_config.get(
                        "use_gold_likelihood_fallback",
                        True,
                    )
                )
                and float(self.vb_score_pair_count.item()) >= minimum_pairs
                and float(cumulative_auc.item()) >= minimum_auc
            )
            self.vb_score_proxy_enabled.fill_(int(enabled))
            self.vb_score_proxy_decided.fill_(1)

        update_epochs = int(
            self.trace_rl_config.get("policy_update_epochs", 2)
        )
        if not 1 <= update_epochs <= 2:
            raise RuntimeError("formal TRACE-VB-v8 permits at most two updates")
        semantic_anchor_weight = self._semantic_anchor_weight()
        target_kl = float(
            self.trace_rl_config.get("stage1_policy_target_kl", 0.01)
        )
        outcome_active = (
            actor_active and float(self.vb_last_stage1_kl.item()) < target_kl
        )
        update_metrics = []
        grad_norms = []
        optimizer_steps = 0
        zero = rollout["actions"].new_zeros(())

        if actor_active:
            for update_index in range(update_epochs):
                optimizer.zero_grad(set_to_none=True)
                metrics = self._trajectory_policy_update(
                    rollout,
                    outcome_active=outcome_active,
                )
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
                    outcome_active
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

                if optimizer_did_step:
                    post_step_kl = self._stage1_policy_kl_on_rollout(rollout)
                else:
                    post_step_kl = metrics["stage1_kl"]
                global_post_step_kl = self._distributed_mean_scalar(
                    post_step_kl
                )
                metrics["post_step_stage1_kl"] = (
                    global_post_step_kl.detach()
                )
                self.vb_last_stage1_kl.copy_(global_post_step_kl)
                if float(global_post_step_kl.item()) >= target_kl:
                    break
        else:
            # Calibration rollouts never mutate the actor or scheduler.
            update_metrics.append(
                {
                    "actor_loss": zero,
                    "stage1_kl": zero,
                    "post_step_stage1_kl": zero,
                    "entropy": zero,
                    "ratio_deviation": zero,
                    "clip_fraction": zero,
                    "objective": zero,
                    "outcome_active": zero,
                    "semantic_anchor": zero,
                    "semantic_anchor_plan": zero,
                    "semantic_anchor_solve_text": zero,
                    "semantic_anchor_refine": zero,
                    "semantic_anchor_weight": zero,
                    "semantic_anchor_applied": zero,
                    "semantic_anchor_truncated_fraction": zero,
                }
            )
            grad_norms.append(zero)

        self.vb_rollout_batches_seen.add_(1)
        reduced = {
            name: torch.stack([item[name] for item in update_metrics]).mean()
            for name in update_metrics[0]
        }
        grad_norm = torch.stack(grad_norms).mean()
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        actor_lr = float(raw_optimizer.param_groups[0]["lr"])
        cumulative_proxy_auc = self.vb_score_pair_credit / (
            self.vb_score_pair_count.clamp_min(1.0)
        )
        batches_seen_after = int(self.vb_rollout_batches_seen.item())
        logs = {
            "train/total_loss": reduced["objective"].detach(),
            "train/trajectory_policy_loss": reduced["actor_loss"].detach(),
            "train/stage1_policy_kl": reduced["stage1_kl"].detach(),
            "train/stage1_policy_kl_post_step": reduced[
                "post_step_stage1_kl"
            ].detach(),
            "train/stage1_policy_kl_global_last": (
                self.vb_last_stage1_kl.detach()
            ),
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
            "train/group_policy_reward": rollout[
                "policy_rewards"
            ].mean().detach(),
            "train/gold_answer_score": rollout[
                "gold_answer_scores"
            ].mean().detach(),
            "train/output_length": rollout[
                "greedy_lengths"
            ].mean().detach(),
            "train/n_latent_forward": torch.tensor(
                float(self.n_trace_steps),
                device=self.device,
            ),
            "train/grad_norm": grad_norm.detach(),
            "train/actor_lr": torch.tensor(actor_lr, device=self.device),
            "train/critic_lr": zero,
            "train/optimizer_did_step": torch.tensor(
                float(optimizer_steps) / float(max(1, len(update_metrics))),
                device=self.device,
            ),
            "train/policy_update_epochs": torch.tensor(
                float(update_epochs), device=self.device
            ),
            "train/effective_policy_update_epochs": torch.tensor(
                float(optimizer_steps), device=self.device
            ),
            "train/answer_decoder_frozen": torch.ones((), device=self.device),
            "train/head_only_ppo": zero,
            "train/role_local_policy_update": torch.ones(
                (), device=self.device
            ),
            "train/critic_free_group_rl": torch.ones((), device=self.device),
            "train/actor_active": torch.tensor(
                float(actor_active), device=self.device
            ),
            "train/outcome_update_active": torch.tensor(
                float(outcome_active), device=self.device
            ),
            "train/kl_budget_active": torch.tensor(
                float(float(self.vb_last_stage1_kl.item()) < target_kl),
                device=self.device,
            ),
            "train/score_calibration_remaining_batches": torch.tensor(
                float(max(0, calibration_batches - batches_seen_after)),
                device=self.device,
            ),
            "train/score_proxy_pair_count": self.vb_score_pair_count.detach(),
            "train/score_proxy_auc": cumulative_proxy_auc.detach(),
            "train/score_proxy_decided": self.vb_score_proxy_decided.float(),
            "train/score_proxy_enabled": self.vb_score_proxy_enabled.float(),
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
        validation_path = str(
            self.trace_config.get("validation_path", "student_commit")
        )
        if validation_path == "capability_teacher_all_roles":
            trajectory = self._capability_teacher_trajectory(questions)
            output_ids = self._generate_capability_answers_from_trajectory(
                trajectory
            )
            n_latent = torch.full(
                (len(questions), 1),
                fill_value=self.n_trace_steps,
                device=self.device,
                dtype=torch.long,
            )
            return output_ids, n_latent, trajectory
        if validation_path != "student_commit":
            raise ValueError(
                f"unknown TRACE-VB validation_path={validation_path}"
            )
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
        # Visualization exposes the trained, frozen Stage-1 sufficiency probe.
        # The legacy value head is never treated as v7 mechanism evidence.
        value_head = self.sufficiency_head
        value_head_type = "frozen_stage1_sufficiency"
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
        if "source_id" not in batch:
            raise RuntimeError(
                "formal TRACE-VB validation requires immutable source_id"
            )
        for index, source_id in zip(
            batch["idx"].tolist(),
            batch["source_id"].tolist(),
        ):
            prediction = self.sample_logs[index]["pred_answer"][-1]
            output_string = self.sample_logs[index]["output_string"][-1]
            self._validation_question_records.append(
                (
                    int(source_id),
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
                "schema_version": "trace_vb_v8_validation_behavior_v1",
                "validation_path": str(
                    self.trace_config.get(
                        "validation_path", "student_commit"
                    )
                ),
                "epoch_index": int(self.current_epoch),
                "global_step": int(self.global_step),
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
