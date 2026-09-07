#!/usr/bin/env python3
"""Shared safe checkpoint facts for v8 orchestration contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.utils.safe_checkpoint import safe_load_checkpoint  # noqa: E402


ZERO_ACTION_RESET_SCHEMA = "trace_vb_v8_zero_action_reset_v1"
ZERO_ACTION_RESET_TARGET_NAMES = [
    "trajectory_policy.mean_heads.plan.weight",
    "trajectory_policy.mean_heads.plan.bias",
    "trajectory_policy.mean_heads.solve.weight",
    "trajectory_policy.mean_heads.solve.bias",
    "trajectory_policy.mean_heads.check.weight",
    "trajectory_policy.mean_heads.check.bias",
    "trajectory_policy.mean_heads.commit.weight",
    "trajectory_policy.mean_heads.commit.bias",
    "trajectory_policy.action_projector.0.bias",
]
REGISTERED_CAPABILITY_CHECKPOINT_SHA256 = (
    "d27ef63b1d462aa94fbbecc636cec31985a3687e5d2d036de9cfd87b01ec1525"
)
REGISTERED_CAPABILITY_PAYLOAD_SHA256 = (
    "b1e9a973bbf4f2eeaa18b7d49df16c38db50cea87e998008ba18a46cdff9e049"
)
COT_ENCODER_CHECKPOINT_SHA256 = (
    "df90292c2da854852651a9e75b1e8221484c6dc58b8c81a56400897a8dec7f54"
)
METRIC_SAFE_BASELINE_PATH = str(
    Path(
        "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
        "20260818-213000_trace_vb_v7_full_seed0/student_initial_gate.json"
    ).resolve()
)
METRIC_SAFE_BASELINE_SHA256 = (
    "9b888b36956f70affae153526b716ad26343164b4aa66d6d3d54f70561144aa2"
)
REGISTERED_CAPABILITY_VALIDATION_PATH = str(
    Path(
        "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
        "20260818-213000_trace_vb_v7_full_seed0/capability_parity_gate.json"
    ).resolve()
)
REGISTERED_CAPABILITY_VALIDATION_SHA256 = (
    "59123d2bfc020335f54d70862a904e1b70e5ea1ce764483692809ae4e6772116"
)


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = safe_load_checkpoint(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint payload is not a dictionary")
    return checkpoint


def checkpoint_global_step(checkpoint: dict[str, Any]) -> int:
    if "global_step" not in checkpoint:
        raise RuntimeError("checkpoint has no authoritative top-level global_step")
    step = int(checkpoint["global_step"])
    if step < 0:
        raise RuntimeError(f"checkpoint has invalid global_step={step}")
    return step


def checkpoint_epoch(checkpoint: dict[str, Any]) -> int:
    if "epoch" not in checkpoint:
        raise RuntimeError("checkpoint has no authoritative top-level epoch")
    epoch = int(checkpoint["epoch"])
    if epoch < 0:
        raise RuntimeError(f"checkpoint has invalid epoch={epoch}")
    return epoch


def checkpoint_stage(checkpoint: dict[str, Any]) -> int:
    if checkpoint.get("trace_vb_schema_version") != "trace_vb_v8":
        raise RuntimeError("checkpoint schema is not trace_vb_v8")
    stage = int(checkpoint.get("trace_policy_training_stage", -1))
    if stage not in (1, 2):
        raise RuntimeError(f"checkpoint has invalid training stage={stage}")
    return stage


def _normalized_sha256(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"checkpoint has invalid {label}")
    return normalized


def checkpoint_provenance(
    checkpoint: dict[str, Any],
    *,
    expected_registered_checkpoint_sha256: str,
    expected_registered_payload_sha256: str,
    expected_v7_source_sha256: str,
    expected_cot_encoder_sha256: str,
    expected_rewind_tensors: int = 504,
) -> dict[str, Any]:
    """Validate immutable one-time-rewind provenance on every v8 artifact."""
    checkpoint_stage(checkpoint)
    if checkpoint.get("trace_vb_capability_spine_rewound") is not True:
        raise RuntimeError("checkpoint does not attest the one-time capability rewind")
    rewind_count = checkpoint.get("trace_vb_capability_spine_rewind_count")
    if type(rewind_count) is not int:
        raise RuntimeError("checkpoint rewind count is not a strict integer")
    if rewind_count != int(expected_rewind_tensors):
        raise RuntimeError(
            f"checkpoint rewind count={rewind_count}, expected "
            f"{expected_rewind_tensors}"
        )
    reset_contract = {
        "zero_action_reset_schema_version": checkpoint.get(
            "trace_vb_zero_action_reset_schema_version"
        ),
        "zero_action_reset_applied": checkpoint.get(
            "trace_vb_zero_action_reset_applied"
        ),
        "zero_action_reset_operation_count": checkpoint.get(
            "trace_vb_zero_action_reset_operation_count"
        ),
        "zero_action_reset_tensor_count": checkpoint.get(
            "trace_vb_zero_action_reset_tensor_count"
        ),
        "zero_action_reset_target_names": checkpoint.get(
            "trace_vb_zero_action_reset_target_names"
        ),
        "zero_action_reset_source_schema": checkpoint.get(
            "trace_vb_zero_action_reset_source_schema"
        ),
    }
    expected_reset_contract = {
        "zero_action_reset_schema_version": ZERO_ACTION_RESET_SCHEMA,
        "zero_action_reset_applied": True,
        "zero_action_reset_operation_count": 1,
        "zero_action_reset_tensor_count": 9,
        "zero_action_reset_target_names": ZERO_ACTION_RESET_TARGET_NAMES,
        "zero_action_reset_source_schema": "trace_vb_v7",
    }
    for key, expected_value in expected_reset_contract.items():
        observed = reset_contract[key]
        if key in {
            "zero_action_reset_operation_count",
            "zero_action_reset_tensor_count",
        } and type(observed) is not int:
            raise RuntimeError(f"checkpoint {key} is not a strict integer")
        if key == "zero_action_reset_applied" and observed is not True:
            raise RuntimeError("checkpoint zero-action reset is not applied")
        if observed != expected_value:
            raise RuntimeError(f"checkpoint has invalid {key}")
    facts = {
        "registered_capability_checkpoint_sha256": _normalized_sha256(
            checkpoint.get(
                "trace_vb_registered_capability_checkpoint_sha256"
            ),
            "registered capability checkpoint SHA256",
        ),
        "registered_capability_payload_sha256": _normalized_sha256(
            checkpoint.get("trace_vb_registered_capability_payload_sha256"),
            "registered capability payload SHA256",
        ),
        "v7_source_checkpoint_sha256": _normalized_sha256(
            checkpoint.get("trace_vb_v7_source_checkpoint_sha256"),
            "v7 source checkpoint SHA256",
        ),
        "cot_encoder_checkpoint_sha256": _normalized_sha256(
            checkpoint.get("trace_vb_cot_encoder_checkpoint_sha256"),
            "CoT encoder checkpoint SHA256",
        ),
        "rewind_tensor_count": rewind_count,
    }
    expected = {
        "registered_capability_checkpoint_sha256": _normalized_sha256(
            expected_registered_checkpoint_sha256,
            "expected registered capability checkpoint SHA256",
        ),
        "registered_capability_payload_sha256": _normalized_sha256(
            expected_registered_payload_sha256,
            "expected registered capability payload SHA256",
        ),
        "v7_source_checkpoint_sha256": _normalized_sha256(
            expected_v7_source_sha256,
            "expected v7 source checkpoint SHA256",
        ),
        "cot_encoder_checkpoint_sha256": _normalized_sha256(
            expected_cot_encoder_sha256,
            "expected CoT encoder checkpoint SHA256",
        ),
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if facts[key] != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "checkpoint provenance mismatch: " + ", ".join(mismatches)
        )
    reset_source_sha = _normalized_sha256(
        checkpoint.get(
            "trace_vb_zero_action_reset_source_checkpoint_sha256"
        ),
        "zero-action reset source checkpoint SHA256",
    )
    if reset_source_sha != facts["v7_source_checkpoint_sha256"]:
        raise RuntimeError(
            "zero-action reset source SHA256 differs from v7 provenance"
        )
    metric_contract = {
        "metric_safe_baseline_path": checkpoint.get(
            "trace_vb_metric_safe_baseline_path"
        ),
        "metric_safe_baseline_sha256": checkpoint.get(
            "trace_vb_metric_safe_baseline_sha256"
        ),
        "metric_safe_baseline_correct_count": checkpoint.get(
            "trace_vb_metric_safe_baseline_correct_count"
        ),
        "metric_safe_baseline_questions": checkpoint.get(
            "trace_vb_metric_safe_baseline_questions"
        ),
        "registered_capability_validation_path": checkpoint.get(
            "trace_vb_registered_capability_validation_path"
        ),
        "registered_capability_validation_sha256": checkpoint.get(
            "trace_vb_registered_capability_validation_sha256"
        ),
        "registered_capability_validation_correct_count": checkpoint.get(
            "trace_vb_registered_capability_validation_correct_count"
        ),
        "registered_capability_validation_questions": checkpoint.get(
            "trace_vb_registered_capability_validation_questions"
        ),
    }
    expected_metric_contract = {
        "metric_safe_baseline_path": METRIC_SAFE_BASELINE_PATH,
        "metric_safe_baseline_sha256": METRIC_SAFE_BASELINE_SHA256,
        "metric_safe_baseline_correct_count": 527,
        "metric_safe_baseline_questions": 747,
        "registered_capability_validation_path": (
            REGISTERED_CAPABILITY_VALIDATION_PATH
        ),
        "registered_capability_validation_sha256": (
            REGISTERED_CAPABILITY_VALIDATION_SHA256
        ),
        "registered_capability_validation_correct_count": 540,
        "registered_capability_validation_questions": 747,
    }
    for key, expected_value in expected_metric_contract.items():
        observed = metric_contract[key]
        if key.endswith(("correct_count", "questions")) and (
            type(observed) is not int
        ):
            raise RuntimeError(f"checkpoint {key} is not a strict integer")
        if observed != expected_value:
            raise RuntimeError(f"checkpoint has invalid {key}")
    return {
        **facts,
        **reset_contract,
        "zero_action_reset_source_checkpoint_sha256": reset_source_sha,
        **metric_contract,
    }


def rollout_batches_seen(checkpoint: dict[str, Any]) -> int:
    state = checkpoint.get("state_dict", {})
    if "vb_rollout_batches_seen" not in state:
        raise RuntimeError("checkpoint is missing vb_rollout_batches_seen")
    value = state["vb_rollout_batches_seen"]
    if hasattr(value, "item"):
        value = value.item()
    batches = int(value)
    if batches < 0:
        raise RuntimeError("checkpoint has a negative rollout counter")
    return batches


def require_full_state(checkpoint: dict[str, Any]) -> None:
    required = ("loops", "optimizer_states", "lr_schedulers", "global_step")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(f"resume checkpoint lacks full-state keys: {missing}")
    if not isinstance(checkpoint["loops"], dict) or not checkpoint["loops"]:
        raise RuntimeError("resume checkpoint has no fit-loop state")
    optimizers = checkpoint["optimizer_states"]
    schedulers = checkpoint["lr_schedulers"]
    if not isinstance(optimizers, list) or len(optimizers) != 1:
        raise RuntimeError("resume checkpoint must contain exactly one optimizer state")
    if not isinstance(schedulers, list) or len(schedulers) != 1:
        raise RuntimeError("resume checkpoint must contain exactly one scheduler state")


def validate_stage2_policy_reference(
    stage2_checkpoint: dict[str, Any],
    stage1_checkpoint: dict[str, Any],
) -> int:
    """Bind a Stage-2 artifact to the exact selected Stage-1 policy tensors."""
    if checkpoint_stage(stage2_checkpoint) != 2:
        raise RuntimeError("policy-reference binding requires a Stage-2 checkpoint")
    if checkpoint_stage(stage1_checkpoint) != 1:
        raise RuntimeError("policy-reference binding requires a Stage-1 source")
    reference = stage2_checkpoint.get("trace_stage1_policy_reference")
    if not isinstance(reference, dict) or not reference:
        raise RuntimeError("Stage-2 checkpoint has no immutable Stage-1 policy reference")
    prefix = "trajectory_policy."
    source = {
        name[len(prefix) :]: value
        for name, value in stage1_checkpoint.get("state_dict", {}).items()
        if name.startswith(prefix)
    }
    if not source:
        raise RuntimeError("Stage-1 checkpoint has no trajectory-policy tensors")
    if set(reference) != set(source):
        missing = sorted(set(source) - set(reference))
        unexpected = sorted(set(reference) - set(source))
        raise RuntimeError(
            "Stage-2 policy-reference key mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    mismatched = []
    for name, source_value in source.items():
        reference_value = reference[name]
        if (
            not isinstance(source_value, torch.Tensor)
            or not isinstance(reference_value, torch.Tensor)
            or source_value.shape != reference_value.shape
            or source_value.dtype != reference_value.dtype
            or not torch.equal(source_value, reference_value)
        ):
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(
            "Stage-2 immutable policy reference differs from the selected "
            "Stage-1 checkpoint: " + ", ".join(mismatched[:8])
        )
    return len(source)
