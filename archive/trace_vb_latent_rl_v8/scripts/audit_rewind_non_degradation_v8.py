#!/usr/bin/env python3
"""Gate metric-safe v7 -> v8 initialization before any optimizer update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.models.trace_vb import (  # noqa: E402
    TRACE_VB_ZERO_ACTION_RESET_SCHEMA,
    TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES,
)
from src.utils.safe_checkpoint import safe_load_checkpoint  # noqa: E402


EXPECTED_QUESTIONS = 747
METRIC_SAFE_BASELINE_ORIGIN = Path(
    "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
    "20260818-213000_trace_vb_v7_full_seed0/student_initial_gate.json"
)
METRIC_SAFE_BASELINE_SHA256 = (
    "9b888b36956f70affae153526b716ad26343164b4aa66d6d3d54f70561144aa2"
)
METRIC_SAFE_BASELINE_CORRECT = 527
REGISTERED_CAPABILITY_VALIDATION_ORIGIN = Path(
    "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
    "20260818-213000_trace_vb_v7_full_seed0/capability_parity_gate.json"
)
REGISTERED_CAPABILITY_VALIDATION_SHA256 = (
    "59123d2bfc020335f54d70862a904e1b70e5ea1ce764483692809ae4e6772116"
)
REGISTERED_CAPABILITY_VALIDATION_CORRECT = 540
BEHAVIOR_THRESHOLDS = {
    "valid_answer_fraction": (0.98, "minimum"),
    "unique_prediction_ratio": (0.20, "minimum"),
    "top1_mode_fraction": (0.20, "maximum"),
    "nonempty_output_fraction": (0.98, "minimum"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def audit_validation_artifact(
    artifact: Path,
    *,
    origin: Path,
    expected_sha256: str,
    validation_path: str,
    correct_count: int,
) -> dict:
    """Validate both the canonical artifact and its byte-identical run copy."""
    if not origin.is_file():
        raise RuntimeError(f"missing canonical validation artifact: {origin}")
    if not artifact.is_file():
        raise RuntimeError(f"missing copied validation artifact: {artifact}")
    origin_sha = sha256(origin)
    artifact_sha = sha256(artifact)
    if origin_sha != expected_sha256 or artifact_sha != expected_sha256:
        raise RuntimeError(
            f"{validation_path} artifact SHA256 mismatch: "
            f"origin={origin_sha}, copy={artifact_sha}"
        )
    value = read_object(artifact)
    failures = []
    if value.get("schema_version") != "trace_vb_v7_validation_behavior_v1":
        failures.append("schema_version")
    if value.get("validation_path") != validation_path:
        failures.append("validation_path")
    if type(value.get("world_size")) is not int or value["world_size"] != 4:
        failures.append("world_size")
    questions = value.get("unique_questions")
    if (
        isinstance(questions, bool)
        or not isinstance(questions, (int, float))
        or not math.isfinite(float(questions))
        or float(questions) != float(EXPECTED_QUESTIONS)
    ):
        failures.append("unique_questions")
    if value.get("status") != "PASS":
        failures.append("status")
    if value.get("failures") != []:
        failures.append("failures")
    observed_correct = value.get("correct_count")
    if type(observed_correct) is not int or observed_correct != correct_count:
        failures.append("correct_count")
    accuracy = value.get("accuracy")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or not math.isclose(
            float(accuracy),
            correct_count / EXPECTED_QUESTIONS,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        failures.append("accuracy")
    behavior = {}
    for key, (threshold, direction) in BEHAVIOR_THRESHOLDS.items():
        observed = value.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not 0.0 <= float(observed) <= 1.0
        ):
            failures.append(key)
            continue
        observed = float(observed)
        behavior[key] = observed
        if direction == "minimum" and observed < threshold:
            failures.append(key)
        elif direction == "maximum" and observed > threshold:
            failures.append(key)
    if failures:
        raise RuntimeError(
            f"{validation_path} artifact contract failed: "
            + ", ".join(failures)
        )
    return {
        "origin": str(origin.resolve()),
        "artifact": str(artifact.resolve()),
        "sha256": expected_sha256,
        "schema_version": value["schema_version"],
        "validation_path": validation_path,
        "world_size": 4,
        "questions": EXPECTED_QUESTIONS,
        "correct_count": correct_count,
        "accuracy": float(accuracy),
        "behavior": behavior,
        "status": "PASS",
    }


def audit_step0_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = safe_load_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("step-0 checkpoint is not a dictionary")
    failures = []
    if checkpoint.get("trace_vb_schema_version") != "trace_vb_v8":
        failures.append("schema")
    if type(checkpoint.get("trace_policy_training_stage")) is not int or (
        checkpoint["trace_policy_training_stage"] != 1
    ):
        failures.append("stage")
    if type(checkpoint.get("global_step")) is not int or (
        checkpoint["global_step"] != 0
    ):
        failures.append("global_step")
    metadata = {
        "schema_version": checkpoint.get(
            "trace_vb_zero_action_reset_schema_version"
        ),
        "applied": checkpoint.get("trace_vb_zero_action_reset_applied"),
        "operation_count": checkpoint.get(
            "trace_vb_zero_action_reset_operation_count"
        ),
        "tensor_count": checkpoint.get(
            "trace_vb_zero_action_reset_tensor_count"
        ),
        "target_names": checkpoint.get(
            "trace_vb_zero_action_reset_target_names"
        ),
        "source_schema": checkpoint.get(
            "trace_vb_zero_action_reset_source_schema"
        ),
        "source_checkpoint_sha256": checkpoint.get(
            "trace_vb_zero_action_reset_source_checkpoint_sha256"
        ),
    }
    v7_source_sha = checkpoint.get("trace_vb_v7_source_checkpoint_sha256")
    if metadata != {
        "schema_version": TRACE_VB_ZERO_ACTION_RESET_SCHEMA,
        "applied": True,
        "operation_count": 1,
        "tensor_count": len(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES),
        "target_names": list(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES),
        "source_schema": "trace_vb_v7",
        "source_checkpoint_sha256": v7_source_sha,
    }:
        failures.append("zero_action_reset_metadata")
    if (
        not isinstance(v7_source_sha, str)
        or len(v7_source_sha) != 64
        or any(character not in "0123456789abcdef" for character in v7_source_sha)
    ):
        failures.append("v7_source_checkpoint_sha256")
    expected_metric_metadata = {
        "trace_vb_metric_safe_baseline_path": str(
            METRIC_SAFE_BASELINE_ORIGIN.resolve()
        ),
        "trace_vb_metric_safe_baseline_sha256": (
            METRIC_SAFE_BASELINE_SHA256
        ),
        "trace_vb_metric_safe_baseline_correct_count": (
            METRIC_SAFE_BASELINE_CORRECT
        ),
        "trace_vb_metric_safe_baseline_questions": EXPECTED_QUESTIONS,
        "trace_vb_registered_capability_validation_path": str(
            REGISTERED_CAPABILITY_VALIDATION_ORIGIN.resolve()
        ),
        "trace_vb_registered_capability_validation_sha256": (
            REGISTERED_CAPABILITY_VALIDATION_SHA256
        ),
        "trace_vb_registered_capability_validation_correct_count": (
            REGISTERED_CAPABILITY_VALIDATION_CORRECT
        ),
        "trace_vb_registered_capability_validation_questions": (
            EXPECTED_QUESTIONS
        ),
    }
    if any(
        checkpoint.get(key) != expected
        for key, expected in expected_metric_metadata.items()
    ):
        failures.append("metric_provenance_metadata")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        failures.append("state_dict")
        state = {}
    observed_mean_names = sorted(
        name
        for name in state
        if name.startswith("trajectory_policy.mean_heads.")
    )
    expected_mean_names = sorted(
        name
        for name in TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES
        if ".mean_heads." in name
    )
    if observed_mean_names != expected_mean_names:
        failures.append("mean_head_coverage")
    nonzero_or_invalid = []
    for name in TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES:
        value = state.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or not torch.isfinite(value).all().item()
            or torch.count_nonzero(value).item() != 0
        ):
            nonzero_or_invalid.append(name)
    if nonzero_or_invalid:
        failures.append(
            "nonzero_or_invalid_reset_tensors=" + ",".join(nonzero_or_invalid)
        )
    if failures:
        raise RuntimeError(
            "step-0 checkpoint reset contract failed: "
            + "; ".join(failures)
        )
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256(checkpoint_path),
        "global_step": 0,
        "training_stage": 1,
        "reset_schema_version": TRACE_VB_ZERO_ACTION_RESET_SCHEMA,
        "reset_operation_count": 1,
        "reset_tensor_count": len(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES),
        "reset_target_names": list(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES),
        "reset_tensors_exact_zero": True,
    }


def audit(args: argparse.Namespace) -> dict:
    trigger = read_object(args.trigger_contract)
    summary = read_object(args.validation_summary)
    failures = []
    try:
        student_baseline = audit_validation_artifact(
            args.metric_safe_baseline,
            origin=METRIC_SAFE_BASELINE_ORIGIN,
            expected_sha256=METRIC_SAFE_BASELINE_SHA256,
            validation_path="student_commit",
            correct_count=METRIC_SAFE_BASELINE_CORRECT,
        )
        capability_validation = audit_validation_artifact(
            args.registered_capability_validation,
            origin=REGISTERED_CAPABILITY_VALIDATION_ORIGIN,
            expected_sha256=REGISTERED_CAPABILITY_VALIDATION_SHA256,
            validation_path="capability_teacher_all_roles",
            correct_count=REGISTERED_CAPABILITY_VALIDATION_CORRECT,
        )
        step0_checkpoint = audit_step0_checkpoint(args.step0_checkpoint)
    except RuntimeError as error:
        raise SystemExit(f"metric-safe rewind gate failed: {error}") from error
    if (
        trigger.get("schema_version")
        != "trace_vb_v8_v7_trigger_contract_v1"
        or trigger.get("status") != "PASS"
    ):
        failures.append("v7 trigger contract is not a formal PASS")
    baseline_correct = trigger.get("best_correct_count")
    if (
        type(baseline_correct) is not int
        or not 0 <= baseline_correct <= EXPECTED_QUESTIONS
    ):
        failures.append("v7 baseline exact count is invalid")
        baseline_correct = -1
    required_correct = max(baseline_correct, METRIC_SAFE_BASELINE_CORRECT)
    if summary.get("schema_version") != args.validation_schema:
        failures.append("step-0 validation schema is wrong")
    if summary.get("validation_path") != "student_commit":
        failures.append("step-0 validation did not use question+COMMIT")
    if type(summary.get("world_size")) is not int or (
        summary["world_size"] != 4
    ):
        failures.append("step-0 validation was not aggregated from four ranks")
    unique_questions = summary.get("unique_questions")
    if (
        isinstance(unique_questions, bool)
        or not isinstance(unique_questions, (int, float))
        or not math.isfinite(float(unique_questions))
        or float(unique_questions) != float(EXPECTED_QUESTIONS)
    ):
        failures.append("step-0 validation is not the strict 747-question set")
    if type(summary.get("global_step")) is not int or (
        summary["global_step"] != 0
    ):
        failures.append("step-0 validation contains optimizer progress")
    if summary.get("status", "PASS") != "PASS":
        failures.append("step-0 validation status is not PASS")
    correct = summary.get("correct_count")
    accuracy = summary.get("accuracy")
    if (
        type(correct) is not int
        or not 0 <= correct <= EXPECTED_QUESTIONS
        or isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or not math.isclose(
            float(accuracy),
            correct / EXPECTED_QUESTIONS,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        failures.append("step-0 exact count and accuracy are inconsistent")
        correct = -1
    if correct < required_correct:
        failures.append(
            "metric-safe initialization degraded validation: "
            f"{correct}<{required_correct} correct"
        )
    behavior = {}
    for key, (threshold, direction) in BEHAVIOR_THRESHOLDS.items():
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            value = float("nan")
        else:
            value = float(value)
        behavior[key] = value
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            failures.append(f"invalid {key}")
        elif direction == "minimum" and value < threshold:
            failures.append(f"{key}={value}<{threshold}")
        elif direction == "maximum" and value > threshold:
            failures.append(f"{key}={value}>{threshold}")
    if failures:
        raise SystemExit(
            "metric-safe rewind gate failed: " + "; ".join(failures)
        )
    report = {
        "schema_version": "trace_vb_v8_metric_safe_initialization_v1",
        "status": "PASS",
        "v7_best_correct_count": baseline_correct,
        "metric_safe_baseline": student_baseline,
        "registered_capability_validation": capability_validation,
        "required_step0_correct_count": required_correct,
        "step0_correct_count": correct,
        "non_degradation_margin": correct - required_correct,
        "validation_questions": EXPECTED_QUESTIONS,
        "step0_checkpoint": step0_checkpoint,
        "behavior_thresholds": BEHAVIOR_THRESHOLDS,
        "behavior": behavior,
        "trigger_contract": str(args.trigger_contract.resolve()),
        "validation_summary": str(args.validation_summary.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger-contract", required=True, type=Path)
    parser.add_argument("--validation-summary", required=True, type=Path)
    parser.add_argument("--validation-schema", required=True)
    parser.add_argument("--step0-checkpoint", required=True, type=Path)
    parser.add_argument("--metric-safe-baseline", required=True, type=Path)
    parser.add_argument(
        "--registered-capability-validation", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
