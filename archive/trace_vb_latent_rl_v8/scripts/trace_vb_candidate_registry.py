#!/usr/bin/env python3
"""Fail-closed exact-count registry for TRACE-VB-v8 validation candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trace_vb_checkpoint_contract import (  # noqa: E402
    checkpoint_epoch,
    checkpoint_global_step,
    checkpoint_provenance,
    checkpoint_stage,
    load_checkpoint,
    rollout_batches_seen,
    validate_stage2_policy_reference,
)


BEHAVIOR_KEYS = (
    "valid_answer_fraction",
    "unique_prediction_ratio",
    "top1_mode_fraction",
    "nonempty_output_fraction",
)
BEHAVIOR_THRESHOLDS = {
    "valid_answer_fraction": (0.98, "minimum"),
    "unique_prediction_ratio": (0.20, "minimum"),
    "top1_mode_fraction": (0.20, "maximum"),
    "nonempty_output_fraction": (0.98, "minimum"),
}
COMPLETION_KEYS = (
    "candidate_index",
    "best_checkpoint",
    "last_checkpoint",
    "finished_at",
)


def _validated_physical_gpus(value: str) -> str:
    identifier = r"(?:0|[1-9][0-9]*)"
    if re.fullmatch(rf"{identifier}(?:,{identifier}){{3}}", value) is None:
        raise SystemExit("physical GPU contract requires exactly four numeric IDs")
    identifiers = value.split(",")
    if len(set(identifiers)) != 4:
        raise SystemExit("physical GPU contract contains duplicate IDs")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected an object in {path}")
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _checkpoint_facts(
    *,
    checkpoint_path: Path,
    phase: str,
    phase_input: bool,
    source_global_step: int,
    rollout_batches: int | None,
    registered_checkpoint_sha256: str,
    registered_payload_sha256: str,
    v7_source_sha256: str,
    cot_encoder_sha256: str,
    stage1_reference_checkpoint: Path | None,
) -> dict:
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        actual_stage = checkpoint_stage(checkpoint)
        expected_stage = 1 if phase == "stage1" or phase_input else 2
        if actual_stage != expected_stage:
            raise RuntimeError(
                f"checkpoint stage={actual_stage}, expected {expected_stage}"
            )
        actual_global_step = checkpoint_global_step(checkpoint)
        if actual_global_step != source_global_step:
            raise RuntimeError(
                f"checkpoint global_step={actual_global_step}, summary "
                f"global_step={source_global_step}"
            )
        actual_rollouts = rollout_batches_seen(checkpoint)
        expected_rollouts = 0
        if phase == "stage2" and not phase_input:
            if rollout_batches is None or int(rollout_batches) <= 0:
                raise RuntimeError("trained Stage-2 candidate has no rollout coordinate")
            expected_rollouts = int(rollout_batches)
        if actual_rollouts != expected_rollouts:
            raise RuntimeError(
                f"checkpoint rollout counter={actual_rollouts}, expected "
                f"{expected_rollouts}"
            )
        provenance = checkpoint_provenance(
            checkpoint,
            expected_registered_checkpoint_sha256=(
                registered_checkpoint_sha256
            ),
            expected_registered_payload_sha256=registered_payload_sha256,
            expected_v7_source_sha256=v7_source_sha256,
            expected_cot_encoder_sha256=cot_encoder_sha256,
        )
        reference_tensors = 0
        if phase == "stage2":
            if stage1_reference_checkpoint is None:
                raise RuntimeError(
                    "Stage-2 candidate audit requires the selected Stage-1 checkpoint"
                )
            stage1_path = stage1_reference_checkpoint.resolve()
            if not stage1_path.is_file():
                raise RuntimeError("selected Stage-1 reference checkpoint is missing")
            stage1_checkpoint = load_checkpoint(stage1_path)
            if checkpoint_stage(stage1_checkpoint) != 1:
                raise RuntimeError("selected Stage-1 reference has the wrong stage")
            checkpoint_provenance(
                stage1_checkpoint,
                expected_registered_checkpoint_sha256=(
                    registered_checkpoint_sha256
                ),
                expected_registered_payload_sha256=registered_payload_sha256,
                expected_v7_source_sha256=v7_source_sha256,
                expected_cot_encoder_sha256=cot_encoder_sha256,
            )
            if phase_input:
                if checkpoint_path.resolve() != stage1_path:
                    raise RuntimeError(
                        "Stage-2 phase input is not the selected Stage-1 checkpoint"
                    )
            else:
                reference_tensors = validate_stage2_policy_reference(
                    checkpoint, stage1_checkpoint
                )
        return {
            "checkpoint_stage": actual_stage,
            "checkpoint_global_step": actual_global_step,
            "checkpoint_epoch": checkpoint_epoch(checkpoint),
            "checkpoint_rollout_batches_seen": actual_rollouts,
            "stage1_policy_reference_tensors": reference_tensors,
            **provenance,
        }
    except (RuntimeError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"candidate checkpoint audit failed: {error}") from error


def register(args: argparse.Namespace) -> None:
    physical_gpus = _validated_physical_gpus(args.physical_gpus)
    summary_path = args.summary.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not summary_path.is_file():
        raise SystemExit(f"missing validation summary: {summary_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"missing candidate checkpoint: {checkpoint_path}")
    summary = _read_json(summary_path)
    failures: list[str] = []
    if summary.get("schema_version") != args.validation_schema:
        failures.append("wrong validation schema")
    if summary.get("validation_path") != "student_commit":
        failures.append("validation did not use question+COMMIT")
    if int(summary.get("world_size", -1)) != args.world_size:
        failures.append("wrong validation world size")
    if int(summary.get("unique_questions", -1)) != args.questions:
        failures.append("validation did not deduplicate to the registered set")
    correct = int(summary.get("correct_count", -1))
    accuracy = float(summary.get("accuracy", float("nan")))
    source_global_step = int(summary.get("global_step", -1))
    if source_global_step < 0:
        failures.append("validation summary has no valid global_step")
    if args.phase_input:
        if args.phase != "stage2" or args.step != 0:
            failures.append("phase input remapping is only valid for Stage-2 step 0")
    elif source_global_step != args.step:
        failures.append(
            f"summary global_step={source_global_step} does not equal "
            f"selection step={args.step}"
        )
    if not 0 <= correct <= args.questions:
        failures.append("invalid exact correct count")
    if not math.isfinite(accuracy) or not math.isclose(
        accuracy, correct / args.questions, rel_tol=0.0, abs_tol=1e-12
    ):
        failures.append("accuracy is not the exact correct-count ratio")
    for key in BEHAVIOR_KEYS:
        value = float(summary.get(key, float("nan")))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            failures.append(f"invalid {key}")
    if failures:
        raise SystemExit("candidate registration failed: " + "; ".join(failures))
    checkpoint_facts = _checkpoint_facts(
        checkpoint_path=checkpoint_path,
        phase=args.phase,
        phase_input=bool(args.phase_input),
        source_global_step=source_global_step,
        rollout_batches=args.rollout_batches,
        registered_checkpoint_sha256=args.registered_checkpoint_sha256,
        registered_payload_sha256=args.registered_payload_sha256,
        v7_source_sha256=args.v7_source_sha256,
        cot_encoder_sha256=args.cot_encoder_sha256,
        stage1_reference_checkpoint=args.stage1_reference_checkpoint,
    )
    record = {
        "schema_version": "trace_vb_v8_candidate_v1",
        "phase": args.phase,
        "physical_gpus": physical_gpus,
        "step": args.step,
        "selection_step": args.step,
        "source_global_step": source_global_step,
        "phase_input": bool(args.phase_input),
        "rollout_batches": args.rollout_batches,
        "correct_count": correct,
        "accuracy": accuracy,
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "behavior": {key: float(summary[key]) for key in BEHAVIOR_KEYS},
        **checkpoint_facts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def build_selection(args: argparse.Namespace) -> dict:
    """Replay every candidate and deterministically rebuild its index."""

    physical_gpus = _validated_physical_gpus(args.physical_gpus)
    records = [_read_json(path.resolve()) for path in args.candidates]
    expected_steps = None
    if args.expected_steps is not None:
        expected_steps = [int(value) for value in args.expected_steps.split(",")]
    expected_rollouts = None
    if args.expected_rollout_batches is not None:
        expected_rollouts = [
            int(value) for value in args.expected_rollout_batches.split(",")
        ]
        if expected_steps is not None and len(expected_rollouts) != len(expected_steps):
            raise SystemExit("expected rollout and selection coordinates differ in length")
    expected_count = len(expected_steps or expected_rollouts or [])
    if expected_count <= 0:
        raise SystemExit("selection requires expected steps or rollout coordinates")
    if len(records) != expected_count:
        raise SystemExit(
            f"expected {expected_count} candidates, found {len(records)}"
        )
    by_step: dict[int, dict] = {}
    for record in records:
        if record.get("schema_version") != "trace_vb_v8_candidate_v1":
            raise SystemExit("candidate has the wrong schema")
        if record.get("phase") != args.phase:
            raise SystemExit("candidate belongs to the wrong phase")
        if record.get("physical_gpus") != physical_gpus:
            raise SystemExit("candidate belongs to a different physical GPU set")
        step_value = record.get("step")
        selection_step_value = record.get("selection_step")
        source_step_value = record.get("source_global_step")
        if any(
            type(value) is not int
            for value in (step_value, selection_step_value, source_step_value)
        ):
            raise SystemExit("candidate coordinates are not strict integers")
        step = int(step_value)
        if selection_step_value != step:
            raise SystemExit("candidate selection_step alias is inconsistent")
        source_global_step = int(source_step_value)
        rollout_value = record.get("rollout_batches")
        if args.phase == "stage1":
            if rollout_value is not None:
                raise SystemExit("Stage-1 candidate unexpectedly has a rollout coordinate")
        elif type(rollout_value) is not int or rollout_value < 0:
            raise SystemExit("Stage-2 rollout coordinate is not a non-negative integer")
        phase_input_value = record.get("phase_input")
        if type(phase_input_value) is not bool:
            raise SystemExit("candidate phase_input is not a strict boolean")
        phase_input = phase_input_value
        if phase_input:
            if args.phase != "stage2" or step != 0 or source_global_step < 0:
                raise SystemExit("invalid Stage-2 input candidate coordinate")
        elif source_global_step != step:
            raise SystemExit(f"candidate global-step continuity failed at step {step}")
        if step in by_step:
            raise SystemExit(f"duplicate candidate step: {step}")
        summary = Path(str(record.get("summary", "")))
        checkpoint = Path(str(record.get("checkpoint", "")))
        if not summary.is_absolute() or not checkpoint.is_absolute():
            raise SystemExit(f"candidate paths are not absolute at step {step}")
        if not summary.is_file() or _sha256(summary) != record.get("summary_sha256"):
            raise SystemExit(f"summary continuity failed at step {step}")
        summary_value = _read_json(summary)
        if summary_value.get("schema_version") != "trace_vb_v8_validation_behavior_v1":
            raise SystemExit(f"summary schema continuity failed at step {step}")
        if summary_value.get("validation_path") != "student_commit":
            raise SystemExit(f"summary validation path failed at step {step}")
        if type(summary_value.get("world_size")) is not int or (
            summary_value["world_size"] != 4
        ):
            raise SystemExit(f"summary world-size continuity failed at step {step}")
        summary_questions = summary_value.get("unique_questions")
        # Validation metrics are reduced through Lightning and therefore the
        # exact question count is serialized as 747.0 in formal summaries.
        # Registration already accepts that representation.  Replay must use
        # the same exact-value contract while still rejecting booleans,
        # non-numeric values, non-finite values, and non-integral counts.
        if (
            isinstance(summary_questions, bool)
            or not isinstance(summary_questions, (int, float))
            or not math.isfinite(float(summary_questions))
            or float(summary_questions) != 747.0
        ):
            raise SystemExit(f"summary question-count continuity failed at step {step}")
        if type(summary_value.get("global_step")) is not int or (
            summary_value["global_step"] != source_global_step
        ):
            raise SystemExit(f"summary global-step continuity failed at step {step}")
        summary_correct = summary_value.get("correct_count")
        record_correct = record.get("correct_count")
        if (
            type(summary_correct) is not int
            or type(record_correct) is not int
            or not 0 <= summary_correct <= 747
            or record_correct != summary_correct
        ):
            raise SystemExit(f"summary correct-count continuity failed at step {step}")
        summary_accuracy = summary_value.get("accuracy")
        record_accuracy = record.get("accuracy")
        if (
            isinstance(summary_accuracy, bool)
            or not isinstance(summary_accuracy, (int, float))
            or isinstance(record_accuracy, bool)
            or not isinstance(record_accuracy, (int, float))
            or not math.isfinite(float(summary_accuracy))
            or not math.isfinite(float(record_accuracy))
            or not math.isclose(
                float(summary_accuracy),
                summary_correct / 747,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(record_accuracy),
                float(summary_accuracy),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise SystemExit(f"summary accuracy continuity failed at step {step}")
        behavior = record.get("behavior")
        if not isinstance(behavior, dict) or set(behavior) != set(BEHAVIOR_KEYS):
            raise SystemExit(f"candidate behavior keys changed at step {step}")
        for key in BEHAVIOR_KEYS:
            summary_behavior = summary_value.get(key)
            record_behavior = behavior.get(key)
            if (
                isinstance(summary_behavior, bool)
                or not isinstance(summary_behavior, (int, float))
                or isinstance(record_behavior, bool)
                or not isinstance(record_behavior, (int, float))
                or not math.isfinite(float(summary_behavior))
                or not 0.0 <= float(summary_behavior) <= 1.0
                or float(summary_behavior) != float(record_behavior)
            ):
                raise SystemExit(f"summary behavior continuity failed for {key}")
        if (
            not checkpoint.is_file()
            or _sha256(checkpoint) != record.get("checkpoint_sha256")
        ):
            raise SystemExit(f"checkpoint continuity failed at step {step}")
        facts = _checkpoint_facts(
            checkpoint_path=checkpoint,
            phase=args.phase,
            phase_input=phase_input,
            source_global_step=source_global_step,
            rollout_batches=record.get("rollout_batches"),
            registered_checkpoint_sha256=args.registered_checkpoint_sha256,
            registered_payload_sha256=args.registered_payload_sha256,
            v7_source_sha256=args.v7_source_sha256,
            cot_encoder_sha256=args.cot_encoder_sha256,
            stage1_reference_checkpoint=args.stage1_reference_checkpoint,
        )
        for key, value in facts.items():
            observed = record.get(key)
            if (
                (type(value) in (int, bool) and type(observed) is not type(value))
                or observed != value
            ):
                raise SystemExit(
                    f"candidate checkpoint fact {key} changed at step {step}"
                )
        by_step[step] = record
    if expected_steps is not None and sorted(by_step) != sorted(expected_steps):
        raise SystemExit(
            f"candidate steps are {sorted(by_step)}, expected {sorted(expected_steps)}"
        )
    if expected_rollouts is not None:
        by_rollout = {}
        for record in records:
            rollout = int(record.get("rollout_batches", -1))
            if rollout in by_rollout:
                raise SystemExit(f"duplicate rollout coordinate: {rollout}")
            by_rollout[rollout] = record
        actual_rollouts = sorted(by_rollout)
        if actual_rollouts != expected_rollouts:
            raise SystemExit(
                f"rollout coordinates are {actual_rollouts}, "
                f"expected {expected_rollouts}"
            )
        ordered_by_progress = [by_rollout[value] for value in expected_rollouts]
        if not bool(ordered_by_progress[0].get("phase_input", False)):
            raise SystemExit("Stage-2 rollout coordinate zero must be a phase input")
        trained_source_steps = [
            int(record["source_global_step"])
            for record in ordered_by_progress[1:]
        ]
        if any(
            current <= previous
            for previous, current in zip(
                trained_source_steps, trained_source_steps[1:]
            )
        ):
            raise SystemExit(
                "trained Stage-2 global steps are not strictly increasing: "
                f"{trained_source_steps}"
            )
    else:
        ordered_by_progress = [by_step[value] for value in expected_steps]
    eligible = []
    for record in records:
        behavior_failures = []
        for key, (threshold, direction) in BEHAVIOR_THRESHOLDS.items():
            value = float(record["behavior"][key])
            if direction == "minimum" and value < threshold:
                behavior_failures.append(f"{key}={value}<{threshold}")
            if direction == "maximum" and value > threshold:
                behavior_failures.append(f"{key}={value}>{threshold}")
        record["selection_eligible"] = not behavior_failures
        record["behavior_failures"] = behavior_failures
        if not behavior_failures:
            eligible.append(record)
    if not eligible:
        raise SystemExit("no candidate passes the deployment behavior gate")
    # Exact integer correctness is authoritative among safe candidates; an
    # exact tie selects the earlier step, making the input candidate a real
    # non-degradation guard.
    selected = max(
        eligible,
        key=lambda item: (
            int(item["correct_count"]),
            -int(
                item["rollout_batches"]
                if item.get("rollout_batches") is not None
                else item["step"]
            ),
        ),
    )
    report = {
        "schema_version": f"trace_vb_v8_{args.phase}_candidate_index_v1",
        "phase": args.phase,
        "physical_gpus": physical_gpus,
        "selection": (
            "maximum_exact_correct_count_then_earliest_progress_coordinate"
        ),
        "expected_steps": expected_steps,
        "observed_selection_steps": [
            int(item["selection_step"]) for item in ordered_by_progress
        ],
        "expected_rollout_batches": expected_rollouts,
        "candidate_count": len(records),
        "eligible_candidate_count": len(eligible),
        "behavior_thresholds": BEHAVIOR_THRESHOLDS,
        "selected_step": int(selected["step"]),
        "selected_source_global_step": int(selected["source_global_step"]),
        "selected_rollout_batches": selected.get("rollout_batches"),
        "selected_phase_input": bool(selected.get("phase_input", False)),
        "selected_correct_count": int(selected["correct_count"]),
        "selected_accuracy": float(selected["accuracy"]),
        "selected_checkpoint": selected["checkpoint"],
        "candidates": ordered_by_progress,
    }
    return report


def select(args: argparse.Namespace) -> None:
    report = build_selection(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.best_checkpoint_record.write_text(
        str(report["selected_checkpoint"]) + "\n", encoding="utf-8"
    )


def _read_exact_pointer(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0]:
        raise SystemExit(f"{label} must contain exactly one path")
    target = Path(lines[0]).resolve()
    if not target.is_file():
        raise SystemExit(f"{label} target is missing: {target}")
    return target


def _read_manifest(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise SystemExit(f"missing completed Stage manifest: {path}")
    values: dict[str, list[str]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line or "=" not in line:
            raise SystemExit(f"malformed Stage manifest line {line_number}")
        key, value = line.split("=", 1)
        if not key or not value:
            raise SystemExit(f"empty Stage manifest field at line {line_number}")
        values.setdefault(key, []).append(value)
    return values


def verify_completed(args: argparse.Namespace) -> dict:
    """Replay and verify an already-complete Stage without rewriting it."""

    report = build_selection(args)
    index_path = args.index.resolve()
    if not index_path.is_file():
        raise SystemExit("completed Stage candidate index is missing")
    index = _read_json(index_path)
    if index != json.loads(json.dumps(report)):
        raise SystemExit("completed Stage candidate index differs from exact replay")
    selected = _read_exact_pointer(
        args.best_checkpoint_record.resolve(), "completed Stage best record"
    )
    if selected != Path(str(report["selected_checkpoint"])).resolve():
        raise SystemExit("completed Stage best record differs from exact replay")
    last = _read_exact_pointer(
        args.last_checkpoint_record.resolve(), "completed Stage last record"
    )
    final_progress = Path(str(report["candidates"][-1]["checkpoint"])).resolve()
    if last != final_progress:
        raise SystemExit("completed Stage last record is not the final boundary")
    expected_last = args.expected_last_checkpoint.resolve()
    if not expected_last.is_file() or expected_last != final_progress:
        raise SystemExit("resume checkpoint is not the completed Stage boundary")
    manifest = _read_manifest(args.manifest.resolve())
    expected_fields = {
        "candidate_index": str(index_path),
        "best_checkpoint": str(selected),
        "last_checkpoint": str(last),
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != [expected]:
            raise SystemExit(
                f"completed Stage manifest {key} is not exactly {expected}"
            )
    finished = manifest.get("finished_at", [])
    if len(finished) != 1 or not finished[0]:
        raise SystemExit("completed Stage manifest must declare finished_at once")
    return report


def finalize_completed(args: argparse.Namespace) -> dict:
    """Idempotently publish or recover one fully replayed Stage surface."""

    report = build_selection(args)
    index_path = args.index.resolve()
    expected_index = json.loads(json.dumps(report))
    if index_path.exists():
        if not index_path.is_file() or _read_json(index_path) != expected_index:
            raise SystemExit("completed Stage candidate index differs from exact replay")
    else:
        _atomic_text(index_path, json.dumps(report, indent=2) + "\n")

    selected = Path(str(report["selected_checkpoint"])).resolve()
    best_record = args.best_checkpoint_record.resolve()
    if best_record.exists():
        if _read_exact_pointer(best_record, "completed Stage best record") != selected:
            raise SystemExit("completed Stage best record differs from exact replay")
    else:
        _atomic_text(best_record, f"{selected}\n")

    final_progress = Path(str(report["candidates"][-1]["checkpoint"])).resolve()
    expected_last = args.expected_last_checkpoint.resolve()
    if not expected_last.is_file() or expected_last != final_progress:
        raise SystemExit("resume/current checkpoint is not the final Stage boundary")

    manifest_path = args.manifest.resolve()
    manifest = _read_manifest(manifest_path)
    present = {key: manifest.get(key, []) for key in COMPLETION_KEYS}
    has_any = any(values for values in present.values())
    has_complete = all(len(values) == 1 for values in present.values())
    expected_fields = {
        "candidate_index": str(index_path),
        "best_checkpoint": str(selected),
        "last_checkpoint": str(final_progress),
    }
    if has_any:
        if not has_complete or any(
            present[key] != [expected]
            for key, expected in expected_fields.items()
        ) or not present["finished_at"][0]:
            raise SystemExit("completed Stage manifest surface is partial or inconsistent")

    last_record = args.last_checkpoint_record.resolve()
    if last_record.exists():
        if _read_exact_pointer(last_record, "completed Stage last record") != final_progress:
            raise SystemExit("completed Stage last record differs from final boundary")
    else:
        # This is the recoverable crash window after the manifest was committed
        # but before the legacy pointer write in an older launcher revision.
        _atomic_text(last_record, f"{final_progress}\n")

    if not has_any:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        lines.extend(
            (
                f"candidate_index={index_path}",
                f"best_checkpoint={selected}",
                f"last_checkpoint={final_progress}",
                f"finished_at={timestamp}",
            )
        )
        _atomic_text(manifest_path, "\n".join(lines) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--phase", required=True, choices=("stage1", "stage2"))
    register_parser.add_argument("--step", required=True, type=int)
    register_parser.add_argument("--summary", required=True, type=Path)
    register_parser.add_argument("--checkpoint", required=True, type=Path)
    register_parser.add_argument("--output", required=True, type=Path)
    register_parser.add_argument("--validation-schema", required=True)
    register_parser.add_argument("--questions", type=int, default=747)
    register_parser.add_argument("--world-size", type=int, default=4)
    register_parser.add_argument("--physical-gpus", required=True)
    register_parser.add_argument("--phase-input", action="store_true")
    register_parser.add_argument("--rollout-batches", type=int, default=None)
    register_parser.add_argument(
        "--registered-checkpoint-sha256", required=True
    )
    register_parser.add_argument("--registered-payload-sha256", required=True)
    register_parser.add_argument("--v7-source-sha256", required=True)
    register_parser.add_argument("--cot-encoder-sha256", required=True)
    register_parser.add_argument(
        "--stage1-reference-checkpoint", type=Path
    )
    register_parser.set_defaults(function=register)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--phase", required=True, choices=("stage1", "stage2"))
    select_parser.add_argument("--physical-gpus", required=True)
    select_parser.add_argument("--expected-steps")
    select_parser.add_argument("--expected-rollout-batches")
    select_parser.add_argument("--candidates", nargs="+", required=True, type=Path)
    select_parser.add_argument("--output", required=True, type=Path)
    select_parser.add_argument("--best-checkpoint-record", required=True, type=Path)
    select_parser.add_argument(
        "--registered-checkpoint-sha256", required=True
    )
    select_parser.add_argument("--registered-payload-sha256", required=True)
    select_parser.add_argument("--v7-source-sha256", required=True)
    select_parser.add_argument("--cot-encoder-sha256", required=True)
    select_parser.add_argument("--stage1-reference-checkpoint", type=Path)
    select_parser.set_defaults(function=select)
    verify_parser = subparsers.add_parser("verify-completed")
    verify_parser.add_argument(
        "--phase", required=True, choices=("stage1", "stage2")
    )
    verify_parser.add_argument("--physical-gpus", required=True)
    verify_parser.add_argument("--expected-steps")
    verify_parser.add_argument("--expected-rollout-batches")
    verify_parser.add_argument("--candidates", nargs="+", required=True, type=Path)
    verify_parser.add_argument("--index", required=True, type=Path)
    verify_parser.add_argument(
        "--best-checkpoint-record", required=True, type=Path
    )
    verify_parser.add_argument(
        "--last-checkpoint-record", required=True, type=Path
    )
    verify_parser.add_argument("--manifest", required=True, type=Path)
    verify_parser.add_argument(
        "--expected-last-checkpoint", required=True, type=Path
    )
    verify_parser.add_argument(
        "--registered-checkpoint-sha256", required=True
    )
    verify_parser.add_argument("--registered-payload-sha256", required=True)
    verify_parser.add_argument("--v7-source-sha256", required=True)
    verify_parser.add_argument("--cot-encoder-sha256", required=True)
    verify_parser.add_argument("--stage1-reference-checkpoint", type=Path)
    verify_parser.set_defaults(function=verify_completed)
    finalize_parser = subparsers.add_parser("finalize-completed")
    finalize_parser.add_argument(
        "--phase", required=True, choices=("stage1", "stage2")
    )
    finalize_parser.add_argument("--physical-gpus", required=True)
    finalize_parser.add_argument("--expected-steps")
    finalize_parser.add_argument("--expected-rollout-batches")
    finalize_parser.add_argument(
        "--candidates", nargs="+", required=True, type=Path
    )
    finalize_parser.add_argument("--index", required=True, type=Path)
    finalize_parser.add_argument(
        "--best-checkpoint-record", required=True, type=Path
    )
    finalize_parser.add_argument(
        "--last-checkpoint-record", required=True, type=Path
    )
    finalize_parser.add_argument("--manifest", required=True, type=Path)
    finalize_parser.add_argument(
        "--expected-last-checkpoint", required=True, type=Path
    )
    finalize_parser.add_argument(
        "--registered-checkpoint-sha256", required=True
    )
    finalize_parser.add_argument("--registered-payload-sha256", required=True)
    finalize_parser.add_argument("--v7-source-sha256", required=True)
    finalize_parser.add_argument("--cot-encoder-sha256", required=True)
    finalize_parser.add_argument("--stage1-reference-checkpoint", type=Path)
    finalize_parser.set_defaults(function=finalize_completed)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
