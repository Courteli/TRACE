#!/usr/bin/env python3
"""Admit only exact TRACE-VB-v8 validated candidate boundaries for resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from trace_vb_checkpoint_contract import (
    checkpoint_epoch,
    checkpoint_global_step,
    checkpoint_provenance,
    checkpoint_stage,
    load_checkpoint,
    require_full_state,
    rollout_batches_seen,
    validate_stage2_policy_reference,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_loop_boundary(
    checkpoint: dict,
    *,
    completed_intervals: int,
    interval_batches: int,
) -> dict[str, object]:
    """Validate the Lightning state saved at one completed validation boundary.

    ``global_step`` cannot identify a Stage-2 boundary because manual RL may
    perform a variable number of optimizer updates.  The current epoch's
    synchronized dataloader progress is the authoritative boundary instead.
    """
    try:
        fit_loop = checkpoint["loops"]["fit_loop"]
        batch_progress = fit_loop["epoch_loop.batch_progress"]
        current_batch = batch_progress["current"]
        epoch_progress = fit_loop["epoch_progress"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "resume checkpoint lacks authoritative Lightning loop progress"
        ) from error
    if not all(
        isinstance(value, dict)
        for value in (fit_loop, batch_progress, current_batch, epoch_progress)
    ):
        raise RuntimeError("Lightning loop progress has the wrong structure")

    if batch_progress.get("is_last_batch") is not True:
        raise RuntimeError("resume checkpoint was not saved at the last batch")
    batch_facts: dict[str, int] = {}
    for key in ("ready", "started", "processed", "completed"):
        value = current_batch.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"current batch progress {key} is not an integer")
        if value != interval_batches:
            raise RuntimeError(
                f"current batch progress {key}={value}, expected "
                f"{interval_batches}"
            )
        batch_facts[key] = value

    expected_epoch = {
        "ready": completed_intervals,
        "started": completed_intervals,
        "processed": completed_intervals,
        # Lightning's last.ckpt is written after validation while the current
        # epoch is still completing.  The top-level epoch is therefore
        # completed_intervals - 1 and epoch_progress.completed trails by one.
        "completed": completed_intervals - 1,
    }
    epoch_facts: dict[str, dict[str, int]] = {}
    for scope in ("current", "total"):
        observed = epoch_progress.get(scope)
        if not isinstance(observed, dict):
            raise RuntimeError(f"epoch progress has no {scope} mapping")
        epoch_facts[scope] = {}
        for key, expected in expected_epoch.items():
            value = observed.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(
                    f"{scope} epoch progress {key} is not an integer"
                )
            if value != expected:
                raise RuntimeError(
                    f"{scope} epoch progress {key}={value}, expected {expected}"
                )
            epoch_facts[scope][key] = value
    return {
        "current_batch_progress": batch_facts,
        "is_last_batch": True,
        "epoch_progress": epoch_facts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--completed-intervals", required=True, type=int)
    parser.add_argument("--completed-candidate", required=True, type=Path)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--interval-rollout-batches", type=int, default=0)
    parser.add_argument("--stage1-interval-steps", type=int, default=0)
    parser.add_argument("--registered-checkpoint-sha256", required=True)
    parser.add_argument("--registered-payload-sha256", required=True)
    parser.add_argument("--v7-source-sha256", required=True)
    parser.add_argument("--cot-encoder-sha256", required=True)
    parser.add_argument("--stage1-reference-checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    completed = int(args.completed_intervals)
    identifier = r"(?:0|[1-9][0-9]*)"
    if re.fullmatch(rf"{identifier}(?:,{identifier}){{3}}", args.physical_gpus) is None:
        raise SystemExit("resume physical GPU contract is malformed")
    if len(set(args.physical_gpus.split(","))) != 4:
        raise SystemExit("resume physical GPU contract contains duplicate IDs")
    if not 0 <= completed <= 4:
        raise SystemExit("completed interval count must be in [0, 4]")
    if not args.completed_candidate.is_file():
        raise SystemExit("recovery requires its completed progress anchor")
    candidate = json.loads(args.completed_candidate.read_text(encoding="utf-8"))
    if candidate.get("schema_version") != "trace_vb_v8_candidate_v1":
        raise SystemExit("completed candidate has the wrong schema")
    expected_phase = f"stage{args.stage}"
    if candidate.get("phase") != expected_phase:
        raise SystemExit("completed candidate belongs to the wrong phase")
    if candidate.get("physical_gpus") != args.physical_gpus:
        raise SystemExit("completed candidate belongs to another physical GPU set")
    candidate_checkpoint_path = Path(candidate.get("checkpoint", ""))
    if (
        not candidate_checkpoint_path.is_file()
        or sha256(candidate_checkpoint_path)
        != candidate.get("checkpoint_sha256")
    ):
        raise SystemExit("completed candidate checkpoint continuity failed")
    if args.checkpoint.resolve() != candidate_checkpoint_path.resolve():
        raise SystemExit(
            "formal resume accepts only the latest validated candidate checkpoint"
        )
    if sha256(args.checkpoint) != candidate.get("checkpoint_sha256"):
        raise SystemExit("resume checkpoint SHA256 differs from its candidate record")
    checkpoint = load_checkpoint(args.checkpoint)
    actual_stage = checkpoint_stage(checkpoint)
    expected_checkpoint_stage = 1 if completed == 0 else args.stage
    if actual_stage != expected_checkpoint_stage:
        raise SystemExit(
            f"boundary checkpoint stage is {actual_stage}, expected "
            f"{expected_checkpoint_stage}"
        )
    provenance = checkpoint_provenance(
        checkpoint,
        expected_registered_checkpoint_sha256=(
            args.registered_checkpoint_sha256
        ),
        expected_registered_payload_sha256=args.registered_payload_sha256,
        expected_v7_source_sha256=args.v7_source_sha256,
        expected_cot_encoder_sha256=args.cot_encoder_sha256,
    )
    loop_boundary = None
    if completed >= 1:
        require_full_state(checkpoint)
    global_step = checkpoint_global_step(checkpoint)
    epoch = checkpoint_epoch(checkpoint)
    rollouts = rollout_batches_seen(checkpoint)
    candidate_source_step = int(candidate.get("source_global_step", -1))
    if global_step != candidate_source_step:
        raise SystemExit("completed candidate checkpoint has the wrong global step")
    recorded_facts = {
        "checkpoint_stage": actual_stage,
        "checkpoint_global_step": global_step,
        "checkpoint_epoch": epoch,
        "checkpoint_rollout_batches_seen": rollouts,
    }
    for key, value in recorded_facts.items():
        if int(candidate.get(key, -1)) != value:
            raise SystemExit(f"candidate boundary fact {key} is inconsistent")
    if args.stage == 1:
        interval_steps = int(args.stage1_interval_steps)
        if interval_steps <= 0:
            raise SystemExit("Stage-1 recovery requires a positive interval size")
        expected_candidate_step = completed * interval_steps
        if (
            int(candidate.get("selection_step", -1))
            != expected_candidate_step
            or int(candidate.get("step", -1)) != expected_candidate_step
            or candidate_source_step != expected_candidate_step
        ):
            raise SystemExit("Stage-1 progress anchor has the wrong coordinate")
        if actual_stage != 1:
            raise SystemExit("Stage-1 progress anchor has the wrong stage")
        if rollouts != 0:
            raise SystemExit("Stage-1 recovery has a nonzero RL rollout counter")
        if completed >= 1 and epoch != completed - 1:
            raise SystemExit("Stage-1 checkpoint epoch is not the exact boundary")
        if completed >= 1:
            loop_boundary = validated_loop_boundary(
                checkpoint,
                completed_intervals=completed,
                interval_batches=interval_steps,
            )
    else:
        interval_size = int(args.interval_rollout_batches)
        if interval_size <= 0:
            raise SystemExit("Stage-2 recovery requires a positive rollout interval")
        lower = completed * interval_size
        if rollouts != lower:
            raise SystemExit("Stage-2 checkpoint is not at a validated boundary")
        if int(candidate.get("rollout_batches", -1)) != lower:
            raise SystemExit("completed Stage-2 candidate has wrong rollout coordinate")
        expected_selection_step = 0 if completed == 0 else candidate_source_step
        if (
            int(candidate.get("selection_step", -1)) != expected_selection_step
            or int(candidate.get("step", -1)) != expected_selection_step
        ):
            raise SystemExit("completed Stage-2 candidate has wrong selection coordinate")
        expected_candidate_stage = 1 if completed == 0 else 2
        if actual_stage != expected_candidate_stage:
            raise SystemExit("Stage-2 progress anchor has the wrong checkpoint stage")
        if bool(candidate.get("phase_input", False)) != (completed == 0):
            raise SystemExit("Stage-2 phase-input marker is inconsistent")
        if completed >= 1 and epoch != completed - 1:
            raise SystemExit("Stage-2 checkpoint epoch is not the exact boundary")
        if args.stage1_reference_checkpoint is None:
            raise SystemExit("Stage-2 recovery requires its selected Stage-1 checkpoint")
        stage1_checkpoint = load_checkpoint(args.stage1_reference_checkpoint)
        checkpoint_provenance(
            stage1_checkpoint,
            expected_registered_checkpoint_sha256=(
                args.registered_checkpoint_sha256
            ),
            expected_registered_payload_sha256=args.registered_payload_sha256,
            expected_v7_source_sha256=args.v7_source_sha256,
            expected_cot_encoder_sha256=args.cot_encoder_sha256,
        )
        if completed == 0:
            if args.checkpoint.resolve() != args.stage1_reference_checkpoint.resolve():
                raise SystemExit(
                    "Stage-2 boundary zero is not the selected Stage-1 checkpoint"
                )
        else:
            validate_stage2_policy_reference(checkpoint, stage1_checkpoint)
            loop_boundary = validated_loop_boundary(
                checkpoint,
                completed_intervals=completed,
                interval_batches=interval_size,
            )

    report = {
        "schema_version": "trace_vb_v8_resume_checkpoint_contract_v1",
        "status": "PASS",
        "checkpoint": str(args.checkpoint.resolve()),
        "requested_stage": args.stage,
        "physical_gpus": args.physical_gpus,
        "checkpoint_stage": actual_stage,
        "global_step": global_step,
        "epoch": epoch,
        "completed_intervals": completed,
        "rollout_batches_seen": rollouts,
        "full_state_required": completed >= 1,
        "optimizer_state_count": 1 if completed >= 1 else None,
        "scheduler_state_count": 1 if completed >= 1 else None,
        "formal_resume_mode": (
            "weights_only_boundary_restart"
            if completed == 0
            else "full_state_validated_boundary"
        ),
        "validated_loop_boundary": loop_boundary,
        "provenance": provenance,
        "completed_candidate": str(args.completed_candidate.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
