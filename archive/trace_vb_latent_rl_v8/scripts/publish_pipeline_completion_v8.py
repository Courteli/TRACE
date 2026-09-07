#!/usr/bin/env python3
"""Atomically publish one complete TRACE-VB-v8 pipeline result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trace_vb_candidate_registry import build_selection  # noqa: E402
from trace_vb_checkpoint_contract import (  # noqa: E402
    COT_ENCODER_CHECKPOINT_SHA256,
    METRIC_SAFE_BASELINE_PATH,
    METRIC_SAFE_BASELINE_SHA256,
    REGISTERED_CAPABILITY_CHECKPOINT_SHA256,
    REGISTERED_CAPABILITY_PAYLOAD_SHA256,
    REGISTERED_CAPABILITY_VALIDATION_PATH,
    REGISTERED_CAPABILITY_VALIDATION_SHA256,
)


COMPLETION_KEYS = {
    "stage1_checkpoint",
    "stage1_checkpoint_sha256",
    "stage1_candidate_index",
    "final_checkpoint",
    "final_checkpoint_sha256",
    "final_candidate_index",
    "finished_at",
}
STAGE_COORDINATES = {
    "stage1": (0, 512, 1024, 1536, 2048),
    "stage2": (0, 256, 512, 768, 1024),
}
METRIC_CONTRACTS = (
    {
        "prefix": "metric_safe_baseline",
        "source": "student_commit",
        "origin": METRIC_SAFE_BASELINE_PATH,
        "sha256": METRIC_SAFE_BASELINE_SHA256,
        "correct": "527",
        "questions": "747",
    },
    {
        "prefix": "registered_capability_validation",
        "source": "capability_teacher_all_roles",
        "origin": REGISTERED_CAPABILITY_VALIDATION_PATH,
        "sha256": REGISTERED_CAPABILITY_VALIDATION_SHA256,
        "correct": "540",
        "questions": "747",
    },
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lexical_absolute(path: Path) -> Path:
    """Normalize dot components without following a final symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def is_single_link_regular_file(path: Path) -> bool:
    return (
        not path.is_symlink()
        and path.is_file()
        and path.stat().st_nlink == 1
    )


def atomic_bytes(path: Path, payload: bytes) -> None:
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


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def read_single_path(path: Path, label: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0]:
        raise RuntimeError(f"{label} must contain exactly one path")
    value = Path(lines[0]).resolve()
    if not value.is_file():
        raise RuntimeError(f"{label} target is missing: {value}")
    return value


def manifest_lines(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    if not path.is_file():
        raise RuntimeError(f"missing pipeline manifest: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    values: dict[str, list[str]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line or "=" not in line:
            raise RuntimeError(f"malformed manifest line {line_number}")
        key, value = line.split("=", 1)
        if not key or not value:
            raise RuntimeError(f"empty manifest field at line {line_number}")
        values.setdefault(key, []).append(value)
    return lines, values


def require_single(values: dict[str, list[str]], key: str, expected: str) -> None:
    if values.get(key) != [expected]:
        raise RuntimeError(
            f"pipeline manifest {key}={values.get(key)!r}, expected {expected!r}"
        )


def single(values: dict[str, list[str]], key: str, scope: str) -> str:
    observed = values.get(key, [])
    if len(observed) != 1:
        raise RuntimeError(f"{scope} manifest must declare {key} exactly once")
    return observed[0]


def candidate_name(phase: str, coordinate: int) -> str:
    stem = "candidate_step" if phase == "stage1" else "candidate_rollout"
    return f"{stem}_{coordinate:04d}.json"


def replay_stage(
    *,
    stage_dir: Path,
    phase: str,
    expected_tag: str,
    physical_gpus: str,
    registered_checkpoint_sha256: str,
    registered_payload_sha256: str,
    v7_source_sha256: str,
    cot_encoder_sha256: str,
    stage1_reference_checkpoint: Path | None,
    pipeline_manifest: dict[str, list[str]],
) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = stage_dir / "manifest.txt"
    _lines, values = manifest_lines(manifest_path)
    require_single(values, "model", "TRACE-VB-v8")
    require_single(values, "run_tag", expected_tag)
    require_single(values, "physical_gpus", physical_gpus)
    require_single(values, "formal_gpus", physical_gpus)
    require_single(values, "fixed_gpus", physical_gpus)
    single(values, "finished_at", phase)
    require_single(
        values,
        "registered_capability_sha256",
        registered_checkpoint_sha256,
    )
    require_single(values, "registered_capability_payload_tensors", "517")
    require_single(
        values,
        "registered_capability_payload_sha256",
        registered_payload_sha256,
    )
    require_single(values, "cot_encoder_checkpoint_sha256", cot_encoder_sha256)
    if phase == "stage1":
        require_single(values, "v7_best_checkpoint_sha256", v7_source_sha256)
    else:
        require_single(values, "v7_source_checkpoint_sha256", v7_source_sha256)
        if stage1_reference_checkpoint is None:
            raise RuntimeError("Stage-2 replay has no selected Stage-1 reference")
        require_single(
            values,
            "stage1_checkpoint",
            str(stage1_reference_checkpoint.resolve()),
        )

    for prefix in (
        "metric_safe_baseline",
        "registered_capability_validation",
    ):
        for suffix in ("source", "origin", "sha256", "correct", "questions"):
            key = f"{prefix}_{suffix}"
            require_single(values, key, single(pipeline_manifest, key, "pipeline"))
        artifact = Path(single(values, f"{prefix}_artifact", phase))
        expected_sha = single(pipeline_manifest, f"{prefix}_sha256", "pipeline")
        expected_artifact_root = stage_dir
        if phase == "stage2":
            expected_artifact_root = (
                stage_dir.parent
                / single(pipeline_manifest, "stage1_tag", "pipeline")
            )
        expected_artifact = lexical_absolute(
            expected_artifact_root / f"{prefix}.json"
        )
        if (
            not artifact.is_absolute()
            or lexical_absolute(artifact) != expected_artifact
            or not is_single_link_regular_file(artifact)
            or sha256(artifact) != expected_sha
        ):
            raise RuntimeError(f"{phase} {prefix} artifact continuity failed")

    coordinates = STAGE_COORDINATES[phase]
    candidate_dir = stage_dir / "candidates"
    expected_paths = [
        (candidate_dir / candidate_name(phase, coordinate)).resolve()
        for coordinate in coordinates
    ]
    pattern = "candidate_step_*.json" if phase == "stage1" else "candidate_rollout_*.json"
    observed_paths = {path.resolve() for path in candidate_dir.glob(pattern)}
    if observed_paths != set(expected_paths) or not all(
        path.is_file() for path in expected_paths
    ):
        raise RuntimeError(f"{phase} does not contain the exact five candidates")
    replay = build_selection(
        Namespace(
            physical_gpus=physical_gpus,
            registered_checkpoint_sha256=registered_checkpoint_sha256,
            registered_payload_sha256=registered_payload_sha256,
            v7_source_sha256=v7_source_sha256,
            cot_encoder_sha256=cot_encoder_sha256,
            stage1_reference_checkpoint=stage1_reference_checkpoint,
            candidates=expected_paths,
            expected_steps=(
                ",".join(map(str, coordinates)) if phase == "stage1" else None
            ),
            expected_rollout_batches=(
                ",".join(map(str, coordinates)) if phase == "stage2" else None
            ),
            phase=phase,
        )
    )
    index_path = stage_dir / "candidate_index.json"
    if not index_path.is_file():
        raise RuntimeError(f"missing {phase} candidate index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    normalized_replay = json.loads(json.dumps(replay))
    if index != normalized_replay:
        raise RuntimeError(f"{phase} candidate index differs from exact replay")
    require_single(values, "candidate_index", str(index_path.resolve()))

    selected = read_single_path(stage_dir / "best_checkpoint.txt", f"{phase} best record")
    if selected != Path(str(replay["selected_checkpoint"])).resolve():
        raise RuntimeError(f"{phase} best record differs from exact replay")
    require_single(values, "best_checkpoint", str(selected))
    final_progress = Path(str(replay["candidates"][-1]["checkpoint"])).resolve()
    last = read_single_path(stage_dir / "last_checkpoint.txt", f"{phase} last record")
    if last != final_progress:
        raise RuntimeError(f"{phase} last record is not the final progress boundary")
    require_single(values, "last_checkpoint", str(last))
    return replay, selected, index_path.resolve()


def publish(args: argparse.Namespace) -> dict[str, Any]:
    identifier = r"(?:0|[1-9][0-9]*)"
    if (
        re.fullmatch(
            rf"{identifier}(?:,{identifier}){{3}}", args.physical_gpus
        )
        is None
        or len(set(args.physical_gpus.split(","))) != 4
    ):
        raise RuntimeError("publication requires four unique physical GPUs")
    pipeline_dir = args.pipeline_dir.resolve()
    stage1_dir = args.stage1_dir.resolve()
    stage2_dir = args.stage2_dir.resolve()
    manifest = pipeline_dir / "manifest.txt"
    lines, values = manifest_lines(manifest)
    pipeline_tags = values.get("pipeline_tag")
    if pipeline_tags is None or len(pipeline_tags) != 1:
        raise RuntimeError("pipeline manifest must declare pipeline_tag exactly once")
    pipeline_tag = pipeline_tags[0]
    if (
        re.fullmatch(r"[A-Za-z0-9._-]+", pipeline_tag) is None
        or pipeline_dir.name != pipeline_tag
    ):
        raise RuntimeError("pipeline manifest tag does not bind its directory")
    require_single(values, "physical_gpus", args.physical_gpus)
    require_single(values, "formal_gpus", args.physical_gpus)
    require_single(values, "fixed_gpus", args.physical_gpus)
    for contract in METRIC_CONTRACTS:
        prefix = contract["prefix"]
        for suffix in ("source", "origin", "sha256", "correct", "questions"):
            require_single(values, f"{prefix}_{suffix}", contract[suffix])
        artifact = Path(single(values, f"{prefix}_artifact", "pipeline"))
        expected_artifact = lexical_absolute(
            pipeline_dir / f"{prefix}.json"
        )
        if (
            not artifact.is_absolute()
            or lexical_absolute(artifact) != expected_artifact
            or not is_single_link_regular_file(artifact)
            or sha256(artifact) != contract["sha256"]
        ):
            raise RuntimeError(f"pipeline {prefix} artifact continuity failed")
    stage1_tag = single(values, "stage1_tag", "pipeline")
    stage2_tag = single(values, "stage2_tag", "pipeline")
    if stage1_dir.name != stage1_tag or stage2_dir.name != stage2_tag:
        raise RuntimeError("Stage directories do not bind the pipeline tags")
    require_single(
        values,
        "registered_capability_sha256",
        REGISTERED_CAPABILITY_CHECKPOINT_SHA256,
    )
    require_single(values, "registered_capability_payload_tensors", "517")
    require_single(
        values,
        "registered_capability_payload_sha256",
        REGISTERED_CAPABILITY_PAYLOAD_SHA256,
    )
    registered_sha = REGISTERED_CAPABILITY_CHECKPOINT_SHA256
    payload_sha = REGISTERED_CAPABILITY_PAYLOAD_SHA256
    v7_sha = single(values, "v7_best_checkpoint_sha256", "pipeline")
    require_single(
        values, "cot_encoder_checkpoint_sha256", COT_ENCODER_CHECKPOINT_SHA256
    )
    cot_sha = COT_ENCODER_CHECKPOINT_SHA256
    stage1_index, stage1_checkpoint, stage1_index_path = replay_stage(
        stage_dir=stage1_dir,
        phase="stage1",
        expected_tag=stage1_tag,
        physical_gpus=args.physical_gpus,
        registered_checkpoint_sha256=registered_sha,
        registered_payload_sha256=payload_sha,
        v7_source_sha256=v7_sha,
        cot_encoder_sha256=cot_sha,
        stage1_reference_checkpoint=None,
        pipeline_manifest=values,
    )
    stage2_index, final_checkpoint, stage2_index_path = replay_stage(
        stage_dir=stage2_dir,
        phase="stage2",
        expected_tag=stage2_tag,
        physical_gpus=args.physical_gpus,
        registered_checkpoint_sha256=registered_sha,
        registered_payload_sha256=payload_sha,
        v7_source_sha256=v7_sha,
        cot_encoder_sha256=cot_sha,
        stage1_reference_checkpoint=stage1_checkpoint,
        pipeline_manifest=values,
    )

    recovery_fields: list[tuple[str, str]] = []
    if args.recovery_phase is not None:
        if not (
            args.recovery_checkpoint
            and args.resume_attempt
            and args.resume_binding
        ):
            raise RuntimeError("recovery publication provenance is incomplete")
        if re.fullmatch(r"[A-Za-z0-9._-]+", args.resume_attempt) is None:
            raise RuntimeError("unsafe recovery attempt identifier")
        recovery_checkpoint = args.recovery_checkpoint.resolve()
        binding_path = args.resume_binding.resolve()
        if not recovery_checkpoint.is_file() or not binding_path.is_file():
            raise RuntimeError("recovery publication input is missing")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        recovery_sha = sha256(recovery_checkpoint)
        latest_record_path = Path(
            str(binding.get("latest_candidate_record", ""))
        )
        if (
            not isinstance(binding, dict)
            or binding.get("schema_version")
            != "trace_vb_v8_pipeline_resume_binding_v1"
            or binding.get("status") != "PASS"
            or binding.get("pipeline_tag") != pipeline_tag
            or binding.get("phase") != args.recovery_phase
            or binding.get("physical_gpus") != args.physical_gpus
            or Path(
                str(binding.get("validated_boundary_checkpoint", ""))
            ).resolve()
            != recovery_checkpoint
            or binding.get("validated_boundary_checkpoint_sha256")
            != recovery_sha
            or not latest_record_path.is_absolute()
            or not latest_record_path.is_file()
        ):
            raise RuntimeError("recovery binding is not the exact PASS contract")
        latest_record = json.loads(latest_record_path.read_text(encoding="utf-8"))
        recovery_index = (
            stage1_index if args.recovery_phase == "stage1" else stage2_index
        )
        recovery_stage_dir = (
            stage1_dir if args.recovery_phase == "stage1" else stage2_dir
        )
        matching_positions = [
            position
            for position, candidate in enumerate(recovery_index["candidates"])
            if Path(str(candidate.get("checkpoint", ""))).resolve()
            == recovery_checkpoint
        ]
        if len(matching_positions) != 1:
            raise RuntimeError("recovery checkpoint is not one exact replayed boundary")
        recovery_position = matching_positions[0]
        recovery_coordinate = STAGE_COORDINATES[args.recovery_phase][
            recovery_position
        ]
        expected_record_path = (
            recovery_stage_dir
            / "candidates"
            / candidate_name(args.recovery_phase, recovery_coordinate)
        ).resolve()
        if (
            not isinstance(latest_record, dict)
            or latest_record_path.resolve() != expected_record_path
            or Path(str(latest_record.get("checkpoint", ""))).resolve()
            != recovery_checkpoint
            or latest_record.get("checkpoint_sha256") != recovery_sha
            or type(binding.get("completed_intervals")) is not int
            or binding.get("completed_intervals") != recovery_position
            or type(binding.get("replayed_candidate_count")) is not int
            or binding.get("replayed_candidate_count") != recovery_position + 1
        ):
            raise RuntimeError("recovery binding latest candidate is inconsistent")
        recovery_fields = [
            ("recovery_finished_at", now_iso()),
            ("recovery_phase", args.recovery_phase),
            ("recovery_attempt", args.resume_attempt),
            ("recovery_binding", str(binding_path)),
            ("recovery_checkpoint", str(recovery_checkpoint)),
            ("recovery_checkpoint_sha256", recovery_sha),
            ("recovery_physical_gpus", args.physical_gpus),
            ("recovery_formal_gpus", args.physical_gpus),
            ("recovery_fixed_gpus", args.physical_gpus),
            (
                "recovery_formal_policy",
                "latest_contiguous_validated_boundary_only",
            ),
            ("recovery_rewind_path_to_capability", "false"),
        ]

    pipeline_stage1_index = pipeline_dir / "stage1_candidate_index.json"
    pipeline_stage2_index = pipeline_dir / "stage2_candidate_index.json"
    atomic_bytes(pipeline_stage1_index, stage1_index_path.read_bytes())
    atomic_bytes(pipeline_stage2_index, stage2_index_path.read_bytes())
    atomic_text(pipeline_dir / "stage1_best.txt", f"{stage1_checkpoint}\n")
    atomic_text(pipeline_dir / "final_best.txt", f"{final_checkpoint}\n")

    retained = [
        line for line in lines if line.split("=", 1)[0] not in COMPLETION_KEYS
    ]
    completion_fields = [
        ("stage1_checkpoint", str(stage1_checkpoint)),
        ("stage1_checkpoint_sha256", sha256(stage1_checkpoint)),
        ("stage1_candidate_index", str(pipeline_stage1_index)),
        ("final_checkpoint", str(final_checkpoint)),
        ("final_checkpoint_sha256", sha256(final_checkpoint)),
        ("final_candidate_index", str(pipeline_stage2_index)),
        ("finished_at", now_iso()),
    ]
    published_lines = retained + [
        f"{key}={value}" for key, value in recovery_fields + completion_fields
    ]
    atomic_text(manifest, "\n".join(published_lines) + "\n")
    report = {
        "schema_version": "trace_vb_v8_pipeline_publication_v1",
        "status": "PASS",
        "pipeline_tag": pipeline_tag,
        "pipeline_directory": str(pipeline_dir),
        "physical_gpus": args.physical_gpus,
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_checkpoint_sha256": sha256(stage1_checkpoint),
        "stage1_candidate_index": str(pipeline_stage1_index),
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256(final_checkpoint),
        "final_candidate_index": str(pipeline_stage2_index),
        "recovery_phase": args.recovery_phase,
        "resume_attempt": args.resume_attempt,
        "published_at": now_iso(),
    }
    atomic_text(
        pipeline_dir / "publication.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--stage1-dir", required=True, type=Path)
    parser.add_argument("--stage2-dir", required=True, type=Path)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--recovery-phase", choices=("stage1", "stage2"))
    parser.add_argument("--recovery-checkpoint", type=Path)
    parser.add_argument("--resume-attempt")
    parser.add_argument("--resume-binding", type=Path)
    publish(parser.parse_args())


if __name__ == "__main__":
    main()
