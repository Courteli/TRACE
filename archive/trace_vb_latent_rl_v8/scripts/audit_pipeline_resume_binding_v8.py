#!/usr/bin/env python3
"""Bind formal resume to the latest contiguous validated v8 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trace_vb_candidate_registry import build_selection  # noqa: E402
from trace_vb_checkpoint_contract import (  # noqa: E402
    COT_ENCODER_CHECKPOINT_SHA256,
    REGISTERED_CAPABILITY_CHECKPOINT_SHA256,
    REGISTERED_CAPABILITY_PAYLOAD_SHA256,
)


STAGE_COORDINATES = {
    "stage1": (0, 512, 1024, 1536, 2048),
    "stage2": (0, 256, 512, 768, 1024),
}
METRIC_CONTRACTS = (
    {
        "prefix": "metric_safe_baseline",
        "source": "student_commit",
        "origin": (
            "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
            "20260818-213000_trace_vb_v7_full_seed0/student_initial_gate.json"
        ),
        "sha256": (
            "9b888b36956f70affae153526b716ad26343164b4aa66d6d3d54f70561144aa2"
        ),
        "correct": "527",
        "questions": "747",
    },
    {
        "prefix": "registered_capability_validation",
        "source": "capability_teacher_all_roles",
        "origin": (
            "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
            "20260818-213000_trace_vb_v7_full_seed0/capability_parity_gate.json"
        ),
        "sha256": (
            "59123d2bfc020335f54d70862a904e1b70e5ea1ce764483692809ae4e6772116"
        ),
        "correct": "540",
        "questions": "747",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def is_single_link_regular_file(path: Path) -> bool:
    return (
        not path.is_symlink()
        and path.is_file()
        and path.stat().st_nlink == 1
    )


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_manifest(path: Path) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line or "=" not in line:
            raise SystemExit(f"malformed pipeline manifest line {line_number}")
        key, value = line.split("=", 1)
        if not key or not value:
            raise SystemExit(f"empty pipeline manifest field at line {line_number}")
        values.setdefault(key, []).append(value)
    return values


def require_single(values: dict[str, list[str]], key: str, expected: str) -> None:
    observed = values.get(key, [])
    if observed != [expected]:
        raise SystemExit(
            f"pipeline manifest {key}={observed!r}, expected exactly {expected!r}"
        )


def record_name(phase: str, coordinate: int) -> str:
    stem = "candidate_step" if phase == "stage1" else "candidate_rollout"
    return f"{stem}_{coordinate:04d}.json"


def read_candidate(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"candidate record is not an object: {path}")
    return value


def latest_contiguous_candidate(
    candidate_dir: Path,
    phase: str,
    *,
    physical_gpus: str,
    registered_checkpoint_sha256: str,
    registered_payload_sha256: str,
    v7_source_sha256: str,
    cot_encoder_sha256: str,
    stage1_reference_checkpoint: Path | None,
) -> tuple[int, Path, dict, dict]:
    if not candidate_dir.is_dir():
        raise SystemExit("formal resume candidate directory is missing")
    coordinates = STAGE_COORDINATES[phase]
    expected_paths = [candidate_dir / record_name(phase, value) for value in coordinates]
    glob_pattern = "candidate_step_*.json" if phase == "stage1" else "candidate_rollout_*.json"
    observed = {path.resolve() for path in candidate_dir.glob(glob_pattern)}
    expected = {path.resolve() for path in expected_paths}
    unexpected = sorted(str(path) for path in observed - expected)
    if unexpected:
        raise SystemExit(f"unexpected candidate records in formal run: {unexpected}")

    latest_index = -1
    gap_seen = False
    for index, path in enumerate(expected_paths):
        if path.is_file():
            if gap_seen:
                raise SystemExit("candidate history is not a contiguous validated prefix")
            latest_index = index
        else:
            gap_seen = True
    if latest_index < 0:
        raise SystemExit("formal resume requires the immutable boundary-zero candidate")

    prefix_paths = [path.resolve() for path in expected_paths[: latest_index + 1]]
    prefix_coordinates = coordinates[: latest_index + 1]
    replay = build_selection(
        argparse.Namespace(
            physical_gpus=physical_gpus,
            registered_checkpoint_sha256=registered_checkpoint_sha256,
            registered_payload_sha256=registered_payload_sha256,
            v7_source_sha256=v7_source_sha256,
            cot_encoder_sha256=cot_encoder_sha256,
            stage1_reference_checkpoint=stage1_reference_checkpoint,
            candidates=prefix_paths,
            expected_steps=(
                ",".join(map(str, prefix_coordinates))
                if phase == "stage1"
                else None
            ),
            expected_rollout_batches=(
                ",".join(map(str, prefix_coordinates))
                if phase == "stage2"
                else None
            ),
            phase=phase,
        )
    )
    latest_path = prefix_paths[-1]
    record = read_candidate(latest_path)
    return latest_index, latest_path, record, replay


def read_single_path(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0]:
        raise SystemExit(f"{label} must contain exactly one path")
    value = Path(lines[0]).resolve()
    if not value.is_file():
        raise SystemExit(f"{label} target is missing")
    return value


def audit(args: argparse.Namespace) -> dict:
    identifier = r"(?:0|[1-9][0-9]*)"
    if re.fullmatch(rf"{identifier}(?:,{identifier}){{3}}", args.physical_gpus) is None:
        raise SystemExit("pipeline physical GPU contract is malformed")
    if len(set(args.physical_gpus.split(","))) != 4:
        raise SystemExit("pipeline physical GPU contract contains duplicate IDs")
    manifest_path = args.manifest.resolve()
    recovery = args.recovery_checkpoint.resolve()
    v7_best = args.v7_best_checkpoint.resolve()
    candidate_dir = args.candidate_dir.resolve()
    if not manifest_path.is_file() or not recovery.is_file() or not v7_best.is_file():
        raise SystemExit("resume binding input is missing")
    values = read_manifest(manifest_path)
    require_single(values, "model", "TRACE-VB-v8")
    require_single(values, "pipeline", "train_and_validation_only")
    require_single(values, "pipeline_tag", args.pipeline_tag)
    require_single(values, "stage1_tag", args.stage1_tag)
    require_single(values, "stage2_tag", args.stage2_tag)
    require_single(values, "train_seed", str(args.train_seed))
    require_single(values, "physical_gpus", args.physical_gpus)
    require_single(values, "formal_gpus", args.physical_gpus)
    require_single(values, "fixed_gpus", args.physical_gpus)
    metric_provenance = {}
    for contract in METRIC_CONTRACTS:
        prefix = contract["prefix"]
        require_single(values, f"{prefix}_source", contract["source"])
        require_single(values, f"{prefix}_origin", contract["origin"])
        require_single(values, f"{prefix}_sha256", contract["sha256"])
        require_single(values, f"{prefix}_correct", contract["correct"])
        require_single(values, f"{prefix}_questions", contract["questions"])
        artifacts = values.get(f"{prefix}_artifact", [])
        if len(artifacts) != 1:
            raise SystemExit(
                f"pipeline manifest has no single {prefix} artifact"
            )
        artifact = Path(artifacts[0])
        expected_artifact = lexical_absolute(
            manifest_path.parent / f"{prefix}.json"
        )
        if (
            not artifact.is_absolute()
            or lexical_absolute(artifact) != expected_artifact
            or not is_single_link_regular_file(artifact)
            or sha256(artifact) != contract["sha256"]
        ):
            raise SystemExit(f"pipeline {prefix} artifact continuity failed")
        metric_provenance[prefix] = {
            "artifact": str(artifact.resolve()),
            "sha256": contract["sha256"],
            "correct_count": int(contract["correct"]),
            "questions": int(contract["questions"]),
        }
    recorded_v7_paths = values.get("v7_best_checkpoint", [])
    if len(recorded_v7_paths) != 1 or Path(recorded_v7_paths[0]).resolve() != v7_best:
        raise SystemExit("pipeline manifest belongs to a different v7 best path")
    actual_v7_sha = sha256(v7_best)
    require_single(values, "v7_best_checkpoint_sha256", actual_v7_sha)
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
    require_single(
        values, "cot_encoder_checkpoint_sha256", COT_ENCODER_CHECKPOINT_SHA256
    )
    expected_stage_tag = args.stage1_tag if args.phase == "stage1" else args.stage2_tag
    if candidate_dir.name != "candidates" or candidate_dir.parent.name != expected_stage_tag:
        raise SystemExit("resume candidate directory is not bound to the pipeline stage tag")
    stage1_reference = None
    if args.phase == "stage2":
        stage1_dir = candidate_dir.parent.parent / args.stage1_tag
        stage1_reference = read_single_path(
            stage1_dir / "best_checkpoint.txt", "selected Stage-1 checkpoint"
        )

    completed, record_path, record, replay = latest_contiguous_candidate(
        candidate_dir,
        args.phase,
        physical_gpus=args.physical_gpus,
        registered_checkpoint_sha256=REGISTERED_CAPABILITY_CHECKPOINT_SHA256,
        registered_payload_sha256=REGISTERED_CAPABILITY_PAYLOAD_SHA256,
        v7_source_sha256=actual_v7_sha,
        cot_encoder_sha256=COT_ENCODER_CHECKPOINT_SHA256,
        stage1_reference_checkpoint=stage1_reference,
    )
    if record.get("physical_gpus") != args.physical_gpus:
        raise SystemExit("latest candidate belongs to another physical GPU set")
    candidate_checkpoint = Path(str(record.get("checkpoint", ""))).resolve()
    candidate_sha = str(record.get("checkpoint_sha256", ""))
    if not candidate_checkpoint.is_file() or sha256(candidate_checkpoint) != candidate_sha:
        raise SystemExit("latest candidate checkpoint continuity failed")
    if recovery != candidate_checkpoint or sha256(recovery) != candidate_sha:
        raise SystemExit(
            "formal resume accepts only the latest contiguous validated candidate"
        )

    report = {
        "schema_version": "trace_vb_v8_pipeline_resume_binding_v1",
        "status": "PASS",
        "pipeline_tag": args.pipeline_tag,
        "phase": args.phase,
        "physical_gpus": args.physical_gpus,
        "formal_gpus": args.physical_gpus,
        "fixed_gpus": args.physical_gpus,
        "train_seed": args.train_seed,
        "v7_best_checkpoint": str(v7_best),
        "v7_best_checkpoint_sha256": actual_v7_sha,
        "candidate_directory": str(candidate_dir),
        "completed_intervals": completed,
        "latest_candidate_record": str(record_path),
        "replayed_candidate_count": completed + 1,
        "replayed_candidate_records": [
            str((candidate_dir / record_name(args.phase, coordinate)).resolve())
            for coordinate in STAGE_COORDINATES[args.phase][: completed + 1]
        ],
        "replayed_candidate_index": replay,
        "validated_boundary_checkpoint": str(recovery),
        "validated_boundary_checkpoint_sha256": candidate_sha,
        "formal_resume_policy": "latest_contiguous_validated_boundary_only",
        "metric_provenance": metric_provenance,
    }
    atomic_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pipeline-tag", required=True)
    parser.add_argument("--stage1-tag", required=True)
    parser.add_argument("--stage2-tag", required=True)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--v7-best-checkpoint", required=True, type=Path)
    parser.add_argument("--recovery-checkpoint", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("stage1", "stage2"))
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
