#!/usr/bin/env python3
"""Atomic attempt-scoped supervisor state for initial and resumed v8 runs."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA = "trace_vb_v8_supervisor_attempt_v1"
SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9._-]+")
KINDS = {"initial", "resume_stage1", "resume_stage2"}
STATUSES = {"WAITING", "RUNNING", "SUCCEEDED", "FAILED"}
TRANSITIONS = {
    "WAITING": {"RUNNING", "SUCCEEDED", "FAILED"},
    "RUNNING": {"SUCCEEDED", "FAILED"},
    "SUCCEEDED": set(),
    "FAILED": set(),
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_status(path: Path, status: int) -> None:
    if type(status) is not int or not 0 <= status <= 255:
        raise RuntimeError("supervisor exit status must be an integer in [0, 255]")
    atomic_bytes(path, f"{status}\n".encode("ascii"))


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"supervisor JSON is not an object: {path}")
    return value


def validate_identifier(value: str, label: str) -> str:
    if SAFE_IDENTIFIER.fullmatch(value) is None:
        raise RuntimeError(f"unsafe {label}: {value!r}")
    return value


def validate_gpu_csv(value: str) -> str:
    identifier = r"(?:0|[1-9][0-9]*)"
    if (
        re.fullmatch(rf"{identifier}(?:,{identifier}){{3}}", value) is None
        or len(set(value.split(","))) != 4
    ):
        raise RuntimeError("supervisor requires four unique physical GPUs")
    return value


def capture_log_byte_offset_floor(path: Path, *, expected: Path) -> tuple[str, int]:
    """Bind a fresh attempt to the pre-launch end of one append-only log."""

    path = path.resolve()
    expected = expected.resolve()
    if path != expected:
        raise RuntimeError(
            f"supervisor Stage log path {path} does not match {expected}"
        )
    if path.exists() and not path.is_file():
        raise RuntimeError(f"supervisor Stage log is not a regular file: {path}")
    byte_offset = path.stat().st_size if path.is_file() else 0
    return str(path), byte_offset


def attempt_paths(
    supervisor_dir: Path, pipeline_tag: str, attempt_id: str
) -> tuple[Path, Path, Path, Path]:
    active = supervisor_dir / f"{pipeline_tag}.active_attempt.json"
    status = supervisor_dir / f"{pipeline_tag}.{attempt_id}.exit_status"
    canonical_status = supervisor_dir / f"{pipeline_tag}.exit_status"
    history = (
        supervisor_dir
        / f"{pipeline_tag}.attempts"
        / f"{attempt_id}.json"
    )
    return active, status, canonical_status, history


def archive_legacy_status(
    supervisor_dir: Path, pipeline_tag: str, canonical_status: Path
) -> None:
    if not canonical_status.is_file():
        return
    raw = canonical_status.read_text(encoding="ascii").strip()
    try:
        code = int(raw)
    except ValueError as error:
        raise RuntimeError("legacy supervisor status is malformed") from error
    if raw != str(code) or not 0 <= code <= 255:
        raise RuntimeError("legacy supervisor status is outside [0, 255]")
    archive_id = datetime.now().astimezone().strftime("legacy_%Y%m%d-%H%M%S")
    archive = supervisor_dir / f"{pipeline_tag}.attempts" / f"{archive_id}.json"
    suffix = 0
    while archive.exists():
        suffix += 1
        archive = supervisor_dir / f"{pipeline_tag}.attempts" / (
            f"{archive_id}_{suffix}.json"
        )
    atomic_json(
        archive,
        {
            "schema_version": "trace_vb_v8_legacy_supervisor_status_v1",
            "pipeline_tag": pipeline_tag,
            "exit_status": code,
            "source": str(canonical_status.resolve()),
            "archived_at": now_iso(),
        },
    )


def validate_resume_binding(
    path: Path, *, pipeline_tag: str, phase: str, physical_gpus: str
) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError("resume attempt has no validated boundary contract")
    value = read_object(path)
    if (
        value.get("schema_version")
        != "trace_vb_v8_pipeline_resume_binding_v1"
        or value.get("status") != "PASS"
        or value.get("pipeline_tag") != pipeline_tag
        or value.get("phase") != phase
        or value.get("physical_gpus") != physical_gpus
    ):
        raise RuntimeError("resume boundary contract does not bind this attempt")
    checkpoint = Path(
        str(value.get("validated_boundary_checkpoint", ""))
    )
    if not checkpoint.is_absolute() or not checkpoint.is_file():
        raise RuntimeError("resume boundary checkpoint is missing")
    return path


def start(args: argparse.Namespace) -> dict[str, Any]:
    supervisor_dir = args.supervisor_dir.resolve()
    pipeline_tag = validate_identifier(args.pipeline_tag, "pipeline tag")
    attempt_id = validate_identifier(args.attempt_id, "attempt identifier")
    attempt_kind = str(args.attempt_kind)
    if attempt_kind not in KINDS:
        raise RuntimeError(f"invalid attempt kind: {attempt_kind}")
    physical_gpus = validate_gpu_csv(args.physical_gpus)
    if args.formal_gpus != physical_gpus or args.fixed_gpus != physical_gpus:
        raise RuntimeError("supervisor physical/formal/fixed GPU sets differ")
    tmux_session = validate_identifier(args.tmux_session, "tmux session")
    artifact_root = supervisor_dir.parent
    stage1_log_path, stage1_log_floor = capture_log_byte_offset_floor(
            args.stage1_train_log,
            expected=(
                artifact_root
                / "training"
                / f"{pipeline_tag}_stage1"
                / "train.log"
            ),
        )
    stage2_log_path, stage2_log_floor = capture_log_byte_offset_floor(
            args.stage2_train_log,
            expected=(
                artifact_root
                / "training"
                / f"{pipeline_tag}_stage2"
                / "train.log"
            ),
        )
    log_byte_offset_floors = {
        stage1_log_path: stage1_log_floor,
        stage2_log_path: stage2_log_floor,
    }
    phase = None
    binding = None
    if attempt_kind.startswith("resume_"):
        phase = attempt_kind.removeprefix("resume_")
        if args.resume_phase != phase or args.validated_boundary_contract is None:
            raise RuntimeError("resume attempt phase/boundary is incomplete")
        binding = validate_resume_binding(
            args.validated_boundary_contract,
            pipeline_tag=pipeline_tag,
            phase=phase,
            physical_gpus=physical_gpus,
        )
    elif args.resume_phase is not None or args.validated_boundary_contract is not None:
        raise RuntimeError("initial attempt cannot declare a resume boundary")

    active, status_path, canonical_status, history = attempt_paths(
        supervisor_dir, pipeline_tag, attempt_id
    )
    if status_path.exists() or history.exists():
        raise RuntimeError("refusing to reuse a supervisor attempt identifier")
    if active.is_file():
        previous = read_object(active)
        if previous.get("status") in {"WAITING", "RUNNING"}:
            raise RuntimeError("another supervisor attempt is still active")
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    archive_legacy_status(supervisor_dir, pipeline_tag, canonical_status)
    timestamp = now_iso()
    value = {
        "schema_version": SCHEMA,
        "pipeline_tag": pipeline_tag,
        "physical_gpus": physical_gpus,
        "formal_gpus": physical_gpus,
        "fixed_gpus": physical_gpus,
        "attempt_id": attempt_id,
        "attempt_kind": attempt_kind,
        "resume_phase": phase,
        "tmux_session": tmux_session,
        "exit_status_file": str(status_path.resolve()),
        "validated_boundary_contract": (
            str(binding) if binding is not None else None
        ),
        "log_byte_offset_floors": log_byte_offset_floors,
        "status": "WAITING",
        "exit_status": None,
        "started_at": timestamp,
        "updated_at": timestamp,
    }
    atomic_json(history, value)
    atomic_json(active, value)
    return value


def transition(args: argparse.Namespace) -> dict[str, Any]:
    supervisor_dir = args.supervisor_dir.resolve()
    pipeline_tag = validate_identifier(args.pipeline_tag, "pipeline tag")
    attempt_id = validate_identifier(args.attempt_id, "attempt identifier")
    target = str(args.status)
    if target not in STATUSES - {"WAITING"}:
        raise RuntimeError(f"invalid supervisor transition target: {target}")
    active, status_path, canonical_status, history = attempt_paths(
        supervisor_dir, pipeline_tag, attempt_id
    )
    if not active.is_file():
        raise RuntimeError("active supervisor attempt is missing")
    value = read_object(active)
    if (
        value.get("schema_version") != SCHEMA
        or value.get("pipeline_tag") != pipeline_tag
        or value.get("attempt_id") != attempt_id
        or Path(str(value.get("exit_status_file", ""))).resolve()
        != status_path.resolve()
    ):
        raise RuntimeError("active supervisor attempt identity mismatch")
    current = str(value.get("status", ""))
    if target not in TRANSITIONS.get(current, set()):
        raise RuntimeError(f"invalid supervisor transition {current}->{target}")
    if target == "SUCCEEDED":
        exit_status = 0
    elif target == "FAILED":
        if args.exit_status is None or not 1 <= args.exit_status <= 255:
            raise RuntimeError("FAILED transition requires status in [1, 255]")
        exit_status = int(args.exit_status)
    else:
        if args.exit_status is not None:
            raise RuntimeError("RUNNING transition cannot publish exit status")
        exit_status = None
    if exit_status is not None:
        atomic_status(status_path, exit_status)
        atomic_status(canonical_status, exit_status)
    value["status"] = target
    value["exit_status"] = exit_status
    value["updated_at"] = now_iso()
    atomic_json(history, value)
    atomic_json(active, value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--supervisor-dir", required=True, type=Path)
    start_parser.add_argument("--pipeline-tag", required=True)
    start_parser.add_argument("--attempt-id", required=True)
    start_parser.add_argument("--attempt-kind", required=True, choices=sorted(KINDS))
    start_parser.add_argument("--physical-gpus", required=True)
    start_parser.add_argument("--formal-gpus", required=True)
    start_parser.add_argument("--fixed-gpus", required=True)
    start_parser.add_argument("--tmux-session", required=True)
    start_parser.add_argument("--stage1-train-log", required=True, type=Path)
    start_parser.add_argument("--stage2-train-log", required=True, type=Path)
    start_parser.add_argument("--resume-phase", choices=("stage1", "stage2"))
    start_parser.add_argument("--validated-boundary-contract", type=Path)
    start_parser.set_defaults(function=start)
    transition_parser = subparsers.add_parser("transition")
    transition_parser.add_argument("--supervisor-dir", required=True, type=Path)
    transition_parser.add_argument("--pipeline-tag", required=True)
    transition_parser.add_argument("--attempt-id", required=True)
    transition_parser.add_argument(
        "--status", required=True, choices=("RUNNING", "SUCCEEDED", "FAILED")
    )
    transition_parser.add_argument("--exit-status", type=int)
    transition_parser.set_defaults(function=transition)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    value = args.function(args)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
