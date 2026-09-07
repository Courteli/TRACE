#!/usr/bin/env python3
"""Read-only health monitor for a TRACE-VB-v8 train/validation-only run.

The monitor never launches training, validation, test/evidence jobs, tmux
sessions, or signals.  Its only writes are atomic JSON snapshots under the
configured state directory.  Formal progress is inferred from registered
full-validation candidates, not from progress bars or diagnostic recovery
checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


MONITOR_SCHEMA = "trace_vb_v8_train_only_monitor_v1"
RUNTIME_SCHEMA = "trace_vb_v8_train_only_monitor_runtime_v1"
VALIDATION_SCHEMA = "trace_vb_v8_validation_behavior_v1"
CANDIDATE_SCHEMA = "trace_vb_v8_candidate_v1"
EXPECTED_COORDINATES = {
    "stage1": (0, 512, 1024, 1536, 2048),
    "stage2": (0, 256, 512, 768, 1024),
}
BEHAVIOR_THRESHOLDS = {
    "valid_answer_fraction": (0.98, "minimum"),
    "unique_prediction_ratio": (0.20, "minimum"),
    "top1_mode_fraction": (0.20, "maximum"),
    "nonempty_output_fraction": (0.98, "minimum"),
}
WAITING_STANZA = re.compile(
    r"^\S+\s+(?:(?P<initial>waiting for fixed GPUs)|"
    r"waiting to resume (?P<resume_phase>\S+) on fixed GPUs)"
    r"(?:\s+(?P<fixed_gpus>.*\S))?\s*$",
    re.MULTILINE,
)
FATAL_PATTERNS = (
    ("cuda_oom", re.compile(r"(?:torch\.OutOfMemoryError|CUDA out of memory)", re.I)),
    ("child_failed", re.compile(r"\bChildFailedError\b")),
    ("process_raised", re.compile(r"\bProcessRaisedException\b")),
    (
        "nccl_timeout",
        re.compile(
            r"NCCL[^\n]{0,500}(?:watchdog|collective)[^\n]{0,500}timeout",
            re.I,
        ),
    ),
    ("traceback", re.compile(r"Traceback \(most recent call last\):")),
    ("disk_full", re.compile(r"No space left on device", re.I)),
    (
        "full_validation_failed",
        re.compile(r"full validation contract failed", re.I),
    ),
    (
        "candidate_registration_failed",
        re.compile(r"candidate registration failed", re.I),
    ),
    (
        "pipeline_nonzero_exit",
        re.compile(r"pipeline exited with status\s+[1-9][0-9]*", re.I),
    ),
)
# The bounded NCCL expression can span roughly one kilobyte.  Keep a larger
# fixed overlap so a marker split across two half-hour reads is never lost.
FATAL_OVERLAP = max(4096, max(len(pattern.pattern) for _, pattern in FATAL_PATTERNS))
INITIAL_LOG_TAIL_BYTES = 256 * 1024
MIN_DISK_FREE_BYTES = 20 * 1024**3
MIN_FREE_GPU_MIB = 21500
MAX_IDLE_GPU_UTILIZATION = 10
METRIC_SAFE_MINIMUM_CORRECT = 527
METRIC_SAFE_BASELINE_SHA256 = (
    "9b888b36956f70affae153526b716ad26343164b4aa66d6d3d54f70561144aa2"
)
REGISTERED_CAPABILITY_VALIDATION_SHA256 = (
    "59123d2bfc020335f54d70862a904e1b70e5ea1ce764483692809ae4e6772116"
)
METRIC_CONTRACTS = (
    {
        "prefix": "metric_safe_baseline",
        "source": "student_commit",
        "origin": (
            "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
            "20260818-213000_trace_vb_v7_full_seed0/student_initial_gate.json"
        ),
        "sha256": METRIC_SAFE_BASELINE_SHA256,
        "correct": 527,
        "questions": 747,
        "validation_path": "student_commit",
    },
    {
        "prefix": "registered_capability_validation",
        "source": "capability_teacher_all_roles",
        "origin": (
            "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
            "20260818-213000_trace_vb_v7_full_seed0/capability_parity_gate.json"
        ),
        "sha256": REGISTERED_CAPABILITY_VALIDATION_SHA256,
        "correct": 540,
        "questions": 747,
        "validation_path": "capability_teacher_all_roles",
    },
)


class MonitorContractError(RuntimeError):
    """An on-disk formal-run contract is malformed or inconsistent."""


@dataclass(frozen=True)
class MonitorConfig:
    tag: str
    artifact_root: Path
    physical_gpus: tuple[int, int, int, int]
    waiter_session: str
    state_dir: Path
    code_root: Path

    @property
    def supervisor_dir(self) -> Path:
        return self.artifact_root / "supervisor"

    @property
    def waiter_log(self) -> Path:
        return self.supervisor_dir / f"{self.tag}_wait.log"

    @property
    def exit_status_file(self) -> Path:
        return self.supervisor_dir / f"{self.tag}.exit_status"

    @property
    def active_attempt_file(self) -> Path:
        return self.supervisor_dir / f"{self.tag}.active_attempt.json"

    @property
    def trigger_report(self) -> Path:
        return self.supervisor_dir / f"{self.tag}.v7_trigger.json"

    @property
    def pipeline_dir(self) -> Path:
        return self.artifact_root / "pipelines" / self.tag

    @property
    def stage1_dir(self) -> Path:
        return self.artifact_root / "training" / f"{self.tag}_stage1"

    @property
    def stage2_dir(self) -> Path:
        return self.artifact_root / "training" / f"{self.tag}_stage2"


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_runtime(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": RUNTIME_SCHEMA,
            "hash_cache": {},
            "log_cursors": {},
            "stagnant_cycles": 0,
            "process_cpu_ticks": {},
            "sticky_fatal": None,
            "fatal_history": [],
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": RUNTIME_SCHEMA,
            "hash_cache": {},
            "log_cursors": {},
            "stagnant_cycles": 0,
            "process_cpu_ticks": {},
            "sticky_fatal": None,
            "fatal_history": [],
        }
    if not isinstance(value, dict) or value.get("schema_version") != RUNTIME_SCHEMA:
        return {
            "schema_version": RUNTIME_SCHEMA,
            "hash_cache": {},
            "log_cursors": {},
            "stagnant_cycles": 0,
            "process_cpu_ticks": {},
            "sticky_fatal": None,
            "fatal_history": [],
        }
    value.setdefault("hash_cache", {})
    value.setdefault("log_cursors", {})
    value.setdefault("stagnant_cycles", 0)
    value.setdefault("process_cpu_ticks", {})
    value.setdefault("sticky_fatal", None)
    value.setdefault("fatal_history", [])
    return value


def run_command(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MonitorContractError(f"missing JSON: {path}") from error
    except json.JSONDecodeError as error:
        raise MonitorContractError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise MonitorContractError(f"JSON is not an object: {path}")
    return value


def read_manifest(path: Path) -> dict[str, list[str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise MonitorContractError(f"missing manifest: {path}") from error
    values: dict[str, list[str]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line or "=" not in line:
            raise MonitorContractError(
                f"malformed manifest line {line_number}: {path}"
            )
        key, value = line.split("=", 1)
        if not key or not value:
            raise MonitorContractError(
                f"empty manifest field at line {line_number}: {path}"
            )
        values.setdefault(key, []).append(value)
    return values


def manifest_last(values: dict[str, list[str]], key: str) -> str | None:
    candidates = values.get(key, [])
    return candidates[-1] if candidates else None


def require_manifest_value(
    values: dict[str, list[str]], key: str, expected: str, errors: list[str]
) -> None:
    observed = values.get(key, [])
    if observed != [expected]:
        errors.append(f"manifest {key}={observed!r}, expected [{expected!r}]")


def lexical_absolute(path: Path) -> Path:
    """Normalize an absolute path without following a symbolic link."""

    return Path(os.path.abspath(os.fspath(path)))


def validate_metric_safe_manifest_contract(
    values: dict[str, list[str]],
    runtime: dict[str, Any],
    *,
    scope: str,
    canonical_artifacts: dict[str, Path] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate the frozen 527 student and 540 capability evidence copies."""

    errors: list[str] = []
    reports: dict[str, Any] = {}
    for contract in METRIC_CONTRACTS:
        contract_error_start = len(errors)
        prefix = str(contract["prefix"])
        expected_fields = {
            f"{prefix}_source": str(contract["source"]),
            f"{prefix}_origin": str(contract["origin"]),
            f"{prefix}_sha256": str(contract["sha256"]),
            f"{prefix}_correct": str(contract["correct"]),
            f"{prefix}_questions": str(contract["questions"]),
        }
        for key, expected in expected_fields.items():
            observed = values.get(key, [])
            if observed != [expected]:
                errors.append(
                    f"{scope} manifest {key}={observed!r}, expected [{expected!r}]"
                )
        artifact_values = values.get(f"{prefix}_artifact", [])
        artifact = Path(artifact_values[0]) if len(artifact_values) == 1 else None
        canonical = (
            canonical_artifacts.get(prefix)
            if canonical_artifacts is not None
            else None
        )
        if canonical is not None:
            canonical = lexical_absolute(canonical)
            if artifact is None or not artifact.is_absolute() or (
                lexical_absolute(artifact) != canonical
            ):
                errors.append(
                    f"{scope} manifest {prefix}_artifact is not the canonical "
                    f"local copy {canonical}"
                )
            canonical_valid = True
            if canonical.is_symlink():
                canonical_valid = False
                errors.append(
                    f"{scope} canonical local {prefix} artifact is a symbolic link"
                )
            elif not canonical.is_file():
                canonical_valid = False
                errors.append(
                    f"{scope} canonical local {prefix} artifact is missing: "
                    f"{canonical}"
                )
            elif canonical.stat().st_nlink != 1:
                canonical_valid = False
                errors.append(
                    f"{scope} canonical local {prefix} artifact is not a "
                    "single-link local copy"
                )
            if (
                canonical_valid
                and sha256_cached(canonical, runtime) != contract["sha256"]
            ):
                errors.append(
                    f"{scope} canonical local {prefix} artifact SHA256 mismatch"
                )
        artifact_valid = True
        if artifact is None or not artifact.is_absolute():
            artifact_valid = False
            errors.append(
                f"{scope} manifest has no single absolute existing {prefix}_artifact"
            )
        elif artifact.is_symlink():
            artifact_valid = False
            errors.append(
                f"{scope} manifest {prefix}_artifact is a symbolic link"
            )
        elif not artifact.is_file():
            artifact_valid = False
            errors.append(
                f"{scope} manifest has no single absolute existing {prefix}_artifact"
            )
        elif artifact.stat().st_nlink != 1:
            artifact_valid = False
            errors.append(
                f"{scope} manifest {prefix}_artifact is not a single-link local copy"
            )
        if not artifact_valid:
            reports[prefix] = {
                "artifact": str(artifact) if artifact is not None else None,
                "canonical_artifact": str(canonical) if canonical is not None else None,
                "expected_sha256": contract["sha256"],
                "verified": False,
            }
            continue
        actual_digest = sha256_cached(artifact, runtime)
        if actual_digest != contract["sha256"]:
            errors.append(f"{scope} {prefix} artifact SHA256 mismatch")
        try:
            report = read_json_object(artifact)
        except MonitorContractError as error:
            errors.append(str(error))
            reports[prefix] = {
                "artifact": str(artifact),
                "canonical_artifact": str(canonical) if canonical is not None else None,
                "expected_sha256": contract["sha256"],
                "actual_sha256": actual_digest,
                "verified": False,
            }
            continue
        if report.get("schema_version") != "trace_vb_v7_validation_behavior_v1":
            errors.append(f"{scope} {prefix} artifact has wrong schema")
        if report.get("validation_path") != contract["validation_path"]:
            errors.append(f"{scope} {prefix} artifact has wrong validation path")
        try:
            world_size = int(report.get("world_size", -1))
            questions = int(report.get("unique_questions", -1))
            correct = int(report.get("correct_count", -1))
            accuracy = float(report.get("accuracy", float("nan")))
        except (TypeError, ValueError):
            world_size = questions = correct = -1
            accuracy = float("nan")
        if world_size != 4:
            errors.append(f"{scope} {prefix} artifact world_size is not 4")
        if questions != int(contract["questions"]):
            errors.append(f"{scope} {prefix} artifact question count is not 747")
        if correct != int(contract["correct"]):
            errors.append(
                f"{scope} {prefix} artifact correct_count={correct}, "
                f"expected {contract['correct']}"
            )
        if not math.isfinite(accuracy) or not math.isclose(
            accuracy, correct / max(1, questions), rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"{scope} {prefix} artifact accuracy is inconsistent")
        if report.get("status") != "PASS" or report.get("failures") not in ([], None):
            errors.append(f"{scope} {prefix} artifact is not a passing gate")
        for key, (threshold, direction) in BEHAVIOR_THRESHOLDS.items():
            value = _finite_unit(
                report.get(key), f"{scope} {prefix} artifact {key}", errors
            )
            if direction == "minimum" and value < threshold:
                errors.append(
                    f"{scope} {prefix} artifact {key}={value}<{threshold}"
                )
            if direction == "maximum" and value > threshold:
                errors.append(
                    f"{scope} {prefix} artifact {key}={value}>{threshold}"
                )
        reports[prefix] = {
            "artifact": str(artifact),
            "canonical_artifact": str(canonical) if canonical is not None else None,
            "expected_sha256": contract["sha256"],
            "actual_sha256": actual_digest,
            "validation_path": report.get("validation_path"),
            "correct_count": correct,
            "questions": questions,
            "verified": len(errors) == contract_error_start,
        }
    return reports, errors


def sha256_cached(path: Path, runtime: dict[str, Any], *, force: bool = False) -> str:
    try:
        stat = path.stat()
    except FileNotFoundError as error:
        raise MonitorContractError(f"missing file for SHA256: {path}") from error
    key = str(path.resolve())
    cache = runtime.setdefault("hash_cache", {})
    cached = cache.get(key)
    if (
        not force
        and isinstance(cached, dict)
        and int(cached.get("size", -1)) == stat.st_size
        and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
        and isinstance(cached.get("sha256"), str)
    ):
        return str(cached["sha256"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    cache[key] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": value,
    }
    return value


def parse_waiter_stanza(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "path": str(path),
            "stanza_start_offset": 0,
            "lines": [],
            "latest_status_line": None,
            "stanza_kind": None,
            "resume_phase": None,
            "trigger": None,
            "v7_processes": None,
            "boundary": None,
            "declared_fixed_gpus": None,
            "declared_fixed_gpus_raw": None,
            "terminal_trigger_false": False,
        }
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    matches = list(WAITING_STANZA.finditer(text))
    latest_match = matches[-1] if matches else None
    start_character = latest_match.start() if latest_match is not None else 0
    stanza = text[start_character:]
    stanza_start_offset = len(text[:start_character].encode("utf-8"))
    lines = stanza.splitlines()
    fixed_gpu_text = (
        latest_match.group("fixed_gpus") if latest_match is not None else None
    )
    fixed_gpu_match = (
        re.fullmatch(r"[0-9]+(?:,[0-9]+){3}", fixed_gpu_text)
        if fixed_gpu_text is not None
        else None
    )
    declared_fixed_gpus = (
        [int(value) for value in fixed_gpu_text.split(",")]
        if fixed_gpu_match is not None
        else None
    )
    stanza_kind = (
        "resume"
        if latest_match is not None and latest_match.group("resume_phase")
        else "initial"
        if latest_match is not None
        else None
    )
    resume_phase = (
        latest_match.group("resume_phase") if latest_match is not None else None
    )
    latest_status_line = None
    trigger = None
    v7_processes = None
    boundary = None
    gpu_status: dict[str, dict[str, int]] = {}
    for line in lines:
        if "v7_processes=" in line and (
            "trigger=" in line or "boundary=" in line
        ):
            latest_status_line = line
            trigger_match = re.search(r"\btrigger=(\w+)", line)
            v7_match = re.search(r"\bv7_processes=(\w+)", line)
            boundary_match = re.search(r"\bboundary=(\w+)", line)
            trigger = trigger_match.group(1) if trigger_match else None
            v7_processes = v7_match.group(1) if v7_match else None
            boundary = boundary_match.group(1) if boundary_match else None
            gpu_status = {
                match.group(1): {
                    "free_mib": int(match.group(2)),
                    "utilization_percent": int(match.group(3)),
                }
                for match in re.finditer(r"\b(\d+):(\d+)MiB:(\d+)%", line)
            }
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": len(raw),
        "stanza_start_offset": stanza_start_offset,
        "lines": lines[-80:],
        "latest_status_line": latest_status_line,
        "stanza_kind": stanza_kind,
        "resume_phase": resume_phase,
        "declared_fixed_gpus_raw": fixed_gpu_text,
        "declared_fixed_gpus": declared_fixed_gpus,
        "trigger": trigger,
        "v7_processes": v7_processes,
        "boundary": boundary,
        "gpu_status": gpu_status,
        "launch_seen": any("launching v8 train-only pipeline" in line for line in lines),
        "terminal_trigger_false": any("terminal trigger_false" in line for line in lines),
        "pipeline_exit_lines": [
            line for line in lines if "pipeline exited with status" in line
        ][-3:],
    }


def scan_log_increment(
    path: Path,
    runtime: dict[str, Any],
    *,
    floor_offset: int = 0,
    strict_floor: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "findings": []}
    stat = path.stat()
    key = str(path.resolve())
    cursors = runtime.setdefault("log_cursors", {})
    previous = cursors.get(key, {})
    same_file = (
        isinstance(previous, dict)
        and int(previous.get("device", -1)) == stat.st_dev
        and int(previous.get("inode", -1)) == stat.st_ino
    )
    previous_offset = int(previous.get("offset", -1)) if same_file else -1
    if floor_offset <= previous_offset <= stat.st_size:
        start = previous_offset
        prior_tail = str(previous.get("tail", ""))
    else:
        start = (
            int(floor_offset)
            if strict_floor
            else max(int(floor_offset), stat.st_size - INITIAL_LOG_TAIL_BYTES)
        )
        prior_tail = ""
    findings_by_label: dict[str, dict[str, str]] = {}
    scan_tail = prior_tail
    bytes_read = 0
    with path.open("rb") as stream:
        stream.seek(start)
        while True:
            raw = stream.read(INITIAL_LOG_TAIL_BYTES)
            if not raw:
                break
            bytes_read += len(raw)
            new_text = raw.decode("utf-8", errors="replace")
            combined = scan_tail + new_text
            prior_character_count = len(scan_tail)
            for label, pattern in FATAL_PATTERNS:
                if label in findings_by_label:
                    continue
                # The retained tail exists only to join a marker split across
                # reads/chunks.  A match wholly contained in that old tail is
                # not a new occurrence.
                match = next(
                    (
                        candidate
                        for candidate in pattern.finditer(combined)
                        if candidate.end() > prior_character_count
                    ),
                    None,
                )
                if match is None:
                    continue
                excerpt_start = max(0, match.start() - 120)
                excerpt_end = min(len(combined), match.end() + 240)
                findings_by_label[label] = {
                    "pattern": label,
                    "excerpt": combined[excerpt_start:excerpt_end].replace(
                        "\x00", ""
                    ),
                }
            scan_tail = combined[-FATAL_OVERLAP:]
    findings = list(findings_by_label.values())
    final_offset = start + bytes_read
    cursors[key] = {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "offset": final_offset,
        "tail": scan_tail,
    }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": max(stat.st_size, final_offset),
        "scan_start_offset": start,
        "new_bytes": bytes_read,
        "findings": findings,
    }


def query_tmux(session: str, runner: CommandRunner) -> dict[str, Any]:
    completed = runner(
        [
            "tmux",
            "list-panes",
            "-s",
            "-t",
            f"={session}",
            "-F",
            "#{session_name}|#{pane_pid}|#{pane_dead}|#{pane_dead_status}|"
            "#{pane_current_command}|#{pane_start_command}",
        ]
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        missing = bool(
            re.search(r"(?:can't find|no such|not found|unknown).*session", stderr, re.I)
        )
        return {
            "session": session,
            "alive": False if missing else None,
            "query_ok": missing,
            "returncode": completed.returncode,
            "error": stderr or completed.stdout.strip(),
            "panes": [],
        }
    panes = []
    for line in completed.stdout.splitlines():
        fields = line.split("|", 5)
        if len(fields) != 6:
            raise MonitorContractError(f"unexpected tmux pane row: {line!r}")
        panes.append(
            {
                "session_name": fields[0],
                "pane_pid": int(fields[1]),
                "pane_dead": fields[2] == "1",
                "pane_dead_status": fields[3] or None,
                "current_command": fields[4],
                "start_command": fields[5],
            }
        )
    exact = bool(panes) and all(item["session_name"] == session for item in panes)
    alive = exact and any(not item["pane_dead"] for item in panes)
    return {
        "session": session,
        "alive": alive,
        "query_ok": True,
        "returncode": 0,
        "exact_session_match": exact,
        "panes": panes,
    }


def _read_process(pid: int, code_root: Path) -> dict[str, Any] | None:
    process_root = Path("/proc") / str(pid)
    try:
        raw_args = (process_root / "cmdline").read_bytes()
        args = [
            token.decode("utf-8", errors="replace")
            for token in raw_args.split(b"\0")
            if token
        ]
        cwd = (process_root / "cwd").resolve()
        stat_fields = (process_root / "stat").read_text(encoding="utf-8").rsplit(
            ") ", 1
        )[1].split()
        state = stat_fields[0]
        cpu_ticks = int(stat_fields[11]) + int(stat_fields[12])
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, IndexError):
        return None
    run_script = code_root / "run.py"
    exact_run_arg = any(
        arg == "run.py"
        or (
            arg.endswith("/run.py")
            and Path(arg).resolve() == run_script.resolve()
        )
        for arg in args
    )
    return {
        "pid": pid,
        "state": state,
        "cwd": str(cwd),
        "argv": args,
        "cpu_ticks": cpu_ticks,
        "exact_v8_rank": cwd == code_root.resolve() and exact_run_arg and state != "Z",
    }


def query_gpus(
    physical_gpus: Sequence[int], code_root: Path, runner: CommandRunner
) -> dict[str, Any]:
    completed = runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if completed.returncode != 0:
        raise MonitorContractError(
            "nvidia-smi GPU query failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    requested = set(physical_gpus)
    rows = []
    uuid_to_index: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise MonitorContractError(f"unexpected nvidia-smi GPU row: {line!r}")
        index = int(fields[0])
        if index not in requested:
            continue
        uuid_to_index[fields[1]] = index
        rows.append(
            {
                "index": index,
                "uuid": fields[1],
                "total_mib": int(fields[2]),
                "used_mib": int(fields[3]),
                "free_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    rows.sort(key=lambda item: physical_gpus.index(item["index"]))
    if [item["index"] for item in rows] != list(physical_gpus):
        raise MonitorContractError(
            f"nvidia-smi did not return requested GPUs {list(physical_gpus)}"
        )
    applications: list[dict[str, Any]] = []
    process_query = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if process_query.returncode == 0:
        process_cache: dict[int, dict[str, Any] | None] = {}
        for line in process_query.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3 or fields[0] not in uuid_to_index:
                continue
            try:
                pid = int(fields[1])
                used_memory = int(fields[2])
            except ValueError:
                continue
            if pid not in process_cache:
                process_cache[pid] = _read_process(pid, code_root)
            process = process_cache[pid]
            applications.append(
                {
                    "gpu_index": uuid_to_index[fields[0]],
                    "pid": pid,
                    "used_memory_mib": used_memory,
                    "process": process,
                }
            )
    exact_pids = sorted(
        {
            int(item["pid"])
            for item in applications
            if isinstance(item.get("process"), dict)
            and bool(item["process"].get("exact_v8_rank"))
        }
    )
    ready = all(
        item["free_mib"] >= MIN_FREE_GPU_MIB
        and item["utilization_percent"] <= MAX_IDLE_GPU_UTILIZATION
        for item in rows
    )
    return {
        "physical_gpus": list(physical_gpus),
        "rows": rows,
        "all_launch_eligible": ready,
        "compute_query_ok": process_query.returncode == 0,
        "applications": applications,
        "exact_v8_rank_pids": exact_pids,
    }


def disk_snapshot(path: Path) -> dict[str, Any]:
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    statvfs = os.statvfs(target)
    return {
        "path": str(target.resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": usage.free / 1024**3,
        "inode_total": statvfs.f_files,
        "inode_free": statvfs.f_favail,
        "minimum_free_bytes": MIN_DISK_FREE_BYTES,
        "enough_free_space": usage.free >= MIN_DISK_FREE_BYTES,
    }


def candidate_filename(phase: str, coordinate: int) -> str:
    prefix = "candidate_step" if phase == "stage1" else "candidate_rollout"
    return f"{prefix}_{coordinate:04d}.json"


def summary_filename(phase: str, coordinate: int) -> str:
    prefix = "validation_step" if phase == "stage1" else "validation_rollout"
    return f"{prefix}_{coordinate:04d}.json"


def _finite_unit(value: Any, label: str, errors: list[str]) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} is not numeric")
        return float("nan")
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        errors.append(f"{label} is outside [0,1]")
    return result


def validate_candidate(
    path: Path,
    phase: str,
    coordinate: int,
    runtime: dict[str, Any],
    expected_physical_gpus: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        record = read_json_object(path)
    except MonitorContractError as error:
        return None, [str(error)]
    prefix = f"{phase} candidate {coordinate}"
    if record.get("schema_version") != CANDIDATE_SCHEMA:
        errors.append(f"{prefix} has wrong schema")
    if record.get("phase") != phase:
        errors.append(f"{prefix} has wrong phase")
    if record.get("physical_gpus") != expected_physical_gpus:
        errors.append(
            f"{prefix} physical_gpus={record.get('physical_gpus')!r}, "
            f"expected {expected_physical_gpus!r}"
        )
    try:
        step = int(record.get("step", -1))
        selection_step = int(record.get("selection_step", -1))
        source_step = int(record.get("source_global_step", -1))
    except (TypeError, ValueError):
        step = selection_step = source_step = -1
        errors.append(f"{prefix} has invalid step coordinates")
    phase_input = bool(record.get("phase_input", False))
    if phase == "stage1":
        if (step, selection_step, source_step) != (coordinate, coordinate, coordinate):
            errors.append(f"{prefix} step coordinates are inconsistent")
        if phase_input:
            errors.append(f"{prefix} cannot be a phase input")
        if record.get("rollout_batches") is not None:
            errors.append(f"{prefix} unexpectedly declares rollout batches")
    else:
        try:
            rollout_batches = int(record.get("rollout_batches", -1))
        except (TypeError, ValueError):
            rollout_batches = -1
        if rollout_batches != coordinate:
            errors.append(f"{prefix} has wrong rollout coordinate")
        if phase_input != (coordinate == 0):
            errors.append(f"{prefix} has wrong phase-input marker")
        expected_selection = 0 if coordinate == 0 else source_step
        if source_step < 0 or step != expected_selection or selection_step != expected_selection:
            errors.append(f"{prefix} selection/global-step coordinates are inconsistent")

    summary_path = Path(str(record.get("summary", ""))).expanduser()
    expected_summary = path.parent / summary_filename(phase, coordinate)
    if not summary_path.is_absolute() or summary_path.resolve() != expected_summary.resolve():
        errors.append(f"{prefix} summary path is not the registered candidate copy")
    summary: dict[str, Any] | None = None
    if not summary_path.is_file():
        errors.append(f"{prefix} summary is missing: {summary_path}")
    else:
        expected_digest = str(record.get("summary_sha256", ""))
        actual_digest = sha256_cached(summary_path, runtime)
        if actual_digest != expected_digest:
            errors.append(f"{prefix} summary SHA256 mismatch")
        try:
            summary = read_json_object(summary_path)
        except MonitorContractError as error:
            errors.append(str(error))
    if summary is not None:
        if summary.get("schema_version") != VALIDATION_SCHEMA:
            errors.append(f"{prefix} validation has wrong schema")
        if summary.get("validation_path") != "student_commit":
            errors.append(f"{prefix} validation path is not student_commit")
        try:
            world_size = int(summary.get("world_size", -1))
            questions = int(summary.get("unique_questions", -1))
            summary_step = int(summary.get("global_step", -1))
            correct_count = int(summary.get("correct_count", -1))
            accuracy = float(summary.get("accuracy", float("nan")))
        except (TypeError, ValueError):
            world_size = questions = summary_step = correct_count = -1
            accuracy = float("nan")
            errors.append(f"{prefix} validation contains nonnumeric contract fields")
        if world_size != 4:
            errors.append(f"{prefix} validation world_size={world_size}, expected 4")
        if questions != 747:
            errors.append(f"{prefix} validation questions={questions}, expected 747")
        if summary_step != source_step:
            errors.append(f"{prefix} summary global step differs from candidate")
        if not 0 <= correct_count <= 747:
            errors.append(f"{prefix} has invalid correct_count")
        if not math.isfinite(accuracy) or not math.isclose(
            accuracy, correct_count / 747, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"{prefix} accuracy is not exact correct_count/747")
        try:
            record_correct = int(record.get("correct_count", -1))
            record_accuracy = float(record.get("accuracy", float("nan")))
        except (TypeError, ValueError):
            record_correct = -1
            record_accuracy = float("nan")
        if record_correct != correct_count or record_accuracy != accuracy:
            errors.append(f"{prefix} summary and candidate metrics disagree")
        record_behavior = record.get("behavior", {})
        if not isinstance(record_behavior, dict):
            record_behavior = {}
            errors.append(f"{prefix} has invalid behavior object")
        for key in BEHAVIOR_THRESHOLDS:
            summary_value = _finite_unit(summary.get(key), f"{prefix} summary {key}", errors)
            record_value = _finite_unit(record_behavior.get(key), f"{prefix} record {key}", errors)
            if summary_value != record_value:
                errors.append(f"{prefix} summary and record {key} disagree")

    checkpoint_path = Path(str(record.get("checkpoint", ""))).expanduser()
    if not checkpoint_path.is_absolute() or not checkpoint_path.is_file():
        errors.append(f"{prefix} checkpoint path is missing/non-absolute: {checkpoint_path}")
    else:
        expected_digest = str(record.get("checkpoint_sha256", ""))
        actual_digest = sha256_cached(checkpoint_path, runtime)
        if actual_digest != expected_digest:
            errors.append(f"{prefix} checkpoint SHA256 mismatch")
    compact = {
        "coordinate": coordinate,
        "record": str(path.resolve()),
        "source_global_step": source_step,
        "phase_input": phase_input,
        "physical_gpus": record.get("physical_gpus"),
        "correct_count": record.get("correct_count"),
        "accuracy": record.get("accuracy"),
        "summary": str(summary_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": record.get("checkpoint_sha256"),
        "behavior": record.get("behavior"),
    }
    return compact, errors


def inspect_candidates(
    stage_dir: Path,
    phase: str,
    runtime: dict[str, Any],
    expected_physical_gpus: str,
) -> dict[str, Any]:
    candidate_dir = stage_dir / "candidates"
    expected = EXPECTED_COORDINATES[phase]
    prefix = "candidate_step_" if phase == "stage1" else "candidate_rollout_"
    errors: list[str] = []
    observed_paths = sorted(candidate_dir.glob(f"{prefix}*.json")) if candidate_dir.is_dir() else []
    coordinate_pattern = re.compile(rf"^{re.escape(prefix)}(\d{{4}})\.json$")
    by_coordinate: dict[int, Path] = {}
    for path in observed_paths:
        match = coordinate_pattern.match(path.name)
        if match is None:
            errors.append(f"unexpected candidate filename: {path}")
            continue
        coordinate = int(match.group(1))
        if coordinate not in expected:
            errors.append(f"unexpected {phase} candidate coordinate: {coordinate}")
            continue
        if coordinate in by_coordinate:
            errors.append(f"duplicate {phase} candidate coordinate: {coordinate}")
            continue
        by_coordinate[coordinate] = path
    observed = [coordinate for coordinate in expected if coordinate in by_coordinate]
    if observed != list(expected[: len(observed)]):
        errors.append(f"{phase} candidates are not a contiguous prefix: {observed}")
    records = []
    per_coordinate_valid: dict[int, bool] = {}
    for coordinate in observed:
        compact, candidate_errors = validate_candidate(
            by_coordinate[coordinate],
            phase,
            coordinate,
            runtime,
            expected_physical_gpus,
        )
        if compact is not None:
            records.append(compact)
        per_coordinate_valid[coordinate] = compact is not None and not candidate_errors
        errors.extend(candidate_errors)
    if phase == "stage2":
        trained_steps = [
            int(item["source_global_step"])
            for item in records
            if int(item["coordinate"]) > 0
        ]
        if any(
            current <= previous
            for previous, current in zip(trained_steps, trained_steps[1:])
        ):
            errors.append(
                f"stage2 trained global steps are not strictly increasing: {trained_steps}"
            )
    # A fatal state may only be cleared by a *fully registered* validation
    # boundary.  Expose a fail-closed contiguous prefix distinct from merely
    # observed filenames so corrupt or wrong-GPU records cannot act as a
    # recovery signal.
    verified_coordinates: list[int] = []
    if not errors:
        for coordinate in expected:
            if coordinate not in by_coordinate or not per_coordinate_valid.get(
                coordinate, False
            ):
                break
            verified_coordinates.append(coordinate)
    return {
        "phase": phase,
        "candidate_directory": str(candidate_dir),
        "expected_coordinates": list(expected),
        "observed_coordinates": observed,
        "verified_coordinates": verified_coordinates,
        "validated_candidate_count": len(records),
        "complete_candidate_set": observed == list(expected) and not errors,
        "candidates": records,
        "errors": errors,
    }


def candidate_is_behavior_eligible(item: dict[str, Any]) -> bool:
    behavior = item.get("behavior", {})
    if not isinstance(behavior, dict):
        return False
    for key, (threshold, direction) in BEHAVIOR_THRESHOLDS.items():
        try:
            value = float(behavior[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        if direction == "minimum" and value < threshold:
            return False
        if direction == "maximum" and value > threshold:
            return False
    return True


def validate_candidate_index(
    stage_dir: Path,
    phase: str,
    candidate_report: dict[str, Any],
    expected_physical_gpus: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    index_path = stage_dir / "candidate_index.json"
    best_path = stage_dir / "best_checkpoint.txt"
    if not index_path.exists() and not best_path.exists():
        return None, []
    errors: list[str] = []
    if not index_path.is_file():
        return None, [f"missing candidate index: {index_path}"]
    try:
        index = read_json_object(index_path)
    except MonitorContractError as error:
        return None, [str(error)]
    expected_schema = f"trace_vb_v8_{phase}_candidate_index_v1"
    if index.get("schema_version") != expected_schema:
        errors.append(f"{phase} index has wrong schema")
    if index.get("phase") != phase:
        errors.append(f"{phase} index has wrong phase")
    if index.get("physical_gpus") != expected_physical_gpus:
        errors.append(
            f"{phase} index physical_gpus={index.get('physical_gpus')!r}, "
            f"expected {expected_physical_gpus!r}"
        )
    expected = list(EXPECTED_COORDINATES[phase])
    if int(index.get("candidate_count", -1)) != 5:
        errors.append(f"{phase} index candidate_count is not 5")
    candidates = candidate_report.get("candidates", [])
    if candidate_report.get("observed_coordinates") != expected or len(candidates) != 5:
        errors.append(f"{phase} index exists before the complete candidate set")
    index_candidates = index.get("candidates", [])
    if not isinstance(index_candidates, list) or len(index_candidates) != 5:
        errors.append(f"{phase} index does not contain five candidates")
    else:
        index_coordinates = [
            int(
                item.get("step", -1)
                if phase == "stage1"
                else item.get("rollout_batches", -1)
            )
            for item in index_candidates
        ]
        if index_coordinates != expected:
            errors.append(f"{phase} index coordinates are {index_coordinates}")
        by_coordinate = {int(item["coordinate"]): item for item in candidates}
        for coordinate, indexed in zip(expected, index_candidates):
            current = by_coordinate.get(coordinate)
            if current is None:
                continue
            for key in ("correct_count", "accuracy", "checkpoint", "checkpoint_sha256"):
                if indexed.get(key) != current.get(key):
                    errors.append(
                        f"{phase} index candidate {coordinate} changed field {key}"
                    )
    eligible = [item for item in candidates if candidate_is_behavior_eligible(item)]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                int(item["correct_count"]),
                -int(item["coordinate"]),
            ),
        )
        selected_coordinate = int(selected["coordinate"])
        observed_coordinate = int(
            index.get(
                "selected_step" if phase == "stage1" else "selected_rollout_batches",
                -1,
            )
        )
        if observed_coordinate != selected_coordinate:
            errors.append(
                f"{phase} index selected coordinate {observed_coordinate}, "
                f"expected {selected_coordinate}"
            )
        if index.get("selected_checkpoint") != selected["checkpoint"]:
            errors.append(f"{phase} index selected checkpoint is inconsistent")
        if int(index.get("selected_correct_count", -1)) != int(selected["correct_count"]):
            errors.append(f"{phase} index selected correct_count is inconsistent")
    else:
        errors.append(f"{phase} index has no behavior-eligible candidate")
    if not best_path.is_file():
        errors.append(f"missing selected checkpoint record: {best_path}")
    else:
        best_value = best_path.read_text(encoding="utf-8").strip()
        if selected is not None and Path(best_value).resolve() != Path(
            selected["checkpoint"]
        ).resolve():
            errors.append(f"{phase} best_checkpoint.txt is inconsistent")
    compact = {
        "path": str(index_path),
        "schema_version": index.get("schema_version"),
        "physical_gpus": index.get("physical_gpus"),
        "selected_coordinate": (
            selected.get("coordinate") if selected is not None else None
        ),
        "selected_correct_count": (
            selected.get("correct_count") if selected is not None else None
        ),
        "selected_accuracy": selected.get("accuracy") if selected is not None else None,
        "selected_checkpoint": selected.get("checkpoint") if selected is not None else None,
    }
    return compact, errors


def inspect_stage(
    stage_dir: Path,
    phase: str,
    runtime: dict[str, Any],
    expected_physical_gpus: str,
    canonical_metric_dir: Path,
) -> dict[str, Any]:
    report = inspect_candidates(
        stage_dir, phase, runtime, expected_physical_gpus
    )
    manifest_path = stage_dir / "manifest.txt"
    manifest = None
    metric_contract = None
    errors = list(report.pop("errors"))
    finished = False
    if manifest_path.is_file():
        try:
            manifest = read_manifest(manifest_path)
        except MonitorContractError as error:
            errors.append(str(error))
        if manifest is not None:
            require_manifest_value(manifest, "model", "TRACE-VB-v8", errors)
            require_manifest_value(manifest, "test_split", "false", errors)
            require_manifest_value(manifest, "evidence_pipeline", "false", errors)
            for field in ("physical_gpus", "formal_gpus", "fixed_gpus"):
                require_manifest_value(
                    manifest, field, expected_physical_gpus, errors
                )
            metric_contract, metric_errors = validate_metric_safe_manifest_contract(
                manifest,
                runtime,
                scope=phase,
                canonical_artifacts={
                    "metric_safe_baseline": (
                        canonical_metric_dir / "metric_safe_baseline.json"
                    ),
                    "registered_capability_validation": (
                        canonical_metric_dir
                        / "registered_capability_validation.json"
                    ),
                },
            )
            errors.extend(metric_errors)
            finished = bool(manifest_last(manifest, "finished_at"))
    index, index_errors = validate_candidate_index(
        stage_dir, phase, report, expected_physical_gpus
    )
    errors.extend(index_errors)
    if finished:
        if not report["complete_candidate_set"]:
            errors.append(f"{phase} manifest finished before five valid candidates")
        if index is None:
            errors.append(f"{phase} manifest finished without candidate index")
        if not (stage_dir / "last_checkpoint.txt").is_file():
            errors.append(f"{phase} manifest finished without last_checkpoint.txt")
    report.update(
        {
            "directory": str(stage_dir),
            "exists": stage_dir.is_dir(),
            "manifest": str(manifest_path) if manifest_path.is_file() else None,
            "manifest_finished": finished,
            "resume_attempts": (
                list(manifest.get("resume_attempt", []))
                if manifest is not None
                else []
            ),
            "metric_safe_contract": metric_contract,
            "candidate_index": index,
            "errors": errors,
        }
    )
    return report


def parse_exit_status(path: Path) -> tuple[int | None, list[str]]:
    if not path.exists():
        return None, []
    try:
        raw = path.read_text(encoding="ascii").strip()
        status = int(raw)
    except (OSError, ValueError) as error:
        return None, [f"invalid supervisor exit status {path}: {error}"]
    if raw != str(status) or status < 0 or status > 255:
        return None, [f"invalid supervisor exit status value: {raw!r}"]
    return status, []


def _validate_pipeline_resume_binding_path(
    config: MonitorConfig,
    phase: str,
    path: Path,
    runtime: dict[str, Any],
    *,
    attempt_id: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    expected_gpu_csv = ",".join(map(str, config.physical_gpus))
    if not path.is_absolute() or path.parent.resolve() != config.pipeline_dir.resolve():
        return None, ["resume boundary binding path is not inside the pipeline directory"]
    expected_name = (
        f"resume_binding_{phase}_{attempt_id}.json"
        if attempt_id is not None
        else None
    )
    if expected_name is not None and path.name != expected_name:
        errors.append(
            f"resume boundary binding filename {path.name!r}, "
            f"expected {expected_name!r}"
        )
    elif re.fullmatch(rf"resume_binding_{re.escape(phase)}_[A-Za-z0-9._-]+\.json", path.name) is None:
        errors.append("resume boundary binding has an unsafe filename")
    try:
        value = read_json_object(path)
    except MonitorContractError as error:
        return None, errors + [str(error)]
    required = {
        "schema_version": "trace_vb_v8_pipeline_resume_binding_v1",
        "status": "PASS",
        "pipeline_tag": config.tag,
        "phase": phase,
        "physical_gpus": expected_gpu_csv,
        "formal_gpus": expected_gpu_csv,
        "fixed_gpus": expected_gpu_csv,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            errors.append(
                f"resume boundary binding {key}={value.get(key)!r}, "
                f"expected {expected!r}"
            )
    checkpoint = Path(str(value.get("validated_boundary_checkpoint", "")))
    if not checkpoint.is_absolute() or not checkpoint.is_file():
        errors.append("resume boundary binding checkpoint is missing/non-absolute")
    else:
        expected_digest = str(value.get("validated_boundary_checkpoint_sha256", ""))
        actual_digest = sha256_cached(checkpoint, runtime)
        if actual_digest != expected_digest:
            errors.append("resume boundary binding checkpoint SHA256 mismatch")
    candidate = Path(str(value.get("latest_candidate_record", "")))
    expected_candidate_dir = (
        config.stage1_dir if phase == "stage1" else config.stage2_dir
    ) / "candidates"
    if (
        not candidate.is_absolute()
        or not candidate.is_file()
        or candidate.parent.resolve() != expected_candidate_dir.resolve()
    ):
        errors.append("resume boundary binding candidate record is missing/out of scope")
    return (
        {
            "path": str(path.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "candidate": str(candidate.resolve()),
        },
        errors,
    )


def _validate_pipeline_resume_binding(
    config: MonitorConfig,
    phase: str,
    checkpoint: Path,
    runtime: dict[str, Any],
) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    matches = []
    for path in sorted(config.pipeline_dir.glob(f"resume_binding_{phase}_*.json")):
        compact, binding_errors = _validate_pipeline_resume_binding_path(
            config, phase, path, runtime
        )
        if binding_errors or compact is None:
            continue
        if Path(compact["checkpoint"]).resolve() == checkpoint.resolve():
            matches.append(path)
    if len(matches) != 1:
        errors.append(
            f"resume {phase} has {len(matches)} matching validated pipeline "
            "boundary bindings, expected exactly one"
        )
        return None, errors
    return str(matches[0].resolve()), errors


def _latest_verified_candidate(report: dict[str, Any]) -> dict[str, Any] | None:
    verified = [int(value) for value in report.get("verified_coordinates", [])]
    if not verified:
        return None
    by_coordinate = {
        int(item["coordinate"]): item for item in report.get("candidates", [])
    }
    return by_coordinate.get(verified[-1])


def _validate_stage_resume_contract(
    config: MonitorConfig,
    phase: str,
    attempt_id: str,
    report: dict[str, Any],
) -> tuple[str | None, Path | None, list[str]]:
    errors: list[str] = []
    stage_dir = config.stage1_dir if phase == "stage1" else config.stage2_dir
    contract_path = stage_dir / f"resume_checkpoint_contract_{attempt_id}.json"
    try:
        contract = read_json_object(contract_path)
    except MonitorContractError as error:
        return str(contract_path), None, [str(error)]
    expected_gpu_csv = ",".join(map(str, config.physical_gpus))
    required = {
        "schema_version": "trace_vb_v8_resume_checkpoint_contract_v1",
        "status": "PASS",
        "physical_gpus": expected_gpu_csv,
        "requested_stage": 1 if phase == "stage1" else 2,
    }
    for key, expected in required.items():
        observed = contract.get(key)
        if key == "requested_stage":
            try:
                observed = int(observed)
            except (TypeError, ValueError):
                observed = None
        if observed != expected:
            errors.append(
                f"resume checkpoint contract {key}={observed!r}, expected {expected!r}"
            )
    checkpoint = Path(str(contract.get("checkpoint", "")))
    completed_candidate = Path(str(contract.get("completed_candidate", "")))
    latest = _latest_verified_candidate(report)
    if not checkpoint.is_absolute() or not checkpoint.is_file():
        errors.append("resume checkpoint contract checkpoint is missing/non-absolute")
    if not completed_candidate.is_absolute() or not completed_candidate.is_file():
        errors.append("resume checkpoint contract candidate is missing/non-absolute")
    if latest is None:
        errors.append("resume checkpoint contract has no contiguous verified candidate")
    else:
        if checkpoint.resolve() != Path(str(latest["checkpoint"])).resolve():
            errors.append("resume checkpoint is not the latest verified candidate boundary")
        if completed_candidate.resolve() != Path(str(latest["record"])).resolve():
            errors.append("resume completed_candidate is not the latest verified record")
    return str(contract_path.resolve()), checkpoint if checkpoint.is_file() else None, errors


def resolve_supervisor_attempt(
    config: MonitorConfig,
    waiter: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Resolve an initial or validated-boundary recovery attempt.

    A future recovery supervisor can publish ``<tag>.active_attempt.json`` with
    schema ``trace_vb_v8_supervisor_attempt_v1``.  Until that exists, a resume
    attempt deliberately ignores the legacy initial ``<tag>.exit_status``.
    """

    errors: list[str] = []
    warnings: list[str] = []
    expected_gpu_csv = ",".join(map(str, config.physical_gpus))
    detected_resumes: list[tuple[str, str]] = []
    for phase, report in (
        ("stage1", stage1),
        ("stage2", stage2),
    ):
        for value in report.get("resume_attempts", []):
            if re.fullmatch(r"[A-Za-z0-9._-]+", str(value)) is None:
                errors.append(f"unsafe {phase} resume attempt identifier: {value!r}")
                continue
            detected_resumes.append((str(value), phase))
    detected_resume = (
        max(detected_resumes, key=lambda item: item[0])
        if detected_resumes
        else None
    )

    active_path = config.active_attempt_file
    active = None
    if active_path.is_file():
        try:
            active = read_json_object(active_path)
        except MonitorContractError as error:
            errors.append(str(error))
        if active is not None:
            active_error_start = len(errors)
            required = {
                "schema_version": "trace_vb_v8_supervisor_attempt_v1",
                "pipeline_tag": config.tag,
                "physical_gpus": expected_gpu_csv,
                "formal_gpus": expected_gpu_csv,
                "fixed_gpus": expected_gpu_csv,
            }
            for key, expected in required.items():
                if active.get(key) != expected:
                    errors.append(
                        f"active supervisor attempt {key}={active.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            attempt_id = str(active.get("attempt_id", ""))
            attempt_kind = str(active.get("attempt_kind", ""))
            if re.fullmatch(r"[A-Za-z0-9._-]+", attempt_id) is None:
                errors.append("active supervisor attempt has unsafe attempt_id")
            if attempt_kind not in ("initial", "resume_stage1", "resume_stage2"):
                errors.append("active supervisor attempt has invalid attempt_kind")
            active_status = str(active.get("status", ""))
            if active_status not in ("WAITING", "RUNNING", "SUCCEEDED", "FAILED"):
                errors.append("active supervisor attempt has invalid status")
            declared_exit_status = active.get("exit_status")
            if active_status in ("WAITING", "RUNNING"):
                if declared_exit_status is not None:
                    errors.append(
                        f"active supervisor attempt status {active_status} "
                        "unexpectedly declares exit_status"
                    )
            elif active_status == "SUCCEEDED":
                if declared_exit_status != 0:
                    errors.append(
                        "SUCCEEDED active supervisor attempt does not declare exit_status=0"
                    )
            elif active_status == "FAILED":
                if (
                    type(declared_exit_status) is not int
                    or not 1 <= declared_exit_status <= 255
                ):
                    errors.append(
                        "FAILED active supervisor attempt has invalid declared exit_status"
                    )
            declared_tmux_session = str(active.get("tmux_session", ""))
            if re.fullmatch(r"[A-Za-z0-9._-]+", declared_tmux_session) is None:
                errors.append("active supervisor attempt has unsafe tmux_session")
            if declared_tmux_session != config.waiter_session:
                errors.append(
                    "active supervisor attempt tmux_session="
                    f"{declared_tmux_session!r}, expected configured waiter session "
                    f"{config.waiter_session!r}"
                )
            # Never let an untrusted active record redirect the liveness query
            # to an unrelated-but-live session.
            tmux_session = config.waiter_session
            expected_log_paths = {
                str((config.stage1_dir / "train.log").resolve()),
                str((config.stage2_dir / "train.log").resolve()),
            }
            raw_log_floors = active.get("log_byte_offset_floors")
            log_byte_offset_floors: dict[str, int] | None = None
            if not isinstance(raw_log_floors, dict):
                errors.append(
                    "active supervisor attempt lacks log_byte_offset_floors object"
                )
            elif set(raw_log_floors) != expected_log_paths:
                errors.append(
                    "active supervisor attempt log floor paths differ from exact "
                    "Stage train.log paths"
                )
            else:
                parsed_log_floors: dict[str, int] = {}
                for log_path, byte_offset in raw_log_floors.items():
                    if type(byte_offset) is not int or byte_offset < 0:
                        errors.append(
                            f"active supervisor attempt log floor for {log_path} "
                            "is not a non-negative integer"
                        )
                    else:
                        parsed_log_floors[log_path] = byte_offset
                if len(parsed_log_floors) == 2:
                    log_byte_offset_floors = parsed_log_floors
            status_path = Path(str(active.get("exit_status_file", "")))
            expected_status_path = config.supervisor_dir / (
                f"{config.tag}.{attempt_id}.exit_status"
            )
            if not status_path.is_absolute() or status_path.resolve() != expected_status_path.resolve():
                errors.append("active supervisor attempt has unsafe exit_status_file")
            boundary_path = active.get("validated_boundary_contract")
            active_boundary_validated = False
            resume_contract_path = None
            resume_binding_path = None
            resume_phase = active.get("resume_phase")
            if attempt_kind == "initial":
                if resume_phase is not None:
                    errors.append("initial supervisor attempt unexpectedly has resume_phase")
                if boundary_path is not None:
                    errors.append(
                        "initial supervisor attempt unexpectedly has a boundary contract"
                    )
            elif attempt_kind.startswith("resume_"):
                expected_phase = attempt_kind.removeprefix("resume_")
                if resume_phase != expected_phase:
                    errors.append("active supervisor attempt has inconsistent resume_phase")
                if (
                    waiter.get("stanza_kind") == "resume"
                    and waiter.get("resume_phase") != expected_phase
                ):
                    errors.append(
                        "active supervisor attempt differs from latest resume waiter stanza"
                    )
                binding_error_start = len(errors)
                if not isinstance(boundary_path, str):
                    errors.append(
                        "active resume supervisor attempt lacks a boundary contract"
                    )
                    binding = None
                else:
                    binding, binding_errors = _validate_pipeline_resume_binding_path(
                        config,
                        expected_phase,
                        Path(boundary_path),
                        runtime,
                        attempt_id=attempt_id,
                    )
                    errors.extend(binding_errors)
                    if binding is not None:
                        resume_binding_path = binding["path"]
                phase_report = stage1 if expected_phase == "stage1" else stage2
                latest = _latest_verified_candidate(phase_report)
                if binding is not None:
                    if latest is None:
                        errors.append(
                            "active resume boundary has no contiguous verified candidate"
                        )
                    elif (
                        Path(binding["checkpoint"]).resolve()
                        != Path(str(latest["checkpoint"])).resolve()
                        or Path(binding["candidate"]).resolve()
                        != Path(str(latest["record"])).resolve()
                    ):
                        errors.append(
                            "active resume boundary is not the latest verified candidate"
                        )
                # The active attempt is written after the pipeline-boundary
                # audit but before the Stage script starts.  WAITING therefore
                # must not require a Stage resume record that cannot exist yet.
                if active_status != "WAITING":
                    matching_stage_attempts = [
                        item
                        for item in detected_resumes
                        if item == (attempt_id, expected_phase)
                    ]
                    if active_status == "RUNNING" and not matching_stage_attempts:
                        # The waiter transitions atomically to RUNNING before
                        # invoking the Stage wrapper.  Heavy checkpoint/data
                        # audits precede the Stage resume contract, so the
                        # already-validated pipeline binding remains the
                        # authoritative boundary during this bounded startup
                        # window.
                        warnings.append(
                            "RUNNING resume has not published its Stage resume "
                            "contract yet; validated pipeline binding remains "
                            "authoritative"
                        )
                    elif len(matching_stage_attempts) != 1:
                        errors.append(
                            "active resume attempt does not have exactly one matching "
                            "Stage resume record"
                        )
                    else:
                        (
                            resume_contract_path,
                            stage_checkpoint,
                            contract_errors,
                        ) = _validate_stage_resume_contract(
                            config,
                            expected_phase,
                            attempt_id,
                            phase_report,
                        )
                        errors.extend(contract_errors)
                        if (
                            binding is not None
                            and stage_checkpoint is not None
                            and stage_checkpoint.resolve()
                            != Path(binding["checkpoint"]).resolve()
                        ):
                            errors.append(
                                "Stage resume checkpoint differs from pipeline boundary"
                            )
                active_boundary_validated = len(errors) == binding_error_start
            active_attempt_valid = len(errors) == active_error_start
            return (
                {
                    "protocol": "explicit_attempt_v1",
                    "attempt_id": attempt_id,
                    "attempt_kind": attempt_kind,
                    "identity": f"{attempt_kind}:{attempt_id}",
                    "active_attempt_file": str(active_path),
                    "exit_status_file": str(status_path),
                    "exit_status_authoritative": active_attempt_valid,
                    "status": active_status,
                    "declared_exit_status": declared_exit_status,
                    "tmux_session": tmux_session,
                    "declared_tmux_session": declared_tmux_session,
                    "log_byte_offset_floors": log_byte_offset_floors,
                    "validated_boundary": active_boundary_validated,
                    "resume_checkpoint_contract": resume_contract_path,
                    "resume_pipeline_binding": resume_binding_path,
                    "clears_prior_fatal": (
                        active_attempt_valid and active_boundary_validated
                    ),
                },
                errors,
                warnings,
            )

    if active_path.exists():
        # Never fall back to the legacy initial status when an explicit active
        # attempt exists but is malformed; that would reintroduce stale-status
        # completion after a recovery.
        return (
            {
                "protocol": "invalid_explicit_attempt",
                "attempt_id": None,
                "attempt_kind": None,
                "identity": "invalid_explicit_attempt",
                "active_attempt_file": str(active_path),
                "exit_status_file": None,
                "exit_status_authoritative": False,
                "status": None,
                "tmux_session": config.waiter_session,
                "validated_boundary": False,
                "resume_checkpoint_contract": None,
                "resume_pipeline_binding": None,
                "clears_prior_fatal": False,
            },
            errors,
            warnings,
        )

    if waiter.get("stanza_kind") == "resume" and detected_resume is None:
        errors.append(
            "resume waiter stanza has no explicit active supervisor attempt"
        )
        return (
            {
                "protocol": "resume_waiter_without_active_attempt",
                "attempt_id": None,
                "attempt_kind": f"resume_{waiter.get('resume_phase')}",
                "identity": (
                    f"resume_{waiter.get('resume_phase')}:missing_active_attempt"
                ),
                "active_attempt_file": None,
                "exit_status_file": None,
                "exit_status_authoritative": False,
                "status": None,
                "tmux_session": config.waiter_session,
                "validated_boundary": False,
                "resume_checkpoint_contract": None,
                "resume_pipeline_binding": None,
                "clears_prior_fatal": False,
                "ignored_legacy_exit_status_file": str(config.exit_status_file),
            },
            errors,
            warnings,
        )

    if detected_resume is not None:
        resume_id, resume_phase = detected_resume
        resume_contract_path, checkpoint, contract_errors = (
            _validate_stage_resume_contract(
                config,
                resume_phase,
                resume_id,
                stage1 if resume_phase == "stage1" else stage2,
            )
        )
        errors.extend(contract_errors)
        resume_binding_path = None
        if checkpoint is not None and not contract_errors:
            resume_binding_path, binding_errors = _validate_pipeline_resume_binding(
                config, resume_phase, checkpoint, runtime
            )
            errors.extend(binding_errors)
        resume_boundary_validated = not contract_errors and resume_binding_path is not None
        warnings.append(
            "resume attempt has no explicit attempt-scoped supervisor status; "
            "legacy initial exit_status is intentionally ignored"
        )
        return (
            {
                "protocol": "resume_without_supervisor_attempt",
                "attempt_id": resume_id,
                "attempt_kind": f"resume_{resume_phase}",
                "identity": f"resume_{resume_phase}:{resume_id}",
                "active_attempt_file": None,
                "exit_status_file": None,
                "exit_status_authoritative": False,
                "status": None,
                "tmux_session": config.waiter_session,
                "validated_boundary": resume_boundary_validated,
                "resume_checkpoint_contract": resume_contract_path,
                "resume_pipeline_binding": resume_binding_path,
                "clears_prior_fatal": resume_boundary_validated,
                "ignored_legacy_exit_status_file": str(config.exit_status_file),
            },
            errors,
            warnings,
        )

    waiter_line = waiter.get("lines", [None])[0] if waiter.get("lines") else None
    identity_material = f"{waiter.get('stanza_start_offset', 0)}:{waiter_line}"
    identity = hashlib.sha256(identity_material.encode("utf-8")).hexdigest()[:16]
    return (
        {
            "protocol": "legacy_initial_waiter",
            "attempt_id": identity,
            "attempt_kind": "initial",
            "identity": f"initial:{identity}",
            "active_attempt_file": None,
            "exit_status_file": str(config.exit_status_file),
            "exit_status_authoritative": True,
            "status": None,
            "tmux_session": config.waiter_session,
            "validated_boundary": False,
            "resume_checkpoint_contract": None,
            "resume_pipeline_binding": None,
            "clears_prior_fatal": False,
        },
        errors,
        warnings,
    )


def check_same_file(left: Path, right: Path, label: str, errors: list[str]) -> None:
    if not left.is_file() or not right.is_file():
        errors.append(f"missing {label}: {left} or {right}")
        return
    if left.read_bytes() != right.read_bytes():
        errors.append(f"{label} is not an exact copy")


def inspect_pipeline_completion(
    config: MonitorConfig,
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    exit_status: int | None,
    runtime: dict[str, Any],
    *,
    allow_no_launch_completion: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = config.pipeline_dir / "manifest.txt"
    manifest = None
    metric_contract = None
    finished = False
    if manifest_path.is_file():
        try:
            manifest = read_manifest(manifest_path)
        except MonitorContractError as error:
            errors.append(str(error))
        if manifest is not None:
            require_manifest_value(manifest, "model", "TRACE-VB-v8", errors)
            require_manifest_value(
                manifest, "pipeline", "train_and_validation_only", errors
            )
            require_manifest_value(manifest, "pipeline_tag", config.tag, errors)
            require_manifest_value(
                manifest,
                "physical_gpus",
                ",".join(map(str, config.physical_gpus)),
                errors,
            )
            require_manifest_value(
                manifest,
                "formal_gpus",
                ",".join(map(str, config.physical_gpus)),
                errors,
            )
            require_manifest_value(
                manifest,
                "fixed_gpus",
                ",".join(map(str, config.physical_gpus)),
                errors,
            )
            metric_contract, metric_errors = validate_metric_safe_manifest_contract(
                manifest,
                runtime,
                scope="pipeline",
                canonical_artifacts={
                    "metric_safe_baseline": (
                        config.pipeline_dir / "metric_safe_baseline.json"
                    ),
                    "registered_capability_validation": (
                        config.pipeline_dir
                        / "registered_capability_validation.json"
                    ),
                },
            )
            errors.extend(metric_errors)
            finished = bool(
                manifest_last(manifest, "finished_at")
                or manifest_last(manifest, "recovery_finished_at")
            )
    elif config.pipeline_dir.exists():
        errors.append("pipeline directory exists without manifest.txt")

    complete_chain = False
    if finished:
        if stage1.get("errors"):
            errors.append(
                "pipeline completion is invalid because Stage 1 has contract errors"
            )
        if stage2.get("errors"):
            errors.append(
                "pipeline completion is invalid because Stage 2 has contract errors"
            )
        if not stage1.get("manifest_finished") or stage1.get("candidate_index") is None:
            errors.append("pipeline finished without complete Stage 1")
        if not stage2.get("manifest_finished") or stage2.get("candidate_index") is None:
            errors.append("pipeline finished without complete Stage 2")
        check_same_file(
            config.pipeline_dir / "stage1_candidate_index.json",
            config.stage1_dir / "candidate_index.json",
            "pipeline Stage-1 candidate index",
            errors,
        )
        check_same_file(
            config.pipeline_dir / "stage2_candidate_index.json",
            config.stage2_dir / "candidate_index.json",
            "pipeline Stage-2 candidate index",
            errors,
        )
        stage1_best = config.pipeline_dir / "stage1_best.txt"
        final_best = config.pipeline_dir / "final_best.txt"
        expected_stage1 = (
            stage1.get("candidate_index") or {}
        ).get("selected_checkpoint")
        expected_final = (stage2.get("candidate_index") or {}).get(
            "selected_checkpoint"
        )
        for label, pointer, expected in (
            ("stage1_best", stage1_best, expected_stage1),
            ("final_best", final_best, expected_final),
        ):
            if not pointer.is_file():
                errors.append(f"pipeline is missing {label}.txt")
                continue
            value = pointer.read_text(encoding="utf-8").strip()
            if expected is None or Path(value).resolve() != Path(expected).resolve():
                errors.append(f"pipeline {label}.txt is inconsistent")
        if manifest is not None:
            manifest_stage1 = manifest_last(manifest, "stage1_checkpoint")
            manifest_final = manifest_last(manifest, "final_checkpoint")
            if expected_stage1 is not None and (
                manifest_stage1 is None
                or Path(manifest_stage1).resolve() != Path(expected_stage1).resolve()
            ):
                errors.append("pipeline manifest Stage-1 checkpoint is inconsistent")
            if expected_final is not None and (
                manifest_final is None
                or Path(manifest_final).resolve() != Path(expected_final).resolve()
            ):
                errors.append("pipeline manifest final checkpoint is inconsistent")
            for label, checkpoint, digest_key in (
                ("Stage-1", expected_stage1, "stage1_checkpoint_sha256"),
                ("final", expected_final, "final_checkpoint_sha256"),
            ):
                if checkpoint is None:
                    continue
                expected_digest = manifest_last(manifest, digest_key)
                if expected_digest is None:
                    errors.append(f"pipeline manifest lacks {digest_key}")
                elif sha256_cached(Path(checkpoint), runtime, force=True) != expected_digest:
                    errors.append(f"pipeline manifest {label} checkpoint SHA256 mismatch")
        if exit_status is None:
            errors.append("pipeline has completion markers but no supervisor exit status")
        elif exit_status != 0:
            errors.append(f"pipeline completion chain has exit status {exit_status}")
        complete_chain = not errors and exit_status == 0
    elif exit_status == 0 and not allow_no_launch_completion:
        errors.append("supervisor exit status is zero but pipeline completion chain is absent")
    return {
        "directory": str(config.pipeline_dir),
        "exists": config.pipeline_dir.is_dir(),
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_finished": finished,
        "metric_safe_contract": metric_contract,
        "complete_chain": complete_chain,
        "errors": errors,
    }


def validate_trigger_false_contract(
    config: MonitorConfig, runtime: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the machine-readable proof for a no-launch terminal result."""

    errors: list[str] = []
    try:
        report = read_json_object(config.trigger_report)
    except MonitorContractError as error:
        return None, [str(error)]
    required = {
        "schema_version": "trace_vb_v8_v7_trigger_contract_v1",
        "status": "TRIGGER_FALSE",
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            errors.append(
                f"v7 trigger report {key}={report.get(key)!r}, expected {expected!r}"
            )
    numeric_fields: dict[str, int] = {}
    for key in (
        "best_correct_count",
        "validation_questions",
        "trigger_maximum_correct_count",
        "formal_epochs_verified",
        "selected_epoch",
    ):
        try:
            numeric_fields[key] = int(report.get(key))
        except (TypeError, ValueError):
            numeric_fields[key] = -1
            errors.append(f"v7 trigger report {key} is not an integer")
    if numeric_fields["validation_questions"] != 747:
        errors.append("v7 trigger report does not bind the 747-question validation")
    if numeric_fields["trigger_maximum_correct_count"] != 522:
        errors.append("v7 trigger report threshold is not 522/747")
    if not 523 <= numeric_fields["best_correct_count"] <= 747:
        errors.append("v7 trigger report does not prove a result strictly above 70%")
    if numeric_fields["formal_epochs_verified"] != 10:
        errors.append("v7 trigger report does not verify ten formal epochs")
    if not 0 <= numeric_fields["selected_epoch"] < 10:
        errors.append("v7 trigger report selected_epoch is outside 0..9")
    best_epochs = report.get("best_epochs")
    if (
        not isinstance(best_epochs, list)
        or numeric_fields["selected_epoch"] not in best_epochs
    ):
        errors.append("v7 trigger report selected epoch is not a best epoch")

    stage1_dir = Path(str(report.get("stage1_dir", "")))
    manifest_path = Path(str(report.get("manifest", "")))
    index_path = Path(str(report.get("validation_summary_index", "")))
    checkpoint = Path(str(report.get("checkpoint", "")))
    if not stage1_dir.is_absolute() or not stage1_dir.is_dir():
        errors.append("v7 trigger report Stage-1 directory is missing/non-absolute")
    if (
        not manifest_path.is_absolute()
        or manifest_path.resolve() != (stage1_dir / "manifest.txt").resolve()
        or not manifest_path.is_file()
    ):
        errors.append("v7 trigger report manifest path is inconsistent")
        manifest = None
    else:
        try:
            manifest = read_manifest(manifest_path)
        except MonitorContractError as error:
            errors.append(str(error))
            manifest = None
    if manifest is not None:
        if not manifest_last(manifest, "finished_at"):
            errors.append("formal v7 Stage-1 manifest is not finished")
        manifest_index = manifest_last(manifest, "validation_summary_index")
        if (
            manifest_index is None
            or Path(manifest_index).resolve() != index_path.resolve()
        ):
            errors.append("v7 trigger report differs from manifest validation index")
        manifest_checkpoint = manifest_last(manifest, "best_checkpoint")
        if manifest_checkpoint is not None and (
            Path(manifest_checkpoint).resolve() != checkpoint.resolve()
        ):
            errors.append("v7 trigger report differs from manifest best checkpoint")
    if not index_path.is_absolute() or not index_path.is_file():
        errors.append("v7 trigger validation index is missing/non-absolute")
    else:
        try:
            index = read_json_object(index_path)
        except MonitorContractError as error:
            errors.append(str(error))
            index = None
        if index is not None:
            if index.get("schema_version") != "trace_vb_v7_stage1_validation_index_v1":
                errors.append("v7 trigger validation index has wrong schema")
            try:
                formal_epochs = int(index.get("formal_epochs", -1))
            except (TypeError, ValueError):
                formal_epochs = -1
            if formal_epochs != 10:
                errors.append("v7 trigger validation index does not declare ten epochs")
            summaries = index.get("summaries")
            if not isinstance(summaries, list) or len(summaries) != 10:
                errors.append("v7 trigger validation index lacks ten summaries")
            published_run_dir = Path(str(index.get("published_run_dir", "")))
            if checkpoint.is_absolute() and (
                published_run_dir.resolve() != checkpoint.parent.parent.resolve()
            ):
                errors.append("v7 trigger validation index belongs to another run")

    checkpoint_match = re.fullmatch(
        r"epoch(?P<epoch>\d+)__step\d+__monitor[-+0-9.eE]+\.ckpt",
        checkpoint.name,
    )
    if not checkpoint.is_absolute() or not checkpoint.is_file():
        errors.append("v7 trigger checkpoint is missing/non-absolute")
    elif checkpoint_match is None or int(checkpoint_match.group("epoch")) != numeric_fields[
        "selected_epoch"
    ]:
        errors.append("v7 trigger checkpoint is not the selected validation epoch")
    else:
        expected_digest = str(report.get("checkpoint_sha256", ""))
        actual_digest = sha256_cached(checkpoint, runtime, force=True)
        if actual_digest != expected_digest:
            errors.append("v7 trigger checkpoint SHA256 mismatch")
    best_record = stage1_dir / "best_checkpoint.txt"
    if not best_record.is_file():
        errors.append("formal v7 Stage-1 best_checkpoint.txt is missing")
    else:
        best_value = best_record.read_text(encoding="utf-8").strip()
        if Path(best_value).resolve() != checkpoint.resolve():
            errors.append("v7 trigger checkpoint differs from best_checkpoint.txt")
    return (
        {
            "path": str(config.trigger_report.resolve()),
            "status": report.get("status"),
            "best_correct_count": numeric_fields["best_correct_count"],
            "validation_questions": numeric_fields["validation_questions"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "verified": not errors,
        },
        errors,
    )


def file_fingerprint(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            result.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return result


def validated_candidate_frontier(
    stage1: dict[str, Any], stage2: dict[str, Any]
) -> dict[str, list[int]]:
    """Return metric-safe, structurally verified validation boundaries.

    Stage-2 coordinate zero is only the copied Stage-1 phase input, not a
    post-failure training boundary, so it cannot independently clear a fatal.
    """

    frontier: dict[str, list[int]] = {"stage1": [], "stage2": []}
    for phase, report in (("stage1", stage1), ("stage2", stage2)):
        verified = {int(value) for value in report.get("verified_coordinates", [])}
        for item in report.get("candidates", []):
            coordinate = int(item["coordinate"])
            if coordinate not in verified or (phase == "stage2" and coordinate == 0):
                continue
            try:
                correct = int(item["correct_count"])
            except (TypeError, ValueError):
                continue
            if correct >= METRIC_SAFE_MINIMUM_CORRECT:
                frontier[phase].append(coordinate)
    return frontier


def _frontier_advanced(
    current: dict[str, list[int]], previous: Any
) -> bool:
    if not isinstance(previous, dict):
        return False
    for phase in ("stage1", "stage2"):
        try:
            old = {int(value) for value in previous.get(phase, [])}
        except (TypeError, ValueError):
            old = set()
        if any(int(value) not in old for value in current.get(phase, [])):
            return True
    return False


def update_sticky_fatal(
    runtime: dict[str, Any],
    attempt: dict[str, Any],
    frontier: dict[str, list[int]],
    log_scans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist fatal evidence until a validated recovery boundary supersedes it."""

    timestamp = now_iso()
    prior = runtime.get("sticky_fatal")
    sticky = dict(prior) if isinstance(prior, dict) and prior.get("active") else None
    clear_reason = None
    if sticky is not None:
        prior_identity = sticky.get("attempt_identity")
        current_identity = attempt.get("identity")
        if (
            current_identity != prior_identity
            and bool(attempt.get("clears_prior_fatal"))
        ):
            clear_reason = "validated_new_supervisor_attempt"
        elif _frontier_advanced(frontier, sticky.get("detection_frontier")):
            clear_reason = "newer_metric_safe_registered_validation"
    if sticky is not None and clear_reason is not None:
        history = runtime.get("fatal_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "event": "cleared",
                "cleared_at": timestamp,
                "reason": clear_reason,
                "fatal_first_seen_at": sticky.get("first_seen_at"),
                "fatal_attempt_identity": sticky.get("attempt_identity"),
                "recovery_attempt_identity": attempt.get("identity"),
                "recovery_frontier": frontier,
            }
        )
        runtime["fatal_history"] = history[-20:]
        sticky = None

    new_findings: list[dict[str, str]] = []
    for scan in log_scans:
        for finding in scan.get("findings", []):
            new_findings.append(
                {
                    "pattern": str(finding.get("pattern")),
                    "path": str(scan.get("path")),
                    "excerpt": str(finding.get("excerpt", "")),
                }
            )
    if new_findings:
        if sticky is None:
            sticky = {
                "active": True,
                "attempt_identity": attempt.get("identity"),
                "first_seen_at": timestamp,
                "detection_frontier": frontier,
                "findings": [],
            }
        existing = {
            (item.get("pattern"), item.get("path"), item.get("excerpt"))
            for item in sticky.get("findings", [])
            if isinstance(item, dict)
        }
        findings = list(sticky.get("findings", []))
        for finding in new_findings:
            key = (finding["pattern"], finding["path"], finding["excerpt"])
            if key not in existing:
                findings.append(finding)
                existing.add(key)
        sticky["findings"] = findings[-20:]
        sticky["last_seen_at"] = timestamp
        sticky["last_observed_attempt_identity"] = attempt.get("identity")

    runtime["sticky_fatal"] = sticky
    return {
        "active": sticky is not None,
        "sticky": sticky,
        "clear_event": (
            {"reason": clear_reason, "cleared_at": timestamp}
            if clear_reason is not None
            else None
        ),
        "new_findings": new_findings,
        "history": runtime.get("fatal_history", []),
    }


def collect_snapshot(
    config: MonitorConfig,
    runtime: dict[str, Any],
    *,
    runner: CommandRunner = run_command,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    environment_errors: list[str] = []
    waiter = parse_waiter_stanza(config.waiter_log)
    expected_gpu_csv = ",".join(map(str, config.physical_gpus))
    if waiter.get("exists"):
        if waiter.get("declared_fixed_gpus") != list(config.physical_gpus):
            errors.append(
                "latest waiter stanza fixed GPU declaration is "
                f"{waiter.get('declared_fixed_gpus')!r}, expected "
                f"{list(config.physical_gpus)!r}"
            )
        if waiter.get("latest_status_line") is not None:
            status_gpu_keys = {int(value) for value in waiter.get("gpu_status", {})}
            if status_gpu_keys != set(config.physical_gpus):
                errors.append(
                    "latest waiter status GPU keys are "
                    f"{sorted(status_gpu_keys)}, expected "
                    f"{sorted(config.physical_gpus)}"
                )
            if (
                waiter.get("stanza_kind") == "resume"
                and waiter.get("boundary") != "validated"
            ):
                errors.append(
                    "latest resume waiter status does not declare boundary=validated"
                )
    try:
        gpus = query_gpus(config.physical_gpus, config.code_root, runner)
    except Exception as error:  # preserve a diagnostic snapshot
        gpus = {"physical_gpus": list(config.physical_gpus), "error": repr(error)}
        environment_errors.append(f"GPU health query failed: {error}")
    try:
        disk = disk_snapshot(config.artifact_root)
        if not disk["enough_free_space"]:
            environment_errors.append(
                f"artifact filesystem has only {disk['free_gib']:.2f} GiB free"
            )
        if int(disk["inode_free"]) < 1024:
            environment_errors.append(
                "artifact filesystem has fewer than 1024 free inodes"
            )
    except Exception as error:
        disk = {"path": str(config.artifact_root), "error": repr(error)}
        environment_errors.append(f"disk health query failed: {error}")

    stage1 = inspect_stage(
        config.stage1_dir,
        "stage1",
        runtime,
        expected_gpu_csv,
        config.stage1_dir,
    )
    stage2 = inspect_stage(
        config.stage2_dir,
        "stage2",
        runtime,
        expected_gpu_csv,
        config.stage1_dir,
    )
    errors.extend(stage1["errors"])
    errors.extend(stage2["errors"])
    attempt, attempt_errors, attempt_warnings = resolve_supervisor_attempt(
        config, waiter, stage1, stage2, runtime
    )
    errors.extend(attempt_errors)
    warnings.extend(attempt_warnings)
    exit_status = None
    exit_errors: list[str] = []
    if attempt.get("exit_status_authoritative"):
        exit_status_path = Path(str(attempt["exit_status_file"]))
        exit_status, exit_errors = parse_exit_status(exit_status_path)
        errors.extend(exit_errors)
    active_status = attempt.get("status")
    if attempt.get("protocol") == "explicit_attempt_v1" and attempt.get(
        "exit_status_authoritative"
    ):
        if active_status in ("WAITING", "RUNNING") and exit_status is not None:
            errors.append(
                f"active attempt status {active_status} already has an exit status"
            )
        elif active_status == "SUCCEEDED" and exit_status != 0:
            errors.append("SUCCEEDED active attempt lacks an authoritative zero exit")
        elif active_status == "FAILED" and (
            exit_status is None or exit_status == 0
        ):
            errors.append("FAILED active attempt lacks an authoritative nonzero exit")
    legacy_exit_status = None
    if not attempt.get("exit_status_authoritative") and config.exit_status_file.exists():
        legacy_exit_status, _ignored_legacy_errors = parse_exit_status(
            config.exit_status_file
        )
    effective_tmux_session = str(
        attempt.get("tmux_session") or config.waiter_session
    )
    try:
        tmux = query_tmux(effective_tmux_session, runner)
    except Exception as error:  # preserve a diagnostic snapshot
        tmux = {
            "session": effective_tmux_session,
            "alive": None,
            "query_ok": False,
            "error": repr(error),
            "panes": [],
        }
    stage1_candidates = {
        int(item["coordinate"]): item for item in stage1.get("candidates", [])
    }
    stage2_candidates_by_rollout = {
        int(item["coordinate"]): item for item in stage2.get("candidates", [])
    }
    if 0 in stage1_candidates and int(stage1_candidates[0]["correct_count"]) < METRIC_SAFE_MINIMUM_CORRECT:
        errors.append(
            "Stage-1 step-0 metric-safe gate is below 527/747: "
            f"{stage1_candidates[0]['correct_count']}/747"
        )
    if stage1.get("candidate_index") is not None and int(
        stage1["candidate_index"]["selected_correct_count"]
    ) < METRIC_SAFE_MINIMUM_CORRECT:
        errors.append("Stage-1 selected candidate is below the 527/747 safety floor")
    if 0 in stage2_candidates_by_rollout:
        phase_input = stage2_candidates_by_rollout[0]
        if int(phase_input["correct_count"]) < METRIC_SAFE_MINIMUM_CORRECT:
            errors.append(
                "Stage-2 phase-input metric-safe gate is below 527/747: "
                f"{phase_input['correct_count']}/747"
            )
        selected_stage1 = stage1.get("candidate_index")
        if selected_stage1 is not None:
            if Path(phase_input["checkpoint"]).resolve() != Path(
                selected_stage1["selected_checkpoint"]
            ).resolve():
                errors.append(
                    "Stage-2 phase input is not the exact selected Stage-1 checkpoint"
                )
            if int(phase_input["correct_count"]) != int(
                selected_stage1["selected_correct_count"]
            ):
                errors.append(
                    "Stage-2 phase-input correct_count differs from selected Stage 1"
                )
    trigger_false = bool(waiter.get("terminal_trigger_false"))
    trigger_false_contract = None
    trigger_false_errors: list[str] = []
    trigger_false_supervisor_valid = False
    if trigger_false:
        trigger_false_contract, trigger_false_errors = (
            validate_trigger_false_contract(config, runtime)
        )
        errors.extend(trigger_false_errors)
        trigger_false_supervisor_valid = bool(
            attempt.get("protocol") == "explicit_attempt_v1"
            and attempt.get("attempt_kind") == "initial"
            and attempt.get("status") == "SUCCEEDED"
            and attempt.get("exit_status_authoritative")
            and exit_status == 0
        )
        if not trigger_false_supervisor_valid:
            errors.append(
                "trigger_false requires an explicit initial SUCCEEDED attempt "
                "with authoritative zero exit"
            )
    trigger_false_authorized = bool(
        trigger_false
        and not trigger_false_errors
        and trigger_false_supervisor_valid
    )
    pipeline = inspect_pipeline_completion(
        config,
        stage1,
        stage2,
        exit_status,
        runtime,
        allow_no_launch_completion=trigger_false_authorized,
    )
    errors.extend(pipeline["errors"])

    completed_without_launch = (
        trigger_false_authorized
        and exit_status == 0
        and not config.pipeline_dir.exists()
        and not config.stage1_dir.exists()
        and not config.stage2_dir.exists()
    )
    if pipeline["complete_chain"] or completed_without_launch:
        lifecycle = "completed"
        phase = "complete" if pipeline["complete_chain"] else "trigger_false_no_launch"
    elif (
        attempt.get("protocol") == "explicit_attempt_v1"
        and attempt.get("status") == "WAITING"
    ):
        lifecycle = "waiting"
        if str(attempt.get("attempt_kind", "")).startswith("resume_"):
            phase = f"waiting_to_{attempt['attempt_kind']}"
        else:
            phase = "waiting_for_fixed_gpus"
    elif config.pipeline_dir.exists():
        lifecycle = "running"
        if not stage1["manifest_finished"]:
            phase = "stage1"
        elif not config.stage2_dir.exists():
            phase = "between_stages"
        elif not stage2["manifest_finished"]:
            phase = "stage2"
        else:
            phase = "finalizing"
    else:
        lifecycle = "waiting"
        phase = "waiting_for_fixed_gpus"

    if lifecycle == "completed":
        warnings.extend(environment_errors)
    else:
        errors.extend(environment_errors)

    if exit_status is not None and exit_status != 0:
        errors.append(f"supervisor exited with nonzero status {exit_status}")
    if lifecycle != "completed":
        session_role = (
            "waiter"
            if attempt.get("protocol") == "legacy_initial_waiter"
            else "active attempt"
        )
        if tmux.get("alive") is False:
            errors.append(
                f"exact {session_role} tmux session is not alive: "
                f"{effective_tmux_session}"
            )
        elif tmux.get("alive") is None:
            errors.append(
                f"exact {session_role} tmux liveness could not be verified: "
                f"{tmux.get('error')}"
            )
    if lifecycle == "waiting" and gpus.get("all_launch_eligible"):
        warnings.append(
            "all configured GPUs are launch-eligible; waiter should transition shortly"
        )
    if lifecycle == "waiting" and not waiter.get("exists"):
        warnings.append("waiter log does not exist yet")
    if lifecycle == "waiting" and waiter.get("exists") and not waiter.get(
        "latest_status_line"
    ):
        warnings.append("latest waiter stanza has no status line yet")

    stage_log_floors = {"stage1": 0, "stage2": 0}
    strict_stage_log_floors = False
    if (
        attempt.get("protocol") == "explicit_attempt_v1"
        and isinstance(attempt.get("log_byte_offset_floors"), dict)
    ):
        strict_stage_log_floors = True
        for stage_name, log_path in (
            ("stage1", config.stage1_dir / "train.log"),
            ("stage2", config.stage2_dir / "train.log"),
        ):
            floor = int(
                attempt["log_byte_offset_floors"][str(log_path.resolve())]
            )
            if log_path.exists() and not log_path.is_file():
                errors.append(f"{stage_name} train.log is not a regular file")
                floor = 0
            elif not log_path.exists() and floor > 0:
                errors.append(
                    f"{stage_name} train.log is missing below its attempt byte floor"
                )
                floor = 0
            elif log_path.is_file() and log_path.stat().st_size < floor:
                errors.append(
                    f"{stage_name} train.log size is below its attempt byte floor"
                )
                floor = 0
            stage_log_floors[stage_name] = floor

    log_scans = [
        scan_log_increment(
            config.waiter_log,
            runtime,
            floor_offset=int(waiter.get("stanza_start_offset", 0)),
            strict_floor=True,
        ),
        scan_log_increment(
            config.stage1_dir / "train.log",
            runtime,
            floor_offset=stage_log_floors["stage1"],
            strict_floor=strict_stage_log_floors,
        ),
        scan_log_increment(
            config.stage2_dir / "train.log",
            runtime,
            floor_offset=stage_log_floors["stage2"],
            strict_floor=strict_stage_log_floors,
        ),
    ]
    fatal_state = update_sticky_fatal(
        runtime,
        attempt,
        validated_candidate_frontier(stage1, stage2),
        log_scans,
    )
    for finding in fatal_state["new_findings"]:
        errors.append(
            f"fatal log marker {finding['pattern']} in {finding['path']}"
        )
    if fatal_state["active"]:
        sticky_patterns = sorted(
            {
                str(item.get("pattern"))
                for item in (fatal_state.get("sticky") or {}).get("findings", [])
                if isinstance(item, dict)
            }
        )
        errors.append(
            "unresolved sticky fatal marker(s): " + ", ".join(sticky_patterns)
        )

    progress_paths = [
        config.pipeline_dir / "manifest.txt",
        config.stage1_dir / "manifest.txt",
        config.stage1_dir / "train.log",
        config.stage1_dir / "candidate_index.json",
        config.stage2_dir / "manifest.txt",
        config.stage2_dir / "train.log",
        config.stage2_dir / "candidate_index.json",
        config.exit_status_file,
        config.active_attempt_file,
    ]
    if attempt.get("exit_status_file"):
        progress_paths.append(Path(str(attempt["exit_status_file"])))
    progress_paths.extend(
        config.stage1_dir / "candidates" / candidate_filename("stage1", coordinate)
        for coordinate in EXPECTED_COORDINATES["stage1"]
    )
    progress_paths.extend(
        config.stage2_dir / "candidates" / candidate_filename("stage2", coordinate)
        for coordinate in EXPECTED_COORDINATES["stage2"]
    )
    fingerprint = file_fingerprint(progress_paths)
    serialized_fingerprint = json.dumps(fingerprint, sort_keys=True)
    previous_fingerprint = runtime.get("progress_fingerprint")
    if lifecycle == "running" and previous_fingerprint == serialized_fingerprint:
        stagnant_cycles = int(runtime.get("stagnant_cycles", 0)) + 1
    else:
        stagnant_cycles = 0
        runtime["last_progress_at"] = now_iso()
    runtime["progress_fingerprint"] = serialized_fingerprint
    runtime["stagnant_cycles"] = stagnant_cycles

    current_ticks: dict[str, int] = {}
    for application in gpus.get("applications", []):
        process = application.get("process")
        if isinstance(process, dict) and process.get("exact_v8_rank"):
            current_ticks[str(process["pid"])] = int(process["cpu_ticks"])
    previous_ticks = {
        str(key): int(value)
        for key, value in runtime.get("process_cpu_ticks", {}).items()
    }
    cpu_active = any(
        pid not in previous_ticks or ticks > previous_ticks[pid]
        for pid, ticks in current_ticks.items()
    )
    runtime["process_cpu_ticks"] = current_ticks
    gpu_rows = gpus.get("rows", [])
    all_gpu_idle = bool(gpu_rows) and all(
        int(item["utilization_percent"]) <= 1 for item in gpu_rows
    )
    if (
        lifecycle == "running"
        and stagnant_cycles >= 2
        and all_gpu_idle
        and not cpu_active
    ):
        errors.append(
            "no artifact/log progress for two monitor intervals and no GPU/CPU rank activity"
        )

    best = None
    chronological_final = None
    stage2_candidates = stage2.get("candidates", [])
    if stage2.get("candidate_index") is not None:
        best = {
            "correct_count": stage2["candidate_index"]["selected_correct_count"],
            "accuracy": stage2["candidate_index"]["selected_accuracy"],
            "checkpoint": stage2["candidate_index"]["selected_checkpoint"],
            "strictly_above_70_percent": int(
                stage2["candidate_index"]["selected_correct_count"]
            )
            >= 523,
        }
    if stage2_candidates and int(stage2_candidates[-1]["coordinate"]) == 1024:
        chronological_final = {
            "correct_count": stage2_candidates[-1]["correct_count"],
            "accuracy": stage2_candidates[-1]["accuracy"],
            "checkpoint": stage2_candidates[-1]["checkpoint"],
            "strictly_above_70_percent": int(
                stage2_candidates[-1]["correct_count"]
            )
            >= 523,
        }

    health = "abnormal" if errors else "healthy"
    return {
        "schema_version": MONITOR_SCHEMA,
        "updated_at": now_iso(),
        "tag": config.tag,
        "artifact_root": str(config.artifact_root),
        "state_directory": str(config.state_dir),
        "physical_gpus": list(config.physical_gpus),
        "lifecycle": lifecycle,
        "phase": phase,
        "health": health,
        "errors": errors,
        "warnings": warnings,
        "supervisor_exit_status": exit_status,
        "ignored_legacy_supervisor_exit_status": legacy_exit_status,
        "supervisor_attempt": attempt,
        "waiter": waiter,
        "tmux": tmux,
        "gpu": gpus,
        "disk": disk,
        "stage1": stage1,
        "stage2": stage2,
        "pipeline": pipeline,
        "trigger_false_contract": trigger_false_contract,
        "validation": {
            "questions": 747,
            "strict_above_70_minimum_correct": 523,
            "metric_safe_minimum_correct": METRIC_SAFE_MINIMUM_CORRECT,
            "metric_safe_baseline_sha256": METRIC_SAFE_BASELINE_SHA256,
            "registered_capability_validation_sha256": (
                REGISTERED_CAPABILITY_VALIDATION_SHA256
            ),
            "best_registered": best,
            "chronological_final": chronological_final,
        },
        "progress": {
            "stagnant_monitor_cycles": stagnant_cycles,
            "last_progress_at": runtime.get("last_progress_at"),
            "exact_rank_cpu_active_since_previous_snapshot": cpu_active,
        },
        "log_scans": log_scans,
        "fatal_state": fatal_state,
        "read_only_guarantee": {
            "training_artifacts_mutated": False,
            "processes_signalled": False,
            "jobs_or_sessions_launched": False,
            "writes_restricted_to_state_directory": True,
        },
    }


def validate_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError("tag must contain only A-Z, a-z, 0-9, ._- ")
    return value


def validate_session(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError(
            "waiter session must contain only A-Z, a-z, 0-9, ._-"
        )
    return value


def parse_gpu_csv(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(token.strip()) for token in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("physical GPUs must be integers") from error
    if len(values) != 4 or len(set(values)) != 4 or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "physical GPUs must be four distinct non-negative IDs"
        )
    return values  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, type=validate_tag)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/disk1/dingxukai/TRACE/trace_vb_v8_runs"),
    )
    parser.add_argument(
        "--physical-gpus", type=parse_gpu_csv, default=parse_gpu_csv("2,3,4,5")
    )
    parser.add_argument(
        "--waiter-session",
        type=validate_session,
        default="trace_vb_v8_conditional_wait_20260820",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--state-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval_seconds < 10:
        raise SystemExit("--interval-seconds must be at least 10")
    artifact_root = args.artifact_root.resolve()
    state_dir = (
        args.state_dir.resolve()
        if args.state_dir is not None
        else artifact_root / "monitor" / args.tag
    )
    code_root = Path(__file__).resolve().parents[1]
    config = MonitorConfig(
        tag=args.tag,
        artifact_root=artifact_root,
        physical_gpus=args.physical_gpus,
        waiter_session=args.waiter_session,
        state_dir=state_dir,
        code_root=code_root,
    )
    runtime_path = state_dir / "runtime.json"
    latest_path = state_dir / "latest.json"
    while True:
        runtime = load_runtime(runtime_path)
        try:
            snapshot = collect_snapshot(config, runtime)
        except Exception as error:  # always leave an actionable snapshot
            snapshot = {
                "schema_version": MONITOR_SCHEMA,
                "updated_at": now_iso(),
                "tag": config.tag,
                "artifact_root": str(config.artifact_root),
                "state_directory": str(config.state_dir),
                "physical_gpus": list(config.physical_gpus),
                "lifecycle": "unknown",
                "phase": "monitor_exception",
                "health": "abnormal",
                "errors": [repr(error)],
                "warnings": [],
                "read_only_guarantee": {
                    "training_artifacts_mutated": False,
                    "processes_signalled": False,
                    "jobs_or_sessions_launched": False,
                    "writes_restricted_to_state_directory": True,
                },
            }
        runtime["last_snapshot_at"] = snapshot["updated_at"]
        # Publish the user-visible snapshot before advancing log cursors.  If
        # the process dies between these two atomic replacements, the next run
        # safely re-scans a small overlap instead of losing a fatal marker.
        atomic_json(latest_path, snapshot)
        atomic_json(runtime_path, runtime)
        print(json.dumps(snapshot, indent=2, sort_keys=True), flush=True)
        if args.once:
            return 2 if snapshot.get("health") == "abnormal" else 0
        if snapshot.get("lifecycle") == "completed":
            return 2 if snapshot.get("health") == "abnormal" else 0
        # Keep observing abnormal states: a validated-boundary recovery may be
        # started externally, and this read-only process must never attempt the
        # repair itself or silently stop producing half-hour snapshots.
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
