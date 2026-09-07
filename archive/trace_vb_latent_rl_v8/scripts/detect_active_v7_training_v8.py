#!/usr/bin/env python3
"""Detect live v7 training workers without matching supervisors or tmux text."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_V7_ROOT = Path("/home/dingxukai/TRACE/trace_vb_latent_rl_v7")
TRAINING_WRAPPERS = frozenset(
    {
        "resume_full_pipeline_vb.sh",
        "run_full_pipeline.sh",
        "run_full_pipeline_vb.sh",
        "run_preflight_validation_v7.sh",
        "run_stage0_cot.sh",
        "run_stage1_formation.sh",
        "run_stage1_vb.sh",
        "run_stage2_and_evidence.sh",
        "run_stage2_refinement.sh",
        "run_stage2_vb.sh",
    }
)
SHELL_EXECUTABLES = frozenset({"ash", "bash", "dash", "ksh", "sh", "zsh"})
DEAD_STATES = frozenset({"X", "Z"})
ACTIVE_EXIT = 0
ERROR_EXIT = 2
INACTIVE_EXIT = 3
PYTHON_OPTIONS_WITH_ARGUMENT = frozenset(
    {"-W", "-X", "--check-hash-based-pycs"}
)
PYTHON_TERMINATING_OPTIONS = frozenset(
    {"-h", "--help", "-V", "--version"}
)
SHELL_OPTIONS_WITH_ARGUMENT = frozenset(
    {"-O", "+O", "-o", "+o", "--init-file", "--rcfile"}
)


def _resolved_token(token: str, cwd: str) -> Path | None:
    if not token or token.startswith("-"):
        return None
    value = Path(token)
    if not value.is_absolute():
        if not cwd:
            return None
        value = Path(cwd) / value
    try:
        return value.resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def _python_script_entry(argv: list[str]) -> str | None:
    """Return only Python's interpreter script slot, never sys.argv data."""
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return argv[index + 1] if index + 1 < len(argv) else None
        if token == "-":
            return None
        if token in PYTHON_TERMINATING_OPTIONS:
            return None
        if token in ("-c", "-m") or token.startswith(("-c", "-m")):
            # Everything following the command/module is business sys.argv,
            # even if it happens to spell the v7 run.py path.
            return None
        if token in PYTHON_OPTIONS_WITH_ARGUMENT:
            index += 2
            continue
        if token.startswith(("-W", "-X", "--check-hash-based-pycs=")):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None


def _shell_script_entry(argv: list[str]) -> str | None:
    """Return only a shell's script operand, never the script's arguments."""
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return argv[index + 1] if index + 1 < len(argv) else None
        if token == "-":
            return None
        if token in SHELL_OPTIONS_WITH_ARGUMENT:
            index += 2
            continue
        if token.startswith(("--init-file=", "--rcfile=")):
            index += 1
            continue
        if token.startswith("--"):
            index += 1
            continue
        if token.startswith(("-", "+")):
            option_letters = token[1:]
            if "c" in option_letters or "s" in option_letters:
                # -c consumes a command string and -s reads stdin; subsequent
                # values are positional parameters rather than script paths.
                return None
            if "o" in option_letters or "O" in option_letters:
                index += 2
            else:
                index += 1
            continue
        return token
    return None


def classify_active_v7_process(
    process: dict,
    *,
    v7_root: Path = DEFAULT_V7_ROOT,
) -> str | None:
    """Return a blocking reason only for a real v7 trainer or stage wrapper."""
    state = str(process.get("state", ""))[:1].upper()
    if not state or state in DEAD_STATES:
        return None
    argv_value = process.get("argv", [])
    if not isinstance(argv_value, list) or not argv_value:
        return None
    argv = [str(value) for value in argv_value]
    executable = Path(argv[0]).name.lstrip("-")
    comm = str(process.get("comm", ""))
    # A tmux server retains the original new-session command in its argv.  It
    # is orchestration metadata, not the command currently executing in a pane.
    if executable == "tmux" or comm == "tmux" or comm.startswith("tmux:"):
        return None

    root = v7_root.resolve(strict=False)
    cwd = str(process.get("cwd", ""))
    run_entry = (root / "run.py").resolve(strict=False)
    script_root = (root / "scripts").resolve(strict=False)

    direct_script = _resolved_token(argv[0], cwd)
    if (
        direct_script is not None
        and direct_script.parent == script_root
        and direct_script.name in TRAINING_WRAPPERS
    ):
        return f"v7_stage_wrapper:{direct_script.name}"

    if executable.startswith("python"):
        entry = _python_script_entry(argv)
        if entry is not None and _resolved_token(entry, cwd) == run_entry:
            return "v7_run_py_rank"

    if executable in SHELL_EXECUTABLES:
        entry = _shell_script_entry(argv)
        resolved = _resolved_token(entry or "", cwd)
        if (
            resolved is not None
            and resolved.parent == script_root
            and resolved.name in TRAINING_WRAPPERS
        ):
            return f"v7_stage_wrapper:{resolved.name}"
    return None


def active_v7_processes(
    processes: Iterable[dict],
    *,
    v7_root: Path = DEFAULT_V7_ROOT,
) -> list[dict]:
    matches = []
    for process in processes:
        reason = classify_active_v7_process(process, v7_root=v7_root)
        if reason is not None:
            matches.append(
                {
                    "pid": int(process.get("pid", -1)),
                    "ppid": int(process.get("ppid", -1)),
                    "state": str(process.get("state", "")),
                    "comm": str(process.get("comm", "")),
                    "argv": [str(value) for value in process.get("argv", [])],
                    "cwd": str(process.get("cwd", "")),
                    "reason": reason,
                }
            )
    return matches


def _read_status(path: Path) -> tuple[str, str]:
    state = ""
    comm = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("State:"):
            match = re.search(r"State:\s*([A-Z])", line)
            state = match.group(1) if match else ""
        elif line.startswith("Name:"):
            comm = line.split(":", 1)[1].strip()
    return state, comm


def live_process_snapshot(proc_root: Path = Path("/proc")) -> list[dict]:
    snapshot = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            raw_argv = (directory / "cmdline").read_bytes().split(b"\0")
            argv = [
                value.decode("utf-8", errors="replace")
                for value in raw_argv
                if value
            ]
            if not argv:
                continue
            state, comm = _read_status(directory / "status")
            stat_fields = (directory / "stat").read_text(
                encoding="utf-8", errors="replace"
            ).rsplit(") ", 1)[1].split()
            ppid = int(stat_fields[1])
            try:
                cwd = os.readlink(directory / "cwd")
            except OSError:
                cwd = ""
            snapshot.append(
                {
                    "pid": int(directory.name),
                    "ppid": ppid,
                    "state": state,
                    "comm": comm,
                    "argv": argv,
                    "cwd": cwd,
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-root", type=Path, default=DEFAULT_V7_ROOT)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="read a JSON process list instead of /proc (contract tests only)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.snapshot is None:
            processes = live_process_snapshot()
        else:
            processes = json.loads(args.snapshot.read_text(encoding="utf-8"))
            if not isinstance(processes, list):
                raise ValueError("process snapshot must be a JSON list")
        matches = active_v7_processes(processes, v7_root=args.v7_root)
    except Exception as error:
        print(f"v7 activity detector failed: {error}", file=sys.stderr)
        raise SystemExit(ERROR_EXIT) from error
    if not args.quiet:
        print(
            json.dumps(
                {
                    "schema_version": "trace_vb_v8_v7_activity_v1",
                    "active": bool(matches),
                    "matches": matches,
                },
                sort_keys=True,
            )
        )
    raise SystemExit(ACTIVE_EXIT if matches else INACTIVE_EXIT)


if __name__ == "__main__":
    main()
