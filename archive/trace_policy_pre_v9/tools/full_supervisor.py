#!/usr/bin/env python3
"""Wait for four empty GPUs and run the independent formal TRACE pipeline."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/disk1/dingxukai/TRACE")
PYTHON = Path("/home/dingxukai/miniconda3/envs/ROT/bin/python")
STATE_ROOT = ROOT / "run_outputs/supervisor"
PIPELINE = ROOT / "scripts/run_full_pipeline.sh"
CONTRACT_AUDIT = ROOT / "tools/contract_audit.py"
CRITICAL_SOURCES = (
    ROOT / "run.py",
    ROOT / "src/models/model_base.py",
    ROOT / "src/models/read.py",
    ROOT / "src/utils/distributed.py",
    ROOT / "src/models/trace_policy.py",
    ROOT / "src/modules/trace_policy.py",
    ROOT / "src/datasets/gsm8k_aug_nl.py",
    ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml",
    ROOT / "src/configs/datasets/gsm8k_aug_nl.yaml",
    ROOT / "scripts/run_stage0_cot.sh",
    ROOT / "scripts/run_stage1_formation.sh",
    ROOT / "scripts/run_stage2_refinement.sh",
    ROOT / "scripts/run_evidence.sh",
    ROOT / "tools/contract_audit.py",
    ROOT / "tools/data_contract_audit.py",
    ROOT / "tools/trace_stage1_target_audit.py",
    ROOT / "tools/trace_prepare_pca_fit_set.py",
    ROOT / "tools/trace_policy_task_summary.py",
    ROOT / "tools/trace_policy_geometry_summary.py",
    ROOT / "tools/trace_policy_stage_comparison.py",
    ROOT / "tools/trace_policy_causal_summary.py",
    ROOT / "tools/trace_policy_mechanism_smoke.py",
    ROOT / "tools/trace_policy_ddp_memory_smoke.py",
    ROOT / "tools/isolated_gpu_ddp_entry.py",
    ROOT / "src/configs/trainer/trace_stage1_gpu4_dynamic.yaml",
    ROOT / "tools/trace_policy_stage2_stability_smoke.py",
    ROOT / "tools/full_supervisor.py",
    ROOT / "tests/test_data_contract.py",
    ROOT / "tests/test_trace_policy.py",
    PIPELINE,
)
FATAL_PIPELINE_MARKERS = (
    "torch.OutOfMemoryError",
    "CUDA out of memory",
    "ChildFailedError",
    "ProcessRaisedException",
    "NCCL watchdog caught collective operation timeout",
)
FATAL_MARKER_TAIL_LENGTH = max(
    len(marker) for marker in FATAL_PIPELINE_MARKERS
) - 1


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_log(path: Path, message: str):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp()} {message}\n")
        handle.flush()


def detect_fatal_marker(previous_tail: str, new_text: str):
    """Detect markers even when a log read splits the marker in two."""
    combined = previous_tail + new_text
    marker = next(
        (
            candidate
            for candidate in FATAL_PIPELINE_MARKERS
            if candidate in combined
        ),
        None,
    )
    return marker, combined[-FATAL_MARKER_TAIL_LENGTH:]


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(int(process_group_id), 0)
    except ProcessLookupError:
        return False
    return True


def terminate_process_group(child, *, timeout_seconds: int = 20):
    """Terminate and verify every process in the pipeline session.

    The pipeline leader can exit before its torch DDP workers. Checking only
    ``child.poll()`` in that state leaves the re-parented workers alive, so the
    process group itself is the cleanup authority.
    """
    if child is None:
        return None
    process_group_id = int(child.pid)
    if not process_group_exists(process_group_id):
        return child.poll()
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return child.poll()
    deadline = time.monotonic() + float(timeout_seconds)
    while process_group_exists(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + float(timeout_seconds)
        while (
            process_group_exists(process_group_id)
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
    if process_group_exists(process_group_id):
        raise RuntimeError(
            f"pipeline process group {process_group_id} survived SIGKILL"
        )
    try:
        return child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        return child.poll()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "memory_used_mib": int(fields[1]),
                "memory_free_mib": int(fields[2]),
                "utilization_percent": int(fields[3]),
            }
        )
    return rows


def select_empty_gpus(rows, count: int, memory_threshold_mib: int):
    eligible = [
        row
        for row in rows
        if row["memory_used_mib"] <= memory_threshold_mib
    ]
    eligible.sort(key=lambda row: (row["memory_used_mib"], row["index"]))
    return [row["index"] for row in eligible[:count]]


def run_contract_audit(run_dir: Path):
    output_path = run_dir / "preflight_contract_audit.json"
    with output_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            [str(PYTHON), str(CONTRACT_AUDIT)],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"TRACE contract audit failed with code "
            f"{completed.returncode}"
        )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("TRACE contract audit did not pass")
    return report


def run_unit_tests(run_dir: Path):
    output_path = run_dir / "preflight_unit_tests.log"
    with output_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            [
                str(PYTHON),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
            ],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"TRACE unit tests failed with code {completed.returncode}"
        )
    return output_path


def verify_source_hashes(expected: dict):
    actual = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in CRITICAL_SOURCES
    }
    changed = sorted(
        path
        for path, digest in expected.items()
        if actual.get(path) != digest
    )
    if changed:
        raise RuntimeError(
            "Critical TRACE sources changed while waiting: "
            + ", ".join(changed)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--required-gpus", type=int, default=4)
    parser.add_argument("--memory-threshold-mib", type=int, default=3072)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--resume-pipeline-tag", default="")
    parser.add_argument("--resume-stage1-checkpoint", type=Path)
    parser.add_argument("--stage0-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.required_gpus != 4:
        raise SystemExit("The formal TRACE run requires exactly four GPUs")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    if bool(args.resume_pipeline_tag) != bool(args.resume_stage1_checkpoint):
        raise SystemExit(
            "--resume-pipeline-tag and --resume-stage1-checkpoint "
            "must be provided together"
        )
    if (
        args.resume_stage1_checkpoint is not None
        and not args.resume_stage1_checkpoint.is_file()
    ):
        raise SystemExit(
            f"Missing Stage-1 resume checkpoint: "
            f"{args.resume_stage1_checkpoint}"
        )
    if (
        args.stage0_checkpoint is not None
        and not args.stage0_checkpoint.is_file()
    ):
        raise SystemExit(
            f"Missing completed fresh Stage-0 checkpoint: "
            f"{args.stage0_checkpoint}"
        )

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = (STATE_ROOT / ".formal_full_pipeline.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A TRACE formal supervisor is already active")

    if args.dry_run:
        rows = gpu_snapshot()
        selected = select_empty_gpus(
            rows,
            args.required_gpus,
            args.memory_threshold_mib,
        )
        print(
            json.dumps(
                {
                    "gpu_snapshot": rows,
                    "selected": selected,
                    "would_start": len(selected) == args.required_gpus,
                },
                indent=2,
            )
        )
        return 0

    run_tag = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    if args.resume_pipeline_tag:
        run_tag += f"_trace_resume_seed{args.train_seed}"
    else:
        run_tag += f"_trace_policy_v3_full_seed{args.train_seed}"
    run_dir = STATE_ROOT / run_tag
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "supervisor.log"
    status_path = run_dir / "status.json"
    pid_path = run_dir / "supervisor.pid"
    latest_path = STATE_ROOT / "latest"
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
    if latest_path.is_symlink() or latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)

    source_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in CRITICAL_SOURCES
    }
    atomic_json(run_dir / "source_hashes.json", source_hashes)
    unit_test_log = run_unit_tests(run_dir)
    append_log(
        log_path,
        f"Preflight unit tests passed; log={unit_test_log}.",
    )
    report = run_contract_audit(run_dir)
    append_log(
        log_path,
        f"Preflight contract audit passed ({report['passed']}/"
        f"{report['total']}).",
    )

    state = {
        "model": "TRACE-Policy-v3",
        "run_tag": run_tag,
        "supervisor_pid": os.getpid(),
        "state": "waiting_for_four_empty_gpus",
        "required_gpus": args.required_gpus,
        "memory_threshold_mib": args.memory_threshold_mib,
        "poll_seconds": args.poll_seconds,
        "train_seed": args.train_seed,
        "resume_pipeline_tag": args.resume_pipeline_tag or None,
        "resume_stage1_checkpoint": (
            str(args.resume_stage1_checkpoint.resolve())
            if args.resume_stage1_checkpoint is not None
            else None
        ),
        "stage0_checkpoint": (
            str(args.stage0_checkpoint.resolve())
            if args.stage0_checkpoint is not None
            else None
        ),
        "created_at": timestamp(),
        "updated_at": timestamp(),
    }
    atomic_json(status_path, state)

    child = None
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        append_log(log_path, f"Received signal {signum}; stopping supervisor.")
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    selected = []
    while not stop_requested:
        try:
            rows = gpu_snapshot()
            selected = select_empty_gpus(
                rows,
                args.required_gpus,
                args.memory_threshold_mib,
            )
            state.update(
                {
                    "gpu_snapshot": rows,
                    "selected_gpus": selected,
                    "updated_at": timestamp(),
                }
            )
            atomic_json(status_path, state)
            if len(selected) == args.required_gpus:
                break
        except Exception as error:
            state.update(
                {
                    "last_gpu_query_error": repr(error),
                    "updated_at": timestamp(),
                }
            )
            atomic_json(status_path, state)
            append_log(log_path, f"GPU query failed: {error!r}")
        time.sleep(args.poll_seconds)

    if stop_requested:
        state.update({"state": "stopped", "updated_at": timestamp()})
        atomic_json(status_path, state)
        return 130

    gpu_csv = ",".join(str(index) for index in selected)
    pipeline_log_path = run_dir / "pipeline.stdout.log"
    try:
        verify_source_hashes(source_hashes)
    except Exception as error:
        state.update(
            {
                "state": "source_integrity_failed",
                "source_integrity_error": str(error),
                "updated_at": timestamp(),
            }
        )
        atomic_json(status_path, state)
        append_log(log_path, str(error))
        return 3
    environment = os.environ.copy()
    environment.update(
        {
            "PIPELINE_TAG": run_tag,
            "TRAIN_SEED": str(args.train_seed),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if args.resume_pipeline_tag:
        environment.update(
            {
                "RESUME_PIPELINE_TAG": args.resume_pipeline_tag,
                "RESUME_STAGE1_CKPT": str(
                    args.resume_stage1_checkpoint.resolve()
                ),
            }
        )
    if args.stage0_checkpoint is not None:
        environment["STAGE0_CKPT_OVERRIDE"] = str(
            args.stage0_checkpoint.resolve()
        )
    append_log(
        log_path,
        f"Four empty GPUs detected once; launching "
        f"{'resumed' if args.resume_pipeline_tag else 'fresh'} formal "
        f"pipeline on {gpu_csv}.",
    )
    with pipeline_log_path.open("a", encoding="utf-8") as pipeline_log:
        child = subprocess.Popen(
            [
                "bash",
                str(PIPELINE),
                gpu_csv,
                str(selected[0]),
            ],
            cwd=ROOT,
            env=environment,
            stdout=pipeline_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        state.update(
            {
                "state": "pipeline_running",
                "physical_training_gpus": selected,
                "evidence_gpu": selected[0],
                "pipeline_pid": child.pid,
                "pipeline_log": str(pipeline_log_path),
                "pipeline_started_at": timestamp(),
                "updated_at": timestamp(),
            }
        )
        atomic_json(status_path, state)
        log_offset = 0
        log_tail = ""
        fatal_marker = None
        source_integrity_error = None
        return_code = None
        while return_code is None:
            return_code = child.poll()
            try:
                with pipeline_log_path.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as health_log:
                    health_log.seek(log_offset)
                    new_text = health_log.read()
                    log_offset = health_log.tell()
                fatal_marker, log_tail = detect_fatal_marker(
                    log_tail,
                    new_text,
                )
            except OSError as error:
                state["pipeline_health_error"] = repr(error)
            try:
                verify_source_hashes(source_hashes)
            except Exception as error:
                source_integrity_error = str(error)
            state.update(
                {
                    "pipeline_child_alive": return_code is None,
                    "pipeline_log_bytes_checked": log_offset,
                    "updated_at": timestamp(),
                }
            )
            atomic_json(status_path, state)
            if fatal_marker is not None and return_code is None:
                append_log(
                    log_path,
                    f"Detected fatal pipeline marker {fatal_marker!r}; "
                    "terminating the complete process group.",
                )
                state.update(
                    {
                        "state": "pipeline_failing_fast",
                        "pipeline_failure_marker": fatal_marker,
                        "updated_at": timestamp(),
                    }
                )
                atomic_json(status_path, state)
                terminate_process_group(child)
                return_code = 86
                break
            if source_integrity_error is not None and return_code is None:
                append_log(
                    log_path,
                    f"{source_integrity_error}; terminating the complete "
                    "process group.",
                )
                state.update(
                    {
                        "state": "pipeline_source_integrity_failed",
                        "source_integrity_error": source_integrity_error,
                        "updated_at": timestamp(),
                    }
                )
                atomic_json(status_path, state)
                terminate_process_group(child)
                return_code = 87
                break
            if stop_requested and return_code is None:
                terminate_process_group(child)
                return_code = 130
                break
            if return_code is None:
                time.sleep(min(args.poll_seconds, 5))

    state.update(
        {
            "state": (
                "pipeline_completed" if return_code == 0 else "pipeline_failed"
            ),
            "pipeline_return_code": return_code,
            "pipeline_finished_at": timestamp(),
            "updated_at": timestamp(),
        }
    )
    atomic_json(status_path, state)
    append_log(
        log_path,
        f"Formal pipeline exited with return code {return_code}.",
    )
    return return_code


if __name__ == "__main__":
    sys.exit(main())
