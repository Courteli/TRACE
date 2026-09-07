#!/usr/bin/env python3
"""Wait for four empty GPUs and run the formal TRACE-Policy pipeline."""

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


ROOT = Path("/disk1/dingxukai/trace_colar")
PYTHON = Path("/home/dingxukai/miniconda3/envs/ROT/bin/python")
STATE_ROOT = ROOT / "run_outputs/trace_policy/supervisor"
PIPELINE = ROOT / "run_trace_policy_full_pipeline.sh"
CONTRACT_AUDIT = ROOT / "tools/trace_policy_contract_audit.py"
CRITICAL_SOURCES = (
    ROOT / "src/models/trace_policy.py",
    ROOT / "src/modules/trace_policy.py",
    ROOT / "src/datasets/trace_rationale_set.py",
    ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml",
    ROOT / "src/configs/datasets/trace_qsa.yaml",
    ROOT / "run_trace_policy_stage1_full.sh",
    ROOT / "run_trace_policy_stage2_full.sh",
    ROOT / "run_trace_policy_full_evidence.sh",
    PIPELINE,
)


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
        row["index"]
        for row in rows
        if row["memory_used_mib"] <= memory_threshold_mib
    ]
    return sorted(eligible)[:count]


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
            f"TRACE-Policy contract audit failed with code "
            f"{completed.returncode}"
        )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("TRACE-Policy contract audit did not pass")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--required-gpus", type=int, default=4)
    parser.add_argument("--memory-threshold-mib", type=int, default=1024)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.required_gpus != 4:
        raise SystemExit("The formal TRACE-Policy run requires exactly four GPUs")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = (STATE_ROOT / ".formal_full_pipeline.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A TRACE-Policy formal supervisor is already active")

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

    run_tag = (
        datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        + f"_trace_policy_full_seed{args.train_seed}"
    )
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
    report = run_contract_audit(run_dir)
    append_log(
        log_path,
        f"Preflight contract audit passed ({report['passed']}/"
        f"{report['total']}).",
    )

    state = {
        "model": "TRACE-Policy",
        "run_tag": run_tag,
        "supervisor_pid": os.getpid(),
        "state": "waiting_for_four_empty_gpus",
        "required_gpus": args.required_gpus,
        "memory_threshold_mib": args.memory_threshold_mib,
        "poll_seconds": args.poll_seconds,
        "train_seed": args.train_seed,
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
            os.killpg(child.pid, signal.SIGTERM)

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
    environment = os.environ.copy()
    environment.update(
        {
            "PIPELINE_TAG": run_tag,
            "TRAIN_SEED": str(args.train_seed),
            "PYTHONUNBUFFERED": "1",
        }
    )
    append_log(
        log_path,
        f"Four empty GPUs detected once; launching formal pipeline on "
        f"{gpu_csv}.",
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
        return_code = child.wait()

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
