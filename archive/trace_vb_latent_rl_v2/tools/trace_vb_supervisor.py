#!/usr/bin/env python3
"""Wait for four eligible GPUs and supervise the complete TRACE-VB pipeline."""

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


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/disk1/dingxukai/TRACE/trace_vb_runs")
DATA_ROOT = Path("/disk1/dingxukai/TRACE")
PYTHON = Path("/home/dingxukai/miniconda3/envs/ROT/bin/python")
PIPELINE = ROOT / "scripts/run_full_pipeline_vb.sh"
CONTRACT_AUDIT = ROOT / "tools/trace_vb_pipeline_contract_audit.py"
STATE_ROOT = ARTIFACT_ROOT / "supervisor"
MIN_FREE_MIB = 21500
CRITICAL_SOURCES = (
    ROOT / "run.py",
    ROOT / "src/models/trace_vb.py",
    ROOT / "src/models/trace_policy.py",
    ROOT / "src/modules/trace_vb.py",
    ROOT / "src/modules/trace_policy.py",
    ROOT / "src/datasets/gsm8k_aug_nl.py",
    ROOT / "src/configs/models/trace_vb_policy_qwen3_instruct.yaml",
    ROOT / "src/configs/datasets/gsm8k_aug_nl.yaml",
    ROOT / "scripts/trace_vb_common.sh",
    ROOT / "scripts/prepare_sufficiency_cache_vb.sh",
    ROOT / "scripts/run_stage1_vb.sh",
    ROOT / "scripts/run_stage2_vb.sh",
    ROOT / "scripts/run_evidence_vb.sh",
    ROOT / "scripts/run_full_pipeline_vb.sh",
    ROOT / "tools/build_trace_vb_sufficiency_cache.py",
    ROOT / "tools/trace_vb_pipeline_contract_audit.py",
    ROOT / "tools/trace_vb_stage1_smoke.py",
    ROOT / "tools/trace_vb_stage1_ddp_smoke.py",
    ROOT / "tools/trace_vb_stage2_smoke.py",
    ROOT / "tools/trace_vb_stage2_ddp_smoke.py",
    ROOT / "tools/trace_vb_supervisor.py",
    ROOT / "tools/trace_prepare_pca_fit_set.py",
    ROOT / "tools/trace_policy_task_summary.py",
    ROOT / "tools/trace_policy_geometry_summary.py",
    ROOT / "tools/trace_policy_stage_comparison.py",
    ROOT / "tools/trace_policy_causal_summary.py",
    ROOT / "tools/verify_evidence_complete.py",
    ROOT / "tests/test_trace_vb_components.py",
    ROOT / "tests/test_trace_vb_stage2_ddp_smoke.py",
    ROOT / "tests/test_trace_vb_sufficiency_cache.py",
)
FATAL_MARKERS = (
    "torch.OutOfMemoryError",
    "CUDA out of memory",
    "ChildFailedError",
    "ProcessRaisedException",
    "NCCL watchdog caught collective operation timeout",
)
FATAL_TAIL_LENGTH = max(len(marker) for marker in FATAL_MARKERS) - 1


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp()} {message}\n")
        handle.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict:
    missing = [str(path) for path in CRITICAL_SOURCES if not path.is_file()]
    if missing:
        raise RuntimeError("missing critical TRACE-VB sources: " + ", ".join(missing))
    return {str(path.relative_to(ROOT)): sha256(path) for path in CRITICAL_SOURCES}


def verify_source_hashes(expected: dict) -> None:
    current = source_hashes()
    changed = sorted(name for name, digest in expected.items() if current.get(name) != digest)
    if changed:
        raise RuntimeError("critical TRACE-VB sources changed while running: " + ", ".join(changed))


def gpu_snapshot() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = None
    for attempt in range(3):
        candidate = subprocess.run(command, check=False, capture_output=True, text=True)
        if candidate.returncode == 0:
            completed = candidate
            break
        if attempt < 2:
            time.sleep(1)
    if completed is None:
        raise RuntimeError(
            "nvidia-smi failed three times: "
            + (candidate.stderr.strip() or f"exit={candidate.returncode}")
        )
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "total_mib": int(fields[1]),
                "used_mib": int(fields[2]),
                "free_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
            }
        )
    return rows


def select_gpus(rows: list[dict], requested: list[int] | None) -> list[int]:
    by_index = {row["index"]: row for row in rows}
    candidates = requested if requested is not None else sorted(by_index)
    eligible = [index for index in candidates if index in by_index and by_index[index]["free_mib"] >= MIN_FREE_MIB]
    if requested is not None:
        return list(requested) if len(eligible) == 4 else []
    return eligible[:4]


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_group(child: subprocess.Popen, timeout: int = 20) -> None:
    group = int(child.pid)
    if not process_group_exists(group):
        return
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while process_group_exists(group) and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_group_exists(group):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            return


def run_preflight(run_dir: Path) -> None:
    with (run_dir / "preflight_unit_tests.log").open("w", encoding="utf-8") as output:
        subprocess.run(
            [str(PYTHON), "-m", "unittest", "discover", "-s", "tests"],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    with (run_dir / "preflight_contract_audit.json").open("w", encoding="utf-8") as output:
        subprocess.run(
            [str(PYTHON), str(CONTRACT_AUDIT)],
            cwd=ROOT,
            env={**os.environ, "TRACE_PROJECT_ROOT": str(ROOT), "TRACE_DATA_ROOT": str(DATA_ROOT)},
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-csv", help="four ordered physical GPU IDs")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    requested = None
    if args.gpu_csv:
        try:
            requested = [int(token.strip()) for token in args.gpu_csv.split(",") if token.strip()]
        except ValueError as error:
            raise SystemExit("--gpu-csv must contain integer IDs") from error
        if len(requested) != 4 or len(set(requested)) != 4:
            raise SystemExit("--gpu-csv must contain four unique IDs")
    if args.dry_run:
        rows = gpu_snapshot()
        selected = select_gpus(rows, requested)
        print(json.dumps({"gpu_snapshot": rows, "selected": selected, "minimum_free_mib": MIN_FREE_MIB, "would_start": len(selected) == 4}, indent=2))
        return 0

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = (STATE_ROOT / ".formal_trace_vb.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("a formal TRACE-VB supervisor is already active")

    run_tag = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_tag += f"_trace_vb_full_seed{args.train_seed}"
    run_dir = STATE_ROOT / run_tag
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "status.json"
    log_path = run_dir / "supervisor.log"
    (run_dir / "supervisor.pid").write_text(f"{os.getpid()}\n", encoding="ascii")
    latest = STATE_ROOT / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    frozen_hashes = source_hashes()
    write_json(run_dir / "source_hashes.json", frozen_hashes)
    run_preflight(run_dir)
    append_log(log_path, "unit tests and TRACE-VB contract audit passed")
    state = {
        "model": "TRACE-VB",
        "run_tag": run_tag,
        "supervisor_pid": os.getpid(),
        "state": "waiting_for_four_eligible_gpus",
        "requested_gpus": requested,
        "minimum_free_gpu_memory_mib": MIN_FREE_MIB,
        "poll_seconds": args.poll_seconds,
        "code_root": str(ROOT),
        "artifact_root": str(ARTIFACT_ROOT),
        "created_at": timestamp(),
        "updated_at": timestamp(),
    }
    write_json(status_path, state)
    child = None
    stop_requested = False

    def stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        append_log(log_path, f"received signal {signum}")
        if child is not None and child.poll() is None:
            terminate_group(child)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    selected = []
    while not stop_requested:
        try:
            rows = gpu_snapshot()
            selected = select_gpus(rows, requested)
            state.update({"gpu_snapshot": rows, "selected_gpus": selected, "updated_at": timestamp()})
            state.pop("last_gpu_query_error", None)
            write_json(status_path, state)
            if len(selected) == 4:
                break
        except Exception as error:
            state.update({"last_gpu_query_error": repr(error), "updated_at": timestamp()})
            write_json(status_path, state)
            append_log(log_path, f"GPU query failed: {error!r}")
        time.sleep(args.poll_seconds)
    if stop_requested:
        state.update({"state": "stopped", "updated_at": timestamp()})
        write_json(status_path, state)
        return 130

    verify_source_hashes(frozen_hashes)
    gpu_csv = ",".join(map(str, selected))
    pipeline_log_path = run_dir / "pipeline.stdout.log"
    environment = {
        **os.environ,
        "PIPELINE_TAG": run_tag,
        "TRAIN_SEED": str(args.train_seed),
        "PYTHONUNBUFFERED": "1",
        "TRACE_PROJECT_ROOT": str(ROOT),
        "TRACE_DATA_ROOT": str(DATA_ROOT),
    }
    append_log(log_path, f"launching complete TRACE-VB pipeline on {gpu_csv}")
    with pipeline_log_path.open("a", encoding="utf-8") as output:
        child = subprocess.Popen(
            ["bash", str(PIPELINE), gpu_csv, str(selected[0])],
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        state.update(
            {
                "state": "pipeline_running",
                "physical_training_gpus": selected,
                "cache_and_evidence_gpu": selected[0],
                "pipeline_pid": child.pid,
                "pipeline_log": str(pipeline_log_path),
                "pipeline_started_at": timestamp(),
                "updated_at": timestamp(),
            }
        )
        write_json(status_path, state)
        offset = 0
        log_tail = ""
        return_code = None
        while return_code is None:
            return_code = child.poll()
            with pipeline_log_path.open("r", encoding="utf-8", errors="replace") as reader:
                reader.seek(offset)
                new_text = reader.read()
                offset = reader.tell()
            combined = log_tail + new_text
            marker = next((item for item in FATAL_MARKERS if item in combined), None)
            log_tail = combined[-FATAL_TAIL_LENGTH:]
            try:
                verify_source_hashes(frozen_hashes)
                integrity_error = None
            except Exception as error:
                integrity_error = str(error)
            state.update({"pipeline_child_alive": return_code is None, "pipeline_log_bytes_checked": offset, "updated_at": timestamp()})
            write_json(status_path, state)
            if return_code is None and (marker or integrity_error or stop_requested):
                state.update(
                    {
                        "state": "pipeline_failing_fast",
                        "pipeline_failure_marker": marker,
                        "source_integrity_error": integrity_error,
                        "updated_at": timestamp(),
                    }
                )
                write_json(status_path, state)
                terminate_group(child)
                return_code = 130 if stop_requested else 86
                break
            if return_code is None:
                time.sleep(min(args.poll_seconds, 5))

    state.update(
        {
            "state": "pipeline_completed" if return_code == 0 else "pipeline_failed",
            "pipeline_return_code": return_code,
            "pipeline_finished_at": timestamp(),
            "updated_at": timestamp(),
        }
    )
    write_json(status_path, state)
    append_log(log_path, f"pipeline exited with return code {return_code}")
    return int(return_code)


if __name__ == "__main__":
    sys.exit(main())
