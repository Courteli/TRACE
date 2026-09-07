#!/usr/bin/env python3
"""Wait for the mainline, then run its single-seed paired proof package."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/disk1/dingxukai/trace_colar")
MAINLINE_ROOT = Path(
    os.environ.get(
        "MAINLINE_ROOT",
        ROOT
        / "run_outputs"
        / "trace_exchangeable"
        / "mainline"
        / "20260718-1749_trace_mainline_full10_seed0",
    )
)
PROOF_ROOT = MAINLINE_ROOT / "paired_proof"
SCRIPT = ROOT / "run_trace_exchangeable_paired_proof_20260718.sh"
STATE_PATH = PROOF_ROOT / "state.json"
EVENTS_PATH = PROOF_ROOT / "events.log"
POLL_SECONDS = 60
MEMORY_LIMIT_MIB = 1024
UTIL_LIMIT = 5
STABLE_POLLS = 3


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class Supervisor:
    def __init__(self):
        PROOF_ROOT.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.child = None
        self.state = {
            "pid": os.getpid(),
            "status": "WAITING_FOR_MAINLINE",
            "mainline_root": str(MAINLINE_ROOT),
            "proof_root": str(PROOF_ROOT),
            "training_seeds": 1,
            "rollout_seeds": 1,
            "test_times": 1,
            "excluded": [
                "additional training seeds",
                "2x2 training package",
                "matched answer-only training branch",
                "component-by-component training ablations",
            ],
            "evidence": [
                "paired Stage1-to-Final full-task accuracy and length",
                "paired 200-question ordered-path and outcome-geometry statistics",
                "complete-path global-PCA figures",
                "correct/wrong complete-path distance heatmaps",
            ],
            "updated_at": timestamp(),
        }
        atomic_json(STATE_PATH, self.state)
        lock_path = MAINLINE_ROOT / ".paired_proof.lock"
        self.lock_handle = lock_path.open("a+")
        fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock_handle.seek(0)
        self.lock_handle.truncate()
        self.lock_handle.write(f"{os.getpid()}\n")
        self.lock_handle.flush()
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def log(self, message: str) -> None:
        line = f"{timestamp()} {message}"
        with EVENTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def update(self, **changes) -> None:
        self.state.update(changes)
        self.state["updated_at"] = timestamp()
        atomic_json(STATE_PATH, self.state)

    def handle_signal(self, signum, _frame) -> None:
        self.stop_requested = True
        self.log(f"Received signal {signum}.")
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    @staticmethod
    def gpu_snapshot() -> list[dict]:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in output.splitlines():
            index, used, utilization = [int(item.strip()) for item in line.split(",")]
            rows.append(
                {
                    "index": index,
                    "memory_used_mib": used,
                    "utilization_percent": utilization,
                }
            )
        return rows

    @staticmethod
    def clean_gpus(snapshot: list[dict]) -> list[int]:
        return [
            row["index"]
            for row in snapshot
            if row["memory_used_mib"] <= MEMORY_LIMIT_MIB
            and row["utilization_percent"] <= UTIL_LIMIT
        ]

    def wait_for_mainline(self) -> None:
        while not (MAINLINE_ROOT / "PIPELINE_COMPLETE").is_file():
            if self.stop_requested:
                raise RuntimeError("Stop requested while waiting for mainline")
            mainline_state_path = MAINLINE_ROOT / "state.json"
            mainline_state = {}
            if mainline_state_path.is_file():
                mainline_state = json.loads(
                    mainline_state_path.read_text(encoding="utf-8")
                )
                if mainline_state.get("status") == "FAILED":
                    raise RuntimeError("Mainline failed before paired proof")
            self.update(
                status="WAITING_FOR_MAINLINE",
                mainline_phase=mainline_state.get("phase"),
                mainline_status=mainline_state.get("status"),
            )
            time.sleep(POLL_SECONDS)

    def wait_for_gpu(self) -> int:
        stable_gpu = None
        stable_count = 0
        while not self.stop_requested:
            snapshot = self.gpu_snapshot()
            clean = self.clean_gpus(snapshot)
            candidate = clean[0] if clean else None
            if candidate is not None and candidate == stable_gpu:
                stable_count += 1
            elif candidate is not None:
                stable_gpu = candidate
                stable_count = 1
            else:
                stable_gpu = None
                stable_count = 0
            self.update(
                status="WAITING_FOR_GPU",
                clean_gpus=clean,
                stable_polls=stable_count,
                gpu_snapshot=snapshot,
            )
            if stable_count >= STABLE_POLLS:
                return int(stable_gpu)
            time.sleep(POLL_SECONDS)
        raise RuntimeError("Stop requested while waiting for a proof GPU")

    def run(self) -> None:
        if (PROOF_ROOT / "PAIRED_PROOF_COMPLETE").is_file():
            self.update(status="COMPLETE", message="Paired proof already exists.")
            return
        self.wait_for_mainline()
        stage1 = MAINLINE_ROOT / "checkpoints" / "best_Stage1.ckpt"
        final_evidence = MAINLINE_ROOT / "evidence" / "best_Stage2"
        if not stage1.is_file():
            raise RuntimeError(f"Missing best Stage1 checkpoint: {stage1}")
        if not (final_evidence / "evidence_done.txt").is_file():
            raise RuntimeError("Mainline completed without final evidence")
        gpu = self.wait_for_gpu()
        log_path = PROOF_ROOT / "paired_proof.log"
        with log_path.open("a", encoding="utf-8") as handle:
            self.child = subprocess.Popen(
                [
                    str(SCRIPT),
                    str(gpu),
                    str(stage1),
                    str(final_evidence),
                    str(PROOF_ROOT),
                    MAINLINE_ROOT.name,
                ],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.update(
                status="RUNNING",
                assigned_gpu=gpu,
                child_pid=self.child.pid,
                log=str(log_path),
            )
            returncode = self.child.wait()
        if returncode:
            raise RuntimeError(f"Paired proof exited with code {returncode}")
        self.update(
            status="COMPLETE",
            assigned_gpu=None,
            completed_at=timestamp(),
            message="Single-seed paired proof package completed.",
        )
        self.log("PAIRED PROOF COMPLETE.")


def main() -> int:
    supervisor = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if supervisor is not None:
            supervisor.log("PAIRED PROOF STOPPED: " + message)
            supervisor.update(
                status="FAILED",
                failed_at=timestamp(),
                message=message,
            )
        else:
            print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
