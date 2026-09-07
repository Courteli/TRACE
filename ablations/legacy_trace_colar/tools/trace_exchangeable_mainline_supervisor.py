#!/usr/bin/env python3
"""Durable supervisor for one complete TRACE Stage 1 -> Stage 2 mainline."""

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
from typing import Dict, List, Sequence


ROOT = Path("/disk1/dingxukai/trace_colar")
PYTHON = Path("/home/dingxukai/miniconda3/envs/ROT/bin/python")
PIPELINE_ID = os.environ.get(
    "PIPELINE_ID", f"{datetime.now():%Y%m%d-%H%M%S}_trace_mainline_seed0"
)
RUN_ROOT = Path(
    os.environ.get(
        "PIPELINE_ROOT",
        ROOT / "run_outputs" / "trace_exchangeable" / "mainline" / PIPELINE_ID,
    )
)
TRAIN_SEED = 0
MEMORY_LIMIT_MIB = 1024
UTIL_LIMIT = 5
STABLE_POLLS = 3
POLL_SECONDS = 60
MIN_DISK_GIB = 100

STAGE1_SCRIPT = ROOT / "run_trace_exchangeable_stage1_best_20260718.sh"
STAGE2_SCRIPT = ROOT / "run_trace_exchangeable_stage2_best_20260718.sh"
EVIDENCE_SCRIPT = ROOT / "run_trace_exchangeable_evidence_20260718.sh"
ATTACH_STAGE1_PID = int(os.environ.get("ATTACH_STAGE1_PID", "0"))
ATTACH_STAGE1_RUN_TAG = os.environ.get("ATTACH_STAGE1_RUN_TAG", "")
STAGE0 = Path(
    "/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/"
    "20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/"
    "checkpoints/epoch0__step6726__monitor0.871.ckpt"
)
EXPECTED_DATA = {
    "/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc/train.json": 6726,
    "/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc/val.json": 747,
    "/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc/test.json": 1319,
    "/home/dingxukai/RoT/data/GSM8k-Hard/readcot_qsa_test/test.json": 1319,
    "/home/dingxukai/RoT/data/SVAMP/readcot_qsa_test/test.json": 1000,
    "/home/dingxukai/RoT/data/Multiarith/readcot_qsa_test/test.json": 180,
}


class PipelineError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class MainlineSupervisor:
    def __init__(self):
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        (RUN_ROOT / "logs").mkdir(exist_ok=True)
        (RUN_ROOT / "training").mkdir(exist_ok=True)
        (RUN_ROOT / "checkpoints").mkdir(exist_ok=True)
        self.state_path = RUN_ROOT / "state.json"
        self.events_path = RUN_ROOT / "events.log"
        self.children: Dict[str, subprocess.Popen] = {}
        self.stop_requested = False
        self.state = {
            "pipeline_id": PIPELINE_ID,
            "pid": os.getpid(),
            "status": "INITIALIZING",
            "phase": "preflight",
            "created_at": timestamp(),
            "updated_at": timestamp(),
            "run_root": str(RUN_ROOT),
            "training_seed_count": 1,
            "training_seed": TRAIN_SEED,
            "rollout_seed_count": 1,
            "rollout_seed": 271828,
            "sequence": [
                "Stage1 formation: up to 6726 x 10 full epochs, full validation every epoch, patience 4",
                "Select the best Stage1 checkpoint by validation accuracy",
                "Stage2 full TRACE: 2048 x 10 epochs, full validation every epoch",
                "Select the single best Stage2 checkpoint by validation accuracy",
                "Evaluate only that best checkpoint on full tasks and 200-question geometry",
            ],
            "active_jobs": {},
            "checkpoints": {},
            "message": "",
        }
        atomic_json(self.state_path, self.state)

        lock_path = (
            ROOT
            / "run_outputs"
            / "trace_exchangeable"
            / "mainline"
            / ".mainline_supervisor.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = lock_path.open("a+")
        try:
            fcntl.flock(
                self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise PipelineError("Another TRACE mainline supervisor is active") from exc
        self.lock_handle.seek(0)
        self.lock_handle.truncate()
        self.lock_handle.write(f"{os.getpid()} {PIPELINE_ID}\n")
        self.lock_handle.flush()

        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def log(self, message: str):
        line = f"{timestamp()} {message}"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def update(self, **changes):
        self.state.update(changes)
        self.state["updated_at"] = timestamp()
        atomic_json(self.state_path, self.state)

    def handle_signal(self, signum, _frame):
        self.stop_requested = True
        self.log(f"Received signal {signum}; stopping owned jobs.")
        for process in self.children.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def free_disk_gib(self) -> float:
        stats = os.statvfs("/disk1")
        return stats.f_bavail * stats.f_frsize / (1024**3)

    def ensure_disk(self):
        free = self.free_disk_gib()
        if free < MIN_DISK_GIB:
            raise PipelineError(
                f"/disk1 free space is {free:.1f} GiB, below {MIN_DISK_GIB} GiB"
            )

    def gpu_snapshot(self) -> List[dict]:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in output.splitlines():
            index, total, used, utilization = [
                int(item.strip()) for item in line.split(",")
            ]
            rows.append(
                {
                    "index": index,
                    "memory_total_mib": total,
                    "memory_used_mib": used,
                    "utilization_percent": utilization,
                }
            )
        return rows

    def clean_gpus(self, snapshot: Sequence[dict]) -> List[int]:
        return [
            gpu["index"]
            for gpu in snapshot
            if gpu["memory_total_mib"] >= 24000
            and gpu["memory_used_mib"] <= MEMORY_LIMIT_MIB
            and gpu["utilization_percent"] <= UTIL_LIMIT
        ]

    def wait_for_gpus(self, count: int, phase: str) -> List[int]:
        stable: List[int] = []
        stable_count = 0
        last_free = None
        while not self.stop_requested:
            self.ensure_disk()
            snapshot = self.gpu_snapshot()
            free = self.clean_gpus(snapshot)
            candidate = free[:count]
            if len(candidate) == count and candidate == stable:
                stable_count += 1
            elif len(candidate) == count:
                stable = candidate
                stable_count = 1
            else:
                stable = []
                stable_count = 0
            if free != last_free:
                self.log(
                    f"{phase}: clean GPUs={free}, need={count}, "
                    f"stable={stable_count}/{STABLE_POLLS}"
                )
                last_free = list(free)
            self.update(
                status="WAITING_FOR_GPUS",
                phase=phase,
                clean_gpus=free,
                stable_polls=stable_count,
                gpu_snapshot=snapshot,
                message=f"Waiting for {count} clean GPUs; no scale reduction.",
            )
            if stable_count >= STABLE_POLLS:
                final_free = self.clean_gpus(self.gpu_snapshot())
                if all(gpu in final_free for gpu in stable):
                    return stable
                stable = []
                stable_count = 0
            time.sleep(POLL_SECONDS)
        raise PipelineError("Stop requested while waiting for GPUs")

    def preflight(self):
        self.ensure_disk()
        if not STAGE0.is_file():
            raise PipelineError(f"Missing Stage0 checkpoint: {STAGE0}")
        for raw_path, expected in EXPECTED_DATA.items():
            path = Path(raw_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or len(payload) != expected:
                raise PipelineError(
                    f"{path}: expected {expected} records, found {len(payload)}"
                )
        log_path = RUN_ROOT / "logs" / "preflight.log"
        commands = [
            [
                "bash",
                "-n",
                str(STAGE1_SCRIPT),
                str(STAGE2_SCRIPT),
                str(EVIDENCE_SCRIPT),
            ],
            [
                str(PYTHON),
                "-m",
                "unittest",
                "-q",
                "tests.test_trace_exchangeable",
            ],
            [
                str(PYTHON),
                "-m",
                "py_compile",
                "src/models/trace_exchangeable.py",
                "tools/trace_exchangeable_mainline_summary.py",
            ],
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            for command in commands:
                handle.write("$ " + " ".join(command) + "\n")
                handle.flush()
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if result.returncode:
                    raise PipelineError(
                        f"Preflight failed: {' '.join(command)}"
                    )
        protocol = {
            "training_seeds": [TRAIN_SEED],
            "stage1": {
                "formation": True,
                "max_epochs": 10,
                "questions_per_epoch": 6726,
                "tiny_dataset": False,
                "validation": "all 747 validation questions after every epoch",
                "early_stopping": {
                    "monitor": "validation accuracy",
                    "mode": "max",
                    "patience": 4,
                },
                "checkpoint_selection": "highest validation accuracy",
                "saved_checkpoints": ["best.ckpt", "last.ckpt"],
            },
            "stage2": {
                "local_ranking": True,
                "epochs": 10,
                "questions_per_epoch": 2048,
                "group_size": 8,
                "gpus": 4,
                "validation": "all 747 validation questions after every epoch",
                "checkpoint_selection": "highest validation accuracy",
                "saved_checkpoints": ["best.ckpt", "last.ckpt"],
            },
            "formal_evaluation": {
                "checkpoints": ["best Stage2 only"],
                "test_times": 1,
                "full_question_counts": {
                    "GSM8K": 1319,
                    "GSMHard": 1319,
                    "SVAMP": 1000,
                    "MultiArith": 180,
                },
            },
            "geometry": {
                "questions": 200,
                "views_per_question": 8,
                "rollout_seeds": [271828],
            },
            "excluded": [
                "S1P",
                "M00",
                "M01",
                "M10",
                "2x2 package",
                "eligibility pilot",
                "DDP smoke",
                "Stage1 formal test",
                "additional training seeds",
                "second geometry rollout seed",
            ],
        }
        atomic_json(RUN_ROOT / "protocol.json", protocol)
        self.log("Preflight PASS for the single Stage1 -> Stage2 mainline.")

    def run_jobs(self, phase: str, jobs: Sequence[dict], gpus: Sequence[int]):
        self.ensure_disk()
        self.children = {}
        handles = {}
        active = {}
        self.update(
            status="RUNNING",
            phase=phase,
            assigned_gpus=list(gpus),
            message="Running the frozen full-budget mainline.",
        )
        for job in jobs:
            label = job["label"]
            log_path = RUN_ROOT / "logs" / f"{label}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles[label] = handle
            environment = os.environ.copy()
            environment.update(job.get("env", {}))
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                [str(item) for item in job["command"]],
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.children[label] = process
            active[label] = {
                "pid": process.pid,
                "log": str(log_path),
                "command": [str(item) for item in job["command"]],
            }
        self.update(active_jobs=active)
        self.log(f"{phase}: started {list(active)} on GPUs {list(gpus)}")

        last_heartbeat = 0.0
        while any(process.poll() is None for process in self.children.values()):
            if self.stop_requested:
                raise PipelineError(f"Stop requested during {phase}")
            if time.time() - last_heartbeat >= 60:
                self.update(
                    active_jobs={
                        label: {
                            **active[label],
                            "returncode": process.poll(),
                        }
                        for label, process in self.children.items()
                    },
                    gpu_snapshot=self.gpu_snapshot(),
                )
                last_heartbeat = time.time()
            time.sleep(10)

        failures = {}
        for label, process in self.children.items():
            returncode = process.wait()
            handles[label].close()
            if returncode:
                failures[label] = returncode
        self.children = {}
        self.update(active_jobs={})
        if failures:
            raise PipelineError(f"{phase} failed: {failures}")
        self.log(f"{phase}: completed.")

    def best_stage1_checkpoint(self, run_tag: str) -> Path:
        manifest = RUN_ROOT / "training" / run_tag / "manifest.txt"
        if not manifest.is_file():
            raise PipelineError(f"Missing Stage1 manifest: {manifest}")
        values = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        path = Path(values.get("best_checkpoint", ""))
        try:
            epoch = int(values["best_epoch"])
            step = int(values["best_step"])
            monitor = float(values["best_monitor"])
        except (KeyError, ValueError) as exc:
            raise PipelineError("Stage1 manifest lacks valid best-checkpoint metadata") from exc
        if (
            not path.is_file()
            or epoch not in range(10)
            or step != (epoch + 1) * 6726
        ):
            raise PipelineError(f"Invalid best Stage1 checkpoint: {path}")
        link = RUN_ROOT / "checkpoints" / "best_Stage1.ckpt"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(path)
        checkpoints = dict(self.state.get("checkpoints", {}))
        checkpoints["best_Stage1"] = str(path)
        checkpoints["best_Stage1_epoch"] = epoch
        checkpoints["best_Stage1_monitor"] = monitor
        self.update(checkpoints=checkpoints)
        return path

    def best_stage2_checkpoint(self, run_tag: str) -> Path:
        manifest = RUN_ROOT / "training" / run_tag / "manifest.txt"
        if not manifest.is_file():
            raise PipelineError(f"Missing Stage2 manifest: {manifest}")
        values = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        path = Path(values.get("best_checkpoint", ""))
        try:
            epoch = int(values["best_epoch"])
            step = int(values["best_step"])
            float(values["best_monitor"])
        except (KeyError, ValueError) as exc:
            raise PipelineError("Stage2 manifest lacks valid best-checkpoint metadata") from exc
        if (
            not path.is_file()
            or epoch not in range(10)
            or step != (epoch + 1) * 512
        ):
            raise PipelineError(f"Invalid best Stage2 checkpoint: {path}")
        link = RUN_ROOT / "checkpoints" / "best_Stage2.ckpt"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(path)
        checkpoints = dict(self.state.get("checkpoints", {}))
        checkpoints["best_Stage2"] = str(path)
        checkpoints["best_epoch"] = epoch
        checkpoints["best_monitor"] = float(values["best_monitor"])
        self.update(checkpoints=checkpoints)
        return path

    def wait_for_attached_stage1(self, pid: int, run_tag: str) -> Path:
        if pid <= 0 or not run_tag:
            raise PipelineError("Attach mode requires a Stage1 PID and run tag")
        self.log(f"Attached to the already-running Stage1 process {pid}.")
        while Path(f"/proc/{pid}").exists():
            if self.stop_requested:
                raise PipelineError("Stop requested while attached to Stage1")
            self.update(
                status="RUNNING",
                phase="Stage1_full",
                assigned_gpus=[0],
                active_jobs={
                    "train_Stage1": {
                        "pid": pid,
                        "mode": "attached_existing_full_run",
                    }
                },
                gpu_snapshot=self.gpu_snapshot(),
                message="Waiting for the existing complete Stage1 run.",
            )
            time.sleep(60)
        self.update(active_jobs={})
        return self.best_stage1_checkpoint(run_tag)

    def run(self):
        self.preflight()

        if ATTACH_STAGE1_PID:
            stage1_tag = ATTACH_STAGE1_RUN_TAG
            stage1 = self.wait_for_attached_stage1(
                ATTACH_STAGE1_PID,
                stage1_tag,
            )
        else:
            initial_gpus = self.wait_for_gpus(4, "mainline_start_gate")
            stage1_tag = f"{PIPELINE_ID}_Stage1_seed0"
            self.run_jobs(
                "Stage1_full",
                [
                    {
                        "label": "train_Stage1",
                        "command": [
                            STAGE1_SCRIPT,
                            str(initial_gpus[0]),
                        ],
                        "env": {
                            "RUN_ROOT": str(RUN_ROOT / "training"),
                            "RUN_TAG": stage1_tag,
                            "TRAIN_SEED": "0",
                        },
                    }
                ],
                [initial_gpus[0]],
            )
            stage1 = self.best_stage1_checkpoint(stage1_tag)

        stage2_gpus = self.wait_for_gpus(4, "Stage2_full")
        final_tag = f"{PIPELINE_ID}_Stage2_best_seed0"
        self.run_jobs(
            "Stage2_full",
            [
                {
                    "label": "train_TRACE",
                    "command": [
                        STAGE2_SCRIPT,
                        ",".join(map(str, stage2_gpus)),
                        str(stage1),
                    ],
                    "env": {
                        "RUN_ROOT": str(RUN_ROOT / "training"),
                        "RUN_TAG": final_tag,
                        "TRAIN_SEED": "0",
                    },
                }
            ],
            stage2_gpus,
        )
        final = self.best_stage2_checkpoint(final_tag)

        evidence_gpus = self.wait_for_gpus(1, "best_Stage2_full_evidence")
        evidence = RUN_ROOT / "evidence" / "best_Stage2"
        self.run_jobs(
            "best_Stage2_full_evidence",
            [
                {
                    "label": "evidence_best_Stage2",
                    "command": [
                        EVIDENCE_SCRIPT,
                        str(evidence_gpus[0]),
                        str(final),
                        f"{PIPELINE_ID}_best_Stage2",
                    ],
                    "env": {
                        "OUT_ROOT": str(evidence),
                        "RUN_SECOND_GSM_GEOMETRY_SEED": "false",
                    },
                },
            ],
            evidence_gpus,
        )

        self.update(
            status="COMPLETE",
            phase="complete",
            assigned_gpus=[],
            active_jobs={},
            completed_at=timestamp(),
            message="Best Stage2 checkpoint and its complete evidence package completed.",
        )
        (RUN_ROOT / "PIPELINE_COMPLETE").write_text(
            timestamp() + "\n", encoding="utf-8"
        )
        self.log("MAINLINE COMPLETE.")


def main() -> int:
    supervisor = None
    try:
        supervisor = MainlineSupervisor()
        supervisor.run()
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if supervisor is not None:
            supervisor.log("MAINLINE STOPPED: " + message)
            supervisor.update(
                status="FAILED",
                assigned_gpus=[],
                active_jobs={},
                failed_at=timestamp(),
                message=message,
            )
        else:
            print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
