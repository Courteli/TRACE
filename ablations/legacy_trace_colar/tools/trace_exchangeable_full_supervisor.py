#!/usr/bin/env python3
"""Durable, fail-closed supervisor for the frozen TRACE 2x2 package."""

from __future__ import annotations

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
from typing import Dict, Iterable, List, Sequence


ROOT = Path(os.environ.get("TRACE_ROOT", "/disk1/dingxukai/trace_colar")).resolve()
PYTHON = Path(
    os.environ.get(
        "TRACE_PYTHON", "/home/dingxukai/miniconda3/envs/ROT/bin/python"
    )
)
PIPELINE_ID = os.environ.get(
    "PIPELINE_ID", f"{datetime.now():%Y%m%d-%H%M%S}_exchangeable_full_seed0"
)
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "0"))
RUN_ROOT = Path(
    os.environ.get(
        "PIPELINE_ROOT",
        ROOT / "run_outputs" / "trace_exchangeable" / "full_pipeline" / PIPELINE_ID,
    )
).resolve()
GLOBAL_LOCK = (
    ROOT
    / "run_outputs"
    / "trace_exchangeable"
    / "full_pipeline"
    / ".full_supervisor.lock"
)
STAGE0_CKPT = Path(
    os.environ.get(
        "STAGE0_CKPT",
        "/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/"
        "20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/"
        "checkpoints/epoch0__step6726__monitor0.871.ckpt",
    )
)

GPU_MEMORY_LIMIT_MIB = int(os.environ.get("GPU_MEMORY_LIMIT_MIB", "1024"))
GPU_UTIL_LIMIT_PERCENT = int(os.environ.get("GPU_UTIL_LIMIT_PERCENT", "5"))
GPU_STABLE_POLLS = int(os.environ.get("GPU_STABLE_POLLS", "3"))
GPU_POLL_SECONDS = int(os.environ.get("GPU_POLL_SECONDS", "60"))
MIN_FREE_DISK_GIB = int(os.environ.get("MIN_FREE_DISK_GIB", "100"))

PHASE_SCRIPT = ROOT / "run_trace_exchangeable_phase_20260718.sh"
EVIDENCE_SCRIPT = ROOT / "run_trace_exchangeable_evidence_20260718.sh"
ELIGIBILITY_SCRIPT = ROOT / "run_trace_exchangeable_eligibility_pilot_20260718.sh"
DDP_SMOKE_SCRIPT = ROOT / "run_trace_exchangeable_ddp_smoke_20260718.sh"
ANALYSIS_SCRIPT = ROOT / "run_trace_exchangeable_2x2_analysis_20260718.sh"

DATASETS = {
    "gsm8k_train": (
        Path("/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc/train.json"),
        6726,
    ),
    "gsm8k_val": (
        Path("/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc/val.json"),
        747,
    ),
    "gsm8k_test": (
        Path("/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc/test.json"),
        1319,
    ),
    "gsmhard_test": (
        Path("/home/dingxukai/RoT/data/GSM8k-Hard/readcot_qsa_test/test.json"),
        1319,
    ),
    "svamp_test": (
        Path("/home/dingxukai/RoT/data/SVAMP/readcot_qsa_test/test.json"),
        1000,
    ),
    "multiarith_test": (
        Path("/home/dingxukai/RoT/data/Multiarith/readcot_qsa_test/test.json"),
        180,
    ),
}


class PipelineError(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def tail(path: Path, lines: int = 80) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(content[-lines:])


class Supervisor:
    def __init__(self) -> None:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        (RUN_ROOT / "logs").mkdir(exist_ok=True)
        (RUN_ROOT / "checkpoints").mkdir(exist_ok=True)
        self.state_path = RUN_ROOT / "state.json"
        self.events_path = RUN_ROOT / "events.log"
        self.stop_requested = False
        self.children: Dict[str, subprocess.Popen] = {}
        self.completed = []
        self.checkpoints: Dict[str, str] = {}

        if self.state_path.exists():
            previous = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.completed = list(previous.get("completed_phases", []))
            self.checkpoints = dict(previous.get("checkpoints", {}))

        self.state = {
            "pipeline_id": PIPELINE_ID,
            "package": "one-seed decisive 2x2",
            "train_seed": TRAIN_SEED,
            "pid": os.getpid(),
            "created_at": now(),
            "updated_at": now(),
            "status": "INITIALIZING",
            "phase": "startup",
            "run_root": str(RUN_ROOT),
            "budgets": {
                "stage1": "2 branches; 6726 questions x 5 epochs; final full validation",
                "stage2": "4 cells; 2048 questions x 5 epochs; group_size=8; 4-GPU DDP",
                "task_eval": {
                    "GSM8K": 1319,
                    "GSMHard": 1319,
                    "SVAMP": 1000,
                    "MultiArith": 180,
                    "test_times": 1,
                },
                "geometry": "200 matched questions x 8 views; two GSM8K rollout seeds",
            },
            "completed_phases": self.completed,
            "checkpoints": self.checkpoints,
            "active_jobs": {},
            "assigned_gpus": [],
            "message": "",
        }
        self._write_state()

        GLOBAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = GLOBAL_LOCK.open("a+")
        try:
            fcntl.flock(
                self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise PipelineError(
                f"Another TRACE full supervisor holds {GLOBAL_LOCK}"
            ) from exc
        self.lock_handle.seek(0)
        self.lock_handle.truncate()
        self.lock_handle.write(f"{os.getpid()} {PIPELINE_ID}\n")
        self.lock_handle.flush()

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, _frame) -> None:
        self.stop_requested = True
        self.log(f"Received signal {signum}; terminating active children.")
        for process in self.children.values():
            if process.poll() is None:
                process.terminate()

    def log(self, message: str) -> None:
        line = f"{now()} {message}"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def _write_state(self, **updates) -> None:
        self.state.update(updates)
        self.state["updated_at"] = now()
        self.state["completed_phases"] = list(self.completed)
        self.state["checkpoints"] = dict(self.checkpoints)
        atomic_json(self.state_path, self.state)

    def mark_completed(self, phase: str) -> None:
        if phase not in self.completed:
            self.completed.append(phase)
        self._write_state()

    def disk_free_gib(self) -> float:
        stat = os.statvfs("/disk1")
        return stat.f_bavail * stat.f_frsize / (1024**3)

    def ensure_disk(self) -> None:
        free_gib = self.disk_free_gib()
        if free_gib < MIN_FREE_DISK_GIB:
            raise PipelineError(
                f"/disk1 has only {free_gib:.1f} GiB free; "
                f"the frozen floor is {MIN_FREE_DISK_GIB} GiB. "
                "No files were deleted."
            )

    def gpu_snapshot(self) -> List[dict]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(command, text=True, cwd=ROOT)
        snapshot = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 5:
                raise PipelineError(f"Unexpected nvidia-smi row: {line}")
            snapshot.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                    "utilization_percent": int(fields[4]),
                }
            )
        return snapshot

    def clean_gpu_ids(self, snapshot: Sequence[dict]) -> List[int]:
        return [
            gpu["index"]
            for gpu in snapshot
            if gpu["memory_total_mib"] >= 24000
            and gpu["memory_used_mib"] <= GPU_MEMORY_LIMIT_MIB
            and gpu["utilization_percent"] <= GPU_UTIL_LIMIT_PERCENT
        ]

    def wait_for_gpus(self, count: int, phase: str) -> List[int]:
        stable_candidate: List[int] = []
        stable_polls = 0
        last_free: List[int] | None = None
        poll_index = 0
        while not self.stop_requested:
            self.ensure_disk()
            snapshot = self.gpu_snapshot()
            free = self.clean_gpu_ids(snapshot)
            candidate = free[:count]
            if len(candidate) == count and candidate == stable_candidate:
                stable_polls += 1
            elif len(candidate) == count:
                stable_candidate = candidate
                stable_polls = 1
            else:
                stable_candidate = []
                stable_polls = 0

            if free != last_free or poll_index % 10 == 0:
                self.log(
                    f"{phase}: clean GPUs={free}; need {count}; "
                    f"stable={stable_polls}/{GPU_STABLE_POLLS}"
                )
                last_free = list(free)
            self._write_state(
                status="WAITING_FOR_GPUS",
                phase=phase,
                assigned_gpus=[],
                gpu_snapshot=snapshot,
                clean_gpus=free,
                stable_polls=stable_polls,
                message=(
                    f"Waiting for {count} clean 24 GB GPUs. "
                    f"No budget reduction is allowed."
                ),
            )
            if stable_polls >= GPU_STABLE_POLLS:
                final_snapshot = self.gpu_snapshot()
                final_free = self.clean_gpu_ids(final_snapshot)
                if all(gpu in final_free for gpu in stable_candidate):
                    self.log(f"{phase}: acquired clean GPUs {stable_candidate}")
                    return stable_candidate
                stable_candidate = []
                stable_polls = 0
            poll_index += 1
            time.sleep(GPU_POLL_SECONDS)
        raise PipelineError("Supervisor stop requested while waiting for GPUs")

    def relevant_source_files(self) -> List[Path]:
        files = [ROOT / "run.py", ROOT / "TRACE_EXCHANGEABLE_LOCAL_RANKING_PROTOCOL.md"]
        files.extend(
            path
            for path in (ROOT / "src").rglob("*")
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
        )
        files.extend(
            [
                PHASE_SCRIPT,
                EVIDENCE_SCRIPT,
                ELIGIBILITY_SCRIPT,
                DDP_SMOKE_SCRIPT,
                ANALYSIS_SCRIPT,
                ROOT / "tools" / "trace_exchangeable_geometry_summary.py",
                ROOT / "tools" / "trace_exchangeable_2x2_task_summary.py",
                ROOT / "tools" / "trace_exchangeable_eligibility_gate.py",
                ROOT / "tools" / "trace_exchangeable_visualize.py",
                ROOT / "tests" / "test_trace_exchangeable.py",
                Path(__file__).resolve(),
            ]
        )
        return sorted(set(path.resolve() for path in files))

    def build_frozen_manifest(self) -> dict:
        source_files = self.relevant_source_files()
        for path in source_files:
            if not path.is_file():
                raise PipelineError(f"Missing frozen source input: {path}")
        dataset_payload = {}
        for label, (path, expected_count) in DATASETS.items():
            if not path.is_file():
                raise PipelineError(f"Missing dataset: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or len(payload) != expected_count:
                raise PipelineError(
                    f"{label} must contain exactly {expected_count} records; "
                    f"found {len(payload) if isinstance(payload, list) else type(payload)}"
                )
            dataset_payload[label] = {
                "path": str(path),
                "count": expected_count,
                "sha256": sha256(path),
            }
        if not STAGE0_CKPT.is_file():
            raise PipelineError(f"Missing Stage 0 checkpoint: {STAGE0_CKPT}")
        return {
            "created_at": now(),
            "pipeline_id": PIPELINE_ID,
            "train_seed": TRAIN_SEED,
            "source": {str(path): sha256(path) for path in source_files},
            "datasets": dataset_payload,
            "stage0_checkpoint": {
                "path": str(STAGE0_CKPT),
                "bytes": STAGE0_CKPT.stat().st_size,
                "sha256": sha256(STAGE0_CKPT),
            },
            "hard_constraints": {
                "tiny_dataset": False,
                "epoch_scaling": 1,
                "stage1_train_questions_per_epoch": 6726,
                "stage1_epochs": 5,
                "stage2_train_questions_per_epoch": 2048,
                "stage2_epochs": 5,
                "stage2_group_size": 8,
                "stage2_gpu_count": 4,
                "test_times": 1,
                "formal_eval_is_full": True,
                "geometry_question_count": 200,
            },
        }

    def verify_frozen(self) -> None:
        manifest_path = RUN_ROOT / "frozen_manifest.json"
        if not manifest_path.is_file():
            raise PipelineError(f"Missing frozen manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed = []
        for raw_path, expected_hash in manifest["source"].items():
            path = Path(raw_path)
            if not path.is_file() or sha256(path) != expected_hash:
                changed.append(str(path))
        for item in manifest["datasets"].values():
            path = Path(item["path"])
            if not path.is_file() or sha256(path) != item["sha256"]:
                changed.append(str(path))
        stage0 = manifest["stage0_checkpoint"]
        stage0_path = Path(stage0["path"])
        if not stage0_path.is_file() or sha256(stage0_path) != stage0["sha256"]:
            changed.append(str(stage0_path))
        if changed:
            raise PipelineError(
                "Frozen inputs changed after supervision began: "
                + ", ".join(changed[:10])
            )

    def run_preflight(self) -> None:
        if "preflight" in self.completed:
            self.verify_frozen()
            return
        self.ensure_disk()
        preflight_log = RUN_ROOT / "logs" / "preflight.log"
        commands = [
            [
                "bash",
                "-n",
                str(PHASE_SCRIPT),
                str(EVIDENCE_SCRIPT),
                str(ELIGIBILITY_SCRIPT),
                str(DDP_SMOKE_SCRIPT),
                str(ANALYSIS_SCRIPT),
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
                "tools/trace_exchangeable_geometry_summary.py",
                "tools/trace_exchangeable_2x2_task_summary.py",
                "tools/trace_exchangeable_eligibility_gate.py",
                "tools/trace_exchangeable_visualize.py",
            ],
        ]
        self.log("Running static preflight and freezing full-data inputs.")
        with preflight_log.open("w", encoding="utf-8") as handle:
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
                if result.returncode != 0:
                    raise PipelineError(
                        f"Preflight command failed ({result.returncode}): "
                        + " ".join(command)
                    )
        manifest = self.build_frozen_manifest()
        atomic_json(RUN_ROOT / "frozen_manifest.json", manifest)
        self.mark_completed("preflight")
        self.log(
            "Preflight PASS: full dataset counts, scripts, unit tests, source, "
            "Stage 0, and disk floor are frozen."
        )

    def run_jobs(
        self,
        phase: str,
        jobs: Sequence[dict],
        assigned_gpus: Sequence[int],
    ) -> None:
        self.verify_frozen()
        self.ensure_disk()
        self.children = {}
        handles = {}
        active = {}
        self._write_state(
            status="RUNNING",
            phase=phase,
            assigned_gpus=list(assigned_gpus),
            message="Running frozen command(s); no automatic scale reduction.",
        )
        self.log(
            f"{phase}: starting {len(jobs)} job(s) on GPUs {list(assigned_gpus)}"
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
                [str(value) for value in job["command"]],
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
                "command": [str(value) for value in job["command"]],
                "log": str(log_path),
            }
        self._write_state(active_jobs=active)

        last_heartbeat = 0.0
        while True:
            unfinished = {
                label: process
                for label, process in self.children.items()
                if process.poll() is None
            }
            if not unfinished:
                break
            if self.stop_requested:
                raise PipelineError(f"Stop requested during {phase}")
            if time.time() - last_heartbeat >= 60:
                heartbeat = {}
                for label, process in self.children.items():
                    heartbeat[label] = {
                        **active[label],
                        "returncode": process.poll(),
                    }
                self._write_state(
                    active_jobs=heartbeat,
                    gpu_snapshot=self.gpu_snapshot(),
                )
                last_heartbeat = time.time()
            time.sleep(10)

        failures = {}
        for label, process in self.children.items():
            returncode = process.wait()
            handles[label].close()
            if returncode != 0:
                failures[label] = {
                    "returncode": returncode,
                    "tail": tail(Path(active[label]["log"])),
                }
        self.children = {}
        self._write_state(active_jobs={})
        if failures:
            raise PipelineError(
                f"{phase} failed; no downstream phase was launched:\n"
                + json.dumps(failures, indent=2)
            )
        self.log(f"{phase}: all jobs completed successfully.")

    def checkpoint_from_manifest(self, label: str, run_tag: str) -> Path:
        manifest_path = RUN_ROOT / "training" / run_tag / "manifest.txt"
        if not manifest_path.is_file():
            raise PipelineError(f"{label}: missing training manifest {manifest_path}")
        values = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        checkpoint = Path(values.get("checkpoint", ""))
        if not checkpoint.is_file():
            raise PipelineError(
                f"{label}: manifest does not name a valid final checkpoint"
            )
        expected_step = 33630 if label in {"S1P", "S1F"} else 2560
        if (
            "epoch4__step" + str(expected_step) + "__monitor"
            not in checkpoint.name
        ):
            raise PipelineError(
                f"{label}: unexpected checkpoint budget encoded by {checkpoint.name}"
            )
        link = RUN_ROOT / "checkpoints" / f"{label}.ckpt"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(checkpoint)
        self.checkpoints[label] = str(checkpoint)
        self._write_state()
        return checkpoint

    def existing_checkpoint(self, label: str) -> Path | None:
        raw = self.checkpoints.get(label)
        if raw and Path(raw).is_file():
            return Path(raw)
        link = RUN_ROOT / "checkpoints" / f"{label}.ckpt"
        if link.is_file():
            checkpoint = link.resolve()
            self.checkpoints[label] = str(checkpoint)
            return checkpoint
        return None

    def phase_environment(self, run_tag: str) -> dict:
        return {
            "RUN_ROOT": str(RUN_ROOT / "training"),
            "RUN_TAG": run_tag,
            "TRAIN_SEED": str(TRAIN_SEED),
            "MAX_STARTUP_MEMORY_MIB": str(GPU_MEMORY_LIMIT_MIB),
        }

    def train_stage1(self) -> Dict[str, Path]:
        definitions = {
            "S1P": "stage1-plain",
            "S1F": "stage1-formation",
        }
        missing = [
            label
            for label in definitions
            if self.existing_checkpoint(label) is None
        ]
        if missing:
            gpus = self.wait_for_gpus(4, "stage1_full_start_gate")
            jobs = []
            for index, label in enumerate(missing):
                run_tag = f"{PIPELINE_ID}_{label}_seed{TRAIN_SEED}"
                jobs.append(
                    {
                        "label": f"train_{label}",
                        "command": [
                            PHASE_SCRIPT,
                            definitions[label],
                            str(gpus[index]),
                        ],
                        "env": self.phase_environment(run_tag),
                    }
                )
            self.run_jobs("stage1_full", jobs, gpus[: len(missing)])
            for label in missing:
                run_tag = f"{PIPELINE_ID}_{label}_seed{TRAIN_SEED}"
                self.checkpoint_from_manifest(label, run_tag)
            self.mark_completed("stage1_full")
        return {
            label: self.existing_checkpoint(label)
            for label in definitions
        }

    def run_gates(self, s1f: Path) -> None:
        eligibility_root = RUN_ROOT / "gates" / "eligibility"
        eligibility_json = eligibility_root / "eligibility_gate.json"
        eligibility_passed = False
        if eligibility_json.is_file():
            eligibility_passed = (
                json.loads(eligibility_json.read_text(encoding="utf-8")).get("status")
                == "PASS"
            )
        if not eligibility_passed:
            gpu = self.wait_for_gpus(1, "stage1_eligibility_gate")
            self.run_jobs(
                "stage1_eligibility_gate",
                [
                    {
                        "label": "eligibility_gate",
                        "command": [
                            ELIGIBILITY_SCRIPT,
                            str(gpu[0]),
                            str(s1f),
                            f"{PIPELINE_ID}_S1F",
                        ],
                        "env": {
                            "OUT_ROOT": str(eligibility_root),
                            "MAX_STARTUP_MEMORY_MIB": str(
                                GPU_MEMORY_LIMIT_MIB
                            ),
                        },
                    }
                ],
                gpu,
            )
            payload = json.loads(eligibility_json.read_text(encoding="utf-8"))
            if payload.get("status") != "PASS":
                raise PipelineError("Stage 1 eligibility gate did not pass")
            self.mark_completed("stage1_eligibility_gate")

        smoke_root = RUN_ROOT / "gates" / "ddp_smoke"
        smoke_json = smoke_root / "smoke_audit.json"
        smoke_passed = False
        if smoke_json.is_file():
            smoke_passed = (
                json.loads(smoke_json.read_text(encoding="utf-8")).get("status")
                == "PASS"
            )
        if not smoke_passed:
            gpus = self.wait_for_gpus(4, "stage2_four_gpu_ddp_gate")
            self.run_jobs(
                "stage2_four_gpu_ddp_gate",
                [
                    {
                        "label": "ddp_smoke",
                        "command": [
                            DDP_SMOKE_SCRIPT,
                            ",".join(map(str, gpus)),
                            str(s1f),
                        ],
                        "env": {
                            "RUN_TAG": f"{PIPELINE_ID}_ddp_smoke",
                            "OUT_ROOT": str(smoke_root),
                            "MAX_STARTUP_MEMORY_MIB": str(
                                GPU_MEMORY_LIMIT_MIB
                            ),
                        },
                    }
                ],
                gpus,
            )
            payload = json.loads(smoke_json.read_text(encoding="utf-8"))
            if payload.get("status") != "PASS":
                raise PipelineError("Four-GPU DDP smoke gate did not pass")
            self.mark_completed("stage2_four_gpu_ddp_gate")

    def train_stage2(self, stage1: Dict[str, Path]) -> Dict[str, Path]:
        definitions = [
            ("M00", "stage2-answer", "S1P", "false"),
            ("M01", "stage2-full", "S1P", "false"),
            ("M10", "stage2-answer", "S1F", "true"),
            ("M11", "stage2-full", "S1F", "true"),
        ]
        for label, phase, parent, formation in definitions:
            if self.existing_checkpoint(label) is not None:
                continue
            gpus = self.wait_for_gpus(4, f"{label}_full_stage2")
            run_tag = f"{PIPELINE_ID}_{label}_seed{TRAIN_SEED}"
            self.run_jobs(
                f"{label}_full_stage2",
                [
                    {
                        "label": f"train_{label}",
                        "command": [
                            PHASE_SCRIPT,
                            phase,
                            ",".join(map(str, gpus)),
                            str(stage1[parent]),
                            formation,
                        ],
                        "env": self.phase_environment(run_tag),
                    }
                ],
                gpus,
            )
            self.checkpoint_from_manifest(label, run_tag)
            self.mark_completed(f"{label}_full_stage2")
        return {
            label: self.existing_checkpoint(label)
            for label, *_ in definitions
        }

    def run_evidence(self, checkpoints: Dict[str, Path]) -> Dict[str, Path]:
        evidence_roots = {
            label: RUN_ROOT / "evidence" / label
            for label in ("S1P", "S1F", "M00", "M01", "M10", "M11")
        }
        pending = [
            label
            for label, root in evidence_roots.items()
            if not (root / "evidence_done.txt").is_file()
        ]
        while pending:
            batch_size = min(4, len(pending))
            gpus = self.wait_for_gpus(batch_size, f"evidence_batch_{pending[0]}")
            batch = pending[:batch_size]
            jobs = []
            for gpu, label in zip(gpus, batch):
                jobs.append(
                    {
                        "label": f"evidence_{label}",
                        "command": [
                            EVIDENCE_SCRIPT,
                            str(gpu),
                            str(checkpoints[label]),
                            f"{PIPELINE_ID}_{label}",
                        ],
                        "env": {
                            "OUT_ROOT": str(evidence_roots[label]),
                            "MAX_STARTUP_MEMORY_MIB": str(
                                GPU_MEMORY_LIMIT_MIB
                            ),
                        },
                    }
                )
            self.run_jobs(f"evidence_{'_'.join(batch)}", jobs, gpus)
            for label in batch:
                done = evidence_roots[label] / "evidence_done.txt"
                if not done.is_file():
                    raise PipelineError(f"{label}: evidence marker missing")
                self.mark_completed(f"evidence_{label}")
            pending = pending[batch_size:]
        return evidence_roots

    def run_analysis(self, evidence: Dict[str, Path]) -> None:
        output = RUN_ROOT / "analysis_2x2"
        if (output / "analysis_done.txt").is_file():
            return
        self.run_jobs(
            "paired_2x2_analysis",
            [
                {
                    "label": "analysis_2x2",
                    "command": [
                        ANALYSIS_SCRIPT,
                        str(evidence["S1P"]),
                        str(evidence["S1F"]),
                        str(evidence["M00"]),
                        str(evidence["M01"]),
                        str(evidence["M10"]),
                        str(evidence["M11"]),
                        str(output),
                    ],
                }
            ],
            [],
        )
        if not (output / "analysis_done.txt").is_file():
            raise PipelineError("Final 2x2 analysis marker is missing")
        self.mark_completed("paired_2x2_analysis")

    def run(self) -> None:
        self.run_preflight()
        stage1 = self.train_stage1()
        self.run_gates(stage1["S1F"])
        stage2 = self.train_stage2(stage1)
        all_checkpoints = {**stage1, **stage2}
        evidence = self.run_evidence(all_checkpoints)
        self.run_analysis(evidence)
        self._write_state(
            status="COMPLETE",
            phase="complete",
            assigned_gpus=[],
            active_jobs={},
            message=(
                "The frozen one-seed full 2x2 training/evaluation package "
                "completed. Statistical acceptance must be read from analysis_2x2."
            ),
            completed_at=now(),
        )
        (RUN_ROOT / "PIPELINE_COMPLETE").write_text(now() + "\n", encoding="utf-8")
        self.log("PIPELINE COMPLETE. No test-based checkpoint selection was used.")


def main() -> int:
    supervisor = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if supervisor is not None:
            supervisor.log("PIPELINE STOPPED: " + message)
            supervisor._write_state(
                status="FAILED",
                phase=supervisor.state.get("phase", "unknown"),
                assigned_gpus=[],
                active_jobs={},
                message=message,
                failed_at=now(),
            )
        else:
            print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
