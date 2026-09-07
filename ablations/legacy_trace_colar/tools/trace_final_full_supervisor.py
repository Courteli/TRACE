#!/usr/bin/env python3
"""Durable four-GPU supervisor for the final TRACE training/evidence line."""

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
from typing import Dict, List, Sequence


ROOT = Path(
    os.environ.get("TRACE_ROOT", "/disk1/dingxukai/trace_colar")
).resolve()
PYTHON = Path(
    os.environ.get(
        "TRACE_PYTHON",
        "/home/dingxukai/miniconda3/envs/ROT/bin/python",
    )
)
PIPELINE_ID = os.environ.get(
    "PIPELINE_ID",
    f"{datetime.now():%Y%m%d-%H%M%S}_trace_final_seed0",
)
RUN_ROOT = Path(
    os.environ.get(
        "PIPELINE_ROOT",
        ROOT / "run_outputs" / "trace_final" / "full_pipeline" / PIPELINE_ID,
    )
).resolve()
GLOBAL_LOCK = (
    ROOT
    / "run_outputs"
    / "trace_final"
    / "full_pipeline"
    / ".trace_final_supervisor.lock"
)
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "0"))
STAGE0_CKPT = Path(
    os.environ.get(
        "STAGE0_CKPT",
        "/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/"
        "20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/"
        "checkpoints/epoch0__step6726__monitor0.871.ckpt",
    )
)
SOURCE_DATA = Path(
    "/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc"
)
RATIONALE_DATA = (
    ROOT
    / "run_outputs"
    / "trace_final"
    / "data"
    / "gsm8k_multirationale_v1"
)
SHARD_ROOT = RATIONALE_DATA / "shards"

GPU_MEMORY_LIMIT_MIB = int(
    os.environ.get("GPU_MEMORY_LIMIT_MIB", "1024")
)
GPU_UTIL_LIMIT_PERCENT = int(
    os.environ.get("GPU_UTIL_LIMIT_PERCENT", "5")
)
GPU_STABLE_POLLS = int(os.environ.get("GPU_STABLE_POLLS", "3"))
GPU_POLL_SECONDS = int(os.environ.get("GPU_POLL_SECONDS", "60"))
MIN_FREE_DISK_GIB = int(os.environ.get("MIN_FREE_DISK_GIB", "100"))

STAGE1_SCRIPT = ROOT / "run_trace_final_stage1_20260718.sh"
STAGE2_SCRIPT = ROOT / "run_trace_final_stage2_20260718.sh"
EVAL_SCRIPT = ROOT / "run_trace_final_eval_20260718.sh"
EVIDENCE_SCRIPT = ROOT / "run_trace_final_evidence_20260718.sh"
CAUSAL_SCRIPT = ROOT / "run_trace_final_causal_suite_20260718.sh"
DDP_SMOKE_SCRIPT = ROOT / "run_trace_final_ddp_smoke_20260718.sh"
DDP_RESUME_SMOKE_SCRIPT = (
    ROOT / "run_trace_final_ddp_resume_smoke_20260718.sh"
)
RATIONALE_TOOL = ROOT / "tools" / "trace_build_rationale_sets.py"

EXPECTED_DATA = {
    "train": 6726,
    "val": 747,
    "test": 1319,
}
OOD_DATA = {
    "gsmhard": (
        Path(
            "/home/dingxukai/RoT/data/GSM8k-Hard/"
            "readcot_qsa_test/test.json"
        ),
        1319,
    ),
    "svamp": (
        Path(
            "/home/dingxukai/RoT/data/SVAMP/"
            "readcot_qsa_test/test.json"
        ),
        1000,
    ),
    "multiarith": (
        Path(
            "/home/dingxukai/RoT/data/Multiarith/"
            "readcot_qsa_test/test.json"
        ),
        180,
    ),
}


class PipelineError(RuntimeError):
    pass


class ResourceContention(PipelineError):
    """A selected GPU was claimed by a process outside this supervisor."""


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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_values(path: Path) -> dict:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


class Supervisor:
    def __init__(self) -> None:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        (RUN_ROOT / "logs").mkdir(exist_ok=True)
        (RUN_ROOT / "markers").mkdir(exist_ok=True)
        self.state_path = RUN_ROOT / "state.json"
        self.events_path = RUN_ROOT / "events.log"
        self.stop_path = RUN_ROOT / "STOP"
        self.stop_requested = False
        self.children: Dict[str, subprocess.Popen] = {}
        self.completed: List[str] = []
        self.checkpoints: Dict[str, str] = {}

        if self.state_path.is_file():
            previous = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
            self.completed = list(
                previous.get("completed_phases", [])
            )
            self.checkpoints = dict(previous.get("checkpoints", {}))

        self.state = {
            "pipeline_id": PIPELINE_ID,
            "pid": os.getpid(),
            "status": "INITIALIZING",
            "phase": "startup",
            "created_at": now(),
            "updated_at": now(),
            "run_root": str(RUN_ROOT),
            "model_story": (
                "multi-rationale set-anchored trajectory formation -> "
                "CoT-free outcome-local refinement"
            ),
            "budgets": {
                "rationale_generation": (
                    "train/val/test, 6 strategy candidates, strict execution "
                    "verification, 4 GPU shards"
                ),
                "stage1": (
                    "6726 questions/epoch, max 10 epochs, complete 747-question "
                    "validation every epoch, patience 4"
                ),
                "stage2": (
                    "2048 questions/epoch, 8 paths/question, 10 epochs, "
                    "four-GPU DDP, complete validation every epoch"
                ),
                "matched_stage2_control": (
                    "same Stage 1, data, seed, GRPO, replay, budget, and "
                    "validation; outcome-local ranking removed"
                ),
                "task_eval": (
                    "GSM8K 1319, GSMHard 1319, SVAMP 1000, "
                    "MultiArith 180, test_times=1"
                ),
                "geometry": "200 questions x 8 complete paths",
                "causal": (
                    "200 paired questions; prefix/order/state/transition "
                    "interventions"
                ),
                "recovery_smoke": (
                    "four-rank Stage 2 full-state save at step 1 and "
                    "continuation to step 2"
                ),
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
                self.lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise PipelineError(
                f"Another final TRACE supervisor holds {GLOBAL_LOCK}"
            ) from exc
        self.lock_handle.seek(0)
        self.lock_handle.truncate()
        self.lock_handle.write(f"{os.getpid()} {PIPELINE_ID}\n")
        self.lock_handle.flush()

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, _frame) -> None:
        self.stop_requested = True
        self.log(f"Received signal {signum}; stopping child process groups.")
        for process in list(self.children.values()):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
        process_group = process.pid
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        # A launcher may return promptly after SIGTERM while one of its DDP
        # children remains alive. Always recheck the group after waiting for
        # the leader instead of assuming that leader exit emptied the group.
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass

    def terminate_children(self) -> None:
        for process in list(self.children.values()):
            self._terminate_process_group(process)
        self.children.clear()

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
        (RUN_ROOT / "markers" / f"{phase}.done").touch()
        self._write_state()

    def check_stop(self) -> None:
        if self.stop_requested or self.stop_path.exists():
            raise PipelineError("A stop request was received")

    def disk_free_gib(self) -> float:
        stat = os.statvfs("/disk1")
        return stat.f_bavail * stat.f_frsize / (1024**3)

    def ensure_disk(self) -> None:
        free = self.disk_free_gib()
        if free < MIN_FREE_DISK_GIB:
            raise PipelineError(
                f"/disk1 has {free:.1f} GiB free; floor is "
                f"{MIN_FREE_DISK_GIB} GiB. No files were deleted."
            )

    def gpu_snapshot(self) -> List[dict]:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,"
                "utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            cwd=ROOT,
        )
        rows = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 5:
                raise PipelineError(f"Unexpected nvidia-smi row: {line}")
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                    "utilization_percent": int(fields[4]),
                }
            )
        return rows

    def clean_gpu_ids(self, snapshot: Sequence[dict]) -> List[int]:
        return [
            row["index"]
            for row in snapshot
            if row["memory_total_mib"] >= 24000
            and row["memory_used_mib"] <= GPU_MEMORY_LIMIT_MIB
            and row["utilization_percent"] <= GPU_UTIL_LIMIT_PERCENT
        ]

    @staticmethod
    def _process_is_descendant(pid: int, roots: Sequence[int]) -> bool:
        roots = {int(root) for root in roots}
        current = int(pid)
        visited = set()
        while current > 1 and current not in visited:
            if current in roots:
                return True
            visited.add(current)
            try:
                status = Path(f"/proc/{current}/status").read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                return False
            parent = None
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    parent = int(line.split(":", 1)[1].strip())
                    break
            if parent is None or parent == current:
                return False
            current = parent
        return current in roots

    def gpu_compute_processes(self, gpu_ids: Sequence[int]) -> List[dict]:
        uuid_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            cwd=ROOT,
        )
        uuid_to_index = {}
        for line in uuid_rows.splitlines():
            index, uuid = [field.strip() for field in line.split(",", 1)]
            uuid_to_index[uuid] = int(index)
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            cwd=ROOT,
        )
        selected = {int(gpu) for gpu in gpu_ids}
        rows = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            gpu = uuid_to_index.get(fields[0])
            if gpu is None or gpu not in selected:
                continue
            try:
                pid = int(fields[1])
                memory = int(fields[2])
            except ValueError:
                continue
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                    b"\0",
                    b" ",
                ).decode("utf-8", errors="replace").strip()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                command = ""
            rows.append(
                {
                    "gpu": gpu,
                    "pid": pid,
                    "memory_used_mib": memory,
                    "command": command,
                }
            )
        return rows

    def foreign_gpu_processes(
        self,
        gpu_ids: Sequence[int],
        allowed_roots: Sequence[int],
    ) -> List[dict]:
        return [
            row
            for row in self.gpu_compute_processes(gpu_ids)
            if not self._process_is_descendant(
                row["pid"],
                allowed_roots,
            )
        ]

    def wait_for_four_gpus(self, phase: str) -> List[int]:
        candidate: List[int] = []
        stable = 0
        last_free = None
        poll = 0
        while True:
            self.check_stop()
            self.ensure_disk()
            snapshot = self.gpu_snapshot()
            free = self.clean_gpu_ids(snapshot)
            selected = free[:4]
            if len(selected) == 4 and selected == candidate:
                stable += 1
            elif len(selected) == 4:
                candidate = selected
                stable = 1
            else:
                candidate = []
                stable = 0
            if free != last_free or poll % 10 == 0:
                self.log(
                    f"{phase}: clean GPUs={free}; stable={stable}/"
                    f"{GPU_STABLE_POLLS}"
                )
                last_free = list(free)
            self._write_state(
                status="WAITING_FOR_FOUR_GPUS",
                phase=phase,
                assigned_gpus=[],
                gpu_snapshot=snapshot,
                clean_gpus=free,
                stable_polls=stable,
                message=(
                    "Waiting for four clean 24 GB GPUs; no scale "
                    "reduction or single-GPU Stage 2 fallback is allowed."
                ),
            )
            if stable >= GPU_STABLE_POLLS:
                final_free = self.clean_gpu_ids(self.gpu_snapshot())
                if all(gpu in final_free for gpu in candidate):
                    self.log(f"{phase}: acquired GPUs {candidate}")
                    return candidate
                candidate = []
                stable = 0
            poll += 1
            time.sleep(GPU_POLL_SECONDS)

    def source_files(self) -> List[Path]:
        files = [
            ROOT / "run.py",
            ROOT / "src" / "models" / "trace_final.py",
            ROOT / "src" / "models" / "trace_exchangeable.py",
            ROOT / "src" / "models" / "trace_bridge.py",
            ROOT / "src" / "datasets" / "trace_rationale_set.py",
            ROOT / "src" / "configs" / "models"
            / "trace_final_qwen3_instruct.yaml",
            ROOT / "src" / "configs" / "datasets" / "trace_qsa.yaml",
            RATIONALE_TOOL,
            ROOT / "tools" / "trace_exchangeable_geometry_summary.py",
            ROOT / "tools" / "trace_exchangeable_visualize.py",
            ROOT / "tools" / "trace_final_outcome_probe.py",
            ROOT / "tools" / "trace_final_causal_summary.py",
            ROOT / "tools" / "trace_final_task_summary.py",
            ROOT
            / "tools"
            / "trace_final_stage_comparison_visualize.py",
            ROOT / "TRACE_FINAL_SELF_AUDIT_20260718.md",
            ROOT / "TRACE_FINAL_FIGURE_CONTRACT_20260718.md",
            Path(__file__).resolve(),
            STAGE1_SCRIPT,
            STAGE2_SCRIPT,
            EVAL_SCRIPT,
            EVIDENCE_SCRIPT,
            CAUSAL_SCRIPT,
            DDP_SMOKE_SCRIPT,
            DDP_RESUME_SMOKE_SCRIPT,
            ROOT / "tests" / "test_trace_final.py",
            ROOT / "tests" / "test_trace_exchangeable.py",
        ]
        # TRACE inherits several latent-reasoning modules. Freeze the complete
        # in-repo source/config dependency surface, not just leaf classes.
        files.extend(
            path
            for path in (ROOT / "src").rglob("*")
            if path.is_file() and path.suffix in {".py", ".yaml"}
        )
        return sorted(set(path.resolve() for path in files))

    def build_frozen_manifest(self) -> dict:
        source = {}
        for path in self.source_files():
            if not path.is_file():
                raise PipelineError(f"Missing source input: {path}")
            source[str(path)] = sha256(path)
        datasets = {}
        for split, expected in EXPECTED_DATA.items():
            path = SOURCE_DATA / f"{split}.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or len(rows) != expected:
                raise PipelineError(
                    f"{path} must contain {expected} records"
                )
            datasets[split] = {
                "path": str(path),
                "count": expected,
                "sha256": sha256(path),
            }
        evaluation_datasets = {}
        for label, (path, expected) in OOD_DATA.items():
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or len(rows) != expected:
                raise PipelineError(
                    f"{path} must contain {expected} records"
                )
            evaluation_datasets[label] = {
                "path": str(path),
                "count": expected,
                "sha256": sha256(path),
            }
        if not STAGE0_CKPT.is_file():
            raise PipelineError(f"Missing Stage 0 checkpoint: {STAGE0_CKPT}")
        hparams = STAGE0_CKPT.parent.parent / "hparams.yaml"
        hparams_text = hparams.read_text(
            encoding="utf-8",
            errors="replace",
        )
        if "sft_method: cot" not in hparams_text:
            raise PipelineError(
                "Stage 0 is not an ordinary CoT SFT checkpoint"
            )
        return {
            "created_at": now(),
            "pipeline_id": PIPELINE_ID,
            "source": source,
            "source_datasets": datasets,
            "evaluation_datasets": evaluation_datasets,
            "stage0": {
                "path": str(STAGE0_CKPT),
                "sha256": sha256(STAGE0_CKPT),
                "kind": "ordinary_CoT_SFT_not_latent_baseline",
            },
            "hard_constraints": {
                "tiny_dataset": False,
                "epoch_scaling": 1,
                "stage1_questions_per_epoch": 6726,
                "stage1_max_epochs": 10,
                "stage1_validation_questions_every_epoch": 747,
                "stage1_patience": 4,
                "stage2_questions_per_epoch": 2048,
                "stage2_epochs": 10,
                "stage2_group_size": 8,
                "stage2_gpu_count": 4,
                "test_times": 1,
                "geometry_questions": 200,
                "checkpoint_retention": "best_plus_rolling_full_state_last",
            },
        }

    def verify_frozen(self) -> None:
        path = RUN_ROOT / "frozen_manifest.json"
        if not path.is_file():
            raise PipelineError("Missing frozen source manifest")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        changed = []
        for raw_path, expected_hash in manifest["source"].items():
            source_path = Path(raw_path)
            if (
                not source_path.is_file()
                or sha256(source_path) != expected_hash
            ):
                changed.append(raw_path)
        for item in manifest["source_datasets"].values():
            data_path = Path(item["path"])
            if (
                not data_path.is_file()
                or sha256(data_path) != item["sha256"]
            ):
                changed.append(str(data_path))
        for item in manifest["evaluation_datasets"].values():
            data_path = Path(item["path"])
            if (
                not data_path.is_file()
                or sha256(data_path) != item["sha256"]
            ):
                changed.append(str(data_path))
        if sha256(STAGE0_CKPT) != manifest["stage0"]["sha256"]:
            changed.append(str(STAGE0_CKPT))
        if changed:
            raise PipelineError(
                "Frozen inputs changed: " + ", ".join(changed[:10])
            )

    def run_command(
        self,
        *,
        name: str,
        command: Sequence[str],
        log_path: Path,
        env: dict | None = None,
        gpu_ids: Sequence[int] = (),
    ) -> None:
        self.check_stop()
        self.ensure_disk()
        if gpu_ids:
            clean = self.clean_gpu_ids(self.gpu_snapshot())
            unavailable = [
                int(gpu) for gpu in gpu_ids if int(gpu) not in clean
            ]
            if unavailable:
                raise ResourceContention(
                    f"{name}: GPUs became busy before launch: {unavailable}"
                )
        merged_env = os.environ.copy()
        if env:
            merged_env.update({key: str(value) for key, value in env.items()})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                list(command),
                cwd=ROOT,
                env=merged_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.children[name] = process
            self.log(f"Started {name}: pid={process.pid}")
            started = time.monotonic()
            last_state_update = -30
            try:
                while process.poll() is None:
                    self.check_stop()
                    elapsed = int(time.monotonic() - started)
                    if gpu_ids:
                        foreign = self.foreign_gpu_processes(
                            gpu_ids,
                            [process.pid],
                        )
                        if foreign:
                            details = ", ".join(
                                f"GPU{row['gpu']}:pid{row['pid']}"
                                for row in foreign
                            )
                            raise ResourceContention(
                                f"{name}: external GPU claim detected "
                                f"after launch ({details})"
                            )
                    if elapsed - last_state_update >= 30:
                        self._write_state(
                            status="RUNNING",
                            active_jobs={
                                name: {
                                    "pid": process.pid,
                                    "log": str(log_path),
                                    "elapsed_seconds": elapsed,
                                }
                            },
                        )
                        last_state_update = elapsed
                    time.sleep(5 if gpu_ids else 30)
            except BaseException:
                self._terminate_process_group(process)
                self.children.pop(name, None)
                self._write_state(active_jobs={})
                raise
            return_code = process.returncode
            self.children.pop(name, None)
        self._write_state(active_jobs={})
        if return_code != 0:
            self._terminate_process_group(process)
            if gpu_ids:
                foreign = self.foreign_gpu_processes(gpu_ids, [])
                if foreign:
                    details = ", ".join(
                        f"GPU{row['gpu']}:pid{row['pid']}"
                        for row in foreign
                    )
                    raise ResourceContention(
                        f"{name}: failed while selected GPUs were claimed "
                        f"externally ({details})"
                    )
            tail = "\n".join(
                log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()[-60:]
            )
            raise PipelineError(
                f"{name} exited with {return_code}\n{tail}"
            )
        self.log(f"Completed {name}")

    def run_parallel(
        self,
        jobs: Sequence[dict],
        *,
        phase: str,
        gpu_ids: Sequence[int] = (),
    ) -> None:
        self.check_stop()
        if gpu_ids:
            clean = self.clean_gpu_ids(self.gpu_snapshot())
            unavailable = [
                int(gpu) for gpu in gpu_ids if int(gpu) not in clean
            ]
            if unavailable:
                raise ResourceContention(
                    f"{phase}: GPUs became busy before launch: {unavailable}"
                )
        handles = {}
        for job in jobs:
            env = os.environ.copy()
            env.update(
                {
                    key: str(value)
                    for key, value in job.get("env", {}).items()
                }
            )
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                list(job["command"]),
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.children[job["name"]] = process
            handles[job["name"]] = (process, handle, log_path)
            self.log(f"Started {job['name']}: pid={process.pid}")
        started = time.monotonic()
        last_state_update = -30
        try:
            while any(
                process.poll() is None
                for process, _, _ in handles.values()
            ):
                self.check_stop()
                early_failures = [
                    name
                    for name, (process, _, _) in handles.items()
                    if process.poll() not in (None, 0)
                ]
                if early_failures:
                    for process, _, _ in handles.values():
                        self._terminate_process_group(process)
                    break
                if gpu_ids:
                    foreign = self.foreign_gpu_processes(
                        gpu_ids,
                        [
                            process.pid
                            for process, _, _ in handles.values()
                        ],
                    )
                    if foreign:
                        details = ", ".join(
                            f"GPU{row['gpu']}:pid{row['pid']}"
                            for row in foreign
                        )
                        raise ResourceContention(
                            f"{phase}: external GPU claim detected "
                            f"after launch ({details})"
                        )
                active = {
                    name: {
                        "pid": process.pid,
                        "log": str(log_path),
                        "elapsed_seconds": int(time.monotonic() - started),
                    }
                    for name, (process, _, log_path) in handles.items()
                    if process.poll() is None
                }
                elapsed = int(time.monotonic() - started)
                if elapsed - last_state_update >= 30:
                    self._write_state(
                        status="RUNNING",
                        phase=phase,
                        active_jobs=active,
                    )
                    last_state_update = elapsed
                time.sleep(5 if gpu_ids else 30)
        except BaseException:
            for process, _, _ in handles.values():
                self._terminate_process_group(process)
            for name, (_, handle, _) in handles.items():
                handle.close()
                self.children.pop(name, None)
            self._write_state(active_jobs={})
            raise
        failures = []
        for name, (process, handle, log_path) in handles.items():
            handle.close()
            self.children.pop(name, None)
            if process.returncode != 0:
                failures.append(
                    f"{name}={process.returncode} ({log_path})"
                )
        self._write_state(active_jobs={})
        if failures:
            if gpu_ids:
                foreign = self.foreign_gpu_processes(gpu_ids, [])
                if foreign:
                    details = ", ".join(
                        f"GPU{row['gpu']}:pid{row['pid']}"
                        for row in foreign
                    )
                    raise ResourceContention(
                        f"{phase}: failed while selected GPUs were claimed "
                        f"externally ({details})"
                    )
            raise PipelineError(
                f"{phase} failed: " + ", ".join(failures)
            )
        self.log(f"Completed parallel phase {phase}")

    @staticmethod
    def latest_resume_checkpoint(tag: str) -> Path | None:
        log_root = (
            ROOT
            / "logs"
            / "trace_final_qwen3_instruct"
            / "trace_qsa-gsm"
        )
        candidates = [
            path
            for path in log_root.glob(f"*_{tag}/checkpoints/last.ckpt")
            if path.is_file()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    def verify_rationale_data_manifest(self) -> None:
        manifest_path = RUN_ROOT / "rationale_data_manifest.json"
        audit_path = RATIONALE_DATA / "rationale_set_audit.json"
        if not manifest_path.is_file() or not audit_path.is_file():
            raise PipelineError("Rationale data manifest or audit is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise PipelineError("Frozen rationale audit is not PASS")
        if manifest.get("audit_sha256") != sha256(audit_path):
            raise PipelineError("Frozen rationale audit hash changed")
        for split, expected in EXPECTED_DATA.items():
            item = manifest.get("files", {}).get(split, {})
            path = Path(item.get("path", ""))
            if (
                not path.is_file()
                or path.resolve() != (RATIONALE_DATA / f"{split}.json").resolve()
                or sha256(path) != item.get("sha256")
            ):
                raise PipelineError(
                    f"Frozen rationale file changed or is missing: {split}"
                )
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or len(rows) != expected:
                raise PipelineError(
                    f"Frozen rationale split {split} must contain {expected} rows"
                )

    def preflight(self) -> None:
        phase = "preflight"
        if phase in self.completed:
            self.verify_frozen()
            return
        self.ensure_disk()
        commands = [
            [
                "bash",
                "-n",
                str(STAGE1_SCRIPT),
                str(STAGE2_SCRIPT),
                str(EVAL_SCRIPT),
                str(EVIDENCE_SCRIPT),
                str(CAUSAL_SCRIPT),
                str(DDP_SMOKE_SCRIPT),
                str(DDP_RESUME_SMOKE_SCRIPT),
            ],
            [
                str(PYTHON),
                "-m",
                "unittest",
                "-q",
                "tests.test_trace_final",
                "tests.test_trace_exchangeable",
            ],
            [
                str(PYTHON),
                "-m",
                "py_compile",
                str(ROOT / "src" / "models" / "trace_final.py"),
                str(RATIONALE_TOOL),
                str(
                    ROOT
                    / "tools"
                    / "trace_final_stage_comparison_visualize.py"
                ),
                str(Path(__file__).resolve()),
            ],
        ]
        log_path = RUN_ROOT / "logs" / "preflight.log"
        for command_index, command in enumerate(commands):
            self.run_command(
                name=f"preflight_{command_index}",
                command=command,
                log_path=log_path,
            )
        smoke_audit = (
            ROOT
            / "run_outputs"
            / "trace_final"
            / "smoke"
            / "20260718_trace_final_stage2_ddp_smoke4_allpaths"
            / "smoke_audit.json"
        )
        smoke = json.loads(smoke_audit.read_text(encoding="utf-8"))
        if smoke.get("status") != "PASS":
            raise PipelineError("Four-GPU Stage 2 smoke did not pass")
        cache_path = (
            ROOT
            / "run_outputs"
            / "trace_final"
            / "smoke"
            / "final_checkpoint"
            / "eval_one_mb4"
            / "logs"
            / "tb"
            / "run"
            / "trace_final_visual_test.pt"
        )
        schema_command = [
            str(PYTHON),
            "-c",
            (
                "import torch; "
                f"r=torch.load({str(cache_path)!r},map_location='cpu',"
                "weights_only=False)[0]; "
                "assert r['path_bottleneck']['question_answer_attention_access']==0; "
                "assert tuple(r['multiview_implicit_residuals'].shape[:2])==(8,8); "
                "assert 'rationale_teacher_residuals' in r; "
                "assert 'multiview_teacher_residuals' not in r"
            ),
        ]
        self.run_command(
            name="preflight_visual_schema",
            command=schema_command,
            log_path=log_path,
        )
        manifest = self.build_frozen_manifest()
        atomic_json(RUN_ROOT / "frozen_manifest.json", manifest)
        self.mark_completed(phase)
        self.log("Preflight passed and source/data inputs were frozen.")

    def generate_rationale_data(self) -> None:
        phase = "rationale_data"
        self.verify_frozen()
        audit_path = RATIONALE_DATA / "rationale_set_audit.json"
        if phase in self.completed:
            self.verify_rationale_data_manifest()
            return
        for split in ("train", "val", "test"):
            while True:
                gpus = self.wait_for_four_gpus(
                    f"rationale_generation_{split}"
                )
                jobs = []
                for shard_index, gpu in enumerate(gpus):
                    output = (
                        SHARD_ROOT
                        / split
                        / f"rationales_shard_{shard_index}.jsonl"
                    )
                    jobs.append(
                        {
                            "name": f"rationale_{split}_shard{shard_index}",
                            "command": [
                                str(PYTHON),
                                str(RATIONALE_TOOL),
                                "generate-shard",
                                "--source-dir",
                                str(SOURCE_DATA),
                                "--split",
                                split,
                                "--output",
                                str(output),
                                "--shard-id",
                                str(shard_index),
                                "--num-shards",
                                "4",
                                "--batch-size",
                                "2",
                                "--candidates",
                                "6",
                                "--max-alternatives",
                                "3",
                                "--max-prompt-tokens",
                                "512",
                                "--max-new-tokens",
                                "192",
                                "--temperature",
                                "0.85",
                                "--top-p",
                                "0.95",
                                "--seed",
                                "20260718",
                            ],
                            "env": {
                                "CUDA_VISIBLE_DEVICES": gpu,
                                "TOKENIZERS_PARALLELISM": "false",
                                "PYTORCH_CUDA_ALLOC_CONF": (
                                    "expandable_segments:True"
                                ),
                            },
                            "log": RUN_ROOT
                            / "logs"
                            / f"rationale_{split}_shard{shard_index}.log",
                        }
                    )
                try:
                    self.run_parallel(
                        jobs,
                        phase=f"rationale_generation_{split}",
                        gpu_ids=gpus,
                    )
                    break
                except ResourceContention as exc:
                    self.log(
                        f"{exc}; preserving completed source_ids and "
                        "returning to the four-GPU queue."
                    )
        self.run_command(
            name="rationale_merge_all",
            command=[
                str(PYTHON),
                str(RATIONALE_TOOL),
                "merge-all",
                "--source-dir",
                str(SOURCE_DATA),
                "--shard-root",
                str(SHARD_ROOT),
                "--output-dir",
                str(RATIONALE_DATA),
                "--num-shards",
                "4",
                "--max-alternatives",
                "3",
                "--min-train-multi-fraction",
                "0.20",
                "--min-eval-multi-fraction",
                "0.20",
            ],
            log_path=RUN_ROOT / "logs" / "rationale_merge.log",
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise PipelineError("Rationale data audit did not pass")
        for split, expected in EXPECTED_DATA.items():
            path = RATIONALE_DATA / f"{split}.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            if len(rows) != expected:
                raise PipelineError(
                    f"Final {split} rationale data has {len(rows)} records"
                )
        atomic_json(
            RUN_ROOT / "rationale_data_manifest.json",
            {
                "audit": audit,
                "audit_sha256": sha256(audit_path),
                "files": {
                    split: {
                        "path": str(RATIONALE_DATA / f"{split}.json"),
                        "sha256": sha256(
                            RATIONALE_DATA / f"{split}.json"
                        ),
                    }
                    for split in EXPECTED_DATA
                },
            },
        )
        self.verify_rationale_data_manifest()
        self.mark_completed(phase)
        self.log("Strict multi-rationale train/val/test data passed.")

    def gpu_resume_smoke(self) -> None:
        phase = "gpu_resume_smoke"
        self.verify_frozen()
        smoke_root = RUN_ROOT / "smoke" / "ddp_fullstate_resume4"
        audit_path = smoke_root / "smoke_audit.json"
        if phase in self.completed:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("status") != "PASS":
                raise PipelineError(
                    "Recorded four-GPU full-state resume smoke is not PASS"
                )
            return
        while True:
            gpus = self.wait_for_four_gpus("ddp_fullstate_resume_smoke")
            self._write_state(
                phase=phase,
                assigned_gpus=gpus,
                message=(
                    "Validating four-rank optimizer/scheduler/loop-state "
                    "checkpoint continuation before formal training."
                ),
            )
            try:
                self.run_command(
                    name="trace_final_ddp_fullstate_resume_smoke",
                    command=[
                        "bash",
                        str(DDP_RESUME_SMOKE_SCRIPT),
                        ",".join(map(str, gpus)),
                        str(STAGE0_CKPT),
                    ],
                    env={
                        "SMOKE_ID": (
                            f"{PIPELINE_ID}_ddp_fullstate_resume4"
                        ),
                        "OUT_ROOT": smoke_root,
                    },
                    log_path=RUN_ROOT
                    / "logs"
                    / "ddp_fullstate_resume_smoke.log",
                    gpu_ids=gpus,
                )
                break
            except ResourceContention as exc:
                self.log(
                    f"{exc}; the smoke created no formal checkpoint and "
                    "will return to the four-card queue."
                )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise PipelineError(
                "Four-GPU full-state save/resume smoke did not pass"
            )
        self.mark_completed(phase)
        self.log("Four-GPU full-state save/resume smoke passed.")

    def train_stage1(self) -> None:
        phase = "stage1"
        self.verify_frozen()
        self.verify_rationale_data_manifest()
        tag = f"{PIPELINE_ID}_stage1_seed{TRAIN_SEED}"
        manifest = RUN_ROOT / "training" / tag / "manifest.txt"
        existing = manifest_values(manifest)
        recorded = Path(existing.get("best_checkpoint", ""))
        if existing.get("finished_at") and recorded.is_file():
            self.checkpoints["stage1_best"] = str(recorded)
            self.mark_completed(phase)
            self.log(f"Recovered completed Stage 1 manifest: {recorded}")
            return
        if phase in self.completed:
            values = manifest_values(manifest)
            checkpoint = Path(values["best_checkpoint"])
            if not checkpoint.is_file():
                raise PipelineError("Recorded Stage 1 best checkpoint is missing")
            self.checkpoints["stage1_best"] = str(checkpoint)
            return
        while True:
            gpus = self.wait_for_four_gpus("stage1_admission")
            self._write_state(
                phase=phase,
                assigned_gpus=[gpus[0]],
                message=(
                    "Four-card admission passed; Stage 1 uses one card at "
                    "batch size 1 to preserve the established optimization "
                    "protocol."
                ),
            )
            resume = self.latest_resume_checkpoint(tag)
            if resume is not None:
                self.log(f"Stage 1 will resume full state from {resume}")
            try:
                self.run_command(
                    name="trace_final_stage1",
                    command=[
                        "bash",
                        str(STAGE1_SCRIPT),
                        str(gpus[0]),
                    ],
                    env={
                        "RUN_TAG": tag,
                        "RUN_ROOT": RUN_ROOT / "training",
                        "DATASET_DIR": RATIONALE_DATA,
                        "STAGE0_CKPT": STAGE0_CKPT,
                        "TRAIN_SEED": TRAIN_SEED,
                        "RESUME_CKPT_PATH": (
                            str(resume) if resume else ""
                        ),
                    },
                    log_path=RUN_ROOT / "logs" / "stage1_wrapper.log",
                    gpu_ids=[gpus[0]],
                )
                break
            except ResourceContention as exc:
                self.log(
                    f"{exc}; Stage 1 state is preserved and will resume "
                    "after the next four-card admission."
                )
        values = manifest_values(manifest)
        checkpoint = Path(values.get("best_checkpoint", ""))
        if not checkpoint.is_file():
            raise PipelineError("Stage 1 did not produce a validation best")
        self.checkpoints["stage1_best"] = str(checkpoint)
        self.mark_completed(phase)
        self.log(f"Stage 1 best: {checkpoint}")

    def _train_stage2_variant(
        self,
        *,
        phase: str,
        tag_suffix: str,
        checkpoint_key: str,
        stage2_variant: str,
        local_ranking_weight: float,
    ) -> None:
        self.verify_frozen()
        self.verify_rationale_data_manifest()
        tag = f"{PIPELINE_ID}_{tag_suffix}_seed{TRAIN_SEED}"
        manifest = RUN_ROOT / "training" / tag / "manifest.txt"
        existing = manifest_values(manifest)
        recorded = Path(existing.get("best_checkpoint", ""))
        if existing.get("finished_at") and recorded.is_file():
            self.checkpoints[checkpoint_key] = str(recorded)
            self.mark_completed(phase)
            self.log(
                f"Recovered completed {stage2_variant} manifest: {recorded}"
            )
            return
        if phase in self.completed:
            values = manifest_values(manifest)
            checkpoint = Path(values["best_checkpoint"])
            if not checkpoint.is_file():
                raise PipelineError(
                    f"Recorded {stage2_variant} best checkpoint is missing"
                )
            self.checkpoints[checkpoint_key] = str(checkpoint)
            return
        stage1 = Path(self.checkpoints["stage1_best"])
        while True:
            gpus = self.wait_for_four_gpus(f"{phase}_training")
            gpu_csv = ",".join(map(str, gpus))
            self._write_state(
                phase=phase,
                assigned_gpus=gpus,
                message=(
                    f"Running complete ten-epoch four-GPU {stage2_variant}."
                ),
            )
            resume = self.latest_resume_checkpoint(tag)
            if resume is not None:
                self.log(
                    f"{stage2_variant} will resume full state from {resume}"
                )
            try:
                self.run_command(
                    name=f"trace_final_{phase}",
                    command=[
                        "bash",
                        str(STAGE2_SCRIPT),
                        gpu_csv,
                        str(stage1),
                    ],
                    env={
                        "RUN_TAG": tag,
                        "RUN_ROOT": RUN_ROOT / "training",
                        "DATASET_DIR": RATIONALE_DATA,
                        "TRAIN_SEED": TRAIN_SEED,
                        "STAGE2_VARIANT": stage2_variant,
                        "LOCAL_RANKING_WEIGHT": local_ranking_weight,
                        "SFT_REPLAY_WEIGHT": 0.05,
                        "ACCURACY_GRADIENT_GUARD": (
                            "true"
                            if local_ranking_weight > 0
                            else "false"
                        ),
                        "RANKING_GRAD_RATIO": 0.25,
                        "RESUME_CKPT_PATH": (
                            str(resume) if resume else ""
                        ),
                    },
                    log_path=RUN_ROOT
                    / "logs"
                    / f"{phase}_wrapper.log",
                    gpu_ids=gpus,
                )
                break
            except ResourceContention as exc:
                self.log(
                    f"{exc}; {stage2_variant} state is preserved and will "
                    "resume after the next four-card admission."
                )
        values = manifest_values(manifest)
        checkpoint = Path(values.get("best_checkpoint", ""))
        if not checkpoint.is_file():
            raise PipelineError(
                f"{stage2_variant} did not produce a validation best"
            )
        self.checkpoints[checkpoint_key] = str(checkpoint)
        self.mark_completed(phase)
        self.log(f"{stage2_variant} best: {checkpoint}")

    def train_stage2(self) -> None:
        self._train_stage2_variant(
            phase="stage2",
            tag_suffix="stage2_final",
            checkpoint_key="stage2_best",
            stage2_variant="outcome_local_refinement",
            local_ranking_weight=0.10,
        )

    def train_stage2_answeronly(self) -> None:
        self._train_stage2_variant(
            phase="stage2_answeronly",
            tag_suffix="stage2_answeronly",
            checkpoint_key="stage2_answeronly_best",
            stage2_variant="matched_answer_only_replay",
            local_ranking_weight=0.0,
        )

    def run_evidence(self) -> None:
        phase = "evidence"
        self.verify_frozen()
        self.verify_rationale_data_manifest()
        evidence_root = RUN_ROOT / "evidence"
        if phase in self.completed:
            if not (evidence_root / "evidence_done.txt").is_file():
                raise PipelineError("Evidence completion marker is missing")
            return
        while True:
            gpus = self.wait_for_four_gpus(
                "full_task_geometry_evidence"
            )
            self._write_state(
                phase=phase,
                assigned_gpus=gpus,
                message=(
                    "Evaluating Stage 1 and Stage 2 on full GSM8K/OOD, "
                    "then 200-question eight-path geometry and publication "
                    "visuals."
                ),
            )
            try:
                self.run_command(
                    name="trace_final_evidence",
                    command=[
                        "bash",
                        str(EVIDENCE_SCRIPT),
                        ",".join(map(str, gpus)),
                        self.checkpoints["stage1_best"],
                        self.checkpoints["stage2_answeronly_best"],
                        self.checkpoints["stage2_best"],
                        str(evidence_root),
                    ],
                    env={"GSM8K_DIR": RATIONALE_DATA},
                    log_path=RUN_ROOT / "logs" / "evidence_wrapper.log",
                    gpu_ids=gpus,
                )
                break
            except ResourceContention as exc:
                self.log(
                    f"{exc}; completed dataset evaluations are retained "
                    "and the evidence wave will resume."
                )
        self.mark_completed(phase)

    def run_causal(self) -> None:
        phase = "causal"
        self.verify_frozen()
        self.verify_rationale_data_manifest()
        causal_root = RUN_ROOT / "causal"
        if phase in self.completed:
            if not (
                causal_root
                / "summary"
                / "trace_final_causal_summary.json"
            ).is_file():
                raise PipelineError("Causal summary is missing")
            return
        while True:
            gpus = self.wait_for_four_gpus("causal_path_audit")
            self._write_state(
                phase=phase,
                assigned_gpus=gpus,
                message=(
                    "Running paired full/no-path/prefix/order/state/"
                    "transition interventions on 200 GSM8K questions."
                ),
            )
            try:
                self.run_command(
                    name="trace_final_causal",
                    command=[
                        "bash",
                        str(CAUSAL_SCRIPT),
                        ",".join(map(str, gpus)),
                        self.checkpoints["stage2_best"],
                        str(causal_root),
                    ],
                    env={"GSM8K_DIR": RATIONALE_DATA},
                    log_path=RUN_ROOT / "logs" / "causal_wrapper.log",
                    gpu_ids=gpus,
                )
                break
            except ResourceContention as exc:
                self.log(
                    f"{exc}; completed causal conditions are retained and "
                    "the suite will resume."
                )
        self.mark_completed(phase)

    def final_self_audit(self) -> None:
        phase = "self_audit"
        self.verify_frozen()
        self.verify_rationale_data_manifest()
        if phase in self.completed:
            return
        task = json.loads(
            (
                RUN_ROOT
                / "evidence"
                / "task_summary"
                / "trace_final_task_summary.json"
            ).read_text(encoding="utf-8")
        )
        causal = json.loads(
            (
                RUN_ROOT
                / "causal"
                / "summary"
                / "trace_final_causal_summary.json"
            ).read_text(encoding="utf-8")
        )
        probe = json.loads(
            (
                RUN_ROOT
                / "evidence"
                / "outcome_probe"
                / "trace_final_outcome_probe.json"
            ).read_text(encoding="utf-8")
        )
        geometry = json.loads(
            (
                RUN_ROOT
                / "evidence"
                / "complete_path_geometry"
                / "trace_exchangeable_geometry_summary.json"
            ).read_text(encoding="utf-8")
        )

        def ci_above_zero(row: dict) -> bool:
            return (
                int(row.get("n", 0)) > 0
                and row.get("ci95_low") is not None
                and float(row["ci95_low"]) > 0.0
            )

        def ci_below_zero(row: dict) -> bool:
            return (
                int(row.get("n", 0)) > 0
                and row.get("ci95_high") is not None
                and float(row["ci95_high"]) < 0.0
            )

        comparisons = geometry["paired_comparisons"]
        stage1_to_final = comparisons[
            "Stage1_GSM8K_to_Final_GSM8K"
        ]["metrics"]
        answeronly_to_final = comparisons[
            "AnswerOnly_GSM8K_to_Final_GSM8K"
        ]["metrics"]
        stage1_refinement_gate = {
            "pass": ci_above_zero(stage1_to_final["outcome_margin"]),
            "criterion": (
                "On the same 200 questions, the Stage 1 to Full TRACE "
                "outcome-margin delta has a strictly positive question-"
                "bootstrap 95% lower bound"
            ),
            "evidence": stage1_to_final["outcome_margin"],
        }
        matched_geometry_checks = {
            "correct_neighborhood_tightens": ci_below_zero(
                answeronly_to_final["correct_local_radius"]
            ),
            "nearest_wrong_path_is_rejected": ci_above_zero(
                answeronly_to_final[
                    "wrong_to_local_correct_distance"
                ]
            ),
            "outcome_margin_rises": ci_above_zero(
                answeronly_to_final["outcome_margin"]
            ),
            "label_null_excess_rises": ci_above_zero(
                answeronly_to_final[
                    "outcome_margin_excess_over_null"
                ]
            ),
        }
        matched_geometry_gate = {
            "pass": all(matched_geometry_checks.values()),
            "criterion": (
                "Relative to the seed-, data-, budget-, and Stage-1-matched "
                "answer-only control, correct local radius decreases while "
                "wrong distance, outcome margin, and label-null excess all "
                "increase with paired two-sided 95% confidence intervals "
                "excluding zero"
            ),
            "checks": matched_geometry_checks,
            "evidence": {
                key: answeronly_to_final[key]
                for key in (
                    "correct_local_radius",
                    "wrong_to_local_correct_distance",
                    "outcome_margin",
                    "outcome_margin_excess_over_null",
                )
            },
        }

        final_geometry = geometry["methods"]["Final_GSM8K"]["aggregate"]
        formation_checks = {
            "ordered_progress_above_null": ci_above_zero(
                final_geometry["position_order_excess_over_null"]
            ),
            "step_correspondence_above_null": ci_above_zero(
                final_geometry["step_alignment_excess_over_null"]
            ),
            "complete_path_alignment_positive": ci_above_zero(
                final_geometry["final_path_alignment_cos"]
            ),
            "path_diversity_nonzero": ci_above_zero(
                final_geometry["model_path_diversity"]
            ),
        }
        formation_gate = {
            "pass": all(formation_checks.values()),
            "criterion": (
                "The final model retains ordered, non-collapsed complete "
                "trajectories whose alignment exceeds matched permutation "
                "nulls on 200 held-out questions"
            ),
            "checks": formation_checks,
        }

        final_probe = probe["crossfit"]["Final_GSM8K"]
        final_probe_auroc = final_probe["metrics"]["auroc"]
        final_probe_null = final_probe[
            "within_question_label_permutation_null"
        ]
        heldout_probe_gate = {
            "pass": (
                final_probe_auroc.get("ci95_low") is not None
                and float(final_probe_auroc["ci95_low"]) > 0.5
                and final_probe_null.get("p_one_sided") is not None
                and float(final_probe_null["p_one_sided"]) < 0.01
            ),
            "criterion": (
                "Question-held-out complete-path AUROC has a 95% lower "
                "bound above 0.5 and within-question label-permutation "
                "p < 0.01"
            ),
            "auroc": final_probe_auroc,
            "permutation_null": final_probe_null,
        }
        ood_probe_checks = {}
        for label, report in probe["transfer"].items():
            auroc = report["metrics"]["auroc"]
            ood_probe_checks[label] = {
                "pass": (
                    auroc.get("ci95_low") is not None
                    and float(auroc["ci95_low"]) > 0.5
                ),
                "auroc": auroc,
            }
        ood_probe_gate = {
            "pass": bool(ood_probe_checks)
            and all(
                item["pass"] for item in ood_probe_checks.values()
            ),
            "criterion": (
                "A PCA/probe fit only on GSM8K separates correct and wrong "
                "complete paths on every evaluated OOD dataset without "
                "target refitting, with each AUROC 95% lower bound above 0.5"
            ),
            "datasets": ood_probe_checks,
        }

        causal_gates = causal["claim_gates"]
        causal_core_labels = (
            "Path is a necessary answer mediator",
            "Ordered path structure is used",
            "Path content is question-specific",
            "Path effect is not a norm artifact",
        )
        causal_core_checks = {
            label: bool(causal_gates[label]["pass"])
            for label in causal_core_labels
        }
        causal_core_gate = {
            "pass": all(causal_core_checks.values()),
            "criterion": (
                "No-path, order-preserving controls, same/cross-question "
                "swaps, and same-norm random paths jointly establish that "
                "the bottlenecked path is used and carries question-specific "
                "ordered content"
            ),
            "checks": causal_core_checks,
        }

        primary_task_pass = bool(task["primary_gate"]["pass"])
        matched_task_pass = bool(
            task["matched_control_task_gate"]["pass"]
        )
        core_results = {
            "accuracy_length_primary": primary_task_pass,
            "matched_answeronly_task": matched_task_pass,
            "stage1_trajectory_formation_retained": formation_gate["pass"],
            "stage1_to_final_outcome_refinement": (
                stage1_refinement_gate["pass"]
            ),
            "matched_answeronly_geometry": matched_geometry_gate["pass"],
            "question_heldout_outcome_separability": (
                heldout_probe_gate["pass"]
            ),
            "core_path_causality": causal_core_gate["pass"],
        }
        in_domain_confirmed = all(core_results.values())
        if in_domain_confirmed and ood_probe_gate["pass"]:
            empirical_status = "CLAIM_READY"
        elif in_domain_confirmed:
            empirical_status = "IN_DOMAIN_MECHANISM_CONFIRMED"
        else:
            empirical_status = "RESULTS_REQUIRE_REVISION"
        audit = {
            "design_status": "PASS",
            "design_gates": {
                "fixed_eight_state_path": True,
                "answer_question_attention_exactly_zero": True,
                "verified_multi_rationale_set_formation": True,
                "permutation_invariant_teacher_matching": True,
                "no_route_id_or_prototype": True,
                "same_complete_path_metric_in_both_stages": True,
                "cot_free_stage2_outcome_refinement": True,
                "question_local_correct_wrong_ranking": True,
                "stage1_replay_and_gradient_guard": True,
                "matched_answeronly_single_difference_control": True,
                "deterministic_center_inference": True,
                "fixed_inference_latent_length": 8,
            },
            "empirical_status": empirical_status,
            "core_result_checks": core_results,
            "empirical_gates": {
                "accuracy_length_primary": task["primary_gate"],
                "matched_answeronly_task": task[
                    "matched_control_task_gate"
                ],
                "trajectory_formation_retained": formation_gate,
                "stage1_to_final_outcome_refinement": (
                    stage1_refinement_gate
                ),
                "matched_answeronly_geometry": matched_geometry_gate,
                "question_heldout_outcome_probe": heldout_probe_gate,
                "frozen_gsm8k_to_ood_probe": ood_probe_gate,
                "core_path_causality": causal_core_gate,
                "secondary_causal_claims": causal_gates,
            },
            "claim_policy": (
                "The final paper may state only empirical claims whose "
                "registered gate passed. 3D plots never override a failed "
                "causal or held-out statistical gate. Per-state necessity is "
                "reported only if its separate secondary gate passes."
            ),
        }
        atomic_json(RUN_ROOT / "FINAL_SELF_AUDIT.json", audit)
        self.mark_completed(phase)

    def run(self) -> None:
        try:
            self.preflight()
            self.verify_frozen()
            self.gpu_resume_smoke()
            self.generate_rationale_data()
            self.train_stage1()
            self.train_stage2()
            self.train_stage2_answeronly()
            self.run_evidence()
            self.run_causal()
            self.final_self_audit()
            self._write_state(
                status="COMPLETE",
                phase="complete",
                assigned_gpus=[],
                active_jobs={},
                message=(
                    "The complete final TRACE training and registered evidence "
                    "pipeline finished. See FINAL_SELF_AUDIT.json."
                ),
            )
            self.log("Final TRACE pipeline completed.")
        except Exception as exc:
            self.terminate_children()
            self._write_state(
                status="STOPPED" if self.stop_requested else "FAILED",
                assigned_gpus=[],
                active_jobs={},
                message=str(exc),
            )
            self.log(f"Pipeline terminated: {exc}")
            raise


def main() -> None:
    supervisor = Supervisor()
    supervisor.run()


if __name__ == "__main__":
    main()
