from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import monitor_train_only_v8 as monitor


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "trace_vb_v8_monitor"
    / "waiter_multiple_stanzas.log"
)
STUDENT_GATE_FIXTURE = FIXTURE.with_name("student_initial_gate.json")
CAPABILITY_GATE_FIXTURE = FIXTURE.with_name("capability_parity_gate.json")
STUDENT_ORIGIN = (
    "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
    "20260818-213000_trace_vb_v7_full_seed0/student_initial_gate.json"
)
CAPABILITY_ORIGIN = (
    "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
    "20260818-213000_trace_vb_v7_full_seed0/capability_parity_gate.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeRunner:
    def __init__(self, session: str, *, tmux_alive: bool = True):
        self.session = session
        self.tmux_alive = tmux_alive
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, arguments):
        command = tuple(str(value) for value in arguments)
        self.calls.append(command)
        if command[0] == "tmux":
            if self.tmux_alive:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        f"{self.session}|4242|0||bash|"
                        "bash scripts/wait_for_four_gpus_and_run_train_only_v8.sh\n"
                    ),
                    stderr="",
                )
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=f"can't find session: {self.session}\n",
            )
        if command[0] == "nvidia-smi" and "--query-gpu=" in command[1]:
            rows = []
            for index in range(8):
                free = 21600 if index in (4, 5, 7) else 13000
                utilization = 0 if index != 6 else 82
                rows.append(
                    f"{index}, GPU-{index}, 24576, {24576 - free}, "
                    f"{free}, {utilization}"
                )
            return subprocess.CompletedProcess(
                command, 0, stdout="\n".join(rows) + "\n", stderr=""
            )
        if command[0] == "nvidia-smi" and "--query-compute-apps=" in command[1]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected external command: {command}")


class MonitorFixture:
    def __init__(self, root: Path):
        self.root = root
        self.artifact = root / "artifacts"
        self.tag = "fixture_v8_seed0"
        self.session = "fixture_v8_waiter"
        self.state = root / "state"
        self.supervisor = self.artifact / "supervisor"
        self.supervisor.mkdir(parents=True)
        shutil.copyfile(FIXTURE, self.supervisor / f"{self.tag}_wait.log")
        self.config = monitor.MonitorConfig(
            tag=self.tag,
            artifact_root=self.artifact,
            physical_gpus=(4, 5, 6, 7),
            waiter_session=self.session,
            state_dir=self.state,
            code_root=Path(__file__).resolve().parents[1],
        )

    @property
    def pipeline(self) -> Path:
        return self.config.pipeline_dir

    @property
    def stage1(self) -> Path:
        return self.config.stage1_dir

    @property
    def stage2(self) -> Path:
        return self.config.stage2_dir

    def start_pipeline(self) -> None:
        self.pipeline.mkdir(parents=True, exist_ok=True)
        metric_safe_artifact = self.pipeline / "metric_safe_baseline.json"
        capability_artifact = (
            self.pipeline / "registered_capability_validation.json"
        )
        shutil.copyfile(STUDENT_GATE_FIXTURE, metric_safe_artifact)
        shutil.copyfile(CAPABILITY_GATE_FIXTURE, capability_artifact)
        (self.pipeline / "manifest.txt").write_text(
            "\n".join(
                (
                    "model=TRACE-VB-v8",
                    "pipeline=train_and_validation_only",
                    f"pipeline_tag={self.tag}",
                    "physical_gpus=4,5,6,7",
                    "formal_gpus=4,5,6,7",
                    "fixed_gpus=4,5,6,7",
                    "metric_safe_baseline_source=student_commit",
                    f"metric_safe_baseline_origin={STUDENT_ORIGIN}",
                    f"metric_safe_baseline_artifact={metric_safe_artifact.resolve()}",
                    f"metric_safe_baseline_sha256={monitor.METRIC_SAFE_BASELINE_SHA256}",
                    "metric_safe_baseline_correct=527",
                    "metric_safe_baseline_questions=747",
                    "registered_capability_validation_source=capability_teacher_all_roles",
                    f"registered_capability_validation_origin={CAPABILITY_ORIGIN}",
                    "registered_capability_validation_artifact="
                    f"{capability_artifact.resolve()}",
                    "registered_capability_validation_sha256="
                    f"{monitor.REGISTERED_CAPABILITY_VALIDATION_SHA256}",
                    "registered_capability_validation_correct=540",
                    "registered_capability_validation_questions=747",
                    "started_at=2026-08-20T21:00:00+08:00",
                )
            )
            + "\n",
            encoding="utf-8",
        )

    def start_stage(self, phase: str) -> Path:
        stage = self.stage1 if phase == "stage1" else self.stage2
        (stage / "candidates").mkdir(parents=True, exist_ok=True)
        if phase == "stage1":
            metric_safe_artifact = stage / "metric_safe_baseline.json"
            capability_artifact = stage / "registered_capability_validation.json"
            shutil.copyfile(STUDENT_GATE_FIXTURE, metric_safe_artifact)
            shutil.copyfile(CAPABILITY_GATE_FIXTURE, capability_artifact)
        else:
            metric_safe_artifact = self.stage1 / "metric_safe_baseline.json"
            capability_artifact = self.stage1 / "registered_capability_validation.json"
        (stage / "manifest.txt").write_text(
            "\n".join(
                (
                    "model=TRACE-VB-v8",
                    f"phase={phase}",
                    "test_split=false",
                    "evidence_pipeline=false",
                    "physical_gpus=4,5,6,7",
                    "formal_gpus=4,5,6,7",
                    "fixed_gpus=4,5,6,7",
                    "metric_safe_baseline_source=student_commit",
                    f"metric_safe_baseline_origin={STUDENT_ORIGIN}",
                    f"metric_safe_baseline_artifact={metric_safe_artifact.resolve()}",
                    f"metric_safe_baseline_sha256={monitor.METRIC_SAFE_BASELINE_SHA256}",
                    "metric_safe_baseline_correct=527",
                    "metric_safe_baseline_questions=747",
                    "registered_capability_validation_source=capability_teacher_all_roles",
                    f"registered_capability_validation_origin={CAPABILITY_ORIGIN}",
                    f"registered_capability_validation_artifact={capability_artifact.resolve()}",
                    "registered_capability_validation_sha256="
                    f"{monitor.REGISTERED_CAPABILITY_VALIDATION_SHA256}",
                    "registered_capability_validation_correct=540",
                    "registered_capability_validation_questions=747",
                    "started_at=2026-08-20T21:00:00+08:00",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return stage

    def candidate(
        self,
        phase: str,
        coordinate: int,
        correct_count: int,
    ) -> dict:
        stage = self.stage1 if phase == "stage1" else self.stage2
        candidate_dir = stage / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        selected_stage1 = None
        stage1_index_path = self.stage1 / "candidate_index.json"
        if phase == "stage2" and coordinate == 0 and stage1_index_path.is_file():
            stage1_index = json.loads(stage1_index_path.read_text(encoding="utf-8"))
            selected_stage1 = next(
                item
                for item in stage1_index["candidates"]
                if item["checkpoint"] == stage1_index["selected_checkpoint"]
            )
        source_step = (
            int(selected_stage1["source_global_step"])
            if selected_stage1 is not None
            else coordinate
            if phase == "stage1"
            else 0
            if coordinate == 0
            else coordinate * 2
        )
        step = coordinate if phase == "stage1" else (0 if coordinate == 0 else source_step)
        summary = candidate_dir / monitor.summary_filename(phase, coordinate)
        behavior = {
            "valid_answer_fraction": 1.0,
            "unique_prediction_ratio": 0.8,
            "top1_mode_fraction": 0.01,
            "nonempty_output_fraction": 1.0,
        }
        summary.write_text(
            json.dumps(
                {
                    "schema_version": monitor.VALIDATION_SCHEMA,
                    "validation_path": "student_commit",
                    "epoch_index": 0,
                    "global_step": source_step,
                    "world_size": 4,
                    "unique_questions": 747,
                    "correct_count": correct_count,
                    "accuracy": correct_count / 747,
                    **behavior,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if selected_stage1 is not None:
            checkpoint = Path(selected_stage1["checkpoint"])
        else:
            checkpoint_dir = stage / "fixture_checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint = checkpoint_dir / f"{phase}_{coordinate:04d}.ckpt"
            checkpoint.write_bytes(f"{phase}:{coordinate}:{correct_count}".encode())
        record = {
            "schema_version": monitor.CANDIDATE_SCHEMA,
            "phase": phase,
            "physical_gpus": "4,5,6,7",
            "step": step,
            "selection_step": step,
            "source_global_step": source_step,
            "phase_input": phase == "stage2" and coordinate == 0,
            "rollout_batches": coordinate if phase == "stage2" else None,
            "correct_count": correct_count,
            "accuracy": correct_count / 747,
            "summary": str(summary.resolve()),
            "summary_sha256": sha256(summary),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "behavior": behavior,
        }
        record_path = candidate_dir / monitor.candidate_filename(phase, coordinate)
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record

    def finish_stage(self, phase: str, counts: list[int]) -> dict:
        coordinates = monitor.EXPECTED_COORDINATES[phase]
        records = [
            self.candidate(phase, coordinate, correct)
            for coordinate, correct in zip(coordinates, counts)
        ]
        selected = max(
            zip(coordinates, records),
            key=lambda item: (int(item[1]["correct_count"]), -int(item[0])),
        )
        selected_coordinate, selected_record = selected
        index = {
            "schema_version": f"trace_vb_v8_{phase}_candidate_index_v1",
            "phase": phase,
            "physical_gpus": "4,5,6,7",
            "selection": "maximum_exact_correct_count_then_earliest_progress_coordinate",
            "expected_steps": list(coordinates) if phase == "stage1" else None,
            "observed_selection_steps": [int(item["selection_step"]) for item in records],
            "expected_rollout_batches": list(coordinates) if phase == "stage2" else None,
            "candidate_count": 5,
            "eligible_candidate_count": 5,
            "behavior_thresholds": monitor.BEHAVIOR_THRESHOLDS,
            "selected_step": int(selected_record["step"]),
            "selected_source_global_step": int(selected_record["source_global_step"]),
            "selected_rollout_batches": (
                selected_coordinate if phase == "stage2" else None
            ),
            "selected_phase_input": bool(selected_record["phase_input"]),
            "selected_correct_count": int(selected_record["correct_count"]),
            "selected_accuracy": float(selected_record["accuracy"]),
            "selected_checkpoint": selected_record["checkpoint"],
            "candidates": records,
        }
        stage = self.stage1 if phase == "stage1" else self.stage2
        (stage / "candidate_index.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "best_checkpoint.txt").write_text(
            selected_record["checkpoint"] + "\n", encoding="utf-8"
        )
        (stage / "last_checkpoint.txt").write_text(
            records[-1]["checkpoint"] + "\n", encoding="utf-8"
        )
        with (stage / "manifest.txt").open("a", encoding="utf-8") as stream:
            stream.write("finished_at=2026-08-21T00:00:00+08:00\n")
        return index

    def finish_pipeline(self, stage1_index: dict, stage2_index: dict) -> None:
        shutil.copyfile(
            self.stage1 / "candidate_index.json",
            self.pipeline / "stage1_candidate_index.json",
        )
        shutil.copyfile(
            self.stage2 / "candidate_index.json",
            self.pipeline / "stage2_candidate_index.json",
        )
        stage1_checkpoint = Path(stage1_index["selected_checkpoint"])
        final_checkpoint = Path(stage2_index["selected_checkpoint"])
        (self.pipeline / "stage1_best.txt").write_text(
            str(stage1_checkpoint) + "\n", encoding="utf-8"
        )
        (self.pipeline / "final_best.txt").write_text(
            str(final_checkpoint) + "\n", encoding="utf-8"
        )
        with (self.pipeline / "manifest.txt").open("a", encoding="utf-8") as stream:
            stream.write(
                f"stage1_checkpoint={stage1_checkpoint}\n"
                f"stage1_checkpoint_sha256={sha256(stage1_checkpoint)}\n"
                f"final_checkpoint={final_checkpoint}\n"
                f"final_checkpoint_sha256={sha256(final_checkpoint)}\n"
                "finished_at=2026-08-21T00:00:01+08:00\n"
            )
        (self.supervisor / f"{self.tag}.exit_status").write_text(
            "0\n", encoding="ascii"
        )

    def resume_binding(self, phase: str, attempt_id: str) -> tuple[Path, dict]:
        stage = self.stage1 if phase == "stage1" else self.stage2
        records = sorted(
            (stage / "candidates").glob(
                "candidate_step_*.json"
                if phase == "stage1"
                else "candidate_rollout_*.json"
            )
        )
        record_path = records[-1]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        binding_path = self.pipeline / (
            f"resume_binding_{phase}_{attempt_id}.json"
        )
        binding_path.write_text(
            json.dumps(
                {
                    "schema_version": "trace_vb_v8_pipeline_resume_binding_v1",
                    "status": "PASS",
                    "pipeline_tag": self.tag,
                    "phase": phase,
                    "physical_gpus": "4,5,6,7",
                    "formal_gpus": "4,5,6,7",
                    "fixed_gpus": "4,5,6,7",
                    "latest_candidate_record": str(record_path.resolve()),
                    "validated_boundary_checkpoint": record["checkpoint"],
                    "validated_boundary_checkpoint_sha256": record[
                        "checkpoint_sha256"
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return binding_path, record

    def stage_resume_contract(
        self, phase: str, attempt_id: str, record: dict
    ) -> Path:
        stage = self.stage1 if phase == "stage1" else self.stage2
        record_path = stage / "candidates" / monitor.candidate_filename(
            phase,
            int(record["step"] if phase == "stage1" else record["rollout_batches"]),
        )
        path = stage / f"resume_checkpoint_contract_{attempt_id}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "trace_vb_v8_resume_checkpoint_contract_v1",
                    "status": "PASS",
                    "checkpoint": record["checkpoint"],
                    "requested_stage": 1 if phase == "stage1" else 2,
                    "physical_gpus": "4,5,6,7",
                    "completed_candidate": str(record_path.resolve()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def log_byte_offset_floors(self) -> dict[str, int]:
        result = {}
        for path in (self.stage1 / "train.log", self.stage2 / "train.log"):
            result[str(path.resolve())] = path.stat().st_size if path.is_file() else 0
        return result

    def active_attempt(
        self,
        attempt_id: str,
        attempt_kind: str,
        status: str,
        *,
        boundary: Path | None = None,
        tmux_session: str | None = None,
        log_floors: dict[str, int] | None = None,
    ) -> dict:
        status_path = self.supervisor / f"{self.tag}.{attempt_id}.exit_status"
        exit_status = 0 if status == "SUCCEEDED" else 2 if status == "FAILED" else None
        if exit_status is not None:
            status_path.write_text(f"{exit_status}\n", encoding="ascii")
            self.config.exit_status_file.write_text(
                f"{exit_status}\n", encoding="ascii"
            )
        resume_phase = (
            attempt_kind.removeprefix("resume_")
            if attempt_kind.startswith("resume_")
            else None
        )
        value = {
            "schema_version": "trace_vb_v8_supervisor_attempt_v1",
            "pipeline_tag": self.tag,
            "physical_gpus": "4,5,6,7",
            "formal_gpus": "4,5,6,7",
            "fixed_gpus": "4,5,6,7",
            "attempt_id": attempt_id,
            "attempt_kind": attempt_kind,
            "resume_phase": resume_phase,
            "tmux_session": tmux_session or self.session,
            "exit_status_file": str(status_path.resolve()),
            "validated_boundary_contract": (
                str(boundary.resolve()) if boundary is not None else None
            ),
            "log_byte_offset_floors": (
                log_floors if log_floors is not None else self.log_byte_offset_floors()
            ),
            "status": status,
            "exit_status": exit_status,
            "started_at": "2026-08-20T21:00:00+08:00",
            "updated_at": "2026-08-20T21:00:00+08:00",
        }
        self.config.active_attempt_file.write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
        return value

    def trigger_false_report(self, *, best_correct: int = 523) -> tuple[Path, Path]:
        v7_stage = self.root / "formal_v7_stage1"
        checkpoint_dir = v7_stage / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / (
            f"epoch0__step3584__monitor{best_correct / 747:.6f}.ckpt"
        )
        checkpoint.write_bytes(b"formal-v7-best-checkpoint")
        (v7_stage / "best_checkpoint.txt").write_text(
            str(checkpoint.resolve()) + "\n", encoding="utf-8"
        )
        index_path = v7_stage / "validation_summary_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "schema_version": "trace_vb_v7_stage1_validation_index_v1",
                    "formal_epochs": 10,
                    "published_run_dir": str(v7_stage.resolve()),
                    "summaries": [{"epoch_index": value} for value in range(10)],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = v7_stage / "manifest.txt"
        manifest.write_text(
            f"validation_summary_index={index_path.resolve()}\n"
            f"best_checkpoint={checkpoint.resolve()}\n"
            "finished_at=2026-08-20T20:00:00+08:00\n",
            encoding="utf-8",
        )
        self.config.trigger_report.write_text(
            json.dumps(
                {
                    "schema_version": "trace_vb_v8_v7_trigger_contract_v1",
                    "status": "TRIGGER_FALSE",
                    "stage1_dir": str(v7_stage.resolve()),
                    "manifest": str(manifest.resolve()),
                    "validation_summary_index": str(index_path.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint),
                    "selected_epoch": 0,
                    "best_epochs": [0],
                    "best_correct_count": best_correct,
                    "validation_questions": 747,
                    "trigger_maximum_correct_count": 522,
                    "formal_epochs_verified": 10,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.config.trigger_report, checkpoint


class V8MonitorContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = MonitorFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def runtime(self):
        return monitor.load_runtime(self.fixture.state / "runtime.json")

    def test_waiting_uses_only_latest_waiter_stanza_and_accepts_gpu_4567(self):
        runner = FakeRunner(self.fixture.session)
        snapshot = monitor.collect_snapshot(
            self.fixture.config, self.runtime(), runner=runner
        )
        self.assertEqual(snapshot["lifecycle"], "waiting")
        self.assertEqual(snapshot["health"], "healthy")
        self.assertEqual(snapshot["waiter"]["trigger"], "pass")
        self.assertEqual(snapshot["waiter"]["v7_processes"], "none")
        self.assertNotIn("Traceback", "\n".join(snapshot["waiter"]["lines"]))
        self.assertFalse(snapshot["gpu"]["all_launch_eligible"])
        self.assertTrue(any(call[0] == "tmux" for call in runner.calls))
        self.assertFalse(any("new-session" in call for call in runner.calls))

    def test_partial_stage1_candidates_are_contiguous_authoritative_progress(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        self.fixture.candidate("stage1", 512, 531)
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["lifecycle"], "running")
        self.assertEqual(snapshot["phase"], "stage1")
        self.assertEqual(snapshot["health"], "healthy")
        self.assertEqual(snapshot["stage1"]["observed_coordinates"], [0, 512])
        self.assertEqual(snapshot["stage1"]["validated_candidate_count"], 2)

    def test_noncontiguous_or_sha_corrupt_candidate_is_abnormal(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        record = self.fixture.candidate("stage1", 1024, 529)
        Path(record["checkpoint"]).write_bytes(b"tampered")
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        joined = "\n".join(snapshot["errors"])
        self.assertIn("not a contiguous prefix", joined)
        self.assertIn("checkpoint SHA256 mismatch", joined)

    def test_candidate_wrong_gpu_contract_is_abnormal(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        record_path = (
            self.fixture.stage1
            / "candidates"
            / monitor.candidate_filename("stage1", 0)
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["physical_gpus"] = "0,1,2,3"
        record_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        self.assertIn("candidate 0 physical_gpus", "\n".join(snapshot["errors"]))

    def test_candidate_index_wrong_gpu_contract_is_abnormal(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.finish_stage("stage1", [527, 528, 529, 530, 531])
        index_path = self.fixture.stage1 / "candidate_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["physical_gpus"] = "0,1,2,3"
        index_path.write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        self.assertIn("index physical_gpus", "\n".join(snapshot["errors"]))

    def test_latest_waiter_stanza_and_status_gpu_sets_must_match_config(self):
        self.supervisor_waiter_log.write_text(
            "2026-08-20T21:00:00+08:00 waiting for fixed GPUs 4,5,6,8\n"
            "2026-08-20T21:00:01+08:00 v7_processes=none trigger=pass "
            "4:22000MiB:0% 5:22000MiB:0% 6:22000MiB:0% 8:22000MiB:0%\n",
            encoding="utf-8",
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        joined = "\n".join(snapshot["errors"])
        self.assertIn("latest waiter stanza fixed GPU declaration", joined)
        self.assertIn("latest waiter status GPU keys", joined)

    def test_incremental_log_scan_reports_only_new_fatal_marker(self):
        self.fixture.start_pipeline()
        stage1 = self.fixture.start_stage("stage1")
        train_log = stage1 / "train.log"
        train_log.write_text("healthy progress\n", encoding="utf-8")
        runtime = self.runtime()
        first = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(first["health"], "healthy")
        with train_log.open("a", encoding="utf-8") as stream:
            stream.write("CUDA out of memory while allocating tensor\n")
        second = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(second["health"], "abnormal")
        self.assertTrue(
            any("fatal log marker cuda_oom" in item for item in second["errors"])
        )

    def test_fatal_is_sticky_until_new_metric_safe_registered_candidate(self):
        self.fixture.start_pipeline()
        stage1 = self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        train_log = stage1 / "train.log"
        train_log.write_text("healthy progress\n", encoding="utf-8")
        runtime = self.runtime()
        monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        with train_log.open("a", encoding="utf-8") as stream:
            stream.write("CUDA out of memory while allocating tensor\n")
        failed = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertTrue(failed["fatal_state"]["active"])
        with train_log.open("a", encoding="utf-8") as stream:
            stream.write("normal output after failure\n" * 300)
        still_failed = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(still_failed["health"], "abnormal")
        self.assertTrue(still_failed["fatal_state"]["active"])
        self.assertEqual(still_failed["fatal_state"]["new_findings"], [])

        self.fixture.candidate("stage1", 512, 529)
        recovered = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(recovered["health"], "healthy")
        self.assertFalse(recovered["fatal_state"]["active"])
        self.assertEqual(
            recovered["fatal_state"]["clear_event"]["reason"],
            "newer_metric_safe_registered_validation",
        )

    def test_nccl_timeout_split_across_log_reads_is_detected(self):
        self.fixture.start_pipeline()
        stage1 = self.fixture.start_stage("stage1")
        train_log = stage1 / "train.log"
        train_log.write_text(
            "NCCL watchdog caught collective operation ", encoding="utf-8"
        )
        runtime = self.runtime()
        first = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(first["health"], "healthy")
        with train_log.open("a", encoding="utf-8") as stream:
            stream.write("timeout\n")
        second = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(second["health"], "abnormal")
        self.assertTrue(
            any("fatal log marker nccl_timeout" in item for item in second["errors"])
        )

    def test_full_completion_chain_reports_best_and_chronological_final(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        stage1_index = self.fixture.finish_stage(
            "stage1", [527, 529, 528, 528, 527]
        )
        self.fixture.start_stage("stage2")
        stage2_index = self.fixture.finish_stage(
            "stage2", [529, 530, 533, 531, 530]
        )
        self.fixture.finish_pipeline(stage1_index, stage2_index)
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session, tmux_alive=False),
        )
        self.assertEqual(snapshot["lifecycle"], "completed")
        self.assertEqual(snapshot["health"], "healthy")
        self.assertTrue(snapshot["pipeline"]["complete_chain"])
        self.assertEqual(snapshot["validation"]["best_registered"]["correct_count"], 533)
        self.assertTrue(
            snapshot["validation"]["best_registered"][
                "strictly_above_70_percent"
            ]
        )
        self.assertEqual(
            snapshot["validation"]["chronological_final"]["correct_count"], 530
        )

    def test_pipeline_complete_chain_is_false_when_stage_manifest_has_error(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        stage1_index = self.fixture.finish_stage(
            "stage1", [527, 529, 528, 528, 527]
        )
        self.fixture.start_stage("stage2")
        stage2_index = self.fixture.finish_stage(
            "stage2", [529, 530, 533, 531, 530]
        )
        self.fixture.finish_pipeline(stage1_index, stage2_index)
        manifest = self.fixture.stage1 / "manifest.txt"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                "formal_gpus=4,5,6,7", "formal_gpus=0,1,2,3"
            ),
            encoding="utf-8",
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session, tmux_alive=False),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        self.assertFalse(snapshot["pipeline"]["complete_chain"])
        self.assertIn(
            "Stage 1 has contract errors", "\n".join(snapshot["pipeline"]["errors"])
        )

    def test_metric_safe_candidate_and_manifest_gates_fail_closed(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 526)
        manifest = self.fixture.stage1 / "manifest.txt"
        text = manifest.read_text(encoding="utf-8").replace(
            monitor.METRIC_SAFE_BASELINE_SHA256, "0" * 64
        )
        manifest.write_text(text, encoding="utf-8")
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        joined = "\n".join(snapshot["errors"])
        self.assertIn("step-0 metric-safe gate is below 527/747", joined)
        self.assertIn("metric_safe_baseline_sha256", joined)

    def test_pipeline_metric_artifact_cannot_redirect_around_missing_local_copy(self):
        self.fixture.start_pipeline()
        canonical = self.fixture.pipeline / "metric_safe_baseline.json"
        canonical.unlink()
        manifest = self.fixture.pipeline / "manifest.txt"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                f"metric_safe_baseline_artifact={canonical.resolve()}",
                f"metric_safe_baseline_artifact={STUDENT_GATE_FIXTURE.resolve()}",
            ),
            encoding="utf-8",
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        joined = "\n".join(snapshot["errors"])
        self.assertIn("is not the canonical local copy", joined)
        self.assertIn("canonical local metric_safe_baseline artifact is missing", joined)

    def test_pipeline_metric_artifact_symlink_or_hardlink_alias_is_abnormal(self):
        self.fixture.start_pipeline()
        canonical = self.fixture.pipeline / "metric_safe_baseline.json"
        for alias_kind in ("symlink", "hardlink"):
            with self.subTest(alias_kind=alias_kind):
                canonical.unlink(missing_ok=True)
                if alias_kind == "symlink":
                    canonical.symlink_to(STUDENT_GATE_FIXTURE)
                    expected_error = (
                        "pipeline canonical local metric_safe_baseline artifact "
                        "is a symbolic link"
                    )
                else:
                    alias_source = self.fixture.root / "pipeline_alias_source.json"
                    shutil.copyfile(STUDENT_GATE_FIXTURE, alias_source)
                    canonical.hardlink_to(alias_source)
                    expected_error = (
                        "pipeline canonical local metric_safe_baseline artifact "
                        "is not a single-link local copy"
                    )
                snapshot = monitor.collect_snapshot(
                    self.fixture.config,
                    self.runtime(),
                    runner=FakeRunner(self.fixture.session),
                )
                self.assertEqual(snapshot["health"], "abnormal")
                self.assertIn(expected_error, "\n".join(snapshot["errors"]))

    def test_stage1_metric_artifact_cannot_redirect_around_missing_local_copy(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        canonical = self.fixture.stage1 / "metric_safe_baseline.json"
        canonical.unlink()
        manifest = self.fixture.stage1 / "manifest.txt"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                f"metric_safe_baseline_artifact={canonical.resolve()}",
                f"metric_safe_baseline_artifact={STUDENT_GATE_FIXTURE.resolve()}",
            ),
            encoding="utf-8",
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        joined = "\n".join(snapshot["errors"])
        self.assertIn("stage1 manifest metric_safe_baseline_artifact", joined)
        self.assertIn("canonical local metric_safe_baseline artifact is missing", joined)

    def test_stage1_metric_artifact_symlink_or_hardlink_alias_is_abnormal(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        canonical = self.fixture.stage1 / "metric_safe_baseline.json"
        for alias_kind in ("symlink", "hardlink"):
            with self.subTest(alias_kind=alias_kind):
                canonical.unlink(missing_ok=True)
                if alias_kind == "symlink":
                    canonical.symlink_to(STUDENT_GATE_FIXTURE)
                    expected_error = (
                        "stage1 canonical local metric_safe_baseline artifact "
                        "is a symbolic link"
                    )
                else:
                    alias_source = self.fixture.root / "stage1_alias_source.json"
                    shutil.copyfile(STUDENT_GATE_FIXTURE, alias_source)
                    canonical.hardlink_to(alias_source)
                    expected_error = (
                        "stage1 canonical local metric_safe_baseline artifact "
                        "is not a single-link local copy"
                    )
                snapshot = monitor.collect_snapshot(
                    self.fixture.config,
                    self.runtime(),
                    runner=FakeRunner(self.fixture.session),
                )
                self.assertEqual(snapshot["health"], "abnormal")
                self.assertIn(expected_error, "\n".join(snapshot["errors"]))

    def test_stage2_metric_artifacts_must_reuse_stage1_canonical_copies(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.start_stage("stage2")
        canonical = self.fixture.stage1 / "metric_safe_baseline.json"
        manifest = self.fixture.stage2 / "manifest.txt"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                f"metric_safe_baseline_artifact={canonical.resolve()}",
                f"metric_safe_baseline_artifact={STUDENT_GATE_FIXTURE.resolve()}",
            ),
            encoding="utf-8",
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        self.assertIn(
            "stage2 manifest metric_safe_baseline_artifact",
            "\n".join(snapshot["errors"]),
        )

    def test_stage2_phase_input_below_527_or_not_equal_stage1_is_abnormal(self):
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.finish_stage("stage1", [527, 527, 527, 527, 527])
        self.fixture.start_stage("stage2")
        self.fixture.candidate("stage2", 0, 526)
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        joined = "\n".join(snapshot["errors"])
        self.assertIn("phase-input metric-safe gate is below 527/747", joined)
        self.assertIn("phase-input correct_count differs", joined)

    def test_dead_exact_tmux_without_terminal_status_is_abnormal(self):
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session, tmux_alive=False),
        )
        self.assertEqual(snapshot["lifecycle"], "waiting")
        self.assertEqual(snapshot["health"], "abnormal")
        self.assertTrue(any("exact waiter tmux" in item for item in snapshot["errors"]))

    def test_terminal_trigger_false_is_a_healthy_completed_no_launch(self):
        waiter = self.supervisor_waiter_log
        waiter.write_text(
            "2026-08-20T21:00:00+08:00 waiting for fixed GPUs 4,5,6,7\n"
            "2026-08-20T21:00:01+08:00 terminal trigger_false: formal v7 best "
            "is already above 70%; v8 will not launch\n",
            encoding="utf-8",
        )
        self.fixture.trigger_false_report()
        self.fixture.active_attempt("initialA", "initial", "SUCCEEDED")
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session, tmux_alive=False),
        )
        self.assertEqual(snapshot["lifecycle"], "completed")
        self.assertEqual(snapshot["phase"], "trigger_false_no_launch")
        self.assertEqual(snapshot["health"], "healthy")
        self.assertTrue(snapshot["trigger_false_contract"]["verified"])

    def test_trigger_false_missing_machine_report_fails_closed(self):
        self.supervisor_waiter_log.write_text(
            "2026-08-20T21:00:00+08:00 waiting for fixed GPUs 4,5,6,7\n"
            "2026-08-20T21:00:01+08:00 terminal trigger_false: formal v7 best "
            "is already above 70%; v8 will not launch\n",
            encoding="utf-8",
        )
        self.fixture.active_attempt("initialA", "initial", "SUCCEEDED")
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session, tmux_alive=False),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        self.assertNotEqual(snapshot["lifecycle"], "completed")
        self.assertIn("missing JSON", "\n".join(snapshot["errors"]))

    def test_trigger_false_tampered_checkpoint_fails_closed(self):
        self.supervisor_waiter_log.write_text(
            "2026-08-20T21:00:00+08:00 waiting for fixed GPUs 4,5,6,7\n"
            "2026-08-20T21:00:01+08:00 terminal trigger_false: formal v7 best "
            "is already above 70%; v8 will not launch\n",
            encoding="utf-8",
        )
        _report, checkpoint = self.fixture.trigger_false_report()
        checkpoint.write_bytes(b"tampered")
        self.fixture.active_attempt("initialA", "initial", "SUCCEEDED")
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session, tmux_alive=False),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        self.assertNotEqual(snapshot["lifecycle"], "completed")
        self.assertIn("checkpoint SHA256 mismatch", "\n".join(snapshot["errors"]))

    def test_resume_without_active_attempt_ignores_old_initial_exit_status(self):
        attempt_id = "resumeA"
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        record = self.fixture.candidate("stage1", 0, 527)
        self.fixture.resume_binding("stage1", attempt_id)
        self.fixture.stage_resume_contract("stage1", attempt_id, record)
        with (self.fixture.stage1 / "manifest.txt").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(f"resume_attempt={attempt_id}\n")
        (self.fixture.supervisor / f"{self.fixture.tag}.exit_status").write_text(
            "0\n", encoding="ascii"
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(
            snapshot["supervisor_attempt"]["protocol"],
            "resume_without_supervisor_attempt",
        )
        self.assertIsNone(snapshot["supervisor_exit_status"])
        self.assertEqual(snapshot["ignored_legacy_supervisor_exit_status"], 0)
        self.assertNotEqual(snapshot["lifecycle"], "completed")
        self.assertTrue(
            any("intentionally ignored" in item for item in snapshot["warnings"])
        )

    def test_waiting_explicit_resume_uses_scoped_status_and_active_tmux(self):
        attempt_id = "resumeB"
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        binding_path, _record = self.fixture.resume_binding("stage1", attempt_id)
        status_path = self.fixture.supervisor / (
            f"{self.fixture.tag}.{attempt_id}.exit_status"
        )
        active = {
            "schema_version": "trace_vb_v8_supervisor_attempt_v1",
            "pipeline_tag": self.fixture.tag,
            "physical_gpus": "4,5,6,7",
            "formal_gpus": "4,5,6,7",
            "fixed_gpus": "4,5,6,7",
            "attempt_id": attempt_id,
            "attempt_kind": "resume_stage1",
            "resume_phase": "stage1",
            "tmux_session": self.fixture.session,
            "exit_status_file": str(status_path.resolve()),
            "validated_boundary_contract": str(binding_path.resolve()),
            "log_byte_offset_floors": self.fixture.log_byte_offset_floors(),
            "status": "WAITING",
            "exit_status": None,
            "started_at": "2026-08-20T21:00:00+08:00",
            "updated_at": "2026-08-20T21:00:00+08:00",
        }
        self.fixture.config.active_attempt_file.write_text(
            json.dumps(active, indent=2) + "\n", encoding="utf-8"
        )
        # This is deliberately stale and must not be consulted.
        self.fixture.config.exit_status_file.write_text("0\n", encoding="ascii")
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "healthy")
        self.assertEqual(snapshot["lifecycle"], "waiting")
        self.assertEqual(snapshot["phase"], "waiting_to_resume_stage1")
        self.assertEqual(
            snapshot["supervisor_attempt"]["protocol"], "explicit_attempt_v1"
        )
        self.assertTrue(snapshot["supervisor_attempt"]["validated_boundary"])
        self.assertIsNone(snapshot["supervisor_exit_status"])
        self.assertIsNone(snapshot["ignored_legacy_supervisor_exit_status"])
        self.assertEqual(snapshot["tmux"]["session"], self.fixture.session)

    def test_running_resume_startup_window_uses_validated_pipeline_binding(self):
        attempt_id = "resumeStartup"
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        binding_path, _record = self.fixture.resume_binding("stage1", attempt_id)
        self.fixture.active_attempt(
            attempt_id,
            "resume_stage1",
            "RUNNING",
            boundary=binding_path,
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "healthy")
        self.assertTrue(
            snapshot["supervisor_attempt"]["exit_status_authoritative"]
        )
        self.assertIsNone(
            snapshot["supervisor_attempt"]["resume_checkpoint_contract"]
        )
        self.assertTrue(
            any("has not published" in item for item in snapshot["warnings"])
        )

    def test_fresh_runtime_resume_log_floor_excludes_old_stage_fatal(self):
        attempt_id = "resumeFresh"
        self.fixture.start_pipeline()
        stage1 = self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        train_log = stage1 / "train.log"
        train_log.write_text(
            "CUDA out of memory from the failed initial attempt\n",
            encoding="utf-8",
        )
        binding_path, _record = self.fixture.resume_binding("stage1", attempt_id)
        self.fixture.active_attempt(
            attempt_id,
            "resume_stage1",
            "WAITING",
            boundary=binding_path,
            log_floors=self.fixture.log_byte_offset_floors(),
        )
        self.supervisor_waiter_log.write_text(
            "2026-08-20T21:01:00+08:00 waiting to resume stage1 on fixed GPUs "
            "4,5,6,7\n"
            "2026-08-20T21:01:01+08:00 4:22000MiB:0% 5:22000MiB:0% "
            "6:22000MiB:0% 7:22000MiB:0% v7_processes=none "
            "boundary=validated\n",
            encoding="utf-8",
        )
        fresh_runtime = monitor.load_runtime(self.fixture.root / "missing-runtime.json")
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            fresh_runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "healthy")
        self.assertFalse(snapshot["fatal_state"]["active"])
        stage1_scan = next(
            item
            for item in snapshot["log_scans"]
            if item["path"] == str(train_log)
        )
        self.assertEqual(stage1_scan["scan_start_offset"], train_log.stat().st_size)

    def test_strict_resume_floor_scans_more_than_tail_limit_without_gap(self):
        attempt_id = "resumeLargeLog"
        self.fixture.start_pipeline()
        stage1 = self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        train_log = stage1 / "train.log"
        train_log.write_text("old healthy output\n", encoding="utf-8")
        binding_path, _record = self.fixture.resume_binding("stage1", attempt_id)
        floors = self.fixture.log_byte_offset_floors()
        self.fixture.active_attempt(
            attempt_id,
            "resume_stage1",
            "WAITING",
            boundary=binding_path,
            log_floors=floors,
        )
        with train_log.open("a", encoding="utf-8") as stream:
            stream.write("CUDA out of memory at start of this resume\n")
            stream.write("healthy tail\n" * 30000)
        self.supervisor_waiter_log.write_text(
            "2026-08-20T21:01:00+08:00 waiting to resume stage1 on fixed GPUs "
            "4,5,6,7\n"
            "2026-08-20T21:01:01+08:00 4:22000MiB:0% 5:22000MiB:0% "
            "6:22000MiB:0% 7:22000MiB:0% v7_processes=none "
            "boundary=validated\n",
            encoding="utf-8",
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            monitor.load_runtime(self.fixture.root / "fresh-large-runtime.json"),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertGreater(
            train_log.stat().st_size - floors[str(train_log.resolve())],
            monitor.INITIAL_LOG_TAIL_BYTES,
        )
        self.assertEqual(snapshot["health"], "abnormal")
        self.assertTrue(
            any("fatal log marker cuda_oom" in item for item in snapshot["errors"])
        )
        stage1_scan = next(
            item
            for item in snapshot["log_scans"]
            if item["path"] == str(train_log)
        )
        self.assertEqual(
            stage1_scan["scan_start_offset"], floors[str(train_log.resolve())]
        )

    def test_fresh_runtime_initial_log_floor_excludes_prelaunch_stage_bytes(self):
        self.fixture.start_pipeline()
        stage1 = self.fixture.start_stage("stage1")
        train_log = stage1 / "train.log"
        train_log.write_text(
            "CUDA out of memory before the new initial attempt\n",
            encoding="utf-8",
        )
        floors = self.fixture.log_byte_offset_floors()
        self.fixture.active_attempt(
            "initialFresh",
            "initial",
            "WAITING",
            log_floors=floors,
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            monitor.load_runtime(self.fixture.root / "missing-runtime.json"),
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(snapshot["health"], "healthy")
        self.assertFalse(snapshot["fatal_state"]["active"])
        stage1_scan = next(
            item
            for item in snapshot["log_scans"]
            if item["path"] == str(train_log)
        )
        self.assertEqual(stage1_scan["scan_start_offset"], floors[str(train_log.resolve())])

    def test_new_initial_attempt_cannot_clear_prior_sticky_fatal(self):
        self.supervisor_waiter_log.write_text(
            "2026-08-20T21:00:00+08:00 waiting for fixed GPUs 4,5,6,7\n"
            "2026-08-20T21:00:01+08:00 4:12000MiB:80% 5:12000MiB:80% "
            "6:12000MiB:80% 7:12000MiB:80% v7_processes=none trigger=wait\n"
            "CUDA out of memory in initialA\n",
            encoding="utf-8",
        )
        runtime = self.runtime()
        failed = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertTrue(failed["fatal_state"]["active"])
        self.fixture.active_attempt("initialB", "initial", "WAITING")
        with self.supervisor_waiter_log.open("a", encoding="utf-8") as stream:
            stream.write(
                "2026-08-20T21:01:00+08:00 waiting for fixed GPUs 4,5,6,7\n"
                "2026-08-20T21:01:01+08:00 4:12000MiB:80% 5:12000MiB:80% "
                "6:12000MiB:80% 7:12000MiB:80% v7_processes=none trigger=wait\n"
            )
        second = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(second["health"], "abnormal")
        self.assertTrue(second["fatal_state"]["active"])
        self.assertIsNone(second["fatal_state"]["clear_event"])

        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        self.fixture.active_attempt("initialB", "initial", "RUNNING")
        third = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(third["health"], "healthy")
        self.assertFalse(third["fatal_state"]["active"])
        self.assertEqual(
            third["fatal_state"]["clear_event"]["reason"],
            "newer_metric_safe_registered_validation",
        )

    def test_unrelated_live_tmux_cannot_replace_configured_waiter(self):
        unrelated = "unrelated_live_tmux"
        self.fixture.active_attempt(
            "initialWrongTmux",
            "initial",
            "WAITING",
            tmux_session=unrelated,
        )
        snapshot = monitor.collect_snapshot(
            self.fixture.config,
            self.runtime(),
            runner=FakeRunner(unrelated),
        )
        self.assertEqual(snapshot["health"], "abnormal")
        self.assertEqual(snapshot["tmux"]["session"], self.fixture.session)
        joined = "\n".join(snapshot["errors"])
        self.assertIn("expected configured waiter session", joined)
        self.assertIn("exact active attempt tmux session is not alive", joined)

    def test_new_resume_waiter_stanza_supersedes_old_initial_fatal(self):
        attempt_id = "resumeC"
        self.fixture.start_pipeline()
        self.fixture.start_stage("stage1")
        self.fixture.candidate("stage1", 0, 527)
        binding_path, _record = self.fixture.resume_binding("stage1", attempt_id)
        self.supervisor_waiter_log.write_text(
            "2026-08-20T21:00:00+08:00 waiting for fixed GPUs 4,5,6,7\n"
            "2026-08-20T21:00:01+08:00 4:12000MiB:80% 5:12000MiB:80% "
            "6:12000MiB:80% 7:12000MiB:80% v7_processes=none trigger=wait\n"
            "CUDA out of memory in the initial attempt\n",
            encoding="utf-8",
        )
        runtime = self.runtime()
        failed = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertTrue(failed["fatal_state"]["active"])

        status_path = self.fixture.supervisor / (
            f"{self.fixture.tag}.{attempt_id}.exit_status"
        )
        self.fixture.config.active_attempt_file.write_text(
            json.dumps(
                {
                    "schema_version": "trace_vb_v8_supervisor_attempt_v1",
                    "pipeline_tag": self.fixture.tag,
                    "physical_gpus": "4,5,6,7",
                    "formal_gpus": "4,5,6,7",
                    "fixed_gpus": "4,5,6,7",
                    "attempt_id": attempt_id,
                    "attempt_kind": "resume_stage1",
                    "resume_phase": "stage1",
                    "tmux_session": self.fixture.session,
                    "exit_status_file": str(status_path.resolve()),
                    "validated_boundary_contract": str(binding_path.resolve()),
                    "log_byte_offset_floors": self.fixture.log_byte_offset_floors(),
                    "status": "WAITING",
                    "exit_status": None,
                    "started_at": "2026-08-20T21:01:00+08:00",
                    "updated_at": "2026-08-20T21:01:00+08:00",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        with self.supervisor_waiter_log.open("a", encoding="utf-8") as stream:
            stream.write(
                "2026-08-20T21:01:00+08:00 waiting to resume stage1 on fixed "
                "GPUs 4,5,6,7\n"
                "2026-08-20T21:01:01+08:00 4:22000MiB:0% 5:22000MiB:0% "
                "6:22000MiB:0% 7:22000MiB:0% v7_processes=none "
                "boundary=validated\n"
            )
        recovered = monitor.collect_snapshot(
            self.fixture.config,
            runtime,
            runner=FakeRunner(self.fixture.session),
        )
        self.assertEqual(recovered["waiter"]["stanza_kind"], "resume")
        self.assertEqual(recovered["waiter"]["resume_phase"], "stage1")
        self.assertEqual(recovered["waiter"]["boundary"], "validated")
        self.assertEqual(recovered["health"], "healthy")
        self.assertFalse(recovered["fatal_state"]["active"])
        self.assertEqual(
            recovered["fatal_state"]["clear_event"]["reason"],
            "validated_new_supervisor_attempt",
        )

    @property
    def supervisor_waiter_log(self) -> Path:
        return self.fixture.supervisor / f"{self.fixture.tag}_wait.log"

    def test_main_atomically_publishes_abnormal_snapshot_on_monitor_exception(self):
        state_dir = Path(self.temporary.name) / "exception-state"
        with mock.patch.object(
            monitor, "collect_snapshot", side_effect=RuntimeError("fixture explosion")
        ), redirect_stdout(io.StringIO()):
            status = monitor.main(
                [
                    "--tag",
                    self.fixture.tag,
                    "--artifact-root",
                    str(self.fixture.artifact),
                    "--physical-gpus",
                    "4,5,6,7",
                    "--waiter-session",
                    self.fixture.session,
                    "--state-dir",
                    str(state_dir),
                    "--once",
                ]
            )
        self.assertEqual(status, 2)
        latest = json.loads((state_dir / "latest.json").read_text(encoding="utf-8"))
        self.assertEqual(latest["health"], "abnormal")
        self.assertIn("fixture explosion", latest["errors"][0])
        self.assertTrue((state_dir / "runtime.json").is_file())


if __name__ == "__main__":
    unittest.main()
