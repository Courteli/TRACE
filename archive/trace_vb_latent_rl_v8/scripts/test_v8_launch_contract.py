#!/usr/bin/env python3
"""CPU-only contract checks for the v8 train-only orchestration."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "candidate_registry", SCRIPT_DIR / "trace_vb_candidate_registry.py"
)
assert spec is not None and spec.loader is not None
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)

audit_spec = importlib.util.spec_from_file_location(
    "train_val_audit", SCRIPT_DIR / "audit_train_val_only_v8.py"
)
assert audit_spec is not None and audit_spec.loader is not None
train_val_audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(train_val_audit)

trigger_spec = importlib.util.spec_from_file_location(
    "v7_trigger_audit", SCRIPT_DIR / "audit_v7_trigger_for_v8.py"
)
assert trigger_spec is not None and trigger_spec.loader is not None
v7_trigger_audit = importlib.util.module_from_spec(trigger_spec)
trigger_spec.loader.exec_module(v7_trigger_audit)

gate_spec = importlib.util.spec_from_file_location(
    "rewind_gate", SCRIPT_DIR / "audit_rewind_non_degradation_v8.py"
)
assert gate_spec is not None and gate_spec.loader is not None
rewind_gate = importlib.util.module_from_spec(gate_spec)
gate_spec.loader.exec_module(rewind_gate)

resume_binding_spec = importlib.util.spec_from_file_location(
    "pipeline_resume_binding",
    SCRIPT_DIR / "audit_pipeline_resume_binding_v8.py",
)
assert resume_binding_spec is not None and resume_binding_spec.loader is not None
pipeline_resume_binding = importlib.util.module_from_spec(resume_binding_spec)
resume_binding_spec.loader.exec_module(pipeline_resume_binding)

activity_spec = importlib.util.spec_from_file_location(
    "v7_activity_detector",
    SCRIPT_DIR / "detect_active_v7_training_v8.py",
)
assert activity_spec is not None and activity_spec.loader is not None
v7_activity_detector = importlib.util.module_from_spec(activity_spec)
activity_spec.loader.exec_module(v7_activity_detector)

publication_spec = importlib.util.spec_from_file_location(
    "pipeline_publication",
    SCRIPT_DIR / "publish_pipeline_completion_v8.py",
)
assert publication_spec is not None and publication_spec.loader is not None
pipeline_publication = importlib.util.module_from_spec(publication_spec)
publication_spec.loader.exec_module(pipeline_publication)

supervisor_spec = importlib.util.spec_from_file_location(
    "supervisor_attempt",
    SCRIPT_DIR / "supervisor_attempt_v8.py",
)
assert supervisor_spec is not None and supervisor_spec.loader is not None
supervisor_attempt = importlib.util.module_from_spec(supervisor_spec)
supervisor_spec.loader.exec_module(supervisor_attempt)


REGISTERED_SHA = "d27ef63b1d462aa94fbbecc636cec31985a3687e5d2d036de9cfd87b01ec1525"
PAYLOAD_SHA = "b1e9a973bbf4f2eeaa18b7d49df16c38db50cea87e998008ba18a46cdff9e049"
V7_SHA = "c" * 64
COT_SHA = "df90292c2da854852651a9e75b1e8221484c6dc58b8c81a56400897a8dec7f54"
RESET_TARGET_NAMES = [
    "trajectory_policy.mean_heads.plan.weight",
    "trajectory_policy.mean_heads.plan.bias",
    "trajectory_policy.mean_heads.solve.weight",
    "trajectory_policy.mean_heads.solve.bias",
    "trajectory_policy.mean_heads.check.weight",
    "trajectory_policy.mean_heads.check.bias",
    "trajectory_policy.mean_heads.commit.weight",
    "trajectory_policy.mean_heads.commit.bias",
    "trajectory_policy.action_projector.0.bias",
]
METRIC_BASELINE_PATH = (
    "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
    "20260818-213000_trace_vb_v7_full_seed0/student_initial_gate.json"
)
METRIC_BASELINE_SHA = (
    "9b888b36956f70affae153526b716ad26343164b4aa66d6d3d54f70561144aa2"
)
CAPABILITY_VALIDATION_PATH = (
    "/disk1/dingxukai/TRACE/trace_vb_v7_runs/pipelines/"
    "20260818-213000_trace_vb_v7_full_seed0/capability_parity_gate.json"
)
CAPABILITY_VALIDATION_SHA = (
    "59123d2bfc020335f54d70862a904e1b70e5ea1ce764483692809ae4e6772116"
)


class V7ActivityDetectorTest(unittest.TestCase):
    V7_ROOT = Path("/home/dingxukai/TRACE/trace_vb_latent_rl_v7")

    @staticmethod
    def process(
        argv: list[str],
        *,
        comm: str,
        cwd: str = "/home/dingxukai",
        state: str = "S",
        pid: int = 100,
    ) -> dict:
        return {
            "pid": pid,
            "ppid": 1,
            "state": state,
            "comm": comm,
            "argv": argv,
            "cwd": cwd,
        }

    def test_tmux_server_and_waiters_are_not_training(self) -> None:
        root = str(self.V7_ROOT)
        fixtures = (
            self.process(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    "trace_vb_v7_resume_wait_20260819",
                    f"bash {root}/scripts/run_stage1_vb.sh 2,3,4,5",
                ],
                comm="tmux: server",
                pid=75601,
            ),
            self.process(
                [
                    "bash",
                    f"{root}/scripts/wait_for_four_gpus_and_run_full_vb.sh",
                ],
                comm="bash",
                pid=75602,
            ),
            self.process(
                [
                    "bash",
                    "-lc",
                    f"bash {root}/scripts/run_stage1_vb.sh 2,3,4,5",
                ],
                comm="bash",
                pid=75603,
            ),
            self.process(
                [
                    "/env/bin/python",
                    "-c",
                    "print(1)",
                    f"{root}/run.py",
                ],
                comm="python",
                pid=75604,
            ),
            self.process(
                [
                    "/env/bin/python",
                    "-m",
                    "unrelated.module",
                    f"{root}/run.py",
                ],
                comm="python",
                pid=75605,
            ),
            self.process(
                [
                    "bash",
                    "/tmp/unrelated.sh",
                    f"{root}/scripts/run_stage1_vb.sh",
                ],
                comm="bash",
                pid=75606,
            ),
        )
        self.assertEqual(
            v7_activity_detector.active_v7_processes(fixtures), []
        )

    def test_real_stage_wrappers_and_run_py_ranks_block(self) -> None:
        root = str(self.V7_ROOT)
        fixtures = (
            self.process(
                ["bash", "scripts/run_stage1_vb.sh", "2,3,4,5"],
                comm="bash",
                cwd=root,
                pid=80001,
            ),
            self.process(
                ["bash", f"{root}/scripts/run_stage2_vb.sh", "2,3,4,5"],
                comm="bash",
                pid=80002,
            ),
            self.process(
                ["/env/bin/python", "run.py", "--devices", "0,1,2,3"],
                comm="python",
                cwd=root,
                pid=80003,
            ),
            self.process(
                ["/env/bin/python", f"{root}/run.py", "--local_rank=1"],
                comm="python",
                pid=80004,
            ),
            self.process(
                [
                    "/env/bin/python",
                    "-X",
                    "dev",
                    f"{root}/run.py",
                    "--local_rank=2",
                ],
                comm="python",
                pid=80005,
            ),
            self.process(
                [
                    "bash",
                    "-euo",
                    "pipefail",
                    f"{root}/scripts/run_stage1_vb.sh",
                ],
                comm="bash",
                pid=80006,
            ),
        )
        matches = v7_activity_detector.active_v7_processes(fixtures)
        self.assertEqual(
            [item["pid"] for item in matches],
            [80001, 80002, 80003, 80004, 80005, 80006],
        )
        self.assertEqual(
            [item["reason"] for item in matches],
            [
                "v7_stage_wrapper:run_stage1_vb.sh",
                "v7_stage_wrapper:run_stage2_vb.sh",
                "v7_run_py_rank",
                "v7_run_py_rank",
                "v7_run_py_rank",
                "v7_stage_wrapper:run_stage1_vb.sh",
            ],
        )

    def test_unrelated_and_dead_processes_do_not_block_cli(self) -> None:
        root = str(self.V7_ROOT)
        inactive = [
            self.process(
                ["/env/bin/python", "run.py"],
                comm="python",
                cwd="/home/dingxukai/TRACE/trace_vb_latent_rl_v8",
                pid=81001,
            ),
            self.process(
                ["/env/bin/python", f"{root}/run.py"],
                comm="python",
                state="Z",
                pid=81002,
            ),
            self.process(
                ["sed", f"{root}/scripts/run_stage1_vb.sh"],
                comm="sed",
                pid=81003,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "processes.json"
            snapshot.write_text(json.dumps(inactive), encoding="utf-8")
            command = [
                sys.executable,
                str(SCRIPT_DIR / "detect_active_v7_training_v8.py"),
                "--snapshot",
                str(snapshot),
                "--quiet",
            ]
            self.assertEqual(subprocess.run(command).returncode, 3)
            inactive.append(
                self.process(
                    ["bash", f"{root}/scripts/run_stage1_vb.sh"],
                    comm="bash",
                    pid=81004,
                )
            )
            snapshot.write_text(json.dumps(inactive), encoding="utf-8")
            self.assertEqual(subprocess.run(command).returncode, 0)
            snapshot.write_text("not-json", encoding="utf-8")
            self.assertEqual(subprocess.run(command).returncode, 2)
            snapshot.write_text(json.dumps([None]), encoding="utf-8")
            self.assertEqual(subprocess.run(command).returncode, 2)

    def test_internal_detector_exceptions_are_never_inactive(self) -> None:
        for error in (AttributeError("broken record"), IndexError("broken proc stat")):
            with self.subTest(error=type(error).__name__), patch.object(
                v7_activity_detector,
                "live_process_snapshot",
                side_effect=error,
            ), patch.object(sys, "argv", ["detect_active_v7_training_v8.py", "--quiet"]), patch(
                "builtins.print"
            ), self.assertRaises(SystemExit) as raised:
                v7_activity_detector.main()
            self.assertEqual(raised.exception.code, 2)


class GPUSetContractTest(unittest.TestCase):
    def run_contract(
        self,
        *,
        formal: str,
        requested: str,
        fixed: str | None = None,
        free_mib: int = 23000,
        utilization: int = 0,
    ) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temporary:
            fake_bin = Path(temporary)
            fake_nvidia_smi = fake_bin / "nvidia-smi"
            fake_nvidia_smi.write_text(
                "#!/usr/bin/env bash\n"
                "printf '24576, %s, %s\\n' "
                '"${FAKE_GPU_FREE_MIB}" "${FAKE_GPU_UTILIZATION}"\n',
                encoding="utf-8",
            )
            fake_nvidia_smi.chmod(0o700)
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "TRACE_VB_FORMAL_GPUS": formal,
                    "TRACE_VB_FIXED_GPUS": fixed or formal,
                    "FAKE_GPU_FREE_MIB": str(free_mib),
                    "FAKE_GPU_UTILIZATION": str(utilization),
                }
            )
            return subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; trace_vb_require_four_gpus "$2"',
                    "gpu-contract",
                    str(SCRIPT_DIR / "trace_vb_common.sh"),
                    requested,
                ],
                env=environment,
                text=True,
                capture_output=True,
            )

    def test_nondefault_four_gpu_override_passes(self) -> None:
        result = self.run_contract(formal="4,5,6,7", requested="4,5,6,7")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cardinality_duplicate_and_inconsistent_sets_fail(self) -> None:
        for requested in ("4,5,6", "4,5,5,7", "2,3,4,5"):
            with self.subTest(requested=requested):
                self.assertEqual(
                    self.run_contract(
                        formal="4,5,6,7", requested=requested
                    ).returncode,
                    2,
                )
        self.assertEqual(
            self.run_contract(
                formal="4,5,6,7",
                fixed="2,3,4,5",
                requested="4,5,6,7",
            ).returncode,
            2,
        )

    def test_memory_and_idle_thresholds_are_enforced(self) -> None:
        self.assertEqual(
            self.run_contract(
                formal="4,5,6,7", requested="4,5,6,7", free_mib=21499
            ).returncode,
            2,
        )
        self.assertEqual(
            self.run_contract(
                formal="4,5,6,7", requested="4,5,6,7", utilization=11
            ).returncode,
            2,
        )


def write_checkpoint(
    path: Path,
    *,
    stage: int,
    global_step: int,
    rollout_batches: int = 0,
    epoch: int = 0,
    full_state: bool = False,
    boundary_batches: int | None = None,
    completed_intervals: int | None = None,
    policy_value: float = 1.0,
    reference_value: float | None = None,
    step0_zero_reset: bool = False,
    registered_sha: str = REGISTERED_SHA,
    payload_sha: str = PAYLOAD_SHA,
    v7_sha: str = V7_SHA,
    cot_sha: str = COT_SHA,
) -> None:
    payload = {
        "trace_vb_schema_version": "trace_vb_v8",
        "trace_policy_training_stage": stage,
        "trace_vb_capability_spine_rewound": True,
        "trace_vb_capability_spine_rewind_count": 504,
        "trace_vb_registered_capability_checkpoint_sha256": registered_sha,
        "trace_vb_registered_capability_payload_sha256": payload_sha,
        "trace_vb_v7_source_checkpoint_sha256": v7_sha,
        "trace_vb_cot_encoder_checkpoint_sha256": cot_sha,
        "trace_vb_zero_action_reset_schema_version": (
            "trace_vb_v8_zero_action_reset_v1"
        ),
        "trace_vb_zero_action_reset_applied": True,
        "trace_vb_zero_action_reset_operation_count": 1,
        "trace_vb_zero_action_reset_tensor_count": 9,
        "trace_vb_zero_action_reset_target_names": RESET_TARGET_NAMES,
        "trace_vb_zero_action_reset_source_schema": "trace_vb_v7",
        "trace_vb_zero_action_reset_source_checkpoint_sha256": v7_sha,
        "trace_vb_metric_safe_baseline_path": METRIC_BASELINE_PATH,
        "trace_vb_metric_safe_baseline_sha256": METRIC_BASELINE_SHA,
        "trace_vb_metric_safe_baseline_correct_count": 527,
        "trace_vb_metric_safe_baseline_questions": 747,
        "trace_vb_registered_capability_validation_path": (
            CAPABILITY_VALIDATION_PATH
        ),
        "trace_vb_registered_capability_validation_sha256": (
            CAPABILITY_VALIDATION_SHA
        ),
        "trace_vb_registered_capability_validation_correct_count": 540,
        "trace_vb_registered_capability_validation_questions": 747,
        "global_step": global_step,
        "epoch": epoch,
        "state_dict": {
            "vb_rollout_batches_seen": torch.tensor(rollout_batches),
            "trajectory_policy.weight": torch.tensor([policy_value]),
        },
    }
    if step0_zero_reset:
        payload["state_dict"].update(
            {name: torch.zeros(1) for name in RESET_TARGET_NAMES}
        )
    if stage == 2:
        payload["trace_stage1_policy_reference"] = {
            "weight": torch.tensor(
                [policy_value if reference_value is None else reference_value]
            )
        }
    if full_state:
        if boundary_batches is None or completed_intervals is None:
            raise ValueError(
                "full-state test checkpoints require an exact interval boundary"
            )
        epoch_counts = {
            "ready": completed_intervals,
            "completed": completed_intervals - 1,
            "started": completed_intervals,
            "processed": completed_intervals,
        }
        batch_counts = {
            key: boundary_batches
            for key in ("ready", "completed", "started", "processed")
        }
        payload.update(
            {
                "loops": {
                    "fit_loop": {
                        "epoch_loop.batch_progress": {
                            "total": dict(batch_counts),
                            "current": dict(batch_counts),
                            "is_last_batch": True,
                        },
                        "epoch_progress": {
                            "total": dict(epoch_counts),
                            "current": dict(epoch_counts),
                        },
                    }
                },
                "optimizer_states": [{}],
                "lr_schedulers": [{}],
            }
        )
    torch.save(payload, path)


def registry_namespace(**values) -> Namespace:
    defaults = {
        "physical_gpus": "4,5,6,7",
        "registered_checkpoint_sha256": REGISTERED_SHA,
        "registered_payload_sha256": PAYLOAD_SHA,
        "v7_source_sha256": V7_SHA,
        "cot_encoder_sha256": COT_SHA,
        "stage1_reference_checkpoint": None,
    }
    defaults.update(values)
    return Namespace(**defaults)


def summary(correct: int, global_step: int = 0) -> dict:
    return {
        "schema_version": "trace_vb_v8_validation_behavior_v1",
        "validation_path": "student_commit",
        "world_size": 4,
        # Match the distributed validation reducer's serialized count.  The
        # registry must preserve exact-count safety for both 747 and 747.0.
        "unique_questions": 747.0,
        "correct_count": correct,
        "accuracy": correct / 747,
        "global_step": global_step,
        "valid_answer_fraction": 0.99,
        "unique_prediction_ratio": 0.80,
        "top1_mode_fraction": 0.05,
        "nonempty_output_fraction": 1.0,
    }


def write_v7_fixture(
    root: Path,
    *,
    best_correct: int = 522,
    summary_count: int = 10,
    checkpoint_name: str | None = None,
) -> Namespace:
    stage1_dir = root / "published_stage1"
    run_dir = root / "formal_run"
    checkpoint_dir = run_dir / "checkpoints"
    stage1_dir.mkdir()
    checkpoint_dir.mkdir(parents=True)
    selected_epoch = 6
    score = best_correct / 747
    checkpoint = checkpoint_dir / (
        checkpoint_name
        or f"epoch{selected_epoch}__step3584__monitor{score:.9f}.ckpt"
    )
    checkpoint.write_bytes(b"formal-v7-best")
    entries = []
    for epoch in range(summary_count):
        correct = best_correct if epoch == selected_epoch else min(520, best_correct - 1)
        payload = {
            "schema_version": "trace_vb_v7_validation_behavior_v1",
            "validation_path": "student_commit",
            "world_size": 4,
            "unique_questions": 747,
            "epoch_index": epoch,
            "correct_count": correct,
            "accuracy": correct / 747,
        }
        path = run_dir / f"validation_epoch_{epoch:03d}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        entries.append(
            {
                "epoch_index": epoch,
                "checkpoint_run_copy": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    index_path = stage1_dir / "validation_summary_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "trace_vb_v7_stage1_validation_index_v1",
                "formal_epochs": 10,
                "published_run_dir": str(run_dir),
                "summaries": entries,
            }
        ),
        encoding="utf-8",
    )
    (stage1_dir / "best_checkpoint.txt").write_text(
        str(checkpoint) + "\n", encoding="utf-8"
    )
    (stage1_dir / "manifest.txt").write_text(
        "\n".join(
            (
                "finished_at=2026-08-20T00:00:00+08:00",
                f"validation_summary_index={index_path}",
                f"best_checkpoint={checkpoint}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return Namespace(
        stage1_dir=stage1_dir,
        checkpoint=None,
        output=root / "trigger.json",
    )


class CandidateRegistryTest(unittest.TestCase):
    def test_exact_selection_keeps_earliest_tie(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for step, correct in ((0, 520), (512, 530), (1024, 530), (1536, 528), (2048, 529)):
                summary_path = root / f"summary_{step}.json"
                checkpoint_path = root / f"checkpoint_{step}.ckpt"
                record_path = root / f"candidate_{step}.json"
                summary_path.write_text(
                    json.dumps(summary(correct, step)), encoding="utf-8"
                )
                write_checkpoint(
                    checkpoint_path, stage=1, global_step=step, epoch=step // 512
                )
                registry.register(
                    registry_namespace(
                        summary=summary_path,
                        checkpoint=checkpoint_path,
                        output=record_path,
                        validation_schema="trace_vb_v8_validation_behavior_v1",
                        questions=747,
                        world_size=4,
                        phase="stage1",
                        step=step,
                        phase_input=False,
                        rollout_batches=None,
                    )
                )
                records.append(record_path)
            output = root / "index.json"
            best = root / "best.txt"
            registry.select(
                registry_namespace(
                    candidates=records,
                    expected_steps="0,512,1024,1536,2048",
                    expected_rollout_batches=None,
                    phase="stage1",
                    output=output,
                    best_checkpoint_record=best,
                )
            )
            index = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(index["candidate_count"], 5)
            self.assertEqual(index["physical_gpus"], "4,5,6,7")
            self.assertEqual(index["selected_step"], 512)
            self.assertEqual(index["selected_correct_count"], 530)
            replay_output = root / "replay_index.json"
            replay_best = root / "replay_best.txt"
            registry.select(
                registry_namespace(
                    candidates=records,
                    expected_steps="0,512,1024,1536,2048",
                    expected_rollout_batches=None,
                    phase="stage1",
                    output=replay_output,
                    best_checkpoint_record=replay_best,
                )
            )
            self.assertEqual(replay_output.read_bytes(), output.read_bytes())
            self.assertEqual(replay_best.read_bytes(), best.read_bytes())
            with self.assertRaises(SystemExit):
                registry.select(
                    registry_namespace(
                        physical_gpus="2,3,4,5",
                        candidates=records,
                        expected_steps="0,512,1024,1536,2048",
                        expected_rollout_batches=None,
                        phase="stage1",
                        output=root / "wrong_gpu_index.json",
                        best_checkpoint_record=root / "wrong_gpu_best.txt",
                    )
                )

    def test_inexact_accuracy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = summary(520)
            payload["accuracy"] += 1e-4
            summary_path = root / "summary.json"
            checkpoint_path = root / "checkpoint.ckpt"
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            write_checkpoint(checkpoint_path, stage=1, global_step=0)
            with self.assertRaises(SystemExit):
                registry.register(
                    registry_namespace(
                        summary=summary_path,
                        checkpoint=checkpoint_path,
                        output=root / "record.json",
                        validation_schema="trace_vb_v8_validation_behavior_v1",
                        questions=747,
                        world_size=4,
                        phase="stage1",
                        step=0,
                        phase_input=False,
                        rollout_batches=None,
                    )
                )

    def test_higher_but_unsafe_candidate_is_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for step, correct in ((0, 520), (512, 540)):
                payload = summary(correct, step)
                if step == 512:
                    payload["top1_mode_fraction"] = 0.50
                summary_path = root / f"summary_{step}.json"
                checkpoint_path = root / f"checkpoint_{step}.ckpt"
                record_path = root / f"candidate_{step}.json"
                summary_path.write_text(json.dumps(payload), encoding="utf-8")
                write_checkpoint(
                    checkpoint_path, stage=1, global_step=step, epoch=step // 512
                )
                registry.register(
                    registry_namespace(
                        summary=summary_path,
                        checkpoint=checkpoint_path,
                        output=record_path,
                        validation_schema="trace_vb_v8_validation_behavior_v1",
                        questions=747,
                        world_size=4,
                        phase="stage1",
                        step=step,
                        phase_input=False,
                        rollout_batches=None,
                    )
                )
                records.append(record_path)
            output = root / "index.json"
            registry.select(
                registry_namespace(
                    candidates=records,
                    expected_steps="0,512",
                    expected_rollout_batches=None,
                    phase="stage1",
                    output=output,
                    best_checkpoint_record=root / "best.txt",
                )
            )
            index = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(index["selected_step"], 0)
            self.assertEqual(index["eligible_candidate_count"], 1)

    def test_global_step_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "summary.json"
            checkpoint_path = root / "checkpoint.ckpt"
            summary_path.write_text(json.dumps(summary(520, 511)), encoding="utf-8")
            write_checkpoint(checkpoint_path, stage=1, global_step=511)
            with self.assertRaises(SystemExit):
                registry.register(
                    registry_namespace(
                        summary=summary_path,
                        checkpoint=checkpoint_path,
                        output=root / "record.json",
                        validation_schema="trace_vb_v8_validation_behavior_v1",
                        questions=747,
                        world_size=4,
                        phase="stage1",
                        step=512,
                        phase_input=False,
                        rollout_batches=None,
                    )
                )

    def test_stage2_input_remap_and_variable_optimizer_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1_reference = root / "checkpoint_0.ckpt"
            write_checkpoint(
                stage1_reference, stage=1, global_step=1536, policy_value=7.0
            )
            specs = (
                (0, 1536, 0, True, 520),
                (401, 401, 256, False, 525),
                (807, 807, 512, False, 526),
            )
            records = []
            for selection_step, source_step, rollout, phase_input, correct in specs:
                summary_path = root / f"summary_{rollout}.json"
                checkpoint_path = root / f"checkpoint_{rollout}.ckpt"
                record_path = root / f"candidate_{rollout}.json"
                summary_path.write_text(
                    json.dumps(summary(correct, source_step)), encoding="utf-8"
                )
                if not phase_input:
                    write_checkpoint(
                        checkpoint_path,
                        stage=2,
                        global_step=source_step,
                        rollout_batches=rollout,
                        policy_value=7.0,
                    )
                registry.register(
                    registry_namespace(
                        summary=summary_path,
                        checkpoint=checkpoint_path,
                        output=record_path,
                        validation_schema="trace_vb_v8_validation_behavior_v1",
                        questions=747,
                        world_size=4,
                        phase="stage2",
                        step=selection_step,
                        phase_input=phase_input,
                        rollout_batches=rollout,
                        stage1_reference_checkpoint=stage1_reference,
                    )
                )
                records.append(record_path)
            output = root / "index.json"
            registry.select(
                registry_namespace(
                    candidates=records,
                    expected_steps=None,
                    expected_rollout_batches="0,256,512",
                    phase="stage2",
                    output=output,
                    best_checkpoint_record=root / "best.txt",
                    stage1_reference_checkpoint=stage1_reference,
                )
            )
            index = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(index["observed_selection_steps"], [0, 401, 807])
            self.assertEqual(index["candidates"][0]["source_global_step"], 1536)
            self.assertTrue(index["candidates"][0]["phase_input"])

    def test_checkpoint_summary_and_rollout_binding_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "summary.json"
            checkpoint_path = root / "candidate.ckpt"
            summary_path.write_text(json.dumps(summary(520, 512)), encoding="utf-8")
            write_checkpoint(checkpoint_path, stage=1, global_step=511)
            with self.assertRaises(SystemExit):
                registry.register(
                    registry_namespace(
                        summary=summary_path,
                        checkpoint=checkpoint_path,
                        output=root / "record.json",
                        validation_schema="trace_vb_v8_validation_behavior_v1",
                        questions=747,
                        world_size=4,
                        phase="stage1",
                        step=512,
                        phase_input=False,
                        rollout_batches=None,
                    )
                )

    def test_stage2_wrong_stage1_policy_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1 = root / "stage1.ckpt"
            stage2 = root / "stage2.ckpt"
            summary_path = root / "summary.json"
            write_checkpoint(stage1, stage=1, global_step=512, policy_value=1.0)
            write_checkpoint(
                stage2,
                stage=2,
                global_step=400,
                rollout_batches=256,
                policy_value=1.0,
                reference_value=2.0,
            )
            summary_path.write_text(json.dumps(summary(525, 400)), encoding="utf-8")
            with self.assertRaises(SystemExit):
                registry.register(
                    registry_namespace(
                        summary=summary_path,
                        checkpoint=stage2,
                        output=root / "record.json",
                        validation_schema="trace_vb_v8_validation_behavior_v1",
                        questions=747,
                        world_size=4,
                        phase="stage2",
                        step=400,
                        phase_input=False,
                        rollout_batches=256,
                        stage1_reference_checkpoint=stage1,
                    )
                )


class TriggerAndRewindGateTest(unittest.TestCase):
    def test_trigger_resolves_formal_best_without_injected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = write_v7_fixture(Path(temporary))
            v7_trigger_audit.audit(args)
            report = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["best_correct_count"], 522)
            self.assertIn("epoch6__step3584", report["checkpoint"])

    def test_trigger_rejects_incomplete_or_nonvalidation_best(self) -> None:
        for summary_count, checkpoint_name in (
            (9, None),
            (10, "last.ckpt"),
            (10, "recovery-004.ckpt"),
        ):
            with self.subTest(summary_count=summary_count, name=checkpoint_name):
                with tempfile.TemporaryDirectory() as temporary:
                    args = write_v7_fixture(
                        Path(temporary),
                        summary_count=summary_count,
                        checkpoint_name=checkpoint_name,
                    )
                    with self.assertRaises(SystemExit):
                        v7_trigger_audit.audit(args)

    def test_trigger_false_is_terminal_and_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = write_v7_fixture(Path(temporary), best_correct=523)
            with self.assertRaises(v7_trigger_audit.TriggerFalse):
                v7_trigger_audit.audit(args)
            report = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "TRIGGER_FALSE")
            self.assertEqual(report["best_correct_count"], 523)

    def test_rewind_gate_rejects_any_exact_count_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trigger = root / "trigger.json"
            validation = root / "validation.json"
            checkpoint = root / "step0.ckpt"
            write_checkpoint(
                checkpoint,
                stage=1,
                global_step=0,
                step0_zero_reset=True,
            )
            trigger.write_text(
                json.dumps(
                    {
                        "schema_version": "trace_vb_v8_v7_trigger_contract_v1",
                        "status": "PASS",
                        "best_correct_count": 522,
                    }
                ),
                encoding="utf-8",
            )
            validation.write_text(json.dumps(summary(526, 0)), encoding="utf-8")
            args = Namespace(
                trigger_contract=trigger,
                validation_summary=validation,
                validation_schema="trace_vb_v8_validation_behavior_v1",
                step0_checkpoint=checkpoint,
                metric_safe_baseline=Path(METRIC_BASELINE_PATH),
                registered_capability_validation=Path(
                    CAPABILITY_VALIDATION_PATH
                ),
                output=root / "gate.json",
            )
            with self.assertRaises(SystemExit):
                rewind_gate.audit(args)
            validation.write_text(json.dumps(summary(527, 0)), encoding="utf-8")
            report = rewind_gate.audit(args)
            self.assertEqual(report["non_degradation_margin"], 0)
            self.assertEqual(report["required_step0_correct_count"], 527)
            self.assertTrue(
                report["step0_checkpoint"]["reset_tensors_exact_zero"]
            )

    def test_metric_baseline_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tampered = root / "student_initial_gate.json"
            tampered.write_bytes(Path(METRIC_BASELINE_PATH).read_bytes() + b" ")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                rewind_gate.audit_validation_artifact(
                    tampered,
                    origin=rewind_gate.METRIC_SAFE_BASELINE_ORIGIN,
                    expected_sha256=rewind_gate.METRIC_SAFE_BASELINE_SHA256,
                    validation_path="student_commit",
                    correct_count=527,
                )


class ResumeCheckpointContractTest(unittest.TestCase):
    def run_resume_audit(
        self,
        *,
        checkpoint: Path,
        candidate: Path,
        stage: int,
        stage1_reference: Path | None = None,
        completed_intervals: int = 1,
        physical_gpus: str = "4,5,6,7",
        expect_success: bool,
    ) -> subprocess.CompletedProcess:
        command = [
            sys.executable,
            str(SCRIPT_DIR / "audit_resume_checkpoint_v8.py"),
            "--checkpoint",
            str(checkpoint),
            "--stage",
            str(stage),
            "--completed-intervals",
            str(completed_intervals),
            "--completed-candidate",
            str(candidate),
            "--physical-gpus",
            physical_gpus,
            "--registered-checkpoint-sha256",
            REGISTERED_SHA,
            "--registered-payload-sha256",
            PAYLOAD_SHA,
            "--v7-source-sha256",
            V7_SHA,
            "--cot-encoder-sha256",
            COT_SHA,
            "--output",
            str(checkpoint.parent / "resume_report.json"),
        ]
        if stage == 1:
            command.extend(("--stage1-interval-steps", "512"))
        else:
            command.extend(("--interval-rollout-batches", "256"))
            assert stage1_reference is not None
            command.extend(("--stage1-reference-checkpoint", str(stage1_reference)))
        result = subprocess.run(command, text=True, capture_output=True)
        if expect_success and result.returncode != 0:
            self.fail(result.stderr)
        if not expect_success:
            self.assertNotEqual(result.returncode, 0)
        return result

    @staticmethod
    def candidate_record(
        path: Path,
        checkpoint: Path,
        *,
        stage: int,
        completed_intervals: int,
    ) -> None:
        checkpoint_value = registry.load_checkpoint(checkpoint)
        checkpoint_stage = registry.checkpoint_stage(checkpoint_value)
        global_step = registry.checkpoint_global_step(checkpoint_value)
        epoch = registry.checkpoint_epoch(checkpoint_value)
        rollouts = registry.rollout_batches_seen(checkpoint_value)
        coordinate = completed_intervals * (512 if stage == 1 else 256)
        phase_input = stage == 2 and completed_intervals == 0
        path.write_text(
            json.dumps(
                {
                    "schema_version": "trace_vb_v8_candidate_v1",
                    "phase": f"stage{stage}",
                    "physical_gpus": "4,5,6,7",
                    "step": coordinate if stage == 1 else (
                        0 if phase_input else global_step
                    ),
                    "selection_step": coordinate if stage == 1 else (
                        0 if phase_input else global_step
                    ),
                    "source_global_step": global_step,
                    "phase_input": phase_input,
                    "rollout_batches": None if stage == 1 else coordinate,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "checkpoint_stage": checkpoint_stage,
                    "checkpoint_global_step": global_step,
                    "checkpoint_epoch": epoch,
                    "checkpoint_rollout_batches_seen": rollouts,
                }
            ),
            encoding="utf-8",
        )

    def test_boundary_zero_weights_only_restart_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundary = root / "boundary0.ckpt"
            record = root / "candidate.json"
            write_checkpoint(boundary, stage=1, global_step=0, epoch=0)
            self.candidate_record(
                record, boundary, stage=1, completed_intervals=0
            )
            self.run_resume_audit(
                checkpoint=boundary,
                candidate=record,
                stage=1,
                completed_intervals=0,
                expect_success=True,
            )
            self.run_resume_audit(
                checkpoint=boundary,
                candidate=record,
                stage=1,
                completed_intervals=0,
                physical_gpus="2,3,4,5",
                expect_success=False,
            )

    def test_full_state_boundary_passes_but_midpoint_and_bad_loop_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boundary = root / "boundary.ckpt"
            midpoint = root / "midpoint.ckpt"
            copied_boundary = root / "copied_boundary.ckpt"
            record = root / "candidate.json"
            write_checkpoint(
                boundary,
                stage=1,
                global_step=512,
                epoch=0,
                full_state=True,
                boundary_batches=512,
                completed_intervals=1,
            )
            self.candidate_record(
                record, boundary, stage=1, completed_intervals=1
            )
            self.run_resume_audit(
                checkpoint=boundary,
                candidate=record,
                stage=1,
                expect_success=True,
            )
            write_checkpoint(
                midpoint,
                stage=1,
                global_step=600,
                epoch=0,
                full_state=True,
                boundary_batches=88,
                completed_intervals=1,
            )
            self.run_resume_audit(
                checkpoint=midpoint,
                candidate=record,
                stage=1,
                expect_success=False,
            )
            copied_boundary.write_bytes(boundary.read_bytes())
            self.run_resume_audit(
                checkpoint=copied_boundary,
                candidate=record,
                stage=1,
                expect_success=False,
            )
            checkpoint_value = registry.load_checkpoint(boundary)
            checkpoint_value["loops"]["fit_loop"][
                "epoch_loop.batch_progress"
            ]["current"]["completed"] = 511
            torch.save(checkpoint_value, boundary)
            self.candidate_record(
                record, boundary, stage=1, completed_intervals=1
            )
            self.run_resume_audit(
                checkpoint=boundary,
                candidate=record,
                stage=1,
                expect_success=False,
            )

    def test_stage2_variable_optimizer_step_boundary_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1 = root / "stage1.ckpt"
            boundary = root / "stage2_boundary.ckpt"
            record = root / "candidate.json"
            write_checkpoint(stage1, stage=1, global_step=512, policy_value=1.0)
            write_checkpoint(
                boundary,
                stage=2,
                global_step=401,
                rollout_batches=256,
                epoch=0,
                full_state=True,
                boundary_batches=256,
                completed_intervals=1,
                policy_value=1.0,
            )
            self.candidate_record(
                record, boundary, stage=2, completed_intervals=1
            )
            self.run_resume_audit(
                checkpoint=boundary,
                candidate=record,
                stage=2,
                stage1_reference=stage1,
                expect_success=True,
            )

    def test_resume_rejects_wrong_stage1_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1 = root / "stage1.ckpt"
            boundary = root / "boundary.ckpt"
            record = root / "candidate.json"
            write_checkpoint(stage1, stage=1, global_step=512, policy_value=1.0)
            write_checkpoint(
                boundary,
                stage=2,
                global_step=400,
                rollout_batches=256,
                epoch=0,
                full_state=True,
                boundary_batches=256,
                completed_intervals=1,
                policy_value=1.0,
                reference_value=2.0,
            )
            self.candidate_record(
                record, boundary, stage=2, completed_intervals=1
            )
            self.run_resume_audit(
                checkpoint=boundary,
                candidate=record,
                stage=2,
                stage1_reference=stage1,
                expect_success=False,
            )


class PipelineResumeBindingTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Namespace, Path]:
        pipeline_tag = "formal_pipeline"
        stage1_tag = f"{pipeline_tag}_stage1"
        stage2_tag = f"{pipeline_tag}_stage2"
        candidate_dir = root / "training" / stage1_tag / "candidates"
        checkpoints = root / "validated_checkpoints"
        candidate_dir.mkdir(parents=True)
        checkpoints.mkdir(parents=True)
        v7_best = root / "v7_best.ckpt"
        v7_best.write_bytes(b"formal-v7")
        actual_v7_sha = hashlib.sha256(v7_best.read_bytes()).hexdigest()
        recovery = checkpoints / "last_step0512.ckpt"
        boundary0 = checkpoints / "step0000.ckpt"
        write_checkpoint(
            boundary0,
            stage=1,
            global_step=0,
            v7_sha=actual_v7_sha,
        )
        write_checkpoint(
            recovery,
            stage=1,
            global_step=512,
            epoch=0,
            v7_sha=actual_v7_sha,
        )
        for coordinate, checkpoint in ((0, boundary0), (512, recovery)):
            summary_path = candidate_dir / f"validation_step_{coordinate:04d}.json"
            summary_path.write_text(
                json.dumps(summary(527 + coordinate // 512, coordinate)),
                encoding="utf-8",
            )
            registry.register(
                registry_namespace(
                    summary=summary_path,
                    checkpoint=checkpoint,
                    output=(
                        candidate_dir / f"candidate_step_{coordinate:04d}.json"
                    ),
                    validation_schema="trace_vb_v8_validation_behavior_v1",
                    questions=747,
                    world_size=4,
                    phase="stage1",
                    step=coordinate,
                    phase_input=False,
                    rollout_batches=None,
                    v7_source_sha256=actual_v7_sha,
                )
            )
        metric_baseline = root / "metric_safe_baseline.json"
        capability_validation = root / "registered_capability_validation.json"
        metric_baseline.write_bytes(Path(METRIC_BASELINE_PATH).read_bytes())
        capability_validation.write_bytes(
            Path(CAPABILITY_VALIDATION_PATH).read_bytes()
        )
        manifest = root / "manifest.txt"
        manifest.write_text(
            "\n".join(
                (
                    "model=TRACE-VB-v8",
                    "pipeline=train_and_validation_only",
                    f"pipeline_tag={pipeline_tag}",
                    f"v7_best_checkpoint={v7_best}",
                    "v7_best_checkpoint_sha256="
                    + actual_v7_sha,
                    f"registered_capability_sha256={REGISTERED_SHA}",
                    "registered_capability_payload_tensors=517",
                    f"registered_capability_payload_sha256={PAYLOAD_SHA}",
                    f"cot_encoder_checkpoint_sha256={COT_SHA}",
                    f"stage1_tag={stage1_tag}",
                    f"stage2_tag={stage2_tag}",
                    "physical_gpus=4,5,6,7",
                    "formal_gpus=4,5,6,7",
                    "fixed_gpus=4,5,6,7",
                    "metric_safe_baseline_source=student_commit",
                    f"metric_safe_baseline_origin={METRIC_BASELINE_PATH}",
                    f"metric_safe_baseline_artifact={metric_baseline}",
                    f"metric_safe_baseline_sha256={METRIC_BASELINE_SHA}",
                    "metric_safe_baseline_correct=527",
                    "metric_safe_baseline_questions=747",
                    "registered_capability_validation_source=capability_teacher_all_roles",
                    "registered_capability_validation_origin="
                    + CAPABILITY_VALIDATION_PATH,
                    "registered_capability_validation_artifact="
                    + str(capability_validation),
                    "registered_capability_validation_sha256="
                    + CAPABILITY_VALIDATION_SHA,
                    "registered_capability_validation_correct=540",
                    "registered_capability_validation_questions=747",
                    "train_seed=0",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        args = Namespace(
            manifest=manifest,
            pipeline_tag=pipeline_tag,
            stage1_tag=stage1_tag,
            stage2_tag=stage2_tag,
            train_seed=0,
            physical_gpus="4,5,6,7",
            v7_best_checkpoint=v7_best,
            recovery_checkpoint=recovery,
            phase="stage1",
            candidate_dir=candidate_dir,
            output=root / "binding.json",
        )
        return args, recovery

    def test_exact_pipeline_recovery_binding_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.make_fixture(Path(temporary))
            report = pipeline_resume_binding.audit(args)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["pipeline_tag"], "formal_pipeline")
            self.assertEqual(report["physical_gpus"], "4,5,6,7")

    def test_candidate_gpu_set_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.make_fixture(Path(temporary))
            latest = args.candidate_dir / "candidate_step_0512.json"
            record = json.loads(latest.read_text(encoding="utf-8"))
            record["physical_gpus"] = "2,3,4,5"
            latest.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(SystemExit):
                pipeline_resume_binding.audit(args)

    def test_earlier_prefix_candidate_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.make_fixture(Path(temporary))
            earlier = args.candidate_dir / "candidate_step_0000.json"
            record = json.loads(earlier.read_text(encoding="utf-8"))
            record["physical_gpus"] = "2,3,4,5"
            earlier.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(SystemExit):
                pipeline_resume_binding.audit(args)

    def test_pipeline_metric_copy_cannot_redirect_to_canonical_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.make_fixture(Path(temporary))
            local_copy = args.manifest.parent / "metric_safe_baseline.json"
            manifest = args.manifest.read_text(encoding="utf-8").replace(
                f"metric_safe_baseline_artifact={local_copy}",
                f"metric_safe_baseline_artifact={METRIC_BASELINE_PATH}",
            )
            args.manifest.write_text(manifest, encoding="utf-8")
            local_copy.unlink()
            with self.assertRaises(SystemExit):
                pipeline_resume_binding.audit(args)

    def test_pipeline_metric_copy_cannot_be_symlink_to_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.make_fixture(Path(temporary))
            local_copy = args.manifest.parent / "metric_safe_baseline.json"
            local_copy.unlink()
            local_copy.symlink_to(METRIC_BASELINE_PATH)
            with self.assertRaises(SystemExit):
                pipeline_resume_binding.audit(args)

    def test_cross_run_recovery_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _ = self.make_fixture(root)
            wrong_checkpoint = root / "midpoint.ckpt"
            wrong_checkpoint.write_bytes(b"unvalidated-midpoint")
            args.recovery_checkpoint = wrong_checkpoint
            with self.assertRaises(SystemExit):
                pipeline_resume_binding.audit(args)
            args.recovery_checkpoint = self.make_fixture(root / "second")[1]
            with self.assertRaises(SystemExit):
                pipeline_resume_binding.audit(args)

    def test_noncontiguous_candidate_history_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, recovery = self.make_fixture(root)
            (args.candidate_dir / "candidate_step_0512.json").unlink()
            gap_checkpoint = root / "validated_checkpoints" / "step1024.ckpt"
            gap_checkpoint.write_bytes(b"gap")
            (args.candidate_dir / "candidate_step_1024.json").write_text(
                json.dumps(
                    {
                        "schema_version": "trace_vb_v8_candidate_v1",
                        "phase": "stage1",
                        "physical_gpus": "4,5,6,7",
                        "step": 1024,
                        "selection_step": 1024,
                        "source_global_step": 1024,
                        "phase_input": False,
                        "checkpoint": str(gap_checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            gap_checkpoint.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            args.recovery_checkpoint = recovery
            with self.assertRaises(SystemExit):
                pipeline_resume_binding.audit(args)


class SupervisorAttemptContractTest(unittest.TestCase):
    def test_initial_attempt_atomically_records_both_prelaunch_log_floors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = root / "supervisor"
            stage1_log = root / "training" / "pipeline_stage1" / "train.log"
            stage2_log = root / "training" / "pipeline_stage2" / "train.log"
            stage1_log.parent.mkdir(parents=True)
            stage1_log.write_bytes(b"old bytes must be outside this attempt\n")
            active = supervisor_attempt.start(
                Namespace(
                    supervisor_dir=supervisor,
                    pipeline_tag="pipeline",
                    attempt_id="initial_20260820",
                    attempt_kind="initial",
                    physical_gpus="4,5,6,7",
                    formal_gpus="4,5,6,7",
                    fixed_gpus="4,5,6,7",
                    tmux_session="trace_v8_initial",
                    stage1_train_log=stage1_log,
                    stage2_train_log=stage2_log,
                    resume_phase=None,
                    validated_boundary_contract=None,
                )
            )
            self.assertEqual(
                active["log_byte_offset_floors"],
                {
                    str(stage1_log.resolve()): stage1_log.stat().st_size,
                    str(stage2_log.resolve()): 0,
                },
            )

    def test_failed_status_is_preserved_and_both_resume_phases_can_succeed(self) -> None:
        for phase in ("stage1", "stage2"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                supervisor = root / "supervisor"
                supervisor.mkdir()
                tag = f"pipeline_{phase}"
                (supervisor / f"{tag}.exit_status").write_text(
                    "2\n", encoding="ascii"
                )
                checkpoint = root / f"{phase}_boundary.ckpt"
                checkpoint.write_bytes(b"validated")
                stage1_log = root / "training" / f"{tag}_stage1" / "train.log"
                stage2_log = root / "training" / f"{tag}_stage2" / "train.log"
                stage1_log.parent.mkdir(parents=True)
                stage2_log.parent.mkdir(parents=True)
                stage1_log.write_bytes(b"old stage1 failure\n")
                stage2_log.write_bytes(b"old stage2 failure\nmore\n")
                binding = root / f"resume_binding_{phase}_attempt.json"
                binding.write_text(
                    json.dumps(
                        {
                            "schema_version": (
                                "trace_vb_v8_pipeline_resume_binding_v1"
                            ),
                            "status": "PASS",
                            "pipeline_tag": tag,
                            "phase": phase,
                            "physical_gpus": "4,5,6,7",
                            "validated_boundary_checkpoint": str(checkpoint),
                        }
                    ),
                    encoding="utf-8",
                )
                attempt_id = f"resume_{phase}_20260820"
                start_args = Namespace(
                    supervisor_dir=supervisor,
                    pipeline_tag=tag,
                    attempt_id=attempt_id,
                    attempt_kind=f"resume_{phase}",
                    physical_gpus="4,5,6,7",
                    formal_gpus="4,5,6,7",
                    fixed_gpus="4,5,6,7",
                    tmux_session=f"trace_v8_{phase}",
                    stage1_train_log=stage1_log,
                    stage2_train_log=stage2_log,
                    resume_phase=phase,
                    validated_boundary_contract=binding,
                )
                active = supervisor_attempt.start(start_args)
                self.assertEqual(active["status"], "WAITING")
                self.assertEqual(active["resume_phase"], phase)
                self.assertEqual(
                    active["validated_boundary_contract"], str(binding.resolve())
                )
                self.assertEqual(
                    active["log_byte_offset_floors"],
                    {
                        str(stage1_log.resolve()): stage1_log.stat().st_size,
                        str(stage2_log.resolve()): stage2_log.stat().st_size,
                    },
                )
                legacy = list((supervisor / f"{tag}.attempts").glob("legacy_*.json"))
                self.assertEqual(len(legacy), 1)
                self.assertEqual(
                    json.loads(legacy[0].read_text(encoding="utf-8"))[
                        "exit_status"
                    ],
                    2,
                )
                running = supervisor_attempt.transition(
                    Namespace(
                        supervisor_dir=supervisor,
                        pipeline_tag=tag,
                        attempt_id=attempt_id,
                        status="RUNNING",
                        exit_status=None,
                    )
                )
                self.assertEqual(running["status"], "RUNNING")
                succeeded = supervisor_attempt.transition(
                    Namespace(
                        supervisor_dir=supervisor,
                        pipeline_tag=tag,
                        attempt_id=attempt_id,
                        status="SUCCEEDED",
                        exit_status=None,
                    )
                )
                self.assertEqual(succeeded["status"], "SUCCEEDED")
                self.assertEqual(succeeded["exit_status"], 0)
                self.assertEqual(
                    Path(succeeded["exit_status_file"])
                    .read_text(encoding="ascii")
                    .strip(),
                    "0",
                )
                self.assertEqual(
                    (supervisor / f"{tag}.exit_status")
                    .read_text(encoding="ascii")
                    .strip(),
                    "0",
                )

    def test_resume_attempt_rejects_unvalidated_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "boundary.ckpt"
            checkpoint.write_bytes(b"boundary")
            binding = root / "binding.json"
            binding.write_text(
                json.dumps(
                    {
                        "schema_version": "trace_vb_v8_pipeline_resume_binding_v1",
                        "status": "PASS",
                        "pipeline_tag": "wrong_pipeline",
                        "phase": "stage1",
                        "physical_gpus": "4,5,6,7",
                        "validated_boundary_checkpoint": str(checkpoint),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "does not bind"):
                supervisor_attempt.start(
                    Namespace(
                        supervisor_dir=root / "supervisor",
                        pipeline_tag="pipeline",
                        attempt_id="resume_stage1_attempt",
                        attempt_kind="resume_stage1",
                        physical_gpus="4,5,6,7",
                        formal_gpus="4,5,6,7",
                        fixed_gpus="4,5,6,7",
                        tmux_session="trace_v8_resume",
                        stage1_train_log=(
                            root / "training" / "pipeline_stage1" / "train.log"
                        ),
                        stage2_train_log=(
                            root / "training" / "pipeline_stage2" / "train.log"
                        ),
                        resume_phase="stage1",
                        validated_boundary_contract=binding,
                    )
                )


class PipelinePublicationContractTest(unittest.TestCase):
    def make_fixture(self, root: Path, phase: str) -> Namespace:
        pipeline = root / "pipeline"
        stage1_tag = "pipeline_stage1"
        stage2_tag = "pipeline_stage2"
        stage1 = root / "training" / stage1_tag
        stage2 = root / "training" / stage2_tag
        pipeline.mkdir()
        (stage1 / "candidates").mkdir(parents=True)
        (stage1 / "checkpoints").mkdir()
        (stage2 / "candidates").mkdir(parents=True)
        (stage2 / "checkpoints").mkdir()
        v7_best = root / "v7_best.ckpt"
        v7_best.write_bytes(b"formal-v7-publication")
        actual_v7_sha = hashlib.sha256(v7_best.read_bytes()).hexdigest()

        pipeline_metric = pipeline / "metric_safe_baseline.json"
        pipeline_capability = pipeline / "registered_capability_validation.json"
        pipeline_metric.write_bytes(Path(METRIC_BASELINE_PATH).read_bytes())
        pipeline_capability.write_bytes(Path(CAPABILITY_VALIDATION_PATH).read_bytes())
        stage1_metric = stage1 / "metric_safe_baseline.json"
        stage1_capability = stage1 / "registered_capability_validation.json"
        stage1_metric.write_bytes(pipeline_metric.read_bytes())
        stage1_capability.write_bytes(pipeline_capability.read_bytes())

        metric_lines = (
            "metric_safe_baseline_source=student_commit",
            f"metric_safe_baseline_origin={METRIC_BASELINE_PATH}",
            f"metric_safe_baseline_artifact={stage1_metric}",
            f"metric_safe_baseline_sha256={METRIC_BASELINE_SHA}",
            "metric_safe_baseline_correct=527",
            "metric_safe_baseline_questions=747",
            "registered_capability_validation_source=capability_teacher_all_roles",
            f"registered_capability_validation_origin={CAPABILITY_VALIDATION_PATH}",
            f"registered_capability_validation_artifact={stage1_capability}",
            f"registered_capability_validation_sha256={CAPABILITY_VALIDATION_SHA}",
            "registered_capability_validation_correct=540",
            "registered_capability_validation_questions=747",
        )
        pipeline_metric_lines = tuple(
            line.replace(str(stage1_metric), str(pipeline_metric)).replace(
                str(stage1_capability), str(pipeline_capability)
            )
            for line in metric_lines
        )

        stage1_records = []
        stage1_checkpoints = []
        for index, coordinate in enumerate((0, 512, 1024, 1536, 2048)):
            checkpoint = stage1 / "checkpoints" / f"stage1_{coordinate:04d}.ckpt"
            write_checkpoint(
                checkpoint,
                stage=1,
                global_step=coordinate,
                epoch=max(0, index - 1),
                v7_sha=actual_v7_sha,
            )
            summary_path = stage1 / "candidates" / f"validation_step_{coordinate:04d}.json"
            summary_path.write_text(
                json.dumps(summary(527 + index, coordinate)), encoding="utf-8"
            )
            record = stage1 / "candidates" / f"candidate_step_{coordinate:04d}.json"
            registry.register(
                registry_namespace(
                    summary=summary_path,
                    checkpoint=checkpoint,
                    output=record,
                    validation_schema="trace_vb_v8_validation_behavior_v1",
                    questions=747,
                    world_size=4,
                    phase="stage1",
                    step=coordinate,
                    phase_input=False,
                    rollout_batches=None,
                    v7_source_sha256=actual_v7_sha,
                )
            )
            stage1_records.append(record)
            stage1_checkpoints.append(checkpoint)
        registry.select(
            registry_namespace(
                candidates=stage1_records,
                expected_steps="0,512,1024,1536,2048",
                expected_rollout_batches=None,
                phase="stage1",
                output=stage1 / "candidate_index.json",
                best_checkpoint_record=stage1 / "best_checkpoint.txt",
                v7_source_sha256=actual_v7_sha,
            )
        )
        stage1_checkpoint = Path(
            (stage1 / "best_checkpoint.txt").read_text(encoding="utf-8").strip()
        ).resolve()
        (stage1 / "last_checkpoint.txt").write_text(
            f"{stage1_checkpoints[-1]}\n", encoding="utf-8"
        )
        (stage1 / "manifest.txt").write_text(
            "\n".join(
                (
                    "model=TRACE-VB-v8",
                    f"run_tag={stage1_tag}",
                    f"v7_best_checkpoint_sha256={actual_v7_sha}",
                    f"registered_capability_sha256={REGISTERED_SHA}",
                    "registered_capability_payload_tensors=517",
                    f"registered_capability_payload_sha256={PAYLOAD_SHA}",
                    f"cot_encoder_checkpoint_sha256={COT_SHA}",
                    *metric_lines,
                    "physical_gpus=4,5,6,7",
                    "formal_gpus=4,5,6,7",
                    "fixed_gpus=4,5,6,7",
                    f"candidate_index={stage1 / 'candidate_index.json'}",
                    f"best_checkpoint={stage1_checkpoint}",
                    f"last_checkpoint={stage1_checkpoints[-1]}",
                    "finished_at=2026-08-20T10:00:00+08:00",
                )
            )
            + "\n",
            encoding="utf-8",
        )

        stage1_index = json.loads(
            (stage1 / "candidate_index.json").read_text(encoding="utf-8")
        )
        stage1_source_step = int(stage1_index["selected_source_global_step"])
        stage2_records = []
        stage2_checkpoints = []
        for index, rollout in enumerate((0, 256, 512, 768, 1024)):
            if rollout == 0:
                checkpoint = stage1_checkpoint
                source_step = stage1_source_step
                selection_step = 0
                phase_input = True
            else:
                source_step = 100 + rollout
                selection_step = source_step
                phase_input = False
                checkpoint = stage2 / "checkpoints" / f"stage2_{rollout:04d}.ckpt"
                write_checkpoint(
                    checkpoint,
                    stage=2,
                    global_step=source_step,
                    rollout_batches=rollout,
                    epoch=index - 1,
                    v7_sha=actual_v7_sha,
                )
            summary_path = stage2 / "candidates" / f"validation_rollout_{rollout:04d}.json"
            summary_path.write_text(
                json.dumps(summary(531 + index, source_step)), encoding="utf-8"
            )
            record = stage2 / "candidates" / f"candidate_rollout_{rollout:04d}.json"
            registry.register(
                registry_namespace(
                    summary=summary_path,
                    checkpoint=checkpoint,
                    output=record,
                    validation_schema="trace_vb_v8_validation_behavior_v1",
                    questions=747,
                    world_size=4,
                    phase="stage2",
                    step=selection_step,
                    phase_input=phase_input,
                    rollout_batches=rollout,
                    stage1_reference_checkpoint=stage1_checkpoint,
                    v7_source_sha256=actual_v7_sha,
                )
            )
            stage2_records.append(record)
            stage2_checkpoints.append(checkpoint)
        registry.select(
            registry_namespace(
                candidates=stage2_records,
                expected_steps=None,
                expected_rollout_batches="0,256,512,768,1024",
                phase="stage2",
                output=stage2 / "candidate_index.json",
                best_checkpoint_record=stage2 / "best_checkpoint.txt",
                stage1_reference_checkpoint=stage1_checkpoint,
                v7_source_sha256=actual_v7_sha,
            )
        )
        final_checkpoint = Path(
            (stage2 / "best_checkpoint.txt").read_text(encoding="utf-8").strip()
        ).resolve()
        (stage2 / "last_checkpoint.txt").write_text(
            f"{stage2_checkpoints[-1]}\n", encoding="utf-8"
        )
        (stage2 / "manifest.txt").write_text(
            "\n".join(
                (
                    "model=TRACE-VB-v8",
                    f"run_tag={stage2_tag}",
                    f"stage1_checkpoint={stage1_checkpoint}",
                    f"v7_source_checkpoint_sha256={actual_v7_sha}",
                    f"registered_capability_sha256={REGISTERED_SHA}",
                    "registered_capability_payload_tensors=517",
                    f"registered_capability_payload_sha256={PAYLOAD_SHA}",
                    f"cot_encoder_checkpoint_sha256={COT_SHA}",
                    *metric_lines,
                    "physical_gpus=4,5,6,7",
                    "formal_gpus=4,5,6,7",
                    "fixed_gpus=4,5,6,7",
                    f"candidate_index={stage2 / 'candidate_index.json'}",
                    f"best_checkpoint={final_checkpoint}",
                    f"last_checkpoint={stage2_checkpoints[-1]}",
                    "finished_at=2026-08-20T11:00:00+08:00",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        (pipeline / "manifest.txt").write_text(
            "\n".join(
                (
                    "model=TRACE-VB-v8",
                    "pipeline=train_and_validation_only",
                    "pipeline_tag=pipeline",
                    f"stage1_tag={stage1_tag}",
                    f"stage2_tag={stage2_tag}",
                    f"v7_best_checkpoint={v7_best}",
                    f"v7_best_checkpoint_sha256={actual_v7_sha}",
                    f"registered_capability_sha256={REGISTERED_SHA}",
                    "registered_capability_payload_tensors=517",
                    f"registered_capability_payload_sha256={PAYLOAD_SHA}",
                    f"cot_encoder_checkpoint_sha256={COT_SHA}",
                    *pipeline_metric_lines,
                    "physical_gpus=4,5,6,7",
                    "formal_gpus=4,5,6,7",
                    "fixed_gpus=4,5,6,7",
                    "train_seed=0",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        recovery_checkpoint = (
            stage1_checkpoints[-1] if phase == "stage1" else stage2_checkpoints[-1]
        )
        binding = pipeline / f"resume_binding_{phase}_attempt.json"
        pipeline_resume_binding.audit(
            Namespace(
                manifest=pipeline / "manifest.txt",
                pipeline_tag="pipeline",
                stage1_tag=stage1_tag,
                stage2_tag=stage2_tag,
                train_seed=0,
                physical_gpus="4,5,6,7",
                v7_best_checkpoint=v7_best,
                recovery_checkpoint=recovery_checkpoint,
                phase=phase,
                candidate_dir=(stage1 if phase == "stage1" else stage2)
                / "candidates",
                output=binding,
            )
        )
        return Namespace(
            pipeline_dir=pipeline,
            stage1_dir=stage1,
            stage2_dir=stage2,
            physical_gpus="4,5,6,7",
            recovery_phase=phase,
            recovery_checkpoint=recovery_checkpoint,
            resume_attempt=f"resume_{phase}_attempt",
            resume_binding=binding,
            v7_source_sha256=actual_v7_sha,
        )

    @staticmethod
    def completed_stage_args(args: Namespace, phase: str) -> Namespace:
        stage = args.stage1_dir if phase == "stage1" else args.stage2_dir
        coordinates = (0, 512, 1024, 1536, 2048) if phase == "stage1" else (
            0,
            256,
            512,
            768,
            1024,
        )
        stem = "candidate_step" if phase == "stage1" else "candidate_rollout"
        records = [
            stage / "candidates" / f"{stem}_{coordinate:04d}.json"
            for coordinate in coordinates
        ]
        final_record = json.loads(records[-1].read_text(encoding="utf-8"))
        return registry_namespace(
            candidates=records,
            expected_steps=(
                "0,512,1024,1536,2048" if phase == "stage1" else None
            ),
            expected_rollout_batches=(
                "0,256,512,768,1024" if phase == "stage2" else None
            ),
            phase=phase,
            index=stage / "candidate_index.json",
            best_checkpoint_record=stage / "best_checkpoint.txt",
            last_checkpoint_record=stage / "last_checkpoint.txt",
            manifest=stage / "manifest.txt",
            expected_last_checkpoint=Path(final_record["checkpoint"]),
            stage1_reference_checkpoint=(
                Path(
                    (args.stage1_dir / "best_checkpoint.txt")
                    .read_text(encoding="utf-8")
                    .strip()
                )
                if phase == "stage2"
                else None
            ),
            v7_source_sha256=args.v7_source_sha256,
        )

    def test_stage1_and_stage2_recovery_publish_normal_completion_surface(self) -> None:
        for phase in ("stage1", "stage2"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                args = self.make_fixture(Path(temporary), phase)
                report = pipeline_publication.publish(args)
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["pipeline_tag"], "pipeline")
                pipeline = args.pipeline_dir
                for name in (
                    "stage1_best.txt",
                    "final_best.txt",
                    "stage1_candidate_index.json",
                    "stage2_candidate_index.json",
                    "publication.json",
                ):
                    self.assertTrue((pipeline / name).is_file(), msg=name)
                manifest = pipeline_resume_binding.read_manifest(
                    pipeline / "manifest.txt"
                )
                for key in (
                    "stage1_checkpoint",
                    "stage1_checkpoint_sha256",
                    "stage1_candidate_index",
                    "final_checkpoint",
                    "final_checkpoint_sha256",
                    "final_candidate_index",
                    "finished_at",
                ):
                    self.assertEqual(len(manifest.get(key, [])), 1, msg=key)
                self.assertEqual(manifest["recovery_phase"], [phase])
                self.assertEqual(manifest["recovery_formal_gpus"], ["4,5,6,7"])
                self.assertEqual(manifest["recovery_fixed_gpus"], ["4,5,6,7"])

    def test_incomplete_stage_and_spoofed_index_fail_before_publication(self) -> None:
        for mutation in ("missing_manifest", "spoofed_index"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                args = self.make_fixture(Path(temporary), "stage2")
                if mutation == "missing_manifest":
                    (args.stage2_dir / "manifest.txt").unlink()
                else:
                    index_path = args.stage1_dir / "candidate_index.json"
                    value = json.loads(index_path.read_text(encoding="utf-8"))
                    value["selected_correct_count"] += 1
                    index_path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    pipeline_publication.publish(args)
                self.assertFalse((args.pipeline_dir / "publication.json").exists())
                manifest = (args.pipeline_dir / "manifest.txt").read_text(
                    encoding="utf-8"
                )
                self.assertNotIn("finished_at=", manifest)

    def test_completed_stage_resume_is_idempotent_and_repairs_missing_last_pointer(self) -> None:
        for phase in ("stage1", "stage2"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                args = self.make_fixture(Path(temporary), phase)
                stage_args = self.completed_stage_args(args, phase)
                manifest_path = stage_args.manifest
                with manifest_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"resume_attempt=resume_{phase}_attempt\n")
                registry.finalize_completed(stage_args)
                stage_args.last_checkpoint_record.unlink()
                registry.finalize_completed(stage_args)
                registry.finalize_completed(stage_args)
                manifest = pipeline_resume_binding.read_manifest(manifest_path)
                for key in (
                    "candidate_index",
                    "best_checkpoint",
                    "last_checkpoint",
                    "finished_at",
                ):
                    self.assertEqual(len(manifest.get(key, [])), 1, msg=key)
                self.assertTrue(stage_args.last_checkpoint_record.is_file())
                report = pipeline_publication.publish(args)
                self.assertEqual(report["status"], "PASS")

    def test_missing_or_tampered_pipeline_metric_copy_blocks_publication(self) -> None:
        for artifact_name in (
            "metric_safe_baseline.json",
            "registered_capability_validation.json",
        ):
            for mutation in (
                "missing",
                "tampered",
                "redirect_to_origin",
                "symlink_to_origin",
            ):
                with self.subTest(
                    artifact=artifact_name, mutation=mutation
                ), tempfile.TemporaryDirectory() as temporary:
                    args = self.make_fixture(Path(temporary), "stage2")
                    artifact = args.pipeline_dir / artifact_name
                    if mutation == "missing":
                        artifact.unlink()
                    elif mutation == "redirect_to_origin":
                        origin = (
                            METRIC_BASELINE_PATH
                            if artifact_name == "metric_safe_baseline.json"
                            else CAPABILITY_VALIDATION_PATH
                        )
                        manifest_path = args.pipeline_dir / "manifest.txt"
                        manifest = manifest_path.read_text(encoding="utf-8").replace(
                            f"{artifact_name.removesuffix('.json')}_artifact={artifact}",
                            f"{artifact_name.removesuffix('.json')}_artifact={origin}",
                        )
                        manifest_path.write_text(manifest, encoding="utf-8")
                        artifact.unlink()
                    elif mutation == "symlink_to_origin":
                        origin = (
                            METRIC_BASELINE_PATH
                            if artifact_name == "metric_safe_baseline.json"
                            else CAPABILITY_VALIDATION_PATH
                        )
                        artifact.unlink()
                        artifact.symlink_to(origin)
                    else:
                        artifact.write_bytes(b"tampered")
                    with self.assertRaises(RuntimeError):
                        pipeline_publication.publish(args)
                    self.assertFalse(
                        (args.pipeline_dir / "publication.json").exists()
                    )

    def test_stage_metric_copy_cannot_be_symlink_or_hardlink_alias(self) -> None:
        for alias_kind in ("symlink", "hardlink"):
            with self.subTest(alias=alias_kind), tempfile.TemporaryDirectory() as temporary:
                args = self.make_fixture(Path(temporary), "stage2")
                artifact = args.stage1_dir / "metric_safe_baseline.json"
                artifact.unlink()
                if alias_kind == "symlink":
                    artifact.symlink_to(METRIC_BASELINE_PATH)
                else:
                    os.link(
                        args.pipeline_dir / "metric_safe_baseline.json",
                        artifact,
                    )
                with self.assertRaises(RuntimeError):
                    pipeline_publication.publish(args)
                self.assertFalse((args.pipeline_dir / "publication.json").exists())


class LauncherTextContractTest(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        shell_scripts = sorted(SCRIPT_DIR.glob("*.sh"))
        subprocess.run(["bash", "-n", *map(str, shell_scripts)], check=True)

    def test_fresh_manifest_contract_keys_are_exactly_once(self) -> None:
        scripts = (
            "run_train_only_v8.sh",
            "run_stage1_vb.sh",
            "run_stage2_vb.sh",
        )
        exact_keys = (
            "metric_safe_baseline_source",
            "metric_safe_baseline_origin",
            "metric_safe_baseline_artifact",
            "metric_safe_baseline_sha256",
            "metric_safe_baseline_correct",
            "metric_safe_baseline_questions",
            "registered_capability_validation_source",
            "registered_capability_validation_origin",
            "registered_capability_validation_artifact",
            "registered_capability_validation_sha256",
            "registered_capability_validation_correct",
            "registered_capability_validation_questions",
            "registered_capability_sha256",
            "registered_capability_payload_tensors",
            "registered_capability_payload_sha256",
            "cot_encoder_checkpoint_sha256",
            "physical_gpus",
            "formal_gpus",
            "fixed_gpus",
        )
        for name in scripts:
            source = (SCRIPT_DIR / name).read_text(encoding="utf-8")
            matches = re.findall(
                r'cat > "\$\{[^}]+\}/manifest\.txt" <<EOF\n(.*?)\nEOF',
                source,
                flags=re.DOTALL,
            )
            self.assertEqual(len(matches), 1, msg=name)
            manifest = matches[0].splitlines()
            keys = [line.split("=", 1)[0] for line in manifest if "=" in line]
            for key in exact_keys:
                self.assertEqual(keys.count(key), 1, msg=f"{name}:{key}")

    def test_train_only_and_v8_roots(self) -> None:
        active = [
            "run_stage1_vb.sh",
            "run_stage2_vb.sh",
            "run_train_only_v8.sh",
            "resume_train_only_v8.sh",
            "wait_for_four_gpus_and_run_train_only_v8.sh",
            "wait_for_four_gpus_and_resume_train_only_v8.sh",
        ]
        combined = "\n".join(
            (SCRIPT_DIR / name).read_text(encoding="utf-8") for name in active
        )
        for forbidden in (
            "--do_test",
            "trainer.test",
            "--test_ckpt_path",
            "run_evidence",
            "tools/data_contract_audit.py",
        ):
            self.assertNotIn(forbidden, combined)
        common = (SCRIPT_DIR / "trace_vb_common.sh").read_text(encoding="utf-8")
        self.assertIn("trace_vb_latent_rl_v8", common)
        self.assertIn("trace_vb_v8_runs", common)
        self.assertIn(
            "TRACE_VB_FORMAL_GPUS=${TRACE_VB_FORMAL_GPUS:-2,3,4,5}", common
        )
        self.assertIn(
            "TRACE_VB_FIXED_GPUS=${TRACE_VB_FIXED_GPUS:-${TRACE_VB_FORMAL_GPUS}}",
            common,
        )
        self.assertIn("trace_vb_require_formal_gpu_set", common)
        self.assertIn("utilization <= TRACE_VB_MAX_IDLE_UTILIZATION", common)
        self.assertIn("TRACE_VB_CHECKPOINT_SCHEMA=trace_vb_v8", common)
        self.assertIn(
            "TRACE_VB_REGISTERED_CAPABILITY_PAYLOAD_TENSORS=517", common
        )
        self.assertIn(
            "b1e9a973bbf4f2eeaa18b7d49df16c38db50cea87e998008ba18a46cdff9e049",
            common,
        )
        self.assertIn(METRIC_BASELINE_SHA, common)
        self.assertIn(CAPABILITY_VALIDATION_SHA, common)

    def test_active_data_audit_is_train_val_only(self) -> None:
        stage1 = (SCRIPT_DIR / "run_stage1_vb.sh").read_text(encoding="utf-8")
        self.assertIn("audit_train_val_only_v8.py", stage1)
        self.assertNotIn("tools/data_contract_audit.py", stage1)
        self.assertEqual(
            set(train_val_audit.REGISTERED_FILES),
            {
                "gsm8k_train_processed.jsonl",
                "gsm8k_val_processed.jsonl",
            },
        )

    def test_train_val_audit_actual_read_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = {
                "gsm8k_train_processed.jsonl": {
                    "id": 1,
                    "question": "training question",
                    "cot": "1 + 1 = 2",
                    "answer": "2",
                },
                "gsm8k_val_processed.jsonl": {
                    "id": 2,
                    "question": "validation question",
                    "cot": "2 + 1 = 3",
                    "answer": "3",
                },
            }
            registered = {}
            for name, row in rows.items():
                payload = (json.dumps(row) + "\n").encode()
                (root / name).write_bytes(payload)
                registered[name] = (1, hashlib.sha256(payload).hexdigest())
            output = root / "audit.json"
            accessed_data = set()
            original_open = Path.open

            def guarded_open(path: Path, mode: str = "r", *args, **kwargs):
                if "r" in mode:
                    if "test" in path.name.casefold():
                        raise AssertionError(f"forbidden held-out read: {path}")
                    if path.suffix == ".jsonl":
                        accessed_data.add(path.resolve())
                return original_open(path, mode, *args, **kwargs)

            argv = [
                "audit_train_val_only_v8.py",
                "--dataset-dir",
                str(root),
                "--output",
                str(output),
            ]
            with (
                patch.object(train_val_audit, "REGISTERED_FILES", registered),
                patch.object(Path, "open", guarded_open),
                patch("sys.argv", argv),
            ):
                train_val_audit.main()
            self.assertEqual(
                accessed_data,
                {root / name for name in rows},
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(set(report["files"]), set(rows))

    def test_stage1_short_schedule_and_one_time_rewind(self) -> None:
        stage1 = (SCRIPT_DIR / "run_stage1_vb.sh").read_text(encoding="utf-8")
        self.assertIn("INTERVAL_BATCHES=512", stage1)
        self.assertIn("FORMAL_INTERVALS=4", stage1)
        self.assertIn("EXPECTED_STEPS=0,512,1024,1536,2048", stage1)
        self.assertIn("model.training_kwargs.scheduler.warmup_steps=75", stage1)
        self.assertIn("--save_validation_checkpoint", stage1)
        self.assertIn("--validate_only", stage1)
        self.assertIn("--rewind_path_to_capability", stage1)
        self.assertEqual(stage1.count("\n    --rewind_path_to_capability \\"), 1)
        self.assertEqual(
            stage1.count(
                '--registered_capability_ckpt_path "${TRACE_VB_REGISTERED_CAPABILITY}"'
            ),
            1,
        )
        self.assertIn("resume_rewind_path_to_capability=false", stage1)
        self.assertIn("stage1_capability_kl_scope=full_target", stage1)
        self.assertIn("compact_target_max_equations=2", stage1)
        self.assertIn(
            '--cot_encoder_ckpt_path "${TRACE_VB_COT_ENCODER}"', stage1
        )
        self.assertNotIn(
            '--cot_encoder_ckpt_path "${TRACE_VB_SUFFICIENCY_TEACHER}"', stage1
        )
        self.assertIn("audit_rewind_non_degradation_v8.py", stage1)
        self.assertIn("audit_resume_checkpoint_v8.py", stage1)
        self.assertEqual(stage1.count('--physical-gpus "${physical_gpus}"'), 3)
        self.assertIn("formal_gpus=${TRACE_VB_FORMAL_GPUS}", stage1)
        self.assertIn('"state_norm.",', stage1)
        self.assertLess(
            stage1.index("audit_rewind_non_degradation_v8.py"),
            stage1.index('register_candidate 0 "${initial_summary}"'),
        )

    def test_stage2_short_rl_and_input_candidate(self) -> None:
        stage2 = (SCRIPT_DIR / "run_stage2_vb.sh").read_text(encoding="utf-8")
        self.assertIn("MINIMUM_STAGE1_CORRECT=527", stage2)
        self.assertNotIn("494", stage2)
        self.assertIn("ROLLOUT_BATCHES_PER_INTERVAL=256", stage2)
        self.assertIn("STAGE2_RECOVERY_CHECKPOINT_INTERVAL=64", stage2)
        self.assertIn("FORMAL_INTERVALS=4", stage2)
        self.assertIn("EXPECTED_ROLLOUT_BATCHES=0,256,512,768,1024", stage2)
        self.assertIn("TOTAL_OPTIMIZER_STEPS=2048", stage2)
        self.assertIn('summary.get("global_step", -1)', stage2)
        self.assertIn("actor_head_lr=8.0e-7", stage2)
        self.assertIn("actor_feature_lr=2.0e-7", stage2)
        self.assertIn("step_reward_weight=0.15", stage2)
        self.assertIn("stage1_policy_kl_weight=0.05", stage2)
        self.assertIn(
            'register_candidate 0 0 "${selected_summary}" "${stage1_checkpoint}" true',
            stage2,
        )
        self.assertIn("audit_resume_checkpoint_v8.py", stage2)
        self.assertEqual(stage2.count('--physical-gpus "${physical_gpus}"'), 4)
        self.assertIn("formal_gpus=${TRACE_VB_FORMAL_GPUS}", stage2)
        self.assertIn("--stage1-reference-checkpoint", stage2)
        self.assertNotIn("--registered_capability_ckpt_path", stage2)
        self.assertIn("stage2_recovery_checkpoint_interval", stage2)
        self.assertIn("stage2-recovery-rollout*-globalstep*.ckpt", stage2)
        self.assertLess(
            stage2.rindex("  register_candidate \\"),
            stage2.index("stage2-recovery-rollout*-globalstep*.ckpt"),
        )
        self.assertIn("STAGE1_REPLAY_INDEX=$(mktemp", stage2)
        self.assertIn("candidate_step_*.json", stage2)
        self.assertIn('cmp -s "${STAGE1_REPLAY_INDEX}"', stage2)
        self.assertIn("stage1_candidate_index_registry_replay=exact_byte_match", stage2)
        self.assertLess(
            stage2.index("STAGE1_REPLAY_INDEX=$(mktemp"),
            stage2.index("common_run_args=("),
        )

    def test_four_round_validated_boundary_checkpoint_modes(self) -> None:
        for name in ("run_stage1_vb.sh", "run_stage2_vb.sh"):
            launcher = (SCRIPT_DIR / name).read_text(encoding="utf-8")
            self.assertEqual(
                launcher.count(
                    "if (( interval == 1 && completed_intervals == 0 )); then"
                ),
                1,
            )
            self.assertIn(
                'checkpoint_args=(--resume_ckpt_path "${current_checkpoint}")',
                launcher,
            )
            self.assertIn(
                'checkpoint_args=(--load_ckpt_path "${current_checkpoint}")',
                launcher,
            )
            self.assertIn(
                "diagnostic_only_not_formal_resume_validated_boundary_rollback",
                launcher,
            )

        def pending_modes(completed_intervals: int) -> list[str]:
            return [
                (
                    "load_ckpt_path"
                    if interval == 1 and completed_intervals == 0
                    else "resume_ckpt_path"
                )
                for interval in range(1, 5)
                if interval > completed_intervals
            ]

        self.assertEqual(
            pending_modes(0),
            [
                "load_ckpt_path",
                "resume_ckpt_path",
                "resume_ckpt_path",
                "resume_ckpt_path",
            ],
        )
        self.assertEqual(
            pending_modes(1),
            ["resume_ckpt_path", "resume_ckpt_path", "resume_ckpt_path"],
        )
        self.assertEqual(
            pending_modes(2), ["resume_ckpt_path", "resume_ckpt_path"]
        )
        self.assertEqual(pending_modes(3), ["resume_ckpt_path"])
        self.assertEqual(pending_modes(4), [])

    def test_trigger_precedes_pipeline_creation_and_waiter_resolves_best(self) -> None:
        pipeline = (SCRIPT_DIR / "run_train_only_v8.sh").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            pipeline.index("trace_vb_audit_v7_trigger"),
            pipeline.index('mkdir -p "${PIPELINE_DIR}"'),
        )
        waiter = (
            SCRIPT_DIR / "wait_for_four_gpus_and_run_train_only_v8.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("V7_BEST_CKPT must be injected", waiter)
        self.assertIn("trace_vb_audit_v7_trigger", waiter)
        self.assertIn('"${V7_STAGE1_DIR}" "" "${trigger_report}"', waiter)
        self.assertIn("terminal trigger_false", waiter)
        self.assertIn('export V7_BEST_CKPT="${resolved_v7_best}"', waiter)
        self.assertIn(
            'TRACE_VB_FORMAL_GPUS="${TRACE_VB_FORMAL_GPUS}"', waiter
        )
        self.assertIn(
            'TRACE_VB_FIXED_GPUS="${TRACE_VB_FIXED_GPUS}"', waiter
        )

    def test_v7_activity_gate_uses_exact_detector_not_command_substrings(self) -> None:
        common = (SCRIPT_DIR / "trace_vb_common.sh").read_text(encoding="utf-8")
        waiter = (
            SCRIPT_DIR / "wait_for_four_gpus_and_run_train_only_v8.sh"
        ).read_text(encoding="utf-8")
        combined = common + "\n" + waiter
        self.assertNotIn("pgrep -af", combined)
        self.assertNotIn("tmux has-session", combined)
        self.assertEqual(combined.count("detect_active_v7_training_v8.py"), 1)
        self.assertIn("trace_vb_v7_training_is_active", common)
        self.assertIn("trace_vb_v7_training_is_active", waiter)
        self.assertIn("detector_status == 3", common)
        self.assertIn("trace_vb_require_no_v7_training", common)

    def test_resume_wrapper_binds_pipeline_before_phase_launch(self) -> None:
        resume = (SCRIPT_DIR / "resume_train_only_v8.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("audit_pipeline_resume_binding_v8.py", resume)
        self.assertIn('--recovery-checkpoint "${recovery_checkpoint}"', resume)
        self.assertIn('--candidate-dir "${resume_candidate_dir}"', resume)
        self.assertIn("publish_pipeline_completion_v8.py", resume)
        self.assertIn('--physical-gpus "${physical_gpus}"', resume)
        self.assertIn('--resume-attempt "${resume_binding_attempt}"', resume)
        self.assertIn('TRACE_VB_RESUME_ATTEMPT="${resume_binding_attempt}"', resume)
        self.assertNotIn('--log-parent "${LOG_PARENT}"', resume)
        self.assertLess(
            resume.index("audit_pipeline_resume_binding_v8.py"),
            resume.index('case "${resume_phase}" in'),
        )
        waiter = (
            SCRIPT_DIR / "wait_for_four_gpus_and_resume_train_only_v8.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("supervisor_attempt_v8.py", waiter)
        self.assertIn("--validated-boundary-contract", waiter)
        self.assertIn('TRACE_VB_RESUME_ATTEMPT="${attempt_id}"', waiter)
        binding_audit = waiter.index("audit_pipeline_resume_binding_v8.py")
        attempt_start = waiter.index("supervisor_attempt_v8.py\" start")
        attempt_running = waiter.index("--status RUNNING")
        phase_launch = waiter.index('bash "${SCRIPT_DIR}/resume_train_only_v8.sh"')
        self.assertLess(binding_audit, attempt_start)
        self.assertLess(attempt_start, attempt_running)
        self.assertLess(attempt_running, phase_launch)

        useful_controller = (
            SCRIPT_DIR / "use_three_gpus_until_four_v8.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("TRACE_VB_RESUME_PHASE", useful_controller)
        self.assertIn("TRACE_VB_RECOVERY_CHECKPOINT", useful_controller)
        self.assertIn(
            'IFS=\',\' read -r -a selected <<< "${TRACE_VB_FORMAL_GPUS}"',
            useful_controller,
        )
        self.assertIn(
            'wait_for_four_gpus_and_resume_train_only_v8.sh',
            useful_controller,
        )
        useful_supervisor = (
            SCRIPT_DIR / "supervise_useful_gpus_until_four_v8.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("use_three_gpus_until_four_v8.sh", useful_supervisor)
        self.assertIn("formal_gpu_handoff.txt", useful_supervisor)
        self.assertIn("resource controller exited before handoff", useful_supervisor)

    def test_stage2_requested_global_batch_budget(self) -> None:
        requested_global_batch = 4
        world_size = 4
        local_batch = requested_global_batch // world_size
        requested_questions = 1024
        expected_synchronized_batches = math.ceil(
            requested_questions / (local_batch * world_size)
        )
        self.assertEqual(local_batch, 1)
        self.assertEqual(expected_synchronized_batches, 256)
        stage2 = (SCRIPT_DIR / "run_stage2_vb.sh").read_text(encoding="utf-8")
        self.assertIn("batch_size=4", stage2)
        self.assertIn("QUESTIONS_PER_INTERVAL=1024", stage2)
        self.assertIn("ROLLOUT_BATCHES_PER_INTERVAL=256", stage2)

    def test_stage_completion_publication_is_single_idempotent_registry_call(self) -> None:
        for name in ("run_stage1_vb.sh", "run_stage2_vb.sh"):
            source = (SCRIPT_DIR / name).read_text(encoding="utf-8")
            self.assertEqual(source.count("finalize-completed"), 1, msg=name)
            self.assertNotIn("finished_at=", source, msg=name)

    def test_processed_ddp_batch_contract_is_local_one(self) -> None:
        code_root = SCRIPT_DIR.parent
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        run_spec = importlib.util.spec_from_file_location(
            "trace_vb_run_budget_contract", code_root / "run.py"
        )
        assert run_spec is not None and run_spec.loader is not None
        run_module = importlib.util.module_from_spec(run_spec)
        run_spec.loader.exec_module(run_module)
        argv = [
            "run.py",
            "--model",
            "trace_vb_policy_qwen3_instruct",
            "--dataset",
            "gsm8k_aug_nl",
            "--trainer",
            "default",
            "--devices",
            "0,1,2,3",
            "--no_log",
            "batch_size=4",
            "trainer.limit_train_batches=256",
            "model.model_kwargs.trace_rl_config.n_train_samples_per_epoch=1024",
        ]
        previous_cwd = Path.cwd()
        try:
            os.chdir(code_root)
            with patch("sys.argv", argv):
                _, config = run_module.get_processed_args_and_config()
        finally:
            os.chdir(previous_cwd)
        local_batch = int(config.dataloader.batch_size)
        world_size = len(config.trainer.devices)
        self.assertEqual(local_batch, 1)
        self.assertEqual(world_size, 4)
        self.assertEqual(math.ceil(1024 / (local_batch * world_size)), 256)
        self.assertEqual(math.ceil(6726 / world_size / local_batch), 1682)
        self.assertGreaterEqual(1682, 512)


if __name__ == "__main__":
    unittest.main()
