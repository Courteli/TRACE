#!/usr/bin/env python3
"""Run one strict v7 validation checkpoint without touching the test split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--validation-path",
        choices=("student_commit", "capability_teacher_all_roles"),
        required=True,
    )
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_v7_run(v7_root: Path):
    sys.path.insert(0, str(v7_root))
    os.chdir(v7_root)
    spec = importlib.util.spec_from_file_location("trace_vb_v7_run", v7_root / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the v7 run module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    v7_root = args.v7_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not (v7_root / "run.py").is_file():
        raise SystemExit(f"missing v7 run.py under {v7_root}")
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    if not args.run_tag or any(
        not (character.isalnum() or character in "._-")
        for character in args.run_tag
    ):
        raise SystemExit(f"unsafe run tag: {args.run_tag}")
    log_root = os.environ.get("TRACE_LOG_ROOT")
    if not log_root:
        raise SystemExit("TRACE_LOG_ROOT is required")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("the validation worker requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    # Establish ownership while the large CPU checkpoint is being decoded.
    # The live validation process keeps this small guard; it is not a separate
    # placeholder workload and is released automatically with the process.
    startup_guard_mib = int(os.environ.get("TRACE_VB_STARTUP_GUARD_MIB", "64"))
    if not 64 <= startup_guard_mib <= 8192:
        raise SystemExit(
            "TRACE_VB_STARTUP_GUARD_MIB must be between 64 and 8192"
        )
    startup_guard = torch.empty(
        startup_guard_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda:0",
    )

    run = load_v7_run(v7_root)
    original_argv = sys.argv
    sys.argv = [
        str(v7_root / "run.py"),
        "--model", "trace_vb_policy_qwen3_instruct",
        "--dataset", "gsm8k_aug_nl",
        "--trainer", "default",
        "--devices", "0",
        "--workspace_path", str(v7_root),
        "--load_ckpt_path", str(checkpoint),
        "--test_times", "1",
        "--seed", str(args.seed),
        "--disable_early_stopping",
        "--log_suffix", args.run_tag,
        "data_module.dataset_dir=/disk1/dingxukai/TRACE/data/raw/GSM8k-Aug-NL",
        "data_module.enforce_registered_source=true",
        "data_module.tiny_dataset=false",
        "data_module.epoch_scaling=1",
        "batch_size=4",
        "val_batch_size=1",
        "num_workers=0",
        "pin_memory=false",
        "persistent_workers=false",
        "trainer.limit_val_batches=1.0",
        "trainer.num_sanity_val_steps=0",
        "save_top_k=0",
        "save_last=false",
        f"model.model_kwargs.trace_policy_config.validation_path={args.validation_path}",
        "model.model_kwargs.trace_policy_config.stage1_recovery_checkpoint_interval=0",
        "model.model_kwargs.trace_policy_config.stage1_host_memory_guard_interval=0",
        "model.model_kwargs.trace_policy_config.visual_record_limit=0",
    ]
    try:
        parsed, config = run.get_processed_args_and_config()
    finally:
        sys.argv = original_argv

    run.seed_current_process(parsed.seed)
    run.install_rank_local_ddp_seed_reset()
    data_module = run.instantiate_from_config(
        config.data_module, extra_kwargs={"all_config": config}
    )
    # This v7 data module predates Lightning's ``validate`` stage and only
    # constructs the validation set in ``setup('fit')``.
    data_module.setup("fit")
    model = run.instantiate_from_config(
        config.model, extra_kwargs={"all_config": config}
    )
    state_dict = run.load_full_checkpoint(model, str(checkpoint))
    incompatible = model.load_state_dict(state_dict=state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint has unexpected keys after the model's fail-closed "
            f"coverage checks: {incompatible.unexpected_keys}"
        )
    trainer = run.instantiate_from_config(
        config.trainer,
        extra_kwargs={"callbacks": run.instantiate_callbacks(config.callbacks)},
    )
    # ``trainer.validate`` does not call the training-only ``on_fit_start``
    # hook that legacy v7 uses to create these loggers. Install the same
    # logger types explicitly so the completed validation can publish cleanly.
    from src.utils.log import JsonLogger, TextLogger

    logger_dir = Path(trainer.logger.log_dir).resolve()
    logger_dir.mkdir(parents=True, exist_ok=True)
    model.text_logger = TextLogger(
        model, save_dir=logger_dir, log_file_name="validation_control"
    )
    model.json_logger = JsonLogger(
        model, save_dir=logger_dir, log_file_name="validation_control"
    )
    # Lightning may raise during teardown after the model has already
    # completed all 747 examples and atomically published its strict summary.
    # That happened on the opportunistic GPU-holding controls and caused the
    # worker to release a card despite having finished the useful evaluation.
    # Preserve fail-closed behavior unless the run-local summary below proves
    # that the complete validation contract was satisfied.
    validation_exception = None
    try:
        results = trainer.validate(model=model, datamodule=data_module)
    except Exception as error:  # noqa: BLE001 - audited below before acceptance
        results = []
        validation_exception = {
            "type": type(error).__name__,
            "message": str(error),
        }
    summaries = sorted(logger_dir.glob("validation_epoch_*.json"))
    if len(summaries) != 1:
        missing_summary_error = RuntimeError(
            f"expected one strict validation summary in {logger_dir}, found {len(summaries)}"
        )
        raise missing_summary_error
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    failures = []
    if summary.get("schema_version") != "trace_vb_v7_validation_behavior_v1":
        failures.append("schema_version")
    if summary.get("validation_path") != args.validation_path:
        failures.append("validation_path")
    if int(summary.get("unique_questions", -1)) != 747:
        failures.append("unique_questions")
    if int(summary.get("world_size", -1)) != 1:
        failures.append("world_size")
    correct = summary.get("correct_count")
    accuracy = summary.get("accuracy")
    if type(correct) is not int or not 0 <= correct <= 747:
        failures.append("correct_count")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or type(correct) is not int
        or not math.isclose(
            float(accuracy), correct / 747, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        failures.append("accuracy")
    report = {
        "status": "FAIL" if failures else "PASS",
        "scope": "strict_v7_validation_only_no_test_split",
        "checkpoint": str(checkpoint),
        "validation_path": args.validation_path,
        "seed": args.seed,
        "checkpoint_load": {
            "missing_frozen_or_stage2_keys": len(incompatible.missing_keys),
            "unexpected_keys": len(incompatible.unexpected_keys),
            "startup_guard_bytes": int(startup_guard.numel()),
        },
        "logger_dir": str(logger_dir),
        "validation_summary": str(summaries[0]),
        "results": results,
        "trainer_validate_exception_after_summary": validation_exception,
        "accepted_completed_summary_after_teardown_exception": (
            validation_exception is not None and not failures
        ),
        "summary": summary,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("strict validation contract failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
