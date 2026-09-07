#!/usr/bin/env python3
"""Run five exact, single-device validations for a native-v9 checkpoint.

The four-device training validator necessarily pads the 747-example split to
748 examples.  This evaluator deliberately exposes exactly one CUDA device and
audits the per-question records, so a successful summary is evidence for the
un-padded 747-question protocol rather than merely a Lightning aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_VALIDATION_SHA256 = (
    "5f2ddd39f09f95d834a1b2840ce12d9fd2461bc909a755f1b8c429497bdfa49b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--expected-questions", type=int, default=747)
    parser.add_argument(
        "--expected-validation-sha256",
        default=EXPECTED_VALIDATION_SHA256,
    )
    return parser.parse_args()


def _load_run_module(model_root: Path):
    sys.path.insert(0, str(model_root))
    os.chdir(model_root)
    spec = importlib.util.spec_from_file_location(
        "trace_role_native_v9_run", model_root / "run.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import run.py from {model_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sample_logs(sample_logs, expected_questions: int) -> tuple[int, float]:
    """Return the exact correct count after proving one record per question."""
    expected_indices = set(range(expected_questions))
    observed_indices = {key for key in sample_logs if isinstance(key, int)}
    if observed_indices != expected_indices:
        missing = sorted(expected_indices - observed_indices)[:10]
        extra = sorted(observed_indices - expected_indices)[:10]
        raise RuntimeError(
            "strict validation question coverage mismatch: "
            f"observed={len(observed_indices)}, missing={missing}, extra={extra}"
        )

    correct = 0
    for index in range(expected_questions):
        accuracies = sample_logs[index].get("acc")
        if not isinstance(accuracies, list) or len(accuracies) != 1:
            raise RuntimeError(
                f"question {index} has {accuracies!r}, expected one accuracy value"
            )
        value = float(accuracies[0])
        if value not in (0.0, 1.0):
            raise RuntimeError(f"question {index} has non-binary accuracy {value}")
        correct += int(value)
    return correct, correct / expected_questions


def main() -> None:
    args = parse_args()
    if args.replications != 5:
        raise SystemExit("the formal native-v9 protocol requires exactly 5 replications")
    if args.expected_questions != 747:
        raise SystemExit("the formal native-v9 protocol requires exactly 747 questions")

    model_root = args.model_root.expanduser().resolve(strict=True)
    run_root = args.run_root.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        checkpoint_path.relative_to((run_root / "stage2").resolve(strict=True))
    except ValueError as error:
        raise SystemExit(
            f"checkpoint is not a Stage2 artifact of this fresh run: {checkpoint_path}"
        ) from error

    # Import CUDA-dependent packages only after the caller has restricted
    # CUDA_VISIBLE_DEVICES to one physical card.
    import torch
    import lightning.pytorch as pl
    from omegaconf import OmegaConf

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(
            "strict validation requires exactly one visible CUDA device; "
            f"observed {torch.cuda.device_count()}"
        )

    run = _load_run_module(model_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if run._checkpoint_model_target(checkpoint) != run.NATIVE_MODEL_TARGET:
        raise RuntimeError("checkpoint target is not LitTRACERoleNative")
    if not run._checkpoint_uses_trace_rl(checkpoint):
        raise RuntimeError("formal final validation requires a Stage2 RL checkpoint")

    hparams_path = checkpoint_path.parent.parent / "hparams.yaml"
    config = OmegaConf.load(hparams_path).all_config
    config.trainer.devices = [0]
    config.trainer.strategy = "auto"
    config.trainer.use_distributed_sampler = False
    config.dataloader.val_batch_size = 1
    config.dataloader.num_workers = 0
    config.dataloader.pin_memory = False
    config.dataloader.persistent_workers = False
    config.model.model_kwargs.trace_bridge_config.trace_visual_record_limit = 0
    config.model.model_kwargs.trace_bridge_config.save_trace_visual_info = False
    os.environ["TRACE_NATIVE_RUN_ROOT"] = str(run_root)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_module = run.instantiate_from_config(
        config.data_module, extra_kwargs={"all_config": config}
    )
    validation_path = Path(config.data_module.dataset_dir) / "val.json"
    validation_sha256 = sha256_file(validation_path)
    if validation_sha256 != args.expected_validation_sha256:
        raise RuntimeError(
            "validation data hash mismatch: "
            f"observed={validation_sha256}, "
            f"expected={args.expected_validation_sha256}"
        )
    data_module.setup("fit")
    if len(data_module.val_set) != args.expected_questions:
        raise RuntimeError(
            f"validation dataset has {len(data_module.val_set)} examples, "
            f"expected {args.expected_questions}"
        )

    model = run.instantiate_from_config(
        config.model, extra_kwargs={"all_config": config}
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint has unexpected tensors: {incompatible.unexpected_keys}"
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
    )
    from src.utils.log import JsonLogger, TextLogger

    output_dir.mkdir(parents=True, exist_ok=True)
    replications = []
    for replication in range(args.replications):
        replication_seed = args.seed + replication
        pl.seed_everything(replication_seed, workers=True)
        model.sample_logs = defaultdict(dict)
        model._trace_visual_records = []
        tag = f"replicate_{replication + 1:02d}"
        model.text_logger = TextLogger(
            model, save_dir=output_dir, log_file_name=f"{tag}_log"
        )
        model.json_logger = JsonLogger(
            model, save_dir=output_dir, log_file_name=f"{tag}_samples"
        )
        lightning_results = trainer.validate(model=model, datamodule=data_module)
        if len(lightning_results) != 1 or "val/acc" not in lightning_results[0]:
            raise RuntimeError(
                f"replication {replication + 1} returned invalid metrics: "
                f"{lightning_results}"
            )
        correct, accuracy = audit_sample_logs(
            model.sample_logs, args.expected_questions
        )
        lightning_accuracy = float(lightning_results[0]["val/acc"])
        if not math.isclose(
            lightning_accuracy, accuracy, rel_tol=0.0, abs_tol=1e-6
        ):
            raise RuntimeError(
                f"replication {replication + 1} aggregate mismatch: "
                f"lightning={lightning_accuracy}, exact={accuracy}"
            )
        record = {
            "replication": replication + 1,
            "seed": replication_seed,
            "world_size": 1,
            "unique_questions": args.expected_questions,
            "correct_count": correct,
            "accuracy": accuracy,
            "lightning_metrics": {
                key: float(value) for key, value in lightning_results[0].items()
            },
            "samples": str(output_dir / f"{tag}_samples.json"),
        }
        _atomic_json(output_dir / f"{tag}_summary.json", record)
        replications.append(record)

    accuracies = [record["accuracy"] for record in replications]
    summary = {
        "schema_version": "trace_role_native_v9_strict_validation_v1",
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "model_root": str(model_root),
        "run_root": str(run_root),
        "protocol": {
            "split": "val",
            "replications": 5,
            "world_size_per_replication": 1,
            "unique_questions_per_replication": 747,
            "distributed_padding": False,
            "validation_path": str(validation_path.resolve()),
            "validation_sha256": validation_sha256,
            "base_seed": args.seed,
        },
        "replication_results": replications,
        "accuracy_mean": sum(accuracies) / len(accuracies),
        "accuracy_min": min(accuracies),
        "accuracy_max": max(accuracies),
    }
    _atomic_json(output_dir / "strict_validation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
