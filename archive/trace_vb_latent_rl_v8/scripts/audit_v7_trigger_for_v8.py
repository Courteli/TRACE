#!/usr/bin/env python3
"""Validate the completed v7 Stage-1 artifact that conditionally triggers v8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


EXPECTED_EPOCHS = 10
EXPECTED_QUESTIONS = 747
MAX_TRIGGER_CORRECT = 522
CHECKPOINT_PATTERN = re.compile(
    r"^epoch(?P<epoch>\d+)__step(?P<step>\d+)__monitor(?P<score>[-+0-9.eE]+)\.ckpt$"
)


class TriggerFalse(RuntimeError):
    """A complete, valid v7 run does not satisfy the conditional v8 trigger."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def audit(args: argparse.Namespace) -> None:
    stage1_dir = args.stage1_dir.resolve()
    manifest_path = stage1_dir / "manifest.txt"
    best_record = stage1_dir / "best_checkpoint.txt"
    if not stage1_dir.is_dir() or not manifest_path.is_file():
        raise SystemExit("missing formal v7 Stage-1 directory or manifest")
    if not best_record.is_file():
        raise SystemExit("missing v7 best-checkpoint record")
    recorded_checkpoint = Path(
        best_record.read_text(encoding="utf-8").strip()
    ).resolve()
    requested_checkpoint = getattr(args, "checkpoint", None)
    checkpoint = (
        Path(requested_checkpoint).resolve()
        if requested_checkpoint is not None
        else recorded_checkpoint
    )
    if not checkpoint.is_file():
        raise SystemExit("missing v7 best checkpoint")
    manifest = read_manifest(manifest_path)
    if not manifest.get("finished_at"):
        raise SystemExit("v7 Stage 1 has no finished_at marker")
    if recorded_checkpoint != checkpoint:
        raise SystemExit("V7_BEST_CKPT does not resolve to best_checkpoint.txt")
    manifest_checkpoint = manifest.get("best_checkpoint")
    if manifest_checkpoint and Path(manifest_checkpoint).resolve() != checkpoint:
        raise SystemExit("v7 manifest and best_checkpoint.txt disagree")
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise SystemExit("v7 selected checkpoint is not a validation checkpoint")
    selected_epoch = int(match.group("epoch"))
    if not 0 <= selected_epoch < EXPECTED_EPOCHS:
        raise SystemExit("v7 selected checkpoint has an invalid epoch")

    index_value = manifest.get("validation_summary_index")
    if not index_value:
        raise SystemExit("v7 manifest has no validation_summary_index")
    index_path = Path(index_value).resolve()
    if not index_path.is_file():
        raise SystemExit("v7 validation summary index is missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != "trace_vb_v7_stage1_validation_index_v1":
        raise SystemExit("v7 validation index has the wrong schema")
    if int(index.get("formal_epochs", -1)) != EXPECTED_EPOCHS:
        raise SystemExit("v7 validation index does not declare ten epochs")
    run_dir = checkpoint.parent.parent.resolve()
    if Path(index.get("published_run_dir", "")).resolve() != run_dir:
        raise SystemExit("v7 validation index belongs to a different run")
    entries = index.get("summaries", [])
    if len(entries) != EXPECTED_EPOCHS:
        raise SystemExit("v7 validation index does not contain ten summaries")

    summaries = {}
    for entry in entries:
        epoch = int(entry.get("epoch_index", -1))
        if epoch in summaries or not 0 <= epoch < EXPECTED_EPOCHS:
            raise SystemExit("v7 validation index has duplicate/invalid epochs")
        expected_path = (run_dir / f"validation_epoch_{epoch:03d}.json").resolve()
        if Path(entry.get("checkpoint_run_copy", "")).resolve() != expected_path:
            raise SystemExit(f"v7 epoch {epoch} summary path is inconsistent")
        if not expected_path.is_file() or sha256(expected_path) != entry.get("sha256"):
            raise SystemExit(f"v7 epoch {epoch} summary continuity failed")
        summary = json.loads(expected_path.read_text(encoding="utf-8"))
        if summary.get("schema_version") != "trace_vb_v7_validation_behavior_v1":
            raise SystemExit(f"v7 epoch {epoch} validation schema is wrong")
        if summary.get("validation_path") != "student_commit":
            raise SystemExit(f"v7 epoch {epoch} did not use question+COMMIT")
        if int(summary.get("epoch_index", -1)) != epoch:
            raise SystemExit(f"v7 epoch {epoch} summary has the wrong index")
        if int(summary.get("world_size", -1)) != 4:
            raise SystemExit(f"v7 epoch {epoch} was not aggregated from four ranks")
        if int(summary.get("unique_questions", -1)) != EXPECTED_QUESTIONS:
            raise SystemExit(f"v7 epoch {epoch} is not the strict validation set")
        correct = int(summary.get("correct_count", -1))
        accuracy = float(summary.get("accuracy", float("nan")))
        if not 0 <= correct <= EXPECTED_QUESTIONS or not math.isclose(
            accuracy,
            correct / EXPECTED_QUESTIONS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit(f"v7 epoch {epoch} exact count is inconsistent")
        summaries[epoch] = summary
    if sorted(summaries) != list(range(EXPECTED_EPOCHS)):
        raise SystemExit("v7 validation index does not cover epochs 0..9")

    best_correct = max(int(value["correct_count"]) for value in summaries.values())
    best_epochs = [
        epoch
        for epoch, value in summaries.items()
        if int(value["correct_count"]) == best_correct
    ]
    if selected_epoch not in best_epochs:
        raise SystemExit("best_checkpoint.txt is not an exact validation best")
    selected_accuracy = float(summaries[selected_epoch]["accuracy"])
    filename_score = float(match.group("score"))
    if not math.isclose(filename_score, selected_accuracy, rel_tol=0.0, abs_tol=5.1e-7):
        raise SystemExit("v7 checkpoint filename disagrees with exact validation")
    checkpoint_sha256 = sha256(checkpoint)
    report = {
        "schema_version": "trace_vb_v8_v7_trigger_contract_v1",
        "status": (
            "PASS" if best_correct <= MAX_TRIGGER_CORRECT else "TRIGGER_FALSE"
        ),
        "stage1_dir": str(stage1_dir),
        "manifest": str(manifest_path),
        "validation_summary_index": str(index_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "selected_epoch": selected_epoch,
        "best_epochs": best_epochs,
        "best_correct_count": best_correct,
        "validation_questions": EXPECTED_QUESTIONS,
        "trigger_maximum_correct_count": MAX_TRIGGER_CORRECT,
        "formal_epochs_verified": EXPECTED_EPOCHS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if best_correct > MAX_TRIGGER_CORRECT:
        raise TriggerFalse(
            f"v8 trigger is false: v7 best is "
            f"{best_correct}/{EXPECTED_QUESTIONS}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    try:
        audit(parser.parse_args())
    except TriggerFalse as error:
        print(str(error), file=__import__("sys").stderr)
        raise SystemExit(3) from error


if __name__ == "__main__":
    main()
