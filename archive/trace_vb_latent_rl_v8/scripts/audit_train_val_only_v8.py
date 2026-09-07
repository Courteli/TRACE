#!/usr/bin/env python3
"""Audit only the immutable TRACE-VB-v8 training and validation sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REGISTERED_FILES = {
    "gsm8k_train_processed.jsonl": (
        6726,
        "31e256348cb35ef34bb63c66339a2b8483be44896547c2644d31c0d92e56540c",
    ),
    "gsm8k_val_processed.jsonl": (
        747,
        "c9ef2ef23b44ea661577e5eb02456738b02a133d70a8b29e57adf86342da0b4f",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_question(value: object) -> str:
    return re.sub(r"\W+", " ", str(value).casefold()).strip()


def audit_file(path: Path, expected_count: int, expected_sha256: str) -> tuple[dict, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not JSON") from error
            missing = {"id", "question", "cot", "answer"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} lacks {sorted(missing)}")
            rows.append(row)
    actual_sha256 = sha256(path)
    if len(rows) != expected_count:
        raise ValueError(f"{path} has {len(rows)} rows, expected {expected_count}")
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path} SHA256 {actual_sha256}, expected {expected_sha256}"
        )
    source_ids = [int(row["id"]) for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{path} contains duplicate source ids")
    questions = [normalized_question(row["question"]) for row in rows]
    if len(questions) != len(set(questions)):
        raise ValueError(f"{path} contains duplicate normalized questions")
    return (
        {
            "path": str(path),
            "count": len(rows),
            "sha256": actual_sha256,
            "unique_source_ids": len(set(source_ids)),
            "unique_normalized_questions": len(set(questions)),
        },
        set(questions),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    audited = {
        name: audit_file(dataset_dir / name, count, digest)
        for name, (count, digest) in REGISTERED_FILES.items()
    }
    train_questions = audited["gsm8k_train_processed.jsonl"][1]
    validation_questions = audited["gsm8k_val_processed.jsonl"][1]
    overlap = train_questions & validation_questions
    if overlap:
        raise ValueError(
            f"registered training/validation questions overlap: {len(overlap)}"
        )
    report = {
        "schema_version": "trace_vb_v8_train_val_data_contract_v1",
        "status": "PASS",
        "policy": "read_exactly_registered_training_and_validation_sources",
        "training_validation_overlap": 0,
        "files": {name: value[0] for name, value in audited.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
