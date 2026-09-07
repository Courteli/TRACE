#!/usr/bin/env python3
"""Audit immutable, project-local task data before any TRACE run."""

import hashlib
import json
import re
import stat
from pathlib import Path


ROOT = Path("/disk1/dingxukai/TRACE")
ORIGINAL_GSM8K_DIR = Path(
    "/home/dingxukai/RoT/data/GSM8k-Aug-NL"
)
FILES = {
    "gsm8k_train": (
        ROOT / "data/raw/GSM8k-Aug-NL/gsm8k_train_processed.jsonl",
        6726,
        "31e256348cb35ef34bb63c66339a2b8483be44896547c2644d31c0d92e56540c",
    ),
    "gsm8k_val": (
        ROOT / "data/raw/GSM8k-Aug-NL/gsm8k_val_processed.jsonl",
        747,
        "c9ef2ef23b44ea661577e5eb02456738b02a133d70a8b29e57adf86342da0b4f",
    ),
    "gsm8k_test": (
        ROOT / "data/raw/GSM8k-Aug-NL/gsm8k_test_processed.jsonl",
        1319,
        "5395be51d54d7af531883af873e130d3f280a7e5d9aaf849e7c20ed356e3847b",
    ),
    "gsmhard_test": (
        ROOT / "data/raw/GSM8K-Hard/gsmhard_test_processed.jsonl",
        1319,
        "1be1583e414efb8cf2485f219bf997de1e1e6824a83f24a74405de7183a73119",
    ),
    "svamp_test": (
        ROOT / "data/raw/SVAMP/svamp_test_processed.jsonl",
        1000,
        "0fad09568d6fc5f8c0efb9139477d0d560cf54954e353400fd1bff6cdec3f6f3",
    ),
    "multiarith_test": (
        ROOT / "data/raw/MultiArith/multiarith_test_processed.jsonl",
        180,
        "461cc3e2a3eb4609015f86a3d75572ae9c93a11f01a2f0ec7351224aa752b9b4",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(text: str) -> str:
    return re.sub(r"\W+", " ", str(text).casefold()).strip()


def audit_file(path: Path, expected_count: int, expected_hash: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is not one JSON record"
                ) from error
            missing = {"question", "cot", "answer"} - set(row)
            if missing:
                raise ValueError(
                    f"{path}:{line_number} lacks {sorted(missing)}"
                )
            rows.append(row)
    actual_hash = sha256(path)
    if len(rows) != expected_count:
        raise ValueError(
            f"{path} has {len(rows)} rows, expected {expected_count}"
        )
    if actual_hash != expected_hash:
        raise ValueError(
            f"{path} SHA256 {actual_hash}, expected {expected_hash}"
        )
    ids = [row.get("id") for row in rows if "id" in row]
    if ids and len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate ids")
    normalized_questions = [
        normalize_question(row["question"]) for row in rows
    ]
    unique_normalized_questions = set(normalized_questions)
    return (
        {
            "path": str(path),
            "count": len(rows),
            "sha256": actual_hash,
            "schema": ["id", "question", "cot", "answer"],
            "writable": bool(path.stat().st_mode & stat.S_IWUSR),
            "normalized_duplicate_count": (
                len(normalized_questions)
                - len(unique_normalized_questions)
            ),
        },
        unique_normalized_questions,
    )


def main():
    audited = {
        name: audit_file(path, count, digest)
        for name, (path, count, digest) in FILES.items()
    }
    results = {name: value[0] for name, value in audited.items()}
    questions = {name: value[1] for name, value in audited.items()}
    gsm8k_internal_duplicates = {
        name: results[name]["normalized_duplicate_count"]
        for name in ("gsm8k_train", "gsm8k_val", "gsm8k_test")
    }
    if any(gsm8k_internal_duplicates.values()):
        raise ValueError(
            "Registered GSM8K split contains duplicate normalized "
            f"questions: {gsm8k_internal_duplicates}"
        )
    formal_split_pairs = (
        ("gsm8k_train", "gsm8k_val"),
        ("gsm8k_train", "gsm8k_test"),
        ("gsm8k_val", "gsm8k_test"),
    )
    split_overlaps = {
        f"{left}__{right}": len(questions[left] & questions[right])
        for left, right in formal_split_pairs
    }
    if any(split_overlaps.values()):
        raise ValueError(
            f"Registered GSM8K splits overlap after normalization: "
            f"{split_overlaps}"
        )
    ood_overlaps_with_train = {
        name: len(questions["gsm8k_train"] & questions[name])
        for name in ("gsmhard_test", "svamp_test", "multiarith_test")
    }
    source_mirror = {}
    for name in (
        "gsm8k_train_processed.jsonl",
        "gsm8k_val_processed.jsonl",
        "gsm8k_test_processed.jsonl",
    ):
        original = ORIGINAL_GSM8K_DIR / name
        copied = ROOT / "data/raw/GSM8k-Aug-NL" / name
        if not original.is_file():
            raise FileNotFoundError(original)
        original_hash = sha256(original)
        copied_hash = sha256(copied)
        if original_hash != copied_hash:
            raise ValueError(
                f"Project copy {copied} differs from immutable source "
                f"{original}"
            )
        source_mirror[name] = {
            "original_path": str(original),
            "project_copy_path": str(copied),
            "byte_identical": True,
            "sha256": copied_hash,
        }
    report = {
        "status": "PASS",
        "project_root": str(ROOT),
        "data_policy": {
            "training_source": "gsm8k_train_processed.jsonl",
            "one_original_cot_per_question": True,
            "generated_cots": False,
            "derived_rationale_sets": False,
            "source_files_modified": False,
            "heldout_data_used_for_training": False,
        },
        "split_integrity": {
            "normalization": "casefold_nonword_collapse",
            "gsm8k_internal_duplicate_counts": (
                gsm8k_internal_duplicates
            ),
            "gsm8k_pairwise_overlap_counts": split_overlaps,
            "ood_overlap_with_gsm8k_train_counts": (
                ood_overlaps_with_train
            ),
        },
        "source_mirror": source_mirror,
        "files": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
