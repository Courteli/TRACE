#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    rows = []
    with args.source_file.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < args.count:
        raise ValueError(
            f"{args.source_file} has {len(rows)} rows, fewer than {args.count}"
        )
    selected = rows[: args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "gsm8k_test_processed.jsonl"
    with output_file.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "purpose": "unlabeled_global_pca_fit_only",
        "source": str(args.source_file),
        "source_sha256": sha256(args.source_file),
        "output": str(output_file),
        "output_sha256": sha256(output_file),
        "selection": f"first_{args.count}_training_questions",
        "count": len(selected),
        "outcome_labels_used_for_selection": False,
        "reported_as_task_evaluation": False,
    }
    (args.output_dir / "projection_fit_audit.json").write_text(
        json.dumps(audit, indent=2)
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
