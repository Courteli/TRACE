#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    source = args.source_dir / "train.json"
    rows = json.loads(source.read_text())
    if len(rows) < args.count:
        raise ValueError(
            f"{source} has {len(rows)} rows, fewer than {args.count}"
        )
    selected = rows[: args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test.json").write_text(
        json.dumps(selected, ensure_ascii=False)
    )
    audit = {
        "purpose": "unlabeled_global_pca_fit_only",
        "source": str(source),
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
