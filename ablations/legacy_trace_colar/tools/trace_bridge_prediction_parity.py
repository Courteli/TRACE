#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def first(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def load_records(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        value["question"]: {"idx": int(key), **value}
        for key, value in payload.items()
        if isinstance(value, dict) and "question" in value
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--reference_name", default="native model")
    parser.add_argument("--candidate_name", default="instrumented recorder")
    parser.add_argument("--require_exact", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_records(args.reference)
    candidate = load_records(args.candidate)
    common = sorted(set(reference) & set(candidate))
    fields = ("pred_answer", "output_string", "output_length", "n_latent_forward", "acc")
    mismatch_counts = {field: 0 for field in fields}
    examples = []
    for question in common:
        ref = reference[question]
        cand = candidate[question]
        mismatches = {}
        for field in fields:
            ref_value = first(ref.get(field))
            cand_value = first(cand.get(field))
            if ref_value != cand_value:
                mismatch_counts[field] += 1
                mismatches[field] = {"reference": ref_value, "candidate": cand_value}
        if mismatches and len(examples) < 20:
            examples.append(
                {
                    "idx": ref["idx"],
                    "question": question,
                    "mismatches": mismatches,
                }
            )
    payload = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_name": args.reference_name,
        "candidate_name": args.candidate_name,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "common_count": len(common),
        "same_question_set": set(reference) == set(candidate),
        "mismatch_counts": mismatch_counts,
        "exact_prediction_parity": (
            set(reference) == set(candidate)
            and all(count == 0 for count in mismatch_counts.values())
        ),
        "mismatch_examples": examples,
    }
    (args.out_dir / "prediction_parity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Prediction Parity Audit",
        "",
        f"Reference: **{args.reference_name}**  ",
        f"Candidate: **{args.candidate_name}**",
        "",
        f"- Common questions: **{len(common)}**.",
        f"- Same question set: **{payload['same_question_set']}**.",
        f"- Exact prediction parity: **{payload['exact_prediction_parity']}**.",
    ]
    for field, count in mismatch_counts.items():
        lines.append(f"- `{field}` mismatches: **{count}**.")
    (args.out_dir / "prediction_parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.require_exact and not payload["exact_prediction_parity"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
