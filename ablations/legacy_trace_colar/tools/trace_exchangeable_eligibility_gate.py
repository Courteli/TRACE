#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--label", default="PILOT")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min_questions", type=int, default=100)
    parser.add_argument("--min_eligible_wilson_lower", type=float, default=0.15)
    parser.add_argument("--min_path_diversity", type=float, default=0.01)
    parser.add_argument("--max_teacher_relation_mae", type=float, default=0.10)
    parser.add_argument("--max_identity_excess", type=float, default=0.075)
    parser.add_argument("--max_outcome_position_std", type=float, default=0.10)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    method = summary["methods"][args.label]
    aggregate = method["aggregate"]
    probe = method["seed_index_probe"]
    total = int(aggregate["question_count"])
    eligible = int(aggregate["ranking_eligible_count"])
    eligible_fraction = float(aggregate["ranking_eligible_fraction"] or 0.0)
    eligible_lower = _wilson_lower(eligible, total)
    diversity = aggregate["model_path_diversity"]["mean"]
    relation_mae = aggregate["teacher_relation_mae"]["mean"]
    identity_accuracy = probe["crossfit_accuracy"]
    identity_chance = probe["chance_accuracy"]
    identity_excess = (
        None
        if identity_accuracy is None or identity_chance is None
        else float(identity_accuracy - identity_chance)
    )
    outcome_position_std = probe["outcome_rate_std_by_position"]

    checks = {
        "enough_questions": total >= args.min_questions,
        "ranking_signal_available": (
            eligible_lower >= args.min_eligible_wilson_lower
        ),
        "paths_do_not_collapse": (
            diversity is not None and diversity >= args.min_path_diversity
        ),
        "teacher_relation_is_learnable_scale": (
            relation_mae is not None
            and relation_mae <= args.max_teacher_relation_mae
        ),
        "no_cross_question_seed_position_identity": (
            identity_excess is not None
            and identity_excess <= args.max_identity_excess
        ),
        "outcomes_not_tied_to_seed_position": (
            outcome_position_std is not None
            and outcome_position_std <= args.max_outcome_position_std
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed": {
            "questions": total,
            "ranking_eligible_count": eligible,
            "ranking_eligible_fraction": eligible_fraction,
            "ranking_eligible_wilson95_lower": eligible_lower,
            "model_path_diversity": diversity,
            "teacher_relation_mae": relation_mae,
            "seed_position_probe_accuracy": identity_accuracy,
            "seed_position_probe_chance": identity_chance,
            "seed_position_probe_excess": identity_excess,
            "outcome_rate_std_by_seed_position": outcome_position_std,
        },
        "thresholds": {
            "min_questions": args.min_questions,
            "min_eligible_wilson_lower": args.min_eligible_wilson_lower,
            "min_path_diversity": args.min_path_diversity,
            "max_teacher_relation_mae": args.max_teacher_relation_mae,
            "max_identity_excess": args.max_identity_excess,
            "max_outcome_position_std": args.max_outcome_position_std,
        },
        "failure_action": (
            "Do not start Stage 2 and do not resample questions. Revisit the "
            "exchangeable perturbation before any outcome-ranking training."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
