#!/usr/bin/env python3
"""Fail closed unless the full paired evaluation bundle is complete."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import torch


EXPECTED_DATASETS = {
    "gsm8k": 1319,
    "gsmhard": 1319,
    "svamp": 1000,
    "multiarith": 180,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty evidence artifact: {path}")
    return path


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate(root: Path, stage1: Path, final: Path) -> dict:
    manifest = require_file(root, "manifest.txt").read_text(encoding="utf-8")
    if "finished_at=" not in manifest:
        raise SystemExit("evidence manifest has no finished_at marker")

    task_path = require_file(root, "task_summary/task_summary.json")
    for suffix in ("csv", "md", "tex"):
        require_file(root, f"task_summary/task_summary.{suffix}")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if Path(task["stage1_checkpoint"]).resolve() != stage1.resolve():
        raise SystemExit("task summary Stage-1 checkpoint mismatch")
    if Path(task["final_checkpoint"]).resolve() != final.resolve():
        raise SystemExit("task summary final checkpoint mismatch")
    if int(task.get("test_times", -1)) != 1:
        raise SystemExit("formal task summary must use exactly one replication")
    for name, expected in EXPECTED_DATASETS.items():
        actual = int(task["datasets"][name]["questions"])
        if actual != expected:
            raise SystemExit(
                f"{name} summary contains {actual} questions, expected {expected}"
            )
        if int(task["datasets"][name]["latent_length"]) != 8:
            raise SystemExit(f"{name} did not use exactly eight latent states")

    geometry_reports = {}
    for prefix in ("stage1", "final"):
        base = f"{prefix}_geometry_summary"
        require_file(root, f"{base}/global_train_fit_pca.pt")
        rows = require_file(root, f"{base}/question_geometry.csv")
        if count_csv_rows(rows) != 200:
            raise SystemExit(f"{base} does not contain exactly 200 questions")
        for relative in (
            "geometry_summary.json",
            "geometry_summary.md",
            "source_data.npz",
            "figure_contract.json",
            "figure_qa.json",
            "role_mechanism_evidence.svg",
            "role_mechanism_evidence.pdf",
            "role_mechanism_evidence.tiff",
            "policy_step_profile.svg",
            "policy_step_profile.pdf",
            "policy_step_profile.tiff",
            "diagnostic_outcome_geometry.svg",
            "diagnostic_outcome_geometry.pdf",
            "diagnostic_outcome_geometry.tiff",
        ):
            require_file(root, f"{base}/{relative}")
        geometry_reports[prefix] = json.loads(
            require_file(root, f"{base}/geometry_summary.json").read_text(
                encoding="utf-8"
            )
        )
        role_report = geometry_reports[prefix]
        if role_report.get("evidence_priority", {}).get("questions") != 200:
            raise SystemExit(f"{base} role evidence is not question-level N=200")
        if set(role_report.get("role_semantics", {})) != {
            "PLAN",
            "SOLVE",
            "REFINE",
        }:
            raise SystemExit(f"{base} is missing role-semantic evidence")
        commit = role_report.get("commit_determinism", {})
        if (
            int(commit.get("passing_questions", -1)) != 200
            or int(commit.get("n_questions", -1)) != 200
            or float(commit.get("commit_only_readout_rate", -1.0)) != 1.0
        ):
            raise SystemExit(f"{base} failed the deterministic COMMIT audit")
        qa = json.loads(
            require_file(root, f"{base}/figure_qa.json").read_text(
                encoding="utf-8"
            )
        )
        if qa.get("status") != "PASS":
            raise SystemExit(f"{base} figure QA did not pass")

    left = torch.load(
        root / "stage1_geometry_summary/global_train_fit_pca.pt",
        map_location="cpu",
        weights_only=False,
    )
    right = torch.load(
        root / "final_geometry_summary/global_train_fit_pca.pt",
        map_location="cpu",
        weights_only=False,
    )
    for key in ("mean", "components", "explained_ratio"):
        if not torch.allclose(left[key], right[key], atol=1e-7, rtol=1e-6):
            raise SystemExit("Stage-1 and final reports do not share one PCA")

    comparison = json.loads(
        require_file(root, "stage_comparison/stage_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    if int(comparison.get("paired_questions", -1)) != 200:
        raise SystemExit("stage comparison is not paired on exactly 200 questions")
    if (
        comparison.get("evidence_priority", {}).get("primary")
        != "behavior_and_terminal_outcome_calibration"
    ):
        raise SystemExit(
            "stage comparison does not prioritize behavior/outcome calibration"
        )
    for phase in ("stage1", "final"):
        if not comparison.get("commit_determinism", {}).get(phase, {}).get(
            "pass", False
        ):
            raise SystemExit(f"stage comparison {phase} COMMIT audit failed")
    calibration = comparison.get("terminal_outcome_value_calibration", {})
    if (
        calibration.get("stage1_head_type") != "sufficiency"
        or calibration.get("final_head_type") != "outcome_critic"
        or calibration.get("target") != "terminal_exact_correctness"
    ):
        raise SystemExit("stage comparison has an invalid S-to-V calibration contract")
    for phase in ("stage1", "final"):
        summary = calibration.get(phase, {})
        values = (
            summary.get("brier"),
            summary.get("ece_10bin"),
            summary.get("mean_prediction"),
            summary.get("empirical_accuracy"),
        )
        if (
            int(summary.get("n_paths", -1)) != 1600
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in values
            )
            or len(summary.get("bins", [])) != 10
        ):
            raise SystemExit(
                f"stage comparison {phase} value calibration is incomplete"
            )
    for relative in (
        "stage_comparison.md",
        "stage_comparison_source_data.npz",
        "figure_contract.json",
        "figure_qa.json",
        "stage_accuracy_length.svg",
        "stage_accuracy_length.pdf",
        "stage_accuracy_length.tiff",
        "stage_reliability.svg",
        "stage_reliability.pdf",
        "stage_reliability.tiff",
        "stage_value_calibration.svg",
        "stage_value_calibration.pdf",
        "stage_value_calibration.tiff",
        "stage_role_evidence.svg",
        "stage_role_evidence.pdf",
        "stage_role_evidence.tiff",
        "stage_outcome_geometry.svg",
        "stage_outcome_geometry.pdf",
        "stage_outcome_geometry.tiff",
    ):
        require_file(root, f"stage_comparison/{relative}")

    causal = json.loads(
        require_file(root, "causal_summary/causal_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if int(causal.get("questions", -1)) != 200:
        raise SystemExit("causal audit is not based on exactly 200 questions")
    priority = causal.get("evidence_priority", {})
    if (
        priority.get("primary")
        != "same_norm_transition_replacement_with_suffix_and_commit_recomputation"
        or priority.get("co_primary_elsewhere")
        != "terminal_outcome_value_calibration"
        or priority.get("diagnostic_only")
        != "commit_bottleneck_sanity_curve"
    ):
        raise SystemExit("causal audit assigns an invalid evidence priority")
    if (
        causal.get("answer_context_contract") != "path_only_COMMIT"
        or causal.get("question_attention_access") is not False
    ):
        raise SystemExit(
            "causal audit violates the strict path-only COMMIT contract"
        )
    bottleneck = causal.get("commit_bottleneck_sanity", {})
    if (
        bottleneck.get("status") != "PASS"
        or bottleneck.get("prefix_0_through_7_equal_no_readout") is not True
        or bottleneck.get("prefix_8_enables_commit") is not True
        or "not measure incremental step sufficiency"
        not in str(bottleneck.get("interpretation", ""))
    ):
        raise SystemExit(
            "causal audit misstates or fails the COMMIT-bottleneck sanity check"
        )
    intervention_types = causal.get("transition_intervention_types", [])
    if (
        len(intervention_types) != 8
        or intervention_types[:7]
        != ["same-norm action replacement plus suffix recomputation"] * 7
        or intervention_types[-1] != "COMMIT readout ablation"
        or causal.get("transition_gold_drop", {}).get("family_size") != 8
    ):
        raise SystemExit("causal audit does not cover all seven roles plus COMMIT")
    protocol = causal.get("transition_replacement_protocol", {})
    if (
        protocol.get("prefix")
        != "all deterministic actions before the replaced role held fixed"
        or "conditional means" not in str(protocol.get("suffix", ""))
        or "COMMIT recomputed" not in str(protocol.get("suffix", ""))
    ):
        raise SystemExit(
            "causal transition replacement does not recompute the full suffix"
        )
    figure_contract = json.loads(
        require_file(root, "causal_summary/figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        figure_contract.get("answer_context_contract") != "path_only_COMMIT"
        or int(figure_contract.get("question_attention_access", -1)) != 0
        or "not stepwise sufficiency"
        not in str(figure_contract.get("diagnostic_only", ""))
    ):
        raise SystemExit("causal figure contract overclaims prefix evidence")
    for relative in (
        "question_causal_interventions.csv",
        "causal_source_data.npz",
        "figure_contract.json",
        "figure_qa.json",
        "causal_accuracy.svg",
        "causal_accuracy.pdf",
        "causal_accuracy.tiff",
        "commit_bottleneck_sanity.svg",
        "commit_bottleneck_sanity.pdf",
        "commit_bottleneck_sanity.tiff",
        "transition_effects.svg",
        "transition_effects.pdf",
        "transition_effects.tiff",
    ):
        require_file(root, f"causal_summary/{relative}")
    if count_csv_rows(root / "causal_summary/question_causal_interventions.csv") != 200:
        raise SystemExit("causal intervention CSV is not exactly 200 questions")

    return {
        "status": "COMPLETE",
        "evidence_root": str(root.resolve()),
        "stage1_checkpoint": str(stage1.resolve()),
        "stage1_checkpoint_sha256": sha256(stage1),
        "final_checkpoint": str(final.resolve()),
        "final_checkpoint_sha256": sha256(final),
        "dataset_questions": EXPECTED_DATASETS,
        "geometry_questions_per_stage": 200,
        "causal_questions": 200,
        "test_times": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--write-complete", action="store_true")
    args = parser.parse_args()
    report = validate(
        args.evidence_root.resolve(),
        args.stage1_checkpoint.resolve(),
        args.final_checkpoint.resolve(),
    )
    if args.write_complete:
        output = args.evidence_root / "COMPLETE.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
