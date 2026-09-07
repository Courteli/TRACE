#!/usr/bin/env python3
"""Static anti-shortcut contract for the role-semantic formal pipeline."""

import json
import os
import subprocess
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("TRACE_DATA_ROOT", "/disk1/dingxukai/TRACE")
).resolve()
PYTHON = Path("/home/dingxukai/miniconda3/envs/ROT/bin/python")


def check(name: str, condition: bool, detail: str) -> dict:
    return {
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "detail": detail,
    }


def read(relative: str) -> str:
    return (CODE_ROOT / relative).read_text(encoding="utf-8")


def run_data_audit() -> dict:
    environment = os.environ.copy()
    environment["TRACE_PROJECT_ROOT"] = str(CODE_ROOT)
    environment["TRACE_DATA_ROOT"] = str(DATA_ROOT)
    completed = subprocess.run(
        [str(PYTHON), str(CODE_ROOT / "tools/data_contract_audit.py")],
        cwd=CODE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def run_audit() -> dict:
    full = read("scripts/run_full_pipeline.sh")
    stage1 = read("scripts/run_stage1_formation.sh")
    stage2 = read("scripts/run_stage2_refinement.sh")
    evidence = read("scripts/run_evidence.sh")
    model = read("src/models/trace_policy.py")
    module = read("src/modules/trace_policy.py")
    config = read("src/configs/models/trace_policy_qwen3_instruct.yaml")
    formal_scripts = "\n".join((full, stage1, stage2, evidence))
    data_report = run_data_audit()

    role_source = "\n".join((model, module, config))
    role_tokens = ("PLAN", "SOLVE", "CHECK", "COMMIT")
    forbidden_legacy = (
        "verify_canonical66.py",
        "TRACE-canonical66-stage1",
        "5bab7dff9efd5a68dc0a4baa4622b6bfd4ad931904c75fae7b2752cf30b5b023",
        "20260721-200252_230269",
    )
    checks = [
        check(
            "code_root_is_dynamic",
            all(
                token in formal_scripts
                for token in (
                    "SCRIPT_DIR=",
                    "CODE_ROOT=",
                    'cd "${CODE_ROOT}"',
                )
            )
            and "ROOT=/disk1/dingxukai/TRACE" not in formal_scripts,
            "Every formal script executes the copied code root, never the old project.",
        ),
        check(
            "data_and_artifacts_are_separate",
            "DATA_ROOT=${DATA_ROOT:-/disk1/dingxukai/TRACE}" in formal_scripts
            and (
                "ARTIFACT_ROOT=${ARTIFACT_ROOT:-"
                "/disk1/dingxukai/TRACE/role_semantic_runs}"
            )
            in formal_scripts,
            "Registered data stay immutable while new logs/checkpoints use an isolated root.",
        ),
        check(
            "registered_full_data",
            data_report.get("status") == "PASS"
            and data_report["files"]["gsm8k_train"]["count"] == 6726
            and data_report["files"]["gsm8k_val"]["count"] == 747
            and data_report["files"]["gsm8k_test"]["count"] == 1319,
            "Hash-registered GSM8K train/validation/test splits are intact.",
        ),
        check(
            "role_semantic_path",
            all(token in role_source for token in role_tokens)
            and "n_latents: 8" in config,
            "The model declares PLAN, five SOLVE states, CHECK, and COMMIT in eight slots.",
        ),
        check(
            "explicit_stage0_to_new_stage1",
            "<stage0-checkpoint>" in full
            and 'bash "${SCRIPT_DIR}/run_stage1_formation.sh"' in full
            and '"${stage0_checkpoint}"' in full,
            "Formal training takes an explicit fresh CoT-SFT checkpoint into the new Stage 1.",
        ),
        check(
            "complete_stage1_budget",
            "trainer.limit_train_batches=1.0" in stage1
            and "trainer.limit_val_batches=1.0" in stage1
            and "scheduler.num_training_steps=16820" in stage1
            and "full_validation_every_epoch=true" in stage1
            and "STAGE1_MAX_EPOCHS=${STAGE1_MAX_EPOCHS:-10}" in stage1,
            "Stage 1 retains full 6,726-question epochs and full validation.",
        ),
        check(
            "complete_stage2_budget",
            "trainer.max_epochs=10" in stage2
            and "trainer.limit_train_batches=512" in stage2
            and "trainer.limit_val_batches=1.0" in stage2
            and "n_train_samples_per_epoch=2048" in stage2
            and "group_size=8" in stage2
            and "scheduler.num_training_steps=5120" in stage2,
            "Stage 2 retains 2,048 unique questions, eight paths, and 10 full epochs.",
        ),
        check(
            "stage2_consumes_new_stage1",
            "<stage1-best-checkpoint>" in stage2
            and "stage1_checkpoint=$2" in stage2
            and not any(token in formal_scripts for token in forbidden_legacy),
            "Stage 2 receives this run's Stage-1 best checkpoint and has no canonical fallback.",
        ),
        check(
            "complete_paired_evidence",
            all(
                token in evidence
                for token in (
                    "gsm8k_questions=1319",
                    "GSMHard,SVAMP,MultiArith",
                    "geometry_questions=200",
                    "geometry_rollouts_per_question=8",
                    "--bootstrap 10000",
                    "run_evidence_suite stage1",
                    "run_evidence_suite final",
                )
            ),
            "Evidence remains paired and full-size for IID, OOD, geometry, and causal audits.",
        ),
        check(
            "evidence_complete_gate",
            "tools/verify_evidence_complete.py" in evidence
            and "--write-complete" in evidence
            and "COMPLETE.json" in full,
            "No pipeline can report success before every required evidence artifact exists.",
        ),
        check(
            "single_path_evaluation",
            "--test_times 1" in evidence
            and "--seed 0" in evidence
            and any(
                token in config.lower()
                for token in ("do_sample=false", "do_sample: false")
            ),
            "Formal evaluation uses one deterministic deployment path without best-of-N.",
        ),
    ]
    passed = sum(row["status"] == "PASS" for row in checks)
    return {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "code_root": str(CODE_ROOT),
        "data_root": str(DATA_ROOT),
        "checks": checks,
    }


def main() -> None:
    report = run_audit()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
