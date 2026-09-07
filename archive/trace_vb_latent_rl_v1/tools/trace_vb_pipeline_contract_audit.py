#!/usr/bin/env python3
"""Fail-closed static/data contract for the formal TRACE-VB pipeline."""

import json
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("TRACE_DATA_ROOT", "/disk1/dingxukai/TRACE"))
PYTHON = Path("/home/dingxukai/miniconda3/envs/ROT/bin/python")


def result(name: str, condition: bool, detail: str) -> dict:
    return {
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "detail": detail,
    }


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def data_audit() -> dict:
    environment = os.environ.copy()
    environment["TRACE_PROJECT_ROOT"] = str(ROOT)
    environment["TRACE_DATA_ROOT"] = str(DATA_ROOT)
    completed = subprocess.run(
        [str(PYTHON), str(ROOT / "tools/data_contract_audit.py")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    config = OmegaConf.load(
        ROOT / "src/configs/models/trace_vb_policy_qwen3_instruct.yaml"
    )
    policy = config.model.model_kwargs.trace_policy_config
    rl = config.model.model_kwargs.trace_rl_config
    readcot = config.model.model_kwargs.readcot_config
    stage1 = text("scripts/run_stage1_vb.sh")
    stage2 = text("scripts/run_stage2_vb.sh")
    evidence = text("scripts/run_evidence_vb.sh")
    full = text("scripts/run_full_pipeline_vb.sh")
    common = text("scripts/trace_vb_common.sh")
    model = text("src/models/trace_vb.py")
    math_module = text("src/modules/trace_vb.py")
    stage1_smoke = text("tools/trace_vb_stage1_smoke.py")
    stage2_smoke = text("tools/trace_vb_stage2_smoke.py")
    stage2_ddp_smoke = text("tools/trace_vb_stage2_ddp_smoke.py")
    stage_comparison = text("tools/trace_policy_stage_comparison.py")
    causal_summary = text("tools/trace_policy_causal_summary.py")
    evidence_verifier = text("tools/verify_evidence_complete.py")
    stage2_start = model.index("    def trace_rl_training_step(")
    stage2_end = model.index("    def _legacy_trace_rl_training_step(", stage2_start)
    active_stage2 = model[stage2_start:stage2_end]
    all_formal_scripts = "\n".join((stage1, stage2, evidence, full, common))
    registered = data_audit()
    stage1_best_offset = full.index("stage1_checkpoint=$(")
    stage2_single_smoke_offset = full.index(
        '"${TRACE_VB_PYTHON}" tools/trace_vb_stage2_smoke.py',
        stage1_best_offset,
    )
    stage2_ddp_smoke_offset = full.index(
        "tools/trace_vb_stage2_ddp_smoke.py",
        stage2_single_smoke_offset + 1,
    )
    formal_stage2_offset = full.index(
        'bash "${SCRIPT_DIR}/run_stage2_vb.sh"',
        stage2_ddp_smoke_offset,
    )

    checks = [
        result(
            "vb_model_target",
            str(config.model.target) == "src.models.trace_vb.LitTRACEVB",
            "The dedicated config instantiates the isolated TRACE-VB class.",
        ),
        result(
            "isolated_artifact_root",
            "TRACE_VB_ARTIFACT_ROOT=/disk1/dingxukai/TRACE/trace_vb_runs"
            in common
            and "role_semantic_runs" not in all_formal_scripts,
            "Every formal output is isolated under trace_vb_runs.",
        ),
        result(
            "registered_stage0_is_pinned",
            "TRACE_VB_REGISTERED_STAGE0_SHA256=" in common
            and "trace_vb_require_stage0" in stage1
            and "load_ckpt_path" in stage1,
            "Stage 1 reuses only the hash-pinned registered CoT checkpoint.",
        ),
        result(
            "registered_full_data",
            registered.get("status") == "PASS"
            and registered["files"]["gsm8k_train"]["count"] == 6726
            and registered["files"]["gsm8k_val"]["count"] == 747
            and registered["files"]["gsm8k_test"]["count"] == 1319,
            "Registered GSM8K train/validation/test splits are intact.",
        ),
        result(
            "fixed_role_program",
            int(readcot.n_latents) == 8
            and "PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT"
            in all_formal_scripts,
            "The physical path is PLAN, five SOLVE states, REFINE, COMMIT.",
        ),
        result(
            "strict_commit_bottleneck",
            str(policy.answer_context_mode) == "path_only"
            and "answer_context_mode=path_only" in stage1
            and "answer_context_mode=path_only" in stage2,
            "Answer decoding cannot read the raw question or pre-COMMIT states.",
        ),
        result(
            "simplified_stage1",
            int(policy.stage1_stochastic_paths) == 1
            and int(policy.stage1_posterior_samples) == 0
            and not bool(readcot.use_hybrid)
            and not bool(readcot.use_anchor_loss)
            and float(policy.stage1_answer_weight) == 1.0,
            "Stage 1 contains one mean path, one stochastic prior path, and one answer loss.",
        ),
        result(
            "semantic_value_bridge",
            float(policy.stage1_plan_forecast_weight) > 0
            and float(policy.stage1_sufficiency_weight) > 0
            and "sufficiency_cache_path" in policy
            and all(
                token in model
                for token in (
                    "PlanForecastHead",
                    "plan_forecaster",
                    "sufficiency_head",
                    "value_critic",
                )
            ),
            "PLAN forecast and offline sufficiency initialize the outcome critic.",
        ),
        result(
            "action_efficacy_gate",
            float(policy.stage1_minimum_action_efficacy_ratio) > 0.0
            and float(policy.stage1_action_efficacy_weight) > 0.0
            and "action_efficacy_hinge" in math_module
            and "trace_vb_action_efficacy_loss" in model
            and "trace_vb_action_efficacy_ratio" in model
            and "action_efficacy_in_total_loss" in stage1_smoke
            and '"smoke_gate": 0.01' in stage2_smoke,
            "Stage 1 enforces action-to-transition efficacy and both smokes fail closed on its contract.",
        ),
        result(
            "no_cot_posterior",
            "self.trajectory_posterior = None" in model
            and "self.trajectory_posterior = CoTConditioned" not in model,
            "Formal TRACE-VB has no train-only CoT-conditioned latent posterior.",
        ),
        result(
            "terminal_only_rl",
            bool(rl.use_terminal_exact_reward)
            and float(rl.dense_outcome_weight) == 0
            and float(rl.step_reward_weight) == 0
            and float(rl.trajectory_length_weight) == 0,
            "The only RL reward is final exact correctness.",
        ),
        result(
            "gae_credit",
            bool(rl.use_gae)
            and float(rl.gamma) == 1.0
            and float(rl.gae_lambda) == 0.95
            and "masked_terminal_reward_gae" in math_module
            and "pre_action_states" in model,
            "GAE assigns terminal outcome credit to saved pre-action states.",
        ),
        result(
            "head_only_ppo",
            bool(rl.use_head_only_ppo)
            and int(rl.policy_update_epochs) == 4
            and int(rl.rollout_micro_batch_size) == 1
            and "num_training_steps=20480" in stage2,
            "Each rollout receives four actor/critic head-only updates.",
        ),
        result(
            "strict_stage2_tensor_and_trainability_contract",
            "rollout_float32_fields" in stage2_smoke
            and all(
                token in stage2_smoke
                for token in (
                    '"policy_trunk"',
                    '"policy_step_embedding"',
                    '"dynamics_step_embedding"',
                    '"mean_heads.commit"',
                    '("plan", "solve", "check")',
                )
            ),
            "Stage-2 smoke permits only stochastic heads/value critic and requires FP32 PPO tensors.",
        ),
        result(
            "real_four_rank_stage2_ddp_gate",
            all(
                token in stage2_ddp_smoke
                for token in (
                    "world_size != 4",
                    "group_size_per_rank",
                    "terminal_exact_reward_only",
                    "FLOAT32_ROLLOUT_FIELDS",
                    "validate_trainability",
                    "warmup_output[\"objective\"].backward()",
                    "args.actor_updates < 2",
                    "cross_rank_max_difference",
                    "nonunit_ratio_observed_on_later_update",
                )
            )
            and "tools/trace_vb_stage2_ddp_smoke.py" in full
            and "--actor-updates 2" in full
            and stage1_best_offset
            < stage2_single_smoke_offset
            < stage2_ddp_smoke_offset
            < formal_stage2_offset,
            "A real Stage-1 checkpoint must pass single-GPU then four-rank cached-state PPO before formal Stage 2.",
        ),
        result(
            "critic_warmup",
            int(rl.critic_warmup_batches) == 256
            and "critic_warmup_batches=256" in stage2
            and "vb_rollout_batches_seen" in active_stage2
            and "actor_active=" in active_stage2.replace(" ", ""),
            "The critic is calibrated for 256 rollout batches before actor updates.",
        ),
        result(
            "active_stage2_is_vb_only",
            all(
                token not in active_stage2
                for token in (
                    "_decode_single_gold_cots",
                    "role_targets",
                    "semantic_step_rewards",
                    "trajectory_rewards",
                    "answer_rewards",
                )
            )
            and "actor_active=actor_active" in active_stage2.replace(" ", "")
            and "terminal_rewards" in active_stage2,
            "The active training entry cannot call legacy CoT/shaped-reward code.",
        ),
        result(
            "full_stage1_budget",
            "for target_max_epochs in $(seq 1 10)" in stage1
            and "trainer.limit_train_batches=1.0" in stage1
            and "trainer.limit_val_batches=1.0" in stage1
            and "early_stopping=false" in stage1,
            "Stage 1 runs ten complete epochs with full 747-question validation.",
        ),
        result(
            "full_stage2_budget",
            "trainer.max_epochs=10" in stage2
            and "trainer.limit_train_batches=512" in stage2
            and "trainer.limit_val_batches=1.0" in stage2
            and "n_train_samples_per_epoch=2048" in stage2,
            "Stage 2 runs ten full 2,048-question rollout epochs and full validation.",
        ),
        result(
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
                    "verify_evidence_complete.py",
                    "--write-complete",
                )
            ),
            "IID/OOD, paired geometry, causal bootstrap, and completeness gate are retained.",
        ),
        result(
            "terminal_value_calibration_evidence",
            all(
                token in stage_comparison
                for token in (
                    "terminal_outcome_value_calibration",
                    "question_final_action_brier",
                    "ece_10bin",
                    "stage_value_calibration",
                    "behavior_and_terminal_outcome_calibration",
                )
            )
            and all(
                token in evidence_verifier
                for token in (
                    "terminal_outcome_value_calibration",
                    '"sufficiency"',
                    '"outcome_critic"',
                    '"stage_value_calibration.svg"',
                )
            ),
            "Paired evidence treats terminal-outcome calibration as primary and verifies its complete artifacts.",
        ),
        result(
            "strict_path_only_causal_semantics",
            all(
                token in causal_summary
                for token in (
                    "same_norm_transition_replacement_with_suffix_and_commit_",
                    "commit_bottleneck_sanity_curve",
                    "prefix 1--7 must equal no-readout",
                    "the question influences the score only through the",
                    "commit_bottleneck_sanity",
                )
            )
            and all(
                token in evidence_verifier
                for token in (
                    "same_norm_transition_replacement_with_suffix_and_commit_recomputation",
                    "prefix_0_through_7_equal_no_readout",
                    "commit_bottleneck_sanity.svg",
                    "question_attention_access",
                )
            )
            and "prefix_curve_interpretation=COMMIT_bottleneck_sanity_only_not_stepwise_contribution"
            in evidence,
            "Causal evidence is suffix-recomputed transition replacement; the prefix curve is only a COMMIT-bottleneck sanity check.",
        ),
        result(
            "resource_gate",
            "TRACE_VB_MIN_FREE_GPU_MIB=21500" in common
            and "trace_vb_require_four_gpus" in stage1
            and "trace_vb_require_four_gpus" in stage2,
            "Every four-GPU training phase fails closed below 21,500 MiB free per GPU.",
        ),
        result(
            "vb_names_are_consistent",
            "--model \"${TRACE_VB_MODEL_CONFIG}\"" in stage1
            and "--model \"${TRACE_VB_MODEL_CONFIG}\"" in stage2
            and "--model \"${TRACE_VB_MODEL_CONFIG}\"" in evidence
            and "trace_policy_qwen3_instruct.yaml" not in all_formal_scripts,
            "All formal commands resolve the TRACE-VB model configuration.",
        ),
    ]
    passed = sum(item["status"] == "PASS" for item in checks)
    report = {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "project_root": str(ROOT),
        "artifact_root": "/disk1/dingxukai/TRACE/trace_vb_runs",
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
