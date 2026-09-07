#!/usr/bin/env python3
"""Fail-closed static and mathematical audit for formal TRACE-VB-v6 Stage 2."""

import ast
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modules.trace_vb import evidence_gated_group_advantages


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def main() -> None:
    model_path = ROOT / "src/models/trace_vb.py"
    module_path = ROOT / "src/modules/trace_vb.py"
    launcher_path = ROOT / "scripts/run_stage2_vb.sh"
    common_path = ROOT / "scripts/trace_vb_common.sh"
    config_path = (
        ROOT / "src/configs/models/trace_vb_policy_qwen3_instruct.yaml"
    )
    model_source = model_path.read_text(encoding="utf-8")
    module_source = module_path.read_text(encoding="utf-8")
    launcher = launcher_path.read_text(encoding="utf-8")
    common = common_path.read_text(encoding="utf-8")
    ast.parse(model_source, filename=str(model_path))
    ast.parse(module_source, filename=str(module_path))

    rollout = section(
        model_source,
        "    def trace_policy_rollout(",
        "    def _legacy_semantic_rollout(",
    )
    update = section(
        model_source,
        "    def _trajectory_policy_update(",
        "    def _legacy_trajectory_policy_update(",
    )
    training = section(
        model_source,
        "    def trace_rl_training_step(",
        "    def _legacy_trace_rl_training_step(",
    )
    config = OmegaConf.load(config_path)
    rl = config.model.model_kwargs.trace_rl_config

    checks = {
        "critic_free_actor_signal": (
            not bool(rl.use_gae)
            and "value_critic" not in rollout
            and "value_critic" not in update
            and "masked_terminal_reward_gae" not in rollout
        ),
        "exact_mixed_group_signal": (
            bool(rl.use_terminal_exact_reward)
            and bool(rl.use_evidence_gated_group_rl)
            and "evidence_gated_group_advantages(" in rollout
            and "terminal_rewards" in rollout
        ),
        "calibrated_fail_closed_proxy": (
            int(rl.score_calibration_batches) == 128
            and int(rl.score_proxy_minimum_pairs) == 64
            and float(rl.score_proxy_minimum_auc) == 0.60
            and "proxy_pair_credit" in training
            and "vb_score_proxy_enabled" in training
            and "all_wrong" in module_source
        ),
        "post_update_stage1_kl_guard": (
            float(rl.stage1_policy_target_kl) == 0.01
            and "_stage1_policy_kl_on_rollout(rollout)" in training
            and "global_post_step_kl" in training
        ),
        "role_semantics_retained": (
            bool(rl.use_semantic_anchor)
            and float(rl.semantic_anchor_minimum_weight) == 0.01
            and "_stage2_role_semantic_anchor(batch)" in training
            and "semantic_anchor_changes_terminal_reward=false" in launcher
        ),
        "head_only_trainability": (
            "trainable_modules=stochastic_role_actor_heads_only" in launcher
            and "unexpected" in model_source
            and "actor_parameters" in model_source
        ),
        "formal_evaluation_budget": (
            "trainer.max_epochs=10" in launcher
            and "trainer.limit_train_batches=512" in launcher
            and "trainer.limit_val_batches=1.0" in launcher
            and "full_747_validation_every_epoch=true" in launcher
        ),
        "isolated_artifacts": (
            "TRACE_VB_ARTIFACT_ROOT=/disk1/dingxukai/TRACE/trace_vb_v6_runs"
            in common
        ),
        "question_commit_bridge": (
            "answer_context_mode=question_and_commit" in launcher
            and "private_latent_answer_access=false" in launcher
        ),
        "forbidden_shaping_zero": all(
            f"trace_rl_config.{key}=0.0" in launcher
            for key in (
                "dense_outcome_weight",
                "step_reward_weight",
                "trajectory_length_weight",
            )
        ),
    }
    for name, passed in checks.items():
        require(bool(passed), f"TRACE-VB-v6 contract failed: {name}")

    # Mathematical regression: no evidence means exactly zero; mixed exact
    # outcomes dominate; a calibrated all-wrong score gives only local ranks.
    correctness = torch.tensor(
        [1.0] * 8
        + [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        + [0.0] * 8
    )
    scores = torch.tensor(
        list(range(8))
        + [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        + list(range(8)),
        dtype=torch.float32,
    )
    disabled = evidence_gated_group_advantages(
        correctness,
        scores,
        group_size=8,
        enable_likelihood_fallback=False,
        minimum_gold_score_gap=0.002,
    )
    require(torch.equal(disabled.path_advantages[:8], torch.zeros(8)), "all-correct group changed actor")
    require(float(disabled.path_advantages[8]) > 0.0, "mixed correct path lacks positive credit")
    require(torch.all(disabled.path_advantages[9:16] < 0.0), "mixed wrong paths lack negative credit")
    require(torch.equal(disabled.path_advantages[16:], torch.zeros(8)), "uncalibrated all-wrong group changed actor")
    enabled = evidence_gated_group_advantages(
        torch.zeros(8),
        torch.arange(8, dtype=torch.float32),
        group_size=8,
        enable_likelihood_fallback=True,
        minimum_gold_score_gap=0.002,
    )
    require(bool(enabled.likelihood_group_mask.item()), "calibrated fallback did not activate")
    require(torch.all(enabled.path_advantages[1:] > enabled.path_advantages[:-1]), "local likelihood ranks are not monotonic")

    print(
        json.dumps(
            {
                "status": "PASS",
                "contract": "TRACE-VB-v6 Stage-2",
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
