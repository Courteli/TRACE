#!/usr/bin/env python3
"""Fail-closed static/data contract for the formal TRACE-VB pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.datasets.gsm8k_aug_nl import split_cot_steps
from src.models.trace_vb import LitTRACEVB
from src.modules.trace_policy import contiguous_cot_chunk_spans


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


def solve_text_train_audit(max_tokens: int, model_path: str) -> dict:
    """Verify full train-CoT coverage without reading question or answer text."""
    if int(max_tokens) < 2:
        raise ValueError("solve text token bound must include content plus EOS")
    train_path = (
        DATA_ROOT
        / "data/raw/GSM8k-Aug-NL/gsm8k_train_processed.jsonl"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    rows = 0
    active_chunks = 0
    maximum_tokens_with_eos = 0
    overflow_chunks = 0
    exact_coverage = True
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            steps = [
                str(step).strip()
                for step in split_cot_steps(payload["cot"])
            ]
            if not steps or any(not step for step in steps):
                raise RuntimeError(
                    f"empty training CoT at line {line_number}"
                )
            token_ids = tokenizer.encode(
                "\n".join(steps),
                add_special_tokens=False,
            )
            if not token_ids:
                raise RuntimeError(
                    f"token-empty training CoT at line {line_number}"
                )
            spans = contiguous_cot_chunk_spans(
                len(token_ids),
                n_chunks=5,
            )
            rebuilt = []
            for start, end in spans:
                if end <= start:
                    continue
                chunk_tokens = int(end - start + 1)
                maximum_tokens_with_eos = max(
                    maximum_tokens_with_eos,
                    chunk_tokens,
                )
                overflow_chunks += int(chunk_tokens > int(max_tokens))
                active_chunks += 1
                rebuilt.extend(token_ids[start:end])
            exact_coverage = exact_coverage and rebuilt == token_ids
            rows += 1
    return {
        "rows": rows,
        "active_chunks": active_chunks,
        "maximum_tokens_with_eos": maximum_tokens_with_eos,
        "overflow_chunks": overflow_chunks,
        "exact_coverage": exact_coverage,
        "source_fields": ["cot"],
    }


def compact_target_train_audit(config) -> dict:
    """Verify the train-only compact target keeps answer and final CoT math."""
    train_path = (
        DATA_ROOT
        / "data/raw/GSM8k-Aug-NL/gsm8k_train_processed.jsonl"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model.model_kwargs.llm_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = LitTRACEVB.__new__(LitTRACEVB)
    model.tokenizer = tokenizer
    model.anchor_header = "Anchors:"
    model.thinking_separator = "###"
    model.answer_template = "Answer:{}"
    model.trace_config = config.model.model_kwargs.trace_policy_config
    model.readcot_config = config.model.model_kwargs.readcot_config
    model.model_kwargs = SimpleNamespace(
        hybrid_generation_config=SimpleNamespace(
            max_new_tokens=int(
                config.model.model_kwargs.hybrid_generation_config.max_new_tokens
            )
        )
    )
    budget = int(model.trace_config.compact_target_max_new_tokens)
    rows = 0
    overflow_targets = 0
    protected_answers = 0
    protected_final_clauses = 0
    total_equation_lines = 0
    maximum_tokens = 0
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            steps = [
                str(step).strip()
                for step in split_cot_steps(payload["cot"])
            ]
            answer = str(payload["answer"])
            if not steps or any(not step for step in steps):
                raise RuntimeError(
                    f"empty compact-target CoT at line {line_number}"
                )
            spans = contiguous_cot_chunk_spans(len(steps), n_chunks=5)
            target = model._build_role_compact_targets(
                [{"steps": steps}],
                [answer],
                [spans],
            )[0]
            token_count = model._target_token_count(target)
            maximum_tokens = max(maximum_tokens, token_count)
            overflow_targets += int(token_count > budget)
            protected_answers += int(
                target.endswith(
                    model.thinking_separator
                    + model.answer_template.format(answer)
                )
            )
            final_rendering = model._format_anchor_step(steps[-1])
            final_clauses = [
                clause.strip()
                for clause in final_rendering.split(";")
                if clause.strip()
            ]
            if not final_clauses:
                raise RuntimeError(
                    f"empty final compact clause at line {line_number}"
                )
            protected_final_clauses += int(final_clauses[-1] in target)
            total_equation_lines += sum(
                line.startswith("- ") for line in target.splitlines()
            )
            rows += 1
    return {
        "rows": rows,
        "budget": budget,
        "maximum_tokens": maximum_tokens,
        "overflow_targets": overflow_targets,
        "protected_answers": protected_answers,
        "protected_final_clauses": protected_final_clauses,
        "mean_equation_lines": total_equation_lines / max(rows, 1),
        "source_fields": ["cot", "answer"],
    }


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
    stage1_ddp_smoke = text("tools/trace_vb_stage1_ddp_smoke.py")
    stage2_smoke = text("tools/trace_vb_stage2_smoke.py")
    stage1_trainer = text("src/configs/trainer/trace_vb_stage1_v2.yaml")
    stage2_ddp_smoke = text("tools/trace_vb_stage2_ddp_smoke.py")
    stage_comparison = text("tools/trace_policy_stage_comparison.py")
    causal_summary = text("tools/trace_policy_causal_summary.py")
    evidence_verifier = text("tools/verify_evidence_complete.py")
    stage2_start = model.index("    def trace_rl_training_step(")
    stage2_end = model.index("    def _legacy_trace_rl_training_step(", stage2_start)
    active_stage2 = model[stage2_start:stage2_end]
    anchor_start = model.index("    def _stage2_role_semantic_anchor(")
    anchor_end = model.index("    def _role_step_rewards(", anchor_start)
    semantic_anchor = model[anchor_start:anchor_end]
    rollout_start = model.index("    def trace_policy_rollout(")
    rollout_end = model.index("    def _legacy_semantic_rollout(", rollout_start)
    active_rollout = model[rollout_start:rollout_end]
    stage1_start = model.index("    def _stage1_forward_impl(")
    stage1_end = model.index("    def _legacy_role_forward(", stage1_start)
    active_stage1 = model[stage1_start:stage1_end]
    deployment_start = model.index("    def read_generate_with_trajectory(")
    deployment_end = model.index("    def read_generate(", deployment_start)
    deployment_generation = model[deployment_start:deployment_end]
    eval_start = model.index("    def eval_generation(")
    eval_end = model.index("    def on_validation_epoch_start(", eval_start)
    eval_method = model[eval_start:eval_end]
    all_formal_scripts = "\n".join((stage1, stage2, evidence, full, common))
    registered = data_audit()
    solve_text_registered = solve_text_train_audit(
        int(policy.solve_text_decoder_max_tokens),
        str(config.model.model_kwargs.llm_path),
    )
    compact_target_registered = compact_target_train_audit(config)
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
            "TRACE_VB_ARTIFACT_ROOT=/disk1/dingxukai/TRACE/trace_vb_v5_runs"
            in common
            and "role_semantic_runs" not in all_formal_scripts,
            "Every formal output is isolated under trace_vb_v5_runs.",
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
            "registered_train_solve_text_coverage",
            solve_text_registered["rows"] == 6726
            and solve_text_registered["active_chunks"] == 5 * 6726
            and solve_text_registered["overflow_chunks"] == 0
            and solve_text_registered["exact_coverage"]
            and solve_text_registered["source_fields"] == ["cot"]
            and solve_text_registered["maximum_tokens_with_eos"]
            <= int(policy.solve_text_decoder_max_tokens),
            "All 6,726 training CoTs activate five ordered SOLVE spans, "
            "cover every token once, read only the CoT field, and fit the "
            f"bound (max={solve_text_registered['maximum_tokens_with_eos']}).",
        ),
        result(
            "registered_train_compact_target_coverage",
            compact_target_registered["rows"] == 6726
            and compact_target_registered["overflow_targets"] == 0
            and compact_target_registered["protected_answers"] == 6726
            and compact_target_registered["protected_final_clauses"] == 6726
            and compact_target_registered["maximum_tokens"]
            <= int(policy.compact_target_max_new_tokens)
            and compact_target_registered["source_fields"] == ["cot", "answer"],
            "All 6,726 train-only compact targets fit the unchanged 48-token "
            "budget and preserve both the protected answer and the final "
            "observed CoT equation.",
        ),
        result(
            "fixed_role_program",
            int(readcot.n_latents) == 8
            and "PLAN_SOLVE1_SOLVE2_SOLVE3_SOLVE4_SOLVE5_REFINE_COMMIT"
            in all_formal_scripts,
            "The physical path is PLAN, five SOLVE states, REFINE, COMMIT.",
        ),
        result(
            "reliable_question_commit_bridge",
            str(policy.answer_context_mode) == "question_and_commit"
            and "answer_context_mode=question_and_commit" in stage1
            and "answer_context_mode=question_and_commit" in stage2
            and "_commit_only_latent_mask" in model
            and "self.answer_reads_question = True" in model
            and bool(policy.commit_residual_bridge)
            and float(policy.stage1_commit_weight) > 0.0
            and "current_state + previous_state" in model
            and 'commit_weight * semantic_losses["commit"]' in active_stage1,
            "Answer decoding reads raw question plus a residual-preserved, "
            "text-CoT-aligned COMMIT, never the private role states.",
        ),
        result(
            "teacher_deployment_stage1",
            int(policy.stage1_stochastic_paths) == 1
            and int(policy.stage1_posterior_samples) == 1
            and float(policy.stage1_posterior_kl_weight) > 0.0
            and float(policy.stage1_sampled_answer_weight) > 0.0
            and float(policy.stage1_map_compact_weight) > 0.0
            and float(policy.stage1_map_answer_suffix_weight) > 0.0
            and str(readcot.anchor_text_mode) == "compact_equation"
            and int(readcot.compact_anchor_max_chars) == 32
            and bool(policy.deployment_compact_reasoning)
            and bool(policy.stage1_answer_activation_checkpoint)
            and "stage1_answer_activation_checkpoint=true" in stage1
            and "logits = checkpoint(" in model
            and "use_reentrant=False" in model
            and not bool(readcot.use_hybrid)
            and not bool(readcot.use_anchor_loss)
            and "posterior_context=cot_contexts" in model
            and "include_hybrid_header=True" in model,
            "One sample-local CoT teacher transfers role semantics to one "
            "question-only deployment path trained with its exact compact "
            "generation protocol.",
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
            "residual_only_step_text_grounding",
            float(policy.stage1_solve_text_weight) > 0.0
            and float(policy.stage1_solve_weight)
            < float(policy.stage1_solve_text_weight)
            and int(policy.solve_text_decoder_max_tokens) >= 2
            and bool(policy.solve_text_decoder_fail_on_truncation)
            and "solve_text_decoder_fail_on_truncation=true" in stage1
            and "LatentStepTextDecoder" in math_module
            and "per_sequence_token_cross_entropy" in math_module
            and "_solve_text_chunk_records" in model
            and "_solve_text_token_records" in model
            and "contiguous_cot_chunk_spans" in model
            and "self.embedding(previous_ids).detach()" in model
            and "_frozen_lm_head_logits" in model
            and "solve_decoder_question_access=false" in stage1
            and "solve_decoder_answer_field_access=false" in stage1
            and "detected CoT truncation" in stage1_smoke
            and "detected CoT truncation" in stage2_smoke
            and "solve_text_decoder." in stage2
            and "solve_text_decoder." in evidence,
            "Every SOLVE residual reconstructs one balanced, contiguous span "
            "of its sample-local gold CoT; all five spans cover the CoT once, "
            "and question/answer access or silent truncation is excluded.",
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
            "single_training_only_cot_teacher",
            "self.trajectory_posterior = CoTConditionedTrajectoryPosterior(" in model
            and "posterior_context=cot_contexts"
            in active_stage1.replace(" ", "")
            and "deterministic=True"
            in deployment_generation.replace(" ", "")
            and "posterior_context" not in deployment_generation
            and eval_method.index("read_generate_with_trajectory")
            < eval_method.index("_decode_single_gold_cots")
            and "deployment_compact_reasoning: true"
            in text("src/configs/models/trace_vb_policy_qwen3_instruct.yaml")
            and not bool(policy.stage1_posterior_activation_offload),
            "Gold CoT conditions exactly one training-only teacher posterior; "
            "validation and deployment use the question-only prior mean.",
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
            "decayed_mean_path_semantic_anchor",
            bool(rl.use_semantic_anchor)
            and float(rl.semantic_anchor_initial_weight)
            > float(rl.semantic_anchor_minimum_weight)
            >= 0.0
            and int(rl.semantic_anchor_decay_batches) > 0
            and "_stage2_role_semantic_anchor(batch)" in active_stage2
            and "update_index == 0" in active_stage2
            and "self.manual_backward(weighted_anchor)" in active_stage2
            and "deterministic=True" in semantic_anchor.replace(" ", "")
            and "_solve_text_decoder_loss" in semantic_anchor
            and "not self._solve_text_decoder_loaded" in model
            and "semantic_anchor_changes_terminal_reward=false" in stage2
            and "terminal_rewards = greedy_accuracy.detach()" in active_rollout,
            "After critic warm-up, one deterministic mean path per rollout "
            "receives a frozen-decoder semantic loss whose coefficient decays; "
            "the environment reward remains exact terminal correctness.",
        ),
        result(
            "active_stage2_keeps_reward_unshaped",
            all(
                token not in active_rollout
                for token in (
                    "semantic_step_rewards",
                    "trajectory_rewards",
                    "answer_rewards",
                    "dense_outcome_weight *",
                )
            )
            and "actor_active=actor_active" in active_stage2.replace(" ", "")
            and "_stage2_role_semantic_anchor(batch)" in active_stage2
            and "terminal_rewards = greedy_accuracy.detach()" in active_rollout,
            "Gold CoT is consumed only by the auxiliary anchor; rollout reward "
            "and GAE remain terminal exact-correctness only.",
        ),
        result(
            "full_stage1_budget",
            "for target_max_epochs in $(seq 1 10)" in stage1
            and "target_max_epochs <= initial_completed_epochs" in stage1
            and "initial_resume_monitor" in stage1
            and "trainer.limit_train_batches=1.0" in stage1
            and "trainer.limit_val_batches=1.0" in stage1
            and "early_stopping=false" in stage1,
            "Stage 1 runs ten complete epochs with full 747-question validation.",
        ),
        result(
            "epoch1_failure_gate",
            "TRACE_VB_EPOCH1_MIN_ACCURACY=0.60" in common
            and "checkpoint_current_monitor" in stage1
            and "Epoch-1 behavior gate failed" in stage1
            and "epoch1_behavior_gate.json" in stage1
            and "unique_prediction_ratio" in stage1
            and "top1_mode_fraction" in stage1,
            "Accuracy and prediction-diversity gates stop a collapsed first epoch.",
        ),
        result(
            "partial_epoch_resume_contract",
            'epoch_progress.get("processed")' in stage1
            and 'checkpoint.get("epoch", -1)' not in stage1
            and "initial_completed_epochs > 0" in stage1
            and "--resume_ckpt_path" in stage1,
            "Recovery checkpoints count only fully processed epochs; a "
            "mid-epoch checkpoint resumes the remaining batches instead of "
            "skipping directly to the next epoch.",
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
            "question_commit_causal_semantics",
            all(
                token in causal_summary
                for token in (
                    "same_norm_transition_replacement_with_suffix_and_commit_",
                    "commit_bottleneck_sanity_curve",
                    "prefix 1--7 must equal question-only",
                    "question_and_commit",
                    "commit_bottleneck_sanity",
                )
            )
            and all(
                token in evidence_verifier
                for token in (
                    "same_norm_transition_replacement_with_suffix_and_commit_recomputation",
                    "prefix_0_through_7_equal_question_only",
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
            and "trace_vb_require_four_gpus" in stage2
            and not bool(policy.stage1_posterior_activation_offload)
            and "stage1_posterior_activation_offload=false" in stage1
            and "TRACE-VB-v5 forbids saved-tensor CPU activation offload"
            in model
            and "stage1_host_memory_guard_interval=10" in stage1
            and "stage1_maximum_rank_rss_gib=20.0" in stage1
            and "--trainer trace_vb_stage1_v2" in stage1
            and "ddp_gradient_as_bucket_view=true" in stage1
            and "find_unused_parameters: false" in stage1_trainer
            and "gradient_as_bucket_view: true" in stage1_trainer
            and "static_graph: true" in stage1_trainer
            and "broadcast_buffers: false" in stage1_trainer
            and "bucket_cap_mb: 4" in stage1_trainer
            and "find_unused_parameters=False" in stage1_ddp_smoke
            and "gradient_as_bucket_view=True" in stage1_ddp_smoke
            and "static_graph=True" in stage1_ddp_smoke
            and "broadcast_buffers=False" in stage1_ddp_smoke
            and "bucket_cap_mb=4" in stage1_ddp_smoke,
            "Four-GPU phases use GPU-resident activations, fail closed on GPU/host-memory limits, and share the audited low-overhead DDP configuration.",
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
        "artifact_root": "/disk1/dingxukai/TRACE/trace_vb_v5_runs",
        "solve_text_train_audit": solve_text_registered,
        "compact_target_train_audit": compact_target_registered,
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
