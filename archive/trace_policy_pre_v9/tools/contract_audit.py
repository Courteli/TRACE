#!/usr/bin/env python3
"""Static and data self-audit for the independent TRACE implementation."""

import argparse
import ast
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path("/disk1/dingxukai/TRACE")
MODEL = ROOT / "src/models/trace_policy.py"
MODULE = ROOT / "src/modules/trace_policy.py"
DATASET = ROOT / "src/datasets/gsm8k_aug_nl.py"
MODEL_CONFIG = ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml"
DATA_CONFIG = ROOT / "src/configs/datasets/gsm8k_aug_nl.yaml"
STAGE0 = ROOT / "scripts/run_stage0_cot.sh"
STAGE1 = ROOT / "scripts/run_stage1_formation.sh"
STAGE2 = ROOT / "scripts/run_stage2_refinement.sh"
EVIDENCE = ROOT / "scripts/run_evidence.sh"
PIPELINE = ROOT / "scripts/run_full_pipeline.sh"
GEOMETRY = ROOT / "tools/trace_policy_geometry_summary.py"
CAUSAL = ROOT / "tools/trace_policy_causal_summary.py"
TARGET_AUDIT = ROOT / "tools/trace_stage1_target_audit.py"
GSM8K_DATA = ROOT / "data/raw/GSM8k-Aug-NL"


def check(name, condition, detail):
    return {
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "detail": detail,
    }


def class_bases(source: str, class_name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [ast.unparse(base) for base in node.bases]
    return []


def run_data_audit():
    result = subprocess.run(
        [
            "/home/dingxukai/miniconda3/envs/ROT/bin/python",
            str(ROOT / "tools/data_contract_audit.py"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def run_audit():
    model = MODEL.read_text()
    module = MODULE.read_text()
    dataset = DATASET.read_text()
    model_config_text = MODEL_CONFIG.read_text()
    data_config_text = DATA_CONFIG.read_text()
    stage0 = STAGE0.read_text()
    stage1 = STAGE1.read_text()
    stage2 = STAGE2.read_text()
    evidence = EVIDENCE.read_text()
    pipeline = PIPELINE.read_text()
    geometry = GEOMETRY.read_text()
    causal = CAUSAL.read_text()
    target_audit = TARGET_AUDIT.read_text()
    formal_sources = "\n".join(
        (
            model,
            model_config_text,
            data_config_text,
            stage0,
            stage1,
            stage2,
            evidence,
            pipeline,
        )
    )
    config = OmegaConf.load(MODEL_CONFIG).model.model_kwargs
    trace = config.trace_policy_config
    rl = config.trace_rl_config
    data_report = run_data_audit()
    prohibited_paths = (
        "/disk1/dingxukai/trace_colar",
        "trace_final/data",
        "gsm8k_multirationale",
        "readcot_qsa",
    )
    fixed_route_patterns = (
        "center_seeds",
        "center_indices",
        "view_ids_tensor",
        "_sample_path_seeds",
        "trace_view_embeddings",
        "stage1_teacher_set_size",
        "stage1_max_semantic_modes",
        "build_teacher_rationale_schedule",
        "permutation_invariant_set_matching",
        "_build_action_conditioned_anchors",
    )
    checks = [
        check(
            "independent_project",
            all(path not in formal_sources for path in prohibited_paths),
            "Formal code and scripts reference only /disk1/dingxukai/TRACE.",
        ),
        check(
            "registered_raw_jsonl_only",
            (
                data_report["status"] == "PASS"
                and data_report["data_policy"][
                    "one_original_cot_per_question"
                ]
                and not data_report["data_policy"]["generated_cots"]
                and "enforce_registered_source: True" in data_config_text
            ),
            "Training uses the hash-registered project-local JSONL mirror.",
        ),
        check(
            "evidence_uses_registered_gsm8k_path",
            (
                GSM8K_DATA.is_dir()
                and f'dataset_dir="${{ROOT}}/data/raw/GSM8k-Aug-NL"' in evidence
                and (
                    f'--source-file "${{ROOT}}/data/raw/GSM8k-Aug-NL/'
                    'gsm8k_train_processed.jsonl"'
                )
                in evidence
                and "data/raw/GSM8K-Aug-NL" not in evidence
            ),
            "Formal evaluation and PCA fitting use the same case-sensitive "
            "registered GSM8K mirror as training.",
        ),
        check(
            "single_explicit_cot_contract",
            (
                "_decode_single_gold_cots" in model
                and "_collect_single_cot_features" in model
                and 'answers=[""] * len(gold_cots)' in model
                and "rationale_set_json" not in dataset
                and "generated_cots=false" in stage1
            ),
            "The posterior reads question plus one original CoT. No separate "
            "answer field is appended, although the original CoT may state "
            "the final answer.",
        ),
        check(
            "cot_conditioned_latent_posterior",
            (
                "CoTConditionedTrajectoryPosterior" in model
                and "posterior_context=repeated_contexts" in model
                and "stage1_posterior_samples" in model
                and "stage1_posterior_kl_weight" in model
                and "diagonal_gaussian_kl" in model
            ),
            "Stage 1 samples IID paths from q(P|question, CoT) and distills "
            "them into the question-only prior.",
        ),
        check(
            "single_cot_monotone_corridor",
            (
                "_single_cot_corridor" in model
                and "_build_single_cot_corridors" in model
                and "student_path.detach()" in model
                and "student_actions.detach()" in model
                and "action_conditioned_progress_centers" in model
                and "corridor_progress_logits" not in model
                and "cumulative / cumulative" in module
                and "stochastic_monotone_assignment" in model
                and float(trace.corridor_progress_action_scale) != 0.0
            ),
            "Each sampled action path defines its own ordered compression "
            "schedule against one observed CoT; no synthetic teacher set or "
            "globally shared progress template is created.",
        ),
        check(
            "no_fixed_or_pseudo_routes",
            (
                all(pattern not in model for pattern in fixed_route_patterns)
                and "_compress_explicit_reasoning(" not in (
                    model[
                        model.index("def forward(self, batch)") :
                        model.index("def _rollout_micro_batch_size")
                    ]
                )
            ),
            "No center path, fixed view identity, route table, rationale "
            "schedule, set-matching pseudo-route, or inherited fixed-query "
            "compressor remains in the Stage-1 path.",
        ),
        check(
            "iid_shared_policy_paths",
            (
                "repeated_questions" in model
                and "deterministic=False" in model
                and "GaussianTrajectoryPolicy" in model
                and "gaussian_log_prob" in model
            ),
            "All sampled paths share one autoregressive policy and differ "
            "only through fresh IID action samples.",
        ),
        check(
            "stable_stage0_preserving_recurrence",
            (
                "states.float() + self.base_projector(states.float())"
                in module
                and "nn.init.zeros_(self.base_projector[-1].weight)"
                in module
            ),
            "Fresh TRACE dynamics are an identity-centered residual update, "
            "so Stage 1 does not discard the Stage-0 hidden representation.",
        ),
        check(
            "anti_collapse_without_fixed_routes",
            (
                "action_transition_identifiability_loss" in model
                and "action_transition_retrieval_accuracy" in model
                and "minimum_action_entropy_loss" in model
                and "minimum_action_gate" in model_config_text
                and "action_gate" in module
                and "stage1_action_identifiability_weight" in model_config_text
                and "stage1_minimum_action_std" in model_config_text
                and "fixed_view_ids=false" in stage1
            ),
            "A minimum entropy preserves support, a bounded dynamics gate "
            "prevents action erasure, and within-question InfoNCE requires "
            "sampled actions to remain identifiable in transitions.",
        ),
        check(
            "three_layer_path_structure",
            (
                float(trace.path_position_weight) > 0
                and float(trace.path_direction_weight) > 0
                and float(trace.path_step_weight) > 0
                and "trajectory_distance_components" in model
            ),
            "Position, direction, and step-scale constraints remain active.",
        ),
        check(
            "strict_answer_path_bottleneck",
            (
                str(trace.answer_context_mode) == "path_only"
                and bool(trace.require_path_bottleneck)
                and "include_question=self.answer_reads_question" in model
                and "build_path_bottleneck_mask" in model
                and "strict_answer_bottleneck=true" in stage1
                and "question_KV_masked" in stage1
            ),
            "Answer queries cannot read question K/V and must use the full "
            "eight-state latent path.",
        ),
        check(
            "exchangeable_stage1_and_deployment_consistent_risk",
            (
                int(trace.stage1_posterior_samples) == 4
                and float(trace.stage1_compact_weight) > 0
                and 0.0 < float(trace.stage1_deployment_risk_mix) < 1.0
                and "map_outputs" not in (
                    model[
                        model.index("def forward(self, batch)") :
                        model.index("def _rollout_micro_batch_size")
                    ]
                )
                and "direct_local_indices = torch.randint" in model
                and "deployment_outputs = self._stage1_trajectory_latents"
                in model
                and "deterministic=True" in model
                and "privileged_sampled_paths_per_question=0" in stage1
                and "canonical_deployment_path_is_rollout_member=false" in stage1
                and "canonical_deployment_path_task_supervision=true" in stage1
                and "all_4_exchangeable_paths_complete_trace" in stage1
            ),
            "All four IID posterior paths remain exchangeable, while the exact "
            "question-only deterministic deployment path receives explicit "
            "task and structure risk without becoming a sampled mode.",
        ),
        check(
            "compact_targets_match_deployment_budget",
            (
                int(trace.compact_target_max_new_tokens)
                == int(config.hybrid_generation_config.max_new_tokens)
                and str(trace.stage1_target_mode)
                == "complete_answer_causal_arithmetic_trace"
                and "answer_causal_equation_slice" in model
                and "is_numerically_valid_equation" in model
                and "_complete_computation_trace_target" in model
                and "visible_targets_define_route_identity=false" in stage1
                and "numerically_invalid_equations" in target_audit
                and "trace_stage1_target_audit.py" in pipeline
                and "trace_stage1_compact_target_tokens_max" in model
            ),
            "Every exchangeable path receives the same answer-causal "
            "numerically validated computation trace compiled from the single "
            "original CoT, within the deployed generation budget.",
        ),
        check(
            "exact_stage1_answer_microbatch",
            (
                int(trace.stage1_sampled_answer_micro_batch_size) == 1
                and "_teacher_force_bottleneck_chunk" in model
                and "chunk_loss * chunk_tokens" in model
                and "token_counts" in model
                and (
                    "sampled_answer_decoder_micro_batch_size="
                    "1_exact_token_weighted_CE"
                )
                in stage1
            ),
            "Stage-1 exchangeable paths are decoded one at a time, then "
            "recombined with the exact valid-token weighting of the original "
            "full-batch causal cross entropy.",
        ),
        check(
            "four_rank_gpu_only_stage1",
            (
                not bool(trace.stage1_posterior_activation_offload)
                and int(trace.stage1_offload_cache_release_interval) == 0
                and "STAGE1_WORLD_SIZE=${STAGE1_WORLD_SIZE:-4}" in stage1
                and "stage1_accumulate_grad_batches=$((4 / STAGE1_WORLD_SIZE))"
                in stage1
                and "posterior_saved_activation_offload=disabled" in stage1
                and "stage1_memory_placement=gpu_only" in stage1
                and "trajectory_activation_checkpoint="
                "gpu_recompute_preserve_rng" in stage1
                and "answer_decoder_activation_checkpoint=gpu_recompute"
                in stage1
                and "stage1_execution=four_rank_gpu_only_ddp" in stage1
                and "--trainer trace_stage1_gpu4_dynamic" in stage1
                and "tools/isolated_gpu_ddp_entry.py" in stage1
                and "ddp_launcher=torchrun_four_rank" in stage1
                and "ddp_gpu_visibility=one_physical_gpu_per_rank" in stage1
                and (
                    "ddp_strategy="
                    "standard_dynamic_graph_gradient_bucket_views"
                )
                in stage1
                and (
                    "trace_policy_config."
                    "stage1_posterior_activation_offload=false"
                )
                in stage1
                and bool(trace.stage1_answer_activation_checkpoint)
                and bool(trace.stage1_trajectory_activation_checkpoint)
            ),
            "Formal Stage 1 uses four GPU DDP ranks at local batch one with "
            "standard dynamic-graph torchrun DDP with one visible physical "
            "GPU per rank, CPU activation offload disabled, and activations "
            "recomputed on GPU during backward.",
        ),
        check(
            "stage2_has_no_cot",
            (
                "cot_visible_in_stage2=false" in stage2
                and "trace_rl_training_step" in model
                and "_decode_single_gold_cots" not in (
                    model[
                        model.index("def trace_rl_training_step") :
                        model.index("def read_generate_with_trajectory")
                    ]
                )
            ),
            "Stage 2 rolls out and updates only the question-conditioned prior.",
        ),
        check(
            "outcome_relations_and_hard_negative",
            (
                "mine_question_local_hard_pairs" in model
                and "correct_peer_index" in module
                and "has_correct_peer" in module
                and "wrong_index" in module
                and "local_relation_weight" in model
                and "wrong_wrong_dispersion=false" in stage2
            ),
            "When available, the nearest correct peer defines local positive "
            "structure; a singleton correct path still pairs with the nearest "
            "wrong path for counterfactual credit.",
        ),
        check(
            "transition_counterfactual_credit",
            (
                "counterfactual_action_batch" in model
                and "counterfactual_transition_credits" in model
                and "forced_action_mask" in model
                and "correct_to_wrong_and_wrong_to_correct" in stage2
                and "4_rotating_positions_per_update_covering_all_8" in stage2
                and "counterfactual_path_slot_coverage" in model
            ),
            "Bidirectional action replacement recomputes causal suffixes. "
            "Training rotates a four-position subset across all eight "
            "transitions and logs realized coverage.",
        ),
        check(
            "transition_advantages_are_position_local",
            (
                "preserve_trailing_positions=True" in module
                and "reduce_dims = (" in module
            ),
            "Rollout advantages are standardized independently at each "
            "transition, so credit for one position cannot rescale another.",
        ),
        check(
            "stationary_stage1_path_only_scorer",
            (
                'decoder_role="stage1_path_value"' in model
                and "self._activate_path_adapter()" in model
                and (
                    "frozen_stage1_path_only_direct_answer_scorer"
                    in stage2
                )
                and "question_fixed_path_intervened" in stage2
            ),
            "Counterfactual scores use the frozen Stage-1 path-only channel "
            "rather than the drifting Stage-2 answer adapter.",
        ),
        check(
            "noise_floored_continuous_outcome_credit",
            (
                float(rl.dense_outcome_weight) > 0
                and "frozen_gold_scores" in model
                and "dense_outcome_advantages" in model
                and "group_standardize_with_floor" in model
                and float(rl.minimum_gold_score_std) > 0
                and float(rl.minimum_gold_score_gap) > 0
                and "score_ranked_pair_only_above_registered_numerical_floor"
                in stage2
            ),
            "Homogeneous exact-answer groups receive score-ranked credit only "
            "when frozen gold likelihood variation clears registered floors.",
        ),
        check(
            "joint_stage1_policy_trust_region",
            (
                float(rl.stage1_policy_kl_weight) > 0
                and float(rl.stage1_answer_kl_weight) > 0
                and float(rl.answer_policy_weight) == 1.0
                and float(rl.stage1_policy_kl_target) > 0
                and float(rl.stage1_answer_kl_target) > 0
                and "sampled_forward_kl" in model
                and "stage1_answer_log_probs" in model
                and "target_KL_for_latent_policy_and_answer_behavior" in stage2
            ),
            "Both deployed policy factors receive full PPO objectives and "
            "target-KL protection around Stage 1.",
        ),
        check(
            "every_transition_claim_controls_familywise_error",
            (
                "simultaneous_familywise95_low" in causal
                and "Bonferroni bootstrap percentile" in causal
                and "args.bootstrap) < 10000" in causal
                and "--bootstrap 10000" in evidence
            ),
            "The all-eight-transition claim uses at least 10,000 question "
            "bootstrap draws and family-wise 95% Bonferroni intervals.",
        ),
        check(
            "real_trajectory_policy_optimization",
            (
                bool(rl.use_trajectory_policy_loss)
                and bool(rl.use_answer_policy_loss)
                and int(rl.policy_update_epochs) >= 1
                and 'rollout["policy_states"]' in model
                and "current_log_probs" in model
                and "self._trajectory_latents(" not in model[
                    model.index("def _trajectory_policy_update") :
                    model.index("def _answer_policy_update")
                ]
                and "action_ratio_deviation_final_update" in model
            ),
            "Stage 2 performs clipped updates on both policies and recomputes "
            "action density on saved on-policy states without another LLM "
            "trajectory pass.",
        ),
        check(
            "immutable_stage1_prior",
            (
                "stage1_policy_reference" in model
                and "set_stage2_trainability" in model
                and "trace_stage1_policy_reference" in model
            ),
            "Stage 2 may redistribute path probability while preserving "
            "Stage-1 action semantics and answer behavior.",
        ),
        check(
            "versioned_stage_handoff",
            (
                'trace_policy_version = "TRACE-Policy-v3"' in model
                and 'checkpoint["trace_policy_version"]' in model
                and '"TRACE-Policy-v3"' in stage2
            ),
            "Checkpoint metadata prevents legacy fixed-view and the failed "
            "posterior-only v2 state from entering the v3 Stage-2 pipeline.",
        ),
        check(
            "fresh_warm_start",
            (
                "old_checkpoint_reused=false" in stage0
                and "--load_ckpt_path" not in stage0
                and "generic_cot_sft_warm_start" in stage0
                and "--load_ckpt_path \"${stage0_checkpoint}\"" in stage1
                and "src.models.cot.LitCot" in stage1
                and "Stage 0 used an unregistered dataset" in stage1
                and "Stage 0 was not trained from the fresh base model"
                in stage1
            ),
            "The formal protocol defines a fresh CoT warm start from the "
            "registered raw data before Stage 1; a completed checkpoint from "
            "that exact protocol may be reused without importing an older "
            "latent model.",
        ),
        check(
            "real_stage1_stress_preflight",
            (
                "trace_policy_mechanism_smoke.py" in pipeline
                and "--stage0-checkpoint" in pipeline
                and "--stress-cases 3" in pipeline
                and "stage1_stress_preflight.json" in pipeline
            ),
            "Before Stage 1, the exact Stage-0 checkpoint must pass a "
            "long-example real-model forward/backward/AdamW/deployment stress "
            "preflight.",
        ),
        check(
            "complete_matched_stage2_budget",
            (
                "trainer.max_epochs=3" in stage0
                and (
                    "trainer.max_epochs=10" in stage1
                    or (
                        "STAGE1_MAX_EPOCHS=${STAGE1_MAX_EPOCHS:-10}"
                        in stage1
                        and 'trainer.max_epochs="${target_max_epochs}"'
                        in stage1
                        and "next_completed != target_max_epochs" in stage1
                        and "process_recycling=one_full_epoch_per_process"
                        in stage1
                    )
                )
                and "trainer.max_epochs=10" in stage2
                and "trainer.limit_train_batches=1.0" in stage1
                and "trainer.limit_train_batches=512" in stage2
                and "scheduler.num_training_steps=16820" in stage1
                and "scheduler.num_training_steps=5120" in stage2
                and "n_train_samples_per_epoch=2048" in stage2
                and "policy_update_epochs=1" in stage2
                and "full_validation_every_epoch=true" in stage1
                and "full_validation_every_epoch=true" in stage2
                and "test_times=1" in stage2
            ),
            "Stage 2 uses the old answer-only control's 2,048-question, "
            "one-update budget for ten epochs while retaining the intact 6,726-"
            "question source split and full validation.",
        ),
        check(
            "stage_handoff_matches_dynamic_compressor",
            (
                "corridor_progress_logits" not in stage1
                and "corridor_progress_logits" not in stage2
                and (
                    "compression_schedule="
                    "sampled_action_conditioned_strictly_monotone"
                )
                in stage1
                and '"trajectory_policy."' in stage2
                and '"trajectory_posterior."' in stage2
                and '"transition_action_decoder."' in stage2
            ),
            "The Stage-1 manifest records action-conditioned compression and "
            "the Stage-2 checkpoint gate accepts exactly the new mechanism.",
        ),
        check(
            "single_path_deployment",
            (
                "deterministic=True" in model
                and "conditional_policy_mean" in model
                and int(config.readcot_config.n_latents) == 8
            ),
            "Inference uses one eight-state conditional-mean path without "
            "the training CoT, a full verbal CoT, verifier, or best-of-N search.",
        ),
        check(
            "complete_evidence_suite",
            (
                "test_times=1" in evidence
                and "geometry_questions=200" in evidence
                and "GSMHard,SVAMP,MultiArith" in evidence
                and "trace_policy_geometry_summary.py" in evidence
                and "trace_policy_causal_summary.py" in evidence
                and "trace_policy_stage_comparison.py" in evidence
                and "trace_final/data" not in evidence
            ),
            "Best-checkpoint evidence includes GSM8K, three OOD sets, shared "
            "global PCA, heatmaps, 200-question geometry, and interventions.",
        ),
        check(
            "formation_evidence_matches_dynamic_compressor",
            (
                "rollout_actions" in geometry
                and "corridor_progress_centers" in geometry
                and "formation_geometry" in geometry
                and "action_path_permutation_null" in geometry
                and "progress_schedule_diversity" in geometry
                and "action_path_distance_correlation" in geometry
                and "plot_formation_evidence" in geometry
            ),
            "The 200-question suite directly tests path-specific monotone "
            "progress schedules and action-to-trajectory coupling against a "
            "within-question permutation null.",
        ),
    ]
    passed = sum(item["status"] == "PASS" for item in checks)
    return {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "scope": (
            "Architecture, data provenance, and execution contract. Empirical "
            "accuracy and geometry claims remain pending formal training."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "run_outputs/contract_audit",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_audit()
    output = args.output_dir / "contract_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
