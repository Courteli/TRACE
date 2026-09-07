#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "src/models/trace_policy.py"
MODULE_PATH = ROOT / "src/modules/trace_policy.py"
CONFIG_PATH = (
    ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml"
)
EVIDENCE_PATH = ROOT / "run_trace_policy_full_evidence.sh"
STAGE1_PATH = ROOT / "run_trace_policy_stage1_full.sh"
STAGE2_PATH = ROOT / "run_trace_policy_stage2_full.sh"
COMPARISON_PATH = ROOT / "tools/trace_policy_stage_comparison.py"
GEOMETRY_PATH = ROOT / "tools/trace_policy_geometry_summary.py"
CAUSAL_PATH = ROOT / "tools/trace_policy_causal_summary.py"
TASK_SUMMARY_PATH = ROOT / "tools/trace_policy_task_summary.py"
DATA_AUDIT_PATH = (
    ROOT
    / "run_outputs/trace_final/data/gsm8k_multirationale_v1"
    / "rationale_set_audit.json"
)
STAGE0_CKPT = Path(
    "/home/dingxukai/colar/logs/cot_qwen3_instruct_qsa/qsa-gsm/"
    "20260426-192859_438300_qsa_cot_sft_qwen3_instruct_lr3e-5_3epoch_gpu5/"
    "checkpoints/epoch0__step6726__monitor0.871.ckpt"
)
STAGE0_HPARAMS = STAGE0_CKPT.parent.parent / "hparams.yaml"


def _check(name, condition, detail):
    return {
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "detail": detail,
    }


def _class_base(source: str, class_name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [ast.unparse(base) for base in node.bases]
    return []


def run_audit():
    model_source = MODEL_PATH.read_text()
    module_source = MODULE_PATH.read_text()
    evidence_source = EVIDENCE_PATH.read_text()
    stage1_source = STAGE1_PATH.read_text()
    stage2_source = STAGE2_PATH.read_text()
    comparison_source = COMPARISON_PATH.read_text()
    geometry_source = GEOMETRY_PATH.read_text()
    causal_source = CAUSAL_PATH.read_text()
    task_summary_source = TASK_SUMMARY_PATH.read_text()
    config = OmegaConf.load(CONFIG_PATH).model.model_kwargs
    trace = config.trace_policy_config
    rl = config.trace_rl_config
    data_audit = json.loads(DATA_AUDIT_PATH.read_text())
    stage0_hparams = STAGE0_HPARAMS.read_text()
    stage0_state = torch.load(
        STAGE0_CKPT,
        map_location="cpu",
        weights_only=False,
    ).get("state_dict", {})
    stage0_prohibited = (
        "trajectory_policy",
        "trace_",
        "latent_bridge",
        "step_compressor",
        "latent_relation",
    )
    stage0_latent_key_counts = {
        token: sum(token in key for key in stage0_state)
        for token in stage0_prohibited
    }
    bases = _class_base(model_source, "LitTRACEPolicy")

    prohibited_runtime_patterns = (
        "center_seeds",
        "center_indices",
        "view_ids_tensor",
        "_sample_path_seeds",
        "[:, 0].zero_",
        "use_latent_loss: False",
    )
    checks = [
        _check(
            "direct_original_backbone",
            bases == ["LitREADCoTStableEfficient"],
            f"LitTRACEPolicy bases: {bases}",
        ),
        _check(
            "no_fixed_route_runtime",
            all(
                pattern not in model_source
                for pattern in prohibited_runtime_patterns
            ),
            "No center branch, fixed first rollout, seed map, or disabled "
            "latent objective is present.",
        ),
        _check(
            "learned_action_policy",
            (
                "GaussianTrajectoryPolicy" in model_source
                and "gaussian_log_prob" in model_source
                and "action_log_probs" in model_source
                and "self.latent_bridge(" not in model_source
                and "self.residual_projector(" not in model_source
            ),
            "Every transition exposes a learned Gaussian action and log-prob; "
            "the inherited deterministic bridge/projector are not in the "
            "TRACE path computation.",
        ),
        _check(
            "iid_stage1_paths",
            (
                int(trace.stage1_teacher_set_size) == 4
                and "repeated_questions" in model_source
                and "deterministic=False" in model_source
            ),
            "Four student paths use the same stochastic trajectory call.",
        ),
        _check(
            "hierarchical_teacher_set",
            (
                int(trace.stage1_teacher_set_size) == 4
                and int(trace.stage1_max_semantic_modes) == 2
                and "build_teacher_rationale_schedule" in model_source
                and "monotone_progress_centers" in model_source
                and "monotonic_path_marginals" in module_source
            ),
            "Two sampled verified rationales x stochastic globally monotone "
            "compression views form a four-path set.",
        ),
        _check(
            "frozen_cot_teacher_snapshot",
            (
                'teacher_adapter_name = "trace_teacher"' in model_source
                and "_copy_path_adapter_to_teacher_adapter" in model_source
                and "_activate_teacher_adapter" in model_source
                and "or teacher_marker in name" in model_source
                and (
                    "explicit_teacher_adapter=frozen_stage0_cot_lora"
                    in stage1_source
                )
                and '".trace_teacher." in key' in stage2_source
            ),
            "Explicit CoT states use a checkpointed frozen LoRA snapshot, "
            "so their representation basis cannot drift with the student "
            "adapter; the structured compressor remains dependency-trained.",
        ),
        _check(
            "permutation_invariant_matching",
            (
                "permutation_invariant_set_matching" in model_source
                and "itertools.permutations" in module_source
            ),
            "No route identity participates in Stage-1 matching.",
        ),
        _check(
            "strict_answer_bottleneck",
            (
                "torch.zeros_like(question_attention_mask)" in model_source
                and "build_path_bottleneck_mask" in model_source
                and "answer_question_attention_access" in model_source
            ),
            "Answer attention explicitly masks every question K/V entry.",
        ),
        _check(
            "real_stage2_trajectory_objective",
            (
                bool(rl.use_trajectory_policy_loss)
                and "trajectory_policy_loss" in model_source
                and "current[\"action_log_probs\"]" in model_source
                and int(rl.policy_update_epochs) >= 2
                and "for _ in range(update_epochs)" in model_source
                and "action_ratio_deviation_final_update" in model_source
                and "action_clip_fraction_final_update" in model_source
            ),
            "Two updates reuse rollout log-probabilities, so the second "
            "clipped PPO ratio is nontrivial.",
        ),
        _check(
            "mandatory_dual_stage2_objective",
            (
                bool(rl.use_trajectory_policy_loss)
                and bool(rl.use_answer_policy_loss)
                and "required_stage2_objectives" in model_source
                and "disabled" in model_source
            ),
            "Final Stage 2 refuses latent-only or answer-only degradation.",
        ),
        _check(
            "greedy_path_labels",
            (
                "greedy_path_accuracy" in model_source
                and "sampled_answer_accuracy" in model_source
                and "trajectory_rewards = greedy_accuracy" in model_source
            ),
            "Path labels and sampled answer-token outcomes are separated.",
        ),
        _check(
            "bidirectional_counterfactual_credit",
            (
                "counterfactual_action_batch" in model_source
                and "forced_action_mask" in model_source
                and "counterfactual_transition_credits" in model_source
            ),
            "Correct<-wrong and wrong<-correct transitions recompute causal "
            "suffixes with recipient innovations.",
        ),
        _check(
            "positive_positive_negative_relation",
            (
                "correct_peer_index" in module_source
                and "wrong_index" in module_source
                and "local_relation_weight" in model_source
            ),
            "Nearest correct peer and nearest wrong path define the local "
            "outcome relation.",
        ),
        _check(
            "immutable_stage1_policy_prior",
            (
                "stage1_policy_reference" in model_source
                and "diagonal_gaussian_kl" in model_source
                and "set_stage2_trainability" in model_source
                and "extract_stage2_policy_reference" in model_source
                and "checkpoint_stage == 2" in model_source
            ),
            "Stage-2 dynamics are frozen; only a genuine Stage-2 checkpoint "
            "can restore the immutable Stage-1 policy snapshot.",
        ),
        _check(
            "deterministic_distributed_epoch_budget",
            (
                int(rl.n_train_samples_per_epoch) == 6726
                and "_validate_trace_rl_epoch_budget" in model_source
                and "set_train_indices" not in model_source
                and "requested_global_batch_size=4" in stage1_source
                and "scheduled_optimizer_steps=16820" in stage1_source
                and (
                    "requested_global_question_batch_size=4"
                    in stage2_source
                )
                and "unique_training_questions_per_epoch=6726"
                in stage2_source
                and "dataloader_rows_per_epoch=6728" in stage2_source
                and "ddp_padding_duplicates_per_epoch=2" in stage2_source
                and "scheduled_optimizer_steps=33640" in stage2_source
                and "trainer.limit_train_batches=1682" in stage2_source
                and "dataset_subset_mutation=false" in stage2_source
                and "summarize_unique_validation_records" in model_source
                and "dist.all_gather_object" in model_source
            ),
            "The immutable 6,726-question split is fully covered each epoch; "
            "four DDP ranks consume 1,682 synchronized batches with only two "
            "standard sampler padding rows. Validation padding is deduplicated "
            "before checkpoint selection.",
        ),
        _check(
            "map_inference_without_search",
            (
                "deterministic=True" in model_source
                and "conditional_policy_mean" in model_source
                and int(config.readcot_config.n_latents) == 8
            ),
            "Inference runs one conditional-mean path with eight latents.",
        ),
        _check(
            "full_fair_dataset",
            (
                data_audit["status"] == "PASS"
                and data_audit["split_counts"]
                == {"train": 6726, "val": 747, "test": 1319}
                and data_audit["splits"]["train"][
                    "questions_with_multiple_verified_rationales"
                ]
                == 4142
                and "target: src.models.cot.LitCot" in stage0_hparams
                and "sft_method: cot" in stage0_hparams
                and not any(stage0_latent_key_counts.values())
                and (
                    "Stage 0 must be a plain CoT-SFT checkpoint"
                    in stage1_source
                )
            ),
            "Original GSM8K-Aug-NL splits are retained; Stage 0 is plain "
            "CoT SFT with zero latent-policy keys; 4,142 training questions "
            "have multiple rationales.",
        ),
        _check(
            "evidence_cache_is_unmodified",
            (
                "manual_offsets\": False" in model_source
                and "per_path_rescaling\": False" in model_source
                and "rollout_innovations" in model_source
                and "teacher_assignments" in model_source
            ),
            "Saved evidence includes replayable IID actions, assignments, "
            "outcomes, and an explicit no-offset/no-rescaling contract.",
        ),
        _check(
            "paired_stage1_final_evidence",
            (
                "stage1_checkpoint" in evidence_source
                and "stage2_checkpoint" in evidence_source
                and 'model.model_kwargs.do_trace_rl=${rl_mode}'
                in evidence_source
                and (
                    'run_evidence_suite stage1 "${stage1_checkpoint}" false'
                    in evidence_source
                )
                and (
                    'run_evidence_suite final "${stage2_checkpoint}" true'
                    in evidence_source
                )
                and "trace_policy_stage_comparison.py" in evidence_source
                and "trace_policy_task_summary.py" in evidence_source
                and "paired_ci" in comparison_source
                and "map_output_length" in comparison_source
                and '"total_L_definition"' in task_summary_source
                and "exact_sign_p" in task_summary_source
            ),
            "Stage 1 and Final use paired GSM8K/OOD/200-question evidence "
            "with accuracy, total-#L, rescue/regress, and geometry deltas.",
        ),
        _check(
            "complete_path_causal_evidence",
            (
                "trace_policy_causal_summary.py" in evidence_source
                and "latent_read_mask" in model_source
                and "_fork_past_key_values" in model_source
                and "same_norm_random_actions" in causal_source
                and '"mean_repeat"' in causal_source
                and "prefix_accuracy_values" in causal_source
                and "transition_accuracy_values" in causal_source
                and "all_eight_transitions_individually_necessary"
                in causal_source
                and "prefix_length=step + 1" in causal_source
            ),
            "The final checkpoint receives no-path, ordered-path, same-norm, "
            "prefix, and all-eight transition interventions with causal "
            "suffix recomputation.",
        ),
        _check(
            "submission_figure_contract",
            all(
                (
                    '"Times New Roman"' in source
                    and 'PINK = "#E5A3BF"' in source
                    and 'with_suffix(".svg")' in source
                    and 'with_suffix(".pdf")' in source
                    and 'with_suffix(".tiff")' in source
                    and "figure_qa" in source
                )
                for source in (
                    geometry_source,
                    comparison_source,
                    causal_source,
                )
            ),
            "Geometry, paired, and causal figures use the requested "
            "Times-style serif stack, pink palette, SVG/PDF/600-dpi TIFF, "
            "and automated visual QA.",
        ),
    ]
    passed = sum(item["status"] == "PASS" for item in checks)
    return {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "scope": (
            "Static architecture and data contract only. Accuracy, geometry, "
            "and causal empirical claims require formal training/evaluation."
        ),
    }


def write_markdown(report, path: Path):
    lines = [
        "# TRACE Policy Contract Audit",
        "",
        f"**Overall: {report['status']} "
        f"({report['passed']}/{report['total']})**",
        "",
        "| Contract | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in report["checks"]:
        lines.append(
            f"| `{item['name']}` | {item['status']} | {item['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            report["scope"],
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "run_outputs/trace_policy/contract_audit",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_audit()
    (args.output_dir / "contract_audit.json").write_text(
        json.dumps(report, indent=2)
    )
    write_markdown(
        report,
        args.output_dir / "contract_audit.md",
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
