import ast
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from src.models.trace_vb import (
    LitTRACEVB,
    Stage1HostMemoryGuard,
    Stage1RecoveryCheckpoint,
    _host_memory_guard_reasons,
    _parse_linux_memory_gib,
    build_path_bottleneck_mask,
    summarize_unique_validation_records,
)
from src.modules.trace_vb import (
    checkpointed_projected_token_cross_entropy,
    per_sequence_token_cross_entropy,
)


class QuestionCommitBridgeContractTests(unittest.TestCase):
    def test_formal_mask_exposes_question_and_commit_only(self):
        question = torch.tensor([[1, 1, 0]], dtype=torch.long)
        latent = torch.ones((1, 8), dtype=torch.long)
        commit_only = LitTRACEVB._commit_only_latent_mask(latent)
        answer = torch.tensor([[1, 1]], dtype=torch.long)
        mask = build_path_bottleneck_mask(
            question,
            commit_only,
            answer,
            include_question=True,
        )
        self.assertTrue(torch.equal(mask[:, :3], question))
        self.assertTrue(
            torch.equal(
                mask[:, 3:11],
                torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]]),
            )
        )
        self.assertTrue(torch.equal(mask[:, 11:], answer))

    def test_full_validation_summary_detects_constant_prediction(self):
        shards = [
            [
                (0, 1.0, 5, '[["10"]]', True, True),
                (1, 0.0, 5, '[["10"]]', True, True),
            ],
            [
                (2, 0.0, 5, '[["10"]]', True, True),
                (3, 0.0, 5, '[["10"]]', True, True),
            ],
        ]
        summary = summarize_unique_validation_records(
            shards,
            expected_count=4,
        )
        self.assertEqual(summary["unique_questions"], 4.0)
        self.assertEqual(summary["unique_predictions"], 1.0)
        self.assertEqual(summary["unique_prediction_ratio"], 0.25)
        self.assertEqual(summary["top1_mode_fraction"], 1.0)
        self.assertEqual(summary["valid_answer_fraction"], 1.0)


class SufficiencyConsumptionContractTests(unittest.TestCase):
    @staticmethod
    def _minimal_model(row):
        model = LitTRACEVB.__new__(LitTRACEVB)
        pl.LightningModule.__init__(model)
        model.n_trace_steps = 8
        model._sufficiency_cache_validated = True
        model._sufficiency_cache = {"by_idx": {0: row}}
        return model

    def test_short_cot_skips_empty_slots_without_losing_next_gain(self):
        question = "What is two plus three?"
        row = {
            "idx": 0,
            "question_sha256": hashlib.sha256(
                question.encode("utf-8")
            ).hexdigest(),
            "n_steps": 2,
            "role_scores": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.8, 1.0],
            "role_valid_mask": [True, True, True, False, False, False, True, True],
            "role_source_prefix_index": [-1, -1, 0, None, None, None, 1, 1],
        }
        model = self._minimal_model(row)
        targets, mask, importance = model._batch_sufficiency_targets(
            {"idx": torch.tensor([0]), "question": [question]},
            [[(0, 1), (1, 1), (1, 1), (1, 1), (1, 2)]],
        )
        self.assertTrue(torch.equal(mask[0, :2], torch.ones(2, dtype=torch.bool)))
        self.assertTrue(torch.equal(mask[0, 3:6], torch.zeros(3, dtype=torch.bool)))
        self.assertFalse(bool(mask[0, 7]))
        self.assertAlmostEqual(float(targets[0, 6]), 0.8)
        self.assertAlmostEqual(float(importance[0, 0]), 0.2)
        self.assertAlmostEqual(float(importance[0, 4]), 0.6)

    def test_cache_alignment_divergence_fails_closed(self):
        question = "Q"
        row = {
            "idx": 0,
            "question_sha256": hashlib.sha256(
                question.encode("utf-8")
            ).hexdigest(),
            "n_steps": 1,
            "role_scores": [0.0] * 8,
            "role_valid_mask": [True] * 8,
            "role_source_prefix_index": [-1, -1, 0, None, None, None, 0, 0],
        }
        model = self._minimal_model(row)
        with self.assertRaisesRegex(RuntimeError, "alignment diverged"):
            model._batch_sufficiency_targets(
                {"idx": torch.tensor([0]), "question": [question]},
                [[(0, 0), (0, 0), (0, 0), (0, 0), (0, 1)]],
            )


class SolveTextLeakageContractTests(unittest.TestCase):
    def test_targets_use_only_same_sample_gold_steps(self):
        records = LitTRACEVB._solve_text_chunk_records(
            [
                {
                    "steps": ["2 + 3 = 5"],
                    "answer": "DO_NOT_READ_THIS_ANSWER_FIELD",
                }
            ],
            [[(0, 0), (0, 0), (0, 0), (0, 0), (0, 1)]],
        )
        self.assertEqual(records, [(0, 4, "2 + 3 = 5")])
        self.assertNotIn("DO_NOT_READ", records[0][2])

    class _ByteTokenizer:
        eos_token = "<eos>"
        eos_token_id = 256
        bos_token_id = 257
        pad_token_id = 258

        @staticmethod
        def encode(text, add_special_tokens=False):
            if add_special_tokens:
                raise AssertionError("test tokenizer forbids special tokens")
            return list(str(text).encode("utf-8"))

    @classmethod
    def _minimal_text_model(cls, max_tokens=96):
        model = LitTRACEVB.__new__(LitTRACEVB)
        pl.LightningModule.__init__(model)
        model.tokenizer = cls._ByteTokenizer()
        model.trace_config = {
            "solve_text_decoder_max_tokens": max_tokens,
            "solve_text_decoder_fail_on_truncation": True,
        }
        return model

    def test_token_targets_cover_each_sample_once_across_all_five_roles(self):
        model = self._minimal_text_model()
        texts = [
            "alpha beta gamma delta epsilon zeta",
            "one two three four five six seven",
        ]
        gold_cots = [
            {"steps": [text], "answer": "DO_NOT_READ_THIS_ANSWER_FIELD"}
            for text in texts
        ]
        spans = [
            [(0, 0), (0, 0), (0, 0), (0, 0), (0, 1)],
            [(0, 0), (0, 0), (0, 0), (0, 0), (0, 1)],
        ]

        records = model._solve_text_token_records(gold_cots, spans)

        self.assertEqual(len(records), 10)
        for sample_index, text in enumerate(texts):
            sample_records = [
                record for record in records if record[0] == sample_index
            ]
            self.assertEqual([record[1] for record in sample_records], list(range(5)))
            reconstructed = [
                token
                for _, _, chunk in sample_records
                for token in chunk
            ]
            self.assertEqual(reconstructed, model.tokenizer.encode(text))

    def test_token_target_bound_fails_closed_instead_of_truncating(self):
        model = self._minimal_text_model(max_tokens=4)
        with self.assertRaisesRegex(RuntimeError, "refusing silent CoT truncation"):
            model._tokenize_solve_text_targets([[1, 2, 3, 4]])

    def test_noncontiguous_or_cross_sample_span_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "sample-local"):
            LitTRACEVB._solve_text_chunk_records(
                [{"steps": ["a", "b"]}],
                [[(0, 1), (0, 0), (0, 0), (0, 0), (0, 1)]],
            )

    def test_compact_formatter_retains_final_calculation_under_char_budget(self):
        model = self._minimal_text_model()
        model.readcot_config = {
            "anchor_text_mode": "compact_equation",
            "compact_anchor_max_chars": 32,
        }
        compact = model._format_anchor_step(
            "First find risk: 100% - 40% = 60% "
            "Finally multiply: 60% * 40% = 24%"
        )
        self.assertLessEqual(len(compact), 32)
        self.assertTrue(compact.endswith("60 * 40 = 24"))

    def test_compact_budget_protects_final_observed_equation(self):
        model = self._minimal_text_model()
        model.trace_config["compact_target_max_new_tokens"] = 32
        model.model_kwargs = SimpleNamespace(
            hybrid_generation_config=SimpleNamespace(max_new_tokens=32)
        )
        model.anchor_header = "Anchors:"
        model.thinking_separator = "###"
        model.answer_template = "Answer:{}"
        target = (
            "Anchors:\n"
            "- this early explanation is deliberately far too verbose\n"
            "- 8*7=56\n"
            "###Answer:56"
        )
        fitted = model._fit_compact_target_to_generation_budget(target, "56")
        self.assertIn("- 8*7=56", fitted)
        self.assertIn("###Answer:56", fitted)
        self.assertLessEqual(model._target_token_count(fitted), 32)


class SolveTextChunkedProjectionTests(unittest.TestCase):
    def test_chunked_projection_matches_full_loss_and_hidden_gradient(self):
        torch.manual_seed(7)
        hidden = torch.randn(2, 5, 4, dtype=torch.float64)
        weight = torch.randn(11, 4, dtype=torch.float64)
        bias = torch.randn(11, dtype=torch.float64)
        labels = torch.tensor(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            dtype=torch.long,
        )
        mask = torch.tensor(
            [[True, True, True, False, False], [True] * 5]
        )

        full_hidden = hidden.clone().requires_grad_(True)
        full_loss = per_sequence_token_cross_entropy(
            F.linear(full_hidden, weight, bias),
            labels,
            mask,
            token_chunk_size=2,
        )
        full_gradient = torch.autograd.grad(
            full_loss.sum(),
            full_hidden,
        )[0]

        projected_token_counts = []
        chunked_hidden = hidden.clone().requires_grad_(True)

        def project(states):
            projected_token_counts.append(int(states.shape[1]))
            return F.linear(states, weight, bias)

        chunked_loss = checkpointed_projected_token_cross_entropy(
            chunked_hidden,
            labels,
            mask,
            project,
            token_chunk_size=2,
        )
        chunked_gradient = torch.autograd.grad(
            chunked_loss.sum(),
            chunked_hidden,
        )[0]

        torch.testing.assert_close(chunked_loss, full_loss)
        torch.testing.assert_close(chunked_gradient, full_gradient)
        self.assertTrue(projected_token_counts)
        self.assertLessEqual(max(projected_token_counts), 2)


class TeacherDeploymentPathContractTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = (
            cls.root
            / "src/configs/models/trace_vb_policy_qwen3_instruct.yaml"
        ).read_text(encoding="utf-8")
        cls.launcher = (
            cls.root / "scripts/run_stage1_vb.sh"
        ).read_text(encoding="utf-8")
        cls.stage2_launcher = (
            cls.root / "scripts/run_stage2_vb.sh"
        ).read_text(encoding="utf-8")
        cls.supervisor = (
            cls.root / "scripts/wait_for_four_gpus_and_run_full_vb.sh"
        ).read_text(encoding="utf-8")
        cls.source = (
            cls.root / "src/models/trace_vb.py"
        ).read_text(encoding="utf-8")

    def test_formal_config_keeps_one_teacher_and_one_deployment_path(self):
        for token in (
            "deployment_compact_reasoning: true",
            "use_capability_anchor: true",
            "capability_expected_lora_tensors: 504",
            "stage1_posterior_samples: 1",
            "stage1_posterior_kl_weight: 0.05",
            "stage1_sampled_answer_weight: 0.0",
            "stage1_map_compact_weight: 1.0",
            "stage1_map_answer_suffix_weight: 0.50",
            "stage1_capability_kl_weight: 0.25",
            "stage1_solve_text_weight: 0.05",
            "stage1_commit_weight: 0.10",
            "stage1_answer_activation_checkpoint: true",
            "anchor_text_mode: compact_equation",
            "compact_anchor_max_chars: 32",
            "commit_causal_summary: true",
        ):
            self.assertIn(token, self.config)
            self.assertIn(token.replace(": ", "="), self.launcher)

    def test_registered_capability_parity_components_are_fail_closed(self):
        for token in (
            "capability_trace_view_scale: 0.30",
            "capability_trace_step_view_scale: 0.15",
            "capability_anchor_gate_scale: 0.50",
            "stage0_expected_lora_tensors: 504",
        ):
            self.assertIn(token, self.config)
            self.assertIn(token.replace(": ", "="), self.launcher)
        for token in (
            '"anchor_gate_predictor.0.weight"',
            '"trace_view_embeddings.weight"',
            '"trace_step_view_embeddings.weight"',
            '"capability_trace_view"',
            '"capability_trace_step_views"',
            "def load_cot_encoder_state_dict",
            "capability_teacher_all_roles",
        ):
            self.assertIn(token, self.source)
        run_source = (self.root / "run.py").read_text(encoding="utf-8")
        self.assertIn("--cot_encoder_ckpt_path", run_source)
        self.assertIn("load_cot_encoder_state_dict", run_source)
        preflight = (
            self.root / "scripts/run_preflight_validation_v7.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("exact_correct=540", preflight)
        self.assertIn("minimum_correct=449", preflight)
        self.assertIn("validation_path=capability_teacher_all_roles", preflight)

    def test_capability_parity_uses_historical_full_prefill_and_dtype_order(self):
        trajectory_start = self.source.index(
            "    def _capability_teacher_trajectory("
        )
        trajectory_end = self.source.index(
            "    def _trajectory_latents(", trajectory_start
        )
        trajectory = self.source[trajectory_start:trajectory_end]
        for token in (
            "base_latent + query_scale * latent_queries",
            "latent_inputs + view_scale * view_embeds",
            "latent_inputs + step_view_scale * step_view_embeds",
            '"context_inputs_embeds": context_inputs_embeds',
        ):
            self.assertIn(token, trajectory)
        self.assertNotIn("previous_state.float()", trajectory)

        generation_start = self.source.index(
            "    def _generate_capability_answers_from_trajectory("
        )
        generation_end = self.source.index(
            "    def _answer_token_log_probs(", generation_start
        )
        generation = self.source[generation_start:generation_end]
        self.assertIn("inputs_embeds=all_inputs_embeds", generation)
        self.assertIn('trajectory_outputs["context_inputs_embeds"]', generation)
        self.assertNotIn("past_key_values=", generation)
        self.assertNotIn("position_ids=", generation)

        student_start = self.source.index("    def _trajectory_latents(")
        student_end = self.source.index(
            "    def _prompt_ids_for_answer(", student_start
        )
        student = self.source[student_start:student_end]
        for token in (
            "capability_base + step_scale * step_prior",
            "anchored_input + view_scale * view_prior",
            "anchored_input + step_view_scale * step_view_prior",
            "anchored_input.float()",
            "action_scale * action_residual.float()",
        ):
            self.assertIn(token, student)
        self.assertNotIn("capability_question_state.float()", student)

    def test_restarted_stage1_publishes_one_indexed_summary_per_epoch(self):
        for token in (
            "trace_vb_v7_stage1_validation_index_v1",
            "validation_summary_index.json",
            "checkpoint_run_copy",
        ):
            self.assertIn(token, self.launcher)
            self.assertIn(token, self.stage2_launcher)
        self.assertIn("expected exactly one", self.launcher)
        self.assertIn("hashlib.sha256(expected_path.read_bytes())", self.stage2_launcher)
        self.assertIn("sorted(indexed_epochs)", self.stage2_launcher)

    def test_supervisor_can_pin_gpus_and_publishes_exit_status(self):
        for token in (
            "TRACE_VB_FIXED_GPUS",
            "fixed_gpu_array",
            "exit_status_file",
            "run_status=$?",
            '"${selected}" "${selected%%,*}"',
        ):
            self.assertIn(token, self.supervisor)

    def test_supervisor_waits_for_fixed_gpus_before_guarded_recovery(self):
        for token in (
            "PIPELINE_RESUME_STAGE1_CKPT",
            "TRACE_VB_SUPERVISOR_ATTEMPT",
            "formal recovery requires TRACE_VB_FIXED_GPUS",
            'bash "${SCRIPT_DIR}/resume_full_pipeline_vb.sh"',
        ):
            self.assertIn(token, self.supervisor)

    def test_deployment_generation_cannot_read_gold_cot(self):
        start = self.source.index("    def read_generate_with_trajectory(")
        end = self.source.index("    def read_generate(", start)
        deployment = self.source[start:end]
        self.assertIn("deterministic=True", deployment.replace(" ", ""))
        self.assertNotIn("posterior_context", deployment)

        start = self.source.index("    def eval_generation(")
        end = self.source.index("    def on_validation_epoch_start(", start)
        evaluation = self.source[start:end]
        self.assertLess(
            evaluation.index("read_generate_with_trajectory"),
            evaluation.index("_decode_single_gold_cots"),
        )

    def test_compact_targets_are_sample_local_and_checkpointed(self):
        start = self.source.index("    def _build_role_compact_targets(")
        end = self.source.index("    @staticmethod\n    def _masked_role_kl", start)
        builder = self.source[start:end]
        self.assertIn(
            "zip(gold_cots, answers, solve_spans)",
            builder.replace("\n", " ").replace("  ", " "),
        )
        checkpoint = self.source[
            self.source.index("    def on_save_checkpoint("):
            self.source.index("    @staticmethod\n    def _masked_role_kl")
        ]
        self.assertIn("trajectory_posterior.", checkpoint)
        teacher_start = self.source.index("    def _teacher_force_bottleneck(")
        teacher_end = self.source.index(
            "    def _prompt_ids_for_answer(", teacher_start
        )
        teacher_force = self.source[teacher_start:teacher_end]
        self.assertIn("logits = checkpoint(", teacher_force)
        self.assertIn("use_reentrant=False", teacher_force)
        self.assertGreaterEqual(
            teacher_force.count("self._fork_past_key_values("),
            1,
        )
        self.assertIn("posterior_context_norm.", checkpoint)

    def test_commit_and_protected_answer_objectives_are_active(self):
        start = self.source.index("    def _stage1_forward_impl(")
        end = self.source.index("    def _legacy_role_forward(", start)
        active = self.source[start:end]
        for token in (
            "protected_suffixes=protected_answer_suffixes",
            'map_answer_suffix_weight * map_answer_suffix_loss',
            'capability_weight * capability_suffix_kl',
            'commit_weight * semantic_losses["commit"]',
            '"trace_vb_commit_causal_summary": total_loss.new_ones(())',
        ):
            self.assertIn(token, active)
        self.assertEqual(
            active.count("+ posterior_kl_weight * posterior_kl"),
            1,
        )
        self.assertIn("uses its normalized state directly", self.source)
        self.assertNotIn("current_state + previous_state", self.source)

    def test_stage1_metric_dictionary_has_scalar_solve_text_lookup(self):
        tree = ast.parse(self.source)
        matches = [
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant)
            and key.value == "trace_vb_solve_text_truncated_fraction"
        ]
        self.assertEqual(len(matches), 1)
        value = matches[0]
        self.assertIsInstance(value, ast.Subscript)
        self.assertIsInstance(value.value, ast.Name)
        self.assertEqual(value.value.id, "solve_text")
        self.assertIsInstance(value.slice, ast.Constant)
        self.assertEqual(value.slice.value, "truncated_fraction")



class HostMemoryGuardContractTests(unittest.TestCase):
    def test_fresh_artifact_root_exists_before_checkpoint_lookup(self):
        project_root = Path(__file__).resolve().parents[1]
        launcher = (project_root / "scripts/run_stage1_vb.sh").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            launcher.index("mkdir -p \"${log_root}\""),
            launcher.index("guard_before=$(newest_host_guard_checkpoint)"),
        )

    def test_stage1_resume_skips_completed_epochs_and_selects_canonical_best(self):
        project_root = Path(__file__).resolve().parents[1]
        launcher_text = (
            project_root / "scripts/run_stage1_vb.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'epoch_progress.get("processed")',
            launcher_text,
        )
        self.assertNotIn(
            'print(int(checkpoint.get("epoch", -1)) + 1)',
            launcher_text,
        )
        self.assertIn(
            "initial_completed_epochs=$(checkpoint_completed_epochs",
            launcher_text,
        )
        self.assertIn(
            "target_max_epochs <= initial_completed_epochs",
            launcher_text,
        )
        self.assertNotIn(
            '\"${STAGE1_RESUME_CKPT}\" \"${initial_resume_monitor}\"',
            launcher_text,
        )
        self.assertNotIn("invalid initial Stage-1 best-checkpoint candidate", launcher_text)
        self.assertIn("filename monitor disagrees with validation", launcher_text)
        self.assertIn("expected exactly one", launcher_text)
        self.assertIn("Select with the exact, unrounded JSON accuracy", launcher_text)
        self.assertIn('validation_path\") != \"student_commit\"', launcher_text)

    def test_stage1_resume_preserves_original_manifest(self):
        project_root = Path(__file__).resolve().parents[1]
        launcher_text = (
            project_root / "scripts/run_stage1_vb.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ -z "${STAGE1_RESUME_CKPT}" ]]', launcher_text)
        self.assertIn("Stage-1 recovery requires the original manifest", launcher_text)
        self.assertIn("resume_checkpoint_sha256=", launcher_text)
        self.assertIn("original Stage-1 manifest used a different", launcher_text)

    def test_vb_v5_formal_path_forbids_cpu_activation_offload(self):
        project_root = Path(__file__).resolve().parents[1]
        config_text = (
            project_root
            / "src/configs/models/trace_vb_policy_qwen3_instruct.yaml"
        ).read_text(encoding="utf-8")
        launcher_text = (
            project_root / "scripts/run_stage1_vb.sh"
        ).read_text(encoding="utf-8")
        source_text = (
            project_root / "src/models/trace_vb.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "stage1_posterior_activation_offload: false",
            config_text,
        )
        self.assertIn(
            "stage1_posterior_activation_offload=false",
            launcher_text,
        )
        self.assertIn(
            "TRACE-VB-v7 forbids saved-tensor CPU activation offload",
            source_text,
        )

    def test_proc_memory_parser_and_fail_closed_thresholds(self):
        self.assertEqual(
            _parse_linux_memory_gib(
                "Name:\tpython\nVmRSS:\t2097152 kB\n",
                "VmRSS",
            ),
            2.0,
        )
        with self.assertRaisesRegex(RuntimeError, "missing"):
            _parse_linux_memory_gib("VmSize: 1 kB", "VmRSS")

        self.assertEqual(
            _host_memory_guard_reasons(
                maximum_rank_rss_gib=6.0,
                minimum_host_available_gib=400.0,
                maximum_allowed_rank_rss_gib=20.0,
                minimum_required_host_available_gib=192.0,
            ),
            [],
        )
        reasons = _host_memory_guard_reasons(
            maximum_rank_rss_gib=20.0,
            minimum_host_available_gib=192.0,
            maximum_allowed_rank_rss_gib=20.0,
            minimum_required_host_available_gib=192.0,
        )
        self.assertEqual(len(reasons), 2)

    def test_callbacks_are_stage1_only_and_keep_both_safety_layers(self):
        model = LitTRACEVB.__new__(LitTRACEVB)
        pl.LightningModule.__init__(model)
        model.trace_config = {
            "stage1_recovery_checkpoint_interval": 200,
            "stage1_host_memory_guard_interval": 10,
            "stage1_maximum_rank_rss_gib": 20.0,
            "stage1_minimum_host_available_gib": 192.0,
        }
        model.do_trace_rl = False
        callbacks = model.configure_callbacks()
        self.assertEqual(
            [type(callback) for callback in callbacks],
            [Stage1RecoveryCheckpoint, Stage1HostMemoryGuard],
        )
        guard = callbacks[1]
        self.assertEqual(guard.every_n_train_steps, 10)
        self.assertEqual(guard.maximum_rank_rss_gib, 20.0)
        self.assertEqual(guard.minimum_host_available_gib, 192.0)

        model.do_trace_rl = True
        self.assertEqual(model.configure_callbacks(), [])


class SemanticAnchorScheduleTests(unittest.TestCase):
    @staticmethod
    def _minimal_model(batches_seen):
        model = LitTRACEVB.__new__(LitTRACEVB)
        pl.LightningModule.__init__(model)
        model.trace_rl_config = {
            "use_semantic_anchor": True,
            "semantic_anchor_initial_weight": 0.02,
            "semantic_anchor_minimum_weight": 0.01,
            "semantic_anchor_decay_batches": 100,
            "score_calibration_batches": 10,
        }
        model.register_buffer(
            "vb_rollout_batches_seen",
            torch.tensor(batches_seen, dtype=torch.long),
        )
        return model

    def test_anchor_is_off_during_warmup_then_decays_to_floor(self):
        self.assertEqual(
            self._minimal_model(9)._semantic_anchor_weight(),
            0.0,
        )
        self.assertAlmostEqual(
            self._minimal_model(10)._semantic_anchor_weight(),
            0.02,
        )
        self.assertAlmostEqual(
            self._minimal_model(110)._semantic_anchor_weight(),
            0.01,
        )


if __name__ == "__main__":
    unittest.main()
