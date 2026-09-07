import inspect
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DistributedSampler
from transformers import DynamicCache

from src.models.read_stable_efficient import LitREADCoTStableEfficient
from src.models.trace_policy import (
    LitTRACEPolicy,
    build_path_bottleneck_mask,
    extract_stage2_policy_reference,
    summarize_unique_validation_records,
)
from src.modules.trace_policy import (
    GaussianTrajectoryPolicy,
    HardPathPair,
    build_teacher_rationale_schedule,
    build_transition_advantages,
    clipped_policy_loss,
    counterfactual_action_batch,
    counterfactual_transition_credits,
    diagonal_gaussian_kl,
    gaussian_log_prob,
    mine_question_local_hard_pairs,
    monotonic_path_marginals,
    monotone_progress_centers,
    permutation_invariant_set_matching,
    stochastic_monotone_assignment,
)
from tools.trace_policy_task_summary import exact_sign_p, summarize_pair


class GaussianPolicyTests(unittest.TestCase):
    def test_log_prob_has_policy_gradient(self):
        means = torch.zeros(3, 4, requires_grad=True)
        log_stds = torch.full(
            (3, 4),
            -0.5,
            requires_grad=True,
        )
        actions = torch.ones(3, 4)
        loss = -gaussian_log_prob(actions, means, log_stds).mean()
        loss.backward()
        self.assertGreater(float(means.grad.abs().sum()), 0.0)
        self.assertGreater(float(log_stds.grad.abs().sum()), 0.0)

    def test_reference_kl_is_zero_only_for_equal_policies(self):
        means = torch.randn(2, 8, 4)
        log_stds = torch.randn(2, 8, 4).clamp(-2.0, 0.2)
        equal = diagonal_gaussian_kl(
            means,
            log_stds,
            means,
            log_stds,
        )
        shifted = diagonal_gaussian_kl(
            means + 0.5,
            log_stds,
            means,
            log_stds,
        )
        self.assertTrue(torch.allclose(equal, torch.zeros_like(equal)))
        self.assertTrue(torch.all(shifted > 0))

    def test_stage2_freezes_action_semantics(self):
        policy = GaussianTrajectoryPolicy(
            hidden_size=12,
            action_dim=4,
            n_steps=8,
            policy_hidden_size=16,
        )
        policy.set_stage2_trainability()
        trainable = {
            name
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            any(name.startswith("mean_head") for name in trainable)
        )
        self.assertTrue(
            any(name.startswith("log_std_head") for name in trainable)
        )
        self.assertFalse(
            any(name.startswith("action_projector") for name in trainable)
        )
        self.assertFalse(
            any(name.startswith("base_projector") for name in trainable)
        )

    def test_second_policy_update_has_nonunit_ratio(self):
        means = torch.zeros(2, 3, requires_grad=True)
        log_stds = torch.full((2, 3), -0.5)
        actions = torch.ones(2, 3)
        old_log_probs = gaussian_log_prob(
            actions,
            means,
            log_stds,
        ).detach()
        optimizer = torch.optim.SGD([means], lr=0.05)

        first_loss = clipped_policy_loss(
            gaussian_log_prob(actions, means, log_stds),
            old_log_probs,
            torch.ones(2),
            clip_epsilon=0.12,
        )
        first_loss.backward()
        optimizer.step()

        second_ratio = torch.exp(
            gaussian_log_prob(actions, means, log_stds)
            - old_log_probs
        )
        self.assertFalse(
            torch.allclose(second_ratio, torch.ones_like(second_ratio))
        )


class StochasticTeacherTests(unittest.TestCase):
    def test_progress_centers_are_strictly_ordered(self):
        logits = torch.zeros(4, 8)
        noise = torch.randn(4, 8)
        centers = monotone_progress_centers(
            logits,
            noise,
            noise_scale=0.7,
        )
        self.assertTrue(torch.all(centers[:, 1:] > centers[:, :-1]))
        self.assertTrue(
            torch.allclose(
                centers[:, -1],
                torch.ones(4),
                atol=1e-6,
            )
        )

    def test_progress_prior_moves_assignments_forward(self):
        semantic_scores = torch.zeros(8, 6)
        centers = torch.linspace(0.1, 1.0, 8)
        assignment = stochastic_monotone_assignment(
            semantic_scores,
            centers,
            sigma=0.12,
            progress_strength=2.0,
        )
        expected_step = (
            assignment
            * torch.arange(6, dtype=assignment.dtype).unsqueeze(0)
        ).sum(dim=-1)
        self.assertTrue(torch.all(expected_step[1:] > expected_step[:-1]))
        self.assertTrue(
            torch.allclose(
                assignment.sum(dim=-1),
                torch.ones(8),
                atol=1e-6,
            )
        )

    def test_monotone_alignment_survives_adversarial_semantics(self):
        logits = torch.full((8, 6), -8.0)
        logits[:4, 5] = 20.0
        logits[4:, 0] = 20.0
        assignment = monotonic_path_marginals(logits)
        expected_step = (
            assignment
            * torch.arange(6, dtype=assignment.dtype).unsqueeze(0)
        ).sum(dim=-1)
        self.assertTrue(
            torch.all(expected_step[1:] >= expected_step[:-1] - 1e-6)
        )
        self.assertTrue(
            torch.allclose(
                assignment.sum(dim=-1),
                torch.ones(8),
                atol=1e-6,
            )
        )

    def test_teacher_schedule_balances_selected_rationales(self):
        generator = torch.Generator().manual_seed(11)
        rationale_indices, semantic_modes = (
            build_teacher_rationale_schedule(
                n_available=4,
                set_size=4,
                max_semantic_modes=2,
                generator=generator,
            )
        )
        self.assertEqual(len(set(rationale_indices)), 2)
        counts = sorted(
            rationale_indices.count(index)
            for index in set(rationale_indices)
        )
        self.assertEqual(counts, [2, 2])
        self.assertEqual(sorted(semantic_modes), [0, 0, 1, 1])

    def test_gold_rationale_is_not_a_persistent_primary_route(self):
        selections = set()
        for seed in range(24):
            schedule, _ = build_teacher_rationale_schedule(
                n_available=4,
                set_size=4,
                max_semantic_modes=2,
                generator=torch.Generator().manual_seed(seed),
            )
            selections.update(schedule)
        self.assertEqual(selections, {0, 1, 2, 3})


class SetMatchingTests(unittest.TestCase):
    def test_four_path_matching_is_permutation_invariant(self):
        torch.manual_seed(3)
        model = torch.randn(2, 4, 8, 10)
        teacher = torch.randn(2, 4, 8, 10)
        original = permutation_invariant_set_matching(
            model,
            teacher,
        )["loss"]
        model_permutation = torch.tensor([2, 0, 3, 1])
        teacher_permutation = torch.tensor([1, 3, 0, 2])
        permuted = permutation_invariant_set_matching(
            model.index_select(1, model_permutation),
            teacher.index_select(1, teacher_permutation),
        )["loss"]
        self.assertTrue(torch.allclose(original, permuted, atol=1e-6))


class CounterfactualCreditTests(unittest.TestCase):
    @staticmethod
    def _paths():
        return torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.03], [1.0, 0.03]],
                [[1.0, 0.08], [1.0, 0.08]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        )

    def test_mining_uses_nearest_wrong_as_hard_negative(self):
        pairs = mine_question_local_hard_pairs(
            self._paths(),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            group_size=4,
            margin=0.1,
            max_pairs_per_group=1,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].wrong_index, 2)
        self.assertIn(pairs[0].correct_peer_index, {0, 1})

    def test_swap_keeps_prefix_and_recomputes_suffix(self):
        actions = torch.arange(
            4 * 3 * 2,
            dtype=torch.float32,
        ).reshape(4, 3, 2)
        innovations = actions + 100
        pair = HardPathPair(
            group_index=0,
            correct_index=0,
            correct_peer_index=1,
            wrong_index=2,
            correct_radius=0.1,
            wrong_distance=0.05,
            hinge=0.15,
        )
        batch = counterfactual_action_batch(
            actions,
            innovations,
            [pair],
        )
        # Rows are [c<-w, w<-c] for transition 0, then transition 1, etc.
        row = 2
        self.assertTrue(
            torch.equal(batch["forced_actions"][row, 0], actions[0, 0])
        )
        self.assertTrue(
            torch.equal(batch["forced_actions"][row, 1], actions[2, 1])
        )
        self.assertTrue(
            torch.equal(
                batch["forced_mask"][row],
                torch.tensor([True, True, False]),
            )
        )
        self.assertTrue(
            torch.equal(batch["innovations"][row], innovations[0])
        )

    def test_bidirectional_scores_produce_per_step_credit(self):
        pair = HardPathPair(
            group_index=0,
            correct_index=0,
            correct_peer_index=1,
            wrong_index=2,
            correct_radius=0.1,
            wrong_distance=0.05,
            hinge=0.15,
        )
        metadata = {
            "pair_indices": torch.tensor([0, 0, 0, 0]),
            "step_indices": torch.tensor([0, 0, 1, 1]),
            "directions": torch.tensor([0, 1, 0, 1]),
        }
        base_scores = torch.tensor([1.0, 0.9, 0.0])
        counterfactual_scores = torch.tensor([0.4, 0.6, 0.9, 0.1])
        credits = counterfactual_transition_credits(
            base_scores,
            counterfactual_scores,
            metadata,
            [pair],
            n_steps=2,
        )
        self.assertTrue(torch.allclose(credits, torch.tensor([[0.6, 0.1]])))

    def test_credit_raises_correct_and_suppresses_wrong_transition(self):
        pair = HardPathPair(
            group_index=0,
            correct_index=0,
            correct_peer_index=1,
            wrong_index=2,
            correct_radius=0.1,
            wrong_distance=0.05,
            hinge=0.15,
        )
        advantages = build_transition_advantages(
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            n_steps=2,
            group_size=4,
            pairs=[pair],
            counterfactual_credits=torch.tensor([[1.0, 0.1]]),
            counterfactual_weight=0.5,
            local_weight=0.1,
            credit_temperature=1.0,
            local_temperature=0.1,
        )
        self.assertGreater(float(advantages[0, 0]), float(advantages[0, 1]))
        self.assertLess(float(advantages[2, 0]), float(advantages[2, 1]))


class ArchitectureContractTests(unittest.TestCase):
    def test_answer_mask_has_no_question_access(self):
        question = torch.ones(2, 5, dtype=torch.long)
        latent = torch.ones(2, 8, dtype=torch.long)
        answer = torch.ones(2, 4, dtype=torch.long)
        mask = build_path_bottleneck_mask(question, latent, answer)
        self.assertTrue(torch.equal(mask[:, :5], torch.zeros_like(question)))
        self.assertTrue(torch.equal(mask[:, 5:13], latent))
        self.assertTrue(torch.equal(mask[:, 13:], answer))

    def test_no_path_mask_hides_question_and_all_latents(self):
        question = torch.ones(1, 5, dtype=torch.long)
        latent = torch.zeros(1, 8, dtype=torch.long)
        answer = torch.ones(1, 3, dtype=torch.long)
        mask = build_path_bottleneck_mask(question, latent, answer)
        self.assertEqual(int(mask[:, :13].sum().item()), 0)
        self.assertTrue(torch.equal(mask[:, 13:], answer))

    def test_answer_cache_fork_does_not_mutate_path_prefix(self):
        cache = DynamicCache()
        key = torch.randn(1, 1, 2, 4)
        value = torch.randn(1, 1, 2, 4)
        cache.update(key, value, layer_idx=0)
        fork = LitTRACEPolicy._fork_past_key_values(cache)
        fork.update(
            torch.randn(1, 1, 1, 4),
            torch.randn(1, 1, 1, 4),
            layer_idx=0,
        )
        self.assertEqual(cache.get_seq_length(), 2)
        self.assertEqual(fork.get_seq_length(), 3)

    def test_only_stage2_checkpoint_can_restore_policy_reference(self):
        stale = {"mean_head.weight": torch.ones(1)}
        stage1 = {
            "trace_policy_training_stage": 1,
            "trace_stage1_policy_reference": stale,
        }
        self.assertIsNone(extract_stage2_policy_reference(stage1))
        with self.assertRaises(RuntimeError):
            extract_stage2_policy_reference(
                {"trace_policy_training_stage": 2}
            )
        stage2 = {
            "trace_policy_training_stage": 2,
            "trace_stage1_policy_reference": stale,
        }
        self.assertIs(extract_stage2_policy_reference(stage2), stale)

    def test_model_inherits_original_bridge_backbone_directly(self):
        self.assertIn(LitREADCoTStableEfficient, LitTRACEPolicy.__bases__)

    def test_stage1_teacher_is_a_frozen_checkpointed_adapter(self):
        source = inspect.getsource(LitTRACEPolicy)
        self.assertIn('teacher_adapter_name = "trace_teacher"', source)
        self.assertIn("_copy_path_adapter_to_teacher_adapter", source)
        self.assertIn("_activate_teacher_adapter", source)
        self.assertIn("or teacher_marker in name", source)
        self.assertIn(
            'parameter.requires_grad_(False)',
            inspect.getsource(
                LitTRACEPolicy._set_adapter_parameter_trainability
            ),
        )
        trajectory_source = inspect.getsource(
            LitTRACEPolicy._trajectory_latents
        )
        self.assertNotIn("self.latent_bridge(", trajectory_source)
        self.assertNotIn("self.residual_projector(", trajectory_source)

    def test_added_adapter_copy_matches_path_dtype_and_values(self):
        class AdapterHarness:
            path_adapter_name = "default"
            _match_adapter_storage_to_path = (
                LitTRACEPolicy._match_adapter_storage_to_path
            )
            _copy_path_adapter = LitTRACEPolicy._copy_path_adapter

        harness = AdapterHarness()
        harness.llm = torch.nn.Module()
        harness.llm.lora_A = torch.nn.Module()
        harness.llm.lora_A.default = torch.nn.Linear(
            3,
            2,
            bias=False,
            dtype=torch.float32,
        )
        harness.llm.lora_A.trace_teacher = torch.nn.Linear(
            3,
            2,
            bias=False,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            harness.llm.lora_A.default.weight.normal_()
        copied = harness._copy_path_adapter("trace_teacher")
        source = harness.llm.lora_A.default.weight
        target = harness.llm.lora_A.trace_teacher.weight
        self.assertEqual(copied, 1)
        self.assertEqual(source.dtype, target.dtype)
        self.assertTrue(torch.equal(source, target))

    def test_source_has_no_fixed_first_rollout_or_route_table(self):
        source = inspect.getsource(LitTRACEPolicy)
        forbidden = (
            "center_seeds",
            "center_indices",
            "view_ids_tensor",
            "_sample_path_seeds",
            "trace_view_embeddings =",
            "[:, 0].zero_",
        )
        for phrase in forbidden:
            self.assertNotIn(phrase, source)

    def test_formal_config_enables_real_trajectory_policy(self):
        path = Path(
            "/disk1/dingxukai/trace_colar/src/configs/models/"
            "trace_policy_qwen3_instruct.yaml"
        )
        text = path.read_text()
        self.assertIn("use_trajectory_policy_loss: True", text)
        self.assertIn("stage1_teacher_set_size: 4", text)
        self.assertNotIn("use_latent_loss: False", text)
        self.assertNotIn("center_path", text)
        source = inspect.getsource(LitTRACEPolicy.trace_rl_training_step)
        self.assertIn("action_ratio_deviation_final_update", source)
        self.assertIn("action_clip_fraction_final_update", source)

    def test_formal_four_gpu_scheduler_matches_global_batch_budget(self):
        root = Path("/disk1/dingxukai/trace_colar")
        stage1 = (root / "run_trace_policy_stage1_full.sh").read_text()
        stage2 = (root / "run_trace_policy_stage2_full.sh").read_text()
        evidence = (
            root / "run_trace_policy_full_evidence.sh"
        ).read_text()
        run_source = (root / "run.py").read_text()
        self.assertIn("config.dataloader.batch_size = bs_per_device", run_source)
        self.assertIn("requested_global_batch_size=4", stage1)
        self.assertIn("effective_per_device_batch_size=1", stage1)
        self.assertIn("scheduled_optimizer_steps=16820", stage1)
        self.assertIn(
            "explicit_teacher_adapter=frozen_stage0_cot_lora",
            stage1,
        )
        self.assertIn(
            "Stage 0 must be a plain CoT-SFT checkpoint",
            stage1,
        )
        self.assertIn('"trajectory_policy",', stage1)
        self.assertIn(
            "model.training_kwargs.scheduler.num_training_steps=16820",
            stage1,
        )
        self.assertIn("requested_global_question_batch_size=4", stage2)
        self.assertIn(
            "effective_per_device_question_batch_size=1",
            stage2,
        )
        self.assertIn("policy_updates_per_rollout=2", stage2)
        self.assertIn(
            "unique_training_questions_per_epoch=6726",
            stage2,
        )
        self.assertIn("dataloader_rows_per_epoch=6728", stage2)
        self.assertIn("ddp_padding_duplicates_per_epoch=2", stage2)
        self.assertIn("scheduled_optimizer_steps=33640", stage2)
        self.assertIn("trainer.limit_train_batches=1682", stage2)
        self.assertIn("dataset_subset_mutation=false", stage2)
        self.assertIn(
            'if not any(".trace_teacher." in key for key in keys)',
            stage2,
        )
        self.assertIn(
            "model.training_kwargs.scheduler.num_training_steps=33640",
            stage2,
        )
        self.assertIn(
            '"trace_stage1_policy_reference" not in checkpoint',
            evidence,
        )

    def test_four_rank_budget_covers_full_6726_question_split(self):
        dataset = range(6726)
        rank_indices = []
        for rank in range(4):
            sampler = DistributedSampler(
                dataset,
                num_replicas=4,
                rank=rank,
                shuffle=True,
                seed=0,
                drop_last=False,
            )
            sampler.set_epoch(3)
            rank_indices.append(list(sampler))
        flattened = [
            index for local_indices in rank_indices for index in local_indices
        ]
        self.assertEqual(len(flattened), 6728)
        self.assertEqual(len(set(flattened)), 6726)
        duplicate_rows = len(flattened) - len(set(flattened))
        self.assertEqual(duplicate_rows, 2)

    def test_validation_summary_removes_ddp_padding_duplicate(self):
        shards = [
            [(0, 1.0, 4), (4, 0.0, 7)],
            [(1, 0.0, 5), (0, 1.0, 4)],
            [(2, 1.0, 6)],
            [(3, 1.0, 8)],
        ]
        summary = summarize_unique_validation_records(
            shards,
            expected_count=5,
        )
        self.assertEqual(summary["unique_questions"], 5.0)
        self.assertAlmostEqual(summary["accuracy"], 0.6)
        with self.assertRaises(RuntimeError):
            summarize_unique_validation_records(
                [[(0, 1.0, 4)], [(0, 0.0, 4)]],
                expected_count=1,
            )


class TaskSummaryTests(unittest.TestCase):
    def test_total_length_is_eight_latents_plus_generated_tokens(self):
        stage1 = {
            0: {"accuracy": 0.0, "total_L": 44.0},
            1: {"accuracy": 1.0, "total_L": 42.0},
        }
        final = {
            0: {"accuracy": 1.0, "total_L": 43.0},
            1: {"accuracy": 1.0, "total_L": 41.0},
        }
        summary = summarize_pair(
            stage1,
            final,
            rng=np.random.default_rng(7),
            draws=100,
        )
        self.assertEqual(summary["latent_length"], 8)
        self.assertEqual(summary["rescued"], 1)
        self.assertEqual(summary["regressed"], 0)
        self.assertAlmostEqual(summary["stage1_total_L"]["mean"], 43.0)
        self.assertAlmostEqual(summary["final_total_L"]["mean"], 42.0)
        self.assertAlmostEqual(summary["total_L_delta"]["mean"], -1.0)

    def test_exact_paired_sign_test_uses_only_discordant_questions(self):
        self.assertEqual(exact_sign_p(0, 0), 1.0)
        self.assertEqual(exact_sign_p(3, 3), 1.0)
        self.assertAlmostEqual(exact_sign_p(10, 0), 2.0 / 1024.0)


if __name__ == "__main__":
    unittest.main()
