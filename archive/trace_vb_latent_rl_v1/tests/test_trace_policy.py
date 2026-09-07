import inspect
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    CoTConditionedTrajectoryPosterior,
    GaussianTrajectoryPolicy,
    HardPathPair,
    action_conditioned_progress_centers,
    action_transition_identifiability_loss,
    build_transition_advantages,
    clipped_policy_loss,
    counterfactual_action_batch,
    counterfactual_transition_credits,
    diagonal_gaussian_kl,
    gaussian_log_prob,
    group_standardize,
    minimum_action_entropy_loss,
    mine_question_local_hard_pairs,
    monotonic_path_marginals,
    monotone_progress_centers,
    sampled_forward_kl,
    stochastic_monotone_assignment,
)
from tools.trace_policy_task_summary import exact_sign_p, summarize_pair
from tools.trace_policy_geometry_summary import (
    action_path_permutation_null,
    formation_geometry,
    question_heldout_probe,
)
from tools.trace_policy_causal_summary import (
    matrix_mean_ci,
    summarize_commit_bottleneck_sanity,
)
from tools.full_supervisor import (
    detect_fatal_marker,
    process_group_exists,
    terminate_process_group,
)
from run import load_full_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class GaussianPolicyTests(unittest.TestCase):
    def test_fresh_dynamics_is_a_stable_residual_recurrence(self):
        policy = GaussianTrajectoryPolicy(
            hidden_size=12,
            action_dim=4,
            n_steps=8,
            policy_hidden_size=16,
        )
        states = torch.randn(3, 12)
        actions = torch.randn(3, 4)
        latent_input = policy.latent_input(
            states,
            actions,
            0,
            action_scale=0.0,
            step_scale=0.0,
        )
        self.assertTrue(torch.equal(latent_input, states))

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

    def test_sampled_answer_kl_is_nonnegative_and_has_gradient(self):
        current = torch.tensor(
            [[-1.0, -2.0, -3.0]],
            requires_grad=True,
        )
        reference = torch.tensor([[-1.0, -1.5, -3.5]])
        mask = torch.tensor([[1, 1, 0]])
        equal = sampled_forward_kl(
            current,
            current.detach(),
            mask=mask,
        )
        shifted = sampled_forward_kl(
            current,
            reference,
            mask=mask,
        )
        self.assertEqual(float(equal), 0.0)
        self.assertGreater(float(shifted), 0.0)
        shifted.backward()
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertEqual(float(current.grad[0, 2]), 0.0)

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

    def test_each_action_path_defines_its_own_progress_schedule(self):
        actions = torch.zeros(2, 8, 4)
        actions[0, :, 0] = torch.linspace(-2.0, 2.0, 8)
        actions[1, :, 0] = torch.linspace(2.0, -2.0, 8)
        centers = action_conditioned_progress_centers(
            actions,
            progress_dim=0,
            action_scale=1.0,
        )
        self.assertTrue(torch.all(centers[:, 1:] > centers[:, :-1]))
        self.assertTrue(
            torch.allclose(
                centers[:, -1],
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertFalse(torch.allclose(centers[0], centers[1]))

    def test_progress_schedule_has_no_external_route_identity(self):
        actions = torch.randn(4, 8, 16)
        first = action_conditioned_progress_centers(
            actions,
            progress_dim=0,
            action_scale=1.0,
        )
        second = action_conditioned_progress_centers(
            actions.clone(),
            progress_dim=0,
            action_scale=1.0,
        )
        self.assertTrue(torch.equal(first, second))
        changed = actions.clone()
        changed[2, :, 0] = changed[2, :, 0].flip(dims=(0,))
        third = action_conditioned_progress_centers(
            changed,
            progress_dim=0,
            action_scale=1.0,
        )
        self.assertTrue(torch.equal(first[:2], third[:2]))
        self.assertFalse(torch.allclose(first[2], third[2]))

    def test_progress_schedule_rejects_zero_action_scale(self):
        with self.assertRaises(ValueError):
            action_conditioned_progress_centers(
                torch.randn(4, 8, 16),
                progress_dim=0,
                action_scale=0.0,
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

    def test_action_identifiability_uses_exchangeable_path_differences(self):
        predictions = torch.zeros(2, 4, 8, 3, requires_grad=True)
        actions = torch.randn(2, 4, 8, 3)
        loss = action_transition_identifiability_loss(
            predictions,
            actions,
        )
        loss.backward()
        self.assertGreater(float(predictions.grad.abs().sum()), 0.0)

    def test_entropy_floor_prevents_distribution_collapse(self):
        healthy = minimum_action_entropy_loss(
            torch.full((2, 8, 4), -1.0),
            minimum_std=0.2,
        )
        collapsed = minimum_action_entropy_loss(
            torch.full((2, 8, 4), -4.0),
            minimum_std=0.2,
        )
        self.assertEqual(float(healthy), 0.0)
        self.assertGreater(float(collapsed), 0.0)

    def test_cot_posterior_is_a_distribution_not_a_route_table(self):
        posterior = CoTConditionedTrajectoryPosterior(
            hidden_size=12,
            action_dim=4,
            n_steps=8,
            posterior_hidden_size=16,
        )
        states = torch.randn(3, 12)
        context = torch.randn(3, 12)
        prior_means = torch.zeros(3, 4)
        prior_log_stds = torch.full((3, 4), -0.7)
        means, log_stds = posterior.distribution_parameters(
            states,
            context,
            2,
            prior_means,
            prior_log_stds,
        )
        self.assertEqual(tuple(means.shape), (3, 4))
        self.assertEqual(tuple(log_stds.shape), (3, 4))
        self.assertGreater(float(torch.exp(log_stds).min()), 0.0)
        self.assertFalse(
            any("route" in name or "view" in name for name, _ in posterior.named_parameters())
        )


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

    def test_single_correct_rollout_still_receives_counterfactual_credit(self):
        pairs = mine_question_local_hard_pairs(
            self._paths(),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            group_size=4,
            margin=0.1,
            max_pairs_per_group=1,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].correct_index, 0)
        self.assertEqual(pairs[0].correct_peer_index, 0)
        self.assertFalse(pairs[0].has_correct_peer)
        self.assertEqual(pairs[0].correct_radius, 0.0)

    def test_all_wrong_group_uses_only_real_continuous_score_gap(self):
        pairs = mine_question_local_hard_pairs(
            self._paths(),
            torch.zeros(4),
            group_size=4,
            margin=0.1,
            outcome_scores=torch.tensor([-1.0, -1.1, -1.4, -2.0]),
            minimum_score_gap=0.05,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].source, "continuous_outcome")
        rejected = mine_question_local_hard_pairs(
            self._paths(),
            torch.zeros(4),
            group_size=4,
            margin=0.1,
            outcome_scores=torch.tensor(
                [-1.0, -1.00001, -1.00002, -1.00003]
            ),
            minimum_score_gap=0.05,
        )
        self.assertEqual(rejected, [])

    def test_all_correct_group_does_not_invent_a_negative_path(self):
        pairs = mine_question_local_hard_pairs(
            self._paths(),
            torch.ones(4),
            group_size=4,
            margin=0.1,
            outcome_scores=torch.tensor([-1.0, -1.1, -1.4, -2.0]),
            minimum_score_gap=0.05,
        )
        self.assertEqual(pairs, [])

    def test_transition_standardization_does_not_mix_step_positions(self):
        values = torch.tensor(
            [
                [100.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 3.0],
            ]
        )
        changed = values.clone()
        changed[1, 0] = 50.0
        first = group_standardize(
            values,
            group_size=4,
            preserve_trailing_positions=True,
        )
        second = group_standardize(
            changed,
            group_size=4,
            preserve_trailing_positions=True,
        )
        self.assertTrue(torch.equal(first[:, 1], second[:, 1]))
        self.assertFalse(torch.equal(first[:, 0], second[:, 0]))

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

    def test_rotating_swap_batch_can_select_a_strict_step_subset(self):
        actions = torch.randn(4, 8, 2)
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
            torch.randn_like(actions),
            [pair],
            step_indices=[0, 2, 4, 6],
        )
        self.assertEqual(batch["forced_actions"].shape[0], 8)
        self.assertEqual(
            set(batch["step_indices"].tolist()),
            {0, 2, 4, 6},
        )
        credits = counterfactual_transition_credits(
            torch.tensor([1.0, 0.9, 0.0, 0.0]),
            torch.zeros(8),
            batch,
            [pair],
            n_steps=8,
        )
        self.assertTrue(torch.equal(credits[0, 1::2], torch.zeros(4)))

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

    def test_base_scoring_selects_only_paths_used_by_hard_pairs(self):
        pairs = [
            HardPathPair(
                group_index=0,
                correct_index=5,
                correct_peer_index=6,
                wrong_index=2,
                correct_radius=0.1,
                wrong_distance=0.05,
                hinge=0.15,
            ),
            HardPathPair(
                group_index=1,
                correct_index=9,
                correct_peer_index=10,
                wrong_index=13,
                correct_radius=0.1,
                wrong_distance=0.05,
                hinge=0.15,
            ),
        ]
        self.assertEqual(
            LitTRACEPolicy._hard_pair_path_indices(pairs),
            [2, 5, 9, 13],
        )

    def test_incomplete_counterfactual_metadata_is_rejected(self):
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
            "pair_indices": torch.tensor([0, 0, 0]),
            "step_indices": torch.tensor([0, 0, 1]),
            "directions": torch.tensor([0, 1, 0]),
        }
        with self.assertRaises(ValueError):
            counterfactual_transition_credits(
                torch.tensor([1.0, 0.9, 0.0]),
                torch.tensor([0.4, 0.6, 0.9]),
                metadata,
                [pair],
                n_steps=2,
            )

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

    def test_dense_outcome_keeps_homogeneous_groups_trainable(self):
        exact_correctness = torch.ones(4)
        frozen_gold_scores = torch.tensor([-2.0, -1.0, -3.0, -1.5])
        dense = group_standardize(
            frozen_gold_scores.unsqueeze(-1),
            group_size=4,
        ).squeeze(-1)
        rewards = exact_correctness + 0.25 * dense
        advantages = build_transition_advantages(
            rewards,
            n_steps=2,
            group_size=4,
            pairs=[],
            counterfactual_credits=None,
            counterfactual_weight=0.5,
            local_weight=0.1,
            credit_temperature=1.0,
            local_temperature=0.1,
        )
        self.assertGreater(
            float((advantages.abs() > 1e-6).float().mean()),
            0.0,
        )


class ArchitectureContractTests(unittest.TestCase):
    def test_test_loader_restores_checkpoint_level_state(self):
        class Harness:
            def __init__(self):
                self.loaded_marker = None

            def on_load_checkpoint(self, checkpoint):
                self.loaded_marker = checkpoint["auxiliary_marker"]

        checkpoint = {
            "state_dict": {"weight": torch.ones(1)},
            "auxiliary_marker": "restored",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save(checkpoint, path)
            harness = Harness()
            state_dict = load_full_checkpoint(harness, str(path))
        self.assertEqual(harness.loaded_marker, "restored")
        self.assertTrue(torch.equal(state_dict["weight"], torch.ones(1)))

    def test_supervisor_detects_an_oom_across_log_read_boundaries(self):
        marker, tail = detect_fatal_marker("", "torch.OutOfMem")
        self.assertIsNone(marker)
        marker, _ = detect_fatal_marker(tail, "oryError: CUDA failed")
        self.assertEqual(marker, "torch.OutOfMemoryError")

    def test_compact_targets_fit_deployment_budget_without_truncation(self):
        class FakeTokenizer:
            eos_token = "</s>"

            @staticmethod
            def encode(text, add_special_tokens=False):
                del add_special_tokens
                return re.findall(
                    r"###|</s>|[A-Za-z]+|\d+(?:\.\d+)?|\S",
                    text,
                )

        class TargetHarness:
            tokenizer = FakeTokenizer()
            trace_config = {}
            model_kwargs = SimpleNamespace(
                hybrid_generation_config=SimpleNamespace(
                    max_new_tokens=48,
                )
            )
            anchor_header = "Anchors:"
            thinking_separator = "###"
            answer_template = "Answer:{}"
            _target_token_count = LitTRACEPolicy._target_token_count
            _fit_compact_target_to_generation_budget = (
                LitTRACEPolicy._fit_compact_target_to_generation_budget
            )

        harness = TargetHarness()
        answer_suffix = "###Answer:11280"
        first_clause = "- 123 + 456 = 579"
        harness.trace_config["compact_target_max_new_tokens"] = (
            harness._target_token_count(
                first_clause + "\n" + answer_suffix
            )
        )
        target = (
            "Anchors:\n"
            "- 123 + 456 = 579; 579 * 20 = 11580\n"
            "- 11580 - 300 = 11280\n"
            f"{answer_suffix}"
        )
        fitted = harness._fit_compact_target_to_generation_budget(
            target,
            "11280",
        )
        self.assertLessEqual(
            harness._target_token_count(fitted),
            harness.trace_config["compact_target_max_new_tokens"],
        )
        self.assertTrue(fitted.endswith(answer_suffix))
        self.assertIn(first_clause, fitted)
        self.assertNotIn("579 * 20 =", fitted)

    def test_path_only_ablation_has_no_question_access(self):
        question = torch.ones(2, 5, dtype=torch.long)
        latent = torch.ones(2, 8, dtype=torch.long)
        answer = torch.ones(2, 4, dtype=torch.long)
        mask = build_path_bottleneck_mask(question, latent, answer)
        self.assertTrue(torch.equal(mask[:, :5], torch.zeros_like(question)))
        self.assertTrue(torch.equal(mask[:, 5:13], latent))
        self.assertTrue(torch.equal(mask[:, 13:], answer))

    def test_competitive_decoder_reads_question_and_commit_only(self):
        question = torch.tensor([[1, 1, 0]], dtype=torch.long)
        latent = torch.ones(1, 8, dtype=torch.long)
        answer = torch.ones(1, 4, dtype=torch.long)
        latent = LitTRACEPolicy._commit_only_latent_mask(latent)
        mask = build_path_bottleneck_mask(
            question,
            latent,
            answer,
            include_question=True,
        )
        self.assertTrue(torch.equal(mask[:, :3], question))
        self.assertEqual(int(mask[:, 3:10].sum().item()), 0)
        self.assertEqual(int(mask[:, 10].item()), 1)
        self.assertTrue(torch.equal(mask[:, 11:], answer))

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

    def test_rollout_uses_one_path_for_frozen_answer_reward_and_step_credit(self):
        source = inspect.getsource(LitTRACEPolicy.trace_policy_rollout)
        self.assertEqual(source.count("self._trajectory_latents("), 1)
        self.assertIn(
            "self._generate_answers_from_trajectory(\n"
            "                sampled_path,",
            source,
        )
        self.assertIn(
            "self._gold_answer_scores(sampled_path, chunk_answers)",
            source,
        )
        self.assertIn(
            "self._role_step_rewards(sampled_path, chunk_targets)",
            source,
        )
        self.assertNotIn("_answer_token_log_probs", source)
        self.assertNotIn("_answer_policy_update", source)

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

    def test_model_reuses_the_audited_latent_sft_implementation(self):
        self.assertIn(LitREADCoTStableEfficient, LitTRACEPolicy.__bases__)

    def test_stage1_cot_encoder_is_a_frozen_checkpointed_adapter(self):
        source = inspect.getsource(LitTRACEPolicy)
        self.assertIn(
            "self.latent_bridge = torch.nn.Identity()",
            source,
        )
        self.assertIn(
            "self.residual_projector = torch.nn.Identity()",
            source,
        )
        self.assertIn(
            "for legacy_module in (self.step_compressor, self.latent_relation)",
            source,
        )
        self.assertIn(
            'cot_encoder_adapter_name = "trace_cot_encoder"',
            source,
        )
        self.assertIn(
            "_copy_path_adapter_to_cot_encoder_adapter",
            source,
        )
        self.assertIn("_activate_cot_encoder_adapter", source)
        self.assertIn("or cot_encoder_marker in name", source)
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
        harness.llm.lora_A.trace_cot_encoder = torch.nn.Linear(
            3,
            2,
            bias=False,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            harness.llm.lora_A.default.weight.normal_()
        copied = harness._copy_path_adapter("trace_cot_encoder")
        source = harness.llm.lora_A.default.weight
        target = harness.llm.lora_A.trace_cot_encoder.weight
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
        self.assertNotIn("corridor_progress_logits", source)
        corridor_source = inspect.getsource(
            LitTRACEPolicy._single_cot_corridor
        )
        self.assertIn(
            "action_conditioned_progress_centers",
            corridor_source,
        )
        self.assertIn("student_actions.detach()", corridor_source)

    def test_trajectory_recurrence_uses_hidden_backbone_without_lm_logits(self):
        source = inspect.getsource(LitTRACEPolicy._trajectory_latents)
        self.assertIn("self.llm.get_base_model()", source)
        self.assertIn("last_hidden_state", source)
        self.assertNotIn("self.llm.forward(", source)
        self.assertNotIn('"question_inputs_embeds"', source)
        self.assertNotIn('"context_inputs_embeds"', source)

    def test_formal_config_enables_real_trajectory_policy(self):
        path = ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml"
        text = path.read_text()
        self.assertIn("use_trajectory_policy_loss: True", text)
        self.assertIn("use_answer_policy_loss: False", text)
        self.assertIn("stage1_posterior_samples: 3", text)
        self.assertIn("stage1_map_compact_weight: 1.0", text)
        self.assertIn("answer_context_mode: question_and_path", text)
        self.assertIn("use_hybrid: True", text)
        self.assertIn("hybrid_seed_anchor_header: True", text)
        self.assertIn("stage1_posterior_kl_weight:", text)
        self.assertIn("stage1_plan_weight:", text)
        self.assertIn("stage1_solve_weight:", text)
        self.assertIn("stage1_check_weight:", text)
        self.assertIn("stage1_role_minimum_stds:", text)
        self.assertIn("dense_outcome_weight: 0.25", text)
        self.assertIn("step_reward_weight: 0.30", text)
        self.assertIn("step_reward_discount: 0.90", text)
        self.assertIn("role_entropy_weights:", text)
        self.assertIn(
            "stage1_posterior_activation_offload: true",
            text,
        )
        self.assertIn(
            "stage1_offload_cache_release_interval: 10",
            text,
        )
        self.assertNotIn("use_latent_loss: False", text)
        self.assertNotIn("center_path", text)
        source = inspect.getsource(LitTRACEPolicy.trace_rl_training_step)
        self.assertIn("self._trajectory_policy_update(rollout)", source)
        self.assertIn("role_entropy", source)
        self.assertNotIn("self._answer_policy_update(rollout)", source)
        rollout_source = inspect.getsource(
            LitTRACEPolicy.trace_policy_rollout
        )
        self.assertIn("frozen_gold_scores", rollout_source)
        self.assertIn("semantic_step_rewards", rollout_source)
        self.assertIn("trajectory_returns", rollout_source)
        self.assertNotIn("stage1_answer_log_probs", rollout_source)
        self.assertNotIn("counterfactual", rollout_source)

    def test_formal_four_gpu_scheduler_matches_global_batch_budget(self):
        root = ROOT
        stage0 = (root / "scripts/run_stage0_cot.sh").read_text()
        stage1 = (root / "scripts/run_stage1_formation.sh").read_text()
        stage2 = (root / "scripts/run_stage2_refinement.sh").read_text()
        stage2_pipeline = (
            root / "scripts/run_stage2_and_evidence.sh"
        ).read_text()
        evidence = (root / "scripts/run_evidence.sh").read_text()
        run_source = (root / "run.py").read_text()
        self.assertIn("config.dataloader.batch_size = bs_per_device", run_source)
        self.assertLess(
            run_source.index("torch.cuda.set_device(local_rank_index)"),
            run_source.index("seed_current_process(args.seed)"),
        )
        self.assertIn("install_rank_local_ddp_seed_reset()", run_source)
        self.assertIn("old_checkpoint_reused=false", stage0)
        self.assertIn("requested_global_batch_size=4", stage1)
        self.assertIn("effective_per_device_batch_size=1", stage1)
        self.assertIn("GPU IDs must be unique", stage1)
        self.assertIn("GPU IDs must be unique", stage2)
        self.assertIn("src.models.cot.LitCot", stage1)
        self.assertIn(
            "Stage 0 used an unregistered dataset",
            stage1,
        )
        self.assertIn(
            "Stage 0 was not trained from the fresh base model",
            stage1,
        )
        self.assertIn("scheduled_optimizer_steps=16820", stage1)
        self.assertIn(
            "posterior_saved_activation_offload=cpu_exact_autograd",
            stage1,
        )
        self.assertIn(
            "offload_cache_release_interval=10_batches_after_optimizer_step",
            stage1,
        )
        self.assertIn("STAGE1_NUM_WORKERS=${STAGE1_NUM_WORKERS:-0}", stage1)
        self.assertIn("STAGE1_PIN_MEMORY=${STAGE1_PIN_MEMORY:-false}", stage1)
        self.assertIn("STAGE1_MAX_EPOCHS=${STAGE1_MAX_EPOCHS:-10}", stage1)
        self.assertIn(
            "process_recycling=one_full_epoch_per_process",
            stage1,
        )
        self.assertIn('trainer.max_epochs="${target_max_epochs}"', stage1)
        self.assertIn("next_completed != target_max_epochs", stage1)
        self.assertIn(
            "trainer.strategy=ddp_find_unused_parameters_true",
            stage1,
        )
        self.assertIn("run_stage1_ddp_stress_preflight", (
            root / "scripts/run_full_pipeline.sh"
        ).read_text())
        self.assertIn(
            "explicit_cots_per_question=1_original_only",
            stage1,
        )
        self.assertIn(
            "Stage 0 contains TRACE state",
            stage1,
        )
        self.assertIn(
            "stage1_decoder_supervision=sampled_answer_plus_MAP_role_compact_target",
            stage1,
        )
        self.assertIn(
            "role_target_construction=PLAN_teacher_summary_SOLVE_five_monotone_CoT_chunks_CHECK_endpoint_error",
            stage1,
        )
        self.assertIn(
            "short_CoT_handling=empty_SOLVE_slots_masked_final_step_anchored",
            stage1,
        )
        self.assertIn(
            "answer_context=question_plus_COMMIT_latent_only",
            stage1,
        )
        self.assertIn(
            "model.training_kwargs.scheduler.num_training_steps=16820",
            stage1,
        )
        self.assertIn("requested_global_question_batch_size=4", stage2)
        self.assertIn(
            "effective_per_device_question_batch_size=1",
            stage2,
        )
        self.assertIn("policy_updates_per_rollout=1", stage2)
        self.assertIn(
            "unique_training_questions_per_epoch=2048",
            stage2,
        )
        self.assertIn("source_training_split_questions=6726", stage2)
        self.assertIn("dataloader_rows_per_epoch=2048", stage2)
        self.assertIn("ddp_padding_duplicates_per_epoch=0", stage2)
        self.assertIn("scheduled_optimizer_steps=5120", stage2)
        self.assertIn("trainer.limit_train_batches=512", stage2)
        self.assertIn("dataset_subset_mutation=false", stage2)
        self.assertIn(
            'if not any(".trace_cot_encoder." in key for key in keys)',
            stage2,
        )
        self.assertIn(
            "model.training_kwargs.scheduler.num_training_steps=5120",
            stage2,
        )
        self.assertIn(
            "ddp_gradient_sync=standard_lightning_ddp_each_micro_batch",
            stage2,
        )
        self.assertIn(
            "trace_rl_config.rollout_micro_batch_size=2",
            stage2,
        )
        self.assertIn(
            "trace_rl_config.exp_batch_size=2",
            stage2,
        )
        self.assertIn(
            "answer_decoder_trainable_in_stage2=false",
            stage2,
        )
        self.assertIn(
            "step_credit=frozen_stage1_role_semantic_scores_plus_terminal_answer_reward",
            stage2,
        )
        self.assertIn(
            "role_entropy=PLAN_high_SOLVE_decay_CHECK_low_COMMIT_deterministic",
            stage2,
        )
        self.assertIn(
            "return_definition=position_local_discounted_role_rewards_plus_terminal_reward",
            stage2,
        )
        self.assertIn(
            "trace_policy_stage2_stability_smoke.py",
            stage2_pipeline,
        )
        self.assertIn(
            '"trace_stage1_policy_reference" not in checkpoint',
            evidence,
        )
        help_result = subprocess.run(
            [
                sys.executable,
                str(root / "tools/trace_policy_ddp_memory_smoke.py"),
                "--help",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)

    def test_four_rank_sampler_draws_2048_unique_questions_per_epoch(self):
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
            rank_indices.append(list(sampler)[:512])
        flattened = [
            index for local_indices in rank_indices for index in local_indices
        ]
        self.assertEqual(len(flattened), 2048)
        self.assertEqual(len(set(flattened)), 2048)
        duplicate_rows = len(flattened) - len(set(flattened))
        self.assertEqual(duplicate_rows, 0)

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


class GeometryEvidenceTests(unittest.TestCase):
    def test_minimum_progress_gap_includes_origin_to_first_step(self):
        actions = np.zeros((4, 3, 2), dtype=np.float64)
        centers = np.asarray(
            [
                [0.01, 0.50, 1.00],
                [0.02, 0.51, 1.00],
                [0.03, 0.52, 1.00],
                [0.04, 0.53, 1.00],
            ]
        )
        path_distances = np.zeros((4, 4), dtype=np.float64)
        metrics, _ = formation_geometry(
            actions,
            centers,
            path_distances,
        )
        self.assertAlmostEqual(metrics["minimum_progress_gap"], 0.01)

    def test_formation_geometry_detects_action_path_coupling(self):
        actions = np.zeros((4, 3, 2), dtype=np.float64)
        actions[:, :, 0] = np.arange(4)[:, None]
        centers = np.stack(
            [
                np.cumsum(np.exp(np.linspace(-scale, scale, 3)))
                for scale in (0.2, 0.5, 0.8, 1.1)
            ]
        )
        centers = centers / centers[:, -1:]
        signatures = actions.reshape(4, -1)
        differences = signatures[:, None] - signatures[None]
        path_distances = np.linalg.norm(differences, axis=-1)
        metrics, action_distances = formation_geometry(
            actions,
            centers,
            path_distances,
        )
        self.assertGreater(
            metrics["action_path_distance_correlation"],
            0.99,
        )
        self.assertGreater(metrics["progress_schedule_diversity"], 0.0)
        self.assertTrue(np.allclose(action_distances, action_distances.T))

    def test_action_path_pairing_beats_question_local_null(self):
        rng = np.random.default_rng(7)
        actions = []
        paths = []
        for _ in range(20):
            signatures = rng.normal(size=(8, 6))
            differences = signatures[:, None] - signatures[None]
            distance = np.linalg.norm(differences, axis=-1)
            actions.append(distance)
            paths.append(distance.copy())
        report = action_path_permutation_null(
            np.stack(actions),
            np.stack(paths),
            permutations=128,
            rng=np.random.default_rng(8),
        )
        self.assertGreater(report["observed_mean"], 0.99)
        self.assertGreater(report["excess"], 0.5)
        self.assertLess(report["p_value"], 0.02)

    def test_question_heldout_probe_uses_whole_question_folds(self):
        rng = np.random.default_rng(19)
        labels = np.tile(np.asarray([0, 1, 0, 1]), (10, 1))
        features = rng.normal(scale=0.05, size=(10, 4, 3))
        features[..., 0] += labels * 2.0 - 1.0
        report = question_heldout_probe(
            features,
            labels,
            folds=5,
        )
        self.assertGreater(report["auroc"], 0.99)
        self.assertGreater(report["balanced_accuracy"], 0.95)

    def test_question_heldout_probe_rejects_single_class(self):
        with self.assertRaises(ValueError):
            question_heldout_probe(
                np.zeros((10, 4, 3)),
                np.ones((10, 4)),
                folds=5,
            )


class CausalEvidenceTests(unittest.TestCase):
    def test_commit_bottleneck_curve_is_not_stepwise_evidence(self):
        accuracy = np.asarray(
            [[0.0] * 8 + [1.0], [1.0] * 8 + [0.0]],
            dtype=np.float64,
        )
        score = np.asarray(
            [[-4.0] * 8 + [-1.0], [-2.0] * 8 + [-3.0]],
            dtype=np.float64,
        )
        report = summarize_commit_bottleneck_sanity(accuracy, score)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["prefix_0_through_7_equal_no_readout"])
        self.assertIn(
            "not measure incremental step sufficiency",
            report["interpretation"],
        )

    def test_commit_bottleneck_curve_fails_if_private_state_is_readable(self):
        accuracy = np.zeros((2, 9), dtype=np.float64)
        score = np.zeros((2, 9), dtype=np.float64)
        score[0, 4] = 0.01
        report = summarize_commit_bottleneck_sanity(accuracy, score)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["prefix_0_through_7_equal_no_readout"])

    def test_transition_familywise_interval_is_more_conservative(self):
        rng = np.random.default_rng(17)
        values = rng.normal(loc=0.05, scale=0.3, size=(200, 8))
        report = matrix_mean_ci(
            values,
            rng=np.random.default_rng(18),
            bootstrap=10000,
        )
        point_low = np.asarray(report["ci95_low"])
        point_high = np.asarray(report["ci95_high"])
        family_low = np.asarray(
            report["simultaneous_familywise95_low"]
        )
        family_high = np.asarray(
            report["simultaneous_familywise95_high"]
        )
        self.assertTrue(np.all(family_low <= point_low))
        self.assertTrue(np.all(family_high >= point_high))
        self.assertEqual(report["family_size"], 8)
        self.assertEqual(
            report["simultaneous_interval_correction"],
            "Bonferroni bootstrap percentile",
        )


class SupervisorCleanupTests(unittest.TestCase):
    def test_cleanup_kills_orphaned_descendant_after_leader_exits(self):
        leader = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess; "
                    "subprocess.Popen(['sleep', '60']); "
                    "raise SystemExit(0)"
                ),
            ],
            start_new_session=True,
        )
        process_group_id = leader.pid
        leader.wait(timeout=5)
        deadline = time.monotonic() + 2.0
        while (
            not process_group_exists(process_group_id)
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        try:
            self.assertTrue(process_group_exists(process_group_id))
            terminate_process_group(leader, timeout_seconds=2)
            self.assertFalse(process_group_exists(process_group_id))
        finally:
            if process_group_exists(process_group_id):
                os.killpg(process_group_id, 9)


if __name__ == "__main__":
    unittest.main()
