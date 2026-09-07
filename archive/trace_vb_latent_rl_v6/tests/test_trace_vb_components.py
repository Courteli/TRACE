import unittest

import torch

from src.modules.trace_policy import GaussianTrajectoryPolicy
from src.modules.trace_vb import (
    LatentStepTextDecoder,
    PlanForecastHead,
    RoleConditionedScalarHead,
    action_efficacy_hinge,
    evidence_gated_group_advantages,
    masked_ppo_actor_loss,
    masked_terminal_reward_gae,
    masked_value_loss,
    per_sequence_token_cross_entropy,
)


class EvidenceGatedGroupAdvantageTests(unittest.TestCase):
    def test_mixed_groups_use_exact_outcomes_and_homogeneous_groups_fail_closed(self):
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
        result = evidence_gated_group_advantages(
            correctness,
            scores,
            group_size=8,
            enable_likelihood_fallback=False,
            minimum_gold_score_gap=0.002,
        )

        self.assertTrue(torch.equal(result.all_correct_group_mask, torch.tensor([1, 0, 0], dtype=torch.bool)))
        self.assertTrue(torch.equal(result.mixed_group_mask, torch.tensor([0, 1, 0], dtype=torch.bool)))
        self.assertTrue(torch.equal(result.all_wrong_group_mask, torch.tensor([0, 0, 1], dtype=torch.bool)))
        self.assertTrue(torch.equal(result.path_advantages[:8], torch.zeros(8)))
        self.assertGreater(float(result.path_advantages[8]), 0.0)
        self.assertTrue(torch.all(result.path_advantages[9:16] < 0.0))
        self.assertTrue(torch.equal(result.path_advantages[16:], torch.zeros(8)))
        self.assertEqual(float(result.proxy_pair_count), 7.0)
        self.assertEqual(float(result.proxy_pair_credit), 7.0)

    def test_calibrated_all_wrong_fallback_uses_only_question_local_rank(self):
        correctness = torch.zeros(8)
        scores = torch.tensor([0.2, 0.8, 0.4, 0.1, 0.7, 0.6, 0.3, 0.5])
        result = evidence_gated_group_advantages(
            correctness,
            scores,
            group_size=8,
            enable_likelihood_fallback=True,
            minimum_gold_score_gap=0.002,
        )

        self.assertTrue(bool(result.likelihood_group_mask.item()))
        self.assertTrue(torch.equal(result.failure_entropy_mask, torch.ones(8, dtype=torch.bool)))
        order = scores.argsort()
        ordered_advantages = result.path_advantages[order]
        self.assertTrue(torch.all(ordered_advantages[1:] > ordered_advantages[:-1]))
        self.assertAlmostEqual(float(result.path_advantages.mean()), 0.0, places=6)
        self.assertAlmostEqual(
            float(result.path_advantages.std(unbiased=False)),
            1.0,
            places=5,
        )

    def test_numerically_flat_scores_cannot_manufacture_preference(self):
        result = evidence_gated_group_advantages(
            torch.zeros(8),
            torch.linspace(0.0, 0.001, 8),
            group_size=8,
            enable_likelihood_fallback=True,
            minimum_gold_score_gap=0.002,
        )
        self.assertFalse(bool(result.likelihood_group_mask.item()))
        self.assertTrue(torch.equal(result.path_advantages, torch.zeros(8)))


class ActionEfficacyHingeTests(unittest.TestCase):
    def test_rms_normalized_hand_calculation_and_commit_mask(self):
        # Four unit action changes have RMS distance 1; nine transition
        # changes of .5 have RMS distance .5.  With minimum_ratio=.75 the
        # active hinge is .25 and the active efficacy ratio is .5.
        sampled_actions = torch.ones(1, 2, 4)
        map_actions = torch.zeros_like(sampled_actions)
        sampled_transitions = torch.full((1, 2, 9), 0.5)
        map_transitions = torch.zeros_like(sampled_transitions)
        action_mask = torch.tensor([[1, 0]], dtype=torch.bool)

        result = action_efficacy_hinge(
            sampled_actions,
            map_actions,
            sampled_transitions,
            map_transitions,
            action_mask,
            0.75,
        )

        self.assertAlmostEqual(float(result.loss), 0.25, places=6)
        self.assertAlmostEqual(
            float(result.active_efficacy_ratio),
            0.5,
            places=6,
        )

        # Make deterministic COMMIT maximally violating.  Its zero mask must
        # keep both returned active statistics unchanged.
        sampled_transitions[:, 1] = 0.0
        remasked = action_efficacy_hinge(
            sampled_actions,
            map_actions,
            sampled_transitions,
            map_transitions,
            action_mask,
            0.75,
        )
        self.assertAlmostEqual(float(remasked.loss), 0.25, places=6)
        self.assertAlmostEqual(
            float(remasked.active_efficacy_ratio),
            0.5,
            places=6,
        )

    def test_action_distance_is_detached_and_transition_receives_gradient(self):
        sampled_actions = torch.ones(1, 2, 4, requires_grad=True)
        map_actions = torch.zeros_like(sampled_actions)
        sampled_transitions = torch.full(
            (1, 2, 9),
            0.25,
            requires_grad=True,
        )
        map_transitions = torch.zeros_like(sampled_transitions)
        action_mask = torch.tensor([[1, 0]], dtype=torch.bool)

        result = action_efficacy_hinge(
            sampled_actions,
            map_actions,
            sampled_transitions,
            map_transitions,
            action_mask,
            0.75,
        )
        result.loss.backward()

        self.assertIsNone(sampled_actions.grad)
        self.assertGreater(
            float(sampled_transitions.grad[:, 0].abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(sampled_transitions.grad[:, 1].abs().sum()),
            0.0,
        )

    def test_strict_shape_and_positive_ratio_validation(self):
        actions = torch.zeros(1, 2, 4)
        transitions = torch.zeros(1, 2, 6)
        with self.assertRaises(ValueError):
            action_efficacy_hinge(
                actions,
                actions,
                transitions,
                transitions,
                torch.ones(1, 2, dtype=torch.bool),
                0.0,
            )
        with self.assertRaises(ValueError):
            action_efficacy_hinge(
                actions,
                actions,
                transitions,
                transitions,
                torch.ones(2, dtype=torch.bool),
                0.5,
            )


class HeadOnlyActorContractTests(unittest.TestCase):
    def test_stage2_update_cannot_change_commit_mapping(self):
        torch.manual_seed(13)
        policy = GaussianTrajectoryPolicy(
            hidden_size=12,
            action_dim=4,
            n_steps=8,
            policy_hidden_size=16,
        )
        states = torch.randn(3, 12)
        before = policy.distribution_parameters(states, 7)[0].detach().clone()
        policy.set_stage2_trainability()
        trainable_names = {
            name for name, value in policy.named_parameters()
            if value.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(
                name.startswith("mean_heads.")
                or name.startswith("log_std_heads.")
                for name in trainable_names
            )
        )
        self.assertFalse(
            any(".commit." in f".{name}." for name in trainable_names)
        )
        optimizer = torch.optim.SGD(
            [value for value in policy.parameters() if value.requires_grad],
            lr=0.1,
        )
        mean, log_std = policy.distribution_parameters(states, 0)
        (mean.square().mean() + log_std.square().mean()).backward()
        optimizer.step()
        after = policy.distribution_parameters(states, 7)[0].detach()
        self.assertTrue(torch.equal(before, after))


class PlanForecastHeadTests(unittest.TestCase):
    def test_preserves_leading_axes_and_predicts_five_targets(self):
        head = PlanForecastHead(
            hidden_size=12,
            target_dim=4,
            n_targets=5,
            head_hidden_size=16,
        )
        states = torch.randn(2, 3, 12, requires_grad=True)
        predictions = head(states)

        self.assertEqual(tuple(predictions.shape), (2, 3, 5, 4))
        predictions.square().mean().backward()
        self.assertGreater(float(states.grad.abs().sum()), 0.0)


class LatentStepTextDecoderTests(unittest.TestCase):
    def test_decoder_is_conditioned_on_residual_and_backpropagates_to_it(self):
        torch.manual_seed(17)
        decoder = LatentStepTextDecoder(
            hidden_size=12,
            decoder_hidden_size=8,
            n_solve_roles=5,
        )
        residuals = torch.randn(3, 12, requires_grad=True)
        previous_tokens = torch.randn(3, 4, 12)
        roles = torch.tensor([0, 2, 4], dtype=torch.long)

        states = decoder(residuals, previous_tokens, roles)

        self.assertEqual(tuple(states.shape), (3, 4, 12))
        states.square().mean().backward()
        self.assertGreater(float(residuals.grad.abs().sum()), 0.0)

    def test_decoder_rejects_invalid_role_and_extra_context_shape(self):
        decoder = LatentStepTextDecoder(
            hidden_size=6,
            decoder_hidden_size=4,
            n_solve_roles=5,
        )
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            decoder(
                torch.randn(1, 6),
                torch.randn(1, 2, 6),
                torch.tensor([5]),
            )
        with self.assertRaisesRegex(ValueError, "previous_token_embeddings"):
            decoder(
                torch.randn(1, 6),
                torch.randn(1, 2, 7),
                torch.tensor([0]),
            )

    def test_per_sequence_ce_masks_padding_and_normalizes_per_record(self):
        logits = torch.tensor(
            [
                [[3.0, 0.0], [0.0, 3.0]],
                [[0.0, 3.0], [3.0, 0.0]],
            ],
            requires_grad=True,
        )
        labels = torch.tensor([[0, 1], [1, 1]], dtype=torch.long)
        mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

        losses = per_sequence_token_cross_entropy(
            logits,
            labels,
            mask,
            token_chunk_size=1,
        )

        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[3.0, 0.0]]),
            torch.tensor([0]),
        )
        self.assertEqual(tuple(losses.shape), (2,))
        self.assertTrue(torch.allclose(losses, expected.expand_as(losses)))
        losses.sum().backward()
        self.assertEqual(float(logits.grad[1, 1].abs().sum()), 0.0)


class RoleConditionedScalarHeadTests(unittest.TestCase):
    def test_role_conditioned_shape_and_exact_sufficiency_to_value_copy(self):
        torch.manual_seed(7)
        sufficiency = RoleConditionedScalarHead(
            hidden_size=10,
            n_roles=4,
            role_embedding_dim=3,
            head_hidden_size=11,
        )
        critic = RoleConditionedScalarHead(
            hidden_size=10,
            n_roles=4,
            role_embedding_dim=3,
            head_hidden_size=11,
        )
        states = torch.randn(2, 8, 10)
        role_ids = torch.tensor(
            [
                [0, 1, 1, 1, 1, 1, 2, 3],
                [0, 1, 1, 1, 1, 1, 2, 3],
            ]
        )

        critic.copy_from(sufficiency)
        source_values = sufficiency(states, role_ids)
        copied_values = critic(states, role_ids)

        self.assertEqual(tuple(source_values.shape), (2, 8))
        self.assertTrue(torch.equal(source_values, copied_values))
        for source_parameter, critic_parameter in zip(
            sufficiency.parameters(),
            critic.parameters(),
        ):
            self.assertNotEqual(
                source_parameter.data_ptr(),
                critic_parameter.data_ptr(),
            )

    def test_copy_rejects_nonisomorphic_head(self):
        source = RoleConditionedScalarHead(hidden_size=10, n_roles=4)
        destination = RoleConditionedScalarHead(hidden_size=10, n_roles=3)
        with self.assertRaises(ValueError):
            destination.copy_from(source)


class TerminalRewardGAETests(unittest.TestCase):
    def test_hand_computed_gae_masks_deterministic_commit(self):
        # The final slot represents deterministic COMMIT.  With zero values,
        # gamma=1, and lambda=.95, a unit terminal reward gives exactly
        # [lambda^2, lambda, 1] on the three preceding stochastic actions.
        values = torch.zeros(2, 4, requires_grad=True)
        terminal_rewards = torch.tensor([1.0, 0.0])
        stochastic_mask = torch.tensor([1, 1, 1, 0], dtype=torch.bool)

        result = masked_terminal_reward_gae(
            terminal_rewards,
            values,
            stochastic_mask,
            gamma=1.0,
            gae_lambda=0.95,
        )
        expected = torch.tensor(
            [
                [0.95**2, 0.95, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertTrue(torch.allclose(result.advantages, expected))
        self.assertTrue(torch.allclose(result.returns, expected))
        self.assertFalse(result.advantages.requires_grad)
        self.assertFalse(result.returns.requires_grad)

    def test_masked_holes_are_skipped_not_treated_as_terminal(self):
        values = torch.zeros(1, 4)
        result = masked_terminal_reward_gae(
            torch.ones(1),
            values,
            torch.tensor([[1, 0, 1, 0]], dtype=torch.bool),
            gamma=1.0,
            gae_lambda=0.5,
        )
        self.assertTrue(
            torch.equal(
                result.advantages,
                torch.tensor([[0.5, 0.0, 1.0, 0.0]]),
            )
        )


class MaskedActorCriticLossTests(unittest.TestCase):
    def test_ppo_mask_removes_commit_loss_and_gradient(self):
        current_log_probs = torch.zeros(1, 4, requires_grad=True)
        old_log_probs = torch.zeros(1, 4)
        advantages = torch.tensor([[1.0, 2.0, 3.0, 1000.0]])
        action_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

        loss = masked_ppo_actor_loss(
            current_log_probs,
            old_log_probs,
            advantages,
            action_mask,
        )
        self.assertAlmostEqual(float(loss), -2.0)
        loss.backward()
        self.assertEqual(float(current_log_probs.grad[0, 3]), 0.0)

    def test_second_ppo_epoch_has_nonunit_probability_ratio(self):
        current_log_probs = torch.nn.Parameter(torch.zeros(1, 4))
        old_log_probs = current_log_probs.detach().clone()
        advantages = torch.ones(1, 4)
        action_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        optimizer = torch.optim.SGD([current_log_probs], lr=0.05)

        first_loss = masked_ppo_actor_loss(
            current_log_probs,
            old_log_probs,
            advantages,
            action_mask,
        )
        first_loss.backward()
        optimizer.step()

        second_ratio = torch.exp(
            current_log_probs.detach() - old_log_probs
        )
        self.assertFalse(
            torch.allclose(
                second_ratio[:, :3],
                torch.ones_like(second_ratio[:, :3]),
            )
        )
        self.assertEqual(float(second_ratio[0, 3]), 1.0)

    def test_value_loss_ignores_commit_target_and_gradient(self):
        predicted = torch.tensor(
            [[0.0, 0.0, 100.0]],
            requires_grad=True,
        )
        targets = torch.tensor([[1.0, 3.0, -100.0]])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

        loss = masked_value_loss(
            predicted,
            targets,
            mask,
            loss_type="mse",
        )
        self.assertEqual(float(loss), 5.0)
        loss.backward()
        self.assertEqual(float(predicted.grad[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
