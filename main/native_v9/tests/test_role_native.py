import unittest

import torch

from src.modules.role_native import (
    COMMIT_INDEX,
    N_ROLES,
    RoleLatentPolicy,
    build_role_targets,
    clipped_policy_loss,
    contiguous_spans,
    discounted_role_returns,
    group_standardize,
)
from src.models.trace_bridge import LitTRACEBridge


class RoleTargetTests(unittest.TestCase):
    def test_contiguous_partition_preserves_every_step(self):
        for length in range(1, 17):
            spans = contiguous_spans(length)
            flattened = [index for start, end in spans for index in range(start, end)]
            self.assertEqual(flattened, list(range(length)))
            self.assertEqual(len(spans), 5)

    def test_targets_use_observed_ordered_transitions(self):
        residuals = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        states = residuals + 100.0
        targets = build_role_targets(
            [{"step_residuals": residuals, "step_states": states}]
        )
        self.assertEqual(tuple(targets.solve.shape), (1, 5, 2))
        self.assertTrue(targets.solve_mask.all())
        self.assertTrue(torch.equal(targets.solve[0, 0], residuals[:2].sum(0)))
        self.assertTrue(torch.equal(targets.refine[0], residuals[-1]))
        self.assertTrue(torch.equal(targets.commit[0], states[-1]))

    def test_one_step_progress_metrics_keep_distributed_log_schema(self):
        model = LitTRACEBridge.__new__(LitTRACEBridge)
        torch.nn.Module.__init__(model)
        model.trace_config = {
            "stage1_progress_anchor_mix": 0.45,
            "stage1_progress_anchor_sigma": 0.18,
        }
        outputs = model._apply_trace_progress_anchors(
            explicit_features=[{"step_residuals": torch.randn(1, 4)}],
            compression_outputs={
                "assignments": torch.ones(1, 8, 1),
                "aggregated_explicit_residuals": torch.zeros(1, 8, 4),
            },
        )
        self.assertIn("trace_progress_anchor_span", outputs)
        self.assertIn("trace_progress_anchor_mean", outputs)
        self.assertEqual(outputs["trace_progress_anchor_span"].item(), 0.0)
        self.assertEqual(outputs["trace_progress_anchor_mean"].item(), 0.0)


class RolePolicyTests(unittest.TestCase):
    def test_zero_initial_mean_preserves_block_prior(self):
        torch.manual_seed(3)
        policy = RoleLatentPolicy(hidden_size=24, action_dim=4, policy_hidden_size=16)
        states = torch.randn(2, 24)
        for role_index in range(N_ROLES):
            action, log_prob, mean, _ = policy.realize(
                states, role_index, deterministic=True
            )
            self.assertTrue(torch.equal(action, mean))
            self.assertTrue(torch.equal(mean, torch.zeros_like(mean)))
            if role_index == COMMIT_INDEX:
                self.assertTrue(torch.equal(log_prob, torch.zeros_like(log_prob)))

    def test_group_credit_is_question_local(self):
        values = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [101.0, 102.0], [103.0, 104.0]]
        )
        normalized = group_standardize(values, group_size=2)
        self.assertTrue(torch.allclose(normalized[:2].mean(0), torch.zeros(2)))
        self.assertTrue(torch.allclose(normalized[2:].mean(0), torch.zeros(2)))

    def test_returns_exclude_deterministic_commit(self):
        scores = torch.ones(2, N_ROLES)
        returns = discounted_role_returns(scores, gamma=1.0)
        self.assertTrue(torch.equal(returns[:, COMMIT_INDEX], torch.zeros(2)))
        self.assertTrue(torch.equal(returns[:, 6], torch.ones(2)))
        self.assertTrue(torch.equal(returns[:, 0], torch.full((2,), 7.0)))

    def test_clipped_policy_loss_is_finite(self):
        current = torch.zeros(2, N_ROLES)
        old = torch.zeros_like(current)
        advantage = torch.linspace(-1.0, 1.0, 2 * N_ROLES).reshape_as(current)
        mask = torch.ones_like(current, dtype=torch.bool)
        loss = clipped_policy_loss(current, old, advantage, mask, 0.12)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
