import copy
from types import SimpleNamespace
import unittest

import torch

from src.modules.trace_policy import GaussianTrajectoryPolicy, gaussian_log_prob
from src.modules.trace_vb import RoleConditionedScalarHead
from tools.trace_vb_stage2_ddp_smoke import (
    FLOAT32_ROLLOUT_FIELDS,
    Stage2HeadDDPProxy,
    validate_rollout,
)


def make_proxy_and_rollout():
    torch.manual_seed(19)
    actor = GaussianTrajectoryPolicy(
        hidden_size=16,
        action_dim=4,
        n_steps=8,
        policy_hidden_size=12,
        step_embedding_size=3,
    )
    actor.set_stage2_trainability()
    reference = copy.deepcopy(actor).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    critic = RoleConditionedScalarHead(
        hidden_size=16,
        n_roles=8,
        role_embedding_dim=4,
        head_hidden_size=8,
    )
    owner = SimpleNamespace(
        trajectory_policy=actor,
        value_critic=critic,
        stage1_policy_reference=reference,
        n_trace_steps=8,
        trace_rl_config={
            "trajectory_clip_epsilon": 0.12,
            "stage1_policy_kl_weight": 0.02,
            "entropy_coefficient": 0.001,
            "value_loss_coefficient": 0.5,
            "value_loss_type": "huber",
            "role_entropy_weights": [1.0] * 7 + [0.0],
        },
    )
    proxy = Stage2HeadDDPProxy(owner)
    states = torch.randn(8, 8, 16)
    parameters = [
        actor.distribution_parameters(states[:, index], index)
        for index in range(8)
    ]
    means = torch.stack([item[0] for item in parameters], dim=1)
    log_stds = torch.stack([item[1] for item in parameters], dim=1)
    actions = (means + log_stds.exp() * torch.randn_like(means)).detach()
    old_log_probs = gaussian_log_prob(actions, means, log_stds).detach()
    action_mask = torch.ones(8, 8, dtype=torch.bool)
    action_mask[:, 7] = False
    terminal_rewards = torch.tensor(
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    )
    rollout = {
        "pre_action_states": states,
        "actions": actions,
        "old_action_log_probs": old_log_probs,
        "advantages": torch.randn(8, 8),
        "raw_advantages": torch.randn(8, 8),
        "returns": torch.rand(8, 8) * action_mask,
        "values": torch.rand(8, 8),
        "action_mask": action_mask,
        "role_ids": torch.arange(8).view(1, -1).expand(8, -1),
        "terminal_rewards": terminal_rewards,
        "greedy_accuracy": terminal_rewards.clone(),
        "greedy_lengths": torch.ones(8),
    }
    rollout["advantages"][:, 7] = 0.0
    rollout["raw_advantages"][:, 7] = 0.0
    return actor, critic, proxy, rollout


class Stage2HeadDDPProxyTests(unittest.TestCase):
    def test_warmup_then_two_active_evaluations_expose_nonunit_ratio(self):
        actor, critic, proxy, rollout = make_proxy_and_rollout()
        actor_parameters = [
            parameter for parameter in actor.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": actor_parameters, "lr": 8.0e-7},
                {"params": critic.parameters(), "lr": 1.0e-4},
            ]
        )

        warmup = proxy(rollout, False)
        warmup["objective"].backward()
        self.assertTrue(all(p.grad is None for p in actor_parameters))
        self.assertTrue(any(p.grad is not None for p in critic.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        first = proxy(rollout, True)
        self.assertEqual(first["objective"].dtype, torch.float32)
        self.assertEqual(float(first["ratio_deviation"]), 0.0)
        first["objective"].backward()
        self.assertTrue(any(p.grad is not None for p in actor_parameters))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        second = proxy(rollout, True)
        self.assertGreater(float(second["ratio_deviation"]), 0.0)

    def test_rollout_contract_fails_closed_on_dtype_and_shaping(self):
        _, _, _, rollout = make_proxy_and_rollout()
        self.assertEqual(
            {rollout[name].dtype for name in FLOAT32_ROLLOUT_FIELDS},
            {torch.float32},
        )
        validate_rollout(rollout, torch.device("cpu"))

        wrong_dtype = dict(rollout)
        wrong_dtype["actions"] = wrong_dtype["actions"].to(torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "not FP32"):
            validate_rollout(wrong_dtype, torch.device("cpu"))

        shaped = dict(rollout)
        shaped["semantic_step_rewards"] = torch.zeros(8, 8)
        with self.assertRaisesRegex(RuntimeError, "shaped rewards"):
            validate_rollout(shaped, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
