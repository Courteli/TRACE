import inspect
import re
import unittest
from pathlib import Path

import torch

from src.models.trace_policy import LitTRACEPolicy
from src.modules import trace_policy as trace_policy_module
from src.modules.trace_policy import GaussianTrajectoryPolicy


ROOT = Path(__file__).resolve().parents[1]


def _role_name(value) -> str:
    value = getattr(value, "value", value)
    return str(value).split(".")[-1].lower()


def _first_callable(owner, names):
    for name in names:
        value = getattr(owner, name, None)
        if callable(value):
            return value
    raise AssertionError(
        "missing role-semantic API; expected one of: " + ", ".join(names)
    )


def _role_head_container(policy, kind: str):
    candidates = (
        f"role_{kind}_heads",
        f"{kind}_heads",
        "role_heads",
    )
    for name in candidates:
        container = getattr(policy, name, None)
        if container is not None:
            return container
    direct = getattr(policy, f"solve_{kind}_head", None)
    if direct is not None:
        return {"solve": direct}
    raise AssertionError(
        f"GaussianTrajectoryPolicy must expose a shared SOLVE {kind} head"
    )


class RoleLayoutContractTests(unittest.TestCase):
    def test_fixed_eight_state_role_sequence(self):
        sequence = getattr(
            trace_policy_module,
            "ROLE_SEQUENCE",
            getattr(trace_policy_module, "TRACE_ROLE_NAMES", None),
        )
        self.assertIsNotNone(
            sequence,
            "src.modules.trace_policy must publish ROLE_SEQUENCE",
        )
        self.assertEqual(
            tuple(_role_name(role) for role in sequence),
            (
                "plan",
                "solve",
                "solve",
                "solve",
                "solve",
                "solve",
                "check",
                "commit",
            ),
        )

    def test_five_solve_positions_share_one_distribution_head(self):
        policy = GaussianTrajectoryPolicy(
            hidden_size=12,
            action_dim=4,
            n_steps=8,
            policy_hidden_size=16,
        )
        role_for_step = getattr(policy, "role_for_step", None)
        if role_for_step is None:
            sequence = getattr(
                policy,
                "role_sequence",
                getattr(trace_policy_module, "TRACE_ROLE_NAMES", None),
            )
            self.assertIsNotNone(
                sequence,
                "policy needs role_for_step() or role_sequence",
            )
            roles = tuple(_role_name(role) for role in sequence)
        else:
            roles = tuple(_role_name(role_for_step(i)) for i in range(8))
        self.assertEqual(roles[1:6], ("solve",) * 5)

        for kind in ("mean", "log_std"):
            container = _role_head_container(policy, kind)
            normalized_keys = {
                _role_name(key): key for key in getattr(container, "keys", lambda: [])()
            }
            if normalized_keys:
                self.assertIn("solve", normalized_keys)
                if kind == "log_std":
                    self.assertNotIn(
                        "commit",
                        normalized_keys,
                        "deterministic COMMIT must not own a log-std head",
                    )
                solve_head = container[normalized_keys["solve"]]
            else:
                solve_head = container["solve"]
            self.assertIsInstance(solve_head, torch.nn.Module)

        numbered_solve_heads = [
            name
            for name, _ in policy.named_modules()
            if re.search(r"solve[_\.](?:1|2|3|4|5)[_\.]", name.lower())
        ]
        self.assertEqual(
            numbered_solve_heads,
            [],
            "SOLVE slots must reuse one head, not five position-specific heads",
        )


class MonotoneSolveTargetTests(unittest.TestCase):
    @staticmethod
    def _target_api():
        return _first_callable(
            trace_policy_module,
            (
                "build_contiguous_cot_targets",
                "build_monotone_solve_targets",
            ),
        )

    def test_chunking_is_contiguous_complete_and_never_exceeds_five(self):
        build_targets = self._target_api()
        for n_steps, expected_chunks in ((1, 1), (4, 4), (5, 5), (8, 5)):
            with self.subTest(n_steps=n_steps):
                residuals = torch.arange(
                    1,
                    n_steps + 1,
                    dtype=torch.float32,
                ).unsqueeze(-1)
                result = build_targets(residuals, n_chunks=5)
                spans = [tuple(map(int, span)) for span in result.spans]
                active_slots = result.mask.nonzero(as_tuple=False).flatten().tolist()
                active_spans = [
                    spans[index]
                    for index in active_slots
                ]
                self.assertEqual(tuple(result.targets.shape), (5, 1))
                self.assertEqual(tuple(result.mask.shape), (5,))
                self.assertEqual(len(active_slots), expected_chunks)
                self.assertEqual(active_slots[-1], 4)
                self.assertEqual(active_spans[0][0], 0)
                self.assertEqual(active_spans[-1][1], n_steps)
                self.assertTrue(
                    all(start < end for start, end in active_spans)
                )
                self.assertTrue(
                    all(
                        active_spans[index][1] == active_spans[index + 1][0]
                        for index in range(len(active_spans) - 1)
                    )
                )
                if n_steps <= 5:
                    self.assertEqual(
                        active_spans,
                        [(index, index + 1) for index in range(n_steps)],
                    )
                self.assertGreaterEqual(
                    float(result.targets[-1, 0]),
                    float(n_steps),
                )

    def test_single_step_target_is_supervised_at_last_solve_slot(self):
        residual = torch.tensor([[7.0, -3.0]])
        result = self._target_api()(residual, n_chunks=5)
        self.assertEqual(
            result.mask.tolist(),
            [False, False, False, False, True],
        )
        self.assertTrue(torch.equal(result.targets[-1], residual[0]))
        self.assertEqual(float(result.targets[:-1].abs().sum()), 0.0)


class CommitReadoutContractTests(unittest.TestCase):
    @staticmethod
    def _mask_api():
        for owner in (LitTRACEPolicy, trace_policy_module):
            for name in (
                "_commit_only_latent_mask",
                "_commit_latent_read_mask",
                "commit_only_latent_mask",
            ):
                value = getattr(owner, name, None)
                if callable(value):
                    return value
        raise AssertionError(
            "missing commit-only mask API on LitTRACEPolicy or trace_policy module"
        )

    def test_commit_mask_reads_exactly_the_last_available_state(self):
        available = torch.ones(2, 8, dtype=torch.long)
        mask = self._mask_api()(available)
        expected = torch.zeros_like(available)
        expected[:, -1] = 1
        self.assertEqual(mask.dtype, available.dtype)
        self.assertEqual(mask.device, available.device)
        self.assertTrue(torch.equal(mask, expected))

        unavailable_commit = available.clone()
        unavailable_commit[1, -1] = 0
        masked = self._mask_api()(unavailable_commit)
        expected[1, -1] = 0
        self.assertTrue(torch.equal(masked, expected))

    def test_all_answer_paths_resolve_the_commit_only_mask(self):
        resolver_source = inspect.getsource(
            LitTRACEPolicy._resolve_latent_read_mask
        ).lower()
        self.assertIn("commit", resolver_source)

        for method_name in (
            "_teacher_force_bottleneck",
            "_generate_answers_from_trajectory",
            "_answer_token_log_probs",
        ):
            with self.subTest(method=method_name):
                source = inspect.getsource(
                    getattr(LitTRACEPolicy, method_name)
                )
                self.assertIn(
                    "_resolve_latent_read_mask",
                    source,
                    f"{method_name} must use the same commit-only readout contract",
                )


class StepReturnContractTests(unittest.TestCase):
    def test_discounted_returns_match_hand_calculation(self):
        discounted_returns = _first_callable(
            trace_policy_module,
            (
                "build_discounted_role_returns",
                "discounted_step_returns",
                "discounted_role_returns",
                "reverse_discounted_returns",
            ),
        )
        terminal = torch.tensor([10.0, -1.0])
        step_rewards = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 99.0],
                [0.0, -2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 99.0],
            ]
        )
        actual = discounted_returns(
            terminal,
            step_rewards,
            gamma=0.5,
            step_reward_weight=0.25,
        )
        expected = torch.tensor(
            [
                [10.6875, 10.875, 10.75, 10.0, 10.0, 10.0, 10.0, 10.0],
                [-1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            ]
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

        undiscounted = discounted_returns(
            terminal,
            step_rewards,
            gamma=0.0,
            step_reward_weight=1.0,
        )
        expected_undiscounted = terminal.unsqueeze(-1) + step_rewards
        expected_undiscounted[:, -1] = terminal
        self.assertTrue(torch.equal(undiscounted, expected_undiscounted))


class Stage2LatentOnlyContractTests(unittest.TestCase):
    def test_stage2_never_updates_or_invokes_answer_policy(self):
        training_source = inspect.getsource(
            LitTRACEPolicy.trace_rl_training_step
        )
        rollout_source = inspect.getsource(
            LitTRACEPolicy.trace_policy_rollout
        )
        self.assertNotIn("_answer_policy_update", training_source)
        self.assertNotIn("answer_temperature", rollout_source)
        self.assertNotIn("answer_top_p", rollout_source)
        self.assertNotRegex(rollout_source, r"do_sample\s*=\s*True")

        config_text = (
            ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml"
        ).read_text(encoding="utf-8")
        self.assertNotRegex(
            config_text.lower(),
            r"use_answer_policy_loss\s*:\s*true",
        )
        for script in (ROOT / "scripts").glob("*.sh"):
            text = script.read_text(encoding="utf-8")
            self.assertNotRegex(
                text.lower(),
                r"use_answer_policy_loss\s*=\s*true",
                msg=str(script),
            )


class NewTreeIsolationContractTests(unittest.TestCase):
    def test_scripts_do_not_hardcode_the_old_source_tree(self):
        forbidden = (
            "/disk1/dingxukai/TRACE/canonical66/stage2_adaptive",
            "canonical66/stage2_adaptive",
        )
        for script in (ROOT / "scripts").glob("*.sh"):
            text = script.read_text(encoding="utf-8")
            for old_root in forbidden:
                self.assertNotIn(old_root, text, msg=str(script))

    def test_training_entrypoints_resolve_code_from_their_own_tree(self):
        entrypoints = (
            "run_stage0_cot.sh",
            "run_stage1_formation.sh",
            "run_stage2_refinement.sh",
            "run_full_pipeline.sh",
        )
        for name in entrypoints:
            with self.subTest(script=name):
                text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn("BASH_SOURCE[0]", text)
                self.assertRegex(
                    text,
                    r"(?:PROJECT_ROOT|ROOT)=\$\(cd\s+\"?\$?\{?SCRIPT_DIR",
                )
                self.assertNotRegex(
                    text,
                    r"(?m)^(?:PROJECT_ROOT|ROOT)=/disk1/dingxukai/TRACE\s*$",
                )


if __name__ == "__main__":
    unittest.main()
