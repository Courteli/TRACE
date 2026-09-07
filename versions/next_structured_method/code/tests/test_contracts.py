from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from fixtures import examples, tiny_config, tiny_model
from trace_structured.config import Config
from trace_structured.data import answer_matches, audit_data, load_jsonl, load_split, normalize_number, output_parts, parse_row, progress_anchors, question_prompt
from trace_structured.model import token_mask
from trace_structured.policy import group_advantage, process_returns, surrogate
from trace_structured.targets import build_targets, process_scores, role_quantities
from trace_structured.training import rl_backward


class DataTests(unittest.TestCase):
    def test_actual_plain_splits(self):
        root = Path(__file__).resolve().parents[4]
        result = audit_data(root / "data/GSM8k-Aug-NL")
        self.assertEqual([result[x]["count"] for x in ("train", "val", "test")], [6726, 747, 1319])

    def test_actual_tokenizer_budgets_and_numeric_labels(self):
        from transformers import AutoTokenizer
        root = Path(__file__).resolve().parents[4]
        tokenizer = AutoTokenizer.from_pretrained(root / 'models/base_reference', local_files_only=True)
        config = Config()
        def length(text):
            return len(tokenizer.encode(text, add_special_tokens=False))
        for split in ('train', 'val', 'test'):
            for row in load_split(root / 'data/GSM8k-Aug-NL', split):
                question = length(question_prompt(row.question))
                teacher = question + sum(length(step) + length('\n') for step in row.steps)
                _, anchors, answer = output_parts(row, config)
                self.assertLessEqual(question, config.max_question_tokens)
                self.assertLessEqual(teacher, config.max_teacher_tokens)
                self.assertLessEqual(length(anchors) + length(answer) + 1, config.max_target_tokens)
                self.assertIsNotNone(normalize_number(row.answer))

    def test_graph_rows_rejected(self):
        row = {"id": 1, "question": "q", "cot": "s", "answer": "1"}
        for key in ("dependency_matrix", "confidence_matrix", "anchor_indices", "unused"):
            with self.assertRaises(ValueError):
                parse_row({**row, key: []}, "train")
        with self.assertRaises(TypeError):
            Config(**{"lambda_dep": 0.0})

    def test_jsonl_unicode_physical_lines_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "train.jsonl"
            row = {"id": 1, "question": "q\u2028text", "cot": "step", "answer": "1"}
            p.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(len(load_jsonl(p, "train")), 1)
            with self.assertRaises(ValueError):
                load_jsonl(p, "train", "wrong")
            p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_jsonl(p, "train")

    def test_annotated_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "readcot_qsa_qwen_dc"
            folder.mkdir()
            p = folder / "train.jsonl"
            p.write_text('{}\n')
            with self.assertRaises(ValueError):
                load_jsonl(p, "train")

    def test_graph_free_anchors_and_question_boundary(self):
        self.assertEqual(progress_anchors(("a", "b", "c", "d", "e", "f")), ("c", "e"))
        self.assertEqual(progress_anchors(("compute <<2+2=4>> items",)), ("2+2=4",))
        a = examples()[0]
        self.assertEqual(question_prompt(a.question), question_prompt(replace(a, steps=("WRONG",), answer="999").question))
        self.assertNotIn(a.steps[0], question_prompt(a.question))

    def test_answer_parser_not_anchor_reward(self):
        self.assertTrue(answer_matches("x\nAnswer: 1,000", "1000.0"))
        self.assertTrue(answer_matches("Answer: -0.25.", "-.25"))
        for text in ("Anchors: 1000", "Answer: 1001", "Answer: NaN", "Answer: 1000 or 2000"):
            self.assertFalse(answer_matches(text, "1000"))


class TargetPolicyTests(unittest.TestCase):
    def test_partition_closure_and_short_cot(self):
        for length in (1, 2, 6, 11):
            b = torch.randn(length + 1, 8)
            target = build_targets(b, 6)
            torch.testing.assert_close(target.solve.sum(0), b[-1] - b[0])
            self.assertEqual(int(target.span_mask.sum()), min(length, 6))
            torch.testing.assert_close(target.cumulative, target.solve.cumsum(0))

    def test_zero_targets_and_empty_rejection(self):
        target = build_targets(torch.zeros(3, 8), 2)
        states = torch.zeros(4, 8, requires_grad=True)
        plan = torch.zeros(2, 8, requires_grad=True)
        values = role_quantities(states, plan, target)
        self.assertTrue(all(torch.isfinite(x).all() for x in values))
        self.assertEqual(sum(float(x.sum()) for x in values), 0.0)
        with self.assertRaises(ValueError):
            build_targets(torch.zeros(1, 8), 2)

    def test_one_target_shared_by_losses_and_scores(self):
        boundaries = torch.randn(5, 8)
        target = build_targets(boundaries, 2)
        plan_state = boundaries[0]
        states = torch.stack([plan_state, plan_state + target.cumulative[0],
                              plan_state + target.cumulative[1], target.end])
        quantities = role_quantities(states, target.solve.clone(), target)
        self.assertLess(sum(float(x.abs().sum()) for x in quantities), 1e-5)
        scores, valid = process_scores(states, target.solve.clone(), target)
        torch.testing.assert_close(scores, torch.ones_like(scores))
        self.assertTrue(valid.all())

    def test_group_isolation_and_small_process_differences(self):
        values = torch.tensor([0., 1., 100., 100.])
        advantage = group_advantage(values, 2)
        torch.testing.assert_close(advantage, torch.tensor([-1., 1., 0., 0.]))
        tiny = group_advantage(torch.tensor([0.5, 0.500001]), 2, std_floor=0.1, bound=2.)
        self.assertLess(float(tiny.abs().max()), 1e-4)
        with self.assertRaises(ValueError):
            group_advantage(torch.ones(3), 2)

    def test_readout_process_credit_and_bound(self):
        scores = torch.tensor([[0., 0., 0., 1.]])
        valid = torch.ones_like(scores, dtype=torch.bool)
        result = process_returns(scores, valid, 0.9)
        self.assertGreater(float(result[0, -1]), 0)
        self.assertAlmostEqual(float(result[0, -1]), 0.9 / 1.9, places=6)
        torch.testing.assert_close(process_returns(torch.ones_like(scores), valid, 1.), torch.ones(1, 3))
        self.assertEqual(tuple(result.shape), (1, 3))

    def test_fixed_scale_padding_and_length(self):
        current = torch.zeros(2, requires_grad=True)
        loss = surrogate(current, current.detach(), torch.tensor(1.), torch.ones(2), .12, 8)
        padded = surrogate(torch.zeros(5), torch.zeros(5), torch.tensor(1.), torch.tensor([1, 1, 0, 0, 0]), .12, 8)
        self.assertEqual(float(loss), float(padded))
        longer = surrogate(torch.zeros(4), torch.zeros(4), torch.tensor(1.), torch.ones(4), .12, 8)
        self.assertEqual(float(longer), float(loss) * 2)

    def test_config_and_annealing(self):
        config = Config()
        self.assertEqual(config.role_names, ("PLAN", "SOLVE1", "SOLVE2", "SOLVE3", "SOLVE4", "SOLVE5", "SOLVE6", "READOUT"))
        self.assertEqual(config.process_coefficient(100, 100), 0)
        self.assertEqual(config.process_coefficient(99, 100), 0)
        self.assertEqual(config.process_coefficient(0, 100), config.process_weight)
        for kwargs in ({"solve_roles": 0}, {"group_size": 1}, {"semantic_dim": 3.5}, {"sft_lr": float('nan')}):
            with self.assertRaises(ValueError):
                Config(**kwargs)


class ModelTests(unittest.TestCase):
    def test_raw_sampling_replay_and_readout_gradient(self):
        model = tiny_model()
        model.train()
        self.assertFalse(model.language_model.training)
        with torch.no_grad():
            original = model.roles("2+2?", stochastic=True)
            completion = model.generate(original, sample=True)
        replay = model.roles("2+2?", forced_actions=original.actions)
        current, mask = model.answer_logprobs(replay, completion.tokens)
        torch.testing.assert_close(replay.actions[:-1], original.actions[:-1], atol=0, rtol=0)
        torch.testing.assert_close(replay.log_probs[:-1], original.log_probs[:-1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(current, completion.log_probs, atol=1e-6, rtol=1e-6)
        (-current[mask].sum()).backward()
        self.assertGreater(float(model.policy.means['readout'].weight.grad.norm()), 0)
        self.assertIsNone(model.policy.means['solve'].weight.grad)

    def test_readout_is_recomputed_after_parameter_change(self):
        model = tiny_model()
        with torch.no_grad():
            old = model.roles("q", stochastic=True)
            model.policy.means['readout'].bias.add_(0.1)
        replay = model.roles("q", forced_actions=old.actions)
        torch.testing.assert_close(replay.actions[:-1], old.actions[:-1], atol=0, rtol=0)
        self.assertFalse(torch.equal(replay.actions[-1], old.actions[-1]))
        self.assertEqual(float(replay.log_probs[-1]), 0)

    def test_padding_eos_same_id(self):
        mask = token_mask(torch.tensor([3, 2, 2, 2]), {2}, 2)
        self.assertEqual(mask.tolist(), [True, True, False, False])
        self.assertEqual(token_mask(torch.tensor([3, 4, 2]), {2}, 2).tolist(), [True, True, False])

    def test_sft_noise_shared_mechanism_fixed_std(self):
        model = tiny_model()
        torch.manual_seed(17)
        first = model.roles("q", stochastic=True, sft_std=.12)
        torch.manual_seed(17)
        second = model.roles("q", stochastic=True, sft_std=.12)
        torch.testing.assert_close(first.actions, second.actions, atol=0, rtol=0)
        torch.testing.assert_close(first.log_stds[:-1].exp(), torch.full_like(first.log_stds[:-1], .12))
        _, stage2_std = model.policy.distribution(first.pre_states[0], 0)
        torch.testing.assert_close(stage2_std.exp(), torch.full_like(stage2_std, .12))
        torch.manual_seed(17)
        before = torch.get_rng_state()
        model.roles("q", stochastic=False)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_fixed_scorer_and_role_reference_survive_rl_update(self):
        model = tiny_model()
        model.initialize_stage2_reference()
        target = model.teacher_target(examples()[0])
        fixed = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(('reference_policy.', 'score_plan_head.'))}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        metrics = rl_backward(model, examples()[0], target, 0, 10, audit_gradients=True)
        self.assertIn('head_probe_grad/answer', metrics)
        self.assertLess(metrics['old_current_logprob_max_error'], 1e-5)
        optimizer.step()
        for key, value in fixed.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        self.assertTrue(model.plan_head.weight.requires_grad)
        with self.assertRaises(ValueError):
            model.initialize_stage2_reference()

    def test_lora_checkpoint_has_no_frozen_base_weights(self):
        model = tiny_model(lora=True)
        state = model.checkpoint_state()
        llm_keys = [k for k in state if k.startswith('language_model.')]
        self.assertTrue(llm_keys)
        self.assertTrue(all('lora_' in k for k in llm_keys))
        replacement = tiny_model(lora=True)
        replacement.load_checkpoint_state(state)
        for k in state:
            self.assertTrue(torch.equal(state[k], replacement.state_dict()[k]))

    def test_nonzero_answer_advantage_reaches_readout_in_rl(self):
        model = tiny_model(tiny_config(replay_weight=0.))
        model.initialize_stage2_reference()
        target = model.teacher_target(examples()[0])
        with patch('trace_structured.training.answer_matches', side_effect=[False, True]):
            metrics = rl_backward(model, examples()[0], target, 0, 10, audit_gradients=True)
        self.assertGreater(metrics['head_probe_grad/answer'], 0.)
        self.assertTrue(torch.isfinite(model.policy.means['readout'].weight.grad).all())


if __name__ == '__main__':
    unittest.main()
