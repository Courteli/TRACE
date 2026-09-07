import json
import unittest

import torch

from src.datasets.trace_rationale_set import (
    TraceRationaleSetDataset,
    normalize_rationale_set,
)
from src.models.trace_final import (
    build_path_bottleneck_attention_mask,
    path_noncollapse_loss,
    permutation_invariant_path_set_loss,
    replace_cached_latent_suffix,
)
from src.models.trace_bridge import LitTRACEBridge
from tools.trace_build_rationale_sets import (
    _merge_generated_split,
    answers_match,
    evaluate_expression,
    rationale_fingerprint,
    verify_generated_rationale,
)
from tools.trace_final_outcome_probe import (
    binary_metrics,
    fit_linear_probe,
    predict_probe,
)


class PathBottleneckTests(unittest.TestCase):
    def test_question_access_is_exactly_zero(self):
        question = torch.tensor([[0, 1, 1], [1, 1, 1]])
        latent = torch.ones(2, 8, dtype=torch.long)
        tail = torch.ones(2, 4, dtype=torch.long)
        mask = build_path_bottleneck_attention_mask(
            question,
            latent,
            tail,
        )
        self.assertEqual(tuple(mask.shape), (2, 15))
        self.assertTrue(torch.equal(mask[:, :3], torch.zeros_like(question)))
        self.assertTrue(torch.equal(mask[:, 3:11], latent))
        self.assertTrue(torch.equal(mask[:, 11:], tail))

    def test_transition_replacement_preserves_slot_norm(self):
        class Dummy:
            training = False
            trace_config = {
                "trace_eval_intervention": "replace_transition_3",
            }

        latents = torch.randn(2, 8, 16)
        controlled = LitTRACEBridge._apply_trace_eval_intervention(
            Dummy(),
            latents,
            ["q0", "q1"],
        )
        self.assertTrue(torch.equal(controlled[:, :3], latents[:, :3]))
        self.assertTrue(torch.equal(controlled[:, 4:], latents[:, 4:]))
        self.assertTrue(
            torch.allclose(
                controlled[:, 3].norm(dim=-1),
                latents[:, 3].norm(dim=-1),
                atol=1e-5,
            )
        )

    def test_same_norm_random_path_preserves_each_slot_norm(self):
        class Dummy:
            training = False
            trace_config = {
                "trace_eval_intervention": "same_norm_random_path",
                "trace_eval_intervention_seed": 7,
            }

            @staticmethod
            def _stable_question_seed(question, seed):
                return seed + len(question)

        latents = torch.randn(2, 8, 16)
        controlled = LitTRACEBridge._apply_trace_eval_intervention(
            Dummy(),
            latents,
            ["q0", "q1"],
        )
        self.assertTrue(
            torch.allclose(
                controlled.norm(dim=-1),
                latents.norm(dim=-1),
                atol=1e-5,
            )
        )
        self.assertFalse(torch.allclose(controlled, latents))

    def test_cached_path_swap_changes_only_latent_suffix(self):
        class Layer:
            def __init__(self, offset):
                values = torch.arange(
                    2 * 1 * 11 * 3,
                    dtype=torch.float32,
                ).reshape(2, 1, 11, 3)
                self.keys = values + offset
                self.values = values + offset + 1000

        class Cache:
            def __init__(self, offset):
                self.layers = [Layer(offset), Layer(offset + 100)]

        target = Cache(0)
        donor = Cache(10000)
        question_keys = [
            layer.keys[..., :3, :].clone() for layer in target.layers
        ]
        donor_indices = torch.tensor([1, 0])
        replace_cached_latent_suffix(
            target,
            donor,
            n_latents=8,
            donor_indices=donor_indices,
        )
        for layer_index, layer in enumerate(target.layers):
            self.assertTrue(
                torch.equal(layer.keys[..., :3, :], question_keys[layer_index])
            )
            expected = donor.layers[layer_index].keys.index_select(
                0,
                donor_indices,
            )[..., -8:, :]
            self.assertTrue(torch.equal(layer.keys[..., -8:, :], expected))


class SetFormationTests(unittest.TestCase):
    def test_matching_is_teacher_and_model_permutation_invariant(self):
        torch.manual_seed(4)
        model = torch.randn(3, 2, 8, 12)
        teacher = torch.randn(3, 2, 8, 12)
        original = permutation_invariant_path_set_loss(model, teacher)["loss"]
        teacher_swapped = permutation_invariant_path_set_loss(
            model,
            teacher.flip(dims=[1]),
        )["loss"]
        model_swapped = permutation_invariant_path_set_loss(
            model.flip(dims=[1]),
            teacher,
        )["loss"]
        self.assertTrue(torch.allclose(original, teacher_swapped, atol=1e-6))
        self.assertTrue(torch.allclose(original, model_swapped, atol=1e-6))

    def test_noncollapse_is_unary(self):
        zeros = torch.zeros(2, 2, 8, 16)
        moving = torch.ones_like(zeros)
        self.assertGreater(
            float(path_noncollapse_loss(zeros, margin=0.02)),
            0.0,
        )
        self.assertEqual(
            float(path_noncollapse_loss(moving, margin=0.02)),
            0.0,
        )

    def test_linear_outcome_probe_recovers_separable_paths(self):
        features = torch.tensor(
            [
                [[-2.0, 0.0], [-1.0, 0.0]],
                [[1.0, 0.0], [2.0, 0.0]],
            ]
        ).numpy()
        labels = torch.tensor([[0, 0], [1, 1]]).numpy()
        probe = fit_linear_probe(features, labels, ridge=0.01)
        scores = predict_probe(probe, features).reshape(labels.shape)
        metrics = binary_metrics(labels, scores)
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["auprc"], 1.0)


class RationaleVerificationTests(unittest.TestCase):
    def test_safe_arithmetic_and_answer_equivalence(self):
        self.assertEqual(evaluate_expression("48 / 2"), 24)
        self.assertTrue(answers_match("$10.0", "10"))
        self.assertFalse(answers_match("11", "10"))
        self.assertIsNone(evaluate_expression("__import__('os').system('x')"))

    def test_generated_rationale_requires_distinct_valid_equations(self):
        gold_steps = [
            "Natalia sold 48/2 = 24 clips in May.",
            "She sold 48+24 = 72 clips in total.",
        ]
        gold_fingerprint = rationale_fingerprint(gold_steps)
        accepted, reason = verify_generated_rationale(
            "1. Half of 48 is 48 / 2 = 24.\n"
            "2. Combining both months gives 48 + 24 = 72.\n"
            "Answer: 72",
            "72",
            {gold_fingerprint},
        )
        self.assertIsNone(accepted)
        self.assertEqual(reason, "surface_duplicate")

        accepted, reason = verify_generated_rationale(
            "1. April plus half of April can be grouped as "
            "48 * (1 + 0.5) = 72.\nAnswer: 72",
            "72",
            {gold_fingerprint},
        )
        self.assertEqual(reason, "accepted")
        self.assertTrue(accepted["verified"])

        rejected, reason = verify_generated_rationale(
            "1. Compute 48 * 1.5 = 70.\nAnswer: 72",
            "72",
            set(),
        )
        self.assertIsNone(rejected)
        self.assertEqual(reason, "invalid_intermediate_equation")

    def test_generated_split_merge_keeps_verified_alternatives(self):
        source = [
            {
                "source_id": 7,
                "question": "What is 2+3?",
                "answer": "5",
                "steps": ["2+3=5."],
            }
        ]
        alternative = {
            "steps": ["3+2=5."],
            "fingerprint": "alternative",
            "source": "verified",
            "verified": True,
        }
        merged, histogram, _ = _merge_generated_split(
            source_rows=source,
            generated={
                7: {
                    "source_id": 7,
                    "accepted": [alternative],
                    "rejection_reasons": {},
                }
            },
            max_alternatives=3,
        )
        self.assertEqual(histogram[2], 1)
        self.assertEqual(merged[0]["rationale_set"], [alternative])


class RationaleDatasetTests(unittest.TestCase):
    @staticmethod
    def _item():
        return {
            "source_id": 3,
            "question": "What is 2+3?",
            "answer": "5",
            "steps": ["Add the values: 2+3=5."],
            "dependency_matrix": [[0]],
            "confidence_matrix": [[0.0]],
            "rationale_fingerprint": "gold-fp",
            "rationale_set": [
                {
                    "steps": ["Use a number line: 2+3=5."],
                    "fingerprint": "alt-fp",
                    "source": "generated",
                    "verified": True,
                },
                {
                    "steps": ["Unverified text"],
                    "fingerprint": "bad",
                    "verified": False,
                },
            ],
        }

    def test_gold_is_first_and_unverified_routes_are_removed(self):
        rationales = normalize_rationale_set(self._item())
        self.assertEqual(len(rationales), 2)
        self.assertEqual(rationales[0]["source"], "gold")
        dataset = TraceRationaleSetDataset([self._item()])
        row = dataset[0]
        self.assertEqual(row["n_rationales"], 2)
        decoded = json.loads(row["rationale_set_json"])
        self.assertEqual([item["fingerprint"] for item in decoded], ["gold-fp", "alt-fp"])


if __name__ == "__main__":
    unittest.main()
