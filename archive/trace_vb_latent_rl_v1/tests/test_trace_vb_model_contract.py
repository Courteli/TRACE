import hashlib
import unittest

import torch
import lightning.pytorch as pl

from src.models.trace_vb import LitTRACEVB


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


if __name__ == "__main__":
    unittest.main()
