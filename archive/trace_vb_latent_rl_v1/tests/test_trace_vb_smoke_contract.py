import unittest

from tools.trace_vb_stage1_smoke import select_stress_indices


class _WhitespaceTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return str(text).split()


class _Dataset:
    def __init__(self):
        self.data = [
            {"question": "short question", "steps": "a = 1"},
            {"question": "medium length question", "steps": "b = 2 then c = 3"},
            {
                "question": "invalid " * 100,
                "steps": "leaked = 4 " * 100,
            },
        ]

    @staticmethod
    def get_all_indices():
        return [0, 1, 2]


class _Model:
    tokenizer = _WhitespaceTokenizer()
    _sufficiency_cache = {
        "by_idx": {
            0: {"role_valid_mask": [True, True, True, False, False, False, False, False]},
            1: {"role_valid_mask": [True, True, True, False, True, False, False, False]},
            # The longest row intentionally has no non-leaking target.
            2: {"role_valid_mask": [False] * 8},
        }
    }


class TraceVBStage1SmokeSelectionTests(unittest.TestCase):
    def test_stress_selection_excludes_unsupervised_cache_rows(self):
        selected = select_stress_indices(_Model(), _Dataset(), 2)
        self.assertEqual(
            {row["dataset_index"] for row in selected},
            {0, 1},
        )
        self.assertTrue(
            all(row["active_sufficiency_roles"] > 0 for row in selected)
        )


if __name__ == "__main__":
    unittest.main()
