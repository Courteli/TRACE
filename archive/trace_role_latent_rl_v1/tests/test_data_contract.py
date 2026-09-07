import json
import stat
import unittest
from pathlib import Path

from src.datasets.gsm8k_aug_nl import (
    FORMAL_FILES,
    FORMAL_GSM8K_DIR,
    GSM8KAugNLProcessedDataset,
    sha256_file,
    split_cot_steps,
)
from tools.data_contract_audit import normalize_question


class RawDataContractTests(unittest.TestCase):
    def test_registered_gsm8k_splits_have_no_question_overlap(self):
        split_questions = {}
        for name in (
            "gsm8k_train_processed.jsonl",
            "gsm8k_val_processed.jsonl",
            "gsm8k_test_processed.jsonl",
        ):
            path = FORMAL_GSM8K_DIR / name
            with path.open(encoding="utf-8") as handle:
                split_questions[name] = {
                    normalize_question(json.loads(line)["question"])
                    for line in handle
                    if line.strip()
                }
        names = list(split_questions)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                self.assertFalse(
                    split_questions[left] & split_questions[right]
                )

    def test_registered_files_are_exact_and_read_only(self):
        for name, (expected_count, expected_hash) in FORMAL_FILES.items():
            path = FORMAL_GSM8K_DIR / name
            with path.open(encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
            self.assertEqual(count, expected_count)
            self.assertEqual(sha256_file(path), expected_hash)
            self.assertFalse(path.stat().st_mode & stat.S_IWUSR)

    def test_original_cot_becomes_ordered_steps_without_new_rationales(self):
        source = FORMAL_GSM8K_DIR / "gsm8k_train_processed.jsonl"
        with source.open(encoding="utf-8") as handle:
            raw = json.loads(next(handle))
        dataset = GSM8KAugNLProcessedDataset([raw])
        row = dataset[0]
        steps = json.loads(row["step_list_json"])
        self.assertEqual(steps, split_cot_steps(raw["cot"]))
        self.assertEqual(row["steps"], "\n".join(steps))
        self.assertNotIn("rationale_set_json", row)
        self.assertNotIn("generated_cot", row)

    def test_decimal_period_does_not_create_a_false_step(self):
        steps = split_cot_steps(
            "The rate is 12/60 = 0.2. Then 0.2 * 50 = 10."
        )
        self.assertEqual(len(steps), 2)
        self.assertIn("0.2", steps[0])

    def test_titles_and_units_do_not_fragment_reasoning_steps(self):
        steps = split_cot_steps(
            "Mr. Lee buys 8 ft. of rope. He cuts 3 ft. off. "
            "So 8-3=5 ft. The answer is 5."
        )
        self.assertEqual(
            steps,
            [
                "Mr. Lee buys 8 ft. of rope.",
                "He cuts 3 ft. off.",
                "So 8-3=5 ft.",
                "The answer is 5.",
            ],
        )

    def test_compound_area_units_remain_one_equation_step(self):
        steps = split_cot_steps(
            "The area is 8 ft. * 20 ft. = 160 sq. ft. "
            "Therefore the answer is 160."
        )
        self.assertEqual(len(steps), 2)
        self.assertIn("160 sq. ft.", steps[0])


if __name__ == "__main__":
    unittest.main()
