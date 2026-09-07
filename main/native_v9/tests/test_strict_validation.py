import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.run_native_v9_strict_validation import audit_sample_logs, sha256_file


class StrictValidationAuditTests(unittest.TestCase):
    def test_accepts_exact_unique_binary_question_records(self):
        sample_logs = {
            index: {"acc": [1 if index < 500 else 0]} for index in range(747)
        }
        correct, accuracy = audit_sample_logs(sample_logs, 747)
        self.assertEqual(correct, 500)
        self.assertEqual(accuracy, 500 / 747)

    def test_rejects_missing_question(self):
        sample_logs = {index: {"acc": [1]} for index in range(746)}
        with self.assertRaises(RuntimeError):
            audit_sample_logs(sample_logs, 747)

    def test_rejects_repeated_or_nonbinary_accuracy(self):
        sample_logs = {index: {"acc": [1]} for index in range(747)}
        sample_logs[12]["acc"] = [1, 1]
        with self.assertRaises(RuntimeError):
            audit_sample_logs(sample_logs, 747)
        sample_logs[12]["acc"] = [0.5]
        with self.assertRaises(RuntimeError):
            audit_sample_logs(sample_logs, 747)

    def test_streaming_file_hash_is_stable(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "val.json"
            path.write_bytes(b"native-v9-validation-contract\n")
            self.assertEqual(
                sha256_file(path),
                "4442bb4de98d5c3813c3298980fa5ee682ea6a20e7102829d8c4c4df2d9f60c6",
            )


if __name__ == "__main__":
    unittest.main()
