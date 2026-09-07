import unittest
from pathlib import Path


class Stage1ScriptContractTests(unittest.TestCase):
    def test_recycled_epoch_checkpoint_globs_match_logger_versions(self):
        script = (
            Path(__file__).parents[1]
            / "scripts"
            / "run_stage1_formation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '*/${RUN_TAG}_epoch_process_*/checkpoints/last.ckpt',
            script,
        )
        self.assertIn(
            'f"{run_tag}_epoch_process_*/checkpoints/"',
            script,
        )
        self.assertNotIn(
            '-path "*_${RUN_TAG}/checkpoints/last.ckpt"',
            script,
        )


if __name__ == "__main__":
    unittest.main()
