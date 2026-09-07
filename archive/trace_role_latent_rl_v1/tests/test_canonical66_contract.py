"""Formal contracts for the isolated role-semantic TRACE experiment.

The filename is retained so existing CI discovery keeps exercising the
pipeline, but the old canonical66 architecture/checkpoint lock is deliberately
gone.  This version must train its own Stage 1 from an explicit Stage-0 CoT
checkpoint and may never fall back to the previous experiment's sources.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("TRACE_DATA_ROOT", "/disk1/dingxukai/TRACE")
)


class RoleSemanticPipelineContractTests(unittest.TestCase):
    def test_static_pipeline_and_registered_data_audit_pass(self):
        environment = os.environ.copy()
        environment["TRACE_PROJECT_ROOT"] = str(ROOT)
        environment["TRACE_DATA_ROOT"] = str(DATA_ROOT)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/role_pipeline_contract_audit.py"),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["passed"], report["total"])

    def test_launcher_trains_new_stage1_then_stage2(self):
        full = (ROOT / "scripts/run_full_pipeline.sh").read_text()
        stage1 = (ROOT / "scripts/run_stage1_formation.sh").read_text()
        stage2 = (ROOT / "scripts/run_stage2_refinement.sh").read_text()
        combined = "\n".join((full, stage1, stage2))

        self.assertIn("<stage0-checkpoint>", full)
        self.assertIn("run_stage1_formation.sh", full)
        self.assertIn("run_stage2_refinement.sh", full)
        self.assertIn("<stage1-best-checkpoint>", stage2)
        self.assertIn("stage1_checkpoint=$2", stage2)
        self.assertNotIn("verify_canonical66.py", combined)
        self.assertNotIn("TRACE-canonical66-stage1", combined)
        self.assertNotIn("20260721-200252_230269", combined)

    def test_fixed_role_program_and_commit_only_answer_contract(self):
        config = (
            ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml"
        ).read_text()
        model = (ROOT / "src/models/trace_policy.py").read_text()
        module = (ROOT / "src/modules/trace_policy.py").read_text()

        self.assertIn("n_latents: 8", config)
        self.assertIn("TRACE_ROLE_NAMES", module)
        for role in ("PLAN", "SOLVE", "CHECK", "COMMIT"):
            self.assertIn(f'"{role}"', module)
        self.assertIn("_commit_only_latent_mask", model)
        self.assertIn("build_contiguous_cot_targets", model)
        self.assertIn("stage1_plan_weight", config)
        self.assertIn("stage1_solve_weight", config)
        self.assertIn("stage1_check_weight", config)

    def test_stage2_is_latent_only_with_step_level_credit(self):
        config = (
            ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml"
        ).read_text()
        model = (ROOT / "src/models/trace_policy.py").read_text()

        self.assertIn("use_trajectory_policy_loss: True", config)
        self.assertIn("use_answer_policy_loss: False", config)
        self.assertIn("step_reward_weight: 0.30", config)
        self.assertIn("step_reward_discount: 0.90", config)
        self.assertIn("role_entropy_weights:", config)
        self.assertIn("_role_step_rewards", model)
        self.assertIn("build_discounted_role_advantages", model)
        self.assertNotIn(
            "self._answer_policy_update(rollout)",
            model[model.index("def trace_rl_training_step"):model.index(
                "def _legacy_answer_joint_training_step"
            )],
        )

    def test_full_budget_and_fail_closed_evidence_gate(self):
        stage1 = (ROOT / "scripts/run_stage1_formation.sh").read_text()
        stage2 = (ROOT / "scripts/run_stage2_refinement.sh").read_text()
        evidence = (ROOT / "scripts/run_evidence.sh").read_text()

        self.assertIn("scheduled_optimizer_steps=16820", stage1)
        self.assertIn("trainer.limit_train_batches=1.0", stage1)
        self.assertIn("trainer.limit_val_batches=1.0", stage1)
        self.assertIn("unique_training_questions_per_epoch=2048", stage2)
        self.assertIn("trainer.limit_train_batches=512", stage2)
        self.assertIn("trainer.limit_val_batches=1.0", stage2)
        self.assertIn("scheduler.num_training_steps=5120", stage2)
        self.assertIn("--test_times 1", evidence)
        self.assertIn("--seed 0", evidence)
        self.assertIn("tools/verify_evidence_complete.py", evidence)
        self.assertIn("--write-complete", evidence)


if __name__ == "__main__":
    unittest.main()
