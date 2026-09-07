import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from run import (
    NATIVE_MODEL_TARGET,
    NATIVE_STAGE0_TARGET,
    _validated_native_initialization,
    _validated_native_resume,
)


def checkpoint(target, do_trace_rl=False):
    all_config = OmegaConf.create(
        {
            "model": {
                "target": target,
                "model_kwargs": {"do_trace_rl": do_trace_rl},
            }
        }
    )
    return {
        "state_dict": {},
        "optimizer_states": [{"state": {}, "param_groups": []}],
        "hyper_parameters": {"all_config": all_config},
    }


def target_config(do_trace_rl):
    return OmegaConf.create(
        {
            "model": {
                "target": NATIVE_MODEL_TARGET,
                "model_kwargs": {"do_trace_rl": do_trace_rl},
            }
        }
    )


class LineageGuardTests(unittest.TestCase):
    def test_accepts_only_direct_fresh_predecessors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage0 = root / "stage0"
            stage1 = root / "stage1"
            stage0.mkdir()
            stage1.mkdir()
            stage0_path = stage0 / "formation.ckpt"
            stage1_path = stage1 / "roles.ckpt"
            torch.save(checkpoint(NATIVE_STAGE0_TARGET), stage0_path)
            torch.save(checkpoint(NATIVE_MODEL_TARGET), stage1_path)
            with patch.dict(os.environ, {"TRACE_NATIVE_RUN_ROOT": str(root)}):
                loaded0 = _validated_native_initialization(
                    stage0_path, target_config(False)
                )
                loaded1 = _validated_native_initialization(
                    stage1_path, target_config(True)
                )
            self.assertEqual(loaded0["state_dict"], {})
            self.assertEqual(loaded1["state_dict"], {})

    def test_rejects_historical_or_rl_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stage0").mkdir()
            (root / "stage1").mkdir()
            outside = root.parent / f"{root.name}_historical.ckpt"
            wrong_stage = root / "stage0" / "old_native.ckpt"
            rl_stage1 = root / "stage1" / "old_rl.ckpt"
            torch.save(checkpoint(NATIVE_STAGE0_TARGET), outside)
            torch.save(checkpoint(NATIVE_MODEL_TARGET), wrong_stage)
            torch.save(checkpoint(NATIVE_MODEL_TARGET, True), rl_stage1)
            try:
                with patch.dict(os.environ, {"TRACE_NATIVE_RUN_ROOT": str(root)}):
                    with self.assertRaises(RuntimeError):
                        _validated_native_initialization(outside, target_config(False))
                    with self.assertRaises(RuntimeError):
                        _validated_native_initialization(wrong_stage, target_config(True))
                    with self.assertRaises(RuntimeError):
                        _validated_native_initialization(rl_stage1, target_config(True))
            finally:
                outside.unlink(missing_ok=True)

    def test_resume_is_confined_to_the_same_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for stage in ("stage0", "stage1", "stage2"):
                (root / stage).mkdir()
            stage1_path = root / "stage1" / "last.ckpt"
            stage2_path = root / "stage2" / "last.ckpt"
            torch.save(checkpoint(NATIVE_MODEL_TARGET, False), stage1_path)
            torch.save(checkpoint(NATIVE_MODEL_TARGET, True), stage2_path)
            with patch.dict(os.environ, {"TRACE_NATIVE_RUN_ROOT": str(root)}):
                loaded = _validated_native_resume(stage1_path, target_config(False))
                self.assertTrue(loaded["optimizer_states"])
                with self.assertRaises(RuntimeError):
                    _validated_native_resume(stage2_path, target_config(False))

    def test_resume_rejects_weights_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stage0").mkdir()
            path = root / "stage0" / "last.ckpt"
            value = checkpoint(NATIVE_STAGE0_TARGET, False)
            value["optimizer_states"] = []
            torch.save(value, path)
            stage0_config = OmegaConf.create(
                {
                    "model": {
                        "target": NATIVE_STAGE0_TARGET,
                        "model_kwargs": {"do_trace_rl": False},
                    }
                }
            )
            with patch.dict(os.environ, {"TRACE_NATIVE_RUN_ROOT": str(root)}):
                with self.assertRaises(RuntimeError):
                    _validated_native_resume(path, stage0_config)


if __name__ == "__main__":
    unittest.main()
