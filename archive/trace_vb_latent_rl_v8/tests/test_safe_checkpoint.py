import pickle
import tempfile
import unittest
from pathlib import Path

import torch
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from omegaconf import DictConfig, OmegaConf

from src.utils.safe_checkpoint import (
    install_safe_checkpoint_globals,
    safe_load_checkpoint,
)


class _UntrustedPayload:
    pass


class SafeCheckpointLifecycleTests(unittest.TestCase):
    def test_helper_keeps_omegaconf_allowlist_for_lightning_resume(self):
        install_safe_checkpoint_globals()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            torch.save(
                {
                    "state_dict": {"weight": torch.ones(1)},
                    "hyper_parameters": {
                        "all_config": OmegaConf.create(
                            {"model": {"target": "data.only"}}
                        )
                    },
                    "optimizer_states": [],
                    "lr_schedulers": [],
                },
                path,
            )

            loaded = safe_load_checkpoint(path)
            self.assertEqual(
                loaded["hyper_parameters"]["all_config"].model.target,
                "data.only",
            )
            self.assertIn(DictConfig, torch.serialization.get_safe_globals())

            # This is the independent loader Lightning invokes for ckpt_path.
            resumed = TorchCheckpointIO().load_checkpoint(
                path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(torch.equal(resumed["state_dict"]["weight"], torch.ones(1)))

    def test_unregistered_project_class_remains_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "untrusted.ckpt"
            torch.save({"payload": _UntrustedPayload()}, path)
            with self.assertRaises(pickle.UnpicklingError) as raised:
                safe_load_checkpoint(path)
            self.assertIn("Unsupported global", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
