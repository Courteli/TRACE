import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from run import (
    validate_execution_mode_args,
    validate_rewind_source_request,
)

from src.datasets.gsm8k_aug_nl import GSM8KAugNLDataModule
from src.models.read import LitREADCoT
from src.models.trace_vb import (
    LitTRACEVB,
    Stage2RecoveryCheckpoint,
    TRACE_VB_ZERO_ACTION_RESET_SCHEMA,
    TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES,
    assert_capability_spine_parity,
    canonical_registered_capability_payload,
    collect_stage1_role_only_parameters,
    ensure_validation_json_logger,
    rewind_capability_spine_tensors,
    reset_v8_zero_action_policy_tensors,
    select_capability_kl_mask,
    validate_required_state_coverage,
    validate_v8_checkpoint_provenance,
    verify_v7_registered_capability_payload,
    write_v8_checkpoint_provenance,
)
from src.modules.trace_policy import GaussianTrajectoryPolicy


METRIC_BASELINE_PATH = "/metric-safe-baseline.json"
METRIC_BASELINE_SHA = "e" * 64
CAPABILITY_VALIDATION_PATH = "/capability-validation.json"
CAPABILITY_VALIDATION_SHA = "f" * 64


def _metric_config() -> dict:
    return {
        "metric_safe_baseline_path": METRIC_BASELINE_PATH,
        "metric_safe_baseline_sha256": METRIC_BASELINE_SHA,
        "metric_safe_baseline_correct_count": 527,
        "metric_safe_baseline_questions": 747,
        "registered_capability_validation_path": (
            CAPABILITY_VALIDATION_PATH
        ),
        "registered_capability_validation_sha256": (
            CAPABILITY_VALIDATION_SHA
        ),
        "registered_capability_validation_correct_count": 540,
        "registered_capability_validation_questions": 747,
    }


class _CharacterTokenizer:
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in str(text)]


class _CompactTargetHarness:
    readcot_config = {"compact_anchor_max_chars": 64}
    trace_config = {
        "compact_target_max_equations": 2,
        "compact_target_max_new_tokens": 512,
    }
    model_kwargs = SimpleNamespace(
        hybrid_generation_config=SimpleNamespace(max_new_tokens=48)
    )
    tokenizer = _CharacterTokenizer()
    anchor_header = "Anchors:"
    thinking_separator = "###"
    answer_template = "Answer:{}"

    _normalize_compact_equation = LitREADCoT._normalize_compact_equation
    _complete_compact_equations = LitREADCoT._complete_compact_equations
    _target_token_count = LitTRACEVB._target_token_count
    _fit_compact_target_to_generation_budget = (
        LitTRACEVB._fit_compact_target_to_generation_budget
    )
    _build_role_compact_targets = LitTRACEVB._build_role_compact_targets


def _tiny_lora_parameters():
    return {
        "llm.block.lora_A.default.weight": torch.nn.Parameter(
            torch.zeros(2, 3), requires_grad=False
        ),
        "llm.block.lora_A.trace_capability.weight": torch.nn.Parameter(
            torch.arange(6.0).reshape(2, 3), requires_grad=False
        ),
        "llm.block.lora_B.default.weight": torch.nn.Parameter(
            torch.zeros(3, 2), requires_grad=False
        ),
        "llm.block.lora_B.trace_capability.weight": torch.nn.Parameter(
            torch.arange(6.0).reshape(3, 2) + 10.0,
            requires_grad=False,
        ),
    }


def _tiny_registered_capability_state(dtype=torch.float64):
    state = {
        "llm.block.lora_A.default.weight": torch.arange(
            6, dtype=dtype
        ).reshape(2, 3),
        "llm.block.lora_B.default.weight": (
            torch.arange(6, dtype=dtype).reshape(3, 2) + 10
        ),
        "state_norm.weight": torch.arange(3, dtype=dtype),
        "state_norm.bias": torch.arange(3, dtype=dtype) + 1,
        "latent_bridge.0.weight": torch.arange(
            6, dtype=dtype
        ).reshape(2, 3),
        "latent_bridge.0.bias": torch.arange(2, dtype=dtype),
        "latent_bridge.2.weight": torch.arange(
            6, dtype=dtype
        ).reshape(3, 2),
        "latent_bridge.2.bias": torch.arange(3, dtype=dtype),
        "step_compressor.latent_queries": torch.arange(
            6, dtype=dtype
        ).reshape(2, 3),
        "anchor_gate_predictor.0.weight": torch.arange(
            6, dtype=dtype
        ).reshape(2, 3),
        "anchor_gate_predictor.0.bias": torch.arange(2, dtype=dtype),
        "anchor_gate_predictor.2.weight": torch.arange(
            4, dtype=dtype
        ).reshape(2, 2),
        "anchor_gate_predictor.2.bias": torch.arange(2, dtype=dtype),
        "trace_view_embeddings.weight": torch.arange(
            6, dtype=dtype
        ).reshape(2, 3),
        "trace_step_view_embeddings.weight": torch.arange(
            12, dtype=dtype
        ).reshape(4, 3),
    }
    return state


class TraceVBV8ContractTest(unittest.TestCase):
    def test_stage2_recovery_uses_persistent_rollout_coordinate(self):
        callback = Stage2RecoveryCheckpoint(every_n_rollout_batches=64)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory) / "checkpoints"
            saves = []

            def save_checkpoint(path, *, weights_only):
                saves.append((Path(path), weights_only))
                Path(path).touch()

            trainer = SimpleNamespace(
                checkpoint_callback=SimpleNamespace(
                    dirpath=str(checkpoint_directory)
                ),
                default_root_dir=directory,
                current_epoch=0,
                global_step=31,
                is_global_zero=True,
                save_checkpoint=save_checkpoint,
            )
            module = SimpleNamespace(
                do_trace_rl=True,
                vb_rollout_batches_seen=torch.tensor(63),
            )

            callback.on_train_batch_end(
                trainer, module, None, None, batch_idx=62
            )
            self.assertEqual(saves, [])

            module.vb_rollout_batches_seen.fill_(64)
            trainer.global_step = 32
            callback.on_train_batch_end(
                trainer, module, None, None, batch_idx=63
            )
            self.assertEqual(len(saves), 1)
            self.assertFalse(saves[0][1])
            self.assertEqual(
                saves[0][0].name,
                "stage2-recovery-rollout000064-globalstep000032.ckpt",
            )

            # Re-entering the same hook coordinate cannot duplicate a save.
            callback.on_train_batch_end(
                trainer, module, None, None, batch_idx=63
            )
            self.assertEqual(len(saves), 1)

            module.vb_rollout_batches_seen.fill_(128)
            trainer.global_step = 80
            callback.on_train_batch_end(
                trainer, module, None, None, batch_idx=127
            )
            self.assertEqual(len(saves), 2)
            self.assertFalse(saves[1][1])
            self.assertEqual(
                saves[1][0].name,
                "stage2-recovery-rollout000128-globalstep000080.ckpt",
            )
            remaining = list(
                checkpoint_directory.glob(callback.filename_glob)
            )
            self.assertEqual(remaining, [saves[1][0]])

            # The Stage-2 callback is inert for Stage 1 even at a boundary.
            module.do_trace_rl = False
            module.vb_rollout_batches_seen.fill_(192)
            callback.on_train_batch_end(
                trainer, module, None, None, batch_idx=191
            )
            self.assertEqual(len(saves), 2)

        config = OmegaConf.load(
            "src/configs/models/trace_vb_policy_qwen3_instruct.yaml"
        )
        policy = config.model.model_kwargs.trace_policy_config
        self.assertEqual(policy.stage2_recovery_checkpoint_interval, 64)

    def test_registered_capability_canonical_mapping_and_registry(self):
        registered = _tiny_registered_capability_state()
        payload = canonical_registered_capability_payload(
            registered,
            expected_lora_tensors=2,
            n_trace_steps=2,
        )
        self.assertEqual(len(payload), 15)
        self.assertIn(
            "llm.block.lora_A.trace_capability.weight", payload
        )
        self.assertTrue(
            torch.equal(
                payload["capability_trace_view"],
                registered["trace_view_embeddings.weight"][0],
            )
        )
        self.assertTrue(
            torch.equal(
                payload["capability_trace_step_views"],
                registered["trace_step_view_embeddings.weight"][:2],
            )
        )
        v7_state = {
            name: value.float().clone() for name, value in payload.items()
        }
        report = verify_v7_registered_capability_payload(
            v7_state,
            registered,
            expected_lora_tensors=2,
            n_trace_steps=2,
        )
        self.assertEqual(report["tensor_count"], 15)
        self.assertEqual(report["canonical_cast_count"], 15)
        self.assertEqual(len(report["canonical_payload_sha256"]), 64)
        v7_state["capability_trace_view"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "differs"):
            verify_v7_registered_capability_payload(
                v7_state,
                registered,
                expected_lora_tensors=2,
                n_trace_steps=2,
            )

        config = OmegaConf.load(
            "src/configs/models/trace_vb_policy_qwen3_instruct.yaml"
        )
        policy = config.model.model_kwargs.trace_policy_config
        self.assertEqual(policy.registered_capability_payload_tensors, 517)
        self.assertEqual(
            policy.registered_capability_payload_sha256,
            "b1e9a973bbf4f2eeaa18b7d49df16c38db50cea87e998008ba18a46cdff9e049",
        )

    def test_validate_only_json_logger_lifecycle_is_idempotent(self):
        harness = SimpleNamespace(
            all_config=SimpleNamespace(
                args=SimpleNamespace(no_log=True)
            )
        )
        self.assertTrue(ensure_validation_json_logger(harness))
        logger_identity = id(harness.json_logger)
        self.assertFalse(ensure_validation_json_logger(harness))
        self.assertEqual(id(harness.json_logger), logger_identity)
        self.assertIn(
            "ensure_validation_json_logger(self)",
            inspect.getsource(LitTRACEVB.on_validation_start),
        )

    def test_retained_warm_start_state_is_complete_and_shape_checked(self):
        expected = {
            "trajectory_policy.weight": torch.zeros(2, 3),
            "trajectory_posterior.bias": torch.zeros(4),
            "unrelated.weight": torch.zeros(1),
        }
        source = {
            "trajectory_policy.weight": torch.ones(2, 3),
            "trajectory_posterior.bias": torch.ones(4),
        }
        names = validate_required_state_coverage(
            expected,
            source,
            required_prefixes=(
                "trajectory_policy.",
                "trajectory_posterior.",
            ),
        )
        self.assertEqual(len(names), 2)
        with self.assertRaisesRegex(RuntimeError, "missing retained"):
            validate_required_state_coverage(
                expected,
                {"trajectory_policy.weight": torch.ones(2, 3)},
                required_prefixes=(
                    "trajectory_policy.",
                    "trajectory_posterior.",
                ),
            )
        wrong_shape = dict(source)
        wrong_shape["trajectory_posterior.bias"] = torch.ones(5)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            validate_required_state_coverage(
                expected,
                wrong_shape,
                required_prefixes=(
                    "trajectory_policy.",
                    "trajectory_posterior.",
                ),
            )
        self.assertIn(
            '"state_norm.",',
            inspect.getsource(LitTRACEVB.validate_v8_warm_start_coverage),
        )

    def test_resume_and_weights_only_rewind_modes_are_disjoint(self):
        resume_args = SimpleNamespace(
            validate_only=False,
            do_test=False,
            test_ckpt_path="",
            resume_ckpt_path="resume.ckpt",
            load_ckpt_path=None,
            rewind_path_to_capability=False,
            registered_capability_ckpt_path=None,
            save_validation_checkpoint=None,
        )
        validate_execution_mode_args(resume_args)
        resume_args.rewind_path_to_capability = True
        with self.assertRaisesRegex(ValueError, "resume"):
            validate_execution_mode_args(resume_args)
        with self.assertRaisesRegex(RuntimeError, "requires explicit"):
            validate_rewind_source_request(
                "trace_vb_v7", rewind_requested=False
            )
        validate_rewind_source_request(
            "trace_vb_v7", rewind_requested=True
        )
        validate_rewind_source_request(
            "trace_vb_v8", rewind_requested=False
        )
        with self.assertRaisesRegex(RuntimeError, "second"):
            validate_rewind_source_request(
                "trace_vb_v8", rewind_requested=True
            )
        run_source = Path("run.py").read_text(encoding="utf-8")
        self.assertLess(
            run_source.index("model.on_load_checkpoint(checkpoint)"),
            run_source.index("model.register_v8_capability_provenance("),
        )
        self.assertLess(
            run_source.index("model.register_v8_capability_provenance("),
            run_source.index("incompatible = model.load_state_dict("),
        )

    def test_full_target_kl_mask_is_not_suffix_only(self):
        full = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
        suffix = torch.tensor([[0, 0, 0, 1]], dtype=torch.long)
        selected = select_capability_kl_mask(
            full, suffix, scope="full_target"
        )
        self.assertTrue(torch.equal(selected, full))
        self.assertFalse(torch.equal(selected, suffix))
        with self.assertRaisesRegex(ValueError, "full_target"):
            select_capability_kl_mask(full, suffix, scope="unknown")

    def test_rewind_has_exact_coverage_and_preserves_role_tensor(self):
        parameters = _tiny_lora_parameters()
        capability_queries = torch.arange(12.0).reshape(3, 4)
        dynamics_prior = torch.zeros_like(capability_queries)
        role_tensor = torch.nn.Parameter(torch.randn(4, 4))
        role_hash_before = hashlib.sha256(
            role_tensor.detach().numpy().tobytes()
        ).hexdigest()
        copied = rewind_capability_spine_tensors(
            parameters,
            capability_queries,
            dynamics_prior,
            path_adapter_name="default",
            capability_adapter_name="trace_capability",
            expected_tensors=2,
        )
        self.assertEqual(copied, 2)
        self.assertEqual(
            assert_capability_spine_parity(
                parameters,
                capability_queries,
                dynamics_prior,
                path_adapter_name="default",
                capability_adapter_name="trace_capability",
                expected_tensors=2,
            ),
            2,
        )
        self.assertEqual(
            hashlib.sha256(
                role_tensor.detach().numpy().tobytes()
            ).hexdigest(),
            role_hash_before,
        )
        with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
            rewind_capability_spine_tensors(
                parameters,
                capability_queries,
                dynamics_prior,
                path_adapter_name="default",
                capability_adapter_name="trace_capability",
                expected_tensors=3,
            )

    def test_zero_action_reset_changes_exactly_nine_policy_tensors(self):
        policy = GaussianTrajectoryPolicy(
            hidden_size=8,
            action_dim=3,
            n_steps=8,
            policy_hidden_size=6,
            step_embedding_size=4,
        )
        with torch.no_grad():
            for index, parameter in enumerate(policy.parameters(), start=1):
                parameter.fill_(float(index) / 10.0)
        before = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
        }
        report = reset_v8_zero_action_policy_tensors(policy)
        after = policy.state_dict()
        changed = {
            f"trajectory_policy.{name}"
            for name, value in after.items()
            if not torch.equal(value, before[name])
        }
        self.assertEqual(changed, set(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES))
        self.assertEqual(report["schema_version"], TRACE_VB_ZERO_ACTION_RESET_SCHEMA)
        self.assertEqual(report["operation_count"], 1)
        self.assertEqual(report["tensor_count"], 9)
        self.assertEqual(
            report["tensor_names"],
            list(TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES),
        )
        for name, value in after.items():
            qualified = f"trajectory_policy.{name}"
            if qualified in TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES:
                self.assertEqual(torch.count_nonzero(value).item(), 0)
            else:
                self.assertTrue(
                    torch.equal(value, before[name]),
                    msg=f"unexpected reset mutation: {qualified}",
                )

    def test_zero_action_reset_rejects_invalid_policy_before_mutation(self):
        policy = GaussianTrajectoryPolicy(
            hidden_size=8,
            action_dim=3,
            n_steps=8,
            policy_hidden_size=6,
            step_embedding_size=4,
        )
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.fill_(0.25)
        policy.mean_heads["commit"] = torch.nn.Linear(7, 3)
        before = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
        }
        with self.assertRaisesRegex(RuntimeError, "feature width"):
            reset_v8_zero_action_policy_tensors(policy)
        for name, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), msg=name)

        policy = GaussianTrajectoryPolicy(
            hidden_size=8,
            action_dim=3,
            n_steps=8,
            policy_hidden_size=6,
            step_embedding_size=4,
        )
        with torch.no_grad():
            policy.mean_heads["solve"].weight[0, 0] = float("nan")
        before = policy.action_projector[0].bias.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            reset_v8_zero_action_policy_tensors(policy)
        self.assertTrue(torch.equal(policy.action_projector[0].bias, before))

    def test_v7_initialization_rewinds_only_once(self):
        registered_sha = "a" * 64
        payload_sha = "b" * 64
        v7_sha = "c" * 64
        parameters = _tiny_lora_parameters()
        llm = SimpleNamespace(
            named_parameters=lambda: list(parameters.items())
        )
        policy = GaussianTrajectoryPolicy(
            hidden_size=4,
            action_dim=2,
            n_steps=8,
            policy_hidden_size=6,
            step_embedding_size=3,
        )
        with torch.no_grad():
            for head in policy.mean_heads.values():
                head.weight.fill_(0.75)
                head.bias.fill_(0.5)
            policy.action_projector[0].bias.fill_(0.25)
        projector_weight_before = (
            policy.action_projector[0].weight.detach().clone()
        )
        harness = SimpleNamespace(
            trace_config={
                "capability_expected_lora_tensors": 2,
                "registered_capability_checkpoint_sha256": registered_sha,
                "registered_capability_payload_tensors": 15,
                "registered_capability_payload_sha256": payload_sha,
                **_metric_config(),
            },
            path_adapter_name="default",
            capability_adapter_name="trace_capability",
            llm=llm,
            capability_latent_queries=torch.arange(32.0).reshape(8, 4),
            trajectory_policy=policy,
            _capability_anchor_loaded=True,
            _capability_spine_rewound=False,
            _capability_spine_rewind_count=0,
            _capability_spine_rewind_source=None,
            _zero_action_reset_schema_version=None,
            _zero_action_reset_applied=False,
            _zero_action_reset_operation_count=0,
            _zero_action_reset_tensor_count=0,
            _zero_action_reset_target_names=(),
            _zero_action_reset_source_schema=None,
            _registered_capability_checkpoint_sha256=registered_sha,
            _registered_capability_payload_tensor_count=15,
            _registered_capability_payload_sha256=payload_sha,
            _v7_source_checkpoint_sha256=v7_sha,
            _set_adapter_parameter_trainability=lambda: None,
            _assert_registered_capability_provenance=lambda: None,
            _assert_capability_spine_contract=lambda: {"ok": True},
        )
        with self.assertRaisesRegex(RuntimeError, "explicit"):
            LitTRACEVB.initialize_v8_capability_spine(
                harness, {"trace_vb_schema_version": "trace_vb_v7"}
            )
        result = LitTRACEVB.initialize_v8_capability_spine(
            harness,
            {"trace_vb_schema_version": "trace_vb_v7"},
            allow_v7_rewind=True,
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(harness._capability_spine_rewind_count, 2)
        self.assertEqual(harness._zero_action_reset_tensor_count, 9)
        for name, value in harness.trajectory_policy.state_dict().items():
            if f"trajectory_policy.{name}" in (
                TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES
            ):
                self.assertEqual(torch.count_nonzero(value).item(), 0)
        self.assertTrue(
            torch.equal(
                harness.trajectory_policy.action_projector[0].weight,
                projector_weight_before,
            )
        )
        self.assertFalse(
            harness.trajectory_policy.dynamics_step_embedding.weight.requires_grad
        )
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            LitTRACEVB.initialize_v8_capability_spine(
                harness,
                {"trace_vb_schema_version": "trace_vb_v7"},
                allow_v7_rewind=True,
            )

    def test_stage1_optimizer_partition_is_role_only(self):
        role = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        frozen_path = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        frozen_query = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        names, parameters = collect_stage1_role_only_parameters(
            [
                ("trajectory_policy.policy_trunk.1.weight", role),
                ("llm.layer.lora_A.default.weight", frozen_path),
                (
                    "trajectory_policy.dynamics_step_embedding.weight",
                    frozen_query,
                ),
            ],
            path_adapter_name="default",
        )
        self.assertEqual(
            names, ["trajectory_policy.policy_trunk.1.weight"]
        )
        self.assertEqual(parameters, [role])
        frozen_path.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "deployment LoRA"):
            collect_stage1_role_only_parameters(
                [("llm.layer.lora_A.default.weight", frozen_path)],
                path_adapter_name="default",
            )
        frozen_path.requires_grad_(False)
        frozen_query.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "query prior"):
            collect_stage1_role_only_parameters(
                [
                    (
                        "trajectory_policy.dynamics_step_embedding.weight",
                        frozen_query,
                    )
                ],
                path_adapter_name="default",
            )

    def test_compact_target_keeps_last_two_complete_equations(self):
        harness = _CompactTargetHarness()
        target = harness._build_role_compact_targets(
            [
                {
                    "steps": [
                        "First 1 + 1 = 2.",
                        "Then 2 + 2 = 4.",
                        "Finally 3 + 3 = 6.",
                    ]
                }
            ],
            ["6"],
            [[(0, 1), (1, 2), (2, 3)]],
        )[0]
        self.assertEqual(target.count("\n- ") + target.startswith("- "), 2)
        self.assertNotIn("1+1=2", target)
        self.assertLess(target.index("2+2=4"), target.index("3+3=6"))
        self.assertTrue(target.endswith("###Answer:6"))

    def test_overlong_or_missing_equation_goes_directly_to_answer(self):
        harness = _CompactTargetHarness()
        long_equation = "+".join(["1234567890"] * 8) + "=8"
        self.assertGreater(len(long_equation), 64)
        self.assertEqual(
            harness._complete_compact_equations(long_equation), []
        )
        target = harness._build_role_compact_targets(
            [{"steps": ["There is no explicit equation here."]}],
            ["7"],
            [[(0, 1)]],
        )[0]
        self.assertEqual(target, "###Answer:7")
        self.assertNotIn("no compact equation", target)

    def test_checkpoint_metadata_supports_resume_and_weights_only(self):
        expected_sha = "a" * 64
        registered_sha = "b" * 64
        payload_sha = "c" * 64
        v7_source_sha = "d" * 64
        checkpoint = {}
        metric_config = _metric_config()
        write_v8_checkpoint_provenance(
            checkpoint,
            rewound=True,
            rewind_count=504,
            rewind_source="trace_vb_v7_capability",
            cot_encoder_sha256=expected_sha,
            registered_capability_checkpoint_sha256=registered_sha,
            registered_capability_payload_tensors=517,
            registered_capability_payload_sha256=payload_sha,
            v7_source_checkpoint_sha256=v7_source_sha,
            zero_action_reset_schema_version=(
                TRACE_VB_ZERO_ACTION_RESET_SCHEMA
            ),
            zero_action_reset_applied=True,
            zero_action_reset_operation_count=1,
            zero_action_reset_tensor_count=9,
            zero_action_reset_target_names=(
                TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES
            ),
            zero_action_reset_source_schema="trace_vb_v7",
            metric_safe_baseline_path=metric_config[
                "metric_safe_baseline_path"
            ],
            metric_safe_baseline_sha256=metric_config[
                "metric_safe_baseline_sha256"
            ],
            metric_safe_baseline_correct_count=527,
            metric_safe_baseline_questions=747,
            registered_capability_validation_path=metric_config[
                "registered_capability_validation_path"
            ],
            registered_capability_validation_sha256=metric_config[
                "registered_capability_validation_sha256"
            ],
            registered_capability_validation_correct_count=540,
            registered_capability_validation_questions=747,
        )
        # A trained v8 checkpoint may legitimately have nonzero heads.  Load
        # validates the one-time reset attestation, never the current values.
        checkpoint["state_dict"] = {
            name: torch.ones(1)
            for name in TRACE_VB_ZERO_ACTION_RESET_TENSOR_NAMES
        }
        self.assertEqual(checkpoint["trace_vb_schema_version"], "trace_vb_v8")
        self.assertTrue(checkpoint["trace_vb_capability_spine_rewound"])
        self.assertEqual(
            checkpoint["trace_vb_capability_spine_rewind_count"], 504
        )
        self.assertEqual(
            checkpoint["trace_vb_cot_encoder_checkpoint_sha256"],
            expected_sha,
        )
        self.assertEqual(
            checkpoint[
                "trace_vb_registered_capability_checkpoint_sha256"
            ],
            registered_sha,
        )
        self.assertEqual(
            checkpoint["trace_vb_registered_capability_payload_sha256"],
            payload_sha,
        )
        self.assertEqual(
            checkpoint["trace_vb_registered_capability_payload_tensors"],
            517,
        )
        self.assertEqual(
            checkpoint["trace_vb_v7_source_checkpoint_sha256"],
            v7_source_sha,
        )
        validation_kwargs = {
            "expected_lora_tensors": 504,
            "expected_cot_sha256": expected_sha,
            "expected_registered_capability_checkpoint_sha256": (
                registered_sha
            ),
            "expected_registered_capability_payload_tensors": 517,
            "expected_registered_capability_payload_sha256": payload_sha,
            "expected_metric_safe_baseline_path": metric_config[
                "metric_safe_baseline_path"
            ],
            "expected_metric_safe_baseline_sha256": metric_config[
                "metric_safe_baseline_sha256"
            ],
            "expected_metric_safe_baseline_correct_count": 527,
            "expected_metric_safe_baseline_questions": 747,
            "expected_registered_capability_validation_path": metric_config[
                "registered_capability_validation_path"
            ],
            "expected_registered_capability_validation_sha256": metric_config[
                "registered_capability_validation_sha256"
            ],
            "expected_registered_capability_validation_correct_count": 540,
            "expected_registered_capability_validation_questions": 747,
        }
        metadata = validate_v8_checkpoint_provenance(
            checkpoint,
            **validation_kwargs,
        )
        self.assertEqual(metadata["rewind_count"], 504)
        self.assertEqual(metadata["cot_encoder_sha256"], expected_sha)
        self.assertEqual(
            metadata["registered_capability_payload_sha256"], payload_sha
        )
        self.assertEqual(
            metadata["v7_source_checkpoint_sha256"], v7_source_sha
        )
        weights_only_harness = SimpleNamespace(
            trace_config={
                "capability_expected_lora_tensors": 504,
                "stage0_cot_encoder_checkpoint_sha256": expected_sha,
                "registered_capability_checkpoint_sha256": registered_sha,
                "registered_capability_payload_tensors": 517,
                "registered_capability_payload_sha256": payload_sha,
                **metric_config,
            },
            _capability_spine_rewound=False,
            _capability_spine_rewind_count=0,
            _capability_spine_rewind_source=None,
            _cot_encoder_checkpoint_sha256=None,
            _registered_capability_checkpoint_sha256=None,
            _registered_capability_payload_tensor_count=0,
            _registered_capability_payload_sha256=None,
            _v7_source_checkpoint_sha256=None,
            _assert_capability_spine_contract=lambda: {"ok": True},
        )
        with self.assertRaisesRegex(RuntimeError, "must never request"):
            LitTRACEVB.initialize_v8_capability_spine(
                weights_only_harness,
                checkpoint,
                allow_v7_rewind=True,
            )
        self.assertEqual(
            LitTRACEVB.initialize_v8_capability_spine(
                weights_only_harness, checkpoint
            ),
            {"ok": True},
        )
        self.assertEqual(
            weights_only_harness._cot_encoder_checkpoint_sha256,
            expected_sha,
        )
        broken = dict(checkpoint)
        broken.pop("trace_vb_cot_encoder_checkpoint_sha256")
        with self.assertRaisesRegex(RuntimeError, "CoT encoder provenance"):
            validate_v8_checkpoint_provenance(
                broken,
                **validation_kwargs,
            )
        broken = dict(checkpoint)
        broken["trace_vb_zero_action_reset_operation_count"] = True
        with self.assertRaisesRegex(RuntimeError, "operation count"):
            validate_v8_checkpoint_provenance(broken, **validation_kwargs)

    def test_validate_setup_does_not_open_train_or_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            val_path = directory_path / "val.jsonl"
            val_path.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "question": "q",
                        "cot": "1 + 1 = 2.",
                        "answer": "2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            module = GSM8KAugNLDataModule(
                dataset_name="unit",
                dataset_dir=directory,
                train_file="missing_train.jsonl",
                val_file="val.jsonl",
                test_file="missing_test.jsonl",
            )
            module.setup("validate")
            self.assertEqual(len(module.val_set), 1)
            self.assertIsNone(module.train_set)
            self.assertIsNone(module.test_set)


if __name__ == "__main__":
    unittest.main()
