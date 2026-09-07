#!/usr/bin/env python3
"""Real single-GPU Stage-1 mechanism and device-wide memory gate for TRACE-VB."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate


ROOT = Path(os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))

from src.modules.trace_policy import TRACE_COMMIT_INDEX  # noqa: E402
from src.utils.utils import instantiate_from_config  # noqa: E402


MIB = 1024**2
MIN_HEADROOM_MIB = 4096.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config():
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "src/configs/trainer/default.yaml"),
        OmegaConf.load(ROOT / "src/configs/models/trace_vb_policy_qwen3_instruct.yaml"),
        OmegaConf.load(ROOT / "src/configs/datasets/gsm8k_aug_nl.yaml"),
    )
    config.dataloader.batch_size = 1
    config.dataloader.val_batch_size = 1
    config.dataloader.num_workers = 0
    config.dataloader.pin_memory = False
    config.dataloader.persistent_workers = False
    config.args = OmegaConf.create({"workspace_path": str(ROOT), "no_log": True})
    return config


def load_stage0(model, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", {})
    if not state:
        raise RuntimeError("Stage-0 checkpoint has no model state")
    if any(
        token in name
        for name in state
        for token in ("trajectory_policy", "trajectory_posterior", "trace_vb")
    ):
        raise RuntimeError("Stage-0 checkpoint unexpectedly contains TRACE state")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("unexpected Stage-0 keys: " + ", ".join(incompatible.unexpected_keys[:10]))
    if hasattr(model, "_copy_path_adapter_to_cot_encoder_adapter"):
        model._copy_path_adapter_to_cot_encoder_adapter()
    if hasattr(model, "_load_and_validate_sufficiency_cache"):
        model._load_and_validate_sufficiency_cache()
    del checkpoint, state
    gc.collect()


def select_stress_indices(model, dataset, count: int):
    # The cache intentionally masks rows whose first CoT prefix already
    # exposes the answer.  Such a row is valid training data for the answer
    # and semantic objectives, but it cannot exercise the sufficiency head.
    # A mechanism smoke must therefore choose the longest *supervised* rows
    # so that its non-zero-gradient assertion tests a real target rather than
    # failing on an intentionally absent label.
    cache = getattr(model, "_sufficiency_cache", None)
    if not isinstance(cache, dict) or "by_idx" not in cache:
        raise RuntimeError("validated sufficiency cache is unavailable")
    supervised = {
        int(index): int(sum(bool(value) for value in row["role_valid_mask"][:-1]))
        for index, row in cache["by_idx"].items()
        if any(bool(value) for value in row["role_valid_mask"][:-1])
    }
    scored = []
    for index in dataset.get_all_indices():
        if int(index) not in supervised:
            continue
        row = dataset.data[index]
        q_len = len(model.tokenizer.encode(str(row["question"]), add_special_tokens=False))
        cot_len = len(model.tokenizer.encode(str(row["steps"]), add_special_tokens=False))
        scored.append(
            (
                q_len + cot_len + 8 * str(row["steps"]).count("="),
                cot_len,
                q_len,
                int(index),
                supervised[int(index)],
            )
        )
    requested = max(2, int(count))
    selected = sorted(scored, reverse=True)[:requested]
    if len(selected) != requested:
        raise RuntimeError(
            f"need {requested} supervised stress rows, found {len(selected)}"
        )
    return [
        {
            "dataset_index": i,
            "risk_score": int(s),
            "cot_tokens": int(c),
            "question_tokens": int(q),
            "active_sufficiency_roles": int(active),
        }
        for s, c, q, i, active in selected
    ]


def module_grad_norm(module: torch.nn.Module) -> float:
    grads = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(grads).norm().cpu()) if grads else 0.0


def path_adapter_grad_norm(model) -> float:
    marker = f".{model.path_adapter_name}."
    grads = [
        p.grad.detach().float().norm()
        for name, p in model.llm.named_parameters()
        if marker in name and p.grad is not None
    ]
    return float(torch.stack(grads).norm().cpu()) if grads else 0.0


def memory_snapshot(device: torch.device, phase: str) -> dict:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    return {
        "phase": phase,
        "free_mib": float(free / MIB),
        "total_mib": float(total / MIB),
        "reserved_mib": float(reserved / MIB),
        "external_or_nonallocator_mib": float(max(0, total - free - reserved) / MIB),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--stress-cases", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("TRACE-VB Stage-1 smoke requires CUDA")
    config = build_config()
    if str(config.model.model_kwargs.trace_policy_config.answer_context_mode) != "path_only":
        raise RuntimeError("Stage-1 smoke did not load path_only")
    if int(config.model.model_kwargs.trace_policy_config.stage1_stochastic_paths) != 1:
        raise RuntimeError("Stage-1 smoke requires exactly one stochastic path")
    if int(config.model.model_kwargs.trace_policy_config.stage1_posterior_samples) != 0:
        raise RuntimeError("Stage-1 smoke forbids posterior paths")
    policy_config = config.model.model_kwargs.trace_policy_config
    if (
        float(policy_config.stage1_minimum_action_efficacy_ratio) <= 0.0
        or float(policy_config.stage1_action_efficacy_weight) <= 0.0
    ):
        raise RuntimeError("Stage-1 action-efficacy objective is not active")

    data = instantiate_from_config(config.data_module, extra_kwargs={"all_config": config})
    data.setup("fit")
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    load_stage0(model, args.stage0_checkpoint)
    for name in ("plan_forecaster", "semantic_projection", "sufficiency_head", "value_critic"):
        if not hasattr(model, name):
            raise RuntimeError(f"TRACE-VB model is missing {name}")
    if getattr(model, "trajectory_posterior", None) is not None:
        raise RuntimeError("TRACE-VB must not instantiate a CoT posterior")
    if tuple(model.semantic_projection.shape) != (256, model.hidden_size):
        raise RuntimeError("semantic projection has the wrong fixed shape")
    if bool(model.answer_reads_question):
        raise RuntimeError("path_only model still exposes the raw question to the answer decoder")
    if model.trajectory_policy.is_stochastic_step(TRACE_COMMIT_INDEX):
        raise RuntimeError("COMMIT is incorrectly stochastic")
    mask = model.trajectory_policy.stochastic_action_mask(batch_size=1)
    if bool(mask[0, TRACE_COMMIT_INDEX]) or int(mask.sum().item()) != 7:
        raise RuntimeError("stochastic action mask must contain seven actions and exclude COMMIT")

    stress_cases = select_stress_indices(model, data.train_set, args.stress_cases)
    device = torch.device("cuda:0")
    model = model.to(device).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.model.training_kwargs.optimizer.lr),
        weight_decay=float(config.model.training_kwargs.optimizer.weight_decay),
        foreach=False,
    )
    snapshots = [memory_snapshot(device, "post_initialization")]
    torch.cuda.reset_peak_memory_stats(device)
    reports = []
    for case in stress_cases:
        row = data.train_set[case["dataset_index"]]
        batch = default_collate([row])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward(batch)
        snapshots.append(memory_snapshot(device, "post_forward"))
        loss = output["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Stage-1 smoke produced non-finite total loss")
        efficacy_keys = (
            "trace_vb_action_efficacy_loss",
            "trace_vb_action_efficacy_ratio",
            "lambda_action_efficacy_eff",
        )
        missing_efficacy = [key for key in efficacy_keys if key not in output]
        if missing_efficacy:
            raise RuntimeError(
                "Stage-1 output is missing action efficacy: "
                + ", ".join(missing_efficacy)
            )
        if not all(
            bool(torch.isfinite(output[key].detach()).all())
            for key in efficacy_keys
        ):
            raise RuntimeError("Stage-1 action-efficacy outputs are non-finite")
        efficacy_loss = output["trace_vb_action_efficacy_loss"]
        efficacy_ratio = output["trace_vb_action_efficacy_ratio"]
        efficacy_weight = output["lambda_action_efficacy_eff"]
        if not efficacy_loss.requires_grad:
            raise RuntimeError("Stage-1 action-efficacy loss is detached")
        if float(efficacy_weight.detach().cpu()) <= 0.0:
            raise RuntimeError("Stage-1 action-efficacy loss has zero weight")
        reconstructed_loss = (
            output["lambda_answer_eff"] * output["answer_loss"]
            + output["lambda_plan_forecast_eff"]
            * output["trace_vb_plan_forecast_loss"]
            + output["lambda_solve_eff"] * output["trace_vb_solve_loss"]
            + output["lambda_refine_eff"] * output["trace_vb_refine_loss"]
            + output["lambda_sufficiency_eff"]
            * output["trace_vb_sufficiency_loss"]
            + output["lambda_variance_eff"]
            * output["trace_vb_variance_floor_loss"]
            + efficacy_weight * efficacy_loss
        )
        if not torch.allclose(
            loss.float(),
            reconstructed_loss.float(),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(
                "Stage-1 total loss does not include the efficacy objective"
            )
        efficacy_parameters = [
            parameter
            for parameter in model.trajectory_policy.parameters()
            if parameter.requires_grad
        ]
        efficacy_gradients = torch.autograd.grad(
            efficacy_loss,
            efficacy_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        efficacy_gradient_norm = float(
            torch.stack(
                [
                    gradient.detach().float().norm()
                    for gradient in efficacy_gradients
                    if gradient is not None
                ]
                or [loss.detach().new_zeros(())]
            ).norm().cpu()
        )
        if (
            float(efficacy_loss.detach().cpu()) > 1e-8
            and not efficacy_gradient_norm > 0.0
        ):
            raise RuntimeError(
                "active Stage-1 efficacy hinge has no policy gradient"
            )
        loss.backward()
        snapshots.append(memory_snapshot(device, "post_backward"))
        gradients = {
            "trajectory_policy": module_grad_norm(model.trajectory_policy),
            "plan_forecaster": module_grad_norm(model.plan_forecaster),
            "sufficiency_head": module_grad_norm(model.sufficiency_head),
            "path_adapter": path_adapter_grad_norm(model),
        }
        missing = [name for name, value in gradients.items() if not value > 0.0]
        if missing:
            raise RuntimeError("missing required Stage-1 gradients: " + ", ".join(missing))
        for key in (
            "trace_vb_mean_path_answer_only",
            "trace_vb_question_only_paths",
            "trace_vb_path_only_commit_readout",
            "trace_vb_stochastic_path_count",
        ):
            if key not in output or int(output[key].item()) != 1:
                raise RuntimeError(f"Stage-1 runtime contract failed: {key}")
        if int(output.get("trace_answer_question_access", loss.new_ones(())).item()) != 0:
            raise RuntimeError("Stage-1 answer readout accessed the raw question")
        torch.nn.utils.clip_grad_norm_(trainable, 0.3)
        optimizer.step()
        snapshots.append(memory_snapshot(device, "post_optimizer"))
        reports.append(
            {
                **case,
                "source_id": int(row["source_id"]),
                "total_loss": float(loss.detach().float().cpu()),
                "action_efficacy_loss": float(
                    efficacy_loss.detach().float().cpu()
                ),
                "action_efficacy_ratio": float(
                    efficacy_ratio.detach().float().cpu()
                ),
                "action_efficacy_weight": float(
                    efficacy_weight.detach().float().cpu()
                ),
                "action_efficacy_gradient_norm": efficacy_gradient_norm,
                "action_efficacy_in_total_loss": True,
                "gradients": gradients,
            }
        )
        del batch, output, loss
        gc.collect()
        torch.cuda.empty_cache()
        snapshots.append(memory_snapshot(device, "post_cleanup"))

    peak_reserved = float(torch.cuda.max_memory_reserved(device) / MIB)
    total_mib = snapshots[0]["total_mib"]
    max_external = max(item["external_or_nonallocator_mib"] for item in snapshots)
    effective_headroom = total_mib - peak_reserved - max_external
    minimum_free = min(item["free_mib"] for item in snapshots)
    failures = []
    if effective_headroom < MIN_HEADROOM_MIB:
        failures.append(f"effective_headroom={effective_headroom:.1f}<{MIN_HEADROOM_MIB:.1f}")
    if minimum_free < MIN_HEADROOM_MIB:
        failures.append(f"minimum_free={minimum_free:.1f}<{MIN_HEADROOM_MIB:.1f}")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "scope": "real_trace_vb_stage1_smoke_not_an_experiment_result",
        "stage0_checkpoint": str(args.stage0_checkpoint.resolve()),
        "stage0_sha256": sha256_file(args.stage0_checkpoint),
        "registered_train_questions": len(data.train_set),
        "role_schema": ["PLAN", "SOLVE1", "SOLVE2", "SOLVE3", "SOLVE4", "SOLVE5", "REFINE", "COMMIT"],
        "answer_context": "path_only",
        "stochastic_paths": 1,
        "posterior_paths": 0,
        "minimum_action_efficacy_ratio": float(
            policy_config.stage1_minimum_action_efficacy_ratio
        ),
        "action_efficacy_weight": float(
            policy_config.stage1_action_efficacy_weight
        ),
        "optimizer_steps": len(reports),
        "stress_cases": reports,
        "peak_reserved_mib": peak_reserved,
        "max_external_or_nonallocator_mib": max_external,
        "effective_peak_headroom_mib": effective_headroom,
        "minimum_observed_free_mib": minimum_free,
        "required_headroom_mib": MIN_HEADROOM_MIB,
        "memory_snapshots": snapshots,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("TRACE-VB Stage-1 memory gate failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
