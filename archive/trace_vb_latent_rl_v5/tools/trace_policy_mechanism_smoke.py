#!/usr/bin/env python3
"""Real Stage-1 role-semantic stress preflight for formal TRACE.

This is not an experiment result. It loads the exact fresh Stage-0 checkpoint,
selects long registered training examples, exercises the complete Stage-1
objective at the deployed 48-token budget, allocates AdamW state, and checks a
deterministic COMMIT-only deployment generation before training is launched.
"""

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

ROOT = Path(
    os.environ.get("TRACE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(ROOT))

from src.utils.utils import instantiate_from_config  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gradient_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().float().norm()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).norm().cpu())


def path_adapter_gradient_norm(model) -> float:
    values = [
        parameter.grad.detach().float().norm()
        for name, parameter in model.llm.named_parameters()
        if f".{model.path_adapter_name}." in name
        and parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).norm().cpu())


def named_gradient_norm(model, prefix: str) -> float:
    values = [
        parameter.grad.detach().float().norm()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).norm().cpu())


def build_config():
    trainer = OmegaConf.load(ROOT / "src/configs/trainer/default.yaml")
    model = OmegaConf.load(
        ROOT / "src/configs/models/trace_policy_qwen3_instruct.yaml"
    )
    dataset = OmegaConf.load(
        ROOT / "src/configs/datasets/gsm8k_aug_nl.yaml"
    )
    config = OmegaConf.merge(trainer, model, dataset)
    config.dataloader.batch_size = 1
    config.dataloader.val_batch_size = 1
    config.dataloader.num_workers = 0
    config.dataloader.persistent_workers = False
    config.args = OmegaConf.create(
        {
            "workspace_path": str(ROOT),
            "no_log": True,
        }
    )
    return config


def select_stress_indices(model, train_set, count: int):
    scored = []
    for index in train_set.get_all_indices():
        row = train_set.data[index]
        question_tokens = len(
            model.tokenizer.encode(
                model.question_template.format(row["question"])
                + model.speed_template.format(1)
                + model.thinking_separator,
                add_special_tokens=False,
            )
        )
        cot_tokens = len(
            model.tokenizer.encode(
                row["steps"],
                add_special_tokens=False,
            )
        )
        equation_count = row["steps"].count("=")
        score = question_tokens + cot_tokens + 8 * equation_count
        scored.append(
            (
                score,
                cot_tokens,
                question_tokens,
                equation_count,
                index,
            )
        )
    selected = sorted(scored, reverse=True)[: max(2, int(count))]
    return [
        {
            "dataset_index": int(index),
            "risk_score": int(score),
            "cot_tokens": int(cot_tokens),
            "question_tokens": int(question_tokens),
            "equation_count": int(equation_count),
        }
        for (
            score,
            cot_tokens,
            question_tokens,
            equation_count,
            index,
        ) in selected
    ]


def load_stage0(model, checkpoint_path: Path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", {})
    if not state_dict:
        raise RuntimeError("Stage-0 checkpoint has no model state")
    prohibited = [
        name
        for name in state_dict
        if "trajectory_policy" in name
        or "trajectory_posterior" in name
        or ".trace_" in name
    ]
    if prohibited:
        raise RuntimeError(
            "Stress preflight requires a fresh non-TRACE Stage-0 checkpoint"
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            "Unexpected Stage-0 keys: " + ", ".join(unexpected[:10])
        )
    del checkpoint, state_dict
    gc.collect()
    model._copy_path_adapter_to_cot_encoder_adapter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--stress-cases", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "run_outputs/smoke/mechanism_stress.json",
    )
    args = parser.parse_args()
    if not args.stage0_checkpoint.is_file():
        raise SystemExit(
            f"Missing Stage-0 checkpoint: {args.stage0_checkpoint}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("The real-model stress preflight requires one GPU")

    torch.manual_seed(20260720)
    config = build_config()
    deployed_budget = int(
        config.model.model_kwargs.hybrid_generation_config.max_new_tokens
    )
    target_budget = int(
        config.model.model_kwargs.trace_policy_config
        .compact_target_max_new_tokens
    )
    if target_budget != deployed_budget:
        raise RuntimeError(
            "Stage-1 compact target and deployment budgets do not match"
        )

    data_module = instantiate_from_config(
        config.data_module,
        extra_kwargs={"all_config": config},
    )
    data_module.setup("fit")
    model = instantiate_from_config(
        config.model,
        extra_kwargs={"all_config": config},
    )
    load_stage0(model, args.stage0_checkpoint)
    stress_cases = select_stress_indices(
        model,
        data_module.train_set,
        args.stress_cases,
    )
    model = model.cuda()
    model.train()
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.model.training_kwargs.optimizer.lr),
        weight_decay=float(
            config.model.training_kwargs.optimizer.weight_decay
        ),
        foreach=False,
    )
    torch.cuda.reset_peak_memory_stats()

    case_reports = []
    posterior_context_gradient_seen = False
    for case_number, case in enumerate(stress_cases):
        row = data_module.train_set[case["dataset_index"]]
        batch = default_collate([row])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward(batch)
        total_loss = output["total_loss"]
        if not torch.isfinite(total_loss):
            raise RuntimeError("Stage-1 stress produced a non-finite loss")
        total_loss.backward()
        gradients = {
            "trajectory_posterior": gradient_norm(
                model.trajectory_posterior
            ),
            "trajectory_policy": gradient_norm(model.trajectory_policy),
            "plan_policy_head": named_gradient_norm(
                model,
                "trajectory_policy.mean_heads.plan.",
            ),
            "solve_policy_head": named_gradient_norm(
                model,
                "trajectory_policy.mean_heads.solve.",
            ),
            "check_policy_head": named_gradient_norm(
                model,
                "trajectory_policy.mean_heads.check.",
            ),
            "commit_policy_head": named_gradient_norm(
                model,
                "trajectory_policy.mean_heads.commit.",
            ),
            "trajectory_dynamics": (
                gradient_norm(model.trajectory_policy.action_projector)
                + gradient_norm(model.trajectory_policy.base_projector)
            ),
            "posterior_context": gradient_norm(
                model.posterior_context_norm
            ),
            "path_adapter": path_adapter_gradient_norm(model),
        }
        required = {
            name: value
            for name, value in gradients.items()
            if name != "posterior_context"
        }
        missing = [name for name, value in required.items() if value <= 0.0]
        if missing:
            raise RuntimeError(
                "Required Stage-1 gradients are absent: "
                + ", ".join(missing)
            )
        if gradients["posterior_context"] > 0.0:
            posterior_context_gradient_seen = True
        compact_target_max = int(
            output["trace_stage1_compact_target_tokens_max"].item()
        )
        if compact_target_max > deployed_budget:
            raise RuntimeError(
                f"Compact target has {compact_target_max} tokens, "
                f"deployment budget is {deployed_budget}"
            )
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.3)
        optimizer.step()
        case_reports.append(
            {
                **case,
                "source_id": int(row["source_id"]),
                "total_loss": float(total_loss.detach().float().cpu()),
                "compact_target_tokens": int(
                    output["trace_stage1_compact_target_tokens"].item()
                ),
                "compact_target_tokens_max": compact_target_max,
                "plan_loss": float(
                    output["trace_stage1_plan_loss"].item()
                ),
                "solve_loss": float(
                    output["trace_stage1_solve_loss"].item()
                ),
                "check_loss": float(
                    output["trace_stage1_check_loss"].item()
                ),
                "posterior_prior_kl": float(
                    output["trace_stage1_posterior_prior_kl"].item()
                ),
                "role_entropy_floor_loss": float(
                    output["trace_stage1_role_entropy_loss"].item()
                ),
                "posterior_std": float(
                    output["trace_stage1_posterior_std"].item()
                ),
                "prior_std": float(
                    output["trace_stage1_prior_std"].item()
                ),
                "role_contract": int(
                    output["trace_stage1_role_contract"].item()
                ),
                "commit_is_deterministic": int(
                    output[
                        "trace_stage1_commit_is_deterministic"
                    ].item()
                ),
                "commit_only_readout": int(
                    output["trace_stage1_commit_only_readout"].item()
                ),
                "gradients": gradients,
                "cuda_memory_allocated_mib": float(
                    torch.cuda.memory_allocated() / (1024**2)
                ),
            }
        )
        del batch, output, total_loss
        gc.collect()
        torch.cuda.empty_cache()

    if not posterior_context_gradient_seen:
        raise RuntimeError(
            "CoT posterior context remained disconnected after a real "
            "optimizer update"
        )

    if any(
        report[contract] != 1
        for report in case_reports
        for contract in (
            "role_contract",
            "commit_is_deterministic",
            "commit_only_readout",
        )
    ):
        raise RuntimeError("Stage-1 role or COMMIT contract was not enforced")

    model.zero_grad(set_to_none=True)
    model.eval()
    deployment_row = data_module.train_set[
        stress_cases[0]["dataset_index"]
    ]
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        trajectory = model._trajectory_latents(
            [deployment_row["question"]],
            deterministic=True,
        )
        generated = model._generate_answers_from_trajectory(
            trajectory,
            do_sample=False,
        )
    generated_tokens = int(
        generated.ne(model.tokenizer.pad_token_id).sum().item()
    )
    if generated_tokens > deployed_budget:
        raise RuntimeError(
            f"Deployment generated {generated_tokens} tokens under a "
            f"{deployed_budget}-token budget"
        )

    report = {
        "status": "PASS",
        "scope": "real_stage1_stress_preflight_not_an_experiment_result",
        "stage0_checkpoint": str(args.stage0_checkpoint.resolve()),
        "stage0_checkpoint_sha256": sha256_file(
            args.stage0_checkpoint
        ),
        "registered_train_split_size": len(data_module.train_set),
        "explicit_cots_per_question": 1,
        "role_schema": [
            "PLAN",
            "SOLVE1",
            "SOLVE2",
            "SOLVE3",
            "SOLVE4",
            "SOLVE5",
            "CHECK",
            "COMMIT",
        ],
        "posterior_paths_per_question": int(
            config.model.model_kwargs.trace_policy_config
            .stage1_posterior_samples
        ),
        "supervised_map_paths_per_question": 1,
        "compact_target_budget": target_budget,
        "deployment_generation_budget": deployed_budget,
        "optimizer": "AdamW_with_state_allocation",
        "optimizer_steps": len(case_reports),
        "stress_cases": case_reports,
        "posterior_context_gradient_after_optimizer_step": True,
        "generated_token_count": generated_tokens,
        "answer_question_attention_access": int(
            model.answer_reads_question
        ),
        "answer_latent_attention_access": 1,
        "answer_latent_attention_role": "COMMIT",
        "peak_cuda_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated() / (1024**2)
        ),
        "peak_cuda_memory_reserved_mib": float(
            torch.cuda.max_memory_reserved() / (1024**2)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
