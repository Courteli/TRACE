#!/usr/bin/env python3
"""Run paired causal interventions on a trained final TRACE-VB checkpoint.

Figure contract
---------------
Core conclusion:
    The deployed answer decoder has no direct question access and reads only
    deterministic COMMIT.  The primary causal test therefore replaces one
    PLAN/SOLVE/REFINE action at equal norm, recomputes every downstream state
    and COMMIT, and measures the paired answer change.  Masking COMMIT tests
    the sole deployed readout channel.
Evidence logic:
    All interventions are paired within the same 200 questions. Reverse,
    shuffle, and random controls preserve latent count. Random action controls
    preserve action norm. PLAN through REFINE interventions keep the original
    prefix, replace one action, and regenerate the suffix. COMMIT is not a
    stochastic action, so its registered eighth-position intervention masks
    the COMMIT readout instead of inventing a random COMMIT action.  Requested
    prefixes 1--7 are intentionally identical to no-path because the decoder
    cannot read private states; the 0--8 curve is only a COMMIT-bottleneck
    sanity check, never evidence of incremental step sufficiency.
Review risks:
    Equal-norm replacement can move a state off the learned action manifold.
    The result establishes behavioral sensitivity to a role transition under
    the registered intervention, not human-readable semantics, necessity of a
    particular computation, or a claim that prefixes 1--7 are independently
    readable by the answer decoder.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.utils import instantiate_from_config


PINK = "#E5A3BF"
PINK_DARK = "#B95C88"
BLUE = "#6687B8"
GREEN = "#69AD7C"
ORANGE = "#E5A11A"
INK = "#30343B"
GRID = "#DDE2E8"


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Liberation Serif",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def save_figure(fig, output_base: Path):
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def load_model(checkpoint_path: Path, device: torch.device):
    hparams_path = checkpoint_path.parent.parent / "hparams.yaml"
    if not hparams_path.exists():
        raise FileNotFoundError(f"missing checkpoint hparams: {hparams_path}")
    saved = OmegaConf.load(hparams_path)
    config = saved.all_config
    target = str(config.model.target)
    if target != "src.models.trace_vb.LitTRACEVB":
        raise ValueError(
            "formal TRACE-VB causal evidence requires LitTRACEVB, got "
            f"{target}"
        )
    if str(
        config.model.model_kwargs.trace_policy_config.answer_context_mode
    ) != "question_and_commit":
        raise ValueError("TRACE-VB-v4 causal audit requires question_and_commit")
    if not bool(config.model.model_kwargs.do_trace_rl):
        raise ValueError("causal audit requires a final Stage-2 checkpoint")
    if int(config.model.model_kwargs.readcot_config.n_latents) != 8:
        raise ValueError("formal causal audit requires exactly eight latents")
    config.model.model_kwargs.trace_policy_config.visual_record_limit = 0
    model = instantiate_from_config(
        config.model,
        extra_kwargs={"all_config": config},
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("trace_policy_training_stage", -1)) != 2:
        raise ValueError("causal audit requires a genuine Stage-2 checkpoint")
    model.on_load_checkpoint(checkpoint)
    if not model._reference_restored:
        raise ValueError(
            "causal audit could not restore the immutable Stage-1 policy "
            "reference"
        )
    load_result = model.load_state_dict(
        checkpoint["state_dict"],
        strict=False,
    )
    unexpected = [
        key
        for key in load_result.unexpected_keys
        if not key.startswith("stage1_policy_reference.")
    ]
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected[:8]}")
    model.to(device)
    model.eval()
    if not bool(model.answer_reads_question):
        raise ValueError(
            "TRACE-VB-v4 causal evidence requires the reliable raw-question "
            "channel in the answer decoder"
        )
    return model


def load_records(path: Path, count: int):
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or len(records) < int(count):
        raise ValueError(f"{path} must contain at least {count} records")
    records = records[: int(count)]
    required = (
        "idx",
        "question",
        "answer",
        "map_correct",
        "rollout_schema",
        "visualization_contract",
    )
    for row, record in enumerate(records):
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"record {row} is missing {missing}")
        if record["rollout_schema"] not in {
            "iid_conditional_gaussian",
            "iid_role_conditioned_gaussian_with_deterministic_commit",
        }:
            raise ValueError(f"record {row} is not an IID policy record")
        contract = record["visualization_contract"]
        if contract.get("manual_offsets") is not False:
            raise ValueError(f"record {row} permits manual path offsets")
        if contract.get("per_path_rescaling") is not False:
            raise ValueError(f"record {row} permits per-path rescaling")
    return records


def decode_accuracy(model, output_ids, answer: str) -> Tuple[float, int]:
    text = model.tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]
    prediction = model.extract_answer_from_output(text)
    accuracy = float(
        model.verify_answer(
            gt_answer=answer,
            pred_answer=prediction,
        )
    )
    length = int(
        output_ids[0]
        .ne(model.tokenizer.pad_token_id)
        .sum()
        .item()
    )
    return accuracy, length


def evaluate_path(
    model,
    trajectory: Dict[str, torch.Tensor],
    answer: str,
    *,
    latent_read_mask: torch.Tensor = None,
) -> Dict[str, float]:
    output_ids = model._generate_answers_from_trajectory(
        trajectory,
        do_sample=False,
        latent_read_mask=latent_read_mask,
    )
    accuracy, length = decode_accuracy(model, output_ids, answer)
    score = float(
        model._gold_answer_scores(
            trajectory,
            [answer],
            latent_read_mask=latent_read_mask,
        )[0].item()
    )
    return {
        "accuracy": accuracy,
        "length": float(length),
        "gold_score": score,
    }


def same_norm_random_actions(
    actions: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    direction = torch.randn(
        actions.shape,
        generator=generator,
        dtype=torch.float32,
    ).to(actions.device)
    direction = direction / direction.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    return direction.to(actions.dtype) * actions.norm(
        dim=-1,
        keepdim=True,
    )


def forced_trajectory(
    model,
    question: str,
    actions: torch.Tensor,
    *,
    prefix_length: int,
) -> Dict[str, torch.Tensor]:
    n_steps = actions.shape[1]
    mask = torch.zeros(
        1,
        n_steps,
        device=actions.device,
        dtype=torch.bool,
    )
    mask[:, : int(prefix_length)] = True
    innovations = torch.zeros_like(actions)
    return model._trajectory_latents(
        [question],
        innovations=innovations,
        forced_actions=actions,
        forced_action_mask=mask,
    )


def paired_drop(
    baseline: Sequence[float],
    intervention: Sequence[float],
    *,
    rng: np.random.Generator,
    bootstrap: int,
) -> Dict[str, object]:
    base = np.asarray(baseline, dtype=np.float64)
    changed = np.asarray(intervention, dtype=np.float64)
    if base.shape != changed.shape:
        raise ValueError("paired intervention vectors have different shapes")
    drop = base - changed
    indices = rng.integers(
        0,
        len(drop),
        size=(int(bootstrap), len(drop)),
    )
    sampled = drop[indices].mean(axis=1)
    return {
        "baseline": float(base.mean()),
        "intervention": float(changed.mean()),
        "drop": float(drop.mean()),
        "drop_ci95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "positive_fraction": float((drop > 0).mean()),
        "n_questions": int(len(drop)),
    }


def matrix_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap: int,
    familywise_alpha: float = 0.05,
) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("matrix_mean_ci expects a nonempty [question, test] matrix")
    if not np.isfinite(values).all():
        raise ValueError("matrix_mean_ci received non-finite values")
    if int(bootstrap) < 1:
        raise ValueError("bootstrap must be positive")
    if not 0.0 < float(familywise_alpha) < 1.0:
        raise ValueError("familywise_alpha must lie in (0, 1)")
    indices = rng.integers(
        0,
        values.shape[0],
        size=(int(bootstrap), values.shape[0]),
    )
    means = values[indices].mean(axis=1)
    family_size = int(values.shape[1])
    tail = float(familywise_alpha) / (2.0 * family_size)
    return {
        "mean": values.mean(axis=0).tolist(),
        "ci95_low": np.quantile(means, 0.025, axis=0).tolist(),
        "ci95_high": np.quantile(means, 0.975, axis=0).tolist(),
        "simultaneous_familywise95_low": np.quantile(
            means,
            tail,
            axis=0,
        ).tolist(),
        "simultaneous_familywise95_high": np.quantile(
            means,
            1.0 - tail,
            axis=0,
        ).tolist(),
        "simultaneous_interval_correction": "Bonferroni bootstrap percentile",
        "familywise_alpha": float(familywise_alpha),
        "family_size": family_size,
        "n_questions": int(values.shape[0]),
    }


def summarize_commit_bottleneck_sanity(
    accuracy_values: np.ndarray,
    gold_score_values: np.ndarray,
    *,
    atol: float = 1e-7,
) -> Dict[str, object]:
    """Verify the question+COMMIT meaning of the requested-prefix curve.

    Columns 0--7 retain the raw question but have no COMMIT readout: column
    zero requests no latent and columns 1--7 request private states that the
    deployed decoder is forbidden to read.  Column eight adds COMMIT.
    Consequently, the first eight columns must agree up to numerical
    tolerance; this is an architectural sanity check and is not a step-level
    causal attribution.
    """
    accuracy_values = np.asarray(accuracy_values, dtype=np.float64)
    gold_score_values = np.asarray(gold_score_values, dtype=np.float64)
    if (
        accuracy_values.ndim != 2
        or gold_score_values.shape != accuracy_values.shape
        or accuracy_values.shape[0] < 1
        or accuracy_values.shape[1] != 9
    ):
        raise ValueError(
            "COMMIT-bottleneck curves must share shape [question, 9]"
        )
    if not (
        np.isfinite(accuracy_values).all()
        and np.isfinite(gold_score_values).all()
    ):
        raise ValueError("COMMIT-bottleneck curves contain non-finite values")
    accuracy_delta = float(
        np.max(np.abs(accuracy_values[:, :8] - accuracy_values[:, [0]]))
    )
    score_delta = float(
        np.max(np.abs(gold_score_values[:, :8] - gold_score_values[:, [0]]))
    )
    passed = accuracy_delta <= float(atol) and score_delta <= float(atol)
    return {
        "status": "PASS" if passed else "FAIL",
        "prefix_0_through_7_equal_question_only": bool(passed),
        "max_abs_accuracy_delta": accuracy_delta,
        "max_abs_gold_score_delta": score_delta,
        "absolute_tolerance": float(atol),
        "prefix_8_enables_commit": True,
        "interpretation": (
            "architectural COMMIT-bottleneck sanity only; columns 1--7 do "
            "not measure incremental step sufficiency or contribution"
        ),
    }


def plot_accuracy(report: dict, output_base: Path):
    names = (
        "baseline",
        "no_path",
        "reverse",
        "shuffle",
        "mean_repeat",
        "random",
    )
    labels = (
        "Full path",
        "No path",
        "Reverse",
        "Shuffle",
        "Mean repeat",
        "Random",
    )
    values = [report["accuracy"][name]["intervention"] for name in names]
    values[0] = report["accuracy"]["baseline"]["baseline"]
    colors = [BLUE, ORANGE, GREEN, PINK, BLUE, PINK_DARK]
    fig, ax = plt.subplots(figsize=(4.25, 2.45))
    bars = ax.bar(
        np.arange(len(names)),
        np.asarray(values) * 100.0,
        color=colors,
        edgecolor=INK,
        linewidth=0.45,
        width=0.68,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value * 100.0:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_xticks(np.arange(len(names)), labels, rotation=18, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Answer sensitivity to path interventions",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    fig.tight_layout(pad=0.6)
    save_figure(fig, output_base)


def plot_commit_bottleneck(
    prefix_score: dict,
    prefix_accuracy: dict,
    output_base: Path,
):
    steps = np.arange(len(prefix_score["mean"]))
    fig, axes = plt.subplots(1, 2, figsize=(5.25, 2.35))
    for ax, values, ylabel, scale in (
        (axes[0], prefix_accuracy, "Accuracy (%)", 100.0),
        (
            axes[1],
            prefix_score,
            "Gold-answer log probability",
            1.0,
        ),
    ):
        mean = np.asarray(values["mean"]) * scale
        low = np.asarray(values["ci95_low"]) * scale
        high = np.asarray(values["ci95_high"]) * scale
        ax.plot(
            steps,
            mean,
            color=PINK_DARK,
            marker="o",
            markersize=3.8,
            linewidth=1.7,
        )
        ax.fill_between(steps, low, high, color=PINK, alpha=0.24)
        ax.set_xticks(steps)
        ax.axvline(8, color=INK, linestyle="--", linewidth=0.8)
        ax.set_xlabel("Available states (only state 8/COMMIT is readable)")
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID, linewidth=0.5)
    axes[0].set_title(
        "COMMIT-bottleneck sanity",
        loc="left",
        fontweight="bold",
    )
    axes[1].set_title(
        "No-readout vs COMMIT support",
        loc="left",
        fontweight="bold",
    )
    fig.tight_layout(pad=0.6, w_pad=1.25)
    save_figure(fig, output_base)


def plot_transition(
    transition_score: dict,
    transition_accuracy: dict,
    output_base: Path,
):
    steps = np.arange(1, len(transition_score["mean"]) + 1)
    role_labels = (
        "PLAN",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "REFINE",
        "COMMIT",
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.25, 2.35))
    for ax, values, ylabel, scale in (
        (axes[0], transition_accuracy, "Accuracy drop (pp)", 100.0),
        (axes[1], transition_score, "Gold-score drop", 1.0),
    ):
        mean = np.asarray(values["mean"]) * scale
        low = (
            np.asarray(values["simultaneous_familywise95_low"]) * scale
        )
        high = (
            np.asarray(values["simultaneous_familywise95_high"]) * scale
        )
        errors = np.stack([mean - low, high - mean])
        ax.axhline(0.0, color=INK, linestyle="--", linewidth=0.9)
        ax.bar(
            steps,
            mean,
            color=PINK,
            edgecolor=PINK_DARK,
            linewidth=0.65,
            width=0.68,
        )
        ax.errorbar(
            steps,
            mean,
            yerr=errors,
            fmt="none",
            color=INK,
            linewidth=0.8,
            capsize=2.2,
        )
        ax.set_xticks(steps, role_labels, rotation=35, ha="right")
        ax.set_xlabel("Role intervention")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    axes[0].set_title("Answer behavior", loc="left", fontweight="bold")
    axes[1].set_title("Gold-answer support", loc="left", fontweight="bold")
    fig.tight_layout(pad=0.6, w_pad=1.25)
    save_figure(fig, output_base)


def figure_qa(output_dir: Path):
    from PIL import Image, ImageStat

    figures = []
    for svg in sorted(output_dir.glob("*.svg")):
        pdf = svg.with_suffix(".pdf")
        tiff = svg.with_suffix(".tiff")
        if "<text" not in svg.read_text():
            raise ValueError(f"{svg} does not preserve editable text")
        if not pdf.exists() or not tiff.exists():
            raise ValueError(f"incomplete export bundle for {svg.stem}")
        image = Image.open(tiff).convert("RGB")
        extrema = ImageStat.Stat(image).extrema
        if all(low == high for low, high in extrema):
            raise ValueError(f"{tiff} is visually blank")
        figures.append(
            {
                "name": svg.stem,
                "tiff_pixels": list(image.size),
                "svg_editable_text": True,
                "nonblank": True,
            }
        )
    if len(figures) != 3:
        raise ValueError(f"expected three causal figures, found {len(figures)}")
    return {"status": "PASS", "figures": figures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if int(args.count) != 200:
        raise ValueError("formal causal evidence requires exactly 200 questions")
    if int(args.bootstrap) < 10000:
        raise ValueError(
            "formal causal evidence requires at least 10,000 bootstrap draws"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the formal causal audit")
    records = load_records(args.records, args.count)
    model = load_model(args.checkpoint, device)
    if model.n_trace_steps != 8:
        raise ValueError("loaded model does not use eight transitions")

    shuffle_order = torch.tensor(
        [2, 0, 5, 1, 7, 3, 6, 4],
        device=device,
        dtype=torch.long,
    )
    rows = []
    torch.set_grad_enabled(False)
    for row_index, record in enumerate(records):
        question = str(record["question"])
        answer = str(record["answer"])
        trajectory = model._trajectory_latents(
            [question],
            deterministic=True,
        )
        baseline = evaluate_path(model, trajectory, answer)
        if baseline["accuracy"] != float(record["map_correct"]):
            raise RuntimeError(
                f"MAP replay mismatch at question {record['idx']}"
            )
        latent_mask = trajectory["latent_attention_mask"]
        no_path = evaluate_path(
            model,
            trajectory,
            answer,
            latent_read_mask=torch.zeros_like(latent_mask),
        )
        prefix_results = [no_path]
        for prefix in range(1, model.n_trace_steps):
            prefix_mask = torch.zeros_like(latent_mask)
            prefix_mask[:, :prefix] = 1
            resolved = model._resolve_latent_read_mask(
                trajectory,
                prefix_mask,
            )
            if bool(resolved.any()):
                raise RuntimeError(
                    "a pre-COMMIT prefix unexpectedly became readable"
                )
            prefix_result = evaluate_path(
                model,
                trajectory,
                answer,
                latent_read_mask=prefix_mask,
            )
            for metric in ("accuracy", "length", "gold_score"):
                if abs(prefix_result[metric] - no_path[metric]) > 1e-7:
                    raise RuntimeError(
                        "prefix 1--7 must equal question-only under the "
                        f"COMMIT bottleneck; prefix={prefix}, metric={metric}"
                    )
            prefix_results.append(prefix_result)
        prefix_results.append(baseline)

        actions = trajectory["actions"].detach()
        random_actions = same_norm_random_actions(
            actions,
            seed=20260719 + int(record["idx"]),
        )
        intervention_actions = {
            "reverse": actions.flip(dims=[1]),
            "shuffle": actions.index_select(1, shuffle_order),
            "mean_repeat": actions.mean(dim=1, keepdim=True).expand_as(
                actions
            ),
            "random": random_actions,
        }
        interventions = {}
        for name, changed_actions in intervention_actions.items():
            changed = forced_trajectory(
                model,
                question,
                changed_actions,
                prefix_length=model.n_trace_steps,
            )
            interventions[name] = evaluate_path(model, changed, answer)
            del changed

        transition_score_drops = []
        transition_accuracy_drops = []
        for step in range(model.n_trace_steps):
            if step == model.n_trace_steps - 1:
                # COMMIT has no stochastic action by construction. Its causal
                # intervention is removal of the sole latent readout channel.
                changed_result = no_path
                changed = None
            else:
                changed_actions = actions.clone()
                changed_actions[:, step] = random_actions[:, step]
                changed = forced_trajectory(
                    model,
                    question,
                    changed_actions,
                    prefix_length=step + 1,
                )
                if step > 0 and not torch.allclose(
                    changed["actions"][:, :step].float(),
                    actions[:, :step].float(),
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise RuntimeError(
                        "transition intervention did not preserve its prefix"
                    )
                if not torch.allclose(
                    changed["actions"][:, step].float().norm(dim=-1),
                    actions[:, step].float().norm(dim=-1),
                    atol=1e-5,
                    rtol=1e-5,
                ):
                    raise RuntimeError(
                        "transition intervention did not preserve action norm"
                    )
                changed_result = evaluate_path(model, changed, answer)
            transition_score_drops.append(
                baseline["gold_score"] - changed_result["gold_score"]
            )
            transition_accuracy_drops.append(
                baseline["accuracy"] - changed_result["accuracy"]
            )
            if changed is not None:
                del changed

        output = {
            "idx": int(record["idx"]),
            "baseline_accuracy": baseline["accuracy"],
            "baseline_length": baseline["length"],
            "baseline_gold_score": baseline["gold_score"],
            "no_path_accuracy": no_path["accuracy"],
            "no_path_length": no_path["length"],
            "no_path_gold_score": no_path["gold_score"],
        }
        for name, values in interventions.items():
            for metric, value in values.items():
                output[f"{name}_{metric}"] = value
        for available_states, values in enumerate(prefix_results):
            output[
                f"commit_sanity_available_{available_states}_accuracy"
            ] = values["accuracy"]
            output[
                f"commit_sanity_available_{available_states}_gold_score"
            ] = values["gold_score"]
        for step, (score_drop, accuracy_drop) in enumerate(
            zip(transition_score_drops, transition_accuracy_drops),
            start=1,
        ):
            output[f"transition_{step}_accuracy_drop"] = accuracy_drop
            output[f"transition_{step}_gold_drop"] = score_drop
        rows.append(output)
        del trajectory
        if device.type == "cuda" and (row_index + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (row_index + 1) % 10 == 0:
            print(f"causal audit: {row_index + 1}/{len(records)}")

    csv_path = args.output_dir / "question_causal_interventions.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.default_rng(20260719)
    accuracy = {}
    gold_score = {}
    baseline_accuracy = [row["baseline_accuracy"] for row in rows]
    baseline_score = [row["baseline_gold_score"] for row in rows]
    accuracy["baseline"] = paired_drop(
        baseline_accuracy,
        baseline_accuracy,
        rng=rng,
        bootstrap=args.bootstrap,
    )
    gold_score["baseline"] = paired_drop(
        baseline_score,
        baseline_score,
        rng=rng,
        bootstrap=args.bootstrap,
    )
    for name in (
        "no_path",
        "reverse",
        "shuffle",
        "mean_repeat",
        "random",
    ):
        accuracy[name] = paired_drop(
            baseline_accuracy,
            [row[f"{name}_accuracy"] for row in rows],
            rng=rng,
            bootstrap=args.bootstrap,
        )
        gold_score[name] = paired_drop(
            baseline_score,
            [row[f"{name}_gold_score"] for row in rows],
            rng=rng,
            bootstrap=args.bootstrap,
        )
    prefix_score_values = np.asarray(
        [
            [
                row[f"commit_sanity_available_{count}_gold_score"]
                for count in range(9)
            ]
            for row in rows
        ]
    )
    prefix_accuracy_values = np.asarray(
        [
            [
                row[f"commit_sanity_available_{count}_accuracy"]
                for count in range(9)
            ]
            for row in rows
        ]
    )
    transition_score_values = np.asarray(
        [
            [
                row[f"transition_{step}_gold_drop"]
                for step in range(1, 9)
            ]
            for row in rows
        ]
    )
    transition_accuracy_values = np.asarray(
        [
            [
                row[f"transition_{step}_accuracy_drop"]
                for step in range(1, 9)
            ]
            for row in rows
        ]
    )
    prefix_score = matrix_mean_ci(
        prefix_score_values,
        rng=rng,
        bootstrap=args.bootstrap,
    )
    prefix_accuracy = matrix_mean_ci(
        prefix_accuracy_values,
        rng=rng,
        bootstrap=args.bootstrap,
    )
    transition_score = matrix_mean_ci(
        transition_score_values,
        rng=rng,
        bootstrap=args.bootstrap,
    )
    transition_accuracy = matrix_mean_ci(
        transition_accuracy_values,
        rng=rng,
        bootstrap=args.bootstrap,
    )
    transition_score["positive_fraction"] = (
        transition_score_values > 0
    ).mean(axis=0).tolist()
    strict_transition_gate = [
        lower > 0.0
        for lower in transition_score["simultaneous_familywise95_low"]
    ]
    commit_bottleneck_sanity = summarize_commit_bottleneck_sanity(
        prefix_accuracy_values,
        prefix_score_values,
    )
    if commit_bottleneck_sanity["status"] != "PASS":
        raise RuntimeError(
            "requested pre-COMMIT prefixes violated the COMMIT-only latent "
            "contract"
        )
    commit_vs_question_only = {
        "accuracy": paired_drop(
            prefix_accuracy_values[:, -1],
            prefix_accuracy_values[:, 0],
            rng=rng,
            bootstrap=args.bootstrap,
        ),
        "gold_score": paired_drop(
            prefix_score_values[:, -1],
            prefix_score_values[:, 0],
            rng=rng,
            bootstrap=args.bootstrap,
        ),
    }
    checks = {
        "commit_readout_ablation_reduces_support": (
            accuracy["no_path"]["drop_ci95"][0] > 0.0
            and gold_score["no_path"]["drop_ci95"][0] > 0.0
        ),
        "order_sensitivity": (
            gold_score["reverse"]["drop_ci95"][0] > 0.0
            and gold_score["shuffle"]["drop_ci95"][0] > 0.0
        ),
        "mean_repeat_collapse_sensitivity": (
            gold_score["mean_repeat"]["drop_ci95"][0] > 0.0
        ),
        "same_norm_direction_sensitivity": (
            gold_score["random"]["drop_ci95"][0] > 0.0
        ),
        "commit_adds_information_over_no_latent_readout": (
            commit_vs_question_only["accuracy"]["drop_ci95"][0] > 0.0
            and commit_vs_question_only["gold_score"]["drop_ci95"][0] > 0.0
        ),
        "seven_action_replacements_and_commit_ablation_reduce_support": all(
            strict_transition_gate
        ),
    }
    report = {
        "status": "ANALYSIS_COMPLETE",
        "checkpoint": str(args.checkpoint),
        "questions": len(rows),
        "evidence_priority": {
            "primary": (
                "same_norm_transition_replacement_with_suffix_and_commit_"
                "recomputation"
            ),
            "co_primary_elsewhere": "terminal_outcome_value_calibration",
            "diagnostic_only": "commit_bottleneck_sanity_curve",
        },
        "answer_context_contract": "question_plus_COMMIT_latent_only",
        "question_attention_access": True,
        "accuracy": accuracy,
        "gold_score": gold_score,
        "commit_bottleneck_accuracy_curve": prefix_accuracy,
        "commit_bottleneck_gold_score_curve": prefix_score,
        "commit_bottleneck_sanity": commit_bottleneck_sanity,
        "commit_vs_question_only": commit_vs_question_only,
        "transition_accuracy_drop": transition_accuracy,
        "transition_gold_drop": transition_score,
        "transition_intervention_types": [
            "same-norm action replacement plus suffix recomputation"
        ]
        * 7
        + ["COMMIT readout ablation"],
        "transition_replacement_protocol": {
            "roles": "PLAN, SOLVE1--SOLVE5, REFINE",
            "prefix": (
                "all deterministic actions before the replaced role held fixed"
            ),
            "replacement": "random direction with the original action L2 norm",
            "suffix": (
                "all downstream actions and states regenerated from their "
                "conditional means; deterministic COMMIT recomputed"
            ),
            "scope": (
                "behavioral sensitivity under this intervention; not a "
                "human-readable role-semantics or necessity claim"
            ),
        },
        "claim_checks": {
            name: "PASS" if value else "FAIL"
            for name, value in checks.items()
        },
        "claim_boundary": (
            "The deployed decoder reads the raw question and deterministic "
            "COMMIT, but cannot read private role states. Prefix points 1--7 "
            "are question-only controls and make no incremental-step "
            "claim. Transition replacement with suffix recomputation tests "
            "behavioral sensitivity, but an equal-norm random direction may "
            "be off-manifold and does not identify human-readable semantics. "
            "The all-position gate uses family-wise 95% Bonferroni bootstrap "
            "intervals over seven stochastic roles plus COMMIT ablation."
        ),
        "gold_score_protocol": (
            "frozen Stage-1 answer adapter under the same question+COMMIT "
            "mask; private role states influence the score only through the "
            "recomputed COMMIT state"
        ),
    }
    (args.output_dir / "causal_summary.json").write_text(
        json.dumps(report, indent=2)
    )
    plot_accuracy(report, args.output_dir / "causal_accuracy")
    plot_commit_bottleneck(
        prefix_score,
        prefix_accuracy,
        args.output_dir / "commit_bottleneck_sanity",
    )
    plot_transition(
        transition_score,
        transition_accuracy,
        args.output_dir / "transition_effects",
    )
    np.savez_compressed(
        args.output_dir / "causal_source_data.npz",
        commit_bottleneck_accuracy=prefix_accuracy_values,
        commit_bottleneck_gold_scores=prefix_score_values,
        transition_accuracy_drops=transition_accuracy_values,
        transition_gold_drops=transition_score_values,
        baseline_accuracy=np.asarray(baseline_accuracy),
        baseline_gold_score=np.asarray(baseline_score),
    )
    contract = {
        "paired_questions": 200,
        "path_length": 8,
        "question_attention_access": int(model.answer_reads_question),
        "answer_context_contract": "question_plus_COMMIT_latent_only",
        "gold_score_protocol": (
            "frozen Stage-1 answer adapter under the same question+COMMIT "
            "mask; private role states influence the score only through the "
            "recomputed COMMIT state"
        ),
        "primary_evidence": [
            "terminal outcome value calibration (stage comparison)",
            "same-norm transition replacement with suffix and COMMIT recomputation",
        ],
        "diagnostic_only": (
            "the 0--8 availability curve is a COMMIT-bottleneck sanity check; "
            "points 1--7 are question-only controls, not stepwise sufficiency"
        ),
        "controls": [
            "no path",
            "reverse actions",
            "fixed shuffle",
            "mean-repeat path collapse",
            "same-norm random actions",
            "same-norm PLAN-through-REFINE replacement with suffix recomputation",
            "deterministic COMMIT readout ablation",
        ],
        "uncertainty": (
            "question-bootstrap pointwise 95% intervals; transition-wide "
            "claims use family-wise 95% Bonferroni bootstrap intervals"
        ),
        "selection": "first 200 test records; no outcome or geometry selection",
        "exports": ["SVG", "PDF", "TIFF 600 dpi"],
    }
    (args.output_dir / "figure_contract.json").write_text(
        json.dumps(contract, indent=2)
    )
    qa = figure_qa(args.output_dir)
    (args.output_dir / "figure_qa.json").write_text(
        json.dumps(qa, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
