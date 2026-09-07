#!/usr/bin/env python3
"""Visualize TRACE teacher/student hidden-space trajectories."""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.utils import get_position_ids_from_attention_mask, instantiate_from_config


DEFAULT_DATASET = "/home/dingxukai/RoT/data/GSM8k-Aug-NL/gsm8k_test_processed.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET)
    parser.add_argument("--indices", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--trace_steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="run_outputs/trace/trajectory_visualizations/latest")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--origin_ckpt", default="")
    parser.add_argument("--origin_speed", default="5")
    parser.add_argument("--origin_sample", action="store_true")
    return parser.parse_args()


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    return plt


def load_dataset(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "idx": idx,
                    "source_id": item.get("id", idx),
                    "question": item["question"],
                    "steps": item["cot"],
                    "answer": str(item["answer"]),
                }
            )
    return rows


def fit_pca(x: np.ndarray, dim: int = 3):
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(dim, vt.shape[0])]
    return mean, components


def project(x: np.ndarray, mean: np.ndarray, components: np.ndarray, dim: int = 3):
    y = (x - mean) @ components.T
    if y.shape[1] < dim:
        y = np.concatenate([y, np.zeros((y.shape[0], dim - y.shape[1]), dtype=y.dtype)], axis=1)
    return y[:, :dim]


def load_config(ckpt: Path):
    hparams = OmegaConf.load(ckpt.parent.parent / "hparams.yaml")
    config = hparams.get("all_config", None)
    if config is None:
        config = hparams
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config.trainer.logger = False
    config.dataloader.batch_size = 1
    config.dataloader.val_batch_size = 1
    config.dataloader.num_workers = 0
    config.dataloader.persistent_workers = False
    return config


def load_trace_model(args):
    ckpt = Path(args.ckpt)
    config = load_config(ckpt)
    config.model.model_kwargs.trace_trajectory_config.trace_steps = args.trace_steps
    config.model.model_kwargs.trace_trajectory_config.teacher_speed = args.trace_steps
    config.model.model_kwargs.trace_trajectory_config.student_speed = args.trace_steps
    config.model.model_kwargs.latent_generation_config.max_n_latent_forward = args.trace_steps
    config.model.model_kwargs.latent_generation_config.min_n_latent_forward = args.trace_steps
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    print(model.load_state_dict(state, strict=False))
    model.to(args.device)
    model.eval()
    return model


def load_origin_model(args):
    ckpt = Path(args.origin_ckpt)
    config = load_config(ckpt)
    config.model.model_kwargs.latent_generation_config.max_n_latent_forward = args.trace_steps
    config.model.model_kwargs.latent_generation_config.min_n_latent_forward = args.trace_steps
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    print(model.load_state_dict(state, strict=False))
    model.to(args.device)
    model.eval()
    return model


def to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def collect_paths(model, items, trace_steps):
    rows = []
    metrics = []
    for item in items:
        teacher = model.extract_teacher_trajectory(
            questions=[item["question"]],
            steps=[item["steps"]],
            answers=[item["answer"]],
            trace_steps=trace_steps,
        )[0]
        student = model.extract_student_trajectory([item["question"]], trace_steps=trace_steps)[0]
        teacher_delta = teacher[1:] - teacher[:-1]
        student_delta = student[1:] - student[:-1]
        state_cos = F.cosine_similarity(student[1:], teacher[1:], dim=-1)
        transition_cos = F.cosine_similarity(student_delta, teacher_delta, dim=-1)
        if trace_steps > 1:
            step_cos = F.cosine_similarity(student[2:], student[1:-1], dim=-1).mean()
        else:
            step_cos = torch.tensor(0.0, device=student.device)
        rows.append(
            {
                "idx": item["idx"],
                "question": item["question"],
                "answer": item["answer"],
                "teacher": to_numpy(teacher),
                "student": to_numpy(student),
            }
        )
        metrics.append(
            {
                "idx": item["idx"],
                "state_cos": float(state_cos.mean().item()),
                "transition_cos": float(transition_cos.mean().item()),
                "student_step_cos": float(step_cos.item()),
                "teacher_delta_norm": float(teacher_delta.norm(dim=-1).mean().item()),
                "student_delta_norm": float(student_delta.norm(dim=-1).mean().item()),
            }
        )
    return rows, metrics


@torch.no_grad()
def extract_origin_latent_path(model, questions, trace_steps: int, speed: str, use_mean: bool = True):
    suffix = model.speed_template.format(speed) + model.thinking_separator
    question_input_ids, question_attention_mask = model.prepare_inputs(
        questions,
        padding_side="left",
        part="question",
        suffix=suffix,
    )
    latent_temperature = float(model.model_kwargs.latent_generation_config.get("latent_temperature", 1.0))
    question_embeds = model.embedding(question_input_ids)
    attention_mask = question_attention_mask
    outputs = model.llm.forward(
        inputs_embeds=question_embeds,
        attention_mask=attention_mask,
        position_ids=get_position_ids_from_attention_mask(attention_mask),
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    current_position_ids = get_position_ids_from_attention_mask(attention_mask)[:, -1:]
    latent_inputs = []
    for _ in range(trace_steps):
        distributions = model.latent_policy.forward(
            outputs.hidden_states[-1][:, -1:, :],
            temperature=latent_temperature,
        )
        if use_mean:
            current_inputs_embeds = distributions.mean * model.embeds_std
        else:
            current_inputs_embeds = distributions.rsample() * model.embeds_std
        latent_inputs.append(current_inputs_embeds)
        step_mask = torch.ones(
            (question_input_ids.shape[0], 1),
            device=model.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, step_mask], dim=1)
        current_position_ids = current_position_ids + 1
        outputs = model.llm.forward(
            inputs_embeds=current_inputs_embeds,
            attention_mask=attention_mask,
            position_ids=current_position_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

    latent_inputs = torch.cat(latent_inputs, dim=1)
    inputs_embeds = torch.cat([question_embeds, latent_inputs], dim=1)
    full_attention_mask = torch.cat(
        [
            question_attention_mask,
            torch.ones((len(questions), trace_steps), device=model.device, dtype=question_attention_mask.dtype),
        ],
        dim=1,
    )
    outputs = model.llm.forward(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention_mask,
        position_ids=get_position_ids_from_attention_mask(full_attention_mask),
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states[-1].float()
    question_length = question_input_ids.shape[1]
    h0 = hidden[:, question_length - 1 : question_length, :]
    latent_steps = hidden[:, question_length : question_length + trace_steps, :]
    return torch.cat([h0, latent_steps], dim=1)


@torch.no_grad()
def collect_origin_paths(model, items, trace_steps, speed, use_mean=True):
    questions = [item["question"] for item in items]
    origin_paths = extract_origin_latent_path(
        model=model,
        questions=questions,
        trace_steps=trace_steps,
        speed=str(speed),
        use_mean=use_mean,
    )
    return [to_numpy(path) for path in origin_paths]


def add_origin_to_rows(rows, origin_paths, metrics):
    for row, origin, metric in zip(rows, origin_paths, metrics):
        row["origin"] = origin
        teacher = torch.tensor(row["teacher"])
        origin_tensor = torch.tensor(origin)
        teacher_delta = teacher[1:] - teacher[:-1]
        origin_delta = origin_tensor[1:] - origin_tensor[:-1]
        metric["origin_state_cos"] = float(F.cosine_similarity(origin_tensor[1:], teacher[1:], dim=-1).mean().item())
        metric["origin_transition_cos"] = float(F.cosine_similarity(origin_delta, teacher_delta, dim=-1).mean().item())


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = {k: v for k, v in row.items() if k not in {"teacher", "student", "origin"}}
            payload["teacher"] = row["teacher"].tolist()
            payload["student"] = row["student"].tolist()
            if "origin" in row:
                payload["origin"] = row["origin"].tolist()
            f.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )


def plot_paths(rows, out_dir: Path, dpi: int):
    plt = ensure_matplotlib()
    all_points = []
    for row in rows:
        all_points.append(row["teacher"])
        all_points.append(row["student"])
        if "origin" in row:
            all_points.append(row["origin"])
    stacked = np.vstack(all_points)
    mean, components = fit_pca(stacked, dim=3)
    fig = plt.figure(figsize=(4.3 * len(rows), 4.0), constrained_layout=True)
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    for panel_idx, row in enumerate(rows, start=1):
        ax = fig.add_subplot(1, len(rows), panel_idx, projection="3d")
        teacher = project(row["teacher"], mean, components, dim=3)
        student = project(row["student"], mean, components, dim=3)
        origin = project(row["origin"], mean, components, dim=3) if "origin" in row else None
        color = palette[(panel_idx - 1) % len(palette)]
        ax.plot(teacher[:, 0], teacher[:, 1], teacher[:, 2], color="black", linestyle="--", linewidth=1.4)
        ax.scatter(teacher[:, 0], teacher[:, 1], teacher[:, 2], color="black", s=24, marker="^", label="teacher")
        if origin is not None:
            ax.plot(origin[:, 0], origin[:, 1], origin[:, 2], color="#d95f02", linestyle=":", linewidth=1.4)
            ax.scatter(origin[:, 0], origin[:, 1], origin[:, 2], color="#d95f02", s=22, marker="s", label="CoLaR origin")
        ax.plot(student[:, 0], student[:, 1], student[:, 2], color=color, linewidth=1.6)
        ax.scatter(student[:, 0], student[:, 1], student[:, 2], color=color, s=28, marker="o", label="TRACE")
        for k in range(len(student)):
            ax.text(student[k, 0], student[k, 1], student[k, 2], str(k), fontsize=7)
        ax.set_title(f"q{row['idx']}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.view_init(elev=22, azim=-58)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    path = out_dir / "trace_teacher_student_paths_3d.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(Path(args.dataset_path))
    items = [dataset[i] for i in args.indices]
    model = load_trace_model(args)
    rows, metrics = collect_paths(model, items, args.trace_steps)
    release_model(model)
    del model
    if args.origin_ckpt:
        origin_model = load_origin_model(args)
        origin_paths = collect_origin_paths(
            model=origin_model,
            items=items,
            trace_steps=args.trace_steps,
            speed=args.origin_speed,
            use_mean=not args.origin_sample,
        )
        add_origin_to_rows(rows, origin_paths, metrics)
        release_model(origin_model)
        del origin_model
    write_jsonl(out_dir / "trace_teacher_student_paths.jsonl", rows)
    write_csv(out_dir / "trace_teacher_student_metrics.csv", metrics)
    path = plot_paths(rows, out_dir, args.dpi)
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
