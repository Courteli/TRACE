#!/usr/bin/env python3
"""TRACE latent-trajectory visualizations.

This script builds the visual evidence requested for TRACE v2:

1. same-question rollout maps;
2. trajectory path plots;
3. semantic paraphrase stability plots and metrics;
4. same-question similarity heatmaps;
5. training dynamics from TensorBoard event files.

The plotted representations intentionally use the same latent trajectory object
as TRACE rewards: `latent_inputs_embeds` masked by `latent_attention_mask`.
"""

import argparse
import csv
import gc
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.utils import instantiate_from_config


DEFAULT_TRACE_CKPT = (
    "logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/"
    "20260704-044959_549439_trace_v2_rl_qwen3_c5_g8_gpu7/"
    "checkpoints/epoch2__step12288__monitor0.341.ckpt"
)
DEFAULT_ANSWER_CKPT = (
    "logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/"
    "20260704-112325_182870_trace_answer_only_rl_qwen3_c5_gpu5/"
    "checkpoints/epoch0__step2048__monitor0.261.ckpt"
)
DEFAULT_ORIGIN_CKPT = (
    "logs/colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/"
    "20260525-172642_884973_origin_colar_qwen3_c5_cotinit_rawgsm_lr3e-5_50epoch_gpu5_20260525/"
    "checkpoints/epoch15__step107616__monitor0.246.ckpt"
)
DEFAULT_TRACE_LOGDIR = (
    "logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl/"
    "20260704-044959_549439_trace_v2_rl_qwen3_c5_g8_gpu7"
)
DEFAULT_DATASET = "/home/dingxukai/RoT/data/GSM8k-Aug-NL/gsm8k_test_processed.jsonl"

TRACE_DYNAMICS_TAGS = [
    "train/trace/bonus",
    "train/trace/pos_neg_sim",
    "train/trace/hard_pos_sim",
    "train/trace/pos_pos_sim",
    "train/trace/pos_count",
    "train/trace/neg_count",
]

COLORS = {
    "correct": "#1f77b4",
    "wrong": "#ff7f0e",
    "hard_negative": "#d62728",
    "all_correct": "#9aa0a6",
    "all_wrong": "#6f7378",
    "unknown": "#bdbdbd",
}


@dataclass
class MethodSpec:
    name: str
    ckpt: Path
    kind: str


@dataclass
class QuestionItem:
    idx: int
    source_id: str
    question: str
    answer: str
    steps: str


@dataclass
class RolloutGroup:
    method: str
    question_idx: int
    question: str
    answer: str
    outputs: List[str]
    pred_answers: List[str]
    acc: np.ndarray
    answer_conf: np.ndarray
    latent: np.ndarray
    mask: np.ndarray
    pooled: np.ndarray
    labels: List[str]
    n_latent_forward: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["rollout", "dynamics"],
        choices=["all", "rollout", "paraphrase", "dynamics"],
        help="Which visualization families to generate.",
    )
    parser.add_argument("--out_dir", default="run_outputs/trace/visualizations/latest")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET)
    parser.add_argument("--indices", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--max_questions", type=int, default=0, help="Use the first N items if --indices is empty.")
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--compression_factor", type=int, default=5)
    parser.add_argument("--latent_temperature", type=float, default=1.0)
    parser.add_argument("--max_n_latent_forward", type=int, default=64)
    parser.add_argument("--origin_ckpt", default=DEFAULT_ORIGIN_CKPT)
    parser.add_argument("--answer_ckpt", default=DEFAULT_ANSWER_CKPT)
    parser.add_argument("--trace_ckpt", default=DEFAULT_TRACE_CKPT)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["origin", "answer", "trace"],
        choices=["origin", "answer", "trace"],
        help="Methods to compare for rollout/paraphrase plots.",
    )
    parser.add_argument(
        "--candidate_scan",
        type=int,
        default=0,
        help="Before plotting selected indices, scan the first N questions and prefer mixed TRACE groups.",
    )
    parser.add_argument(
        "--require_mixed",
        action="store_true",
        help="When candidate scanning, fail if not enough mixed TRACE groups are found.",
    )
    parser.add_argument("--path_question_index", type=int, default=-1)
    parser.add_argument("--heatmap_question_index", type=int, default=-1)
    parser.add_argument("--paraphrase_indices", nargs="*", type=int, default=[])
    parser.add_argument("--paraphrase_variants", type=int, default=20)
    parser.add_argument(
        "--paraphrase_file",
        default="",
        help="Optional JSON/JSONL file with {'idx': int, 'variants': [str, ...]} records.",
    )
    parser.add_argument("--trace_logdir", default=DEFAULT_TRACE_LOGDIR)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--plot_dim",
        choices=["2", "3", "both"],
        default="both",
        help="PCA projection dimensionality for scatter/path plots. Default writes SemCOT-style 3D plus 2D debug plots.",
    )
    parser.add_argument("--no_show_text", action="store_true", help="Use compact titles for camera-ready drafts.")
    return parser.parse_args()


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    return plt, sns


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def masked_mean(latent: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = mask.astype(np.float32)
    denom = np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
    pooled = (latent * weights[:, :, None]).sum(axis=1) / denom
    return normalize_rows(pooled.astype(np.float32))


def fit_pca(x: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"PCA expects a 2D array, got {x.shape}")
    n_components = min(n_components, x.shape[0], x.shape[1])
    if n_components < 1:
        raise ValueError("PCA needs at least one component")
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return mean.astype(np.float32), components.astype(np.float32)


def transform_pca(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return ((x - mean) @ components.T).astype(np.float32)


def project_pca(x: np.ndarray, mean: np.ndarray, components: np.ndarray, dim: int) -> np.ndarray:
    projected = transform_pca(x, mean, components)
    if projected.shape[1] < dim:
        pad = np.zeros((projected.shape[0], dim - projected.shape[1]), dtype=projected.dtype)
        projected = np.concatenate([projected, pad], axis=1)
    return projected[:, :dim]


def requested_plot_dims(args) -> List[int]:
    if args.plot_dim == "2":
        return [2]
    if args.plot_dim == "3":
        return [3]
    return [3, 2]


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    y = normalize_rows(x.astype(np.float32))
    return y @ y.T


def mean_pairwise_distance(x: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    dists = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dists.append(float(np.linalg.norm(x[i] - x[j])))
    return float(np.mean(dists)) if dists else 0.0


def silhouette_score_np(x: np.ndarray, labels: Sequence[int]) -> Optional[float]:
    labels = np.asarray(labels)
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2 or len(x) <= len(unique):
        return None
    scores = []
    for i, label in enumerate(labels):
        same = labels == label
        other = labels != label
        if same.sum() <= 1 or not other.any():
            continue
        a = np.linalg.norm(x[i] - x[same], axis=1)
        a = float(a[a > 0].mean()) if np.any(a > 0) else 0.0
        b = math.inf
        for other_label in unique:
            if other_label == label:
                continue
            cluster = x[labels == other_label]
            if len(cluster):
                b = min(b, float(np.linalg.norm(x[i] - cluster, axis=1).mean()))
        if math.isfinite(b) and max(a, b) > 0:
            scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else None


def load_jsonl_dataset(path: Path) -> List[QuestionItem]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(
                QuestionItem(
                    idx=idx,
                    source_id=str(row.get("id", idx)),
                    question=row["question"],
                    answer=str(row["answer"]),
                    steps=str(row.get("cot", "")),
                )
            )
    return items


def select_items(dataset: List[QuestionItem], indices: Sequence[int], max_questions: int) -> List[QuestionItem]:
    if indices:
        return [dataset[i] for i in indices]
    n = max_questions if max_questions > 0 else 3
    return dataset[:n]


def parse_methods(args) -> List[MethodSpec]:
    ckpts = {
        "origin": Path(args.origin_ckpt),
        "answer": Path(args.answer_ckpt),
        "trace": Path(args.trace_ckpt),
    }
    names = {
        "origin": "CoLaR origin",
        "answer": "answer-only RL",
        "trace": "TRACE v2",
    }
    return [MethodSpec(name=names[key], ckpt=ckpts[key], kind=key) for key in args.methods]


def hparams_path_from_ckpt(ckpt: Path) -> Path:
    candidates = [
        ckpt.parent.parent / "hparams.yaml",
        ckpt.parent / "hparams.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find hparams.yaml near checkpoint: {ckpt}")


def load_model_from_checkpoint(spec: MethodSpec, args):
    hparams = OmegaConf.load(hparams_path_from_ckpt(spec.ckpt))
    config = hparams.get("all_config", None)
    if config is None:
        config = hparams
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config.trainer.logger = False
    config.dataloader.batch_size = 1
    config.dataloader.val_batch_size = 1
    config.dataloader.num_workers = 0
    config.dataloader.persistent_workers = False
    config.model.model_kwargs.rl_config.group_size = int(args.group_size)
    config.model.model_kwargs.latent_generation_config.compression_factor = int(args.compression_factor)
    config.model.model_kwargs.latent_generation_config.latent_temperature = float(args.latent_temperature)
    config.model.model_kwargs.latent_generation_config.max_n_latent_forward = int(args.max_n_latent_forward)
    config.model.model_kwargs.answer_generation_config.max_new_tokens = int(args.max_new_tokens)
    if spec.kind == "answer" and "trace_config" in config.model.model_kwargs:
        config.model.model_kwargs.trace_config.enable_trace_reward = False
        config.model.model_kwargs.trace_config.trace_resample_mixed_rollout = False
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    state = torch.load(spec.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[warn] {spec.name}: load_state_dict missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )
    model.to(args.device)
    model.eval()
    return model


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def answer_confidence(experience) -> np.ndarray:
    mask = experience.answer_attention_mask.float()
    conf = (experience.answer_logprobs.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return to_numpy(conf)


def classify_rollouts(acc: np.ndarray, answer_conf: np.ndarray, hard_top_frac: float = 0.5) -> List[str]:
    labels = ["correct" if v > 0.5 else "wrong" for v in acc.tolist()]
    pos_count = sum(label == "correct" for label in labels)
    neg_indices = [i for i, label in enumerate(labels) if label == "wrong"]
    if pos_count == len(labels):
        return ["all_correct"] * len(labels)
    if pos_count == 0:
        return ["all_wrong"] * len(labels)
    if neg_indices:
        n_hard = max(1, int(round(len(neg_indices) * hard_top_frac)))
        ranked = sorted(neg_indices, key=lambda i: float(answer_conf[i]), reverse=True)
        for i in ranked[:n_hard]:
            labels[i] = "hard_negative"
    return labels


@torch.no_grad()
def collect_rollout_group(model, spec: MethodSpec, item: QuestionItem) -> RolloutGroup:
    device = next(model.parameters()).device
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        experience = model.rollout(questions=[item.question], gt_answers=[item.answer])
    outputs = model.tokenizer.batch_decode(experience.answer_input_ids, skip_special_tokens=True)
    pred_answers = [model.extract_answer_from_output(text) for text in outputs]
    acc = to_numpy(experience.accuracies.view(-1))
    latent = to_numpy(experience.latent_inputs_embeds)
    mask = to_numpy(experience.latent_attention_mask)
    pooled = masked_mean(latent, mask)
    conf = answer_confidence(experience)
    labels = classify_rollouts(acc=acc, answer_conf=conf)
    n_latent_forward = to_numpy(experience.n_latent_forward.view(-1))
    return RolloutGroup(
        method=spec.name,
        question_idx=item.idx,
        question=item.question,
        answer=item.answer,
        outputs=outputs,
        pred_answers=pred_answers,
        acc=acc,
        answer_conf=conf,
        latent=latent,
        mask=mask,
        pooled=pooled,
        labels=labels,
        n_latent_forward=n_latent_forward,
    )


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_rollouts(methods: List[MethodSpec], items: List[QuestionItem], args) -> List[RolloutGroup]:
    groups: List[RolloutGroup] = []
    for spec in methods:
        print(f"[info] loading {spec.name}: {spec.ckpt}", flush=True)
        model = load_model_from_checkpoint(spec, args)
        for item in items:
            print(f"[info] rollout {spec.name} q={item.idx}", flush=True)
            groups.append(collect_rollout_group(model, spec, item))
        unload_model(model)
        del model
    return groups


def scan_trace_mixed_items(methods: List[MethodSpec], dataset: List[QuestionItem], args) -> List[int]:
    trace = next((m for m in methods if m.kind == "trace"), None)
    if trace is None or args.candidate_scan <= 0:
        return [item.idx for item in select_items(dataset, args.indices, args.max_questions)]
    model = load_model_from_checkpoint(trace, args)
    selected = []
    fallback = []
    scan_items = dataset[: args.candidate_scan]
    target_n = max(1, args.max_questions or len(args.indices) or 3)
    for item in scan_items:
        group = collect_rollout_group(model, trace, item)
        pos = int((group.acc > 0.5).sum())
        if 0 < pos < args.group_size:
            selected.append(item.idx)
        else:
            fallback.append(item.idx)
        if len(selected) >= target_n:
            break
    unload_model(model)
    del model
    if len(selected) < target_n and args.require_mixed:
        raise RuntimeError(
            f"candidate_scan={args.candidate_scan} found only {len(selected)} mixed TRACE groups, "
            f"but {target_n} were requested. Increase --candidate_scan or lower --max_questions."
        )
    if selected:
        if len(selected) < target_n:
            print(
                f"[warn] candidate_scan found only {len(selected)} mixed TRACE groups; using those only.",
                file=sys.stderr,
            )
        return selected[:target_n]
    print(
        "[warn] candidate_scan found no mixed TRACE groups; falling back to non-mixed questions. "
        "These are not suitable for the main separation figure.",
        file=sys.stderr,
    )
    return fallback[:target_n]


def group_records(groups: Sequence[RolloutGroup]) -> List[dict]:
    records = []
    for group in groups:
        pos_count = int((group.acc > 0.5).sum())
        records.append(
            {
                "method": group.method,
                "question_idx": group.question_idx,
                "question": group.question,
                "answer": group.answer,
                "group_size": int(len(group.acc)),
                "pos_count": pos_count,
                "neg_count": int(len(group.acc) - pos_count),
                "mean_acc": float(group.acc.mean()),
                "mean_n_latent_forward": float(group.n_latent_forward.mean()),
                "labels": group.labels,
                "pred_answers": group.pred_answers,
                "answer_conf": [float(x) for x in group.answer_conf.tolist()],
                "n_latent_forward": [float(x) for x in group.n_latent_forward.tolist()],
            }
        )
    return records


def write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def label_style(label: str) -> Tuple[str, str, float]:
    if label == "correct":
        return COLORS["correct"], "o", 0.95
    if label == "wrong":
        return COLORS["wrong"], "o", 0.85
    if label == "hard_negative":
        return COLORS["hard_negative"], "X", 1.0
    if label == "all_correct":
        return COLORS["all_correct"], "o", 0.7
    if label == "all_wrong":
        return COLORS["all_wrong"], "o", 0.7
    return COLORS["unknown"], "o", 0.8


def short_question(text: str, max_len: int = 68) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= max_len else compact[: max_len - 3] + "..."


def make_projection_subplots(plt, nrows: int, ncols: int, dim: int, figsize: Tuple[float, float]):
    subplot_kw = {"projection": "3d"} if dim == 3 else {}
    return plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
        subplot_kw=subplot_kw,
    )


def set_projection_labels(ax, dim: int):
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if dim == 3:
        ax.set_zlabel("PC3")


def plot_same_question_rollout_map(groups: Sequence[RolloutGroup], out_dir: Path, args):
    plt, _ = ensure_matplotlib()
    all_pooled = np.vstack([g.pooled for g in groups])
    method_names = list(dict.fromkeys(g.method for g in groups))
    question_ids = list(dict.fromkeys(g.question_idx for g in groups))
    nrows, ncols = len(question_ids), len(method_names)
    paths = []
    for dim in requested_plot_dims(args):
        mean, components = fit_pca(all_pooled, dim)
        fig, axes = make_projection_subplots(
            plt,
            nrows=nrows,
            ncols=ncols,
            dim=dim,
            figsize=(4.3 * ncols, 3.7 * nrows),
        )
        for r, qid in enumerate(question_ids):
            for c, method in enumerate(method_names):
                ax = axes[r][c]
                group = next(g for g in groups if g.question_idx == qid and g.method == method)
                xy = project_pca(group.pooled, mean, components, dim)
                for label in ["all_correct", "all_wrong", "correct", "wrong", "hard_negative"]:
                    idxs = [i for i, v in enumerate(group.labels) if v == label]
                    if not idxs:
                        continue
                    color, marker, alpha = label_style(label)
                    if dim == 3:
                        ax.scatter(
                            xy[idxs, 0],
                            xy[idxs, 1],
                            xy[idxs, 2],
                            s=78 if marker == "X" else 58,
                            c=color,
                            marker=marker,
                            alpha=alpha,
                            edgecolors="white",
                            linewidths=0.6,
                            label=label.replace("_", " "),
                        )
                        for i in idxs:
                            ax.text(
                                xy[i, 0],
                                xy[i, 1],
                                xy[i, 2],
                                str(i),
                                fontsize=7,
                                ha="center",
                                va="center",
                                color="black",
                            )
                    else:
                        ax.scatter(
                            xy[idxs, 0],
                            xy[idxs, 1],
                            s=70 if marker == "X" else 52,
                            c=color,
                            marker=marker,
                            alpha=alpha,
                            edgecolors="white",
                            linewidths=0.6,
                            label=label.replace("_", " "),
                        )
                        for i in idxs:
                            ax.text(
                                xy[i, 0],
                                xy[i, 1],
                                str(i),
                                fontsize=7,
                                ha="center",
                                va="center",
                                color="white",
                            )
                pos = int((group.acc > 0.5).sum())
                title = f"{method}\nq{qid}: {pos}/{len(group.acc)} correct"
                if not args.no_show_text:
                    title += f"\n{short_question(group.question, 54)}"
                ax.set_title(title, fontsize=9)
                if dim == 2:
                    ax.axhline(0, color="#dddddd", linewidth=0.7)
                    ax.axvline(0, color="#dddddd", linewidth=0.7)
                set_projection_labels(ax, dim)
                if dim == 3:
                    ax.view_init(elev=22, azim=-58)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=min(5, len(labels)),
                frameon=False,
            )
        path = out_dir / f"same_question_rollout_map_{dim}d.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_trajectory_paths(groups: Sequence[RolloutGroup], out_dir: Path, args):
    plt, _ = ensure_matplotlib()
    method_names = list(dict.fromkeys(g.method for g in groups))
    qid = args.path_question_index if args.path_question_index >= 0 else groups[0].question_idx
    selected = [g for g in groups if g.question_idx == qid]
    if not selected:
        selected = [groups[0]]
        qid = selected[0].question_idx
    all_steps = []
    for group in selected:
        for rollout_idx in range(len(group.acc)):
            length = int(group.mask[rollout_idx].sum())
            if length > 0:
                all_steps.append(group.latent[rollout_idx, :length])
    stacked = np.vstack(all_steps)
    paths = []
    for dim in requested_plot_dims(args):
        mean, components = fit_pca(stacked, dim)
        fig, axes = make_projection_subplots(
            plt,
            nrows=1,
            ncols=len(method_names),
            dim=dim,
            figsize=(4.5 * len(method_names), 4.0),
        )
        for c, method in enumerate(method_names):
            ax = axes[0][c]
            group = next((g for g in selected if g.method == method), None)
            if group is None:
                ax.axis("off")
                continue
            for rollout_idx in range(len(group.acc)):
                length = int(group.mask[rollout_idx].sum())
                if length <= 0:
                    continue
                xy = project_pca(group.latent[rollout_idx, :length], mean, components, dim)
                label = group.labels[rollout_idx]
                color, marker, alpha = label_style(label)
                if dim == 3:
                    ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], color=color, alpha=0.42, linewidth=1.2)
                    ax.scatter(xy[0, 0], xy[0, 1], xy[0, 2], s=18, c=color, alpha=0.5, marker=".")
                    ax.scatter(
                        xy[-1, 0],
                        xy[-1, 1],
                        xy[-1, 2],
                        s=80 if marker == "X" else 58,
                        c=color,
                        marker=marker,
                        edgecolors="white",
                        linewidths=0.6,
                        alpha=alpha,
                    )
                    ax.text(
                        xy[-1, 0],
                        xy[-1, 1],
                        xy[-1, 2],
                        str(rollout_idx),
                        fontsize=7,
                        color="black",
                        ha="center",
                        va="center",
                    )
                else:
                    ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.42, linewidth=1.2)
                    ax.scatter(xy[0, 0], xy[0, 1], s=18, c=color, alpha=0.5, marker=".")
                    ax.scatter(
                        xy[-1, 0],
                        xy[-1, 1],
                        s=72 if marker == "X" else 52,
                        c=color,
                        marker=marker,
                        edgecolors="white",
                        linewidths=0.6,
                        alpha=alpha,
                    )
                    ax.text(
                        xy[-1, 0],
                        xy[-1, 1],
                        str(rollout_idx),
                        fontsize=7,
                        color="white",
                        ha="center",
                        va="center",
                    )
            pos = int((group.acc > 0.5).sum())
            ax.set_title(f"{method}\nq{qid}: {pos}/{len(group.acc)} correct", fontsize=9)
            set_projection_labels(ax, dim)
            if dim == 3:
                ax.view_init(elev=22, azim=-58)
        path = out_dir / f"trajectory_path_plot_q{qid}_{dim}d.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def rollout_geometry_metrics(groups: Sequence[RolloutGroup]) -> List[dict]:
    rows = []
    for group in groups:
        sim = cosine_matrix(group.pooled)
        pos = group.acc > 0.5
        hard = np.array([label == "hard_negative" for label in group.labels])
        neg = ~pos
        pos_neg = float(sim[pos][:, neg].mean()) if pos.any() and neg.any() else None
        pos_pos = None
        if pos.sum() > 1:
            block = sim[pos][:, pos]
            off = ~np.eye(int(pos.sum()), dtype=bool)
            pos_pos = float(block[off].mean())
        hard_pos = float(sim[hard][:, pos].mean()) if hard.any() and pos.any() else None
        rows.append(
            {
                "method": group.method,
                "question_idx": group.question_idx,
                "pos_count": int(pos.sum()),
                "neg_count": int(neg.sum()),
                "hard_count": int(hard.sum()),
                "mean_acc": float(group.acc.mean()),
                "mean_n_latent_forward": float(group.n_latent_forward.mean()),
                "pos_neg_sim": pos_neg,
                "pos_pos_sim": pos_pos,
                "hard_pos_sim": hard_pos,
                "pos_neg_margin": None if pos_neg is None else 1.0 - pos_neg,
                "hard_neg_margin": None if hard_pos is None else 1.0 - hard_pos,
            }
        )
    return rows


def plot_similarity_heatmaps(groups: Sequence[RolloutGroup], out_dir: Path, args):
    plt, sns = ensure_matplotlib()
    qid = args.heatmap_question_index if args.heatmap_question_index >= 0 else groups[0].question_idx
    selected = [g for g in groups if g.question_idx == qid]
    if not selected:
        selected = [groups[0]]
        qid = selected[0].question_idx
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(selected),
        figsize=(3.7 * len(selected), 3.4),
        squeeze=False,
        constrained_layout=True,
    )
    order_rank = {"correct": 0, "all_correct": 0, "wrong": 1, "all_wrong": 1, "hard_negative": 2}
    for c, group in enumerate(selected):
        order = sorted(range(len(group.labels)), key=lambda i: (order_rank.get(group.labels[i], 9), i))
        sim = cosine_matrix(group.pooled)[np.ix_(order, order)]
        tick_labels = [f"{i}:{group.labels[i].replace('_negative', '')}" for i in order]
        ax = axes[0][c]
        sns.heatmap(
            sim,
            ax=ax,
            vmin=-1,
            vmax=1,
            cmap="vlag",
            square=True,
            cbar=c == len(selected) - 1,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            linewidths=0.3,
            linecolor="#eeeeee",
        )
        ax.set_title(group.method, fontsize=9)
        ax.tick_params(axis="x", labelrotation=70, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
    path = out_dir / f"similarity_heatmap_q{qid}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def paraphrase_templates(question: str, n: int) -> List[str]:
    templates = [
        "{q}",
        "Please solve this problem: {q}",
        "Find the answer to the following question. {q}",
        "Work out this math problem: {q}",
        "Can you determine the final answer? {q}",
        "Solve carefully: {q}",
        "What is the result for this problem? {q}",
        "Use the information below to compute the answer. {q}",
        "Here is a math word problem: {q}",
        "Please calculate the answer. {q}",
        "Determine the unknown value. {q}",
        "Answer this question: {q}",
        "Compute the final number. {q}",
        "Read the problem and solve it. {q}",
        "Figure out the answer. {q}",
        "Give the numerical answer for: {q}",
        "Solve the following step by step internally. {q}",
        "Use arithmetic reasoning to answer. {q}",
        "What number satisfies the question? {q}",
        "Please provide the final result. {q}",
        "Find the final value described here. {q}",
        "Calculate it: {q}",
        "Resolve this word problem. {q}",
        "What is the correct answer? {q}",
    ]
    return [templates[i % len(templates)].format(q=question) for i in range(n)]


def load_paraphrase_file(path: Path) -> Dict[int, List[str]]:
    if not path:
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    records = []
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = {}
    for record in records:
        out[int(record["idx"])] = list(record["variants"])
    return out


def centroid_for_variant(group: RolloutGroup) -> Tuple[np.ndarray, dict]:
    pos = group.acc > 0.5
    hard = np.array([label == "hard_negative" for label in group.labels])
    if pos.any():
        centroid = normalize_rows(group.pooled[pos].mean(axis=0, keepdims=True))[0]
    else:
        best = int(np.argmax(group.answer_conf))
        centroid = group.pooled[best]
    sim = cosine_matrix(group.pooled)
    pos_neg_sim = float(sim[pos][:, ~pos].mean()) if pos.any() and (~pos).any() else None
    hard_pos_sim = float(sim[hard][:, pos].mean()) if hard.any() and pos.any() else None
    return centroid, {
        "has_correct": bool(pos.any()),
        "pos_count": int(pos.sum()),
        "neg_count": int((~pos).sum()),
        "hard_count": int(hard.sum()),
        "pos_neg_margin": None if pos_neg_sim is None else 1.0 - pos_neg_sim,
        "hard_neg_margin": None if hard_pos_sim is None else 1.0 - hard_pos_sim,
    }


@torch.no_grad()
def collect_paraphrase_groups(
    methods: Sequence[MethodSpec],
    items: Sequence[QuestionItem],
    paraphrase_map: Dict[int, List[str]],
    args,
) -> Tuple[List[dict], List[dict]]:
    centroid_rows = []
    metric_rows = []
    for spec in methods:
        print(f"[info] loading {spec.name} for paraphrase stability", flush=True)
        model = load_model_from_checkpoint(spec, args)
        for item in items:
            variants = paraphrase_map.get(item.idx) or paraphrase_templates(item.question, args.paraphrase_variants)
            variants = variants[: args.paraphrase_variants]
            for variant_idx, variant in enumerate(variants):
                q_item = QuestionItem(
                    idx=item.idx,
                    source_id=f"{item.source_id}:para{variant_idx}",
                    question=variant,
                    answer=item.answer,
                    steps=item.steps,
                )
                group = collect_rollout_group(model, spec, q_item)
                centroid, metrics = centroid_for_variant(group)
                centroid_rows.append(
                    {
                        "method": spec.name,
                        "question_idx": item.idx,
                        "variant_idx": variant_idx,
                        "question": variant,
                        "centroid": centroid,
                    }
                )
                metric_rows.append(
                    {
                        "method": spec.name,
                        "question_idx": item.idx,
                        "variant_idx": variant_idx,
                        **metrics,
                    }
                )
        unload_model(model)
        del model
    return centroid_rows, metric_rows


def summarize_paraphrase_metrics(centroid_rows: List[dict], metric_rows: List[dict]) -> List[dict]:
    out = []
    methods = sorted(set(row["method"] for row in centroid_rows))
    for method in methods:
        rows = [row for row in centroid_rows if row["method"] == method]
        qids = sorted(set(row["question_idx"] for row in rows))
        if not rows:
            continue
        x = np.vstack([row["centroid"] for row in rows])
        labels = [qids.index(row["question_idx"]) for row in rows]
        intra_vals = []
        centroids = []
        for qid in qids:
            qx = np.vstack([row["centroid"] for row in rows if row["question_idx"] == qid])
            qcent = qx.mean(axis=0)
            centroids.append(qcent)
            intra_vals.append(float(np.linalg.norm(qx - qcent[None, :], axis=1).mean()))
        metric_subset = [row for row in metric_rows if row["method"] == method]
        pos_neg_margin = [row["pos_neg_margin"] for row in metric_subset if row["pos_neg_margin"] is not None]
        hard_margin = [row["hard_neg_margin"] for row in metric_subset if row["hard_neg_margin"] is not None]
        out.append(
            {
                "method": method,
                "n_questions": len(qids),
                "n_variants": len(rows),
                "intra_paraphrase_variance": float(np.mean(intra_vals)) if intra_vals else None,
                "inter_question_distance": mean_pairwise_distance(np.vstack(centroids)) if len(centroids) > 1 else None,
                "silhouette_score": silhouette_score_np(x, labels),
                "pos_neg_margin": float(np.mean(pos_neg_margin)) if pos_neg_margin else None,
                "hard_neg_margin": float(np.mean(hard_margin)) if hard_margin else None,
                "correct_variant_frac": float(np.mean([row["has_correct"] for row in metric_subset]))
                if metric_subset
                else None,
            }
        )
    return out


def plot_paraphrase_stability(centroid_rows: List[dict], out_dir: Path, args):
    plt, _ = ensure_matplotlib()
    x = np.vstack([row["centroid"] for row in centroid_rows])
    method_names = list(dict.fromkeys(row["method"] for row in centroid_rows))
    qids = sorted(set(row["question_idx"] for row in centroid_rows))
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    paths = []
    for dim in requested_plot_dims(args):
        mean, components = fit_pca(x, dim)
        xy = project_pca(x, mean, components, dim)
        fig, axes = make_projection_subplots(
            plt,
            nrows=1,
            ncols=len(method_names),
            dim=dim,
            figsize=(4.3 * len(method_names), 3.9),
        )
        offset = 0
        for c, method in enumerate(method_names):
            ax = axes[0][c]
            method_rows = [row for row in centroid_rows if row["method"] == method]
            count = len(method_rows)
            method_xy = xy[offset : offset + count]
            offset += count
            for qi, qid in enumerate(qids):
                idxs = [i for i, row in enumerate(method_rows) if row["question_idx"] == qid]
                if not idxs:
                    continue
                color = palette[qi % len(palette)]
                qxy = method_xy[idxs]
                center = qxy.mean(axis=0)
                if dim == 3:
                    ax.scatter(
                        qxy[:, 0],
                        qxy[:, 1],
                        qxy[:, 2],
                        c=color,
                        s=32,
                        alpha=0.78,
                        edgecolors="white",
                        linewidths=0.4,
                        label=f"q{qid}",
                    )
                    ax.scatter(
                        center[0],
                        center[1],
                        center[2],
                        c=color,
                        s=125,
                        marker="*",
                        edgecolors="black",
                        linewidths=0.4,
                    )
                else:
                    ax.scatter(
                        qxy[:, 0],
                        qxy[:, 1],
                        c=color,
                        s=30,
                        alpha=0.78,
                        edgecolors="white",
                        linewidths=0.4,
                        label=f"q{qid}",
                    )
                    ax.scatter(
                        center[0],
                        center[1],
                        c=color,
                        s=110,
                        marker="*",
                        edgecolors="black",
                        linewidths=0.4,
                    )
            ax.set_title(method, fontsize=10)
            set_projection_labels(ax, dim)
            if dim == 3:
                ax.view_init(elev=22, azim=-58)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=min(len(labels), 5),
                frameon=False,
            )
        path = out_dir / f"semantic_paraphrase_stability_{dim}d.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def read_tensorboard_scalars(logdir: Path, tags: Sequence[str]) -> List[dict]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows = []
    event_files = sorted(logdir.glob("events.out.tfevents.*"))
    for event_file in event_files:
        try:
            acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            acc.Reload()
        except Exception as exc:
            print(f"[warn] could not read {event_file}: {exc}", file=sys.stderr)
            continue
        available = set(acc.Tags().get("scalars", []))
        for tag in tags:
            if tag not in available:
                continue
            for event in acc.Scalars(tag):
                rows.append(
                    {
                        "tag": tag,
                        "step": int(event.step),
                        "wall_time": float(event.wall_time),
                        "value": float(event.value),
                        "event_file": str(event_file),
                    }
                )
    rows.sort(key=lambda row: (row["tag"], row["step"], row["wall_time"]))
    deduped = []
    seen = set()
    for row in rows:
        key = (row["tag"], row["step"], row["value"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def plot_training_dynamics(logdir: Path, out_dir: Path, args):
    plt, _ = ensure_matplotlib()
    rows = read_tensorboard_scalars(logdir, TRACE_DYNAMICS_TAGS)
    write_csv(out_dir / "training_dynamics.csv", rows)
    if not rows:
        print(f"[warn] no TRACE scalar rows found under {logdir}", file=sys.stderr)
        return None
    tags = [tag for tag in TRACE_DYNAMICS_TAGS if any(row["tag"] == tag for row in rows)]
    nrows = len(tags)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(7.0, max(2.0, 1.35 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    for r, tag in enumerate(tags):
        ax = axes[r][0]
        tag_rows = [row for row in rows if row["tag"] == tag]
        steps = np.array([row["step"] for row in tag_rows], dtype=np.float32)
        values = np.array([row["value"] for row in tag_rows], dtype=np.float32)
        order = np.argsort(steps)
        steps, values = steps[order], values[order]
        ax.plot(steps, values, linewidth=1.3, color="#1f77b4")
        ax.set_ylabel(tag.replace("train/trace/", ""), fontsize=8)
        ax.grid(True, color="#eeeeee", linewidth=0.7)
    axes[-1][0].set_xlabel("training step")
    path = out_dir / "training_dynamics.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def run_rollout_plots(args, out_dir: Path, methods: List[MethodSpec], dataset: List[QuestionItem]):
    if args.candidate_scan > 0:
        indices = scan_trace_mixed_items(methods, dataset, args)
        items = [dataset[i] for i in indices]
        print(f"[info] candidate scan selected indices: {indices}", flush=True)
    else:
        items = select_items(dataset, args.indices, args.max_questions)
    groups = collect_rollouts(methods, items, args)
    write_jsonl(out_dir / "rollout_records.jsonl", group_records(groups))
    write_csv(out_dir / "rollout_geometry_metrics.csv", rollout_geometry_metrics(groups))
    paths = []
    paths.extend(plot_same_question_rollout_map(groups, out_dir, args))
    paths.extend(plot_trajectory_paths(groups, out_dir, args))
    paths.append(plot_similarity_heatmaps(groups, out_dir, args))
    return groups, paths


def run_paraphrase_plots(args, out_dir: Path, methods: List[MethodSpec], dataset: List[QuestionItem]):
    indices = args.paraphrase_indices or args.indices or [0, 1, 2]
    items = [dataset[i] for i in indices]
    paraphrase_map = load_paraphrase_file(Path(args.paraphrase_file)) if args.paraphrase_file else {}
    centroid_rows, metric_rows = collect_paraphrase_groups(methods, items, paraphrase_map, args)
    json_rows = [
        {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in row.items()}
        for row in centroid_rows
    ]
    write_jsonl(out_dir / "paraphrase_centroids.jsonl", json_rows)
    write_csv(out_dir / "paraphrase_variant_metrics.csv", metric_rows)
    summary_rows = summarize_paraphrase_metrics(centroid_rows, metric_rows)
    write_csv(out_dir / "paraphrase_summary_metrics.csv", summary_rows)
    paths = plot_paraphrase_stability(centroid_rows, out_dir, args)
    return centroid_rows, metric_rows, summary_rows, paths


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    requested = set(args.plots)
    if "all" in requested:
        requested = {"rollout", "paraphrase", "dynamics"}
    methods = parse_methods(args)
    dataset = None
    if requested & {"rollout", "paraphrase"}:
        dataset = load_jsonl_dataset(Path(args.dataset_path))
    output_paths = []
    if "rollout" in requested:
        _, paths = run_rollout_plots(args, out_dir, methods, dataset)
        output_paths.extend(paths)
    if "paraphrase" in requested:
        _, _, _, paths = run_paraphrase_plots(args, out_dir, methods, dataset)
        output_paths.extend(paths)
    if "dynamics" in requested:
        path = plot_training_dynamics(Path(args.trace_logdir), out_dir, args)
        if path is not None:
            output_paths.append(path)
    print("[done] wrote:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
