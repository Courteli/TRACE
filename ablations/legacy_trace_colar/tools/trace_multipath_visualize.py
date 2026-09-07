#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.utils import instantiate_from_config


DEFAULT_DATASET = "/home/dingxukai/RoT/data/GSM8k-Aug-NL/gsm8k_test_processed.jsonl"

DEFAULT_TRACE_VIS_CFG = {
    "signature_mean_weight": 0.5,
    "signature_last_weight": 0.5,
    "signature_trend_weight": 1.0,
    "signature_delta_weight": 1.0,
    "center_path_signature": True,
    "path_signature_raw_mix": 0.25,
    "max_modes": 3,
    "min_positive_modes": 2,
    "target_positive_modes": 3,
    "mode_merge_threshold": 0.65,
    "hard_threshold": 0.25,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET)
    parser.add_argument("--indices", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--auto_select", action="store_true")
    parser.add_argument("--candidate_count", type=int, default=24)
    parser.add_argument("--num_questions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_l", type=int, default=40)
    parser.add_argument("--min_l", type=int, default=0)
    parser.add_argument("--latent_temperature", type=float, default=None)
    parser.add_argument("--eol_temperature", type=float, default=None)
    parser.add_argument("--compression_factor", type=int, default=None)
    parser.add_argument("--lp_deterministic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="run_outputs/trace_multipath/visualizations/latest")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--trajectory_space", choices=["residual", "raw", "delta"], default="residual")
    parser.add_argument("--pca_scope", choices=["per_group", "global"], default="per_group")
    return parser.parse_args()


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    return plt, sns


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
                    "steps": item.get("cot", ""),
                    "answer": str(item["answer"]),
                }
            )
    return rows


def load_model(args):
    ckpt = Path(args.ckpt)
    hparams = OmegaConf.load(ckpt.parent.parent / "hparams.yaml")
    config = hparams.get("all_config", hparams)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config.trainer.logger = False
    config.dataloader.batch_size = 1
    config.dataloader.val_batch_size = 1
    config.dataloader.num_workers = 0
    config.dataloader.persistent_workers = False
    config.model.model_kwargs.latent_generation_config.max_n_latent_forward = args.max_l
    config.model.model_kwargs.latent_generation_config.min_n_latent_forward = args.min_l
    if args.latent_temperature is not None:
        config.model.model_kwargs.latent_generation_config.latent_temperature = args.latent_temperature
    if args.eol_temperature is not None:
        config.model.model_kwargs.latent_generation_config.eol_temperature = args.eol_temperature
    if args.compression_factor is not None:
        config.model.model_kwargs.latent_generation_config.compression_factor = args.compression_factor
    if getattr(args, "lp_deterministic", False):
        config.model.model_kwargs.latent_policy_config.lp_determinisitc = True
    config.model.model_kwargs.rl_config.group_size = args.group_size
    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    print(model.load_state_dict(state, strict=False))
    model.to(args.device)
    model.eval()
    return model


def fit_pca(x: np.ndarray, dim: int = 3):
    mean = x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        components = np.eye(x.shape[1], dtype=x.dtype)[:dim]
        return mean, components
    centered = x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(dim, vt.shape[0])]
    return mean, components


def project(x: np.ndarray, mean: np.ndarray, components: np.ndarray, dim: int = 3):
    y = (x - mean) @ components.T
    if y.shape[1] < dim:
        y = np.concatenate([y, np.zeros((y.shape[0], dim - y.shape[1]), dtype=y.dtype)], axis=1)
    return y[:, :dim]


def masked_path_mean(latents, mask):
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (latents * mask.unsqueeze(-1)).sum(dim=1) / denom
    return F.normalize(pooled.float(), dim=-1)


def cfg_float(cfg, key, default):
    return float(cfg.get(key, default)) if cfg is not None else float(default)


def cfg_bool(cfg, key, default):
    if cfg is None:
        return bool(default)
    value = cfg.get(key, default)
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def cfg_dict(cfg):
    merged = dict(DEFAULT_TRACE_VIS_CFG)
    if cfg is None:
        return merged
    if hasattr(cfg, "items"):
        merged.update(dict(cfg.items()))
    return merged


def get_trace_cfg(model):
    return cfg_dict(getattr(model, "trace_config", None))


def generic_positive_modes(pos_paths, pos_conf, cfg):
    max_modes = max(1, int(cfg.get("max_modes", 3)))
    merge_threshold = float(cfg.get("mode_merge_threshold", 0.65))
    n_pos = pos_paths.shape[0]
    min_modes = min(max_modes, n_pos, max(1, int(cfg.get("min_positive_modes", 1))))
    target_modes = min(max_modes, n_pos, max(min_modes, int(cfg.get("target_positive_modes", max_modes))))
    order = torch.argsort(pos_conf, descending=True)
    prototypes = []
    seed_indices = []
    assignments = torch.zeros(n_pos, device=pos_paths.device, dtype=torch.long)

    for local_idx in order.tolist():
        prototypes.append(pos_paths[local_idx])
        seed_indices.append(local_idx)
        break

    selected = set(seed_indices)
    while len(prototypes) < min_modes:
        proto_tensor = torch.stack(prototypes, dim=0)
        nearest = (pos_paths @ proto_tensor.T).max(dim=1).values
        if selected:
            selected_tensor = torch.tensor(list(selected), device=pos_paths.device, dtype=torch.long)
            nearest[selected_tensor] = 2.0
        local_idx = int(torch.argmin(nearest).item())
        prototypes.append(pos_paths[local_idx])
        seed_indices.append(local_idx)
        selected.add(local_idx)

    for local_idx in order.tolist():
        if local_idx in selected:
            continue
        if len(prototypes) >= target_modes:
            break
        candidate = pos_paths[local_idx]
        proto_tensor = torch.stack(prototypes, dim=0)
        max_sim = (candidate.unsqueeze(0) @ proto_tensor.T).max()
        if max_sim < merge_threshold and len(prototypes) < max_modes:
            prototypes.append(candidate)
            seed_indices.append(local_idx)
            selected.add(local_idx)

    proto_tensor = F.normalize(torch.stack(prototypes, dim=0), dim=-1)
    for _ in range(2):
        sims = pos_paths @ proto_tensor.T
        assignments = sims.argmax(dim=1)
        new_protos = []
        for mode_idx in range(proto_tensor.shape[0]):
            members = pos_paths[assignments == mode_idx]
            if members.numel() == 0:
                new_protos.append(proto_tensor[mode_idx])
            else:
                new_protos.append(F.normalize(members.mean(dim=0), dim=-1))
        proto_tensor = torch.stack(new_protos, dim=0)

    counts = torch.stack([(assignments == i).sum() for i in range(proto_tensor.shape[0])]).float()
    return proto_tensor, assignments, counts


def positive_modes(model, pos_paths, pos_conf, cfg):
    if hasattr(model, "build_positive_modes"):
        return model.build_positive_modes(pos_paths, pos_conf)
    return generic_positive_modes(pos_paths, pos_conf, cfg)


def mode_entropy(model, counts):
    if hasattr(model, "mode_entropy"):
        return model.mode_entropy(counts)
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
    return entropy, torch.exp(entropy)


def path_signature(latents, mask, cfg):
    latents = latents.float()
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_state = (latents * mask.unsqueeze(-1)).sum(dim=1) / denom
    lengths = mask.sum(dim=1).long().clamp_min(1)
    gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, latents.shape[-1])
    first_state = latents[:, 0, :]
    last_state = latents.gather(dim=1, index=gather_idx).squeeze(1)
    trend_state = last_state - first_state
    if latents.shape[1] <= 1:
        delta_state = torch.zeros_like(mean_state)
    else:
        deltas = latents[:, 1:] - latents[:, :-1]
        delta_mask = mask[:, 1:] * mask[:, :-1]
        delta_state = (deltas * delta_mask.unsqueeze(-1)).sum(dim=1) / delta_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    parts = [
        cfg_float(cfg, "signature_mean_weight", 0.5) * F.normalize(mean_state, dim=-1),
        cfg_float(cfg, "signature_last_weight", 0.5) * F.normalize(last_state, dim=-1),
        cfg_float(cfg, "signature_trend_weight", 1.0) * F.normalize(trend_state, dim=-1),
        cfg_float(cfg, "signature_delta_weight", 1.0) * F.normalize(delta_state, dim=-1),
    ]
    return F.normalize(torch.cat(parts, dim=-1), dim=-1)


def center_group_paths(path, cfg):
    path = F.normalize(path.float(), dim=-1)
    if not cfg_bool(cfg, "center_path_signature", True):
        return path
    centered = F.normalize(path - path.mean(dim=0, keepdim=True), dim=-1)
    raw_mix = cfg_float(cfg, "path_signature_raw_mix", 0.25)
    if raw_mix <= 0:
        return centered
    return F.normalize(torch.cat([centered, raw_mix * path], dim=-1), dim=-1)


def delta_metrics(latents, mask, embeds_std):
    if latents.shape[1] <= 1:
        zeros = torch.zeros(latents.shape[0], device=latents.device)
        return zeros, zeros
    deltas = latents[:, 1:] - latents[:, :-1]
    delta_mask = mask[:, 1:] * mask[:, :-1]
    delta_norm = deltas.norm(dim=-1) / (np.sqrt(latents.shape[-1]) * float(embeds_std))
    mean_norm = (delta_norm * delta_mask).sum(dim=1) / delta_mask.sum(dim=1).clamp_min(1.0)
    if deltas.shape[1] <= 1:
        return torch.zeros_like(mean_norm), mean_norm
    dirs = F.normalize(deltas, dim=-1)
    pair_mask = delta_mask[:, 1:] * delta_mask[:, :-1]
    step_cos = F.cosine_similarity(dirs[:, 1:], dirs[:, :-1], dim=-1)
    coherence = (step_cos * pair_mask).sum(dim=1) / pair_mask.sum(dim=1).clamp_min(1.0)
    return coherence, mean_norm


@torch.no_grad()
def collect_group(model, item, group_size):
    questions = [item["question"]] * group_size
    device = getattr(model, "device", torch.device("cpu"))
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        question_ids, question_mask, latent_embeds, latent_mask, pred_ids = model.latent_generate(
            questions=questions,
            rl_mode=True,
        )
    pred_strings = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    n_latent = latent_mask.sum(dim=1)
    _, accuracies = model.get_group_rewards_and_acc(
        pred_answers=pred_strings,
        gt_answer=item["answer"],
        n_latent_forward=n_latent,
    )
    trace_cfg = get_trace_cfg(model)
    raw_path = path_signature(latent_embeds.float(), latent_mask.float(), trace_cfg)
    path = center_group_paths(raw_path, trace_cfg)
    step_coherence, delta_norm = delta_metrics(latent_embeds.float(), latent_mask.float(), model.embeds_std)

    pos_mask = accuracies.view(-1).float() > 0.5
    mode_assign = torch.full((group_size,), -1, device=path.device, dtype=torch.long)
    hard_mask = torch.zeros((group_size,), device=path.device, dtype=torch.bool)
    prototypes = torch.empty(0, path.shape[-1], device=path.device)
    nearest_proto_sim = torch.zeros((group_size,), device=path.device)
    if pos_mask.any():
        pos_paths = path[pos_mask]
        pos_conf = torch.zeros(pos_paths.shape[0], device=path.device)
        prototypes, assignments, counts = positive_modes(model, pos_paths, pos_conf, trace_cfg)
        pos_indices = pos_mask.nonzero(as_tuple=False).view(-1)
        mode_assign[pos_indices] = assignments
        sims = path @ prototypes.T
        nearest_proto_sim = sims.max(dim=1).values
        hard_threshold = float(trace_cfg.get("hard_threshold", 0.25))
        hard_mask = (~pos_mask) & (nearest_proto_sim > hard_threshold)
        entropy, effective_modes = mode_entropy(model, counts)
    else:
        counts = torch.empty(0, device=path.device)
        entropy = torch.tensor(0.0, device=path.device)
        effective_modes = torch.tensor(0.0, device=path.device)

    sim = path @ path.T
    if pos_mask.any():
        pos_proto_mean = float(nearest_proto_sim[pos_mask].mean().item())
    else:
        pos_proto_mean = 0.0
    if (~pos_mask).any() and pos_mask.any():
        neg_proto_mean = float(nearest_proto_sim[~pos_mask].mean().item())
        pos_neg_margin = pos_proto_mean - neg_proto_mean
    else:
        neg_proto_mean = 0.0
        pos_neg_margin = 0.0
    if int(pos_mask.sum().item()) > 1:
        pos_paths = path[pos_mask]
        pair_sim = pos_paths @ pos_paths.T
        off_diag = ~torch.eye(pos_paths.shape[0], dtype=torch.bool, device=path.device)
        pos_pair_sim = float(pair_sim[off_diag].mean().item())
    else:
        pos_pair_sim = 0.0
    metrics = {
        "idx": item["idx"],
        "acc": float(accuracies.float().mean().item()),
        "pos_count": int(pos_mask.sum().item()),
        "neg_count": int((~pos_mask).sum().item()),
        "hard_count": int(hard_mask.sum().item()),
        "mode_count": int(prototypes.shape[0]),
        "effective_modes": float(effective_modes.item()),
        "mode_entropy": float(entropy.item()),
        "mean_n_latent_forward": float(n_latent.float().mean().item()),
        "mean_step_coherence": float(step_coherence.mean().item()),
        "mean_delta_norm": float(delta_norm.mean().item()),
        "mean_nearest_proto_sim": float(nearest_proto_sim.mean().item()),
        "pos_proto_sim": pos_proto_mean,
        "neg_proto_sim": neg_proto_mean,
        "pos_neg_margin": pos_neg_margin,
        "pos_pair_sim": pos_pair_sim,
    }
    return {
        "item": item,
        "latent_embeds": latent_embeds.detach().float().cpu().numpy(),
        "latent_mask": latent_mask.detach().cpu().numpy(),
        "path": path.detach().float().cpu().numpy(),
        "similarity": sim.detach().float().cpu().numpy(),
        "accuracies": accuracies.view(-1).detach().cpu().numpy(),
        "hard_mask": hard_mask.detach().cpu().numpy(),
        "mode_assign": mode_assign.detach().cpu().numpy(),
        "pred_strings": pred_strings,
        "metrics": metrics,
    }


def sorted_indices(group):
    acc = group["accuracies"].reshape(-1)
    hard = group["hard_mask"].reshape(-1)
    modes = group["mode_assign"].reshape(-1)
    order = list(range(len(acc)))
    return sorted(order, key=lambda i: (0 if acc[i] > 0.5 else 2 if hard[i] else 1, int(modes[i]) if modes[i] >= 0 else 99, i))


def transform_group_latents(latents: np.ndarray, mask: np.ndarray, trajectory_space: str):
    out = latents.copy()
    if trajectory_space == "raw":
        return out
    if trajectory_space == "residual":
        for t in range(out.shape[1]):
            valid = mask[:, t].astype(bool)
            if valid.any():
                out[valid, t] = out[valid, t] - out[valid, t].mean(axis=0, keepdims=True)
        return out
    if trajectory_space == "delta":
        delta_out = np.zeros_like(out)
        for i in range(out.shape[0]):
            valid_idx = np.flatnonzero(mask[i].astype(bool))
            if len(valid_idx) <= 1:
                continue
            points = out[i, valid_idx]
            deltas = points[1:] - points[:-1]
            denom = np.linalg.norm(deltas, axis=1, keepdims=True)
            deltas = deltas / np.clip(denom, 1e-8, None)
            delta_out[i, valid_idx[1:]] = np.cumsum(deltas, axis=0)
        return delta_out
    raise ValueError(f"Unknown trajectory_space={trajectory_space}")


def valid_points_from_groups(groups, trajectory_space: str):
    all_points = []
    transformed = []
    for group in groups:
        latents = transform_group_latents(group["latent_embeds"], group["latent_mask"], trajectory_space)
        transformed.append(latents)
        mask = group["latent_mask"]
        for i in range(latents.shape[0]):
            valid = mask[i].astype(bool)
            if valid.any():
                all_points.append(latents[i, valid])
    return transformed, all_points


def plot_3d(groups, out_dir: Path, dpi: int, trajectory_space: str, pca_scope: str):
    plt, _ = ensure_matplotlib()
    transformed_groups, all_points = valid_points_from_groups(groups, trajectory_space)
    if pca_scope == "global":
        stacked = np.vstack(all_points)
        global_mean, global_components = fit_pca(stacked, dim=3)
    else:
        global_mean, global_components = None, None

    fig = plt.figure(figsize=(4.4 * len(groups), 4.2), constrained_layout=True)
    mode_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf"]
    wrong_color = "#d95f02"
    hard_color = "#d62728"
    gray = "#7f7f7f"
    for panel_idx, group in enumerate(groups, start=1):
        ax = fig.add_subplot(1, len(groups), panel_idx, projection="3d")
        plot_latents = transformed_groups[panel_idx - 1]
        if pca_scope == "global":
            mean, components = global_mean, global_components
        else:
            panel_points = []
            for i in range(plot_latents.shape[0]):
                valid = group["latent_mask"][i].astype(bool)
                if valid.any():
                    panel_points.append(plot_latents[i, valid])
            mean, components = fit_pca(np.vstack(panel_points), dim=3)
        order = sorted_indices(group)
        any_pos = group["accuracies"].sum() > 0
        for rollout_idx in order:
            valid = group["latent_mask"][rollout_idx].astype(bool)
            if not valid.any():
                continue
            xyz = project(plot_latents[rollout_idx, valid], mean, components, dim=3)
            is_correct = group["accuracies"][rollout_idx] > 0.5
            is_hard = bool(group["hard_mask"][rollout_idx])
            mode = int(group["mode_assign"][rollout_idx])
            if is_correct:
                color = mode_colors[mode % len(mode_colors)] if mode >= 0 else mode_colors[0]
                marker = "o"
                label = f"correct mode {mode}" if mode >= 0 else "correct"
                lw = 1.7
                alpha = 0.95
            elif is_hard:
                color = hard_color
                marker = "x"
                label = "hard negative"
                lw = 1.5
                alpha = 0.9
            else:
                color = wrong_color if any_pos else gray
                marker = "s"
                label = "wrong"
                lw = 1.1
                alpha = 0.65
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=lw, alpha=alpha)
            ax.scatter(xyz[-1:, 0], xyz[-1:, 1], xyz[-1:, 2], color=color, s=30, marker=marker, label=label)
        metrics = group["metrics"]
        ax.set_title(
            f"q{metrics['idx']} acc={metrics['acc']:.2f} modes={metrics['mode_count']} hard={metrics['hard_count']}",
            fontsize=9,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.view_init(elev=22, azim=-58)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    dedup = {}
    for h, label in zip(handles, labels):
        dedup.setdefault(label, h)
    fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=min(5, len(dedup)), frameon=False)
    path = out_dir / "trace_multipath_rollout_paths_3d.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmaps(groups, out_dir: Path, dpi: int):
    plt, sns = ensure_matplotlib()
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.0), constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        order = sorted_indices(group)
        sim = group["similarity"][np.ix_(order, order)]
        sns.heatmap(sim, vmin=-1, vmax=1, cmap="coolwarm", square=True, cbar=len(groups) == 1, ax=ax)
        ax.set_title(f"q{group['metrics']['idx']} path cosine", fontsize=9)
        ax.set_xlabel("sorted rollout")
        ax.set_ylabel("sorted rollout")
    path = out_dir / "trace_multipath_similarity_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def write_outputs(groups, out_dir: Path):
    metrics_path = out_dir / "trace_multipath_metrics.csv"
    keys = sorted(groups[0]["metrics"].keys())
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for group in groups:
            writer.writerow(group["metrics"])

    jsonl_path = out_dir / "trace_multipath_rollouts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for group in groups:
            item = group["item"]
            for i, pred in enumerate(group["pred_strings"]):
                f.write(
                    json.dumps(
                        {
                            "idx": item["idx"],
                            "rollout": i,
                            "correct": float(group["accuracies"][i]),
                            "hard_negative": bool(group["hard_mask"][i]),
                            "mode": int(group["mode_assign"][i]),
                            "prediction": pred,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return metrics_path, jsonl_path


def group_score(group, group_size: int):
    metrics = group["metrics"]
    mixed = metrics["pos_count"] > 0 and metrics["neg_count"] > 0
    balance = 1.0 - abs(metrics["pos_count"] - group_size / 2.0) / max(group_size / 2.0, 1.0)
    mode_gain = metrics["effective_modes"] + 0.5 * metrics["mode_count"]
    margin_gain = max(metrics.get("pos_neg_margin", 0.0), 0.0)
    diversity_gain = max(1.0 - metrics.get("pos_pair_sim", 1.0), 0.0) if metrics["pos_count"] > 1 else 0.0
    return (
        100.0 * float(mixed)
        + 12.0 * metrics["hard_count"]
        + 8.0 * mode_gain
        + 8.0 * margin_gain
        + 4.0 * diversity_gain
        + balance
    )


def select_groups(model, dataset, args):
    if not args.auto_select:
        return [collect_group(model, dataset[i], args.group_size) for i in args.indices]

    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(len(dataset))
    candidate_count = min(max(args.candidate_count, args.num_questions), len(dataset))
    candidate_indices = rng.choice(all_indices, size=candidate_count, replace=False).tolist()
    groups = []
    for idx in candidate_indices:
        group = collect_group(model, dataset[idx], args.group_size)
        groups.append(group)
        torch.cuda.empty_cache()
    groups = sorted(groups, key=lambda group: group_score(group, args.group_size), reverse=True)
    selected = groups[: args.num_questions]
    print("[auto_select] selected indices:", [group["metrics"]["idx"] for group in selected])
    print("[auto_select] selected scores:", [round(group_score(group, args.group_size), 3) for group in selected])
    return selected


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(Path(args.dataset_path))
    model = load_model(args)
    groups = select_groups(model, dataset, args)
    p3d = plot_3d(groups, out_dir, args.dpi, args.trajectory_space, args.pca_scope)
    heatmap = plot_heatmaps(groups, out_dir, args.dpi)
    metrics_path, jsonl_path = write_outputs(groups, out_dir)
    print(f"[done] wrote {p3d}")
    print(f"[done] wrote {heatmap}")
    print(f"[done] wrote {metrics_path}")
    print(f"[done] wrote {jsonl_path}")


if __name__ == "__main__":
    main()
