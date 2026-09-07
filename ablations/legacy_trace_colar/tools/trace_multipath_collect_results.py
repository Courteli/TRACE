#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev


RUN_RE = re.compile(r"trace_multipath_([^_]+(?:_[^_]+)*)_qwen3_c5_L(\d+)_g(\d+)")


def infer_run(run_name):
    m = RUN_RE.search(run_name)
    if not m:
        return "", "", ""
    return m.group(1), int(m.group(2)), int(m.group(3))


def monitor_score(path):
    m = re.search(r"monitor(-?\d+(?:\.\d+)?)", path.name)
    return float(m.group(1)) if m else None


def best_checkpoint(run_dir):
    scored, lasts = [], []
    for ckpt in (run_dir / "checkpoints").glob("*.ckpt"):
        if ckpt.name == "last.ckpt":
            lasts.append(ckpt)
            continue
        score = monitor_score(ckpt)
        if score is not None:
            scored.append((score, ckpt.stat().st_mtime, ckpt))
    if scored:
        return max(scored, key=lambda x: (x[0], x[1]))[2]
    if lasts:
        return max(lasts, key=lambda p: p.stat().st_mtime)
    return None


def read_hparams(run_dir):
    path = run_dir / "hparams.yaml"
    if not path.exists():
        return {}
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path).all_config
        latent = cfg.model.model_kwargs.latent_generation_config
        rl = cfg.model.model_kwargs.rl_config
        return {
            "max_L": latent.get("max_n_latent_forward", ""),
            "min_L": latent.get("min_n_latent_forward", ""),
            "group_size": rl.get("group_size", ""),
            "dataset_name": cfg.data_module.dataset_name,
        }
    except Exception:
        return {}


def read_scalars(run_dir):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return {}
    events = sorted(run_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    scalars = {}
    for event in events:
        try:
            ea = event_accumulator.EventAccumulator(str(event), size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            vals = ea.Scalars(tag)
            if vals:
                scalars[tag] = vals[-1].value
    return scalars


def split_by_rep(samples, key):
    max_reps = 0
    for item in samples:
        vals = item.get(key, [])
        if isinstance(vals, list):
            max_reps = max(max_reps, len(vals))
    reps = []
    for i in range(max_reps):
        vals = []
        for item in samples:
            seq = item.get(key, [])
            if isinstance(seq, list) and i < len(seq):
                vals.append(float(seq[i]))
        if vals:
            reps.append(mean(vals))
    return reps


def summarize(vals):
    if not vals:
        return "", "", 0
    avg = mean(vals)
    ci = 0.0 if len(vals) == 1 else 1.96 * pstdev(vals) / math.sqrt(len(vals))
    return avg, ci, len(vals)


def dataset_from_payload(path, payload, default):
    meta = payload.get("test_metadata", {})
    data_module = meta.get("data_module", {}) if isinstance(meta, dict) else {}
    if data_module.get("dataset_name"):
        return data_module["dataset_name"]
    for name in ["gsm8k_aug_nl", "gsmhard", "svamp", "multiarith"]:
        if name in path.stem:
            return name
    return default


def collect_tests(run_dir, default_dataset):
    latest = {}
    for path in run_dir.glob("test_*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        dataset = dataset_from_payload(path, payload, default_dataset)
        if dataset not in latest or path.stat().st_mtime > latest[dataset].stat().st_mtime:
            latest[dataset] = path
    rows = []
    for dataset, path in sorted(latest.items()):
        payload = json.loads(path.read_text())
        samples = [v for k, v in payload.items() if str(k).isdigit() and isinstance(v, dict)]
        acc, acc_ci, reps = summarize(split_by_rep(samples, "acc"))
        nlf, nlf_ci, _ = summarize(split_by_rep(samples, "n_latent_forward"))
        rows.append(
            {
                "dataset": dataset,
                "test_acc": acc,
                "test_acc_ci": acc_ci,
                "test_n_latent_forward": nlf,
                "test_n_latent_forward_ci": nlf_ci,
                "test_replications": reps,
                "test_json": str(path),
            }
        )
    return rows


def fmt(v):
    if v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", default="logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl")
    parser.add_argument("--run_contains", default="trace_multipath_")
    parser.add_argument("--out_dir", default="run_outputs/trace_multipath/results")
    parser.add_argument("--out_name", default="trace_multipath_summary")
    parser.add_argument("--no_dedupe_latest", action="store_true")
    args = parser.parse_args()

    rows = []
    root = Path(args.log_root)
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir() or args.run_contains not in run_dir.name:
            continue
        variant, l_from_name, g_from_name = infer_run(run_dir.name)
        hp = read_hparams(run_dir)
        scalars = read_scalars(run_dir)
        ckpt = best_checkpoint(run_dir)
        test_rows = collect_tests(run_dir, hp.get("dataset_name", "gsm8k_aug_nl")) or [{"dataset": hp.get("dataset_name", "")}]
        for test_row in test_rows:
            row = {
                "variant": variant,
                "max_L": hp.get("max_L", l_from_name),
                "min_L": hp.get("min_L", l_from_name),
                "group_size": hp.get("group_size", g_from_name),
                "run": run_dir.name,
                "run_mtime": run_dir.stat().st_mtime,
                "best_checkpoint": str(ckpt) if ckpt else "",
                "train_acc": scalars.get("train/accuracies", ""),
                "train_reward": scalars.get("train/rewards", ""),
                "train_L": scalars.get("train/n_latent_forward", ""),
                "trace_mode_count": scalars.get("train/trace/mode_count", ""),
                "trace_effective_modes": scalars.get("train/trace/effective_modes", ""),
                "trace_hard_count": scalars.get("train/trace/hard_count", ""),
                "trace_mixed_frac": scalars.get("train/trace/mixed_frac", ""),
                "trace_pos_intra_sim": scalars.get("train/trace/pos_intra_sim", ""),
                "trace_proto_inter_sim": scalars.get("train/trace/proto_inter_sim", ""),
                "trace_neg_proto_sim": scalars.get("train/trace/neg_proto_sim", ""),
                "trace_hard_proto_sim": scalars.get("train/trace/hard_proto_sim", ""),
                "trace_step_coherence": scalars.get("train/trace/step_coherence", ""),
                "trace_noncollapse": scalars.get("train/trace/noncollapse", ""),
                "trace_delta_norm": scalars.get("train/trace/delta_norm", ""),
                "val_acc": scalars.get("val/acc", ""),
                "val_L": scalars.get("val/n_latent_forward", ""),
            }
            row.update(test_row)
            rows.append(row)

    if not rows:
        raise SystemExit(f"no runs matching {args.run_contains!r} under {root}")

    if not args.no_dedupe_latest:
        latest = {}
        for row in rows:
            key = (row.get("variant", ""), row.get("max_L", ""), row.get("group_size", ""), row.get("dataset", ""))
            old = latest.get(key)
            if old is None or row["run_mtime"] > old["run_mtime"]:
                latest[key] = row
        rows = sorted(latest.values(), key=lambda r: (str(r.get("variant", "")), str(r.get("dataset", ""))))

    fields = [
        "variant",
        "max_L",
        "min_L",
        "group_size",
        "dataset",
        "test_acc",
        "test_acc_ci",
        "test_n_latent_forward",
        "test_n_latent_forward_ci",
        "val_acc",
        "val_L",
        "train_acc",
        "train_reward",
        "train_L",
        "trace_mode_count",
        "trace_effective_modes",
        "trace_hard_count",
        "trace_mixed_frac",
        "trace_pos_intra_sim",
        "trace_proto_inter_sim",
        "trace_neg_proto_sim",
        "trace_hard_proto_sim",
        "trace_step_coherence",
        "trace_noncollapse",
        "trace_delta_norm",
        "test_replications",
        "run",
        "best_checkpoint",
        "test_json",
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k, "")) for k in fields})
    md_path = out_dir / f"{args.out_name}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields[:25]) + " |\n")
        f.write("| " + " | ".join(["---"] * 25) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(k, "")) for k in fields[:25]) + " |\n")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
