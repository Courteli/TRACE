#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev


VARIANT_RE = re.compile(
    r"trace_trajectory_(full|no_state|no_trans|answer_only|k4|k16|k40)_qwen3_c5_k(\d+)"
)


def infer_variant(run_name):
    match = VARIANT_RE.search(run_name)
    if not match:
        if "colar_origin_c5" in run_name or "origin_colar" in run_name:
            return "colar_origin_c5", ""
        return "", ""
    return match.group(1), int(match.group(2))


def monitor_score(path):
    match = re.search(r"monitor(-?\d+(?:\.\d+)?)", path.name)
    return float(match.group(1)) if match else None


def best_checkpoint(run_dir):
    scored = []
    lasts = []
    for ckpt in (run_dir / "checkpoints").glob("*.ckpt"):
        if ckpt.name == "last.ckpt":
            lasts.append(ckpt)
            continue
        score = monitor_score(ckpt)
        if score is not None:
            scored.append((score, ckpt.stat().st_mtime, ckpt))
    if scored:
        return max(scored, key=lambda item: (item[0], item[1]))[2]
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
        model_kwargs = cfg.model.model_kwargs
        trace_cfg = model_kwargs.get("trace_trajectory_config", {})
        latent_cfg = model_kwargs.get("latent_generation_config", {})
        return {
            "trace_steps": trace_cfg.get("trace_steps", ""),
            "max_n_latent_forward": latent_cfg.get("max_n_latent_forward", ""),
            "dataset_name": cfg.data_module.dataset_name,
            "test_times": cfg.args.test_times,
        }
    except Exception:
        return {}


def read_scalars(run_dir):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return {}

    events = sorted(run_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not events:
        return {}

    scalars = {}
    for event_path in events:
        try:
            ea = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            vals = ea.Scalars(tag)
            if vals:
                scalars[tag] = (vals[-1].value, vals[-1].step)
    return scalars


def split_by_rep(sample_entries, key):
    max_reps = 0
    for entry in sample_entries:
        values = entry.get(key, [])
        if isinstance(values, list):
            max_reps = max(max_reps, len(values))
    reps = []
    for rep_idx in range(max_reps):
        vals = []
        for entry in sample_entries:
            values = entry.get(key, [])
            if isinstance(values, list) and rep_idx < len(values):
                vals.append(float(values[rep_idx]))
        if vals:
            reps.append(mean(vals))
    return reps


def summarize_reps(values):
    if not values:
        return "", "", 0
    avg = mean(values)
    ci = 0.0 if len(values) == 1 else 1.96 * pstdev(values) / math.sqrt(len(values))
    return avg, ci, len(values)


def infer_dataset_from_json(path, payload, default_dataset):
    meta = payload.get("test_metadata", {})
    data_module = meta.get("data_module", {}) if isinstance(meta, dict) else {}
    dataset = data_module.get("dataset_name")
    if dataset:
        return dataset
    stem = path.stem
    for candidate in ["gsm8k_aug_nl", "gsmhard", "svamp", "multiarith"]:
        if candidate in stem:
            return candidate
    return default_dataset


def collect_tests(run_dir, default_dataset):
    latest_by_dataset = {}
    for path in run_dir.glob("test_*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        dataset = infer_dataset_from_json(path, payload, default_dataset)
        previous = latest_by_dataset.get(dataset)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            latest_by_dataset[dataset] = path

    rows = []
    for dataset, path in sorted(latest_by_dataset.items()):
        payload = json.loads(path.read_text())
        meta = payload.get("test_metadata", {})
        latent_cfg = meta.get("latent_generation_config", {}) if isinstance(meta, dict) else {}
        sample_entries = [v for k, v in payload.items() if str(k).isdigit() and isinstance(v, dict)]
        acc_reps = split_by_rep(sample_entries, "acc")
        nlf_reps = split_by_rep(sample_entries, "n_latent_forward")
        out_len_reps = split_by_rep(sample_entries, "output_length")
        acc_mean, acc_ci, reps = summarize_reps(acc_reps)
        nlf_mean, nlf_ci, _ = summarize_reps(nlf_reps)
        out_len_mean, out_len_ci, _ = summarize_reps(out_len_reps)
        rows.append(
            {
                "dataset": dataset,
                "test_json": str(path),
                "test_replications": reps,
                "test_acc": acc_mean,
                "test_acc_ci": acc_ci,
                "test_n_latent_forward": nlf_mean,
                "test_n_latent_forward_ci": nlf_ci,
                "test_output_length": out_len_mean,
                "test_output_length_ci": out_len_ci,
                "metadata_max_L": latent_cfg.get("max_n_latent_forward", ""),
            }
        )
    return rows


def format_value(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_root",
        default="logs/trace_trajectory_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl",
    )
    parser.add_argument("--run_contains", default="trace_trajectory_")
    parser.add_argument("--out_dir", default="run_outputs/trace/results")
    parser.add_argument("--out_name", default="trace_trajectory_summary")
    parser.add_argument("--no_dedupe_latest", action="store_true")
    args = parser.parse_args()

    log_root = Path(args.log_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(log_root.glob("*")):
        if not run_dir.is_dir() or args.run_contains not in run_dir.name:
            continue
        variant, k_from_name = infer_variant(run_dir.name)
        hparams = read_hparams(run_dir)
        scalars = read_scalars(run_dir)
        ckpt = best_checkpoint(run_dir)
        default_dataset = hparams.get("dataset_name", "")
        test_rows = collect_tests(run_dir, default_dataset)
        if not test_rows:
            test_rows = [{"dataset": default_dataset}]

        for test_row in test_rows:
            configured_max_l = hparams.get("max_n_latent_forward", "")
            if configured_max_l == "":
                configured_max_l = test_row.get("metadata_max_L", "")
            row = {
                "run": run_dir.name,
                "run_mtime": run_dir.stat().st_mtime,
                "variant": variant,
                "trace_steps": hparams.get("trace_steps", k_from_name),
                "configured_max_L": configured_max_l if configured_max_l != "" else k_from_name,
                "best_checkpoint": str(ckpt) if ckpt else "",
                "train_total_loss": scalars.get("train/total_loss", ("", ""))[0],
                "train_answer_loss": scalars.get("train/answer_loss", ("", ""))[0],
                "train_state_loss": scalars.get("train/state_loss", ("", ""))[0],
                "train_transition_loss": scalars.get("train/transition_loss", ("", ""))[0],
                "train_state_cos": scalars.get("train/trace/state_cos", ("", ""))[0],
                "train_transition_cos": scalars.get("train/trace/transition_cos", ("", ""))[0],
                "val_acc": scalars.get("val/acc", ("", ""))[0],
                "val_L": scalars.get("val/n_latent_forward", ("", ""))[0],
                "monitor": scalars.get("monitor", ("", ""))[0],
            }
            row.update(test_row)
            rows.append(row)

    if not rows:
        raise SystemExit(f"no runs matching {args.run_contains!r} under {log_root}")

    if not args.no_dedupe_latest:
        latest = {}
        for row in rows:
            key = (row.get("variant", ""), row.get("trace_steps", ""), row.get("dataset", ""))
            old = latest.get(key)
            if old is None or float(row.get("run_mtime", 0.0)) > float(old.get("run_mtime", 0.0)):
                latest[key] = row
        rows = sorted(latest.values(), key=lambda row: (str(row.get("variant", "")), str(row.get("dataset", ""))))

    fieldnames = [
        "variant",
        "trace_steps",
        "configured_max_L",
        "dataset",
        "test_acc",
        "test_acc_ci",
        "test_n_latent_forward",
        "test_n_latent_forward_ci",
        "val_acc",
        "val_L",
        "train_state_cos",
        "train_transition_cos",
        "train_state_loss",
        "train_transition_loss",
        "train_answer_loss",
        "monitor",
        "test_replications",
        "run",
        "best_checkpoint",
        "test_json",
    ]
    csv_path = out_dir / f"{args.out_name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_value(row.get(k, "")) for k in fieldnames})

    md_path = out_dir / f"{args.out_name}.md"
    with md_path.open("w") as f:
        f.write("| " + " | ".join(fieldnames[:14]) + " |\n")
        f.write("| " + " | ".join(["---"] * 14) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_value(row.get(k, "")) for k in fieldnames[:14]) + " |\n")

    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
