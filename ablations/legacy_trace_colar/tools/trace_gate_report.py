#!/home/dingxukai/miniconda3/envs/ROT/bin/python
import argparse
import json
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TRACE_TAGS = [
    "train/trace/bonus",
    "train/trace/pos_count",
    "train/trace/neg_count",
    "train/trace/pos_neg_sim",
    "train/trace/hard_pos_sim",
    "train/trace/raw_pos_neg_sim",
    "train/trace/raw_pos_pos_sim",
    "train/trace_filter/mixed_frac",
    "train/trace_filter/selected_mixed_frac",
]
TRAIN_TAGS = [
    "train/total_loss",
    "train/grad_norm",
    "train/skipped_nonfinite",
    "train/accuracies",
    "train/rewards",
]
EVAL_TAGS = ["val/acc", "monitor", "test/acc"]
KEY_TAGS = EVAL_TAGS + TRAIN_TAGS + TRACE_TAGS
KNOWN_DATASETS = ("gsm8k_aug_nl", "gsmhard", "multiarith", "svamp")


def scalar_summary(values):
    if not values:
        return None
    scalar_values = [float(v.value) for v in values]
    nonzero = sum(abs(v) > 1e-12 for v in scalar_values)
    return {
        "n": len(values),
        "last_step": int(values[-1].step),
        "last": scalar_values[-1],
        "mean": sum(scalar_values) / len(scalar_values),
        "min": min(scalar_values),
        "max": max(scalar_values),
        "nonzero": nonzero,
        "nonzero_frac": nonzero / len(values),
    }


def monitor_score(path):
    match = re.search(r"monitor(-?\d+(?:\.\d+)?)", path.name)
    return float(match.group(1)) if match else None


def checkpoint_summary(run_dir):
    ckpts = sorted(run_dir.glob("checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
    best = None
    for ckpt in ckpts:
        score = monitor_score(ckpt)
        if score is None:
            continue
        item = (score, ckpt.stat().st_mtime, ckpt)
        if best is None or item[:2] > best[:2]:
            best = item
    last = run_dir / "checkpoints" / "last.ckpt"
    return {
        "count": len(ckpts),
        "best": str(best[2]) if best else "",
        "best_monitor": best[0] if best else None,
        "last": str(last) if last.exists() else "",
    }


def flatten_metric(data, key):
    values = []
    for sample_key, sample in data.items():
        if sample_key in {"test_result", "test_metadata"} or not isinstance(sample, dict):
            continue
        value = sample.get(key)
        if isinstance(value, list):
            values.extend(float(v) for v in value if v is not None)
        elif value is not None:
            values.append(float(value))
    return values


def infer_dataset_from_path(path):
    name = path.name.lower()
    for dataset in KNOWN_DATASETS:
        if dataset.lower() in name:
            return dataset
    pair = path.parent.parent.name if path.parent.parent != path.parent else ""
    if "-" in pair:
        return pair.split("-")[-1]
    return pair or "-"


def infer_repetitions(data, key="acc"):
    counts = []
    for sample_key, sample in data.items():
        if sample_key in {"test_result", "test_metadata"} or not isinstance(sample, dict):
            continue
        value = sample.get(key)
        if isinstance(value, list):
            counts.append(len(value))
        elif value is not None:
            counts.append(1)
    return max(counts) if counts else None


def test_summary(run_dir):
    files = sorted(run_dir.glob("test_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    rows = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata = data.get("test_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        dm = metadata.get("data_module", {}) if isinstance(metadata, dict) else {}
        acc = flatten_metric(data, "acc")
        latent = flatten_metric(data, "n_latent_forward")
        inferred_test_times = infer_repetitions(data)
        test_times = metadata.get("test_times")
        rows.append(
            {
                "file": path.name,
                "dataset": metadata.get("dataset") or dm.get("dataset_name") or infer_dataset_from_path(path),
                "test_times": test_times,
                "inferred_test_times": inferred_test_times,
                "effective_test_times": test_times or inferred_test_times,
                "n_items": len(
                    [
                        k
                        for k, v in data.items()
                        if isinstance(v, dict) and k not in {"test_result", "test_metadata"}
                    ]
                ),
                "n_predictions": len(acc),
                "acc": (sum(acc) / len(acc)) if acc else None,
                "n_latent_forward": (sum(latent) / len(latent)) if latent else None,
            }
        )
    return rows[-1] if rows else None


def load_run(run_dir):
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    tags = {}
    if event_files:
        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", []))
        for tag in KEY_TAGS:
            if tag in scalar_tags:
                tags[tag] = scalar_summary(accumulator.Scalars(tag))
    ckpt = checkpoint_summary(run_dir)
    test = test_summary(run_dir)
    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "mtime": run_dir.stat().st_mtime,
        "tags": tags,
        "checkpoint": ckpt,
        "test": test,
        "gate": infer_gate(run_dir.name, tags, ckpt, test),
    }


def infer_gate(run_name, tags, ckpt, test):
    notes = []
    status = "observed"
    if "answer_only" in run_name:
        grad = tags.get("train/grad_norm")
        if not grad:
            status = "pending"
            notes.append("no grad metric yet")
        elif grad["nonzero_frac"] == 0:
            status = "fail"
            notes.append("zero gradient")
    elif "trace_v2" in run_name or "mixed" in run_name:
        bonus = tags.get("train/trace/bonus")
        grad = tags.get("train/grad_norm")
        if not bonus:
            status = "missing"
            notes.append("no TRACE bonus")
        elif bonus["nonzero"] == 0:
            status = "fail"
            notes.append("TRACE bonus always zero")
        if not grad:
            status = "pending" if status == "observed" else status
            notes.append("no grad metric yet")
        elif grad["nonzero_frac"] == 0:
            status = "fail"
            notes.append("zero gradient")
        elif grad["nonzero_frac"] < 0.05:
            status = "weak" if status == "observed" else status
            notes.append(f"sparse gradient {grad['nonzero_frac']:.3f}")
        else:
            notes.append(f"gradient active {grad['nonzero_frac']:.3f}")
        if ckpt["count"] == 0:
            notes.append("no checkpoint yet")
    elif "sft" in run_name:
        monitor = tags.get("monitor")
        if not monitor:
            status = "pending"
            notes.append("no validation yet")
        else:
            notes.append(f"monitor={monitor['last']:.3f}")
    if test:
        acc = test.get("acc")
        if acc is not None:
            notes.append(f"test_acc={acc:.3f}")
        if test.get("dataset") and test.get("dataset") != "-":
            notes.append(f"test_dataset={test['dataset']}")
        if test.get("effective_test_times") is not None:
            notes.append(f"test_times={test['effective_test_times']}")
    return {"status": status, "notes": "; ".join(notes)}


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def tag_value(row, tag, field, default=None):
    item = row["tags"].get(tag)
    if item is None:
        return default
    return item.get(field, default)


def print_markdown(rows):
    headers = [
        "run",
        "gate",
        "step",
        "monitor",
        "grad_nz",
        "trace_nz",
        "acc",
        "pos",
        "neg",
        "ckpt",
        "notes",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        step = tag_value(row, "train/total_loss", "last_step", tag_value(row, "monitor", "last_step", "-"))
        monitor = tag_value(row, "monitor", "last")
        grad_nz = tag_value(row, "train/grad_norm", "nonzero_frac")
        trace_nz = tag_value(row, "train/trace/bonus", "nonzero_frac")
        acc = tag_value(row, "train/accuracies", "last")
        pos = tag_value(row, "train/trace/pos_count", "last")
        neg = tag_value(row, "train/trace/neg_count", "last")
        ckpt_count = row["checkpoint"]["count"]
        print(
            "| "
            + " | ".join(
                [
                    row["run"],
                    row["gate"]["status"],
                    str(step),
                    fmt(monitor),
                    fmt(grad_nz),
                    fmt(trace_nz),
                    fmt(acc),
                    fmt(pos),
                    fmt(neg),
                    str(ckpt_count),
                    row["gate"]["notes"] or "-",
                ]
            )
            + " |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_root",
        default="logs/trace_colar_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl",
    )
    parser.add_argument("--run_contains", default="")
    parser.add_argument("--latest", type=int, default=0)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    root = Path(args.log_root)
    run_dirs = [p for p in root.glob("*") if p.is_dir()]
    if args.run_contains:
        run_dirs = [p for p in run_dirs if args.run_contains in p.name]
    rows = [load_run(run_dir) for run_dir in sorted(run_dirs, key=lambda p: p.stat().st_mtime)]
    if args.latest > 0:
        rows = rows[-args.latest :]
    print_markdown(rows)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
