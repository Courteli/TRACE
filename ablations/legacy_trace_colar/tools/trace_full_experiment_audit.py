#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


DATA_FILES = {
    "gsm8k_train": "/home/dingxukai/RoT/data/GSM8k-Aug-NL/gsm8k_train_processed.jsonl",
    "gsm8k_val": "/home/dingxukai/RoT/data/GSM8k-Aug-NL/gsm8k_val_processed.jsonl",
    "gsm8k_test": "/home/dingxukai/RoT/data/GSM8k-Aug-NL/gsm8k_test_processed.jsonl",
    "gsmhard": "/home/dingxukai/RoT/data/GSM8k-Hard/gsmhard_test_processed.jsonl",
    "svamp": "/home/dingxukai/RoT/data/SVAMP/svamp_test_processed.jsonl",
    "multiarith": "/home/dingxukai/RoT/data/Multiarith/multiarith_test_processed.jsonl",
}


def line_count(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return sum(1 for _ in f)


def latest_run(log_root: Path, contains: str):
    runs = [p for p in log_root.glob("*") if p.is_dir() and contains in p.name]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def monitor_score(path: Path):
    match = re.search(r"monitor(-?\d+(?:\.\d+)?)", path.name)
    return float(match.group(1)) if match else None


def best_checkpoint(run_dir: Path):
    if not run_dir:
        return None
    scored = []
    for ckpt in (run_dir / "checkpoints").glob("epoch*__monitor*.ckpt"):
        score = monitor_score(ckpt)
        if score is not None:
            scored.append((score, ckpt.stat().st_mtime, ckpt))
    if scored:
        return max(scored, key=lambda item: (item[0], item[1]))[2]
    last = run_dir / "checkpoints" / "last.ckpt"
    return last if last.exists() else None


def load_hparams(run_dir: Path):
    if not run_dir or not (run_dir / "hparams.yaml").exists():
        return {}
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(run_dir / "hparams.yaml").all_config
        model_cfg = cfg.model.model_kwargs
        latent = model_cfg.get("latent_generation_config", {})
        rl = model_cfg.get("rl_config", {})
        trace = model_cfg.get("trace_multipath_config", {})
        return {
            "model": cfg.args.model,
            "dataset": cfg.args.dataset,
            "do_rl": bool(model_cfg.get("do_rl", False)),
            "max_epochs": int(cfg.trainer.max_epochs),
            "limit_val_batches": cfg.trainer.get("limit_val_batches", None),
            "batch_size": int(cfg.dataloader.batch_size),
            "val_batch_size": int(cfg.dataloader.val_batch_size),
            "n_train_samples_per_epoch": int(rl.get("n_train_samples_per_epoch", 0) or 0),
            "group_size": int(rl.get("group_size", 0) or 0),
            "exp_batch_size": int(rl.get("exp_batch_size", 0) or 0),
            "max_l": int(latent.get("max_n_latent_forward", 0) or 0),
            "min_l": int(latent.get("min_n_latent_forward", 0) or 0),
            "trace_path_pretraining": bool(trace.get("enable_trace_path_pretraining", False)),
            "trace_multipath_reward": bool(trace.get("enable_trace_multipath_reward", False)),
        }
    except Exception as exc:
        return {"hparams_error": repr(exc)}


def read_scalars(run_dir: Path):
    if not run_dir:
        return {}
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        return {"error": repr(exc)}
    scalars = {}
    for event in sorted(run_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime):
        try:
            ea = event_accumulator.EventAccumulator(str(event), size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            vals = ea.Scalars(tag)
            if vals:
                scalars[tag] = {
                    "n": len(vals),
                    "last_step": int(vals[-1].step),
                    "last": float(vals[-1].value),
                }
    return scalars


def test_json_summary(run_dir: Path):
    latest = {}
    if not run_dir:
        return latest
    for path in run_dir.glob("test_*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        meta = payload.get("test_metadata", {})
        data_module = meta.get("data_module", {}) if isinstance(meta, dict) else {}
        dataset = data_module.get("dataset_name") or "unknown"
        if dataset not in latest or path.stat().st_mtime > latest[dataset]["mtime"]:
            samples = [v for k, v in payload.items() if str(k).isdigit() and isinstance(v, dict)]
            reps = max((len(s.get("acc", [])) for s in samples), default=0)
            latest[dataset] = {
                "path": str(path),
                "mtime": path.stat().st_mtime,
                "samples": len(samples),
                "replications": reps,
            }
    return {k: {kk: vv for kk, vv in v.items() if kk != "mtime"} for k, v in latest.items()}


def baseline_runs(root: Path, max_l: int):
    out = {}
    for dataset in ["gsm8k_aug_nl", "gsmhard", "svamp", "multiarith"]:
        run = latest_run(root, f"trace_baseline_colar_origin_r5_full50_L{max_l}_{dataset}")
        out[dataset] = test_json_summary(run).get(dataset) or test_json_summary(run).get("unknown") if run else None
    return out


def check_visualization_dir(path: Path):
    return {
        "path": str(path),
        "exists": path.exists(),
        "paths_3d": (path / "trace_multipath_rollout_paths_3d.png").exists(),
        "heatmap": (path / "trace_multipath_similarity_heatmap.png").exists(),
        "metrics_csv": (path / "trace_multipath_metrics.csv").exists(),
        "rollouts_jsonl": (path / "trace_multipath_rollouts.jsonl").exists(),
    }


def csv_rows(path: Path):
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def build_audit(args):
    root = Path(args.root)
    artifact_dir = Path(args.artifact_dir)
    trace_root = root / "logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl"
    cot_root = root / "logs/cot_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl"
    baseline_root = root / "logs/colar_qwen3_instruct/trace_baselines"

    stage0_ckpt_path = artifact_dir / "stage0_cot_ckpt.txt"
    stage0_ckpt = Path(stage0_ckpt_path.read_text().strip()) if stage0_ckpt_path.exists() else None
    stage0_run = stage0_ckpt.parent.parent if stage0_ckpt and stage0_ckpt.exists() else None

    stage1_run = latest_run(trace_root, f"{args.run_name}_stage1_trace_path_sft")
    stage2_run = latest_run(trace_root, f"{args.run_name}_stage2_trace_multipath_rl")
    stage2_ckpt_file = artifact_dir / "stage2_trace_best_ckpt.txt"
    stage2_ckpt = Path(stage2_ckpt_file.read_text().strip()) if stage2_ckpt_file.exists() else best_checkpoint(stage2_run)

    watcher_status = artifact_dir / "stage1_patience4_status.json"
    watcher_decision = artifact_dir / "stage1_patience4_decision.json"

    geometry_dir = artifact_dir / "geometry_200"
    geometry_summary = geometry_dir / "trace_multipath_geometry_summary.json"
    geometry = {
        "dir": str(geometry_dir),
        "summary_exists": geometry_summary.exists(),
        "metrics_rows": csv_rows(geometry_dir / "trace_multipath_geometry_metrics.csv"),
        "rollouts_exists": (geometry_dir / "trace_multipath_geometry_rollouts.jsonl").exists(),
    }
    if geometry_summary.exists():
        try:
            payload = json.loads(geometry_summary.read_text())
            geometry["num_questions"] = payload.get("num_questions")
        except Exception as exc:
            geometry["summary_error"] = repr(exc)

    audit = {
        "run_name": args.run_name,
        "artifact_dir": str(artifact_dir),
        "data_counts": {name: line_count(Path(path)) for name, path in DATA_FILES.items()},
        "stage0": {
            "ckpt": str(stage0_ckpt) if stage0_ckpt else None,
            "ckpt_exists": bool(stage0_ckpt and stage0_ckpt.exists()),
            "run_dir": str(stage0_run) if stage0_run else None,
            "hparams": load_hparams(stage0_run),
            "scalars": read_scalars(stage0_run),
        },
        "stage1": {
            "run_dir": str(stage1_run) if stage1_run else None,
            "best_ckpt": str(best_checkpoint(stage1_run)) if best_checkpoint(stage1_run) else None,
            "hparams": load_hparams(stage1_run),
            "scalars": read_scalars(stage1_run),
            "patience_status": json.loads(watcher_status.read_text()) if watcher_status.exists() else None,
            "patience_decision": json.loads(watcher_decision.read_text()) if watcher_decision.exists() else None,
        },
        "stage2": {
            "run_dir": str(stage2_run) if stage2_run else None,
            "best_ckpt": str(stage2_ckpt) if stage2_ckpt else None,
            "best_ckpt_exists": bool(stage2_ckpt and stage2_ckpt.exists()),
            "hparams": load_hparams(stage2_run),
            "scalars": read_scalars(stage2_run),
            "tests": test_json_summary(stage2_run),
        },
        "colar_baseline": baseline_runs(baseline_root, args.max_l),
        "visualizations": {
            "fixed_q012_global": check_visualization_dir(artifact_dir / "visualizations/fixed_q012_global"),
            "auto_global": check_visualization_dir(artifact_dir / f"visualizations/auto{args.vis_candidate_count}_global"),
        },
        "geometry_200": geometry,
    }
    return audit


def summarize_status(audit):
    checks = []
    counts = audit["data_counts"]
    checks.append(("data_full_gsm8k_train", counts.get("gsm8k_train") == 6726))
    checks.append(("data_full_gsm8k_val", counts.get("gsm8k_val") == 747))
    checks.append(("stage0_ckpt_exists", audit["stage0"]["ckpt_exists"]))
    s1 = audit["stage1"]["hparams"]
    checks.append(("stage1_full_train_6726", s1.get("n_train_samples_per_epoch") == 6726))
    checks.append(("stage1_path_pretraining_on", s1.get("trace_path_pretraining") is True))
    checks.append(("stage1_full_val", s1.get("limit_val_batches") in (1.0, None)))
    s2 = audit["stage2"]["hparams"]
    checks.append(("stage2_ckpt_exists", audit["stage2"]["best_ckpt_exists"]))
    checks.append(("stage2_rl_on", s2.get("do_rl") is True))
    checks.append(("stage2_rl_budget_512", s2.get("n_train_samples_per_epoch") == 512))
    checks.append(("stage2_group_8", s2.get("group_size") == 8))
    required_tests = {"gsm8k_aug_nl", "gsmhard", "svamp", "multiarith"}
    checks.append(("trace_ood_all_present", required_tests.issubset(set(audit["stage2"]["tests"].keys()))))
    checks.append(("colar_ood_all_present", all(audit["colar_baseline"].get(name) for name in required_tests)))
    for name, info in audit["visualizations"].items():
        checks.append((f"{name}_3d", info["paths_3d"]))
        checks.append((f"{name}_heatmap", info["heatmap"]))
    checks.append(("geometry_200_rows", audit["geometry_200"].get("metrics_rows") == 200))
    checks.append(("geometry_200_summary", audit["geometry_200"].get("num_questions") == 200))
    return [{"check": name, "ok": bool(ok)} for name, ok in checks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/disk1/dingxukai/trace_colar")
    parser.add_argument("--run-name", default="trace_v3_three_stage_full_20260705_stage1solid")
    parser.add_argument(
        "--artifact-dir",
        default="/disk1/dingxukai/trace_colar/run_outputs/trace_multipath/trace_v3_three_stage_full_20260705_stage1solid",
    )
    parser.add_argument("--max-l", type=int, default=40)
    parser.add_argument("--vis-candidate-count", type=int, default=200)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    audit = build_audit(args)
    audit["checks"] = summarize_status(audit)
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        print(path)
    print(json.dumps({"run_name": audit["run_name"], "checks": audit["checks"]}, indent=2))


if __name__ == "__main__":
    main()
