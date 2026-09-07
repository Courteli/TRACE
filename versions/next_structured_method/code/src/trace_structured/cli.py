"""Offline, explicit resource use. No job reservation, signalling or v9 imports."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import torch
import torch.distributed as dist

from . import VERSION
from .artifacts import TeacherCache, atomic_json, base_fingerprint, build_cache, file_hash, identity, load_checkpoint, object_hash, tensor_hash
from .config import Config
from .data import audit_data, load_split
from .model import load_model
from .runner import fit, stage_evidence, strict_validate, validate_records
from .training import world


CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parents[2]


def safe_output(path, data_root, model_path=None):
    path = Path(path).resolve()
    protected = [CODE_ROOT, REPO_ROOT / "main", REPO_ROOT / "archive", REPO_ROOT / "data", Path(data_root).resolve()]
    if model_path:
        protected.append(Path(model_path).resolve())
    if path == REPO_ROOT or any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise ValueError("runtime outputs must not overlap source, archived runs, data or base weights")
    for parent in (path, *path.parents):
        if (parent / "stage2.log").exists() and (parent / "state").is_dir():
            raise ValueError("legacy/formal native-v9 output is protected")
    return path


def initialize(args, config):
    if not args.model_path:
        raise ValueError("--model-path (or TRACE_MODEL_PATH) must name local base weights")
    device = args.device
    if device == "cuda":
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.device(device).type == "cuda" else "gloo",
                                timeout=timedelta(hours=3))
    # Same initialization on all ranks, independent rollout RNG after loading.
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    base = base_fingerprint(args.model_path)
    model = load_model(args.model_path, config, device)
    run_identity = identity(config, base, model.tokenizer, next(model.language_model.parameters()).dtype)
    random.seed(config.seed + world()[0])
    torch.manual_seed(config.seed + world()[0])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + world()[0])
    return model, run_identity


def verify_lineage(root, config, validation=None):
    root = Path(root)
    validation = load_split(REPO_ROOT / "data/GSM8k-Aug-NL", "val") if validation is None else validation
    paths = {"stage0": root / "stage0/last.ckpt", "stage1": root / "stage1/best.ckpt", "stage2": root / "stage2/best.ckpt"}
    values = {s: load_checkpoint(p, expected_stage=s, require_complete=True) for s, p in paths.items()}
    shared = values["stage0"]["identity"]
    if any(v["identity"] != shared for v in values.values()) or shared["config"] != config.fingerprint():
        raise ValueError("cross-stage method identity mismatch")
    if values["stage0"]["parent_hash"] is not None:
        raise ValueError("Stage0 must originate from declared base weights")
    if values["stage1"]["parent_hash"] != file_hash(paths["stage0"]) or values["stage2"]["parent_hash"] != file_hash(paths["stage1"]):
        raise ValueError("direct predecessor fingerprint mismatch")
    cache = json.loads((root / "teacher_cache/manifest.json").read_text())
    if cache["identity"] != shared or cache["teacher_checkpoint_sha256"] != file_hash(paths["stage0"]):
        raise ValueError("teacher cache is outside the training lineage")
    if cache["targets_sha256"] != file_hash(root / "teacher_cache/targets.pt"):
        raise ValueError("teacher cache content mismatch")
    if any(values[s]["cache_hash"] != object_hash(cache) for s in ("stage1", "stage2")):
        raise ValueError("training did not use the declared fixed cache")
    for prefix, source in (("reference_policy.", "policy."), ("score_plan_head.", "plan_head.")):
        entries = {k: v for k, v in values["stage2"]["model"].items() if k.startswith(prefix)}
        if not entries or any(not torch.equal(v, values["stage1"]["model"][source + k[len(prefix):]]) for k, v in entries.items()):
            raise ValueError("Stage2 fixed reference/scorer differs from selected Stage1")
    for stage, path in paths.items():
        stage_evidence(path, shared, stage, getattr(config, f"{stage}_epochs"), validation)
    evaluation = json.loads((root / "strict_validation/strict_validation_summary.json").read_text())
    if (evaluation["checkpoint_sha256"] != file_hash(paths["stage2"]) or evaluation["identity"] != shared
            or [r["pass"] for r in evaluation["results"]] != list(range(1, 6))
            or evaluation["unique_count"] != len(validation) or evaluation["passes"] != 5
            or evaluation["mode"] != "deterministic_mean_single_device"):
        raise ValueError("strict evaluation identity/pass count mismatch")
    for result in evaluation["results"]:
        path = root / "strict_validation" / f"pass_{result['pass']}.json"
        saved = json.loads(path.read_text())
        correct = validate_records(saved["records"], validation)
        marker = saved["evaluation_identity"]
        if (result["count"] != len(validation) or result["correct"] != correct
                or result["accuracy"] != correct / len(validation) or result["artifact_sha256"] != file_hash(path)
                or any(saved[k] != v for k, v in result.items() if k != "artifact_sha256")
                or any(marker[k] != evaluation[k] for k in marker)):
            raise ValueError("strict evaluation artifact mismatch")
    if evaluation["mean_accuracy"] != sum(x["accuracy"] for x in evaluation["results"]) / 5:
        raise ValueError("strict summary average mismatch")
    manifest = {"version": VERSION, "identity": shared, "checkpoints": {s: {"path": str(p.relative_to(root)), "sha256": file_hash(p)} for s, p in paths.items()},
                "teacher_cache_sha256": object_hash(cache), "strict_summary_sha256": file_hash(root / "strict_validation/strict_validation_summary.json"),
                "verified": True}
    atomic_json(root / "lineage_manifest.json", manifest)
    return manifest


def pipeline(args, config):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("pipeline orchestrator must not itself run under torchrun")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = safe_output(args.output, args.data_root, args.model_path)
    run = str(CODE_ROOT / "run.py")
    common = ["--data-root", str(Path(args.data_root).resolve()), "--device", args.device]
    if args.model_path:
        common += ["--model-path", str(Path(args.model_path).resolve())]
    if args.config:
        common += ["--config", str(Path(args.config).resolve())]
    commands = []
    for stage in ("stage0", "stage1", "stage2"):
        cmd = [sys.executable]
        if args.workers > 1:
            cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.workers}"]
        cmd += [run, "train", stage, *common, "--output", str(root / stage)]
        if stage != "stage0":
            parent = root / ("stage0/last.ckpt" if stage == "stage1" else "stage1/best.ckpt")
            cmd += ["--parent", str(parent), "--cache", str(root / "teacher_cache")]
        if (root / stage / "last.ckpt").exists():
            cmd += ["--resume", str(root / stage / "last.ckpt")]
        commands.append(cmd)
        if stage == "stage0" and not (root / "teacher_cache/manifest.json").exists():
            commands.append([sys.executable, run, "cache", *common, "--parent", str(root / "stage0/last.ckpt"),
                             "--output", str(root / "teacher_cache")])
    commands.append([sys.executable, run, "evaluate", *common, "--checkpoint", str(root / "stage2/best.ckpt"),
                     "--output", str(root / "strict_validation")])
    if args.dry_run:
        print(json.dumps({"version": VERSION, "data": audit_data(args.data_root), "commands": commands,
                          "note": "No model loaded or process started; explicit user-allocated devices only."}, indent=2))
        return
    if not args.model_path:
        raise ValueError("pipeline requires local base weights")
    for command in commands:
        subprocess.run(command, check=True)
    print(json.dumps(verify_lineage(root, config, load_split(args.data_root, "val")), indent=2))


def parser():
    p = argparse.ArgumentParser(description="Graph-free TRACE: independent plain-CoT staged training")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("audit-data", "train", "cache", "evaluate", "pipeline", "verify-lineage"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path)
        cmd.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/GSM8k-Aug-NL")
        if name not in ("audit-data", "verify-lineage"):
            cmd.add_argument("--model-path", default=os.environ.get("TRACE_MODEL_PATH"))
            cmd.add_argument("--device", default="cuda")
        if name != "audit-data":
            cmd.add_argument("--output", type=Path, required=True)
        if name == "train":
            cmd.add_argument("stage", choices=("stage0", "stage1", "stage2"))
            cmd.add_argument("--parent", type=Path)
            cmd.add_argument("--cache", type=Path)
            cmd.add_argument("--resume", type=Path)
            cmd.add_argument("--max-updates", type=int)
            cmd.add_argument("--gradient-audit-every", type=int, default=100)
        if name == "cache":
            cmd.add_argument("--parent", type=Path, required=True)
        if name == "evaluate":
            cmd.add_argument("--checkpoint", type=Path, required=True)
        if name == "pipeline":
            cmd.add_argument("--workers", type=int, default=4)
            cmd.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = Config.load(args.config)
    if args.command == "audit-data":
        print(json.dumps(audit_data(args.data_root), indent=2))
        return
    if args.command == "pipeline":
        return pipeline(args, config)
    if args.command == "verify-lineage":
        safe_output(args.output, args.data_root)
        print(json.dumps(verify_lineage(args.output, config, load_split(args.data_root, "val")), indent=2))
        return
    safe_output(args.output, args.data_root, args.model_path)
    if args.command == "train" and args.max_updates is not None and args.max_updates < 1:
        raise ValueError("max-updates must be positive")
    model, run_identity = initialize(args, config)
    try:
        if args.command == "cache":
            if world()[1] != 1:
                raise ValueError("teacher cache extraction is a single-device operation")
            print(json.dumps(build_cache(model, load_split(args.data_root, "train"), args.output,
                                        run_identity=run_identity, teacher_checkpoint=args.parent), indent=2))
        elif args.command == "evaluate":
            print(json.dumps(strict_validate(model, load_split(args.data_root, "val"), args.checkpoint,
                                             args.output, run_identity), indent=2))
        else:
            train, validation = load_split(args.data_root, "train"), load_split(args.data_root, "val")
            cache = None
            if args.stage != "stage0":
                if not args.parent or not args.cache:
                    raise ValueError("--parent and --cache are required for Stage1/2")
                parent = load_checkpoint(args.parent, expected_identity=run_identity,
                                         expected_stage="stage0" if args.stage == "stage1" else "stage1", require_complete=True)
                model.load_checkpoint_state(parent["model"])
                teacher_hash = file_hash(args.parent) if args.stage == "stage1" else parent["parent_hash"]
                cache = TeacherCache(args.cache, train, model, run_identity, teacher_hash)
            fit(model, train, validation, args.output, args.stage, run_identity, cache=cache,
                parent_checkpoint=args.parent, resume=args.resume, max_updates=args.max_updates,
                gradient_audit_every=args.gradient_audit_every)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
