"""Content-addressed teacher caches and fail-closed full-state recovery."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import tempfile
import torch

from . import SCHEMA_VERSION, VERSION
from .data import SPLITS, file_hash
from .targets import Targets


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tensor_hash(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() + raw).hexdigest()


def source_fingerprint():
    root = Path(__file__).resolve().parents[2]
    return object_hash({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))})


def base_fingerprint(path):
    path = Path(path).resolve(strict=True)
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix in (".safetensors", ".bin", ".json", ".model", ".txt", ".tiktoken")))
    if not (path / "config.json").is_file() or not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError("a local base model with actual weight shards is required, not metadata alone")
    return {"sha256": object_hash({p.name: file_hash(p) for p in files}), "files": [p.name for p in files]}


def tokenizer_fingerprint(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return object_hash({"class": type(tokenizer).__name__, "vocab": tokenizer.get_vocab(),
                        "special_tokens": tokenizer.special_tokens_map,
                        "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id,
                        "backend": backend.to_str() if backend is not None else None})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_initialized():
            raise ValueError("CUDA checkpoint requires an initialized rank-local CUDA device")
        torch.cuda.set_rng_state(state["cuda"])
    elif torch.cuda.is_initialized():
        raise ValueError("CPU RNG checkpoint cannot be exactly resumed on CUDA")


def example_hash(example):
    return object_hash({"id": example.id, "question": example.question,
                        "steps": example.steps, "answer": example.answer})


def identity(config, base, tokenizer, compute_dtype="float32"):
    return {"version": VERSION, "schema": SCHEMA_VERSION, "config": config.fingerprint(),
            "base": base, "tokenizer": tokenizer_fingerprint(tokenizer),
            "source": source_fingerprint(), "data": {s: x[2] for s, x in SPLITS.items()},
            "roles": list(config.role_names), "compute_dtype": str(compute_dtype),
            "runtime": {name: version(name) for name in ("torch", "transformers", "peft")}}


def load_checkpoint(path, *, expected_identity=None, expected_stage=None, require_complete=False):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("version") != VERSION or value.get("schema") != SCHEMA_VERSION:
        raise ValueError("foreign/legacy checkpoint: explicit new Stage0 lineage required")
    if expected_identity is not None and value.get("identity") != expected_identity:
        raise ValueError("checkpoint source/config/base/tokenizer/data identity mismatch")
    if expected_stage is not None and value.get("stage") != expected_stage:
        raise ValueError("checkpoint stage is not the required direct predecessor")
    if require_complete and not (value.get("progress", {}).get("complete") or value.get("stage_run_complete")):
        raise ValueError("predecessor stage has not completed")
    if not value.get("model") or not isinstance(value.get("optimizer"), dict):
        raise ValueError("checkpoint lacks model/optimizer state")
    return value


def save_checkpoint(path, model, optimizer, scheduler, *, run_identity, stage, progress,
                    parent_hash, cache_hash, rng_by_rank):
    value = {"version": VERSION, "schema": SCHEMA_VERSION, "identity": run_identity,
             "stage": stage, "progress": progress, "parent_hash": parent_hash, "cache_hash": cache_hash,
             "model": model.checkpoint_state(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "rng_by_rank": rng_by_rank,
             "world_size": len(rng_by_rank)}
    atomic_torch(path, value)


@torch.no_grad()
def build_cache(model, examples, output_dir, *, run_identity, teacher_checkpoint):
    """Only ordinary training records; no validation/test target cache."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("cache output must be new/empty; existing artifacts are not overwritten")
    parent = load_checkpoint(teacher_checkpoint, expected_identity=run_identity,
                             expected_stage="stage0", require_complete=True)
    model.load_checkpoint_state(parent["model"])
    model.eval().requires_grad_(False)
    rows = {}
    for example in examples:
        if not example.id.startswith("train:") or example.id in rows:
            raise ValueError("cache accepts unique training IDs only")
        target = model.teacher_target(example)
        rows[example.id] = {"example_sha256": example_hash(example), "target": target.payload()}
    if not rows:
        raise ValueError("empty teacher cache")
    manifest = {"version": VERSION, "schema": SCHEMA_VERSION, "identity": run_identity,
                "teacher_checkpoint_sha256": file_hash(teacher_checkpoint), "split": "train",
                "data_sha256": SPLITS["train"][2], "count": len(rows),
                "projection_sha256": tensor_hash(model.semantic_projection),
                "boundary_rule": "question-last then each physical CoT step-last before newline; layer_norm(last_hidden) @ fixed_projection",
                "dtype": "float32", "teacher_forward_dtype": str(next(model.language_model.parameters()).dtype),
                "execution_mode": "eval_no_grad",
                "ids_sha256": object_hash(sorted(rows))}
    atomic_torch(output_dir / "targets.pt", {"rows": rows})
    manifest["targets_sha256"] = file_hash(output_dir / "targets.pt")
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest


class TeacherCache:
    def __init__(self, path, examples, model, run_identity, teacher_hash):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("version") != VERSION or manifest.get("schema") != SCHEMA_VERSION:
            raise ValueError("cache schema/version mismatch")
        if manifest.get("identity") != run_identity or manifest.get("teacher_checkpoint_sha256") != teacher_hash:
            raise ValueError("cache teacher/config/data/source identity mismatch")
        if manifest.get("split") != "train" or manifest.get("data_sha256") != SPLITS["train"][2]:
            raise ValueError("only fixed training-split targets are accepted")
        if manifest.get("projection_sha256") != tensor_hash(model.semantic_projection):
            raise ValueError("cache coordinate projection mismatch")
        if manifest.get("targets_sha256") != file_hash(path / "targets.pt"):
            raise ValueError("corrupted teacher target file")
        self.rows = torch.load(path / "targets.pt", map_location="cpu", weights_only=True)["rows"]
        expected_ids = {x.id for x in examples}
        if (set(self.rows) != expected_ids or manifest.get("count") != len(expected_ids)
                or manifest.get("ids_sha256") != object_hash(sorted(expected_ids))):
            raise ValueError("cache must cover exactly the supplied training examples")
        k, d = model.config.solve_roles, model.config.semantic_dim
        for example in examples:
            entry = self.rows[example.id]
            target = Targets.from_payload(entry["target"])
            if entry["example_sha256"] != example_hash(example):
                raise ValueError("cache ID/question/CoT/answer mismatch")
            if (target.solve.shape != (k, d) or target.cumulative.shape != (k, d)
                    or target.end.shape != (d,) or target.span_mask.shape != (k,)
                    or target.span_mask.dtype != torch.bool):
                raise ValueError("invalid target shape/dtype")
            if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (target.solve, target.cumulative, target.end)):
                raise ValueError("target cache must be finite float32")
            if not torch.allclose(target.solve.cumsum(0), target.cumulative, atol=2e-5, rtol=2e-5):
                raise ValueError("target cumulative geometry mismatch")
            expected_mask = torch.arange(k) < min(k, len(example.steps))
            if not torch.equal(target.span_mask, expected_mask):
                raise ValueError("target span mask is inconsistent with CoT segmentation")
        self.manifest = manifest
        self.fingerprint = object_hash(manifest)

    def get(self, example, device):
        if not example.id.startswith("train:"):
            raise ValueError("teacher targets are training-only")
        return Targets.from_payload(self.rows[example.id]["target"]).to(device)
