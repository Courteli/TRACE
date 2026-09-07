"""Independent staged runs with exact sampler cursor and no GPU acquisition."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import torch
import torch.distributed as dist

from . import VERSION
from .artifacts import atomic_json, atomic_torch, file_hash, load_checkpoint, restore_rng, rng_state, save_checkpoint
from .data import answer_matches
from .training import add_metrics, check_finite, epoch_indices, make_scheduler, reduce_gradients, rl_backward, sft_backward, world


def gather_objects(value):
    _, size = world()
    if size == 1:
        return [value]
    result = [None] * size
    dist.all_gather_object(result, value)
    return result


def barrier():
    if dist.is_initialized():
        dist.barrier()


def prepare_run(path, run_identity, stage, resume):
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if (parent / "stage2.log").exists() and (parent / "state").is_dir():
            raise ValueError("cannot write inside a legacy/formal native-v9 run")
    marker = path / "run_identity.json"
    expected = {"version": VERSION, "identity": run_identity, "stage": stage}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError("output directory belongs to a different experiment")
        if resume is None:
            raise ValueError("existing run requires explicit --resume; no implicit overwrite")
    else:
        if resume is not None:
            raise ValueError("resume must use the checkpoint's existing run directory")
        if path.exists() and any(path.iterdir()):
            raise ValueError("new run directory must be empty")
        atomic_json(marker, expected)
    return path


def checkpoint_all(path, model, optimizer, scheduler, run_identity, stage, progress, parent_hash, cache_hash):
    states = gather_objects(rng_state())
    if world()[0] == 0:
        save_checkpoint(path, model, optimizer, scheduler, run_identity=run_identity, stage=stage,
                        progress=dict(progress), parent_hash=parent_hash, cache_hash=cache_hash, rng_by_rank=states)
    barrier()


@torch.no_grad()
def evaluate_records(model, examples):
    """Unique strided shards, never DistributedSampler padding or gold input."""
    rank, size = world()
    model.eval()
    records = []
    for example in examples[rank::size]:
        trace = model.roles(example.question, stochastic=False)
        completion = model.generate(trace, sample=False)
        records.append({"id": example.id, "question": example.question, "prediction": completion.text,
                        "gold_answer": example.answer, "correct": answer_matches(completion.text, example.answer),
                        "generated_tokens": len(completion.tokens), "stopped_on_eos": completion.stopped_on_eos})
    records = [item for shard in gather_objects(records) for item in shard]
    expected = {x.id for x in examples}
    if len(records) != len(expected) or {r["id"] for r in records} != expected:
        raise ValueError("evaluation did not cover exactly the unique requested IDs")
    return sorted(records, key=lambda x: x["id"])


def fit(model, train, validation, run_dir, stage, run_identity, *, cache=None,
        parent_checkpoint=None, resume=None, max_updates=None, gradient_audit_every=100):
    """max_updates is an explicit smoke-test pause, never a completion marker."""
    config = model.config
    rank, size = world()
    if stage not in ("stage0", "stage1", "stage2") or not train or (stage != "stage0" and not validation):
        raise ValueError("known stage and nonempty required splits expected")
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    path = Path(run_dir).resolve()
    if resume is not None:
        resume = Path(resume).resolve(strict=True)
        if resume.parent != path:
            raise ValueError("resume checkpoint must belong to this exact run directory")
    parent_hash = file_hash(parent_checkpoint) if parent_checkpoint else None
    if stage == "stage0":
        if parent_checkpoint is not None or cache is not None:
            raise ValueError("new Stage0 starts from base weights only")
    else:
        if parent_checkpoint is None or cache is None:
            raise ValueError("Stage1/2 require the completed direct parent and fixed teacher cache")
        predecessor = "stage0" if stage == "stage1" else "stage1"
        parent = load_checkpoint(parent_checkpoint, expected_identity=run_identity,
                                 expected_stage=predecessor, require_complete=True)
        teacher_hash = parent_hash if stage == "stage1" else parent["parent_hash"]
        if cache.manifest["teacher_checkpoint_sha256"] != teacher_hash:
            raise ValueError("parent and cache are from different Stage0 lineages")
        if resume is None:
            model.load_checkpoint_state(parent["model"])
            if stage == "stage2":
                model.initialize_stage2_reference()
    if resume is not None:
        restored = load_checkpoint(resume, expected_identity=run_identity, expected_stage=stage)
        if restored["parent_hash"] != parent_hash or restored["cache_hash"] != (cache.fingerprint if cache else None):
            raise ValueError("resume parent/cache identity mismatch")
        if restored["world_size"] != size:
            raise ValueError("exact recovery requires the original distributed world size")
        model.load_checkpoint_state(restored["model"])
    model.configure_stage(stage)
    if rank == 0:
        prepare_run(path, run_identity, stage, resume)
    barrier()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.rl_lr if stage == "stage2" else config.sft_lr,
                                  weight_decay=config.weight_decay, foreach=False)
    epochs = getattr(config, f"{stage}_epochs")
    per_epoch = min(len(train), config.stage2_samples_per_epoch) if stage == "stage2" else len(train)
    total_steps = epochs * math.ceil(per_epoch / config.global_batch_size)
    scheduler = make_scheduler(optimizer, config.warmup_steps, total_steps)
    progress = {"epoch": 0, "offset": 0, "step": 0, "complete": False,
                "best_acc": -1.0, "best_step": None, "validation": []}
    if resume is not None:
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        progress = restored["progress"]
        restore_rng(restored["rng_by_rank"][rank])
        if progress["complete"]:
            finalize_stage(path, run_identity, stage, epochs, progress, parent_hash,
                           cache.fingerprint if cache else None)
            return progress
        if restored.get("stage_run_complete"):
            raise ValueError("a historical best from a completed stage is a parent, not a continuation point")
    cache_hash = cache.fingerprint if cache else None
    updates_here = 0
    while progress["epoch"] < epochs:
        indices = epoch_indices(len(train), config.seed, progress["epoch"], per_epoch)
        if not 0 <= progress["offset"] <= len(indices):
            raise ValueError("invalid saved sampler cursor")
        while progress["offset"] < len(indices):
            batch = indices[progress["offset"]:progress["offset"] + config.global_batch_size]
            local = batch[rank::size]
            optimizer.zero_grad(set_to_none=True)
            metrics = {}
            model.train()
            for index in local:
                example = train[index]
                if stage == "stage0":
                    loss = model.stage0_loss(example)
                    check_finite(loss)
                    loss.backward()
                    values = {"total_loss": float(loss.detach())}
                elif stage == "stage1":
                    values = sft_backward(model, example, cache.get(example, model.device), progress["step"], total_steps)
                else:
                    audit = gradient_audit_every > 0 and progress["step"] % gradient_audit_every == 0
                    values = rl_backward(model, example, cache.get(example, model.device), progress["step"],
                                         total_steps, audit_gradients=audit)
                add_metrics(metrics, values)
            count = reduce_gradients(model, len(local))
            norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            progress["step"] += 1
            progress["offset"] += len(batch)
            updates_here += 1
            shards = gather_objects(metrics)
            if rank == 0:
                combined = {}
                for shard in shards:
                    add_metrics(combined, shard, 1 / count)
                record = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage,
                          "step": progress["step"], "epoch": progress["epoch"], "questions": count,
                          "preclip_grad_norm": float(norm), "lr": optimizer.param_groups[0]["lr"], **combined}
                with (path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record, allow_nan=False), flush=True)
            if progress["step"] % config.save_every == 0:
                checkpoint_all(path / "last.ckpt", model, optimizer, scheduler, run_identity, stage,
                               progress, parent_hash, cache_hash)
            if max_updates is not None and updates_here >= max_updates:
                checkpoint_all(path / "last.ckpt", model, optimizer, scheduler, run_identity, stage,
                               progress, parent_hash, cache_hash)
                return progress
        if stage != "stage0":
            records = evaluate_records(model, validation)
            accuracy = sum(r["correct"] for r in records) / len(records)
            summary = {"epoch": progress["epoch"], "step": progress["step"], "correct": sum(r["correct"] for r in records),
                       "count": len(records), "accuracy": accuracy}
            if rank == 0:
                atomic_json(path / f"validation_epoch{progress['epoch']}.json", {**summary, "records": records})
            barrier()
            summary["artifact_sha256"] = file_hash(path / f"validation_epoch{progress['epoch']}.json")
            progress["validation"].append(summary)
            if accuracy > progress["best_acc"]:
                progress["best_acc"], progress["best_step"] = accuracy, progress["step"]
                checkpoint_all(path / "best.ckpt", model, optimizer, scheduler, run_identity, stage,
                               progress, parent_hash, cache_hash)
        progress["epoch"] += 1
        progress["offset"] = 0
        progress["complete"] = progress["epoch"] == epochs
        checkpoint_all(path / "last.ckpt", model, optimizer, scheduler, run_identity, stage,
                       progress, parent_hash, cache_hash)
    finalize_stage(path, run_identity, stage, epochs, progress, parent_hash, cache_hash)
    return progress


def finalize_stage(path, run_identity, stage, epochs, progress, parent_hash, cache_hash):
    """Idempotent recovery after last.ckpt but before summary/best finalization."""
    if not progress["complete"] or progress["epoch"] != epochs:
        raise ValueError("cannot finalize an incomplete stage")
    if world()[0] == 0:
        selected = path / ("last.ckpt" if stage == "stage0" else "best.ckpt")
        if stage != "stage0":
            best = load_checkpoint(selected, expected_identity=run_identity, expected_stage=stage)
            if not best.get("stage_run_complete"):
                best["stage_run_complete"] = True
                best["stage_completed_at_step"] = progress["step"]
                atomic_torch(selected, best)
            elif best.get("stage_completed_at_step") != progress["step"]:
                raise ValueError("selected checkpoint completion step mismatch")
        atomic_json(path / "training_summary.json", {"version": VERSION, "stage": stage, "identity": run_identity,
            "epochs": epochs, "steps": progress["step"], "complete": True, "parent_sha256": parent_hash,
            "cache_sha256": cache_hash, "selected_checkpoint": selected.name, "selected_sha256": file_hash(selected),
            "last_sha256": file_hash(path / "last.ckpt"), "validation": progress["validation"]})
    barrier()


def validate_records(records, examples):
    by_id = {x.id: x for x in examples}
    if len(by_id) != len(examples) or len(records) != len(by_id) or {r["id"] for r in records} != set(by_id):
        raise ValueError("validation ID coverage mismatch")
    for row in records:
        example = by_id[row["id"]]
        if (row["question"] != example.question or row["gold_answer"] != example.answer
                or type(row["correct"]) is not bool
                or row["correct"] != answer_matches(row["prediction"], example.answer)):
            raise ValueError("validation result cannot be recomputed")
    return sum(r["correct"] for r in records)


def stage_evidence(checkpoint, run_identity, stage, epochs, examples):
    checkpoint = Path(checkpoint)
    selected = load_checkpoint(checkpoint, expected_identity=run_identity, expected_stage=stage, require_complete=True)
    summary = json.loads((checkpoint.parent / "training_summary.json").read_text())
    last_path = checkpoint.parent / "last.ckpt"
    last = load_checkpoint(last_path, expected_identity=run_identity, expected_stage=stage, require_complete=True)
    if (summary["selected_sha256"] != file_hash(checkpoint) or summary["last_sha256"] != file_hash(last_path)
            or summary["identity"] != run_identity or summary["epochs"] != epochs or not summary["complete"]
            or summary["stage"] != stage or summary["selected_checkpoint"] != checkpoint.name
            or summary["parent_sha256"] != selected["parent_hash"] or summary["cache_sha256"] != selected["cache_hash"]
            or last["progress"]["epoch"] != epochs or not last["progress"]["complete"]
            or summary["steps"] != last["progress"]["step"] or summary["validation"] != last["progress"]["validation"]):
        raise ValueError("stage completion/selection evidence mismatch")
    if stage != "stage0":
        history = summary["validation"]
        if [x["epoch"] for x in history] != list(range(epochs)):
            raise ValueError("incomplete or duplicate validation epochs")
        for item in history:
            artifact = checkpoint.parent / f"validation_epoch{item['epoch']}.json"
            saved = json.loads(artifact.read_text())
            correct = validate_records(saved["records"], examples)
            if (item["artifact_sha256"] != file_hash(artifact) or item["count"] != len(examples)
                    or item["correct"] != correct or item["accuracy"] != correct / len(examples)
                    or any(saved[k] != v for k, v in item.items() if k != "artifact_sha256")):
                raise ValueError("epoch validation evidence mismatch")
        best = max(history, key=lambda x: x["accuracy"])
        if (selected["progress"]["best_acc"] != best["accuracy"]
                or selected["progress"]["step"] != best["step"]
                or selected["stage_completed_at_step"] != summary["steps"]):
            raise ValueError("selected checkpoint is not the first full-stage global best")
    return summary


def strict_validate(model, examples, checkpoint, output, run_identity, *, passes=5, expected_count=747):
    if world()[1] != 1 or passes != 5:
        raise ValueError("strict protocol requires five single-device passes")
    if len(examples) != expected_count or len({x.id for x in examples}) != expected_count or len({x.question for x in examples}) != expected_count:
        raise ValueError("strict validation requires exactly the unique validation split")
    if any(not x.id.startswith("val:") for x in examples):
        raise ValueError("test/train records are not validation model-selection data")
    checkpoint = Path(checkpoint).resolve(strict=True)
    payload = load_checkpoint(checkpoint, expected_identity=run_identity, expected_stage="stage2", require_complete=True)
    stage_evidence(checkpoint, run_identity, "stage2", model.config.stage2_epochs, examples)
    model.load_checkpoint_state(payload["model"])
    output = Path(output)
    identity_value = {"version": VERSION, "checkpoint_sha256": file_hash(checkpoint), "identity": run_identity,
                      "passes": passes, "unique_count": expected_count, "mode": "deterministic_mean_single_device"}
    marker = output / "evaluation_identity.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity_value:
            raise ValueError("evaluation output belongs to a different checkpoint/protocol")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("evaluation output directory must be new/empty")
    else:
        atomic_json(marker, identity_value)
    results = []
    for index in range(passes):
        destination = output / f"pass_{index + 1}.json"
        saved = json.loads(destination.read_text()) if destination.exists() else None
        if saved is not None and (saved.get("evaluation_identity") != identity_value or saved.get("pass") != index + 1):
            raise ValueError("saved pass belongs to a different checkpoint/protocol")
        records = saved["records"] if saved else evaluate_records(model, examples)
        validate_records(records, examples)
        result = {"pass": index + 1, "correct": sum(r["correct"] for r in records), "count": expected_count,
                  "accuracy": sum(r["correct"] for r in records) / expected_count}
        atomic_json(destination, {**result, "evaluation_identity": identity_value, "records": records})
        results.append({**result, "artifact_sha256": file_hash(destination)})
    summary = {**identity_value, "results": results, "mean_accuracy": sum(x["accuracy"] for x in results) / passes,
               "note": "Repeated deterministic passes audit reproducibility; they are not independent statistical samples."}
    atomic_json(output / "strict_validation_summary.json", summary)
    return summary
