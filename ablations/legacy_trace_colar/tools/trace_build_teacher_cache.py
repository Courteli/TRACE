#!/usr/bin/env python3
"""Build frozen teacher hidden-trajectory caches for TRACE trajectory training."""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.utils import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="trace_trajectory_qwen3_instruct")
    parser.add_argument("--dataset", default="gsm8k_aug_nl")
    parser.add_argument("--trainer", default="default")
    parser.add_argument("--workspace_path", default="/home/dingxukai")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--trace_steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load_ckpt_path", default="")
    parser.add_argument("--out_dir", default="run_outputs/trace/teacher_cache")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--tiny_dataset", action="store_true")
    return parser.parse_args()


def load_config(args):
    trainer_config = OmegaConf.load(f"src/configs/trainer/{args.trainer}.yaml")
    model_config = OmegaConf.load(f"src/configs/models/{args.model}.yaml")
    dataset_config = OmegaConf.load(f"src/configs/datasets/{args.dataset}.yaml")
    config = OmegaConf.merge(trainer_config, model_config, DictConfig(dataset_config))
    config.args = {
        "model": args.model,
        "dataset": args.dataset,
        "trainer": args.trainer,
        "workspace_path": args.workspace_path,
        "no_log": True,
        "seed": 0,
    }
    config.trainer.logger = False
    config.trainer.devices = [0]
    config.dataloader.batch_size = args.batch_size
    config.dataloader.val_batch_size = args.batch_size
    config.dataloader.num_workers = 0
    config.dataloader.persistent_workers = False
    config.data_module.tiny_dataset = bool(args.tiny_dataset)
    config.data_module.trace_teacher_cache_dir = None
    config.model.model_kwargs.trace_trajectory_config.trace_steps = args.trace_steps
    config.model.model_kwargs.trace_trajectory_config.teacher_speed = args.trace_steps
    config.model.model_kwargs.trace_trajectory_config.student_speed = args.trace_steps
    config.model.model_kwargs.latent_generation_config.max_n_latent_forward = args.trace_steps
    config.model.model_kwargs.latent_generation_config.min_n_latent_forward = args.trace_steps
    return config


def split_file_name(config, split):
    if split == "train":
        return config.data_module.train_file
    if split == "val":
        return config.data_module.val_file
    return config.data_module.test_file


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args)
    data_module = instantiate_from_config(config.data_module, extra_kwargs={"all_config": config})
    data_module.setup("test" if args.split == "test" else "fit")
    if args.split == "train":
        dataset = data_module.train_set
    elif args.split == "val":
        dataset = data_module.val_set
    else:
        dataset = data_module.test_set

    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    if args.load_ckpt_path:
        state = torch.load(args.load_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        print(model.load_state_dict(state, strict=False))
    model.to(args.device)
    model.eval()

    paths = []
    meta = []
    total = len(dataset) if args.max_items <= 0 else min(len(dataset), args.max_items)
    for i in range(total):
        item = dataset[i]
        teacher_path = model.extract_teacher_trajectory(
            questions=[item["question"]],
            steps=[item["steps"]],
            answers=[item["answer"]],
            trace_steps=args.trace_steps,
        )[0].cpu().to(torch.float16)
        paths.append(teacher_path)
        meta.append({"idx": int(item["idx"]), "source_id": item.get("source_id", int(item["idx"]))})
        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"[cache] {args.split}: {i + 1}/{total}", flush=True)

    teacher_paths = torch.stack(paths, dim=0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(split_file_name(config, args.split)).stem
    suffix = "_tiny" if args.tiny_dataset or args.max_items > 0 else ""
    out_path = out_dir / f"{stem}_trace_teacher_k{args.trace_steps}{suffix}.pt"
    payload = {
        "teacher_paths": teacher_paths,
        "trace_steps": args.trace_steps,
        "split": args.split,
        "model": args.model,
        "dataset": args.dataset,
        "load_ckpt_path": args.load_ckpt_path,
        "meta": meta,
    }
    torch.save(payload, out_path)
    (out_path.with_suffix(".json")).write_text(
        json.dumps({k: v for k, v in payload.items() if k not in {"teacher_paths"}}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] wrote {out_path} shape={tuple(teacher_paths.shape)}")


if __name__ == "__main__":
    main()
