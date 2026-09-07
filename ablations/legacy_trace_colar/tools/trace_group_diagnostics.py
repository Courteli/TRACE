#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.utils import instantiate_from_config


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
        "seed": args.seed,
    }
    config.trainer.devices = [args.device_index]
    config.trainer.logger = False
    config.data_module.tiny_dataset = args.tiny_dataset
    config.dataloader.batch_size = 1
    config.dataloader.val_batch_size = 1
    config.dataloader.num_workers = 0
    config.dataloader.persistent_workers = False
    config.model.model_kwargs.latent_generation_config.compression_factor = args.compression_factor
    config.model.model_kwargs.latent_generation_config.latent_temperature = args.latent_temperature
    config.model.model_kwargs.answer_generation_config.max_new_tokens = args.max_new_tokens
    return config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="trace_colar_qwen3_instruct")
    parser.add_argument("--dataset", default="gsm8k_aug_nl")
    parser.add_argument("--trainer", default="default")
    parser.add_argument("--workspace_path", default="/home/dingxukai")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--compression_factor", type=int, default=5)
    parser.add_argument("--latent_temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--tiny_dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args)
    data_module = instantiate_from_config(config.data_module, extra_kwargs={"all_config": config})
    data_module.setup("test" if args.split == "test" else "fit")
    dataloader = data_module.test_dataloader() if args.split == "test" else data_module.val_dataloader()

    model = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(args.device)
    model.eval()

    rows = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_samples:
            break
        question = batch["question"][0]
        answer = batch["answer"][0]
        group_questions = [question] * args.group_size
        use_cuda_autocast = str(args.device).startswith("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_autocast):
            _, _, latent_embeds, latent_mask, pred_ids = model.latent_generate(group_questions, rl_mode=True)
        outputs = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        accuracies = []
        for text in outputs:
            accuracies.append(model.verify_answer(answer, model.extract_answer_from_output(text)))
        acc = torch.tensor(accuracies, device=latent_embeds.device, dtype=torch.float32)

        mask = latent_mask.float()
        pooled = (latent_embeds * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = F.normalize(pooled.float(), dim=-1)
        sim = pooled @ pooled.T
        centered = F.normalize(pooled - pooled.mean(dim=0, keepdim=True), dim=-1)
        centered_sim = centered @ centered.T
        pos = acc > 0.5
        neg = ~pos
        pos_neg_sim = float((sim[pos][:, neg]).mean().item()) if pos.any() and neg.any() else None
        centered_pos_neg_sim = (
            float((centered_sim[pos][:, neg]).mean().item()) if pos.any() and neg.any() else None
        )
        pos_pos_sim = None
        centered_pos_pos_sim = None
        if pos.sum() > 1:
            pos_sim = sim[pos][:, pos]
            centered_pos_sim = centered_sim[pos][:, pos]
            off_diag = ~torch.eye(int(pos.sum().item()), dtype=torch.bool, device=pos_sim.device)
            pos_pos_sim = float(pos_sim[off_diag].mean().item())
            centered_pos_pos_sim = float(centered_pos_sim[off_diag].mean().item())
        rows.append(
            {
                "idx": int(batch["idx"][0]),
                "answer": answer,
                "group_acc": float(acc.mean().item()),
                "pos_count": int(pos.sum().item()),
                "neg_count": int(neg.sum().item()),
                "mean_latent_len": float(latent_mask.sum(dim=1).float().mean().item()),
                "pos_neg_sim": pos_neg_sim,
                "pos_pos_sim": pos_pos_sim,
                "centered_pos_neg_sim": centered_pos_neg_sim,
                "centered_pos_pos_sim": centered_pos_pos_sim,
                "outputs": outputs,
            }
        )

    def mean_present(key):
        vals = [row[key] for row in rows if row[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "split": args.split,
        "num_questions": len(rows),
        "group_size": args.group_size,
        "mixed_count": sum(1 for row in rows if row["pos_count"] > 0 and row["neg_count"] > 0),
        "mixed_frac": sum(1 for row in rows if row["pos_count"] > 0 and row["neg_count"] > 0) / max(len(rows), 1),
        "mean_group_acc": sum(row["group_acc"] for row in rows) / max(len(rows), 1),
        "mean_pos_count": sum(row["pos_count"] for row in rows) / max(len(rows), 1),
        "mean_neg_count": sum(row["neg_count"] for row in rows) / max(len(rows), 1),
        "mean_latent_len": sum(row["mean_latent_len"] for row in rows) / max(len(rows), 1),
        "mean_pos_neg_sim": mean_present("pos_neg_sim"),
        "mean_pos_pos_sim": mean_present("pos_pos_sim"),
        "mean_centered_pos_neg_sim": mean_present("centered_pos_neg_sim"),
        "mean_centered_pos_pos_sim": mean_present("centered_pos_pos_sim"),
    }
    report = {"summary": summary, "samples": rows}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
