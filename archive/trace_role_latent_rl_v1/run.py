import os
import ast
import json
import random
import shutil
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
import torch
import lightning.pytorch as pl
import torch.distributed as dist

from src.utils.utils import instantiate_from_config, get_timestamp, get_metric_statistics
from src.utils.log import setup_logger 


logger = setup_logger(__name__)
start_time = get_timestamp()


def seed_current_process(seed: int, *, verbose: bool = True) -> int:
    """Seed CPU RNGs and only the CUDA device owned by this process."""
    seed = int(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if verbose:
        logger.info(f"Seed set to {seed}")
    return seed


def install_rank_local_ddp_seed_reset() -> None:
    """Prevent each Lightning DDP worker from opening every visible GPU."""
    import lightning.pytorch.strategies.ddp as ddp_module

    def reset_current_rank_seed() -> None:
        seed_current_process(
            int(os.environ.get("PL_GLOBAL_SEED", "0")),
            verbose=False,
        )

    ddp_module.reset_seed = reset_current_rank_seed


def load_full_checkpoint(model: pl.LightningModule, path: str):
    """Restore model weights and checkpoint-level auxiliary state."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    model.on_load_checkpoint(checkpoint)
    return checkpoint["state_dict"]


def do_test(model: pl.LightningModule, trainer: pl.Trainer, ckpt_path: str, data_module: pl.LightningDataModule, args):
    results = defaultdict(list)
    if ckpt_path == "best":
        checkpoint_path = trainer.checkpoint_callback.best_model_path
    elif ckpt_path == "last":
        checkpoint_path = trainer.checkpoint_callback.last_model_path
    else:
        checkpoint_path = ckpt_path
    state_dict = load_full_checkpoint(model, checkpoint_path)
    logger.info(f"Loading ckpt from {checkpoint_path}")
    logger.info(model.load_state_dict(state_dict=state_dict, strict=False))
    for i in range(args.test_times):
        seed_current_process(args.seed + i)
        res = trainer.test(model=model, datamodule=data_module)[0]
        for k, v in res.items():
            results[k].append(v)
    statistics = {k: get_metric_statistics(v, args.test_times) for k, v in results.items()}
    test_result_in_text = f"Test results: {results}\nTest statistics with {args.test_times} replications: {statistics}"
    model.text_logger.log(test_result_in_text)
    latent_generation_config = {}
    answer_generation_config = {}
    all_config = getattr(model, "all_config", None)
    if all_config is not None:
        model_config = getattr(getattr(all_config, "model", None), "model_kwargs", None)
        if model_config is not None:
            latent_generation_config = model_config.get("latent_generation_config", {})
            answer_generation_config = model_config.get("answer_generation_config", {})
            if OmegaConf.is_config(latent_generation_config):
                latent_generation_config = OmegaConf.to_container(latent_generation_config, resolve=True)
            if OmegaConf.is_config(answer_generation_config):
                answer_generation_config = OmegaConf.to_container(answer_generation_config, resolve=True)
    model.sample_logs["test_metadata"] = {
        "model": args.model,
        "dataset": args.dataset,
        "ckpt_path": str(ckpt_path),
        "test_times": args.test_times,
        "seed": args.seed,
        "latent_generation_config": latent_generation_config,
        "answer_generation_config": answer_generation_config,
        "data_module": {
            "dataset_name": getattr(data_module, "dataset_name", None),
            "dataset_dir": str(getattr(data_module, "dataset_dir", "")),
            "test_file": getattr(data_module, "test_file", None),
        },
    }
    model.sample_logs["test_result"] = test_result_in_text
    model.json_logger.log(model.sample_logs)
    return results, statistics


def instantiate_callbacks(callback_configs: ListConfig):
    callbacks = []
    for callback_cfg in callback_configs:
        callbacks.append(instantiate_from_config(callback_cfg))

    return callbacks


def _preprocess_config(config, args, unknown_args):
    def set_config_key_value(inplace_dict, key_path, value):
        flag = False

        def bfs_set_config_key_value(inplace_dict, key, value):
            nonlocal flag
            if key in inplace_dict.keys():
                inplace_dict[key] = value
                flag = True
            for v in inplace_dict.values():
                if isinstance(v, (DictConfig, dict)):
                    bfs_set_config_key_value(inplace_dict=v, key=key, value=value)
                elif isinstance(v, ListConfig):
                    for item in v:
                        if isinstance(item, (DictConfig, dict)):
                            bfs_set_config_key_value(inplace_dict=item, key=key, value=value)

        keys = key_path.split(".")  # dataset.a.b = 1
        len_keys = len(keys)
        if len_keys == 1:
            bfs_set_config_key_value(inplace_dict, key=key_path, value=value)
            if flag:
                return
            else:
                raise ValueError(f"{key_path} is not found in config")

        for key_idx in range(len_keys - 1):  #
            inplace_dict = inplace_dict[keys[key_idx]]

            if isinstance(inplace_dict, ListConfig):
                for item in inplace_dict:
                    for sub_key_idx in range(key_idx + 1, len_keys - 1):
                        item = item[keys[sub_key_idx]]
                    item[keys[-1]] = value
                return

        inplace_dict[keys[-1]] = value

    is_test = False
    if p := args.test_ckpt_path:
        # load test model config
        config = OmegaConf.load(Path(p).parent.parent / "hparams.yaml").all_config
        is_test = True
    elif p := args.load_ckpt_path:
        # load pretrained ckpt config
        # config.model = OmegaConf.load(Path(p).parent.parent / 'hparams.yaml').all_config.model
        pass

    # set unknown args to config
    for unknown in unknown_args:
        k, v = unknown.split("=")
        v = v.strip("'")
        vlower = v.lower()
        if vlower == "none" or vlower == "~":
            v = None
        else:
            try:
                v = json.loads(vlower)
            except json.decoder.JSONDecodeError:
                pass  # v = v, the str itself
        set_config_key_value(config, k, v)

    # devices
    if (devices := args.devices) is not None:
        if devices == "all":
            devices = ",".join([str(i) for i in range(torch.cuda.device_count())])
        config.trainer.devices = [int(rank) for rank in devices.split(",")]

    if is_test:
        return config

    # ++ begin of training configuration ++#

    # set project name and signature for logging
    if args.no_log:
        config.trainer.logger = False
    else:
        log_root = os.environ.get("TRACE_LOG_ROOT")
        if log_root:
            config.trainer.logger.save_dir = str(Path(log_root) / args.model)
        else:
            config.trainer.logger.save_dir = f"logs/{args.model}"
        config.trainer.logger.name = f"{args.dataset}-{config.data_module.dataset_name}"
        config.trainer.logger.version = (
            start_time
            + "_"
            + str(random.randint(100000, 999999))
            + (f"_{args.log_suffix}" if args.log_suffix != "" else "")
        )

    # batch size for ddp
    total_bs = config.dataloader.batch_size
    num_devices = len(config.trainer.devices)
    bs_per_device = total_bs // num_devices
    real_bs = bs_per_device * num_devices
    if real_bs != total_bs:
        logger.warning(f"real batch size is {real_bs}")
    config.dataloader.batch_size = bs_per_device

    # epoch scaling
    epoch_scaling = config.data_module.get("epoch_scaling")
    if epoch_scaling is not None and epoch_scaling != 1:
        config.trainer.max_epochs = int(config.trainer.max_epochs / epoch_scaling)
        logger.info(
            f"Training epoch length is scaled by {epoch_scaling}, thus the num of epochs is decreased to {config.trainer.max_epochs}"
        )

    # customize anything here
    config = preprocess_config_hook(config)

    return config


def preprocess_config_hook(config):
    return config


def get_processed_args_and_config():
    args, unknown_args = get_args()

    OmegaConf.register_new_resolver("eval", ast.literal_eval)

    # load trainer config
    trainer_config = OmegaConf.load(f"src/configs/trainer/{args.trainer}.yaml")
    OmegaConf.resolve(trainer_config)

    # load model config
    model_config = OmegaConf.load(f"src/configs/models/{args.model}.yaml")
    OmegaConf.resolve(model_config)
    config = OmegaConf.merge(trainer_config, model_config)

    # load dataset config
    dataset_config = OmegaConf.load(f"src/configs/datasets/{args.dataset}.yaml")
    OmegaConf.resolve(dataset_config)
    config = OmegaConf.merge(config, DictConfig(dataset_config))

    config = _preprocess_config(config, args, unknown_args)
    if args.disable_early_stopping:
        config.callbacks = [
            callback
            for callback in config.callbacks
            if "EarlyStopping" not in str(callback.get("target", ""))
        ]

    # merge args into config
    config = OmegaConf.merge(
        config,
        OmegaConf.create({"args": vars(args), "unkown_args": {x.split("=")[0]: x.split("=")[1] for x in unknown_args}}),
    )

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        logger.info(f"running with config: {config}")

    return args, config


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="colar")

    parser.add_argument("--dataset", type=str, default="qsa")

    parser.add_argument("--trainer", type=str, default="default")

    parser.add_argument("--devices", type=str, default="0")

    parser.add_argument("--no_log", help="disable training log", action="store_true")

    parser.add_argument("--log_suffix", type=str, help="add suffix to log dir", default="")

    parser.add_argument("--resume_ckpt_path", type=str, help="resume training from ckpt", default=None)

    parser.add_argument("--load_ckpt_path", type=str, help="load ckpt as initialization", default=None)

    parser.add_argument("--workspace_path", type=str, help="assign the path of user workspace directory", default="/workspace/images-ks3-starfs/workspace/wenhui")

    parser.add_argument("--do_test", help="test after training", action="store_true")

    parser.add_argument(
        "--disable_early_stopping",
        help="run the configured training budget without an EarlyStopping callback",
        action="store_true",
    )

    parser.add_argument("--test_ckpt_path", default="")

    parser.add_argument("--test_times", type=int, default=5)

    parser.add_argument("--seed", type=int, default=0)

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main():
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and torch.cuda.is_available():
        local_rank_index = int(local_rank)
        if not 0 <= local_rank_index < torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank_index} is outside the visible devices"
            )
        # Lightning's subprocess launcher re-enters this file for every DDP
        # worker. Bind before seeding or model construction so nonzero ranks do
        # not create an avoidable CUDA context on local device 0.
        torch.cuda.set_device(local_rank_index)

    args, config = get_processed_args_and_config()

    seed_current_process(args.seed)
    install_rank_local_ddp_seed_reset()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_module: pl.LightningDataModule = instantiate_from_config(
        config.data_module, extra_kwargs={"all_config": config}
    )

    model: pl.LightningModule = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    if p := args.load_ckpt_path:
        logger.info(model.load_state_dict(state_dict=torch.load(p, map_location="cpu", weights_only=False)["state_dict"], strict=False))

    trainer: pl.Trainer = instantiate_from_config(
        config.trainer, extra_kwargs={"callbacks": instantiate_callbacks(config.callbacks)}
    )

    # test only
    if p := args.test_ckpt_path:
        print(do_test(model=model, trainer=trainer, ckpt_path=p, data_module=data_module, args=args))
        return

    # training
    try:
        if trainer.global_rank == 0:
            shutil.copytree("src", os.path.join(trainer.logger.log_dir, "src_backup"))  # backup src directory
    except AttributeError:
        pass
    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

    # test after training
    if args.do_test:
        print(do_test(model=model, trainer=trainer, ckpt_path="best", data_module=data_module, args=args))


if __name__ == "__main__":
    main()
