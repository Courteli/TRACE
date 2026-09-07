import os
import ast
import json
import random
import shutil
import argparse
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig
import torch
import lightning.pytorch as pl
import torch.distributed as dist

from src.utils.utils import instantiate_from_config, get_timestamp, get_metric_statistics
from src.utils.log import setup_logger 


logger = setup_logger(__name__)
start_time = get_timestamp()

NATIVE_MODEL_TARGET = "src.models.trace_role_native.LitTRACERoleNative"
NATIVE_STAGE0_TARGET = "src.models.cot.LitCot"


def _checkpoint_model_target(checkpoint):
    """Read the source model target from a Lightning checkpoint."""
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    all_config = hyper_parameters.get("all_config")
    if all_config is None:
        return None
    try:
        return all_config.model.target
    except AttributeError:
        try:
            return all_config["model"]["target"]
        except (KeyError, TypeError):
            return None


def _checkpoint_uses_trace_rl(checkpoint):
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    all_config = hyper_parameters.get("all_config")
    try:
        return bool(all_config.model.model_kwargs.get("do_trace_rl", False))
    except AttributeError:
        try:
            return bool(
                all_config["model"]["model_kwargs"].get("do_trace_rl", False)
            )
        except (KeyError, TypeError):
            return False


def _validated_native_initialization(path, config):
    """Fail closed unless a native stage consumes its own direct predecessor.

    The formal pipeline sets TRACE_NATIVE_RUN_ROOT to a fresh run directory.
    This rejects every historical TRACE/v7/v8/72 checkpoint by construction,
    even if its tensor names happen to be load-compatible.
    """
    if config.model.target != NATIVE_MODEL_TARGET:
        return torch.load(path, map_location="cpu", weights_only=False)

    allowed_root = os.environ.get("TRACE_NATIVE_RUN_ROOT")
    if not allowed_root:
        raise RuntimeError(
            "role-native training requires TRACE_NATIVE_RUN_ROOT; arbitrary "
            "checkpoint initialization is forbidden"
        )
    resolved_path = Path(path).expanduser().resolve(strict=True)
    resolved_root = Path(allowed_root).expanduser().resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            f"checkpoint {resolved_path} is outside this run: {resolved_root}"
        ) from exc

    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    source_target = _checkpoint_model_target(checkpoint)
    is_stage2 = bool(config.model.model_kwargs.get("do_trace_rl", False))
    expected_target = NATIVE_MODEL_TARGET if is_stage2 else NATIVE_STAGE0_TARGET
    expected_parent = resolved_root / ("stage1" if is_stage2 else "stage0")
    try:
        resolved_path.relative_to(expected_parent.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(
            f"checkpoint {resolved_path} is not inside the required direct "
            f"predecessor stage {expected_parent}"
        ) from exc
    if source_target != expected_target:
        stage = "Stage 2" if is_stage2 else "Stage 1"
        raise RuntimeError(
            f"{stage} must initialize from this run's direct predecessor "
            f"({expected_target}), got {source_target!r}"
        )
    if _checkpoint_uses_trace_rl(checkpoint):
        raise RuntimeError(
            "native initialization must come from a formation checkpoint, "
            "never from an earlier RL checkpoint"
        )
    return checkpoint


def _validated_native_resume(path, config):
    """Validate an in-stage recovery checkpoint from the same fresh run."""
    allowed_root = os.environ.get("TRACE_NATIVE_RUN_ROOT")
    if not allowed_root:
        raise RuntimeError(
            "formal recovery requires TRACE_NATIVE_RUN_ROOT; arbitrary "
            "resume checkpoints are forbidden"
        )
    resolved_path = Path(path).expanduser().resolve(strict=True)
    resolved_root = Path(allowed_root).expanduser().resolve(strict=True)
    target = str(config.model.target)
    do_rl = bool(config.model.model_kwargs.get("do_trace_rl", False))
    if target == NATIVE_STAGE0_TARGET:
        expected_stage = "stage0"
    elif target == NATIVE_MODEL_TARGET and not do_rl:
        expected_stage = "stage1"
    elif target == NATIVE_MODEL_TARGET and do_rl:
        expected_stage = "stage2"
    else:
        raise RuntimeError(f"unsupported formal recovery target: {target}")
    expected_root = (resolved_root / expected_stage).resolve(strict=True)
    try:
        resolved_path.relative_to(expected_root)
    except ValueError as exc:
        raise RuntimeError(
            f"resume checkpoint {resolved_path} is outside {expected_stage}"
        ) from exc
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    source_target = _checkpoint_model_target(checkpoint)
    source_rl = _checkpoint_uses_trace_rl(checkpoint)
    if source_target != target or source_rl != do_rl:
        raise RuntimeError(
            "resume checkpoint stage/model mismatch: "
            f"target={source_target}, do_trace_rl={source_rl}"
        )
    if not checkpoint.get("optimizer_states"):
        raise RuntimeError(
            "formal recovery checkpoint has no optimizer state; refusing an "
            "inexact weights-only continuation"
        )
    return checkpoint


def do_test(model: pl.LightningModule, trainer: pl.Trainer, ckpt_path: str, data_module: pl.LightningDataModule, args):
    results = defaultdict(list)
    if ckpt_path == "best":
        state_dict = torch.load(trainer.checkpoint_callback.best_model_path, weights_only=False)["state_dict"]
    elif ckpt_path == "last":
        state_dict = torch.load(trainer.checkpoint_callback.last_model_path, weights_only=False)["state_dict"]
    else:
        state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
        logger.info(f"Loading ckpt from {ckpt_path}")
    logger.info(model.load_state_dict(state_dict=state_dict, strict=False))
    for i in range(args.test_times):
        pl.seed_everything(args.seed + i)
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
        config.trainer.logger.save_dir = os.environ.get(
            "TRACE_NATIVE_LOG_ROOT", f"logs/{args.model}"
        )
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
    args, config = get_processed_args_and_config()

    if p := args.resume_ckpt_path:
        _validated_native_resume(p, config)

    pl.seed_everything(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_module: pl.LightningDataModule = instantiate_from_config(
        config.data_module, extra_kwargs={"all_config": config}
    )

    model: pl.LightningModule = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    if args.resume_ckpt_path:
        # ModelBase deliberately stores only trainable tensors.  Recovery is
        # exact because formal checkpoints also retain optimizer/scheduler and
        # loop states; non-trainable base-Qwen tensors are reloaded from base.
        model.strict_loading = False
    if p := args.load_ckpt_path:
        checkpoint = _validated_native_initialization(p, config)
        logger.info(
            model.load_state_dict(
                state_dict=checkpoint["state_dict"], strict=False
            )
        )
        if (
            config.model.target == NATIVE_MODEL_TARGET
            and bool(config.model.model_kwargs.get("do_trace_rl", False))
        ):
            model.initialize_stage2_reference_after_load()

    callbacks = instantiate_callbacks(config.callbacks)
    recovery_dir = os.environ.get("TRACE_NATIVE_RECOVERY_DIR")
    if recovery_dir and not args.test_ckpt_path:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=recovery_dir,
                filename="recovery-step{step:08d}",
                every_n_train_steps=int(
                    os.environ.get("TRACE_NATIVE_RECOVERY_EVERY_N_STEPS", "100")
                ),
                save_top_k=1,
                save_last=True,
                save_weights_only=False,
                monitor=None,
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )
    trainer: pl.Trainer = instantiate_from_config(
        config.trainer, extra_kwargs={"callbacks": callbacks}
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
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.resume_ckpt_path,
        # The path has already passed same-run/stage/optimizer validation.
        # PyTorch >=2.6 otherwise defaults to weights_only=True and refuses
        # Lightning's trusted OmegaConf/optimizer recovery payload.
        weights_only=False if args.resume_ckpt_path else None,
    )

    # test after training
    if args.do_test:
        print(do_test(model=model, trainer=trainer, ckpt_path="best", data_module=data_module, args=args))


if __name__ == "__main__":
    main()
