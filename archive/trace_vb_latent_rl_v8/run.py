import os
import ast
import hashlib
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
from src.utils.safe_checkpoint import (
    install_safe_checkpoint_globals,
    safe_load_checkpoint,
)


logger = setup_logger(__name__)
start_time = get_timestamp()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_execution_mode_args(args) -> None:
    """Reject ambiguous validation, resume, rewind, and test combinations."""
    if args.validate_only and (args.do_test or args.test_ckpt_path):
        raise ValueError(
            "--validate_only cannot be combined with any test option"
        )
    if args.validate_only and args.resume_ckpt_path:
        raise ValueError(
            "validate-only requires --load_ckpt_path so the v8 rewind and "
            "provenance contracts can be checked before validation"
        )
    if args.rewind_path_to_capability and args.resume_ckpt_path:
        raise ValueError(
            "resume must restore an already-rewound v8 checkpoint and may "
            "not request another capability rewind"
        )
    if args.rewind_path_to_capability and not args.load_ckpt_path:
        raise ValueError(
            "--rewind_path_to_capability requires --load_ckpt_path"
        )
    if args.rewind_path_to_capability and not (
        args.registered_capability_ckpt_path
    ):
        raise ValueError(
            "--rewind_path_to_capability requires "
            "--registered_capability_ckpt_path"
        )
    if (
        args.registered_capability_ckpt_path
        and not args.rewind_path_to_capability
    ):
        raise ValueError(
            "--registered_capability_ckpt_path is permitted only for the "
            "explicit one-time v7 rewind"
        )
    if args.save_validation_checkpoint and not args.validate_only:
        raise ValueError(
            "--save_validation_checkpoint is restricted to --validate_only"
        )


def validate_rewind_source_request(
    source_schema: str,
    *,
    rewind_requested: bool,
) -> None:
    """Allow the mutating rewind only for an explicitly authorized v7 load."""
    source_schema = str(source_schema)
    if source_schema == "trace_vb_v7" and not rewind_requested:
        raise RuntimeError(
            "v7 weights-only initialization requires explicit "
            "--rewind_path_to_capability"
        )
    if source_schema == "trace_vb_v8" and rewind_requested:
        raise RuntimeError(
            "v8 weights-only loading forbids a second capability rewind"
        )


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
    checkpoint = safe_load_checkpoint(path, map_location="cpu")
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

    parser.add_argument(
        "--rewind_path_to_capability",
        help=(
            "explicitly authorize the one-time v7 capability-to-deployment "
            "rewind; forbidden for v8 checkpoints and resume"
        ),
        action="store_true",
    )

    parser.add_argument(
        "--registered_capability_ckpt_path",
        type=str,
        default=None,
        help=(
            "independent registered legacy capability checkpoint; required "
            "only for the explicit v7-to-v8 rewind"
        ),
    )

    parser.add_argument(
        "--cot_encoder_ckpt_path",
        type=str,
        help=(
            "load a registered Stage-0 checkpoint only into TRACE-VB's "
            "frozen CoT encoder adapter"
        ),
        default=None,
    )

    parser.add_argument("--workspace_path", type=str, help="assign the path of user workspace directory", default="/workspace/images-ks3-starfs/workspace/wenhui")

    parser.add_argument("--do_test", help="test after training", action="store_true")

    parser.add_argument(
        "--validate_only",
        help="run the validation split only; never read the test split",
        action="store_true",
    )

    parser.add_argument(
        "--save_validation_checkpoint",
        "--save_checkpoint_path",
        dest="save_validation_checkpoint",
        type=str,
        default=None,
        help=(
            "after validate-only succeeds, save the parity-checked weights "
            "checkpoint to this explicit path"
        ),
    )

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

    validate_execution_mode_args(args)

    seed_current_process(args.seed)
    install_rank_local_ddp_seed_reset()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    data_module: pl.LightningDataModule = instantiate_from_config(
        config.data_module, extra_kwargs={"all_config": config}
    )

    model: pl.LightningModule = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    if p := args.load_ckpt_path:
        checkpoint = safe_load_checkpoint(p, map_location="cpu")
        source_schema = str(
            checkpoint.get("trace_vb_schema_version", "")
        )
        validate_rewind_source_request(
            source_schema,
            rewind_requested=args.rewind_path_to_capability,
        )
        # Restore/reset checkpoint-level metadata first. For a v7 source the
        # independent registered payload is deliberately registered only
        # after this reset, so it cannot be accidentally cleared before the
        # one-time rewind. A v8 source restores its persisted provenance here.
        model.on_load_checkpoint(checkpoint)
        if hasattr(model, "validate_v8_warm_start_coverage"):
            coverage_report = model.validate_v8_warm_start_coverage(
                checkpoint["state_dict"]
            )
            logger.info("v8 warm-start coverage: %s", coverage_report)
        if source_schema == "trace_vb_v7":
            registered_path = Path(
                str(args.registered_capability_ckpt_path)
            ).resolve()
            expected_registered_path = Path(
                str(
                    model.trace_config.get(
                        "registered_capability_checkpoint_path", ""
                    )
                )
            ).resolve()
            if registered_path != expected_registered_path:
                raise RuntimeError(
                    "registered capability path mismatch: expected "
                    f"{expected_registered_path}, found {registered_path}"
                )
            registered_sha = sha256_file(registered_path)
            expected_registered_sha = str(
                model.trace_config.get(
                    "registered_capability_checkpoint_sha256", ""
                )
            ).strip().lower()
            if registered_sha != expected_registered_sha:
                raise RuntimeError(
                    "registered capability SHA256 mismatch: expected "
                    f"{expected_registered_sha}, found {registered_sha}"
                )
            v7_source_sha = sha256_file(Path(str(p)))
            registered_checkpoint = safe_load_checkpoint(
                str(registered_path), map_location="cpu"
            )
            provenance_report = model.register_v8_capability_provenance(
                v7_source_state=checkpoint["state_dict"],
                v7_source_checkpoint_sha256=v7_source_sha,
                registered_capability_state=(
                    registered_checkpoint["state_dict"]
                ),
                registered_capability_checkpoint_sha256=registered_sha,
            )
            logger.info(
                "v8 registered capability provenance: %s",
                provenance_report,
            )
            del registered_checkpoint
        # Load tensors only after provenance has been restored (v8) or
        # independently proven and registered (v7).
        incompatible = model.load_state_dict(
            state_dict=checkpoint["state_dict"], strict=False
        )
        logger.info(
            "checkpoint migration completed: %d missing keys, %d unexpected "
            "keys (fail-closed capability coverage is checked by the model)",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        if hasattr(model, "initialize_v8_capability_spine"):
            rewind_report = model.initialize_v8_capability_spine(
                checkpoint,
                allow_v7_rewind=args.rewind_path_to_capability,
            )
            logger.info("v8 capability spine: %s", rewind_report)
    if p := args.cot_encoder_ckpt_path:
        if not hasattr(model, "load_cot_encoder_state_dict"):
            raise RuntimeError(
                "--cot_encoder_ckpt_path requires a model with an isolated "
                "CoT encoder adapter"
            )
        cot_path_obj = Path(str(p))
        actual_cot_sha = sha256_file(cot_path_obj)
        expected_cot_sha = str(
            getattr(model, "trace_config", {}).get(
                "stage0_cot_encoder_checkpoint_sha256", ""
            )
        ).strip().lower()
        if actual_cot_sha != expected_cot_sha:
            raise RuntimeError(
                "registered Stage-0 CoT encoder SHA256 mismatch: expected "
                f"{expected_cot_sha}, found {actual_cot_sha}"
            )
        cot_checkpoint = safe_load_checkpoint(p, map_location="cpu")
        model.load_cot_encoder_state_dict(
            cot_checkpoint["state_dict"],
            checkpoint_sha256=actual_cot_sha,
        )
        logger.info(
            "registered Stage-0 CoT encoder migration completed"
        )

    if args.load_ckpt_path and hasattr(
        model, "assert_v8_initialization_contract"
    ):
        initialization_report = model.assert_v8_initialization_contract()
        logger.info(
            "v8 fail-closed initialization contract: %s",
            initialization_report,
        )

    trainer: pl.Trainer = instantiate_from_config(
        config.trainer, extra_kwargs={"callbacks": instantiate_callbacks(config.callbacks)}
    )

    # test only
    if p := args.test_ckpt_path:
        print(do_test(model=model, trainer=trainer, ckpt_path=p, data_module=data_module, args=args))
        return

    if args.validate_only:
        install_safe_checkpoint_globals()
        validation_results = trainer.validate(
            model=model,
            datamodule=data_module,
            verbose=True,
        )
        if args.save_validation_checkpoint:
            save_path = Path(args.save_validation_checkpoint).resolve()
            if trainer.is_global_zero:
                save_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.strategy.barrier("v8_validate_only_checkpoint_parent")
            # This is a selection candidate, not an optimizer-resume point.
            trainer.save_checkpoint(str(save_path), weights_only=True)
            logger.info(
                "saved parity-checked validate-only checkpoint to %s",
                save_path,
            )
        if trainer.is_global_zero:
            print(validation_results)
        return

    # training
    try:
        if trainer.global_rank == 0:
            shutil.copytree("src", os.path.join(trainer.logger.log_dir, "src_backup"))  # backup src directory
    except AttributeError:
        pass
    # ``safe_load_checkpoint`` and Lightning both decode the same data-only
    # OmegaConf metadata. Reinstall immediately before Lightning's internal
    # full-state restore and require restricted unpickling explicitly.
    install_safe_checkpoint_globals()
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.resume_ckpt_path,
        weights_only=True,
    )

    # test after training
    if args.do_test:
        print(do_test(model=model, trainer=trainer, ckpt_path="best", data_module=data_module, args=args))


if __name__ == "__main__":
    main()
