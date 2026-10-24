import argparse
import math
from pathlib import Path

import yaml


KINDS = {
    str: "dataset data_root cache_dir output_dir device model_name pretrained "
    "precision optimizer checkpoint",
    int: "seed num_workers img_height img_width batch_size fold_index epochs "
    "accum_steps early_stop_patience early_stop_start_epoch max_steps",
    float: "lr weight_decay warmup_ratio grad_clip_norm sgd_momentum ema_decay "
    "early_stop_min_delta",
    bool: "grad_checkpointing sgd_nesterov img_aug img_hflip multi_positive ema",
}
TYPES = {name: kind for kind, names in KINDS.items() for name in names.split()}
CHOICES = {
    "dataset": ("cuhk-pedes", "icfg-pedes", "rstpreid"),
    "precision": ("fp32", "amp_fp16"),
    "optimizer": ("adamw", "sgd"),
    "fold_index": range(5),
}
NULLABLE = {"data_root", "cache_dir", "fold_index", "checkpoint"}
EXCLUDED = {
    "train": {"checkpoint"},
    "evaluate": set(
        "fold_index epochs accum_steps lr weight_decay warmup_ratio grad_clip_norm "
        "optimizer sgd_momentum sgd_nesterov img_aug img_hflip multi_positive ema "
        "ema_decay early_stop_min_delta early_stop_patience early_stop_start_epoch "
        "max_steps".split()
    ),
}
POSITIVE = (
    "batch_size",
    "img_height",
    "img_width",
    "epochs",
    "accum_steps",
    "early_stop_patience",
    "early_stop_start_epoch",
)
NONNEGATIVE = ("num_workers", "max_steps", "seed")
FINITE = (
    "lr",
    "weight_decay",
    "warmup_ratio",
    "grad_clip_norm",
    "ema_decay",
    "early_stop_min_delta",
    "sgd_momentum",
)


def build_parser(mode="train"):
    parser = argparse.ArgumentParser(description=f"3T-CLIP {mode}")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="YAML overrides of configs/config.yaml",
    )
    return parser


def parse_args(mode="train", argv=None):
    parser = build_parser(mode)
    overrides = parser.parse_args(argv)
    if len(overrides.config) != 1:
        parser.error("--config accepts exactly one file")
    overrides.config = overrides.config[0]
    base = Path(__file__).resolve().with_name("config.yaml")
    if not base.is_file():
        base = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    merged = {}
    for filename in (base, overrides.config):
        try:
            values = yaml.safe_load(Path(filename).read_text())
        except (OSError, yaml.YAMLError) as error:
            parser.error(f"cannot load configuration {filename}: {error}")
        if values is None:
            values = {}
        if not isinstance(values, dict):
            parser.error("configuration must be a mapping")
        if filename == base and values.keys() != TYPES.keys():
            parser.error("base configuration must contain all supported parameters")
        for key, value in values.items():
            if key not in TYPES:
                parser.error(f"unknown configuration parameter: {key}")
            kind, choices = TYPES[key], CHOICES.get(key)
            if value is not None:
                valid = type(value) is kind or kind is float and type(value) is int
                if not valid or choices is not None and value not in choices:
                    parser.error(f"invalid configuration value for {key}: {value!r}")
                value = kind(value)
            elif key not in NULLABLE:
                parser.error(f"{key} cannot be null")
            merged[key] = value
    args = argparse.Namespace(
        **{key: value for key, value in merged.items() if key not in EXCLUDED[mode]}
    )
    args.config = overrides.config
    required = ["data_root"] + (["checkpoint"] if mode == "evaluate" else [])
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"{name} must be set in the YAML config")
    if args.device != "cuda" and not (
        args.device.startswith("cuda:")
        and args.device[5:].isascii()
        and args.device[5:].isdigit()
    ):
        parser.error("device must be cuda or cuda:N with a nonnegative device index")
    for name in POSITIVE:
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    for name in NONNEGATIVE:
        if hasattr(args, name) and getattr(args, name) < 0:
            parser.error(f"{name} cannot be negative")
    for name in FINITE:
        if hasattr(args, name):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0:
                parser.error(f"{name} must be finite and nonnegative")
    if mode != "evaluate":
        if not 0 <= args.warmup_ratio <= 1 or not 0 < args.ema_decay < 1:
            parser.error("warmup_ratio must be in [0, 1] and ema_decay in (0, 1)")
        if (
            not 0 <= args.sgd_momentum < 1
            or args.optimizer == "sgd"
            and args.sgd_nesterov
            and args.sgd_momentum == 0
        ):
            parser.error("invalid SGD momentum for selected settings")
        if args.dataset == "cuhk-pedes" and args.fold_index is not None:
            parser.error("native validation does not accept fold_index")
    return args


def train_main(argv=None):
    args = parse_args("train", argv)
    from t3clip.train import run

    return run(args)


def eval_main(argv=None):
    args = parse_args("evaluate", argv)
    from t3clip.evaluate import run

    return run(args)
