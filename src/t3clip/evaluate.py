import json
from pathlib import Path

import torch

from t3clip.cli import parse_args
from t3clip.data import build_transform, load_records, make_eval_loaders
from t3clip.engine import autocast, seed_everything
from t3clip.metrics import evaluate
from t3clip.model import create_model


def run(args):
    if not args.checkpoint or not args.data_root:
        raise ValueError("checkpoint and data_root in the YAML config are required")
    device = torch.device(args.device)
    with autocast(device, args.precision):
        pass
    seed_everything(args.seed, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != 1:
        raise ValueError("unsupported checkpoint format")
    for key in ("model_name", "img_height", "img_width", "dataset"):
        if getattr(args, key) != checkpoint["args"][key]:
            raise ValueError(
                f"checkpoint configuration differs for {key}; supply matching YAML settings"
            )
    args.pretrained = checkpoint["args"]["pretrained"]
    model, tokenizer = create_model(args, device, load_pretrained=False)
    model.load_state_dict(checkpoint["model"])
    records = [
        r for r in load_records(args.dataset, args.data_root) if r.split == "test"
    ]
    loaders = make_eval_loaders(
        records,
        tokenizer,
        build_transform(args.img_height, args.img_width),
        args.batch_size,
        args.num_workers,
        device,
    )
    metrics = evaluate(model, *loaders, device, args.precision)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None):
    return run(parse_args("evaluate", argv))


if __name__ == "__main__":
    main()
