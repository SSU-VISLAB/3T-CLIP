import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm

from t3clip.cli import parse_args
from t3clip.data import (
    build_transform,
    load_records,
    make_eval_loaders,
    make_loader,
    split_records,
)
from t3clip.engine import (
    EMA,
    Selection,
    autocast,
    forward_loss,
    make_optimizer,
    make_scheduler,
    optimizer_step,
    seed_everything,
)
from t3clip.metrics import evaluate
from t3clip.model import create_model


class Timing:
    def __init__(self):
        self.train_seconds = 0.0
        self.wall_seconds = 0.0

    def start_wall(self):
        self.wall_started = time.perf_counter()

    def stop_wall(self):
        self.wall_seconds += time.perf_counter() - self.wall_started

    def start(self, device):
        torch.cuda.synchronize(device)
        self.training_started = time.perf_counter()

    def stop(self, device):
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - self.training_started
        self.train_seconds += seconds
        return seconds

    def metrics(self):
        return {
            "train_gpu_hours": self.train_seconds / 3600.0,
            "wall_hours": self.wall_seconds / 3600.0,
        }

    def write(self, output, epoch, step, seconds, complete):
        with (output / "timing.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step,
                        "complete_epoch": complete,
                        "epoch_train_seconds": seconds,
                        "train_seconds": self.train_seconds,
                        **self.metrics(),
                    }
                )
                + "\n"
            )


def save_checkpoint(path, model, ema, args):
    payload = {
        "version": 1,
        "args": vars(args),
        "model": ema.model_state_dict(model) if ema else model.state_dict(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit(args, model, train_loader, image_loader=None, text_loader=None):
    timing = Timing()
    device = torch.device(args.device)
    with autocast(device, args.precision):
        pass
    if not len(train_loader):
        raise ValueError("training loader is empty; reduce batch size")
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "timing.jsonl").write_text("")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, len(train_loader), args)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "amp_fp16")
    ema = EMA(model, args.ema_decay) if args.ema else None
    selection = Selection()
    global_step = 0
    with (output / "steps.jsonl").open("w", buffering=1) as log:
        for epoch in range(args.epochs):
            timing.start_wall()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            timing.start(device)
            iterator = iter(train_loader)
            for batch_index, batch in enumerate(
                tqdm(
                    iterator,
                    total=len(train_loader),
                    desc=f"epoch {epoch + 1}",
                ),
            ):
                batch = {
                    key: value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }
                with autocast(device, args.precision):
                    loss = forward_loss(model, batch, args.multi_positive)
                    raw_loss = loss.detach().item()
                    if not math.isfinite(raw_loss):
                        raise FloatingPointError("non-finite contrastive loss")
                    group_start = batch_index // args.accum_steps * args.accum_steps
                    group_size = min(
                        args.accum_steps, len(train_loader) - group_start
                    )
                    scaled_loss = loss / group_size
                scaler.scale(scaled_loss).backward()
                boundary = (
                    batch_index + 1
                ) % args.accum_steps == 0 or batch_index + 1 == len(train_loader)
                if not boundary:
                    continue
                success = optimizer_step(
                    model, optimizer, scheduler, scaler, args.grad_clip_norm, ema
                )
                if not success:
                    continue
                global_step += 1
                metrics = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": raw_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_scale": scaler.get_scale(),
                    "peak_allocated": torch.cuda.max_memory_allocated(device),
                    "peak_reserved": torch.cuda.max_memory_reserved(device),
                }
                log.write(json.dumps(metrics, allow_nan=False) + "\n")
                if args.max_steps and global_step >= args.max_steps:
                    seconds = timing.stop(device)
                    timing.stop_wall()
                    timing.write(
                        output,
                        epoch + 1,
                        global_step,
                        seconds,
                        batch_index + 1 == len(train_loader),
                    )
                    save_checkpoint(
                        output / "last.pt",
                        model,
                        ema,
                        args,
                    )
                    return selection.best_r1
            seconds = timing.stop(device)
            if image_loader is None or text_loader is None:
                raise ValueError("epoch validation requires image and text loaders")
            torch.cuda.reset_peak_memory_stats(device)
            with ema.average_parameters(model) if ema else nullcontext(model):
                metrics = evaluate(
                    model, image_loader, text_loader, device, args.precision
                )
            torch.cuda.synchronize(device)
            torch.cuda.max_memory_allocated(device)
            torch.cuda.max_memory_reserved(device)
            timing.stop_wall()
            timing.write(output, epoch + 1, global_step, seconds, True)
            best, stop = selection.observe(
                metrics["t2i_r1"],
                epoch + 1,
                args.early_stop_min_delta,
                args.early_stop_patience,
                args.early_stop_start_epoch,
            )
            metrics.update(timing.metrics())
            with (output / "validation.jsonl").open("a") as validation:
                validation.write(json.dumps({"epoch": epoch + 1, **metrics}) + "\n")
            print(json.dumps({"epoch": epoch + 1, **metrics}))
            for name in ["best.pt", "last.pt"] if best else ["last.pt"]:
                save_checkpoint(
                    output / name,
                    model,
                    ema,
                    args,
                )
            if stop:
                break
    return selection.best_r1


def run(args):
    if not args.data_root:
        raise ValueError("data_root in the YAML config is required")
    if args.dataset != "cuhk-pedes" and args.fold_index is None:
        raise ValueError("fold_index in the YAML config is required for five-fold datasets")
    with autocast(args.device, args.precision):
        pass
    seed_everything(args.seed, args.device)
    model, tokenizer = create_model(args, torch.device(args.device))
    training, validation = split_records(
        load_records(args.dataset, args.data_root), args.dataset, args.fold_index
    )
    if not training or not validation:
        raise ValueError("training and validation splits must be nonempty")
    train_transform = build_transform(
        args.img_height, args.img_width, True, args.img_aug, args.img_hflip
    )
    eval_transform = build_transform(args.img_height, args.img_width)
    train_loader = make_loader(
        training,
        tokenizer,
        train_transform,
        args.batch_size,
        args.num_workers,
        args.device,
    )
    image_loader, text_loader = make_eval_loaders(
        validation,
        tokenizer,
        eval_transform,
        args.batch_size,
        args.num_workers,
        args.device,
    )
    return fit(args, model, train_loader, image_loader, text_loader)


def main(argv=None):
    run(parse_args("train", argv))


if __name__ == "__main__":
    main()
