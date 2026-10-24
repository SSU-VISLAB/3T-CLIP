import math
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device, precision):
    device = torch.device(device)
    if precision == "fp32":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.float16)


def contrastive_loss(logits, identities, multi_positive=True):
    if not multi_positive:
        labels = torch.arange(len(identities), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    positives = identities[:, None].eq(identities[None, :])
    counts = positives.sum(dim=1)
    directions = [
        -(F.log_softmax(scores, dim=1) * positives).sum(dim=1) / counts
        for scores in (logits, logits.T)
    ]
    return (directions[0].mean() + directions[1].mean()) * 0.5


def forward_loss(model, batch, multi_positive):
    images = F.normalize(model.encode_image(batch["images"]), dim=-1)
    texts = F.normalize(model.encode_text(batch["text_tokens"]), dim=-1)
    logits = model.logit_scale.exp().clamp(max=100) * images @ texts.T
    return contrastive_loss(logits, batch["person_ids"], multi_positive)


def make_optimizer(model, args):
    buckets = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            exempt = (
                parameter.ndim < 2 or name.endswith(".bias") or "logit_scale" in name
            )
            buckets.setdefault(exempt, []).append(parameter)
    groups = [
        {
            "params": parameters,
            "lr": args.lr,
            "weight_decay": 0.0 if exempt else args.weight_decay,
        }
        for exempt, parameters in buckets.items()
    ]
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            groups, lr=args.lr, momentum=args.sgd_momentum, nesterov=args.sgd_nesterov
        )
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)


def lr_multiplier(step, warmup_steps, total_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return (1 + math.cos(math.pi * min(1.0, progress))) / 2


def make_scheduler(optimizer, loader_length, args):
    total = math.ceil(loader_length / args.accum_steps) * args.epochs
    warmup = int(total * args.warmup_ratio)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, warmup, total)
    )


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    parameter.detach(), alpha=1 - self.decay
                )

    @contextmanager
    def average_parameters(self, model):
        raw = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name in self.shadow
        }
        try:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in self.shadow:
                        p.copy_(self.shadow[name])
            yield model
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in raw:
                        p.copy_(raw[name])

    def model_state_dict(self, model):
        result = dict(model.state_dict())
        canonical = {id(p): name for name, p in model.named_parameters()}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            key = canonical[id(parameter)]
            if key in self.shadow:
                result[name] = self.shadow[key]
        return result


def optimizer_step(model, optimizer, scheduler, scaler, clip_norm, ema=None):
    if clip_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    success = scaler.get_scale() >= before
    if success:
        scheduler.step()
        if ema is not None:
            ema.update(model)
    optimizer.zero_grad(set_to_none=True)
    return success


@dataclass
class Selection:
    best_r1: float = -math.inf
    plateau_best: float = -math.inf
    bad_epochs: int = 0

    def observe(self, r1, epoch, min_delta, patience, start_epoch):
        best = r1 > self.best_r1
        if best:
            self.best_r1 = r1
        if not math.isfinite(self.plateau_best) or r1 > self.plateau_best + min_delta:
            self.plateau_best = r1
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return best, epoch >= start_epoch and self.bad_epochs >= patience
