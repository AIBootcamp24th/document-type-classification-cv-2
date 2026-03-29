from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.optim as optim


def build_optimizer(cfg: Any, model: nn.Module) -> optim.Optimizer:
    optimizer_name = cfg.optimizer.name.lower()

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == "adamw":
        return optim.AdamW(
            trainable_parameters,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
        )

    if optimizer_name == "sgd":
        return optim.SGD(
            trainable_parameters,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            momentum=getattr(cfg.optimizer, "momentum", 0.9),
        )

    msg = f"Unsupported optimizer: {cfg.optimizer.name}"
    raise ValueError(msg)
