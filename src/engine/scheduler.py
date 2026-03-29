from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR


def build_scheduler(cfg: Any, optimizer: Optimizer) -> LRScheduler | None:
    if not cfg.scheduler.use:
        return None

    scheduler_name = cfg.scheduler.name.lower()

    if scheduler_name == "cosineannealinglr":
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.scheduler.t_max,
            eta_min=cfg.scheduler.eta_min,
        )

    if scheduler_name == "steplr":
        return StepLR(
            optimizer=optimizer,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
        )

    msg = f"Unsupported scheduler: {cfg.scheduler.name}"
    raise ValueError(msg)
