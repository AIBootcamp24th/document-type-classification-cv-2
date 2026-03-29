from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_loss_fn(cfg: Any) -> nn.Module:
    loss_name = cfg.loss.name.lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(
            label_smoothing=cfg.loss.label_smoothing,
        )

    msg = f"Unsupported loss function: {cfg.loss.name}"
    raise ValueError(msg)
