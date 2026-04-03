from __future__ import annotations

import sys
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metric import compute_metrics, get_primary_metric


def train_one_epoch(
    cfg: Any,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    progress_bar = tqdm(
        loader,
        desc="train",
        leave=False,
        disable=not cfg.logging.verbose,
        file=sys.stdout,
        dynamic_ncols=True,
    )

    for batch in progress_bar:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = epoch_loss
    metrics["primary_score"] = get_primary_metric(cfg, metrics)

    return metrics


@torch.no_grad()
def valid_one_epoch(
    cfg: Any,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    running_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    progress_bar = tqdm(
        loader,
        desc="validation",
        leave=False,
        disable=not cfg.logging.verbose,
        file=sys.stdout,
        dynamic_ncols=True,
    )

    for batch in progress_bar:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = epoch_loss
    metrics["primary_score"] = get_primary_metric(cfg, metrics)

    return metrics
