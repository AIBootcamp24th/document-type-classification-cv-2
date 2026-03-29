from __future__ import annotations

import csv
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

from src.config import load_config, parse_args
from src.dataset.loader import build_train_valid_loaders
from src.engine.loss import build_loss_fn
from src.engine.optimizer import build_optimizer
from src.engine.scheduler import build_scheduler
from src.engine.trainer import train_one_epoch, valid_one_epoch
from src.models.model_factory import build_model
from src.utils.logger import setup_logger


def get_device(device_config: str) -> torch.device:
    if device_config == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA is not available, but cfg.runtime.device is set to 'cuda'."
            raise RuntimeError(msg)
        return torch.device("cuda")

    if device_config == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def should_log_epoch(epoch: int, total_epochs: int, log_interval: int) -> bool:
    return epoch == 1 or epoch % log_interval == 0 or epoch == total_epochs


def is_metric_improved(
    current_score: float,
    best_score: float,
    mode: str,
) -> bool:
    if mode == "max":
        return current_score > best_score
    if mode == "min":
        return current_score < best_score

    msg = f"Unsupported early_stopping mode: {mode}"
    raise ValueError(msg)


def main() -> None:
    load_dotenv(dotenv_path=Path(".env"), override=True)
    args = parse_args()

    cfg = load_config(
        base_path=args.base,
        data_path=args.data,
        train_path=args.train,
        inference_path=args.inference,
        model_path=args.model,
    )

    experiment_root = Path(args.train).resolve().parents[1]

    cfg.paths.output_dir = str(experiment_root / "outputs")
    cfg.paths.checkpoint_dir = str(experiment_root / "outputs" / "checkpoints")
    cfg.paths.log_dir = str(experiment_root / "outputs" / "logs")
    cfg.inference.checkpoint_path = str(
        experiment_root / "outputs" / "checkpoints" / "best.pt"
    )

    logger = setup_logger("src.train", cfg.paths.log_dir)

    metrics_path = Path(cfg.paths.output_dir) / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "train_f1_micro",
            "train_f1_macro",
            "valid_loss",
            "valid_accuracy",
            "valid_f1_micro",
            "valid_f1_macro",
        ])

    device = get_device(cfg.runtime.device)

    train_loader, valid_loader = build_train_valid_loaders(cfg)

    model = build_model(cfg).to(device)
    criterion = build_loss_fn(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    use_wandb = cfg.logging.use_wandb
    if use_wandb:
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            dir=cfg.paths.output_dir,
            config={
                "model_name": cfg.model.name,
                "epochs": cfg.train.epochs,
                "train_batch_size": cfg.train.train_batch_size,
                "valid_batch_size": cfg.train.valid_batch_size,
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
                "image_size": cfg.image.size,
                "device": str(device),
            },
        )

    early_stopping_cfg = cfg.early_stopping
    use_early_stopping = early_stopping_cfg.use
    early_stopping_mode = early_stopping_cfg.mode
    early_stopping_patience = early_stopping_cfg.patience
    early_stopping_monitor = early_stopping_cfg.monitor

    if early_stopping_mode == "max":
        best_score = float("-inf")
    elif early_stopping_mode == "min":
        best_score = float("inf")
    else:
        msg = f"Unsupported early_stopping mode: {early_stopping_mode}"
        raise ValueError(msg)

    best_model_path = Path(cfg.paths.checkpoint_dir) / "best.pt"
    log_interval = max(1, int(cfg.logging.log_interval))
    early_stopping_counter = 0

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(
            cfg=cfg,
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_metrics = valid_one_epoch(
            cfg=cfg,
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_metrics["f1_micro"],
                train_metrics["f1_macro"],
                valid_metrics["loss"],
                valid_metrics["accuracy"],
                valid_metrics["f1_micro"],
                valid_metrics["f1_macro"],
            ])

        if should_log_epoch(
            epoch=epoch,
            total_epochs=cfg.train.epochs,
            log_interval=log_interval,
        ):
            logger.info(
                f"[Epoch {epoch}/{cfg.train.epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_{cfg.metric.primary}={train_metrics['primary_score']:.4f} "
                f"valid_loss={valid_metrics['loss']:.4f} "
                f"valid_{cfg.metric.primary}={valid_metrics['primary_score']:.4f}"
            )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "train/f1_micro": train_metrics["f1_micro"],
                "train/f1_macro": train_metrics["f1_macro"],
                "valid/loss": valid_metrics["loss"],
                "valid/accuracy": valid_metrics["accuracy"],
                "valid/f1_micro": valid_metrics["f1_micro"],
                "valid/f1_macro": valid_metrics["f1_macro"],
            })

        monitored_score = valid_metrics[early_stopping_monitor]
        improved = is_metric_improved(
            current_score=monitored_score,
            best_score=best_score,
            mode=early_stopping_mode,
        )

        if improved:
            best_score = monitored_score
            early_stopping_counter = 0
            save_checkpoint(model, best_model_path)
            logger.info(
                f"best model saved | path={best_model_path} "
                f"| {early_stopping_monitor}={best_score:.4f}"
            )
        else:
            early_stopping_counter += 1

            if use_early_stopping:
                logger.info(
                    f"early stopping counter | "
                    f"patience={early_stopping_counter}/{early_stopping_patience}"
                )

                if early_stopping_counter >= early_stopping_patience:
                    logger.info(
                        f"early stopping triggered | epoch={epoch} "
                        f"| best_{early_stopping_monitor}={best_score:.4f}"
                    )
                    break

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
