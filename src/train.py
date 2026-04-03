from __future__ import annotations

import csv
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

from src.config import load_config, parse_args
from src.dataset.loader import (
    build_kfold_splits,
    build_train_valid_loaders,
    build_train_valid_loaders_for_fold,
    load_train_dataframe,
)
from src.engine.loss import build_loss_fn
from src.engine.optimizer import build_optimizer
from src.engine.scheduler import build_scheduler
from src.engine.trainer import train_one_epoch, valid_one_epoch
from src.models.model_factory import build_model
from src.utils.logger import setup_logger


def get_device(device_config: str) -> torch.device:
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_config == "cuda":
        msg = "runtime.device is set to 'cuda', but CUDA is not available."
        raise RuntimeError(msg)

    if device_config == "mps":
        if not torch.backends.mps.is_available():
            msg = "runtime.device is set to 'mps', but MPS is not available."
            raise RuntimeError(msg)
        return torch.device("mps")

    if device_config == "cpu":
        return torch.device("cpu")

    msg = f"Unsupported runtime.device: {device_config}"
    raise ValueError(msg)


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


def get_experiment_name(cfg) -> str:
    experiment = getattr(cfg, "experiment", None)
    experiment_name = getattr(experiment, "name", None)

    if experiment_name:
        return experiment_name

    return cfg.model.name


def resolve_execution_root(*path_candidates: str | Path) -> Path:
    for path_candidate in path_candidates:
        path = Path(path_candidate).resolve()
        config_dir = path.parent if path.is_file() else path

        if (
            config_dir.name == "configs"
            and config_dir.parent.parent.name == "experiments"
        ):
            return config_dir.parent

    return Path.cwd()


def initialize_output_paths(cfg, project_root: Path) -> None:
    experiment_name = get_experiment_name(cfg)
    output_root = project_root / "outputs" / experiment_name

    cfg.paths.output_dir = str(output_root)
    cfg.paths.checkpoint_dir = str(output_root / "checkpoints")
    cfg.paths.log_dir = str(output_root / "logs")
    cfg.inference.checkpoint_path = str(output_root / "checkpoints" / "best.pt")


def write_metrics_header(metrics_path: Path) -> None:
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


def append_metrics_row(
    metrics_path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
) -> None:
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


def build_wandb_config(cfg, device: torch.device) -> dict:
    return {
        "experiment_name": get_experiment_name(cfg),
        "model_name": cfg.model.name,
        "epochs": cfg.train.epochs,
        "train_batch_size": cfg.train.train_batch_size,
        "valid_batch_size": cfg.train.valid_batch_size,
        "lr": cfg.optimizer.lr,
        "weight_decay": cfg.optimizer.weight_decay,
        "image_size": cfg.image.size,
        "device": str(device),
        "split_method": cfg.split.method,
        "n_splits": getattr(cfg.split, "n_splits", None),
        "primary_metric": cfg.metric.primary,
    }


def init_wandb_run(
    cfg,
    run_name: str,
    group_name: str | None,
    run_dir: str,
    device: torch.device,
    extra_config: dict | None = None,
):
    config = build_wandb_config(cfg, device)

    if extra_config is not None:
        config.update(extra_config)

    return wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        dir=run_dir,
        name=run_name,
        group=group_name,
        config=config,
        reinit=True,
    )


def run_single_split_training(cfg, device: torch.device, logger) -> None:
    metrics_path = Path(cfg.paths.output_dir) / "metrics.csv"
    write_metrics_header(metrics_path)

    train_loader, valid_loader = build_train_valid_loaders(cfg)

    model = build_model(cfg).to(device)
    criterion = build_loss_fn(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    use_wandb = cfg.logging.use_wandb
    experiment_name = get_experiment_name(cfg)

    if use_wandb:
        init_wandb_run(
            cfg=cfg,
            run_name=f"{experiment_name}-single-split",
            group_name=experiment_name,
            run_dir=cfg.paths.output_dir,
            device=device,
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
            if isinstance(scheduler, dict):
                warmup_epochs = scheduler["warmup_epochs"]
                if epoch <= warmup_epochs:
                    scheduler["warmup"].step()
                else:
                    scheduler["main"].step()
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        append_metrics_row(metrics_path, epoch, train_metrics, valid_metrics)

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
                "lr": current_lr,
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

            if use_wandb:
                wandb.log({
                    "best/epoch": epoch,
                    f"best/{early_stopping_monitor}": best_score,
                })

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


def run_kfold_training(cfg, device: torch.device, logger) -> None:
    full_df = load_train_dataframe(cfg)
    folds = build_kfold_splits(cfg, full_df)

    summary_path = Path(cfg.paths.output_dir) / "kfold_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict[str, float | int]] = []
    log_interval = max(1, int(cfg.logging.log_interval))
    use_wandb = cfg.logging.use_wandb
    kfold_group_name = f"{get_experiment_name(cfg)}-kfold"

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold",
            "best_epoch",
            "best_valid_loss",
            "best_valid_accuracy",
            "best_valid_f1_micro",
            "best_valid_f1_macro",
        ])

    for fold_idx, (train_indices, valid_indices) in enumerate(folds, start=1):
        logger.info(f"[Fold {fold_idx}/{len(folds)}] start")

        fold_output_dir = Path(cfg.paths.output_dir) / f"fold_{fold_idx}"
        fold_checkpoint_dir = fold_output_dir / "checkpoints"
        fold_log_dir = fold_output_dir / "logs"
        fold_metrics_path = fold_output_dir / "metrics.csv"
        fold_best_model_path = fold_checkpoint_dir / "best.pt"

        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fold_log_dir.mkdir(parents=True, exist_ok=True)
        write_metrics_header(fold_metrics_path)

        train_loader, valid_loader = build_train_valid_loaders_for_fold(
            cfg=cfg,
            dataframe=full_df,
            train_indices=train_indices,
            valid_indices=valid_indices,
        )

        model = build_model(cfg).to(device)
        criterion = build_loss_fn(cfg)
        optimizer = build_optimizer(cfg, model)
        scheduler = build_scheduler(cfg, optimizer)

        wandb_run = None
        if use_wandb:
            wandb_run = init_wandb_run(
                cfg=cfg,
                run_name=f"{get_experiment_name(cfg)}-fold-{fold_idx}",
                group_name=kfold_group_name,
                run_dir=str(fold_output_dir),
                device=device,
                extra_config={"fold": fold_idx},
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

        best_epoch = 0
        best_valid_metrics: dict[str, float] | None = None
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
                if isinstance(scheduler, dict):
                    warmup_epochs = scheduler["warmup_epochs"]
                    if epoch <= warmup_epochs:
                        scheduler["warmup"].step()
                    else:
                        scheduler["main"].step()
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            append_metrics_row(fold_metrics_path, epoch, train_metrics, valid_metrics)

            if should_log_epoch(
                epoch=epoch,
                total_epochs=cfg.train.epochs,
                log_interval=log_interval,
            ):
                logger.info(
                    f"[Fold {fold_idx}/{len(folds)}] "
                    f"[Epoch {epoch}/{cfg.train.epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_{cfg.metric.primary}={train_metrics['primary_score']:.4f} "
                    f"valid_loss={valid_metrics['loss']:.4f} "
                    f"valid_{cfg.metric.primary}={valid_metrics['primary_score']:.4f}"
                )

            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "fold": fold_idx,
                    "lr": current_lr,
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
                best_epoch = epoch
                best_valid_metrics = valid_metrics.copy()
                early_stopping_counter = 0
                save_checkpoint(model, fold_best_model_path)

                logger.info(
                    f"[Fold {fold_idx}/{len(folds)}] best model saved "
                    f"| path={fold_best_model_path} "
                    f"| {early_stopping_monitor}={best_score:.4f}"
                )

                if use_wandb:
                    wandb.log({
                        "best/epoch": best_epoch,
                        "best/valid_loss": best_valid_metrics["loss"],
                        "best/valid_accuracy": best_valid_metrics["accuracy"],
                        "best/valid_f1_micro": best_valid_metrics["f1_micro"],
                        "best/valid_f1_macro": best_valid_metrics["f1_macro"],
                    })

            else:
                early_stopping_counter += 1

                if use_early_stopping:
                    logger.info(
                        f"[Fold {fold_idx}/{len(folds)}] early stopping counter | "
                        f"patience={early_stopping_counter}/{early_stopping_patience}"
                    )

                    if early_stopping_counter >= early_stopping_patience:
                        logger.info(
                            f"[Fold {fold_idx}/{len(folds)}] early stopping triggered "
                            f"| epoch={epoch} "
                            f"| best_{early_stopping_monitor}={best_score:.4f}"
                        )
                        break

        if best_valid_metrics is None:
            msg = f"Fold {fold_idx} did not produce best validation metrics."
            raise RuntimeError(msg)

        fold_result = {
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_metrics["loss"],
            "best_valid_accuracy": best_valid_metrics["accuracy"],
            "best_valid_f1_micro": best_valid_metrics["f1_micro"],
            "best_valid_f1_macro": best_valid_metrics["f1_macro"],
        }
        fold_results.append(fold_result)

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                fold_result["fold"],
                fold_result["best_epoch"],
                fold_result["best_valid_loss"],
                fold_result["best_valid_accuracy"],
                fold_result["best_valid_f1_micro"],
                fold_result["best_valid_f1_macro"],
            ])

        logger.info(
            f"[Fold {fold_idx}/{len(folds)}] done "
            f"| best_epoch={best_epoch} "
            f"| best_valid_f1_macro={best_valid_metrics['f1_macro']:.4f}"
        )

        if use_wandb and wandb_run is not None:
            wandb.summary["best_epoch"] = best_epoch
            wandb.summary["best_valid_loss"] = best_valid_metrics["loss"]
            wandb.summary["best_valid_accuracy"] = best_valid_metrics["accuracy"]
            wandb.summary["best_valid_f1_micro"] = best_valid_metrics["f1_micro"]
            wandb.summary["best_valid_f1_macro"] = best_valid_metrics["f1_macro"]
            wandb.finish()

    f1_macro_scores = [float(result["best_valid_f1_macro"]) for result in fold_results]
    mean_f1_macro = sum(f1_macro_scores) / len(f1_macro_scores)
    variance = sum((score - mean_f1_macro) ** 2 for score in f1_macro_scores) / len(
        f1_macro_scores
    )
    std_f1_macro = variance**0.5

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["mean", "", "", "", "", mean_f1_macro])
        writer.writerow(["std", "", "", "", "", std_f1_macro])

    logger.info(f"[KFold] mean_valid_f1_macro={mean_f1_macro:.4f}")
    logger.info(f"[KFold] std_valid_f1_macro={std_f1_macro:.4f}")

    if use_wandb:
        init_wandb_run(
            cfg=cfg,
            run_name=f"{get_experiment_name(cfg)}-kfold-summary",
            group_name=kfold_group_name,
            run_dir=cfg.paths.output_dir,
            device=device,
            extra_config={"summary_only": True},
        )
        wandb.summary["mean_valid_f1_macro"] = mean_f1_macro
        wandb.summary["std_valid_f1_macro"] = std_f1_macro
        for result in fold_results:
            fold_name = int(result["fold"])
            wandb.summary[f"fold_{fold_name}_best_epoch"] = result["best_epoch"]
            wandb.summary[f"fold_{fold_name}_best_valid_f1_macro"] = result[
                "best_valid_f1_macro"
            ]
        wandb.finish()


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

    project_root = resolve_execution_root(
        args.data,
        args.train,
        args.inference,
        args.model,
    )
    initialize_output_paths(cfg, project_root)

    logger = setup_logger("src.train", cfg.paths.log_dir)

    device = get_device(cfg.runtime.device)
    logger.info(f"[DEVICE: {device.type.upper()}]")
    logger.info(f"[EXPERIMENT: {get_experiment_name(cfg)}]")
    logger.info(f"[OUTPUT DIR: {cfg.paths.output_dir}]")

    if cfg.split.method == "stratified_kfold":
        run_kfold_training(cfg=cfg, device=device, logger=logger)
        return

    run_single_split_training(cfg=cfg, device=device, logger=logger)


if __name__ == "__main__":
    main()
