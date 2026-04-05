from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.dataset.loader import (
    build_kfold_splits,
    build_valid_dataset,
    load_train_dataframe,
)
from src.models.model_factory import build_model
from src.utils.logger import setup_logger
from src.utils.metric import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weighted model ensemble validation")
    parser.add_argument("--base", type=str, default="configs/base.yaml")
    parser.add_argument("--data", type=str, default="configs/data.yaml")
    parser.add_argument("--train", type=str, default="configs/train.yaml")
    parser.add_argument("--inference", type=str, default="configs/inference.yaml")
    parser.add_argument("--ensemble", type=str, default="configs/ensemble.yaml")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


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


def initialize_output_paths(cfg, execution_root: Path) -> None:
    experiment_name = getattr(cfg, "experiment", None)
    experiment_name = getattr(experiment_name, "name", cfg.model.name)

    output_root = execution_root / "outputs" / experiment_name

    cfg.paths.output_dir = str(output_root)
    cfg.paths.checkpoint_dir = str(output_root / "checkpoints")
    cfg.paths.log_dir = str(output_root / "logs")
    cfg.inference.checkpoint_path = str(output_root / "checkpoints" / "best.pt")


def get_device(device_config: str) -> torch.device:
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "runtime.device is set to 'cuda', but CUDA is not available."
            )
        return torch.device("cuda")

    if device_config == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "runtime.device is set to 'mps', but MPS is not available."
            )
        return torch.device("mps")

    if device_config == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported runtime.device: {device_config}")


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


def build_member_checkpoint_paths(output_dir: str | Path, n_splits: int) -> list[Path]:
    output_dir = Path(output_dir).resolve()
    checkpoint_paths: list[Path] = []

    for fold_idx in range(1, n_splits + 1):
        checkpoint_path = output_dir / f"fold_{fold_idx}" / "checkpoints" / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_paths.append(checkpoint_path)

    return checkpoint_paths


def load_member_models(
    member_cfg: dict[str, Any],
    common_paths: dict[str, str],
    execution_root: Path,
    device: torch.device,
) -> tuple[Any, list[torch.nn.Module], float]:
    cfg = load_config(
        base_path=common_paths["base"],
        data_path=common_paths["data"],
        train_path=common_paths["train"],
        inference_path=common_paths["inference"],
        model_path=member_cfg["model_path"],
    )

    initialize_output_paths(cfg, execution_root)

    checkpoint_paths = build_member_checkpoint_paths(
        output_dir=cfg.paths.output_dir,
        n_splits=cfg.split.n_splits,
    )

    models: list[torch.nn.Module] = []
    for checkpoint_path in checkpoint_paths:
        model = build_model(cfg).to(device)
        load_checkpoint(model, checkpoint_path, device)
        model.eval()
        models.append(model)

    return cfg, models, float(member_cfg["weight"])


def build_valid_df_for_fold(cfg: Any, fold_idx: int):
    full_df = load_train_dataframe(cfg)
    folds = build_kfold_splits(cfg, full_df)

    _, valid_idx = folds[fold_idx - 1]
    valid_df = full_df.iloc[valid_idx].reset_index(drop=True)

    return valid_df


def run_weighted_valid_ensemble(
    member_entries: list[tuple[Any, list[torch.nn.Module], float]],
    device: torch.device,
    logger,
) -> dict[str, float]:
    if not member_entries:
        raise RuntimeError("No ensemble members were loaded.")

    reference_cfg = member_entries[0][0]
    n_splits = reference_cfg.split.n_splits

    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for fold_idx in range(1, n_splits + 1):
            logger.info(f"[Fold {fold_idx}]")

            valid_df = build_valid_df_for_fold(reference_cfg, fold_idx)

            weighted_logits_sum: torch.Tensor | None = None
            fold_targets: list[int] = []
            total_weight = 0.0

            for member_idx, (cfg, models, member_weight) in enumerate(member_entries):
                if len(models) < fold_idx:
                    raise RuntimeError(
                        f"Missing fold-{fold_idx} model for member: {cfg.model.name}"
                    )

                model = models[fold_idx - 1]

                valid_dataset = build_valid_dataset(cfg, valid_df)
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=cfg.train.valid_batch_size,
                    shuffle=False,
                )

                member_logits_batches: list[torch.Tensor] = []

                for batch in tqdm(
                    valid_loader,
                    desc=f"{cfg.model.name} fold {fold_idx}",
                ):
                    images = batch["image"].to(device)
                    outputs = model(images)
                    member_logits_batches.append(outputs.cpu())

                    if member_idx == 0:
                        fold_targets.extend(batch["label"].tolist())

                if not member_logits_batches:
                    raise RuntimeError(
                        f"No logits produced for member={cfg.model.name}, fold={fold_idx}"
                    )

                member_logits = torch.cat(member_logits_batches, dim=0)

                if weighted_logits_sum is None:
                    weighted_logits_sum = member_logits * member_weight
                else:
                    if weighted_logits_sum.shape != member_logits.shape:
                        raise RuntimeError(
                            "Logit shape mismatch across ensemble members at "
                            f"fold={fold_idx}, member={cfg.model.name}. "
                            f"expected={tuple(weighted_logits_sum.shape)}, "
                            f"got={tuple(member_logits.shape)}"
                        )
                    weighted_logits_sum += member_logits * member_weight

                total_weight += member_weight

            if weighted_logits_sum is None:
                raise RuntimeError(f"No logits produced for fold {fold_idx}.")

            if total_weight == 0:
                raise RuntimeError("Total ensemble weight must be greater than zero.")

            fold_logits = weighted_logits_sum / total_weight
            fold_preds = torch.argmax(fold_logits, dim=1).tolist()

            if len(fold_preds) != len(fold_targets):
                raise RuntimeError(
                    f"Prediction/target length mismatch at fold {fold_idx}: "
                    f"preds={len(fold_preds)}, targets={len(fold_targets)}"
                )

            all_preds.extend(fold_preds)
            all_targets.extend(fold_targets)

    return compute_metrics(all_targets, all_preds)


def main() -> None:
    args = parse_args()
    ensemble_cfg = load_yaml(args.ensemble)

    execution_root = resolve_execution_root(
        args.data,
        args.train,
        args.inference,
        args.ensemble,
    )

    common_paths = {
        "base": args.base,
        "data": args.data,
        "train": args.train,
        "inference": args.inference,
    }

    base_cfg = load_config(
        base_path=args.base,
        data_path=args.data,
        train_path=args.train,
        inference_path=args.inference,
        model_path=ensemble_cfg["ensemble"]["members"][0]["model_path"],
    )
    device = get_device(base_cfg.runtime.device)

    log_output_dir = execution_root / "outputs"
    logger = setup_logger("valid-ensemble", str(log_output_dir))

    member_entries: list[tuple[Any, list[torch.nn.Module], float]] = []
    for member_cfg in ensemble_cfg["ensemble"]["members"]:
        logger.info(
            f"[LOAD MEMBER] name={member_cfg['name']} weight={float(member_cfg['weight']):.4f}"
        )
        member_entries.append(
            load_member_models(member_cfg, common_paths, execution_root, device)
        )

    metrics = run_weighted_valid_ensemble(member_entries, device, logger)

    logger.info("[ENSEMBLE VALID RESULT]")
    logger.info(f"accuracy={metrics['accuracy']:.4f}")
    logger.info(f"f1_micro={metrics['f1_micro']:.4f}")
    logger.info(f"f1_macro={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
