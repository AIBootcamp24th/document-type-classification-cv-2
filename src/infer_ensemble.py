from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.config import load_config
from src.dataset.loader import build_test_loader_from_config
from src.models.model_factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weighted model ensemble inference")
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
        fold_checkpoint_path = (
            output_dir / f"fold_{fold_idx}" / "checkpoints" / "best.pt"
        )
        if not fold_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {fold_checkpoint_path}")
        checkpoint_paths.append(fold_checkpoint_path)

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


def run_weighted_ensemble(
    member_entries: list[tuple[Any, list[torch.nn.Module], float]],
    device: torch.device,
) -> pd.DataFrame:
    image_names: list[str] | None = None
    weighted_logits_sum: torch.Tensor | None = None
    total_weight = 0.0

    with torch.no_grad():
        for member_idx, (cfg, models, member_weight) in enumerate(member_entries):
            test_loader = build_test_loader_from_config(cfg)

            member_image_names: list[str] = []
            member_logits_batches: list[torch.Tensor] = []

            for batch in tqdm(
                test_loader,
                desc=f"{cfg.model.name} inference",
            ):
                images = batch["image"].to(device)

                fold_logits_sum: torch.Tensor | None = None

                for model in models:
                    outputs = model(images)

                    if fold_logits_sum is None:
                        fold_logits_sum = outputs
                    else:
                        fold_logits_sum += outputs

                if fold_logits_sum is None:
                    raise RuntimeError(
                        f"No logits produced for ensemble member: {cfg.model.name}"
                    )

                fold_logits_mean = fold_logits_sum / len(models)
                member_logits_batches.append(fold_logits_mean.cpu())
                member_image_names.extend(batch["image_name"])

            if not member_logits_batches:
                raise RuntimeError(
                    f"No inference outputs produced for member: {cfg.model.name}"
                )

            member_logits = torch.cat(member_logits_batches, dim=0)

            if member_idx == 0:
                image_names = member_image_names
                weighted_logits_sum = member_logits * member_weight
            else:
                if image_names != member_image_names:
                    raise RuntimeError(
                        "Image order mismatch across ensemble members. "
                        f"member={cfg.model.name}"
                    )

                if weighted_logits_sum is None:
                    raise RuntimeError("Weighted ensemble logits were not initialized.")

                weighted_logits_sum += member_logits * member_weight

            total_weight += member_weight

    if image_names is None:
        raise RuntimeError("Image names were not produced.")

    if weighted_logits_sum is None:
        raise RuntimeError("Weighted ensemble logits were not produced.")

    if total_weight == 0:
        raise RuntimeError("Total ensemble weight must be greater than zero.")

    final_logits = weighted_logits_sum / total_weight
    preds = torch.argmax(final_logits, dim=1).tolist()

    return pd.DataFrame({
        member_entries[0][0].data.image_col: image_names,
        member_entries[0][0].inference.prediction_col: preds,
    })


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

    member_entries: list[tuple[Any, list[torch.nn.Module], float]] = []
    for member_cfg in ensemble_cfg["ensemble"]["members"]:
        member_entries.append(
            load_member_models(member_cfg, common_paths, execution_root, device)
        )

    submission = run_weighted_ensemble(member_entries, device)

    save_path = execution_root / "outputs" / ensemble_cfg["ensemble"]["file_name"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_path, index=False)

    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
