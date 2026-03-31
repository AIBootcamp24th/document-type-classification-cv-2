from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config, parse_args
from src.dataset.loader import build_test_loader_from_config
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


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device,
) -> None:
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


def build_kfold_checkpoint_paths(experiment_root: Path, n_splits: int) -> list[Path]:
    checkpoint_paths: list[Path] = []

    for fold_idx in range(1, n_splits + 1):
        checkpoint_path = (
            experiment_root
            / "outputs"
            / f"fold_{fold_idx}"
            / "checkpoints"
            / "best.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"KFold checkpoint not found: {checkpoint_path}"
            )

        checkpoint_paths.append(checkpoint_path)

    return checkpoint_paths


def load_ensemble_models(cfg, device: torch.device, checkpoint_paths: list[Path]) -> list[torch.nn.Module]:
    models_list: list[torch.nn.Module] = []

    for checkpoint_path in checkpoint_paths:
        model = build_model(cfg).to(device)
        load_checkpoint(model, checkpoint_path, device)
        model.eval()
        models_list.append(model)

    return models_list


def main() -> None:
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

    logger = setup_logger("src.infer", cfg.paths.log_dir)

    device = get_device(cfg.runtime.device)
    logger.info(f"[DEVICE: {device.type.upper()}]")

    test_loader = build_test_loader_from_config(cfg)

    if cfg.split.method != "stratified_kfold":
        raise ValueError(
            "This inference script is intended for KFold ensemble inference. "
            f"Current split.method={cfg.split.method}"
        )

    checkpoint_paths = build_kfold_checkpoint_paths(
        experiment_root=experiment_root,
        n_splits=cfg.split.n_splits,
    )
    logger.info(f"[KFold ensemble] checkpoints={len(checkpoint_paths)}")

    ensemble_models = load_ensemble_models(
        cfg=cfg,
        device=device,
        checkpoint_paths=checkpoint_paths,
    )

    preds = []
    image_names = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="inference"):
            images = batch["image"].to(device)

            logits_sum = None

            for model in ensemble_models:
                outputs = model(images)

                if logits_sum is None:
                    logits_sum = outputs
                else:
                    logits_sum += outputs

            if logits_sum is None:
                raise RuntimeError("No ensemble outputs were produced.")

            logits_mean = logits_sum / len(ensemble_models)
            pred = torch.argmax(logits_mean, dim=1)

            preds.extend(pred.cpu().numpy())
            image_names.extend(batch["image_name"])

    submission = pd.DataFrame({
        cfg.data.image_col: image_names,
        cfg.inference.prediction_col: preds,
    })

    save_path = Path(cfg.paths.output_dir) / cfg.submission.file_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(save_path, index=False)

    logger.info(f"submission saved: {save_path}")


if __name__ == "__main__":
    main()