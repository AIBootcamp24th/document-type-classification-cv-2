from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


def build_kfold_checkpoint_paths(
    output_dir: str | Path,
    checkpoint_path: str | Path,
    n_splits: int,
) -> list[Path]:
    checkpoint_paths: list[Path] = []

    output_dir = Path(output_dir).resolve()
    checkpoint_path = Path(checkpoint_path)

    relative_checkpoint_path = Path("checkpoints") / "best.pt"

    for fold_idx in range(1, n_splits + 1):
        fold_checkpoint_path = (
            output_dir / f"fold_{fold_idx}" / relative_checkpoint_path
        )

        if not fold_checkpoint_path.exists():
            raise FileNotFoundError(
                f"KFold checkpoint not found: {fold_checkpoint_path}"
            )

        checkpoint_paths.append(fold_checkpoint_path)

    return checkpoint_paths


def load_ensemble_models(
    cfg,
    device: torch.device,
    checkpoint_paths: list[Path],
) -> list[torch.nn.Module]:
    models_list: list[torch.nn.Module] = []

    for checkpoint_path in checkpoint_paths:
        model = build_model(cfg).to(device)
        load_checkpoint(model, checkpoint_path, device)
        model.eval()
        models_list.append(model)

    return models_list


def build_tta_transforms_from_cfg(cfg):
    tta_cfg = getattr(cfg.inference, "tta", None)

    if tta_cfg is None or not getattr(tta_cfg, "use", False):
        return []

    transforms_cfg = getattr(tta_cfg, "transforms", [])
    tta_transforms = []

    for t in transforms_cfg:
        t_type = t["type"]
        aug_list = []

        if t_type == "original":
            pass

        elif t_type == "rotate":
            if "limit" not in t:
                raise ValueError("rotate TTA requires 'limit' in config")

            limit = t["limit"]

            aug_list.append(
                A.Rotate(limit=(limit, limit), p=1.0)
            )

        elif t_type == "brightness":
            if "brightness_limit" not in t or "contrast_limit" not in t:
                raise ValueError("brightness TTA requires brightness_limit and contrast_limit")

            aug_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=t["brightness_limit"],
                    contrast_limit=t["contrast_limit"],
                    p=1.0
                )
            )

        else:
            raise ValueError(f"Unsupported TTA type: {t_type}")

        # 공통 후처리
        aug_list.append(A.Normalize(mean=cfg.image.mean, std=cfg.image.std))
        aug_list.append(ToTensorV2())

        tta_transforms.append(A.Compose(aug_list))

    return tta_transforms


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


def main() -> None:
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

    logger = setup_logger("src.infer", cfg.paths.log_dir)

    device = get_device(cfg.runtime.device)
    logger.info(f"[DEVICE: {device.type.upper()}]")
    logger.info(f"[EXPERIMENT: {get_experiment_name(cfg)}]")
    logger.info(f"[OUTPUT DIR: {cfg.paths.output_dir}]")

    test_loader = build_test_loader_from_config(cfg)

    if cfg.split.method != "stratified_kfold":
        raise ValueError(
            "This inference script is intended for KFold ensemble inference. "
            f"Current split.method={cfg.split.method}"
        )

    checkpoint_paths = build_kfold_checkpoint_paths(
        output_dir=cfg.paths.output_dir,
        checkpoint_path=cfg.inference.checkpoint_path,
        n_splits=cfg.split.n_splits,
    )
    logger.info(f"[KFold ensemble] checkpoints={len(checkpoint_paths)}")

    ensemble_models = load_ensemble_models(
        cfg=cfg,
        device=device,
        checkpoint_paths=checkpoint_paths,
    )

    # TTA 설정
    tta_cfg = getattr(cfg.inference, "tta", None)

    if tta_cfg is not None:
        use_tta = getattr(tta_cfg, "use", False)
    else:
        use_tta = False
        tta_transforms_cfg = []

    if use_tta:
        tta_transforms = build_tta_transforms_from_cfg(cfg)
        tta_size = len(tta_transforms)
    else:
        tta_transforms = []
        tta_size = 1
       
    logger.info(f"[TTA USE : {use_tta}]")
    logger.info(f"[TTA SIZE: {tta_size}]")

    preds = []
    image_names = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="inference"):

            # TTA OFF
            if not use_tta:
                images = batch["image"]
            
                if images.ndim == 4 and images.shape[-1] == 3:
                    images = images.permute(0, 3, 1, 2)

                images = images.to(device).float()    

                logits_sum = None

                for model in ensemble_models:
                    outputs = model(images)

                    if logits_sum is None:
                        logits_sum = outputs
                    else:
                        logits_sum += outputs

                logits_mean = logits_sum / len(ensemble_models)
                probs = torch.softmax(logits_mean, dim=1)

                preds.extend(probs.cpu().numpy())
                image_names.extend(batch["image_name"])

            # TTA ON  
            else:
                images = batch["image"]
                batch_image_names = batch["image_name"]

                tta_logits_sum = None

                for tta_transform in tta_transforms:
                    tta_images = []

                    for img in images:
                        img_np = img.permute(1, 2, 0).cpu().numpy()

                        augmented = tta_transform(image=img_np)
                        tta_img = augmented["image"]

                        if isinstance(tta_img, np.ndarray):
                            tta_img = torch.from_numpy(tta_img).permute(2, 0, 1).float()
                        elif isinstance(tta_img, torch.Tensor):
                            if tta_img.shape[0] != 3:
                                tta_img = tta_img.permute(2, 0, 1)

                        tta_images.append(tta_img)

                    tta_images = torch.stack(tta_images)

                    if tta_images.shape[1] != 3:
                        tta_images = tta_images.permute(0, 3, 1, 2)

                    tta_images = tta_images.to(device)

                    fold_logits_sum = None

                    for model in ensemble_models:
                        outputs = model(tta_images)

                        if fold_logits_sum is None:
                            fold_logits_sum = outputs
                        else:
                            fold_logits_sum += outputs

                    fold_logits_mean = fold_logits_sum / len(ensemble_models)

                    if tta_logits_sum is None:
                        tta_logits_sum = fold_logits_mean
                    else:
                        tta_logits_sum += fold_logits_mean

                final_logits = tta_logits_sum / tta_size
                probs = torch.softmax(final_logits, dim=1)

                preds.extend(probs.cpu().numpy())
                image_names.extend(batch_image_names)

    probs_array = np.array(preds)

    save_path = Path(cfg.paths.output_dir) / "probs.npy"
    np.save(save_path, probs_array)

    names_path = Path(cfg.paths.output_dir) / "image_names.npy"
    np.save(names_path, np.array(image_names))

    logger.info(f"probs saved: {save_path}")


if __name__ == "__main__":
    main()