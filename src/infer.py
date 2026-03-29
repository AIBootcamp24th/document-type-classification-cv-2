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
    if device_config == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)


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
    cfg.inference.checkpoint_path = str(
        experiment_root / "outputs" / "checkpoints" / "best.pt"
    )

    logger = setup_logger("src.infer", cfg.paths.log_dir)

    device = get_device(cfg.runtime.device)

    test_loader = build_test_loader_from_config(cfg)

    model = build_model(cfg).to(device)
    load_checkpoint(model, cfg.inference.checkpoint_path)

    model.eval()

    preds = []
    image_names = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="inference"):
            images = batch["image"].to(device)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)

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
