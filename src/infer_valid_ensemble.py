from pathlib import Path

import torch
from tqdm import tqdm

from src.config import load_config, parse_args
from src.dataset.loader import build_kfold_splits, load_train_dataframe, build_valid_dataset
from torch.utils.data import DataLoader
from src.models.model_factory import build_model
from src.utils.metric import compute_metrics
from src.utils.logger import setup_logger


def get_device(device_config: str) -> torch.device:
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def load_checkpoint(model, path, device):
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


def main():
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

    logger = setup_logger("valid-ensemble", cfg.paths.output_dir)

    device = get_device(cfg.runtime.device)

    full_df = load_train_dataframe(cfg)
    folds = build_kfold_splits(cfg, full_df)

    all_preds = []
    all_targets = []

    for fold_idx, (train_idx, valid_idx) in enumerate(folds, start=1):
        logger.info(f"[Fold {fold_idx}]")

        valid_df = full_df.iloc[valid_idx].reset_index(drop=True)

        valid_dataset = build_valid_dataset(cfg, valid_df)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.train.valid_batch_size,
            shuffle=False,
        )

        checkpoint_path = (
            Path(cfg.paths.output_dir)
            / f"fold_{fold_idx}"
            / "checkpoints"
            / "best.pt"
        )

        model = build_model(cfg).to(device)
        load_checkpoint(model, checkpoint_path, device)
        model.eval()

        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"fold {fold_idx}"):
                images = batch["image"].to(device)
                targets = batch["label"].to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                fold_preds.extend(preds.cpu().tolist())
                fold_targets.extend(targets.cpu().tolist())

        all_preds.extend(fold_preds)
        all_targets.extend(fold_targets)

    metrics = compute_metrics(all_targets, all_preds)

    logger.info(f"[ENSEMBLE VALID RESULT]")
    logger.info(f"accuracy={metrics['accuracy']:.4f}")
    logger.info(f"f1_micro={metrics['f1_micro']:.4f}")
    logger.info(f"f1_macro={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()