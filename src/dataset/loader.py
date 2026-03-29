from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from src.dataset.dataset import DocumentDataset
from src.dataset.transforms import (
    build_test_transforms,
    build_train_transforms,
    build_valid_transforms,
)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def load_train_dataframe(cfg: Any) -> pd.DataFrame:
    return pd.read_csv(_resolve_path(cfg.paths.train_csv))


def load_test_dataframe(cfg: Any) -> pd.DataFrame:
    sample_submission = pd.read_csv(_resolve_path(cfg.paths.sample_submission_csv))
    return sample_submission[[cfg.data.image_col]].copy()


def split_train_valid_dataframe(
    cfg: Any, dataframe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.split.method != "stratified":
        msg = f"Unsupported split method: {cfg.split.method}"
        raise ValueError(msg)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=cfg.split.valid_ratio,
        random_state=cfg.split.random_state,
    )

    train_indices, valid_indices = next(
        splitter.split(dataframe, dataframe[cfg.data.label_col])
    )

    train_df = dataframe.iloc[train_indices].reset_index(drop=True)
    valid_df = dataframe.iloc[valid_indices].reset_index(drop=True)

    return train_df, valid_df


def build_train_dataset(cfg: Any, train_df: pd.DataFrame) -> DocumentDataset:
    return DocumentDataset(
        dataframe=train_df,
        image_col=cfg.data.image_col,
        label_col=cfg.data.label_col,
        image_dir=_resolve_path(cfg.paths.train_image_dir),
        transform=build_train_transforms(cfg),
        stage="train",
    )


def build_valid_dataset(cfg: Any, valid_df: pd.DataFrame) -> DocumentDataset:
    return DocumentDataset(
        dataframe=valid_df,
        image_col=cfg.data.image_col,
        label_col=cfg.data.label_col,
        image_dir=_resolve_path(cfg.paths.train_image_dir),
        transform=build_valid_transforms(cfg),
        stage="valid",
    )


def build_test_dataset(cfg: Any, test_df: pd.DataFrame) -> DocumentDataset:
    return DocumentDataset(
        dataframe=test_df,
        image_col=cfg.data.image_col,
        label_col=None,
        image_dir=_resolve_path(cfg.paths.test_image_dir),
        transform=build_test_transforms(cfg),
        stage="test",
    )


def build_train_loader(cfg: Any, train_dataset: DocumentDataset) -> DataLoader:
    return DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )


def build_valid_loader(cfg: Any, valid_dataset: DocumentDataset) -> DataLoader:
    return DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.train.valid_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )


def build_test_loader(cfg: Any, test_dataset: DocumentDataset) -> DataLoader:
    return DataLoader(
        dataset=test_dataset,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        num_workers=cfg.inference.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        persistent_workers=cfg.data.persistent_workers
        and cfg.inference.num_workers > 0,
    )


def build_train_valid_loaders(cfg: Any) -> tuple[DataLoader, DataLoader]:
    train_full_df = load_train_dataframe(cfg)
    train_df, valid_df = split_train_valid_dataframe(cfg, train_full_df)

    train_dataset = build_train_dataset(cfg, train_df)
    valid_dataset = build_valid_dataset(cfg, valid_df)

    train_loader = build_train_loader(cfg, train_dataset)
    valid_loader = build_valid_loader(cfg, valid_dataset)

    return train_loader, valid_loader


def build_test_loader_from_config(cfg: Any) -> DataLoader:
    test_df = load_test_dataframe(cfg)
    test_dataset = build_test_dataset(cfg, test_df)
    return build_test_loader(cfg, test_dataset)
