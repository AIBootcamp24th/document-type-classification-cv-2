from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_col: str,
        label_col: str | None = None,
        image_dir: str | Path | None = None,
        transform: Any = None,
        stage: str = "train",
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.transform = transform
        self.stage = stage

        if self.stage not in {"train", "valid", "test"}:
            msg = (
                f"Unsupported stage: {self.stage}. Expected one of: train, valid, test."
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_image_path(self, image_name: str) -> Path:
        image_path = Path(image_name)

        if image_path.is_absolute():
            return image_path

        if self.image_dir is not None:
            return self.image_dir / image_path

        return image_path

    def _load_image(self, image_path: Path) -> Any:
        image = cv2.imread(str(image_path))
        if image is None:
            msg = f"Failed to load image: {image_path}"
            raise FileNotFoundError(msg)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]

        image_path = self._resolve_image_path(str(row[self.image_col]))
        image = self._load_image(image_path)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if self.stage == "test":
            return {
                "image": image,
                "image_name": row[self.image_col],
                "image_path": str(image_path),
            }

        if self.label_col is None:
            msg = "label_col must be provided for train/valid stage."
            raise ValueError(msg)

        label = int(row[self.label_col])

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "image_name": row[self.image_col],
            "image_path": str(image_path),
        }
