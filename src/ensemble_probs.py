from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def load_probs_and_names(base_dir: Path):
    probs_path = base_dir / "probs.npy"
    names_path = base_dir / "image_names.npy"

    if not probs_path.exists():
        raise FileNotFoundError(f"{probs_path} not found")
    if not names_path.exists():
        raise FileNotFoundError(f"{names_path} not found")

    probs = np.load(probs_path)
    names = np.load(names_path)

    return probs.astype(np.float32), names


def main():
    # 프로젝트 루트 기준으로 경로 고정
    root = Path(__file__).resolve().parents[1]

    resnet_dir = root / "outputs" / "resnet50"
    convnext_dir = root / "outputs" / "convnext_tiny"

    print(f"[LOAD] {resnet_dir}")
    print(f"[LOAD] {convnext_dir}")

    resnet_probs, resnet_names = load_probs_and_names(resnet_dir)
    convnext_probs, convnext_names = load_probs_and_names(convnext_dir)

    # 가장 중요: 순서 검증
    if not np.array_equal(resnet_names, convnext_names):
        raise ValueError("image_names mismatch between models")

    # 가중치 앙상블
    w1, w2 = 0.8, 0.2

    final_probs = w1 * resnet_probs + w2 * convnext_probs

    preds = final_probs.argmax(axis=1)

    submission = pd.DataFrame({
        "ID": resnet_names,
        "target": preds
    })

    save_path = root / "final_submission.csv"
    submission.to_csv(save_path, index=False)

    print(f"[SAVED] {save_path}")
    print(f"[INFO] shape={final_probs.shape}")


if __name__ == "__main__":
    main()