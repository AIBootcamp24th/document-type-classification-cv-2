from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def dict_to_namespace(data: dict[str, Any]) -> SimpleNamespace:
    converted: dict[str, Any] = {}

    for key, value in data.items():
        if isinstance(value, dict):
            converted[key] = dict_to_namespace(value)
        else:
            converted[key] = value

    return SimpleNamespace(**converted)


def load_config(
    base_path: str | Path,
    data_path: str | Path,
    train_path: str | Path,
    inference_path: str | Path,
    model_path: str | Path,
) -> SimpleNamespace:
    cfg = load_yaml(base_path)
    cfg = deep_merge_dicts(cfg, load_yaml(data_path))
    cfg = deep_merge_dicts(cfg, load_yaml(train_path))
    cfg = deep_merge_dicts(cfg, load_yaml(inference_path))
    cfg = deep_merge_dicts(cfg, load_yaml(model_path))
    return dict_to_namespace(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config loader")

    parser.add_argument(
        "--base",
        type=str,
        default="configs/base.yaml",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/data.yaml",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="configs/train.yaml",
    )
    parser.add_argument(
        "--inference",
        type=str,
        default="configs/inference.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="configs/model/resnet50.yaml",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(
        base_path=args.base,
        data_path=args.data,
        train_path=args.train,
        inference_path=args.inference,
        model_path=args.model,
    )

    print(cfg)
    print(cfg.project.name)
    print(cfg.model.name)
    print(cfg.train.epochs)
