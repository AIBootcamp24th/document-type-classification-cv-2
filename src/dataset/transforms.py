from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _build_resize(resize_cfg: Any) -> A.BasicTransform:
    return A.Resize(
        height=resize_cfg.height,
        width=resize_cfg.width,
    )


def _build_horizontal_flip(flip_cfg: Any) -> A.BasicTransform:
    return A.HorizontalFlip(
        p=_get_attr(flip_cfg, "p", 0.5),
    )


def _build_random_brightness_contrast(rbc_cfg: Any) -> A.BasicTransform:
    return A.RandomBrightnessContrast(
        brightness_limit=_get_attr(rbc_cfg, "brightness_limit", 0.2),
        contrast_limit=_get_attr(rbc_cfg, "contrast_limit", 0.2),
        p=_get_attr(rbc_cfg, "p", 0.5),
    )


def _build_affine(affine_cfg: Any) -> A.BasicTransform:
    return A.Affine(
        translate_percent=_get_attr(affine_cfg, "translate_percent", 0.05),
        scale=(
            1.0 - _get_attr(affine_cfg, "scale_limit", 0.1),
            1.0 + _get_attr(affine_cfg, "scale_limit", 0.1),
        ),
        rotate=(
            -_get_attr(affine_cfg, "rotate_limit", 5),
            _get_attr(affine_cfg, "rotate_limit", 5),
        ),
        p=_get_attr(affine_cfg, "p", 0.5),
    )


def _build_normalize(cfg: Any, normalize_cfg: Any) -> A.BasicTransform:
    return A.Normalize(
        mean=cfg.image.mean,
        std=cfg.image.std,
        max_pixel_value=255.0,
        p=_get_attr(normalize_cfg, "p", 1.0),
    )


def _build_to_tensor() -> A.BasicTransform:
    return ToTensorV2()


def _build_gaussian_blur(blur_cfg: Any) -> A.BasicTransform:
    return A.GaussianBlur(
        blur_limit=_get_attr(blur_cfg, "blur_limit", (3, 7)),
        p=_get_attr(blur_cfg, "p", 0.5),
    )


def _build_gauss_noise(noise_cfg: Any) -> A.BasicTransform:
    return A.GaussNoise(
        var_limit=_get_attr(noise_cfg, "var_limit", (10.0, 50.0)),
        mean=_get_attr(noise_cfg, "mean", 0),
        p=_get_attr(noise_cfg, "p", 0.5),
    )


def _build_color_jitter(jitter_cfg: Any) -> A.BasicTransform:
    return A.ColorJitter(
        brightness=_get_attr(jitter_cfg, "brightness", 0.2),
        contrast=_get_attr(jitter_cfg, "contrast", 0.2),
        saturation=_get_attr(jitter_cfg, "saturation", 0.2),
        hue=_get_attr(jitter_cfg, "hue", 0.2),
        p=_get_attr(jitter_cfg, "p", 0.5),
    )


def _build_random_rotate_90(rotate_cfg: Any) -> A.BasicTransform:
    return A.RandomRotate90(
        p=_get_attr(rotate_cfg, "p", 0.5),
    )


def _build_vertical_flip(flip_cfg: Any) -> A.BasicTransform:
    return A.VerticalFlip(
        p=_get_attr(flip_cfg, "p", 0.5),
    )


def build_transforms(cfg: Any, stage: str) -> A.Compose:
    if stage not in {"train", "valid", "test"}:
        msg = f"Unsupported stage: {stage}. Expected one of: train, valid, test."
        raise ValueError(msg)

    stage_cfg = getattr(cfg.augmentation, stage)
    transforms: list[A.BasicTransform] = []

    if hasattr(stage_cfg, "resize"):
        transforms.append(_build_resize(stage_cfg.resize))

    if hasattr(stage_cfg, "horizontal_flip"):
        transforms.append(_build_horizontal_flip(stage_cfg.horizontal_flip))

    if hasattr(stage_cfg, "random_brightness_contrast"):
        transforms.append(
            _build_random_brightness_contrast(stage_cfg.random_brightness_contrast)
        )

    if hasattr(stage_cfg, "affine"):
        transforms.append(_build_affine(stage_cfg.affine))

    if hasattr(stage_cfg, "gaussian_blur"):
        transforms.append(_build_gaussian_blur(stage_cfg.gaussian_blur))

    if hasattr(stage_cfg, "gauss_noise"):
        transforms.append(_build_gauss_noise(stage_cfg.gauss_noise))

    if hasattr(stage_cfg, "color_jitter"):
        transforms.append(_build_color_jitter(stage_cfg.color_jitter))

    if hasattr(stage_cfg, "random_rotate_90"):
        transforms.append(_build_random_rotate_90(stage_cfg.random_rotate_90))

    if hasattr(stage_cfg, "vertical_flip"):
        transforms.append(_build_vertical_flip(stage_cfg.vertical_flip))

    if hasattr(stage_cfg, "normalize"):
        transforms.append(_build_normalize(cfg, stage_cfg.normalize))

    if hasattr(stage_cfg, "to_tensor"):
        transforms.append(_build_to_tensor())

    return A.Compose(transforms)


def build_train_transforms(cfg: Any) -> A.Compose:
    return build_transforms(cfg=cfg, stage="train")


def build_valid_transforms(cfg: Any) -> A.Compose:
    return build_transforms(cfg=cfg, stage="valid")


def build_test_transforms(cfg: Any) -> A.Compose:
    return build_transforms(cfg=cfg, stage="test")
