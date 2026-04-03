from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _build_resize(resize_cfg: Any, cfg: Any) -> A.BasicTransform:
    height = resize_cfg.height
    width = resize_cfg.width

    model_name = cfg.model.name.lower()

    if "deit" in model_name or "vit" in model_name:
        height = 224
        width = 224

    return A.Resize(
        height=height,
        width=width,
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


def _build_perspective(p_cfg: Any) -> A.BasicTransform:
    return A.Perspective(
        scale=_get_attr(p_cfg, "scale", (0.05, 0.1)),
        p=_get_attr(p_cfg, "p", 0.5),
    )


def _build_clahe(clahe_cfg: Any) -> A.BasicTransform:
    return A.CLAHE(
        clip_limit=_get_attr(clahe_cfg, "clip_limit", 4.0),
        tile_grid_size=_get_attr(clahe_cfg, "tile_grid_size", (8, 8)),
        p=_get_attr(clahe_cfg, "p", 0.5),
    )


def _build_sharpen(sharpen_cfg: Any) -> A.BasicTransform:
    return A.Sharpen(
        alpha=_get_attr(sharpen_cfg, "alpha", (0.2, 0.5)),
        lightness=_get_attr(sharpen_cfg, "lightness", (0.5, 1.0)),
        p=_get_attr(sharpen_cfg, "p", 0.5),
    )


def _build_grid_distortion(grid_cfg: Any) -> A.BasicTransform:
    return A.GridDistortion(
        num_steps=_get_attr(grid_cfg, "num_steps", 5),
        distort_limit=_get_attr(grid_cfg, "distort_limit", 0.3),
        p=_get_attr(grid_cfg, "p", 0.5),
    )


def build_transforms(cfg: Any, stage: str) -> A.Compose:
    if stage not in {"train", "valid", "test"}:
        msg = f"Unsupported stage: {stage}. Expected one of: train, valid, test."
        raise ValueError(msg)

    stage_cfg = getattr(cfg.augmentation, stage)
    transforms: list[A.BasicTransform] = []

    if hasattr(stage_cfg, "resize"):
        transforms.append(_build_resize(stage_cfg.resize, cfg))

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

    if hasattr(stage_cfg, "perspective"):
        transforms.append(_build_perspective(stage_cfg.perspective))

    if hasattr(stage_cfg, "clahe"):
        transforms.append(_build_clahe(stage_cfg.clahe))

    if hasattr(stage_cfg, "sharpen"):
        transforms.append(_build_sharpen(stage_cfg.sharpen))

    if hasattr(stage_cfg, "grid_distortion"):
        transforms.append(_build_grid_distortion(stage_cfg.grid_distortion))

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
