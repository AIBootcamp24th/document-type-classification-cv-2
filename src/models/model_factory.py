from __future__ import annotations

from typing import Any, Callable, cast

import timm
import torch.nn as nn
from torchvision import models


def freeze_backbone_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_classifier_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def build_torchvision_model(cfg: Any) -> nn.Module:
    model_name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

        in_features = model.fc.in_features
        if cfg.model.dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=cfg.model.dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)

        if cfg.model.freeze_backbone:
            freeze_backbone_parameters(model)
            unfreeze_classifier_parameters(model.fc)

        return model

    msg = f"Unsupported torchvision model: {model_name}"
    raise ValueError(msg)


def build_timm_model(cfg: Any) -> nn.Module:
    model_name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained
    drop_rate = cfg.model.dropout
    drop_path_rate = getattr(cfg.model, "drop_path_rate", 0.0)

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    if not isinstance(model, nn.Module):
        msg = f"timm.create_model did not return nn.Module for model: {model_name}"
        raise TypeError(msg)

    model = cast(nn.Module, model)

    if cfg.model.freeze_backbone:
        freeze_backbone_parameters(model)

        get_classifier_fn = getattr(model, "get_classifier", None)

        if get_classifier_fn is None:
            msg = f"get_classifier method not found for timm model: {model_name}"
            raise AttributeError(msg)

        if not callable(get_classifier_fn):
            msg = f"get_classifier is not callable for timm model: {model_name}"
            raise TypeError(msg)

        classifier = cast(Callable[[], nn.Module | str | None], get_classifier_fn)()

        if isinstance(classifier, str):
            msg = f"Classifier lookup returned string for timm model: {model_name}"
            raise TypeError(msg)

        if classifier is None:
            msg = f"Classifier not found for timm model: {model_name}"
            raise ValueError(msg)

        if not isinstance(classifier, nn.Module):
            msg = f"Classifier is not nn.Module for timm model: {model_name}"
            raise TypeError(msg)

        unfreeze_classifier_parameters(classifier)

    return model


def build_model(cfg: Any) -> nn.Module:
    library = cfg.model.library

    if library == "torchvision":
        return build_torchvision_model(cfg)

    if library == "timm":
        return build_timm_model(cfg)

    msg = f"Unsupported model library: {library}"
    raise ValueError(msg)
