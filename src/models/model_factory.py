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


def _build_resnet_classifier(
    in_features: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    return nn.Linear(in_features, num_classes)


def build_torchvision_model(cfg: Any) -> nn.Module:
    model_name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained
    dropout = cfg.model.dropout

    supported_resnet_models = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    weights_map = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
        "resnet152": models.ResNet152_Weights.DEFAULT,
    }

    model_fn = supported_resnet_models.get(model_name)
    if model_fn is None:
        msg = f"Unsupported torchvision model: {model_name}"
        raise ValueError(msg)

    weights = weights_map[model_name] if pretrained else None
    model = model_fn(weights=weights)

    in_features = model.fc.in_features
    model.fc = _build_resnet_classifier(
        in_features=in_features,
        num_classes=num_classes,
        dropout=dropout,
    )

    if cfg.model.freeze_backbone:
        freeze_backbone_parameters(model)
        unfreeze_classifier_parameters(model.fc)

    return model


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