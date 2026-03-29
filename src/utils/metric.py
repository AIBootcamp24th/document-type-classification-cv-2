from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="micro"))


def compute_f1_macro(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1_micro": compute_f1_micro(y_true, y_pred),
        "f1_macro": compute_f1_macro(y_true, y_pred),
    }


def get_primary_metric(cfg: Any, metrics: dict[str, float]) -> float:
    metric_name = cfg.metric.primary

    if metric_name not in metrics:
        msg = f"Primary metric '{metric_name}' not found in computed metrics."
        raise KeyError(msg)

    return metrics[metric_name]


def to_numpy(predictions: Any, targets: Any) -> tuple[np.ndarray, np.ndarray]:
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    return pred_np, target_np
