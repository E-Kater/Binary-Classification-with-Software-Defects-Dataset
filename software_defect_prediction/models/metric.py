from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Вычисление метрик классификации"""
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    return metrics


def calculate_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None
) -> Dict[str, float]:
    """Вычисление всех метрик, включая ROC-AUC если есть вероятности"""
    metrics = calculate_metrics(y_true, y_pred)

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics
