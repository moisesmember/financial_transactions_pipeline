"""Threshold tuning utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float = 2.0,
    min_precision: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Select a threshold on validation data by maximizing F-beta."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    best_threshold = 0.5
    best_metrics = {"fbeta": -1.0, "precision": 0.0, "recall": 0.0}

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision < min_precision:
            continue
        recall = recall_score(y_true, y_pred, zero_division=0)
        fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if fbeta > best_metrics["fbeta"]:
            best_threshold = float(threshold)
            best_metrics = {
                "fbeta": float(fbeta),
                "precision": float(precision),
                "recall": float(recall),
            }

    if best_metrics["fbeta"] < 0:
        return 0.5, {"fbeta": 0.0, "precision": 0.0, "recall": 0.0}
    return best_threshold, best_metrics
