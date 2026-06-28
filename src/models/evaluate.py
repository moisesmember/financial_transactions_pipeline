"""Evaluation metrics for imbalanced fraud detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    beta: float = 2.0,
) -> dict[str, float]:
    """Evaluate fraud scores using precision/recall-oriented metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fbeta": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "alerts": float(tp + fp),
        "alert_rate": float((tp + fp) / len(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def precision_recall_table(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, np.ndarray]:
    """Return precision, recall and threshold arrays."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}
