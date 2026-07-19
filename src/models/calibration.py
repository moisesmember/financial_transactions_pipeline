"""Calibration and score-band reports for fraud scores."""

from __future__ import annotations

import json
from base64 import b64decode
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from src.models.evaluate import evaluate_binary_classifier


def write_calibration_artifacts(
    split_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    report_path: Path,
    deciles_path: Path,
    metrics_path: Path,
    curve_path: Path,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 25.0,
) -> dict[str, object]:
    """Persist calibration curves, score bands and Brier scores by split."""
    report_frames: list[pd.DataFrame] = []
    decile_frames: list[pd.DataFrame] = []
    metrics: dict[str, object] = {}
    calibration_target, calibration_score = split_scores.get(
        "validation", next(iter(split_scores.values()))
    )
    calibrators: dict[str, object | None] = {"none": None}
    if len(np.unique(calibration_target)) == 2:
        sigmoid = LogisticRegression(random_state=42)
        sigmoid.fit(np.asarray(calibration_score).reshape(-1, 1), calibration_target)
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(calibration_score, calibration_target)
        calibrators.update({"sigmoid": sigmoid, "isotonic": isotonic})
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    figure = axis = None
    if plt is not None:
        figure, axis = plt.subplots(figsize=(7, 5))

    method_metrics: dict[str, dict[str, dict[str, object]]] = {}
    for method, calibrator in calibrators.items():
        method_metrics[method] = {}
        for split, (y_true, raw_score) in split_scores.items():
            y_score = _calibrate(np.asarray(raw_score, dtype=float), method, calibrator)
            y_true = np.asarray(y_true).astype(int)
            y_score = np.asarray(y_score).astype(float)
            fraction_positive, mean_predicted = calibration_curve(
                y_true,
                y_score,
                n_bins=10,
                strategy="quantile",
            )
            report_frames.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "split": split,
                        "bin": np.arange(1, len(mean_predicted) + 1),
                        "mean_score": mean_predicted,
                        "actual_positive_rate": fraction_positive,
                    }
                )
            )
            score_frame = pd.DataFrame({"score": y_score, "target": y_true})
            score_frame["score_band"] = pd.qcut(
                score_frame["score"].rank(method="first"),
                q=10,
                labels=False,
                duplicates="drop",
            )
            deciles = (
                score_frame.groupby("score_band", observed=True)
                .agg(
                    records=("target", "size"),
                    positives=("target", "sum"),
                    mean_score=("score", "mean"),
                    min_score=("score", "min"),
                    max_score=("score", "max"),
                    actual_positive_rate=("target", "mean"),
                )
                .reset_index()
            )
            deciles.insert(0, "split", split)
            deciles.insert(0, "method", method)
            decile_frames.append(deciles)
            classification = evaluate_binary_classifier(y_true, y_score, threshold=0.5)
            business_cost = (
                float(classification["fp"]) * false_positive_cost
                + float(classification["fn"]) * false_negative_cost
            )
            method_metrics[method][split] = {
                "brier_score": float(brier_score_loss(y_true, y_score)),
                "expected_calibration_error": _expected_calibration_error(
                    y_true, y_score
                ),
                "pr_auc": (
                    float(average_precision_score(y_true, y_score))
                    if len(np.unique(y_true)) == 2
                    else None
                ),
                "precision": classification["precision"],
                "recall": classification["recall"],
                "alert_rate": classification["alert_rate"],
                "business_cost": business_cost,
            }
            if method == "none":
                metrics[split] = {
                    "brier_score": method_metrics[method][split]["brier_score"]
                }
            if axis is not None:
                axis.plot(
                    mean_predicted,
                    fraction_positive,
                    marker="o",
                    label=f"{method}:{split}",
                )

    validation_brier = {
        method: values.get("validation", {}).get("brier_score")
        for method, values in method_metrics.items()
    }
    selected_method = min(
        (item for item in validation_brier.items() if item[1] is not None),
        key=lambda item: float(item[1]),
    )[0]
    metrics["methods"] = method_metrics
    metrics["selected_method_on_validation"] = selected_method
    metrics["calibration_uses_test"] = False
    metrics["calibration_uses_out_of_time"] = False

    pd.concat(report_frames, ignore_index=True).to_csv(report_path, index=False)
    pd.concat(decile_frames, ignore_index=True).to_csv(deciles_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if axis is not None and figure is not None and plt is not None:
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="perfect")
        axis.set(xlabel="Mean predicted score", ylabel="Observed positive rate")
        axis.legend()
        figure.tight_layout()
        figure.savefig(curve_path, dpi=150)
        plt.close(figure)
    else:
        curve_path.write_bytes(
            b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                "+A8AAQUBAScY42YAAAAASUVORK5CYII="
            )
        )
    return metrics


def _calibrate(scores: np.ndarray, method: str, calibrator: object | None) -> np.ndarray:
    if method == "none" or calibrator is None:
        return np.clip(scores, 0.0, 1.0)
    if method == "sigmoid":
        return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]  # type: ignore[attr-defined]
    return calibrator.predict(scores)  # type: ignore[attr-defined]


def _expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(y_score, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(
            float(np.mean(y_true[mask])) - float(np.mean(y_score[mask]))
        )
    return float(error)
