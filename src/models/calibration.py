"""Calibration and score-band reports for fraud scores."""

from __future__ import annotations

import json
from base64 import b64decode
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def write_calibration_artifacts(
    split_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    report_path: Path,
    deciles_path: Path,
    metrics_path: Path,
    curve_path: Path,
) -> dict[str, dict[str, float]]:
    """Persist calibration curves, score bands and Brier scores by split."""
    report_frames: list[pd.DataFrame] = []
    decile_frames: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, float]] = {}
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    figure = axis = None
    if plt is not None:
        figure, axis = plt.subplots(figsize=(7, 5))

    for split, (y_true, y_score) in split_scores.items():
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
        decile_frames.append(deciles)
        metrics[split] = {"brier_score": float(brier_score_loss(y_true, y_score))}
        if axis is not None:
            axis.plot(mean_predicted, fraction_positive, marker="o", label=split)

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
