"""Model feature importance reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd


GEO_TOKENS = ("merchant_city", "merchant_state", "zip", "latitude", "longitude")
TEMPORAL_TOKENS = ("hour", "dayofweek", "month", "weekend", "night", "previous")
BEHAVIOR_TOKENS = ("mean_5", "std_5", "transactions_seen", "amount_to_mean")
RISK_TOKENS = ("mcc", "use_chip", "amount")


def build_feature_importance(pipeline) -> pd.DataFrame:
    """Return coefficients/importances enriched with governance metadata."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_)
        values = values[0] if values.ndim > 1 else values
        odds_ratio = np.exp(np.clip(values, -50, 50))
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        odds_ratio = np.full(len(values), np.nan)
    else:
        return pd.DataFrame()
    if len(names) != len(values):
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "feature_name": names,
            "importance": values.astype(float),
            "absolute_importance": np.abs(values).astype(float),
            "direction": np.where(values >= 0, "positive", "negative"),
            "odds_ratio": odds_ratio.astype(float),
        }
    )
    frame["feature_group"] = frame["feature_name"].map(_feature_group)
    frame["is_geo_feature"] = frame["feature_name"].str.contains("|".join(GEO_TOKENS))
    frame["is_temporal_feature"] = frame["feature_name"].str.contains("|".join(TEMPORAL_TOKENS))
    frame["is_behavioral_feature"] = frame["feature_name"].str.contains("|".join(BEHAVIOR_TOKENS))
    frame["is_risk_feature"] = frame["feature_name"].str.contains("|".join(RISK_TOKENS))
    return frame.sort_values("absolute_importance", ascending=False).reset_index(drop=True)


def _feature_group(name: str) -> str:
    lower = name.lower()
    for group, tokens in (
        ("geographic", GEO_TOKENS),
        ("temporal", TEMPORAL_TOKENS),
        ("behavioral", BEHAVIOR_TOKENS),
        ("risk", RISK_TOKENS),
    ):
        if any(token in lower for token in tokens):
            return group
    return "other"
