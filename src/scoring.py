"""
src/scoring.py
--------------
CSS (Composite Scout Score) ranking module.

Applies MinMaxScaler to each metric independently, then computes
a weighted average to produce a CSS_Score in [0, 1].

NaN values in any metric are imputed with the column median before
scaling so that players missing one situational metric are not dropped.
Columns with the scaled values are retained for downstream ML use.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "success_rate": 0.40,
    "explosiveness": 0.35,
    "reliability": 0.25,
}

METRIC_COLS = list(DEFAULT_WEIGHTS.keys())
SCALED_COLS = [f"{m}_scaled" for m in METRIC_COLS]

# Tier thresholds (based on scaled CSS_Score)
TIER_HIGH_CEILING = 0.70
TIER_HIGH_FLOOR = 0.40


def _assign_tier(score: float) -> str:
    if score >= TIER_HIGH_CEILING:
        return "High-Ceiling"
    elif score >= TIER_HIGH_FLOOR:
        return "High-Floor"
    return "Developmental"


def compute_css(
    features: pd.DataFrame,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """
    Scale metrics and compute the Composite Scout Score.

    Parameters
    ----------
    features : pd.DataFrame
        Output of ``features.build_features()``.
    weights : dict
        Mapping of metric name → weight. Must sum to 1.0.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with:
        {metric}_scaled columns, CSS_Score (float in [0,1]),
        tier (str), sorted descending by CSS_Score.
    """
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Weights must sum to 1.0; got {sum(weights.values()):.4f}")

    scored = features.copy()

    # Median imputation for NaN so players missing one metric aren't dropped.
    # If the entire column is NaN (median is also NaN), fall back to 0.0.
    for col in METRIC_COLS:
        median = scored[col].median()
        if pd.isna(median):
            median = 0.0
        n_imputed = scored[col].isna().sum()
        if n_imputed:
            logger.info("Imputing %d NaN(s) in '%s' with median %.4f", n_imputed, col, median)
        scored[col] = scored[col].fillna(median)

    # MinMaxScaler applied independently per metric
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(scored[METRIC_COLS])
    for i, scaled_col in enumerate(SCALED_COLS):
        scored[scaled_col] = scaled_values[:, i]

    # Weighted average CSS_Score
    scored["CSS_Score"] = sum(
        weights[metric] * scored[f"{metric}_scaled"] for metric in METRIC_COLS
    )

    # Tier labels
    scored["tier"] = scored["CSS_Score"].apply(_assign_tier)

    scored = scored.sort_values("CSS_Score", ascending=False).reset_index(drop=True)
    scored.index += 1  # 1-based rank
    scored.index.name = "rank"

    logger.info(
        "CSS computed for %d players. Tier distribution:\n%s",
        len(scored),
        scored["tier"].value_counts().to_string(),
    )
    return scored
