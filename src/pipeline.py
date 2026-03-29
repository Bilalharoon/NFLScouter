"""
src/pipeline.py
---------------
End-to-end orchestration of the NFL Scouting Engine.

Calls ingestion → feature engineering → scoring → visualization
in sequence and returns the final tidy DataFrame suitable for
downstream K-Means clustering or other ML workflows.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from src.ingestion import load_pbp
from src.features import build_features
from src.scoring import compute_css, DEFAULT_WEIGHTS
from src.visualization import plot_css

logger = logging.getLogger(__name__)

DEFAULT_SEASONS: List[int] = [2023, 2024, 2025]
DEFAULT_MIN_SNAPS: int = 50
DEFAULT_OUTPUT_PATH: str = "outputs/css_scatter.png"


def run(
    seasons: List[int] = DEFAULT_SEASONS,
    min_snaps: int = DEFAULT_MIN_SNAPS,
    weights: Optional[Dict[str, float]] = None,
    output_path: str = DEFAULT_OUTPUT_PATH,
    save_plot: bool = True,
) -> pd.DataFrame:
    """
    Execute the full NFL scouting pipeline.

    Parameters
    ----------
    seasons : list of int
        NFL seasons to include (default [2023, 2024, 2025]).
    min_snaps : int
        Minimum offensive snaps to qualify (default 50).
    weights : dict, optional
        CSS metric weights. Keys must be 'success_rate', 'explosiveness',
        'reliability'. Values must sum to 1.0. Uses DEFAULT_WEIGHTS if None.
    output_path : str
        File path for the scatter plot PNG.
    save_plot : bool
        If False, skip the visualization step (useful for unit tests).

    Returns
    -------
    pd.DataFrame
        Ranked player DataFrame with columns:
        player_id, player_name, position, season, total_snaps,
        success_rate, explosiveness, reliability,
        success_rate_scaled, explosiveness_scaled, reliability_scaled,
        CSS_Score, tier.
        Index is 1-based rank.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    logger.info("=== NFL Scouting Pipeline START ===")
    logger.info("Seasons: %s | Min snaps: %d | Weights: %s", seasons, min_snaps, weights)

    # Step 1 — Ingest
    pbp = load_pbp(seasons=seasons)

    # Step 2 — Feature engineering
    features = build_features(pbp=pbp, min_snaps=min_snaps)

    # Step 3 — Scoring
    scored = compute_css(features=features, weights=weights)

    # Step 4 — Visualization
    if save_plot:
        plot_css(scored=scored, output_path=output_path)

    logger.info("=== NFL Scouting Pipeline COMPLETE. %d players ranked. ===", len(scored))
    return scored
