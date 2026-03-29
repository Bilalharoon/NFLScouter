"""
src/features.py
---------------
Feature engineering module for the NFL Scouting Engine.

Takes the tidy per-play DataFrame from ingestion.py and produces
one row per player with three process-oriented metrics:

    success_rate   – % of plays with positive EPA
    explosiveness  – mean EPA on plays with 20+ yards gained
    reliability    – reception rate (WR/TE) or YPC (RB) in
                     4th quarter, score differential within 8 pts

Players with fewer than min_snaps are excluded entirely.
Metrics with no qualifying plays (e.g. no 20+ yard plays) are NaN.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LATE_CLOSE_QTR = 4
_LATE_CLOSE_MAX_DIFF = 8
_BIG_PLAY_YARDS = 20


def _success_rate(group: pd.DataFrame) -> float:
    """Fraction of plays with positive EPA."""
    valid = group["epa"].dropna()
    return float((valid > 0).mean()) if len(valid) > 0 else np.nan


def _explosiveness(group: pd.DataFrame) -> float:
    """Mean EPA on plays where yards_gained >= 20."""
    big_plays = group.loc[group["yards_gained"] >= _BIG_PLAY_YARDS, "epa"].dropna()
    return float(big_plays.mean()) if len(big_plays) > 0 else np.nan


def _reliability(group: pd.DataFrame) -> float:
    """
    Late-and-close situational efficiency:
    - WR / TE: reception rate (completions / targets)
    - RB:      yards per carry
    """
    lc = group[
        (group["qtr"] == _LATE_CLOSE_QTR)
        & (group["score_differential"].abs() <= _LATE_CLOSE_MAX_DIFF)
    ]
    if lc.empty:
        return np.nan

    position = group["position"].iloc[0]
    if position in {"WR", "TE"}:
        targets = lc["targeted"].sum()
        if targets == 0:
            return np.nan
        completions = lc["complete_pass"].fillna(0).sum()
        return float(completions / targets)
    else:  # RB
        carries = lc[lc["play_type_group"] == "rushing"]
        if carries.empty:
            return np.nan
        return float(carries["yards_gained"].mean())


def build_features(pbp: pd.DataFrame, min_snaps: int = 50) -> pd.DataFrame:
    """
    Compute per-player aggregated metrics from a tidy per-play DataFrame.

    Parameters
    ----------
    pbp : pd.DataFrame
        Output of ``ingestion.load_pbp()``.
    min_snaps : int
        Minimum snap count required to include a player (default 50).

    Returns
    -------
    pd.DataFrame
        One row per player with columns:
        player_id, player_name, position, season, total_snaps,
        success_rate, explosiveness, reliability.
        Players below min_snaps are excluded.
        Metrics are NaN when there are no qualifying plays.
    """
    # Apply snap-count filter before aggregation
    eligible = pbp[pbp["total_snaps"].fillna(0) >= min_snaps].copy()
    excluded = pbp["player_id"].nunique() - eligible["player_id"].nunique()
    logger.info(
        "Snap filter (>= %d): %d players retained, %d excluded",
        min_snaps,
        eligible["player_id"].nunique(),
        excluded,
    )

    group_keys = ["player_id", "player_name", "position", "season", "total_snaps"]

    records = []
    for keys, grp in eligible.groupby(group_keys, sort=False):
        row = dict(zip(group_keys, keys))
        row["success_rate"] = _success_rate(grp)
        row["explosiveness"] = _explosiveness(grp)
        row["reliability"] = _reliability(grp)
        records.append(row)

    if not records:
        logger.warning("No players met the min_snaps threshold; returning empty DataFrame.")
        return pd.DataFrame(
            columns=group_keys + ["success_rate", "explosiveness", "reliability"]
        )

    features = pd.DataFrame(records)

    nan_counts = features[["success_rate", "explosiveness", "reliability"]].isna().sum()
    logger.info("NaN counts per metric:\n%s", nan_counts.to_string())

    return features.reset_index(drop=True)
