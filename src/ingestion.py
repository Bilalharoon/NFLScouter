"""
src/ingestion.py
----------------
Data ingestion module for the NFL Scouting Engine.

Loads play-by-play data and roster/snap-count metadata via nflreadpy,
converts from Polars to pandas, and returns a tidy per-play DataFrame
filtered to skill-position rookies.
"""

from __future__ import annotations

import logging
from typing import List

import nflreadpy as nfl
import pandas as pd

logger = logging.getLogger(__name__)

SKILL_POSITIONS: set[str] = {"WR", "RB", "TE"}


def _load_rookies(seasons: List[int]) -> pd.DataFrame:
    """Return a DataFrame of rookie skill-position players indexed by gsis_id."""
    logger.info("Loading rosters for seasons: %s", seasons)
    rosters = nfl.load_rosters(seasons=seasons).to_pandas()

    # nflreadpy roster columns: gsis_id (player key), full_name, position, season
    rookies = (
        rosters[
            (rosters["years_exp"] == 0) & (rosters["position"].isin(SKILL_POSITIONS))
        ][["gsis_id", "full_name", "position", "season"]]
        .drop_duplicates()
        .rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    )

    logger.info("Found %d rookie skill-position player-seasons", len(rookies))
    return rookies


def _snap_counts_from_pbp(pbp_long: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-player snap exposure from PBP play counts.

    Using load_snap_counts() requires matching pfr_player_id → gsis_id across
    different ID systems, which is unreliable. Play count from PBP is a
    consistent proxy for offensive exposure (one row ≈ one snap involvement).
    """
    return (
        pbp_long.groupby(["player_id", "season"], as_index=False)
        .size()
        .rename(columns={"size": "total_snaps"})
    )


def load_pbp(seasons: List[int]) -> pd.DataFrame:
    """
    Load play-by-play data for the given seasons, filtered to rookie
    skill-position players.

    Parameters
    ----------
    seasons : list of int
        NFL seasons to load (e.g. [2023, 2024, 2025]).

    Returns
    -------
    pd.DataFrame
        Tidy per-play DataFrame enriched with player metadata:
        player_id, player_name, position, season, total_snaps.
    """
    logger.info("Loading PBP data for seasons: %s", seasons)
    raw_pbp = nfl.load_pbp(seasons=seasons).to_pandas()

    rookies = _load_rookies(seasons)

    # ------------------------------------------------------------------ #
    # Build a unified per-play table for skill positions.                 #
    # nflverse PBP represents receiving plays and rushing plays           #
    # separately; we stack them into a single "player_id / play" view.   #
    # ------------------------------------------------------------------ #

    # Receiving plays (WR, TE, and receiving RBs)
    recv = raw_pbp[raw_pbp["receiver_player_id"].notna()][
        [
            "game_id", "play_id", "season", "qtr", "score_differential",
            "yards_gained", "epa", "complete_pass",
            "receiver_player_id",
        ]
    ].rename(columns={"receiver_player_id": "player_id"})
    recv["play_type_group"] = "receiving"
    recv["targeted"] = 1  # every receiving row is a target

    # Rushing plays
    rush = raw_pbp[raw_pbp["rusher_player_id"].notna()][
        [
            "game_id", "play_id", "season", "qtr", "score_differential",
            "yards_gained", "epa", "rusher_player_id",
        ]
    ].rename(columns={"rusher_player_id": "player_id"})
    rush["play_type_group"] = "rushing"
    rush["complete_pass"] = pd.NA
    rush["targeted"] = pd.NA

    pbp_long = pd.concat([recv, rush], ignore_index=True)

    # Join to rookies (inner join keeps only rookies)
    pbp_long = pbp_long.merge(
        rookies, on=["player_id", "season"], how="inner"
    )

    # Derive total_snaps from PBP play counts (avoids cross-ID-system join)
    snap_totals = _snap_counts_from_pbp(pbp_long)
    pbp_long = pbp_long.merge(
        snap_totals, on=["player_id", "season"], how="left"
    )

    logger.info(
        "PBP assembled: %d plays across %d unique player-seasons",
        len(pbp_long),
        pbp_long[["player_id", "season"]].drop_duplicates().shape[0],
    )
    return pbp_long
