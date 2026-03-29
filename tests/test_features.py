"""
tests/test_features.py
----------------------
Unit tests for src/features.py
"""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features


def _make_pbp(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal synthetic PBP DataFrame."""
    defaults = {
        "game_id": "2023_01_A_B",
        "play_id": 1,
        "season": 2023,
        "qtr": 2,
        "score_differential": 14,
        "yards_gained": 5,
        "epa": 0.1,
        "complete_pass": 1.0,
        "play_type_group": "receiving",
        "targeted": 1,
        "player_id": "P1",
        "player_name": "Test Player",
        "position": "WR",
        "total_snaps": 100,
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


class TestSnapFilter:
    def test_player_below_min_snaps_excluded(self):
        pbp = _make_pbp(
            [{"player_id": "P_LOW", "total_snaps": 30, "play_id": i} for i in range(10)]
            + [{"player_id": "P_HIGH", "total_snaps": 60, "play_id": i + 100} for i in range(50)]
        )
        features = build_features(pbp, min_snaps=50)
        assert "P_LOW" not in features["player_id"].values
        assert "P_HIGH" in features["player_id"].values

    def test_player_exactly_at_threshold_included(self):
        pbp = _make_pbp(
            [{"player_id": "P_EXACT", "total_snaps": 50, "play_id": i} for i in range(20)]
        )
        features = build_features(pbp, min_snaps=50)
        assert "P_EXACT" in features["player_id"].values

    def test_all_below_min_returns_empty(self):
        pbp = _make_pbp(
            [{"player_id": "P_LOW", "total_snaps": 10, "play_id": i} for i in range(5)]
        )
        features = build_features(pbp, min_snaps=50)
        assert features.empty


class TestSuccessRate:
    def test_all_positive_epa(self):
        pbp = _make_pbp(
            [{"epa": 0.5, "play_id": i, "total_snaps": 100} for i in range(60)]
        )
        features = build_features(pbp, min_snaps=50)
        assert features.loc[0, "success_rate"] == pytest.approx(1.0)

    def test_mixed_epa(self):
        rows = (
            [{"epa": 1.0, "play_id": i, "total_snaps": 100} for i in range(30)]
            + [{"epa": -1.0, "play_id": i + 100, "total_snaps": 100} for i in range(30)]
        )
        pbp = _make_pbp(rows)
        features = build_features(pbp, min_snaps=50)
        assert features.loc[0, "success_rate"] == pytest.approx(0.5)


class TestExplosiveness:
    def test_no_big_plays_returns_nan(self):
        pbp = _make_pbp(
            [{"yards_gained": 5, "play_id": i, "total_snaps": 100} for i in range(60)]
        )
        features = build_features(pbp, min_snaps=50)
        assert np.isnan(features.loc[0, "explosiveness"])

    def test_big_play_epa_averaged(self):
        rows = (
            [{"yards_gained": 25, "epa": 2.0, "play_id": i, "total_snaps": 100} for i in range(10)]
            + [{"yards_gained": 5, "epa": 0.1, "play_id": i + 50, "total_snaps": 100} for i in range(50)]
        )
        pbp = _make_pbp(rows)
        features = build_features(pbp, min_snaps=50)
        assert features.loc[0, "explosiveness"] == pytest.approx(2.0)


class TestReliabilityWR:
    def test_late_close_reception_rate(self):
        rows = (
            # Late-and-close catches
            [
                {"qtr": 4, "score_differential": 3, "complete_pass": 1.0, "targeted": 1,
                 "play_id": i, "total_snaps": 100}
                for i in range(6)
            ]
            # Late-and-close incompletions
            + [
                {"qtr": 4, "score_differential": 3, "complete_pass": 0.0, "targeted": 1,
                 "play_id": i + 50, "total_snaps": 100}
                for i in range(4)
            ]
            # Normal downs (should not affect reliability)
            + [{"qtr": 2, "play_id": i + 100, "total_snaps": 100} for i in range(50)]
        )
        pbp = _make_pbp(rows)
        features = build_features(pbp, min_snaps=50)
        assert features.loc[0, "reliability"] == pytest.approx(0.6)

    def test_no_late_close_plays_returns_nan(self):
        pbp = _make_pbp(
            [{"qtr": 1, "score_differential": 21, "play_id": i, "total_snaps": 100}
             for i in range(60)]
        )
        features = build_features(pbp, min_snaps=50)
        assert np.isnan(features.loc[0, "reliability"])


class TestReliabilityRB:
    def test_rb_ypc_in_late_close(self):
        rows = (
            [
                {
                    "player_id": "RB1", "player_name": "RB Test", "position": "RB",
                    "qtr": 4, "score_differential": 6,
                    "yards_gained": 10, "play_type_group": "rushing",
                    "complete_pass": pd.NA, "targeted": pd.NA,
                    "play_id": i, "total_snaps": 100,
                }
                for i in range(20)
            ]
        )
        pbp = _make_pbp(rows)
        features = build_features(pbp, min_snaps=50)
        assert features.loc[0, "reliability"] == pytest.approx(10.0)
