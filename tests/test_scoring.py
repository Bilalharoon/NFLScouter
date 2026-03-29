"""
tests/test_scoring.py
---------------------
Unit tests for src/scoring.py
"""

import numpy as np
import pandas as pd
import pytest

from src.scoring import compute_css, DEFAULT_WEIGHTS, METRIC_COLS, SCALED_COLS


def _make_features(**kwargs) -> pd.DataFrame:
    """Return a minimal features DataFrame with sensible defaults."""
    base = pd.DataFrame(
        {
            "player_id": ["P1", "P2", "P3"],
            "player_name": ["Alpha", "Bravo", "Charlie"],
            "position": ["WR", "RB", "TE"],
            "season": [2023, 2023, 2023],
            "total_snaps": [100, 80, 60],
            "success_rate": [0.60, 0.50, 0.45],
            "explosiveness": [1.5, 0.8, 1.2],
            "reliability": [0.70, 7.5, 0.55],
        }
    )
    for col, val in kwargs.items():
        base[col] = val
    return base


class TestCSSOutputShape:
    def test_returns_all_expected_columns(self):
        scored = compute_css(_make_features())
        expected = METRIC_COLS + SCALED_COLS + ["CSS_Score", "tier"]
        for col in expected:
            assert col in scored.columns, f"Missing column: {col}"

    def test_row_count_unchanged(self):
        features = _make_features()
        scored = compute_css(features)
        assert len(scored) == len(features)

    def test_index_is_one_based_rank(self):
        scored = compute_css(_make_features())
        assert scored.index.name == "rank"
        assert scored.index[0] == 1
        assert scored.index[-1] == len(scored)


class TestScalingBounds:
    def test_scaled_values_in_0_1(self):
        scored = compute_css(_make_features())
        for col in SCALED_COLS:
            assert scored[col].min() >= 0.0 - 1e-9
            assert scored[col].max() <= 1.0 + 1e-9

    def test_css_score_in_0_1(self):
        scored = compute_css(_make_features())
        assert scored["CSS_Score"].min() >= 0.0 - 1e-9
        assert scored["CSS_Score"].max() <= 1.0 + 1e-9

    def test_sorted_descending_by_css(self):
        scored = compute_css(_make_features())
        assert list(scored["CSS_Score"]) == sorted(scored["CSS_Score"], reverse=True)


class TestTierLabels:
    def test_high_ceiling_assigned_correctly(self):
        # Force a player to have max metrics → should be High-Ceiling
        feat = _make_features(
            success_rate=[1.0, 0.0, 0.0],
            explosiveness=[10.0, 0.0, 0.0],
            reliability=[1.0, 0.0, 0.0],
        )
        scored = compute_css(feat)
        assert scored.iloc[0]["tier"] == "High-Ceiling"

    def test_developmental_assigned_correctly(self):
        feat = _make_features(
            success_rate=[1.0, 0.0, 0.0],
            explosiveness=[10.0, 0.0, 0.0],
            reliability=[1.0, 0.0, 0.0],
        )
        scored = compute_css(feat)
        assert scored.iloc[-1]["tier"] == "Developmental"

    def test_all_tier_values_valid(self):
        scored = compute_css(_make_features())
        valid_tiers = {"High-Ceiling", "High-Floor", "Developmental"}
        assert set(scored["tier"].unique()).issubset(valid_tiers)


class TestNaNHandling:
    def test_nan_imputed_does_not_drop_player(self):
        # Player P2 has NaN explosiveness (no big plays)
        feat = _make_features(explosiveness=[1.5, np.nan, 1.2])
        scored = compute_css(feat)
        assert len(scored) == 3

    def test_all_nan_metric_imputes_with_median(self):
        feat = _make_features(explosiveness=[np.nan, np.nan, np.nan])
        # Should not raise and should produce 0.5 for all scaled values
        scored = compute_css(feat)
        assert scored["explosiveness_scaled"].isna().sum() == 0


class TestWeightValidation:
    def test_weights_not_summing_to_1_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_css(_make_features(), weights={"success_rate": 0.5, "explosiveness": 0.5, "reliability": 0.5})

    def test_custom_weights_accepted(self):
        custom = {"success_rate": 0.5, "explosiveness": 0.3, "reliability": 0.2}
        scored = compute_css(_make_features(), weights=custom)
        assert "CSS_Score" in scored.columns
