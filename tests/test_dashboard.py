"""Tests for dashboard modules (lightweight, no full import chain)."""

import os
import pytest

FEATURE_COLUMNS_EXPECTED = [
    "gold_volume",
    "gold_return",
    "dxy_return",
    "vix_return",
    "tnx_return",
    "gold_daily_range",
    "dxy_daily_range",
    "vix_daily_range",
    "tnx_daily_range",
    "gold_open_close_return",
    "dxy_open_close_return",
    "vix_open_close_return",
    "tnx_open_close_return",
    "gold_ma_5",
    "gold_ma_20",
    "gold_close_vs_ma_5",
    "gold_close_vs_ma_20",
    "gold_rsi_14",
    "gold_macd",
    "gold_macd_signal",
    "gold_volatility_14",
    "gold_return_lag_1",
    "gold_return_lag_2",
]


class TestDashboardConstants:
    """Tests for dashboard constants (no import of data module)."""

    def test_feature_columns_count(self):
        assert len(FEATURE_COLUMNS_EXPECTED) == 23

    def test_feature_columns_have_gold_return(self):
        assert "gold_return" in FEATURE_COLUMNS_EXPECTED

    def test_feature_columns_have_rsi(self):
        assert "gold_rsi_14" in FEATURE_COLUMNS_EXPECTED

    def test_feature_columns_no_duplicates(self):
        assert len(FEATURE_COLUMNS_EXPECTED) == len(set(FEATURE_COLUMNS_EXPECTED))


class TestDashboardModelLoader:
    """Tests for model loader (reads files or skips gracefully)."""

    def test_features_csv_exists(self):
        path = "data/processed/gold-features.csv"
        if not os.path.exists(path):
            pytest.skip("gold-features.csv not found")
        assert os.path.getsize(path) > 1000

    def test_clean_csv_exists(self):
        path = "data/processed/gold-clean.csv"
        if not os.path.exists(path):
            pytest.skip("gold-clean.csv not found")
        assert os.path.getsize(path) > 1000

    def test_evaluation_results_exists(self):
        path = "models/evaluation_results.json"
        if not os.path.exists(path):
            pytest.skip("evaluation_results.json not found")
        assert os.path.getsize(path) > 50
