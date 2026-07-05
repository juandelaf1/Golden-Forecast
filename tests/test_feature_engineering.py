"""Tests for feature_engineering module."""
import pandas as pd
import numpy as np
import pytest

from src.feature_engineering import (
    create_daily_returns,
    create_daily_ranges,
    create_open_close_returns,
    create_moving_averages,
    create_rsi,
    create_macd,
    create_rolling_volatility,
    create_return_lags,
    create_targets,
    create_features,
    run_feature_engineering,
    ABSOLUTE_COLUMNS,
)


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    @pytest.fixture
    def sample_ohlc(self):
        """Create sample OHLC data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        base_price = 1900
        returns = np.random.normal(0, 0.01, n)
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'Date': dates,
            'gold_close': prices,
            'gold_high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'gold_low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'gold_open': prices * (1 + np.random.normal(0, 0.002, n)),
            'gold_volume': np.random.randint(1000, 10000, n),
            'dxy_close': 100 + np.random.normal(0, 0.5, n),
            'dxy_high': 100 + np.random.normal(0.5, 0.5, n),
            'dxy_low': 100 + np.random.normal(-0.5, 0.5, n),
            'dxy_open': 100 + np.random.normal(0, 0.5, n),
            'dxy_volume': np.random.randint(1000, 10000, n),
            'vix_close': 20 + np.random.normal(0, 2, n),
            'vix_high': 22 + np.random.normal(0, 2, n),
            'vix_low': 18 + np.random.normal(0, 2, n),
            'vix_open': 20 + np.random.normal(0, 2, n),
            'vix_volume': np.random.randint(1000, 10000, n),
            'tnx_close': 4.0 + np.random.normal(0, 0.1, n),
            'tnx_high': 4.2 + np.random.normal(0, 0.1, n),
            'tnx_low': 3.8 + np.random.normal(0, 0.1, n),
            'tnx_open': 4.0 + np.random.normal(0, 0.1, n),
            'tnx_volume': np.random.randint(1000, 10000, n),
        })
        return df

    def test_create_daily_returns(self, sample_ohlc):
        df = create_daily_returns(sample_ohlc.copy())
        assert 'gold_return' in df.columns
        assert 'dxy_return' in df.columns
        assert 'vix_return' in df.columns
        assert 'tnx_return' in df.columns
        assert df['gold_return'].notna().sum() == len(df) - 1  # first is NaN

    def test_create_daily_ranges(self, sample_ohlc):
        df = create_daily_ranges(sample_ohlc.copy())
        assert 'gold_daily_range' in df.columns
        assert (df['gold_daily_range'] >= 0).all()

    def test_create_open_close_returns(self, sample_ohlc):
        df = create_open_close_returns(sample_ohlc.copy())
        assert 'gold_open_close_return' in df.columns

    def test_create_moving_averages(self, sample_ohlc):
        df = create_moving_averages(sample_ohlc.copy())
        assert 'gold_ma_5' in df.columns
        assert 'gold_ma_20' in df.columns
        assert 'gold_close_vs_ma_5' in df.columns
        assert 'gold_close_vs_ma_20' in df.columns

    def test_create_rsi(self, sample_ohlc):
        df = create_rsi(sample_ohlc.copy())
        assert 'gold_rsi_14' in df.columns
        assert df['gold_rsi_14'].dropna().between(0, 100).all()

    def test_create_macd(self, sample_ohlc):
        df = create_macd(sample_ohlc.copy())
        assert 'gold_macd' in df.columns
        assert 'gold_macd_signal' in df.columns

    def test_create_rolling_volatility(self, sample_ohlc):
        df = create_daily_returns(sample_ohlc.copy())
        df = create_rolling_volatility(df)
        assert 'gold_volatility_14' in df.columns
        assert df['gold_volatility_14'].notna().sum() > 0

    def test_create_return_lags(self, sample_ohlc):
        df = create_daily_returns(sample_ohlc.copy())
        df = create_return_lags(df)
        assert 'gold_return_lag_1' in df.columns
        assert 'gold_return_lag_2' in df.columns

    def test_create_targets(self, sample_ohlc):
        df = create_targets(sample_ohlc.copy())
        assert 'target_binary' in df.columns
        assert 'target_multiclass' in df.columns
        assert set(df['target_binary'].dropna().unique()).issubset({0, 1})
        assert set(df['target_multiclass'].dropna().unique()).issubset({-1, 0, 1})

    def test_create_features_pipeline(self, sample_ohlc):
        df = create_features(sample_ohlc.copy())
        expected_features = [
            'gold_return', 'dxy_return', 'vix_return', 'tnx_return',
            'gold_daily_range', 'dxy_daily_range', 'vix_daily_range', 'tnx_daily_range',
            'gold_open_close_return', 'dxy_open_close_return', 'vix_open_close_return', 'tnx_open_close_return',
            'gold_ma_5', 'gold_ma_20',
            'gold_close_vs_ma_5', 'gold_close_vs_ma_20',
            'gold_rsi_14', 'gold_macd', 'gold_macd_signal',
            'gold_volatility_14',
            'gold_return_lag_1', 'gold_return_lag_2',
        ]
        for feat in expected_features:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_run_feature_engineering(self, sample_ohlc, tmp_path):
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        sample_ohlc.to_csv(input_path, index=False)

        result = run_feature_engineering(str(input_path), str(output_path))

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert 'target_binary' in df.columns
        assert 'target_multiclass' in df.columns
        for col in ABSOLUTE_COLUMNS:
            assert col not in df.columns, f"Absolute column {col} should be dropped"
        assert df.isna().sum().sum() == 0, "No NaN values should remain"