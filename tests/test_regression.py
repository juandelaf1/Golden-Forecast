"""Tests for regression module."""

import pandas as pd
import numpy as np
import pytest

from src.regression import (
    get_regression_models,
    TARGET_SPECS,
    information_coefficient,
    mean_absolute_percentage_error,
    evaluate_regression,
)


class TestRegressionModels:
    """Tests for regression model definitions."""

    def test_get_regression_models_count(self):
        models = get_regression_models()
        assert len(models) >= 5

    def test_get_regression_models_has_rf(self):
        models = get_regression_models()
        assert "Random Forest" in models

    def test_get_regression_models_has_ridge(self):
        models = get_regression_models()
        assert "Ridge" in models


class TestRegressionTargetSpecs:
    """Tests for target specifications."""

    def test_target_specs_count(self):
        assert len(TARGET_SPECS) == 4

    def test_target_specs_keys(self):
        expected = {"realized_vol_20d", "future_atr_20d", "max_drawdown_20d", "fair_value_dist"}
        assert set(TARGET_SPECS.keys()) == expected

    def test_target_specs_horizon(self):
        for name, spec in TARGET_SPECS.items():
            assert "horizon" in spec
            assert spec["horizon"] >= 1


class TestRegressionMetrics:
    """Tests for regression metric functions."""

    def test_information_coefficient_output(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        rho, r = information_coefficient(y_true, y_pred)
        assert isinstance(rho, float)
        assert isinstance(r, float)
        assert -1 <= rho <= 1
        assert -1 <= r <= 1

    def test_mean_absolute_percentage_error(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 300.0])
        mape = mean_absolute_percentage_error(y_true, y_pred)
        assert isinstance(mape, float)
        assert mape >= 0

    def test_evaluate_regression_output(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        y = np.random.randn(n)
        split = 80
        models = get_regression_models()
        results = evaluate_regression(models, X[:split], X[split:], y[:split], y[split:])
        assert isinstance(results, pd.DataFrame)
        assert "Modelo" in results.columns
        assert "MAE" in results.columns
        assert "RMSE" in results.columns
        assert "R²" in results.columns
        assert len(results) == len(models)
