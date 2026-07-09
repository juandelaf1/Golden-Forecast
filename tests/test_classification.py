"""Tests for classification module."""

import pandas as pd
import numpy as np
import pytest

from src.classification import (
    get_models,
    get_voting_model,
    evaluate,
    overfitting_check,
    backtest,
    feature_importance,
    train_experimental_models,
    _make_horizon_target,
    _eval_model,
)


class TestClassification:
    """Tests for classification module."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 500
        X = pd.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
                "feat3": np.random.randn(n),
            }
        )
        y_binary = pd.Series(np.random.randint(0, 2, n))
        y_multiclass = pd.Series(np.random.randint(-1, 2, n))
        y_reg = pd.Series(np.random.randn(n))
        return X, y_binary, y_multiclass, y_reg

    def test_get_models(self):
        models = get_models()
        assert "Dummy" in models
        assert "Logistic Regression" in models
        assert "Random Forest" in models

    def test_get_voting_model(self):
        voting = get_voting_model()
        assert hasattr(voting, "fit")
        assert hasattr(voting, "predict")

    def test_evaluate(self, sample_data):
        X, y_binary, y_multiclass, _ = sample_data
        models = get_models()

        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_binary[:split], y_binary[split:]

        results = evaluate(models, X_train, X_test, y_train, y_test)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(models)
        assert "Accuracy" in results.columns
        assert "F1" in results.columns

    def test_overfitting_check(self, sample_data):
        X, y_binary, _, _ = sample_data
        models = get_models()

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_binary[:split], y_binary[split:]

        of = overfitting_check(models, X_train, X_test, y_train, y_test)
        assert isinstance(of, pd.DataFrame)
        assert "Overfitting" in of.columns

    def test_backtest(self, sample_data):
        X, y_binary, _, y_reg = sample_data
        models = get_models()

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_binary[:split], y_binary[split:]
        returns_test = y_reg[split:].values

        bt = backtest(models, X_train, X_test, y_train, y_test, returns_test)
        assert isinstance(bt, pd.DataFrame)
        assert "Estrategia" in bt.columns
        assert "Buy & Hold" in bt.columns

    def test_feature_importance(self, sample_data):
        X, y_binary, _, _ = sample_data
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y_binary)

        imp = feature_importance(model, X.columns.tolist())
        assert isinstance(imp, pd.Series)
        assert len(imp) == len(X.columns)
        assert imp.sum() > 0

    def test_make_horizon_target(self, sample_data):
        _, _, _, y_reg = sample_data
        y_train, y_test = y_reg[:400], y_reg[400:]

        for h in [1, 5, 10]:
            t_train, t_test, v_train, v_test = _make_horizon_target(y_train, y_test, h)
            assert len(t_train) == len(y_train)
            assert len(t_test) == len(y_test)
            assert v_train.sum() > 0
            assert v_test.sum() > 0

    def test_eval_model(self, sample_data):
        X, y_binary, _, _ = sample_data
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42, max_iter=1000)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_binary[:split], y_binary[split:]

        res = _eval_model(model, X_train, X_test, y_train, y_test, 1)
        assert "accuracy" in res
        assert "f1" in res
        assert 0 <= res["accuracy"] <= 1

    def test_train_experimental_models(self, sample_data):
        X, _, _, y_reg = sample_data
        X_train, X_test = X[:400], X[400:]
        y_reg_train, y_reg_test = y_reg[:400], y_reg[400:]

        results = train_experimental_models(X_train, X_test, y_reg_train, y_reg_test, X.columns.tolist())
        assert "models" in results
        assert "voting" in results
        assert "best_threshold_1d" in results
