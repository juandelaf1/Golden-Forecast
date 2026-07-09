"""Tests for training and evaluation modules."""

import os
import pickle
import pytest


class TestTrainConstants:
    """Tests for training module constants."""

    def test_multiclass_encode_decode(self):
        from src.models.train import MULTICLASS_ENCODE, MULTICLASS_DECODE

        assert MULTICLASS_ENCODE == {-1: 0, 0: 1, 1: 2}
        assert MULTICLASS_DECODE == {0: -1, 1: 0, 2: 1}

    def test_no_feature_columns(self):
        from src.models.train import NO_FEATURE_COLUMN

        assert "Date" in NO_FEATURE_COLUMN
        assert "target_binary" in NO_FEATURE_COLUMN
        assert "target_multiclass" in NO_FEATURE_COLUMN

    def test_models_defined(self):
        from src.models.train import MODELS

        assert "lr" in MODELS
        assert "rf" in MODELS
        assert "xgb" in MODELS
        assert "lr_strong_reg" in MODELS
        assert len(MODELS) >= 6


class TestEvaluateConstants:
    """Tests for evaluation module constants."""

    def test_evaluate_multiclass_encode(self):
        from src.models.evaluate import MULTICLASS_ENCODE, MULTICLASS_DECODE

        assert MULTICLASS_ENCODE == {-1: 0, 0: 1, 1: 2}
        assert MULTICLASS_DECODE == {0: -1, 1: 0, 2: 1}


class TestModelArtifacts:
    """Tests that model files exist on disk."""

    MODELS_DIR = "models"

    def test_scaler_exists(self):
        path = os.path.join(self.MODELS_DIR, "scaler.pkl")
        if not os.path.exists(path):
            pytest.skip("scaler.pkl not found (run train.py first)")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert hasattr(obj, "transform")

    def test_primary_model_exists(self):
        path = os.path.join(self.MODELS_DIR, "lr_strong_reg_binary.pkl")
        if not os.path.exists(path):
            pytest.skip("lr_strong_reg_binary.pkl not found (run train.py first)")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert hasattr(obj, "predict")

    def test_model_count(self):
        from src.models.train import MODELS

        existing = [n for n in MODELS if os.path.exists(os.path.join(self.MODELS_DIR, f"{n}.pkl"))]
        if len(existing) < 3:
            pytest.skip("Fewer than 3 models found (run train.py first)")
        assert len(existing) >= 3
