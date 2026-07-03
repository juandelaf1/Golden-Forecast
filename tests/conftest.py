"""Shared pytest fixtures for Golden Forecast tests."""
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, 'src')


@pytest.fixture(scope="session")
def gold_features():
    """Load gold-features.csv once per session."""
    return pd.read_csv('data/processed/gold-features.csv', parse_dates=['Date'])


@pytest.fixture(scope="session")
def gold_clean():
    """Load gold-clean.csv once per session."""
    return pd.read_csv('data/processed/gold-clean.csv', parse_dates=['Date'])


@pytest.fixture(scope="session")
def feature_columns():
    """List of feature column names (excludes targets and Date)."""
    exclude = ['Date', 'target_binary', 'target_multiclass']
    df = pd.read_csv('data/processed/gold-features.csv', nrows=1)
    return [c for c in df.columns if c not in exclude]


@pytest.fixture(scope="session")
def classification_targets(gold_features):
    """Target columns for classification."""
    return gold_features['target_binary'], gold_features['target_multiclass']


@pytest.fixture
def sample_data(gold_features, feature_columns):
    """Small sample for fast tests."""
    return gold_features.sample(100, random_state=42).reset_index(drop=True)