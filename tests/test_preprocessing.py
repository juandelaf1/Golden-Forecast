"""Tests for preprocessing module."""
import pandas as pd
import pytest

from src.preprocessing import (
    convert_date,
    rename_columns,
    drop_uninformative_columns,
    remove_missing_values,
    clean_gold_data,
)


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_convert_date(self):
        df = pd.DataFrame({'Date': ['2023-01-01', '2023-01-02']})
        result = convert_date(df)
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])

    def test_rename_columns(self):
        cols = {
            "Gold_('Close', 'GC=F')": "gold_close",
            "DXY_('Close', 'DX-Y.NYB')": "dxy_close",
        }
        df = pd.DataFrame({k: [1.0] for k in cols})
        result = rename_columns(df)
        assert list(result.columns) == ['gold_close', 'dxy_close']

    def test_drop_uninformative_columns(self):
        df = pd.DataFrame({'gold_close': [1], 'dxy_volume': [2], 'vix_volume': [3]})
        result = drop_uninformative_columns(df)
        assert 'dxy_volume' not in result.columns
        assert 'vix_volume' not in result.columns
        assert 'gold_close' in result.columns

    def test_remove_missing_values(self):
        df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, 6]})
        result = remove_missing_values(df)
        assert len(result) == 2
        assert result['a'].notna().all()

    def test_clean_gold_data_integration(self, tmp_path):
        # Create minimal raw data
        raw = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            "Gold_('Close', 'GC=F')": [1900, 1910, 1920],
            "Gold_('High', 'GC=F')": [1910, 1920, 1930],
            "Gold_('Low', 'GC=F')": [1890, 1900, 1910],
            "Gold_('Open', 'GC=F')": [1895, 1905, 1915],
            "Gold_('Volume', 'GC=F')": [100, 100, 100],
            "DXY_('Close', 'DX-Y.NYB')": [100, 101, 102],
            "DXY_('High', 'DX-Y.NYB')": [101, 102, 103],
            "DXY_('Low', 'DX-Y.NYB')": [99, 100, 101],
            "DXY_('Open', 'DX-Y.NYB')": [100, 101, 102],
            "DXY_('Volume', 'DX-Y.NYB')": [100, 100, 100],
            "VIX_('Close', '^VIX')": [20, 21, 22],
            "VIX_('High', '^VIX')": [22, 23, 24],
            "VIX_('Low', '^VIX')": [18, 19, 20],
            "VIX_('Open', '^VIX')": [19, 20, 21],
            "VIX_('Volume', '^VIX')": [100, 100, 100],
            "TNX_('Close', '^TNX')": [4.0, 4.1, 4.2],
            "TNX_('High', '^TNX')": [4.2, 4.3, 4.4],
            "TNX_('Low', '^TNX')": [3.8, 3.9, 4.0],
            "TNX_('Open', '^TNX')": [3.9, 4.0, 4.1],
            "TNX_('Volume', '^TNX')": [100, 100, 100],
        })
        raw_path = tmp_path / "raw.csv"
        clean_path = tmp_path / "clean.csv"
        raw.to_csv(raw_path, index=False)

        result = clean_gold_data(str(raw_path), str(clean_path))

        assert 'gold_close' in result.columns
        assert 'dxy_volume' not in result.columns
        assert result['Date'].dtype == 'datetime64[ns]'
        assert len(result) == 3