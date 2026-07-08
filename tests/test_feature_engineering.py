import sys
from pathlib import Path

import pandas as pd

# Permite importar archivos desde la carpeta src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from feature_engineering import create_features, create_targets


def create_sample_dataframe():
    """
    Crea un dataset pequeño de prueba con las columnas necesarias
    para ejecutar el feature engineering.
    """
    rows = 40

    return pd.DataFrame(
        {
            "Date": pd.date_range(start="2020-01-01", periods=rows, freq="D"),
            "gold_open": range(100, 100 + rows),
            "gold_high": range(102, 102 + rows),
            "gold_low": range(99, 99 + rows),
            "gold_close": range(101, 101 + rows),
            "gold_volume": range(1000, 1000 + rows),
            "dxy_open": range(90, 90 + rows),
            "dxy_high": range(91, 91 + rows),
            "dxy_low": range(89, 89 + rows),
            "dxy_close": range(90, 90 + rows),
            "vix_open": range(15, 15 + rows),
            "vix_high": range(16, 16 + rows),
            "vix_low": range(14, 14 + rows),
            "vix_close": range(15, 15 + rows),
            "tnx_open": range(1, 1 + rows),
            "tnx_high": range(2, 2 + rows),
            "tnx_low": range(1, 1 + rows),
            "tnx_close": range(2, 2 + rows),
        }
    )


def test_create_features_adds_expected_columns():
    """
    Comprueba que create_features crea las variables principales
    del feature engineering.
    """
    df = create_sample_dataframe()

    result = create_features(df)

    expected_columns = [
        "gold_return",
        "dxy_return",
        "vix_return",
        "tnx_return",
        "gold_ma_5",
        "gold_ma_20",
        "gold_rsi_14",
        "gold_macd",
        "gold_macd_signal",
        "gold_volatility_14",
        "gold_return_lag_1",
        "gold_return_lag_2",
        "dxy_return_lag_2",
        "vix_return_lag_2",
        "gold_return_5d",
        "gold_return_10d",
        "vix_ma_20",
        "vix_high_fear",
        "vix_extreme_fear",
        "gold_ma5_minus_ma20",
        "gold_dxy_spread",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_vix_flags_are_binary():
    """
    Comprueba que las variables de miedo del VIX solo contienen 0 y 1.
    """
    df = create_sample_dataframe()

    result = create_features(df)

    assert set(result["vix_high_fear"].dropna().unique()).issubset({0, 1})
    assert set(result["vix_extreme_fear"].dropna().unique()).issubset({0, 1})


def test_lag_features_use_past_values():
    """
    Comprueba que los lags usan información pasada y no futura.
    """
    df = create_sample_dataframe()

    result = create_features(df)

    assert result.loc[3, "gold_return_lag_1"] == result.loc[2, "gold_return"]
    assert result.loc[4, "gold_return_lag_2"] == result.loc[2, "gold_return"]
    assert result.loc[4, "dxy_return_lag_2"] == result.loc[2, "dxy_return"]
    assert result.loc[4, "vix_return_lag_2"] == result.loc[2, "vix_return"]


def test_targets_have_expected_classes():
    """
    Comprueba que el target multiclase solo contiene -1, 0 y 1.
    """
    df = create_sample_dataframe()
    df = create_features(df)

    result = create_targets(df, threshold=0.005)

    assert "target_multiclass" in result.columns
    assert set(result["target_multiclass"].dropna().unique()).issubset({-1, 0, 1})
