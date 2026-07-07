import sys
from pathlib import Path

import pandas as pd

# Permite importar archivos desde la carpeta src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import clean_gold_data


def create_raw_sample_dataframe():
    """
    Crea un dataset pequeño con columnas originales para probar el preprocessing.
    """
    return pd.DataFrame({
        "Date": pd.date_range(start="2020-01-01", periods=5, freq="D"),

        "gold_open": [100, 101, 102, 103, 104],
        "gold_high": [101, 102, 103, 104, 105],
        "gold_low": [99, 100, 101, 102, 103],
        "gold_close": [100, 101, 102, 103, 104],
        "gold_volume": [1000, 1100, 1200, 1300, 1400],

        "dxy_open": [90, 91, 92, 93, 94],
        "dxy_high": [91, 92, 93, 94, 95],
        "dxy_low": [89, 90, 91, 92, 93],
        "dxy_close": [90, 91, 92, 93, 94],
        "dxy_volume": [1, 1, 1, 1, 1],

        "vix_open": [15, 16, 17, 18, 19],
        "vix_high": [16, 17, 18, 19, 20],
        "vix_low": [14, 15, 16, 17, 18],
        "vix_close": [15, 16, 17, 18, 19],
        "vix_volume": [1, 1, 1, 1, 1],

        "tnx_open": [2, 2, 2, 2, 2],
        "tnx_high": [3, 3, 3, 3, 3],
        "tnx_low": [1, 1, 1, 1, 1],
        "tnx_close": [2, 2, 2, 2, 2],
        "tnx_volume": [1, 1, 1, 1, 1],
    })


def test_clean_gold_data_returns_dataframe(tmp_path):
    """
    Comprueba que el preprocessing devuelve un DataFrame válido.
    """
    input_path = tmp_path / "raw_sample.csv"
    output_path = tmp_path / "clean_sample.csv"

    df = create_raw_sample_dataframe()
    df.to_csv(input_path, index=False)

    result = clean_gold_data(input_path, output_path)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert output_path.exists()


def test_volume_columns_are_handled_correctly(tmp_path):
    """
    Comprueba que se eliminan los volúmenes menos útiles
    y se conserva el volumen del oro.
    """
    input_path = tmp_path / "raw_sample.csv"
    output_path = tmp_path / "clean_sample.csv"

    df = create_raw_sample_dataframe()
    df.to_csv(input_path, index=False)

    result = clean_gold_data(input_path, output_path)

    assert "gold_volume" in result.columns
    assert "dxy_volume" not in result.columns
    assert "vix_volume" not in result.columns
    assert "tnx_volume" not in result.columns