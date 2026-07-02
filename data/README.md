# Data Lineage

## Source

- **Provider**: Yahoo Finance (via `yfinance` Python library)
- **Tickers**: GC=F (Gold Futures), DX-Y.NYB (DXY), ^VIX, ^TNX
- **Frequency**: Daily
- **Range**: 2015-01-01 to present
- **Download script**: `src/extract/extract.py`

## Raw Data Schema

| Column | Type | Description |
|--------|------|-------------|
| Gold_Open | float | Opening price of gold |
| Gold_High | float | Daily high of gold |
| Gold_Low | float | Daily low of gold |
| Gold_Close | float | Closing price of gold |
| Gold_Volume | float | Trading volume |
| DXY_Close | float | Dollar index close |
| VIX_Close | float | Volatility index close |
| TNX_Close | float | 10Y Treasury yield close |

## Processing Pipeline

```
extract.py → preprocessing.py → feature_engineering.py → targets.py → pipeline.py
     ↓              ↓                    ↓                   ↓             ↓
  data/raw/      Columnas            Features            Target        Split + Escalado
  gold-macro-    renombradas         técnicas            binario       temporal
  data.csv       y limpiadas         y macro             multiclase
```

## Storage

- **Raw**: `data/raw/gold-macro-data.csv` (immutable, never modified)
- **Processed**: `data/processed/gold-features.csv` (output of full pipeline)

## Reproducibility

```bash
python src/extract/extract.py          # Descargar datos crudos
python src/preprocessing.py            # (Opcional) Generar CSV limpio
python src/feature_engineering.py      # (Opcional) Generar CSV con features
```

O ejecutar los notebooks en orden:
`EDA_Golden_Forecast.ipynb → 02_preprocessing.ipynb → 03_classification.ipynb`
