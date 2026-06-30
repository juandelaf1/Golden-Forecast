# Data Lineage

## Source

- **Provider**: Yahoo Finance (via `yfinance` Python library)
- **Ticker**: `GC=F` (Gold Futures)
- **Frequency**: Daily
- **Range**: 2015-01-01 to present
- **Download script**: `src/data/loader.py`

## Raw Data Schema

| Column | Type | Description |
|--------|------|-------------|
| Open | float | Opening price |
| High | float | Daily high |
| Low | float | Daily low |
| Close | float | Closing price |
| Adj Close | float | Adjusted closing price |
| Volume | int64 | Trading volume |

## Processing Pipeline

```
Raw (yfinance) → 01_eda.ipynb → 02_preprocessing.ipynb → 03_classification.ipynb
                                                       → 04_regression.ipynb
                                                       → 05_evaluation.ipynb
```

## Storage

- **Raw**: `data/raw/gold_raw.csv` (immutable, never modified)
- **Processed**: `data/processed/gold_clean.csv` (output of preprocessing)

## Reproducibility

Run `notebooks/01_eda.ipynb` from scratch to regenerate all data.
