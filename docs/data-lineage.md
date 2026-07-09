# Data Lineage

## Source

- **Provider**: Yahoo Finance (via `yfinance` Python library)
- **Tickers**: GC=F (Gold Futures), DX-Y.NYB (DXY), ^VIX, ^TNX
- **Frequency**: Daily
- **Range**: 2015-01-01 to present
- **Download script**: `src/extract/extract.py`

## Raw Data Schema

- **Gold_Open** (float): Opening price of gold
- **Gold_High** (float): Daily high of gold
- **Gold_Low** (float): Daily low of gold
- **Gold_Close** (float): Closing price of gold
- **Gold_Volume** (float): Trading volume
- **DXY_Close** (float): Dollar index close
- **VIX_Close** (float): Volatility index close
- **TNX_Close** (float): 10Y Treasury yield close

## Processing Pipeline

```
extract.py → preprocessing.py → feature_engineering.py → models/train.py
     ↓              ↓                    ↓                    ↓
    data/raw/      Columnas           35 features          12 modelos
    gold-macro-    renombradas        + 2 targets          .pkl
    data.csv       y limpiadas        (33+2 columnas)      guardados
```

## Storage

- **Raw**: `data/raw/gold-macro-data.csv` (immutable, never modified)
- **Processed**: `data/processed/gold-clean.csv` (cleaned data)
- **Features**: `data/processed/gold-features.csv` (33 features + 2 targets)

## Reproducibility

```bash
python src/extract/extract.py              # Descargar datos crudos
python src/preprocessing.py                # Generar CSV limpio
python src/feature_engineering.py          # Generar CSV con features
python src/models/train.py                 # Entrenar y evaluar modelos
```

O ejecutar el orquestador completo:
```bash
python scripts/run_pipeline.py
```
