# Roadmap — Golden Forecast

## Sprint 1 (Days 1‑3): Data Preparation & Ingestion ✅
| Milestone                     | Assigned to          | Status |
|-------------------------------|----------------------|--------|
| Raw data downloaded & stored  | Juan (SM)            | ✅ |
| Raw schema defined            | Gema (Dev)           | ✅ |
| Preprocessing pipeline built  | Gema (Dev)           | ✅ |
| Feature engineering completed| Gema (Dev)           | ✅ |
| Version‑control & Git setup   | Juan (SM)            | ✅ |

## Sprint 2 (Days 4‑6): Modeling & Validation ✅
| Milestone                               | Assigned to          | Status |
|-----------------------------------------|----------------------|--------|
| Binary & multiclass classification      | Juan                 | ✅ `src/models/train.py` |
| Ensemble model pipeline (`train.py`)    | Joel                 | ✅ 12 pretrained models |
| Automated evaluation pipeline (`evaluate.py`) | Joel           | ✅ Metrics, backtest, overfit check |
| Interactive Plotly Dash dashboard (8 tabs) | Juan           | ✅ |
| Lazy‑loading of models & tabs           | Juan                 | ✅ Startup < 3 s |
| Audio autoplay on first interaction      | Juan                 | ✅ |
| Ticker visibility & readability         | Juan                 | ✅ |
| Tech‑stack refinement (icons, naming)   | Juan                 | ✅ |

## Sprint 3 (Days 7‑8): Release & Documentation 📦
| Milestone                               | Assigned to          | Status |
|-----------------------------------------|----------------------|--------|
| Model deployment & CI/CD automation       | Juan (SM)            | ✅ |
| Final data documentation (Data Lineage)   | Juan (SM)            | ✅ |
| EDA notebook (PR #38)                     | Jose (Dev)           | ✅ |
| Market sentiment features (PR #39)        | Gema (Dev)           | ✅ |
| Re‑train models with new features         | Joel (Dev)           | ⏳ |
| Docs updated & Mintlify site              | Juan (SM)            | ✅ |
| Presentation narrative & QA               | Maria (PO)           | ✅ |
| Final repo cleanup & handover            | All (collaborative)  | ✅ |

**Entrega final**: Dashboard funcional + API health endpoint + repositorio documentado

---

## Data Lineage
The pipeline follows a strict, auditable flow to ensure reproducibility and traceability.

1. **Raw Data Acquisition**  
   - **Source**: Yahoo Finance via `yfinance` Python library.  
   - **Ticklers**: `GC=F` (Gold Futures), `DX‑Y.NYB` (DXY), `^VIX` (Volatility Index), `^TNX` (10‑yr Treasury Yield).  
   - **Frequency**: Daily, covering **2015‑01‑01 → present**.  
   - **Export Script**: `src/extract/extract.py`.  
   - **Schema**:  

| Column            | Type    | Description                          |
|-------------------|---------|--------------------------------------|
| `Date`            | `datetime64[ns]` | Transaction timestamp |
| `Gold_Open`       | `float` | Opening price of gold (USD/oz) |
| `Gold_High`       | `float` | Daily high |
| `Gold_Low`        | `float` | Daily low |
| `Gold_Close`      | `float` | Closing price |
| `Gold_Volume`     | `float` | Trading volume |
| `DXY_Close`       | `float` | Dollar Index close |
| `VIX_Close`       | `float` | CBOE Volatility Index |
| `TNX_Close`       | `float` | 10‑yr Treasury Yield close |

2. **Raw Storage**  
   - File: `data/raw/gold-macro-data.csv` (immutable; never modified after download).  

3. **Processed Data**  
   - **Cleaned CSV**: `data/raw/gold-clean.csv` (after outlier handling, NaN treatment).  
   - **Feature Set**: `data/processed/gold-features.csv` (features + engineered columns, target variables).  

4. **Pipeline Flow**  
   ```
   extract.py  →  preprocessing.py  →  feature_engineering.py  →  classification.py
```

All steps are idempotent, version‑controlled, and documented in the `README.md`.

---

### Updated Module Overview
| File / Folder                         | Purpose |
|---------------------------------------|---------|
| `src/extract/extract.py`              | Download raw Yahoo Finance data |
| `src/preprocessing.py`                | Clean & rename columns |
| `src/feature_engineering.py`          | Create 24 technical & macro features + 9 market sentiment (PR #39) |
| `src/models/train.py`                 | Train LR, RF, XGBoost classifiers |
| `src/models/evaluate.py`              | Compute metrics, backtest, overfit checks |
| `src/dashboard/`                      | Plotly Dash UI with 8 tabs |
| `src/dashboard/model_loader.py`       | On‑demand model loading |
| `render.yaml` / `Dockerfile`          | Deployment configuration |
