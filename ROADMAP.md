# Roadmap — Golden Forecast

## Sprint 1 (Days 1‑3): Data Preparation & Ingestion ✅
- Data extraction, schema definition & preprocessing pipeline
- Feature engineering (technical + macro indicators)
- Git setup, repo structure & version control

## Sprint 2 (Days 4‑6): Modeling & Validation ✅
- Classification modeling (Logistic Regression, Random Forest, XGBoost)
- Model training pipeline & automated evaluation
- Backtesting & overfitting analysis
- Exploratory Data Analysis (EDA)
- Interactive Plotly Dash dashboard (8 tabs)
- Dashboard optimizations (caching, lazy loading, audio, UI)
- Regression module & exploratory experiments
- Feature importance, correlation & macro analysis

## Sprint 3 (Days 7‑8): Release & Documentation 📦
- Project vision, backlog prioritization & sprint coordination
- Presentation storytelling, narrative & QA validation
- Deployment (Docker, Render, CI/CD automation)
- Data lineage, documentation & data dictionary
- EDA notebooks (PR #38)
- Market sentiment & relative spread features (PR #39)
- Model re-training with updated feature set
- Testing (features, models, dashboard)
- Mintlify documentation site
- Github Actions workflow implementation
- Final repo cleanup & handover

**Deliverables**: PPTX + Dashboard funcional + API health endpoint + github repository

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
    - **Predictions**: `data/processed/predictions.csv` (daily feed from github actions workflow)

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
| `src/predict.py`                      | Daily prediction for next day |
| `src/models/train.py`                 | Train LR, RF, XGBoost classifiers |
| `src/models/evaluate.py`              | Compute metrics, backtest, overfit checks |
| `src/dashboard/`                      | Plotly Dash UI with 8 tabs |
| `src/dashboard/model_loader.py`       | On‑demand model loading |
| `src/dashboard/app.py`                | Dash web application endpoint
| `src/dashboard/callbacks.py`          | Defines the visual themes, data mappings, interactive callbacks, and layout structures |
| `src/dashboard/data.py`               | Data engine|
| `src/dashboard/model_loader.py`       | Loads, manages, and aggregates the predictions of the 12 trained machine learning models |
| `render.yaml` / `Dockerfile`          | Deployment configuration |
