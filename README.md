# Golden Forecast

Gold price classification and forecasting system. End-to-end ML pipeline from data extraction to interactive dashboard.

[Live Dashboard](https://golden-forecast.onrender.com) · [Portfolio Case Study](https://juandelaf1.github.io/building) · [Documentation](docs/index.md)

![CI](https://github.com/juandelaf1/Golden-Forecast/actions/workflows/ci.yml/badge.svg)

---

## Problem

Can a supervised model predict daily gold price direction using historical data, technical indicators, and macroeconomic variables? This project explores three modeling strategies — binary classification, multiclass classification, and regression — on publicly available market data.

## Approach

Three modeling strategies applied to gold price movement:

| Strategy | Target | Use Case |
|----------|--------|----------|
| Binary classification | Up (1) / Down (0) | Direction signal |
| Multiclass classification | Buy (>+1%), Hold, Sell (<-1%) | Trading signal |
| Regression | Expected return | Risk estimation |

## Data

Daily price data extracted via yfinance:

| Series | Ticker | Role |
|--------|--------|------|
| Gold | GC=F | Target variable |
| US Dollar Index | DX-Y.NYB | Inverse correlation |
| Volatility Index | ^VIX | Market risk proxy |
| 10Y Treasury | ^TNX | Interest rate context |

Data range: historical daily values, split 80/20 chronologically to prevent leakage.

## Pipeline

extract -> preprocess -> feature_engineering -> train -> evaluate -> dashboard

Each step is a standalone script in src/:

## Architecture

flowchart LR
    A[Yahoo Finance] --> B[extract]
    B --> C[preprocess]
    C --> D[feature_engineering]
    D --> E[train]
    E --> F[evaluate]
    F --> G[Dashboard]
    E --> G

## Models

12 models trained with chronological 80/20 split:

| Model | Target | F1 | Accuracy |
|-------|--------|-----|----------|
| Logistic Regression (L2) | Binary | 0.70 | 56.9% |
| Logistic Regression (default) | Binary | 0.69 | 56.2% |
| XGBoost | Binary | 0.67 | 56.5% |
| Random Forest | Binary | 0.66 | 55.2% |
| Random Forest (deep) | Binary | 0.62 | 53.6% |
| Logistic Regression | Multiclass | 0.31 | 38.1% |

## Results

- Best model: Logistic Regression with L2 regularization (F1 = 0.70, Accuracy = 56.9%)
- Baseline (always predict previous direction): ~53%
- Improvement over baseline: +3.9%

All results on held-out test data with chronological split. Not a trading recommendation.

## Limitations

- Educational/analytical project — not financial advice
- Feature set limited to freely available macro data (no sentiment, news, or alternative data)
- No live trading integration
- Model performance does not account for transaction costs or slippage
- Dashboard deployed on free Render tier (may cold-start)

## Stack

| Area | Tools |
|------|-------|
| Language | Python 3.12 |
| Data | pandas, numpy, yfinance |
| ML | scikit-learn, XGBoost |
| Visualization | Dash, Plotly |
| Infrastructure | Docker, Docker Compose |
| Deployment | Render, Gunicorn |
| CI/CD | GitHub Actions |
| Testing | pytest |

## Reproduce

git clone https://github.com/juandelaf1/Golden-Forecast.git
cd Golden-Forecast
pip install -r requirements.txt
python src/extract/extract.py
python src/preprocessing.py
python src/feature_engineering.py
python src/models/train.py
python src/models/evaluate.py
python src/dashboard/app.py

### Docker

docker compose up --build

### Tests

pytest tests/ -v

## Context

This project was developed as part of an advanced data science program. It demonstrates end-to-end ML engineering: data extraction, feature engineering, model training, evaluation, visualization, containerization, CI/CD, and deployment.

Full case study with detailed architecture decisions at juandelaf1.github.io