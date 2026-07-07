# Architecture — Golden Forecast

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Yahoo      │     │  Extract     │     │  gold-macro-data  │
│  Finance    │────▶│  extract.py  │────▶│  (data/raw/)      │
│  API        │     │              │     │                   │
└─────────────┘     └──────────────┘     └───────────────────┘
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │ Preprocess   │
                                         │ preprocessing│
                                         │ .py          │
                                         └──────────────┘
                                                 │
                                                 ▼
                                         ┌──────────────────┐
                                         │ gold-clean.csv   │
                                         │ (data/processed/)│
                                         └──────────────────┘
                                                 │
                                                 ▼
                                         ┌───────────────────┐
                                         │ Feature Engineer │
                                         │ feature_engineer │
                                         │ ing.py            │
                                         └───────────────────┘
                                                 │
                                                 ▼
                                         ┌──────────────────┐
                                         │ gold-features.csv│
                                         │ (data/processed/)│
                                         └──────────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    ▼                            ▼                            ▼
        ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
        │ Classification      │    │ Regression          │    │ Dashboard           │
        │ train.py + evaluate │    │ regression.py       │    │ app.py + callbacks  │
        │ .py                 │    │                     │    │ + layout.py + data  │
        │                     │    │                     │    │ .py                 │
        │ ↓ 12 modelos .pkl   │    │ ↓ cache .pkl        │    │                     │
        │ + scaler.pkl        │    │ + CV metrics        │    │ ↓ Plotly Dash       │
        │ + evaluation_results│    │                     │    │ + Gunicorn          │
        └─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                                                        │
                                                                        ▼
                                                               ┌──────────────────┐
                                                               │ Render (Docker)  │
                                                               │ port 8050        │
                                                               │ health endpoint  │
                                                               └──────────────────┘
```

## Pipeline Orchestration

All steps can be run via `scripts/run_pipeline.py`:
```
extract.py → preprocessing.py → feature_engineering.py → train.py → evaluate.py
```

## Deployment

- **Container**: Docker (python:3.12-slim)
- **Host**: Render (branch: feature/full-pipeline)
- **CI**: GitHub Actions (lint → test → build → deploy)
- **Entry point**: `gunicorn src.dashboard.app:server`

## Teams

| Module | Owner |
|--------|-------|
| EDA | José |
| Preprocessing + Features | Gema |
| Classification | Joel |
| Regression (complementary) | Juan |
| Dashboard | Juan |
| Slides | María (lead) |
