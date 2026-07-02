# Roadmap - Gold Price Forecast

## Sprint 1 (Days 1-3): Data & Preprocessing

| Milestone | Assigned to | Estado |
|-----------|-------------|--------|
| Full EDA: nulls, statistics, visualizations, correlations | José | ✅ PR #33 mergeado |
| Preprocessing: cleaning, scaling, encoding, train/test split, pipeline | Gema | ✅ PR #31 mergeado + PR #32 refactorización |
| Visual EDA for presentation | María | Pendiente |
| Dataset downloaded and repo operational | Juan (SM) | ✅ `src/extract/extract.py` funcional |

**Deliverable**: Notebooks `EDA_Golden_Forecast.ipynb` and `02_preprocessing.ipynb` working ✅

## Sprint 2 (Days 4-6): Modeling

| Milestone | Assigned to | Estado |
|-----------|-------------|--------|
| Classification model: baseline + Random Forest | Juan | ✅ `src/classification.py` + `03_classification.ipynb` |
| Regression model: baseline + Random Forest | Joel | En progreso |

**Deliverable**: Notebooks `03_classification.ipynb` and `04_regression.ipynb` with trained models and metrics

## Sprint 3 (Days 7-8): Evaluation & Closing

| Milestone | Assigned to | Estado |
|-----------|-------------|--------|
| Comparative evaluation, overfitting detection, critical analysis | Joel | Pendiente |
| Slides presentation with business narrative | María (lead) + all | Pendiente |
| Final README, decision_log and repo closure | Juan (SM) | Pendiente |

**Final deliverable**: Notebook `05_evaluation.ipynb`, presentation, complete repo

---

## Módulos de `src/`

| Archivo | Responsable | Función |
|---------|-------------|---------|
| `extract/extract.py` | Joel | Descarga datos de Yahoo Finance |
| `preprocessing.py` | Gema | Limpieza, renombrado de columnas, eliminación de nulos |
| `feature_engineering.py` | Gema | Retornos, rangos, open-close, medias móviles |
| `targets.py` | Gema | Target binario y multiclase |
| `pipeline.py` | Gema | Split temporal, escalado, pipeline sklearn |
| `classification.py` | Juan | Modelos de clasificación, evaluación, backtesting |
| `dashboard/` | Juan | Dashboard Plotly Dash |
