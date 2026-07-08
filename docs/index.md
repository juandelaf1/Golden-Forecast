# Golden Forecast — DataScope Solutions

Somos **DataScope Solutions**, consultora internacional de análisis de datos. Este proyecto aplica Machine Learning supervisado sobre datos históricos del oro (GC=F) para generar predicciones explicables a través de un dashboard interactivo.

## Problema de Negocio

Predecir el comportamiento del precio del oro usando datos históricos y macroeconómicos (DXY, VIX, TNX). Abordamos el problema desde dos enfoques:

- **Clasificación binaria**: sube (1) o baja (0) al día siguiente
- **Clasificación multiclase**: comprar (+1%), mantener, vender (<-1%)
- **Regresión**: predicción del retorno esperado (MAE, RMSE, R², MAPE)

## Equipo

| Rol | Nombre |
|-----|--------|
| Product Owner | María |
| Scrum Master | Juan |
| Development Team | José, Gema, Joel |

## Dataset

Datos diarios de **GC=F** (Gold Futures) via Yahoo Finance desde 2015, enriquecidos con 3 indicadores macro:

| Indicador | Descripción | Relación con oro |
|-----------|-------------|------------------|
| **DXY** | Índice del dólar (cesta 6 divisas) | Inversa |
| **VIX** | Índice de volatilidad / miedo | Directa (refugio) |
| **TNX** | Bono USA 10 años (tipos de interés) | Inversa |

> **Dashboard en vivo**: [https://golden-forecast.onrender.com](https://golden-forecast.onrender.com)

## Stack Tecnológico

| Area | Tecnologías |
|------|-------------|
| Datos | Python, pandas, numpy, yfinance |
| Modelado | scikit-learn, XGBoost |
| Dashboard | Plotly Dash, Plotly.js, CSS Grid |
| Deployment | Docker, Render |
| Gobernanza | GitHub Flow, Scrum, Conventional Commits |

## Pipeline de ML

```
extract.py → preprocessing.py → feature_engineering.py → train.py → evaluate.py
     ↓              ↓                    ↓                   ↓           ↓
  Yahoo Finance  Columnas          35 features          12 modelos    Metricas +
  (GC=F, DXY,    limpias           + targets            x 2 targets  Backtest
   VIX, TNX)                                                           + EDA
```
