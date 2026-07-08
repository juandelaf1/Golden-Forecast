<p align="center">
  <img src="src/dashboard/assets/banner.png" alt="Golden Forecast Banner" width="800">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-2.0-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-2.0-EC1C24?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Plotly-5.24-3F4F75?logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/Dash-2.18-008DE4?logo=dash&logoColor=white" alt="Dash">
  <img src="https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Yahoo%20Finance-yfinance-6001D2?logo=yahoo&logoColor=white" alt="yfinance">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Render-Live-46E3B7?logo=render&logoColor=white" alt="Render">
  <img src="https://img.shields.io/badge/Mintlify-Docs-6B46C1?logo=mintlify&logoColor=white" alt="Mintlify">
  <img src="https://github.com/juandelaf1/Golden-Forecast/actions/workflows/ci.yml/badge.svg" alt="CI">
</p>

---

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
| Product Owner | Maria |
| Scrum Master | Juan |
| Development Team | Jose, Gema, Joel |

## Dataset

Datos diarios de **GC=F** (Gold Futures) via Yahoo Finance desde 2015, enriquecidos con 3 indicadores macro:

| Indicador | Descripción | Relación con oro |
|-----------|-------------|------------------|
| **DXY** | Índice del dólar (cesta 6 divisas) | Inversa |
| **VIX** | Índice de volatilidad / miedo | Directa (refugio) |
| **TNX** | Bono USA 10 años (tipos de interés) | Inversa |

> **Dashboard en vivo**: [https://golden-forecast.onrender.com](https://golden-forecast.onrender.com)
> **Documentación**: [Mintlify](https://mintlify.com/juandelaf1/golden-forecast) (próximamente)

## Stack Tecnologico

| Área | Tecnologías |
|------|-------------|
| Datos | Python, pandas, numpy, yfinance |
| Modelado | scikit-learn, XGBoost |
| Dashboard | Plotly Dash, Plotly.js, CSS Grid |
| Deployment | Docker, Render |
| Gobernanza | GitHub Flow, Scrum, Conventional Commits |

## Estructura del Repositorio

```
golden-forecast/
├── data/
│   ├── raw/                    # Datos crudos Yahoo Finance
│   └── processed/              # gold-clean.csv, gold-features.csv (24 features)
├── notebooks/                  # EDA, preprocessing, clasificacion
├── src/
│   ├── extract/                # Descarga de datos
│   ├── predict.py              # Predicción diaria con modelos guardados
│   ├── models/                 # Pipeline entrenamiento (train.py, evaluate.py)
│   ├── dashboard/              # App Dash (app.py, callbacks.py, layout.py, data.py, model_loader.py)
│   ├── classification.py       # Módulo de clasificación reutilizable
│   └── regression.py           # Módulo de regresión exploratorio
├── models/                     # 12 modelos pre-entrenados (.pkl) + scaler + metadata
├── docs/                       # Handbook, data dictionary, decision log, ml report, architecture, dashboard guide, screenshots
├── .github/workflows/          # CI + Daily pipeline + Mintlify deploy
├── tests/                      # 7 test files (47 tests)
├── mint.json                   # Documentación Mintlify
└── README.md, ROADMAP.md, requirements.txt, Dockerfile, render.yaml
```

## Pipeline de ML

```
extract.py → preprocessing.py → feature_engineering.py → train.py → evaluate.py
     ↓              ↓                    ↓                   ↓           ↓
  Yahoo Finance  Columnas          35 features          12 modelos    Metricas +
  (GC=F, DXY,    limpias           + targets            x 2 targets  Backtest
   VIX, TNX)                                                           + EDA
```

## Modelos Pre-entrenados

12 modelos entrenados con split temporal 80/20, escalado sin data leakage:

| Modelo | Algoritmo | Target | F1 (test) | Accuracy |
|--------|-----------|--------|-----------|----------|
| lr_strong_reg_binary | Logistic Regression (C=0.1) | Binario | 0.70 | 56.9% |
| lr_binary | Logistic Regression (C=1.0) | Binario | 0.69 | 56.2% |
| xgb_binary | XGBoost (n=100, d=3) | Binario | 0.67 | 56.5% |
| rf_binary | Random Forest (n=100, d=5) | Binario | 0.66 | 55.2% |
| rf_deep_binary | Random Forest (n=200, d=10) | Binario | 0.62 | 53.6% |
| xgb_deep_binary | XGBoost (n=200, d=5) | Binario | 0.62 | 53.9% |
| lr_multiclass | Logistic Regression | Multiclase | 0.31 | 38.1% |

Metricas de regresion disponibles via dashboard (MAE, RMSE, R², MAPE).

## Dashboard Interactivo

Dashboard temático con 9 pestañas:

| Pestaña | Contenido |
|---------|-----------|
| **Panel de Control** | Señal del día, certeza, precio, predicción vs realidad, rendimiento acumulado |
| **Precio** | Gráfico histórico con señales del modelo y MA 21 |
| **Indicadores** | RSI (14), MACD con selector de fechas |
| **Macro** | Correlaciones DXY/VIX/TNX + indicadores normalizados base 100 |
| **Backtest** | Estrategia ML vs Buy & Hold, matriz de confusión, ROC, tabla 12 modelos |
| **Simulación** | Simulador con capital inicial y rango personalizable (default 1 mes) |
| **Métricas** | Importancia de variables, comparativa completa de modelos |
| **Valor y Riesgo** | Valor justo, volatilidad, ATR, riesgo de caída (regresión) |
| **Metodología** | Pipeline, modelos, equipo, stack, limitaciones |

### Funcionalidades destacadas

- **Selector de fechas** (1D, 5D, 1M, 3M, 6M, 1A, HIST) en todos los gráficos temporales
- **Importancia de variables** filtrable por categoría (Técnico, Macro, Precio)
- **Dropdowns explicativos** en pestañas de Macro y Panel de Control
- **Control de volumen** para ambientación sonora
- **Ticker** con datos en tiempo real del oro, DXY, VIX, MA 21
- **Documentación** desplegada automáticamente en Mintlify desde main

### Metricas mostradas

| Tipo | Metricas |
|------|----------|
| Clasificacion | Accuracy, Precision, Recall, F1 Score, ROC-AUC |
| Regresion | MAE, RMSE, R², MAPE |
| Backtest | Retorno estrategia, retorno Buy & Hold, Alpha |

### Ejecutar

```bash
pip install -r requirements.txt
python src/dashboard/app.py
# Abrir http://localhost:8050
```

### Docker

```bash
docker build -t golden-forecast .
docker run -p 8050:8050 golden-forecast
```

## Pipeline Automático (GitHub Actions)

El pipeline diario se ejecuta automáticamente de lunes a viernes a las 23:00 UTC (tras cierre del mercado americano):

```
extract.py → preprocessing.py → feature_engineering.py → predict.py → git commit + push
     ↓              ↓                    ↓                   ↓
  Yahoo Finance  Columnas          35 features          Predicción del día
  (GC=F, DXY,    limpias           + targets            siguiente guardada
   VIX, TNX)                                            en predictions.csv
```

También se puede ejecutar manualmente desde la pestaña Actions de GitHub.

## Pipeline completo (local)

```bash
# 1. Clonar
git clone https://github.com/juandelaf1/Golden-Forecast.git

# 2. Dependencias (entorno virtual recomendado)
pip install -r requirements.txt

# 3. Descargar datos frescos
python src/extract/extract.py

# 4. Re-entrenar modelos
python src/models/train.py
python src/models/evaluate.py

# 5. Generar predicción diaria
python src/predict.py

# 6. Lanzar dashboard
python src/dashboard/app.py

# Documentación
# Las docs se despliegan automáticamente en Mintlify desde main
```

## Gobernanza (Scrum)

- **Sprint Planning** semanal + Daily Syncs
- **GitHub Flow**: ramas feature/ + PRs con revision por pares
- **Main protegido** (sin pushes directos)
- **Decision Log** en `docs/decision_log.md`
- **Data lineage** en `data/README.md`
- **Conventional Commits**: feat, fix, docs, refactor, chore

## Roadmap

Ver [ROADMAP.md](ROADMAP.md) para el plan de desarrollo completo.

## License

Proyecto academico para uso no comercial.
