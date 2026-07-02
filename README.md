<p align="center">
  <img src="src/dashboard/assets/banner.png" alt="Golden Forecast Banner" width="800">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-1.5-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-1.7-EC1C24?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Plotly-5.17-3F4F75?logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/Dash-2.14-008DE4?logo=dash&logoColor=white" alt="Dash">
  <img src="https://img.shields.io/badge/Flask-2.3-000000?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Yahoo%20Finance-yfinance-6001D2?logo=yahoo&logoColor=white" alt="yfinance">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

# Golden Forecast - DataScope Solutions

Somos **DataScope Solutions**, consultora internacional de analisis de datos. Este proyecto aplica Machine Learning supervisado sobre datos historicos del oro para generar predicciones explicables.

## Problema de Negocio

Predecir el comportamiento del precio del oro (GC=F) usando datos historicos y macroeconomicos (DXY, VIX, TNX). Abordamos el problema desde dos enfoques:

- **Clasificacion**: predecir si el oro sube o baja al dia siguiente (binaria y multiclase)
- **Regresion**: predecir el precio de cierre (valor continuo)

## Equipo

| Rol | Nombre |
|-----|--------|
| Product Owner | Maria |
| Scrum Master | Juan |
| Development Team | Jose, Gema, Joel |

## Dataset

Datos diarios de **GC=F** (Gold Futures) via Yahoo Finance desde 2015. Enriquecido con 3 indicadores macro:

- **DXY**: indice del dolar (relacion inversa con el oro)
- **VIX**: indice de volatilidad/miedo (oro como activo refugio)
- **TNX**: bono USA 10 anos (tipos altos = oro menos atractivo)

## Stack Tecnologico

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, yfinance, Plotly Dash

## Estructura del Repositorio

```
golden-forecast/
├── data/
│   ├── raw/                    # Datos crudos (gold-macro-data.csv)
│   ├── processed/              # Datos procesados (gold-clean.csv, gold-features.csv)
│   └── README.md               # Lineage de datos
├── notebooks/
│   ├── EDA_Golden_Forecast.ipynb   # Analisis exploratorio
│   ├── 02_preprocessing.ipynb      # Preprocesamiento
│   └── 03_classification.ipynb     # Clasificacion
├── src/
│   ├── extract/extract.py      # Descarga de datos Yahoo Finance
│   ├── preprocessing.py        # Limpieza y renombrado de columnas
│   ├── feature_engineering.py  # Features tecnicas y macro
│   ├── classification.py       # Modelos de clasificacion
│   └── dashboard/              # Dashboard Plotly Dash
├── docs/
│   ├── project_handbook.md     # Gobernanza del proyecto
│   ├── data_dictionary.md      # Definicion de variables
│   └── decision_log.md         # Registro de decisiones
├── models/                     # Modelos entrenados (.pkl)
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md
├── README.md
├── ROADMAP.md
├── requirements.txt
└── .gitignore
```

## Pipeline de ML

```
extract.py → preprocessing.py → feature_engineering.py → classification.py
     ↓              ↓                    ↓                    ↓
  Datos crudos   Columnas limpias   Features tecnicas    Modelos de
                 y renombradas      y macro              clasificacion
                                                      evaluados
```

## Como Ejecutar

```bash
# 1. Clonar el repositorio
git clone https://github.com/juandelaf1/Golden-Forecast.git

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar datos
python src/extract/extract.py

# 4. Ejecutar notebooks en orden
#    EDA_Golden_Forecast.ipynb → 02_preprocessing.ipynb → 03_classification.ipynb
```

## Dashboard

Dashboard **Wild-West Saloon** disponible en http://localhost:8050:

- **Price**: grafico del oro con RSI, MACD y volatilidad
- **Macro**: correlaciones con DXY, VIX, TNX
- **Models**: comparacion de accuracy entre modelos
- **Backtest**: rendimiento de estrategia vs Buy and Hold
- **Simulation**: simulador de trading en datos historicos

```bash
python src/dashboard/app.py
```

## Gobernanza (Scrum)

- **Sprint Planning** y daily syncs
- **GitHub Flow**: ramas feature/ + PRs con revision por pares
- **Main protegido** (sin pushes directos)
- **Decision Log** documentando decisiones tecnicas clave
- **Data lineage** para trazabilidad y reproducibilidad

## License

Proyecto academico para uso no comercial.
