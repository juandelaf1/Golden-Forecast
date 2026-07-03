# DataScope Solutions: Project Handbook & Governance

**Proyecto:** Machine Learning Pipeline & Predictive Modeling
**Versión:** 2.1.0
**Fecha:** 30 de junio de 2026

---

## 1. Project Charter

Este documento define la metodología, gobernanza y estándares del proyecto DataScope Solutions.

El objetivo es desarrollar una solución de Machine Learning supervisado (clasificación o regresión) desde datos crudos hasta una propuesta de valor explicable para cliente.

El proyecto se desarrollará siguiendo un enfoque Agile adaptado a Scrum, promoviendo la colaboración, la trazabilidad del trabajo y la reproducibilidad técnica.

---

## 2. Equipo del Proyecto

### Miembros

* María
* Juan
* José
* Gema
* Joel

### Roles y Responsabilidades

#### Product Owner (PO) – María

Responsable de la visión de negocio y validación final del proyecto.

Funciones:

* Definir objetivos del proyecto.
* Priorizar backlog y entregables.
* Validar resultados obtenidos.
* Revisar el cumplimiento de objetivos.
* Coordinar la presentación final.
* Garantizar la alineación entre negocio y solución técnica.

#### Scrum Master (SM) – Juan

Responsable de facilitar el proceso Scrum y coordinar el equipo.

Funciones:

* Organizar Sprint Planning.
* Coordinar Daily Sync.
* Gestionar bloqueos.
* Supervisar el tablero Kanban.
* Asegurar el cumplimiento del proceso Scrum.
* Verificar la Definition of Done.
* Coordinar retrospectivas.

#### Development Team

**José, Gema y Joel**

Responsables de la implementación técnica del proyecto.

##### Propuesta inicial de distribución de responsabilidades

**José**

Responsable principal de análisis exploratorio.

Tareas:

* Exploratory Data Analysis (EDA).
* Estadística descriptiva.
* Identificación de valores nulos.
* Detección de anomalías.
* Análisis de correlaciones.
* Elaboración de visualizaciones.
* Documentación de hallazgos.

**Gema**

Responsable principal de preprocesamiento.

Tareas:

* Limpieza de datos.
* Tratamiento de valores faltantes.
* Transformación de variables.
* Codificación de variables categóricas.
* Escalado de variables.
* Feature Engineering.
* Construcción del pipeline de preprocesamiento.
* Validación de ausencia de data leakage.

**Joel**

Responsable principal de modelado.

Tareas:

* Selección de algoritmos candidatos.
* Entrenamiento de modelos.
* Comparación entre enfoques.
* Evaluación mediante métricas.
* Ajuste de hiperparámetros.
* Validación básica.
* Análisis de overfitting.
* Interpretación de resultados.

> La distribución de responsabilidades es orientativa y podrá ajustarse durante Sprint Planning según las necesidades del proyecto.

---

## 3. Problema de Negocio

El cliente dispone de datos históricos pero no tiene claridad sobre cómo aprovecharlos para generar información útil que apoye la toma de decisiones.

Actualmente el equipo está evaluando desarrollar un caso de estudio basado en datos históricos relacionados con el precio del oro.

El objetivo es determinar si es posible construir un modelo predictivo que permita estimar el comportamiento futuro de la variable objetivo e identificar patrones relevantes.

El equipo deberá:

* Abordar el problema desde clasificación (sube/baja) y regresión (precio continuo).
* Definir targets binario y multiclase.
* Construir un pipeline completo de Machine Learning supervisado.
* Evaluar el rendimiento predictivo de los modelos.
* Traducir resultados técnicos a lenguaje de negocio.
* Comunicar conclusiones de manera clara y comprensible.

**Estado actual:** Dataset GC=F + DXY + VIX + TNX (2015-2026). 12 modelos pre-entrenados de clasificación (LR, RF, XGB × binario, multiclase). Regresión implementada como módulo experimental. Dashboard interactivo Plotly Dash desplegable via Docker.

---

## 4. Metodología de Trabajo (Agile - Scrum adaptado)

### 4.1 Rituales

* Sprint Planning: definición de tareas y dataset.
* Daily Sync: avances, bloqueos y próximos pasos.
* Sprint Review: presentación final al cliente.
* Retrospectiva: identificación de mejoras del proceso.

### 4.2 Kanban

Columnas:

* Por hacer
* En progreso
* Hecho

Herramienta recomendada:

* GitHub Projects
* Trello

---

## 5. Flujo de Trabajo Técnico (GitHub Flow)

Reglas:

1. Rama principal protegida (`main`)
2. Ramas por feature:

   * `feature/eda`
   * `feature/preprocessing`
   * `feature/modeling`
   * `feature/classification`
   * `feature/documentation`
3. Pull Requests obligatorios con plantilla (`.github/PULL_REQUEST_TEMPLATE.md`).
4. Revisión por pares antes del merge.
5. Merges de `main` en `feature/` para mantener sincronización.

### Conventional Commits

* feat: nueva funcionalidad
* fix: corrección de errores
* docs: documentación
* refactor: mejora de código
* chore: tareas de mantenimiento

---

## 6. Estructura del Repositorio

```text
golden-forecast/
├── data/
│   ├── raw/                    # gold-macro-data.csv (Yahoo Finance)
│   └── processed/              # gold-clean.csv, gold-features.csv (24 features)
├── notebooks/                  # EDA, preprocessing, clasificacion
├── src/
│   ├── extract/extract.py      # Descarga Yahoo Finance
│   ├── preprocessing.py        # Limpieza y renombrado
│   ├── feature_engineering.py  # 24 features tecnicas + macro
│   ├── models/
│   │   ├── train.py            # Entrenamiento pre-entrenado
│   │   └── evaluate.py         # Evaluacion + backtest
│   └── dashboard/
│       ├── app.py              # Entry point Dash
│       ├── layout.py           # Layout 8 pestañas
│       ├── callbacks.py        # Callbacks + graficos
│       ├── data.py             # Datos + live training
│       ├── model_loader.py     # Carga modelos pre-entrenados
│       └── assets/style.css    # Tema Wild-West
├── models/                     # 12 .pkl + scaler.pkl + metadata JSON
├── docs/                       # Handbook, data dictionary, decision log
├── mock_server/                # API mock para pruebas
├── Dockerfile
├── docker-compose.yml
├── README.md
├── ROADMAP.md
└── requirements.txt
```

---

## 7. Dataset

**Estado:** En evaluación

### Caso de estudio preliminar

Predicción del comportamiento histórico del precio del oro.

### Variables potenciales

* Fecha
* Open
* High
* Low
* Close
* Adj Close
* Volume
* Rendimientos diarios
* Variación porcentual
* Medias móviles
* Indicadores técnicos derivados

### Requisitos

* Dataset supervisado.
* Variables numéricas suficientes.
* Calidad adecuada para modelado.
* Datos públicos.
* Permisos de reutilización.
* Reproducibilidad del proceso.

### Fuentes candidatas

* Kaggle
* Yahoo Finance
* Alpha Vantage
* Investing.com
* FRED
* Hugging Face

### Decisión

Dataset GC=F (Gold Futures) via Yahoo Finance, enriquecido con DXY, VIX, TNX (2015-2026). Procesado en `data/processed/gold-features.csv` con 24 features técnicas y macro.

---

## 8. Pipeline de Machine Learning

```
extract.py → preprocessing.py → feature_engineering.py → train.py → evaluate.py
     ↓              ↓                    ↓                   ↓           ↓
  Yahoo Finance  Columnas          24 features           6 modelos    Metricas +
  (GC=F, DXY,    limpias           tecnicas+macro        x 2 targets  Backtest
   VIX, TNX)                                                                    
```

### Pipeline Dashboard

```
data.py (live training) 
                 ↘
model_loader.py (pre-trained) → callbacks.py → layout.py → app.py → Browser
```

---

## 9. Métricas de Evaluación

### Clasificación binaria

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Clasificación multiclase

* Accuracy
* F1-score (macro)

### Regresión

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² (coeficiente de determinación)
* MAPE (Mean Absolute Percentage Error)

### Backtest

* Retorno acumulado de la estrategia
* Retorno acumulado de Buy & Hold
* Alpha (diferencia estrategia vs B&H)

> La métrica principal de clasificación es F1-score. Para regresión, MAE y RMSE.

---

## 10. Definition of Done (DoD)

Una tarea está completada si:

* Código funcional sin errores.
* No existe data leakage.
* Pipeline reproducible.
* Documentación actualizada.
* Pull Request revisado.
* Peer review aprobado.
* Resultados interpretables.

---

## 11. Data Governance

* No se versionan datos sensibles.
* Se documenta el origen del dataset.
* Se evita data leakage.
* Se mantiene trazabilidad del pipeline.
* Se garantiza reproducibilidad.

---

## 12. Modelo y Validación

* Se entrenará al menos un modelo base.
* Se compararán al menos dos enfoques.
* Validación cruzada opcional.
* Se analizará overfitting.
* Se estudiará la interpretabilidad de resultados.

---

## 13. Registro de Decisiones

Ver `docs/decision_log.md` para el registro completo y actualizado de decisiones técnicas del proyecto.

---

## 14. Registro de Riesgos (Risk Log)

| Riesgo                | Impacto | Mitigación          |
| --------------------- | ------- | ------------------- |
| Overfitting           | Alto    | CV + regularización |
| Mala calidad de datos | Alto    | EDA exhaustivo      |
| Dataset insuficiente  | Medio   | Evaluación temprana |
| Retrasos              | Medio   | Kanban + Daily Sync |

---

## 15. Criterios de Evaluación del Proyecto

* Preparación correcta de datos.
* Modelado justificado.
* Métricas interpretadas adecuadamente.
* Código limpio y modular.
* Documentación completa.
* Comunicación clara.
* Trabajo colaborativo evidenciado.

---

## 16. Entregables

1. Notebook completo (EDA → modelo → evaluación)
2. Repositorio GitHub estructurado
3. Presentación tipo pitch (10–12 minutos)
4. Documentación técnica del proyecto

---

## 17. Presentación Final

Estructura:

1. Equipo
2. Problema de negocio
3. Dataset
4. Metodología
5. Resultados
6. Limitaciones
7. Conclusiones
8. Próximos pasos

---

## 18. Stack Tecnológico

| Área | Tecnologías |
|------|-------------|
| Lenguaje | Python 3.12 |
| Datos | pandas, numpy, yfinance |
| Modelado | scikit-learn, XGBoost |
| Dashboard | Plotly Dash, Plotly.js, CSS Grid |
| Deployment | Docker, Render |
| Control de versiones | Git + GitHub (GitHub Flow) |
| Entorno | VS Code, Jupyter Notebook |

---

## 19. Dashboard Interactivo

Temática **Wild-West Saloon** con 8 pestañas:

| Pestaña | Contenido |
|---------|-----------|
| **Panel de Control** | Señal del día (ALZA/ESTABLE/PRECAUCIÓN), certeza, precio + MA21, predicción vs realidad, rendimiento acumulado |
| **Precio** | Gráfico histórico con RSI, MACD, volatilidad, rango de fechas seleccionable (1D a HIST), selector de unidad (USD/oz, %, indexado) |
| **Indicadores** | RSI, MACD, volumen con selector de fechas |
| **Macro** | Correlaciones DXY/VIX/TNX con desplegable explicativo |
| **Backtest** | Estrategia ML vs Buy & Hold, alpha generado |
| **Simulación** | Simulador de trading con capital inicial y rango de fechas |
| **Métricas** | Importancia de variables (interactivo por categoría), matriz de confusión, ROC, tabla comparativa de modelos (clasificación + regresión) |
| **Metodología** | Pipeline, modelos, equipo, stack, repositorio (QR) |

### Ejecución

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

---

## 20. Modelos Pre-entrenados

12 modelos entrenados con split temporal 80/20, escalado sin data leakage:

| Modelo | Target | F1 (test) | Accuracy |
|--------|--------|-----------|----------|
| lr_strong_reg_binary | Binario | 0.70 | 56.9% |
| lr_binary | Binario | 0.69 | 56.2% |
| xgb_binary | Binario | 0.67 | 56.5% |
| rf_binary | Binario | 0.66 | 55.2% |
| lr_multiclass | Multiclase | 0.31 | 38.1% |

---

## 21. Principio del Proyecto

> "De datos crudos a decisiones de negocio explicables."
