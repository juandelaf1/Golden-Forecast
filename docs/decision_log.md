# Decision Log

| Date | Decision | Justification |
|-------|----------|---------------|
| 29/06/2026 | Selección de dataset: GLD (Gold ETF) por Yahoo Finance | Datos públicos, históricos suficientes, volumen y precio disponibles |
| 30/06/2026 | María designada Product Owner | Responsable de visión de negocio y validación |
| 30/06/2026 | Juan designado Scrum Master | Responsable de coordinación y seguimiento |
| 30/06/2026 | Problema dual: clasificación + regresión | Aprovechar el mismo dataset para ambos enfoques y comparar resultados |
| 30/06/2026 | Ramas feature/ con PR a main protegida | Flujo profesional GitHub Flow con revisión de código |
| 30/06/2026 | Ticker: GC=F (futuros del oro) + DXY + VIX + TNX | Precio real del oro y 3 macroindicadores para enriquecer features |
| 30/06/2026 | Fecha inicio: 2015-01-01 | ~10 años de datos históricos (~2894 registros diarios) |
| 30/06/2026 | Documentación bilingüe | README, ROADMAP y PR template en inglés |
| 30/06/2026 | Repositorio profesional estructurado | README, ROADMAP, PR template, decision_log, project_handbook incluidos |
| 30/06/2026 | Kanban en GitHub Projects | Seguimiento visual del sprint con columnas To Do / In Progress / Done |
| 30/06/2026 | Data lineage documentado en `data/README.md` | Trazabilidad del origen y transformación de datos |
| 30/06/2026 | Data dictionary en `docs/data_dictionary.md` | Definición de todas las variables del modelo |
| 01/07/2026 | PR #2: Script de extracción `src/extract/extract.py` | Descarga automática de GC=F, DXY, VIX, TNX desde Yahoo Finance |
| 01/07/2026 | PR #3: Documentación base (handbook, PR template, data dictionary) | Estructura profesional del repositorio desde el inicio |
| 01/07/2026 | PR #27: Notebook de clasificación `03_classification.ipynb` | Modelos: Dummy, Logistic Regression, Random Forest, XGBoost |
| 01/07/2026 | PR #29: Dependencias de notebooks | Añadidas librerías faltantes en requirements.txt |
| 01/07/2026 | PR #31: Preprocesamiento `02_preprocessing.ipynb` + dataset procesado | Limpieza de datos: 2894 → 2884 filas (10 filas con nulos eliminadas) |
| 01/07/2026 | PR #32: Refactorización preprocessing en módulos `src/` | `preprocessing.py`, `feature_engineering.py`, `targets.py`, `pipeline.py` como scripts reutilizables |
| 01/07/2026 | PR #33: EDA `EDA_Golden_Forecast.ipynb` | Análisis exploratorio: estadísticas, distribuciones, correlaciones, visualizaciones |
| 01/07/2026 | Target multiclase con umbral 0.01 | 3 categorías: comprar (>1%), mantener, vender (<-1%) |
| 01/07/2026 | Features: retornos, rangos, open-close, medias móviles | 4 bloques de feature engineering en `feature_engineering.py` |
| 01/07/2026 | Split temporal (80/20) sin aleatorización | Respeta orden cronológico para series temporales |
| 01/07/2026 | Escalado sin data leakage | StandardScaler ajustado solo con datos de train |
| 02/07/2026 | Refactorización de `03_classification.ipynb` | Notebook limpio: solo clasificación, usa módulos de `src/` |
| 02/07/2026 | Creación de `src/classification.py` | Funciones reutilizables: get_models, evaluate, overfitting_check, backtest, feature_importance |
| 02/07/2026 | Merge de main en `feature/classification` | Integrados EDA, preprocessing, features, targets y pipeline |
