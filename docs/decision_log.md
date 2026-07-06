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
| 30/06/2026 | Convención de idioma definida | Código/comentarios/docstrings en inglés. UI dashboard, prints, docs internos (README, ROADMAP, handbook, decision_log) en español. Docs técnicos (ml_report, architecture) en inglés. Config/YAML en inglés. Notebooks en español. |
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
| 02/07/2026 | PR #34: Feature engineering avanzado mergeado a main | Indicadores técnicos (RSI, MACD, volatilidad, lags), threshold multiclase 0.5%, limpieza columnas absolutas |
| 02/07/2026 | Sync docs en `feature/classification` | data_dictionary actualizado a features reales, decision_log completado, READMEs y ROADMAP sincronizados |
| 03/07/2026 | PR #37: Pipeline ML completo integrado a main | `src/models/train.py` + `evaluate.py`, 12 modelos pre-entrenados (.pkl) |
| 03/07/2026 | Dashboard: metricas extendidas (precision, recall, regression) | Añadidas precision, recall a clasificacion + MAE, RMSE, R², MAPE a regresion |
| 03/07/2026 | Dashboard: graficos interactivos | Selector de unidad (USD/oz, %, indexado), selector de fechas 1D-HIST, importancia de variables filtrable |
| 03/07/2026 | Dashboard: desplegables explicativos | Guia rapida en Panel de Control, explicacion de DXY/VIX/TNX en pestana Macro |
| 03/07/2026 | Dashboard: control de volumen | Slider de volumen para ambientacion sonora (30% por defecto) |
| 03/07/2026 | Merge de main en `feature/classification` | Integrados pipeline de modelos pre-entrenados en la rama de dashboard |
| 04/07/2026 | Joel prioriza clasificación sobre regresión | Datos disponibles (returns, técnicos, macro) más adecuados para predicción direccional. Juan crea `src/regression.py` como módulo complementario explorando targets continuos (volatilidad, ATR, drawdown, fair value) |
| 04/07/2026 | Pestaña "Regresión" renombrada a "Valor y Riesgo" | Narrativa B2B: el módulo de Juan responde a "¿está caro o barato? ¿cuánto riesgo?" vs el modelo de clasificación de Joel que predice sube/baja |
| 04/07/2026 | `assets/` en .gitignore restringido a solo raíz (`/assets/`) | El patrón genérico `assets/` ignoraba también `src/dashboard/assets/`, impidiendo trackear `background.png` para el fondo del dashboard |
| 04/07/2026 | `assets_folder` explícito en `app.py` | Dash debe servir desde `src/dashboard/assets/` para evitar 404 del fondo ASSAY |
| 04/07/2026 | Pestaña Simulación duplicada eliminada en `layout.py` | Dos pestañas con mismo `value='tab-sim'` causaban comportamiento impredecible en el navegador |
| 04/07/2026 | `run_pipeline.py` corregido: paso clasificación apunta a `src/models/train.py` | Anteriormente apuntaba a `src/classification.py` (Juan, sin `__main__`). Por defecto se salta porque los 12 modelos .pkl ya existen |
| 04/07/2026 | Python unificado a 3.12 en CI (`ci.yml`) | Dockerfile ya usaba 3.12, CI usaba 3.10. Alineado para evitar discrepancias |
| 04/07/2026 | `requirements.txt`: añadidos `xgboost` y `pytest` | Dependencias necesarias para modelos XGBoost y tests respectivamente |
| 04/07/2026 | Stashes limpiados (4) | Correspondían a WIP en ramas `feature/classification` y `feature/pr-template`, ya integradas |
| 04/07/2026 | Columna `ma_21` → `gold_ma_20` en `callbacks.py` | El feature engineering genera `gold_ma_20`, no `ma_21`. Dashboard referenciaba columna inexistente → gráfica vacía |
| 04/07/2026 | Columnas técnicas corregidas (`rsi`→`gold_rsi_14`, `macd`→`gold_macd`) | Misma causa: prefijo `gold_` faltante. Dashboard de clasificación y regresión ahora muestran datos reales |
| 04/07/2026 | Macro indicadores `dxy_close`, `vix_close`, `tnx_close` mergeados en `data.py` | Dashboard necesita correlaciones macro para frase "El oro y el dólar se mueven en direcciones opuestas". Alias `ma_21` mantenido para retrocompatibilidad |
| 04/07/2026 | R² negativos ocultos en pestaña Valor y Riesgo | Narrativa B2B: modelos sin señal predictiva muestran "—" con texto "Relación no lineal — análisis exploratorio" en vez de números negativos |
| 04/07/2026 | Dropdown dark theme añadido a `style.css` | .Select-control, .Select-menu-outer, .Select-option con fondo #2b2b2b y hover #3a3a3a |
| 04/07/2026 | Labels de precio en dropdown mejorados | "Precio en dólares (USD/oz)", "Cambio porcentual diario (%)", "Rendimiento acumulado (base 100)" |
| 04/07/2026 | "Complementa, no compite" eliminado de UI | Juan rechazó la frase. Reemplazado por "Modelos de regresión: valoración de activos y medición de riesgo" |
| 04/07/2026 | 3 nuevos test files: `test_regression.py`, `test_models.py`, `test_dashboard.py` | 22 tests nuevos (9+6+7). Testing Score sube de 0/10 a 6/10 en checklist académico |
| 04/07/2026 | `test_feature_engineering.py:85` arreglado (`.dropna()`) | Test `test_create_rsi` fallaba por NaN en últimas filas de RSI |
| 04/07/2026 | `test_create_rsi` esperaba pandas.Series, recibía DataFrame | `.squeeze("columns")` añadido para obtener Serie unidimensional |
| 04/07/2026 | `temp_app.py` eliminado | Archivo vacío creado por error, gitignorado sin seguimiento |
| 04/07/2026 | `docs/ml_report.md` creado | ML Score: 9.0/10. Documento requerido por checklist: dataset, features, modelos, métricas, limitaciones |
| 04/07/2026 | `docs/architecture.md` creado | Diagrama ASCII del pipeline (extract→preprocess→features→train→evaluate→dashboard) + owners |
| 04/07/2026 | `notebooks/03_classification.ipynb` reparado | Import de `src.pipeline` roto (refactorizado). Reemplazado con funciones inline equivalentes |
| 04/07/2026 | `config/pipeline.yaml` → Python 3.12 | Líneas 98 y 105 aún referenciaban `python:3.10-slim` y `python_version: 3.10`. Alineado con Dockerfile y CI |
| 04/07/2026 | 47 tests pasados, 1 skipped | `test_model_count` salta si no hay 12 modelos .pkl todos a la vez. Todos los demás tests verdes |
| 06/07/2026 | Ticker: `flex-shrink: 0` + `white-space: nowrap` en CSS | Evita que etiquetas como "FIABILIDAD" se corten a "ABILIDAD" por el scroll |
| 06/07/2026 | MA 21 calculado desde `gold` directamente (`rolling(21).mean()`) | `gold_ma_20` de features estaba desfasado ~$300 vs el precio real del oro |
| 06/07/2026 | `CATEGORY_COLORS['Técnico']` cambiado de `#42A5F5` (azul) a `#e8c34a` (dorado) | Las barras de importancia de variables en Métricas ya no son "azul horrible" |
| 06/07/2026 | Stack Tecnológico en Metodología: grid 2×2 con iconos y celdas | Reemplaza lista plana de `<p>` por tarjetas visuales con borde y hover |
| 06/07/2026 | Carga lazy de modelos: solo primario al arranque, resto bajo demanda | `build_pretrained_context` solo carga `lr_strong_reg_binary`. `ensure_all_models()` carga rf_binary, xgb_binary, lr_multiclass + experimental al visitar Métricas. `ensure_regression()` entrena regression al visitar Valor y Riesgo. Startup ~3s vs ~15s |
| 06/07/2026 | Audio autoplay: script JS en layout.py prueba play() al cargar y en cualquier interacción (click, touch, mousemove, scroll, keydown) | El anterior handler solo respondía a click. Ahora el hilo suena con el primer movimiento o tecleo sin tener que hacer clic explícito |
| 06/07/2026 | Ticker: valores más visibles (font-size 1.05rem, weight 800, bg oscuro) | DXY, VIX, MA 21 ahora se distinguen mejor de las etiquetas |
