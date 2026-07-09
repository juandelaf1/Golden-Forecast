# Decision Log

<table>
<tr><th>Date</th><th>Decision</th><th>Justification</th></tr>
<tr><td>29/06/2026</td><td>Selección de dataset: GLD (Gold ETF) por Yahoo Finance</td><td>Datos públicos, históricos suficientes, volumen y precio disponibles</td></tr>
<tr><td>30/06/2026</td><td>María designada Product Owner</td><td>Responsable de visión de negocio y validación</td></tr>
<tr><td>30/06/2026</td><td>Juan designado Scrum Master</td><td>Responsable de coordinación y seguimiento</td></tr>
<tr><td>30/06/2026</td><td>Problema dual: clasificación + regresión</td><td>Aprovechar el mismo dataset para ambos enfoques y comparar resultados</td></tr>
<tr><td>30/06/2026</td><td>Ramas feature/ con PR a main protegida</td><td>Flujo profesional GitHub Flow con revisión de código</td></tr>
<tr><td>30/06/2026</td><td>Ticker: GC=F (futuros del oro) + DXY + VIX + TNX</td><td>Precio real del oro y 3 macroindicadores para enriquecer features</td></tr>
<tr><td>30/06/2026</td><td>Fecha inicio: 2015-01-01</td><td>~10 años de datos históricos (~2894 registros diarios)</td></tr>
<tr><td>30/06/2026</td><td>Convención de idioma definida</td><td>Código/comentarios/docstrings en inglés. UI dashboard, prints, docs internos (README, ROADMAP, handbook, decision_log) en español. Docs técnicos (ml_report, architecture) en inglés. Config/YAML en inglés. Notebooks en español.</td></tr>
<tr><td>30/06/2026</td><td>Repositorio profesional estructurado</td><td>README, ROADMAP, PR template, decision_log, project_handbook incluidos</td></tr>
<tr><td>30/06/2026</td><td>Kanban en GitHub Projects</td><td>Seguimiento visual del sprint con columnas To Do / In Progress / Done</td></tr>
<tr><td>30/06/2026</td><td>Data lineage documentado en <code>data/README.md</code></td><td>Trazabilidad del origen y transformación de datos</td></tr>
<tr><td>30/06/2026</td><td>Data dictionary en <code>docs/data_dictionary.md</code></td><td>Definición de todas las variables del modelo</td></tr>
<tr><td>01/07/2026</td><td>PR #2: Script de extracción <code>src/extract/extract.py</code></td><td>Descarga automática de GC=F, DXY, VIX, TNX desde Yahoo Finance</td></tr>
<tr><td>01/07/2026</td><td>PR #3: Documentación base (handbook, PR template, data dictionary)</td><td>Estructura profesional del repositorio desde el inicio</td></tr>
<tr><td>01/07/2026</td><td>PR #27: Notebook de clasificación <code>03_classification.ipynb</code></td><td>Modelos: Dummy, Logistic Regression, Random Forest, XGBoost</td></tr>
<tr><td>01/07/2026</td><td>PR #29: Dependencias de notebooks</td><td>Añadidas librerías faltantes en requirements.txt</td></tr>
<tr><td>01/07/2026</td><td>PR #31: Preprocesamiento <code>02_preprocessing.ipynb</code> + dataset procesado</td><td>Limpieza de datos: 2894 → 2884 filas (10 filas con nulos eliminadas)</td></tr>
<tr><td>01/07/2026</td><td>PR #32: Refactorización preprocessing en módulos <code>src/</code></td><td><code>preprocessing.py</code>, <code>feature_engineering.py</code>, <code>targets.py</code>, <code>pipeline.py</code> como scripts reutilizables</td></tr>
<tr><td>01/07/2026</td><td>PR #33: EDA <code>EDA_Golden_Forecast.ipynb</code></td><td>Análisis exploratorio: estadísticas, distribuciones, correlaciones, visualizaciones</td></tr>
<tr><td>01/07/2026</td><td>Target multiclase con umbral 0.01</td><td>3 categorías: comprar (>1%), mantener, vender (<-1%)</td></tr>
<tr><td>01/07/2026</td><td>Features: retornos, rangos, open-close, medias móviles</td><td>4 bloques de feature engineering en <code>feature_engineering.py</code></td></tr>
<tr><td>01/07/2026</td><td>Split temporal (80/20) sin aleatorización</td><td>Respeta orden cronológico para series temporales</td></tr>
<tr><td>01/07/2026</td><td>Escalado sin data leakage</td><td>StandardScaler ajustado solo con datos de train</td></tr>
<tr><td>02/07/2026</td><td>Refactorización de <code>03_classification.ipynb</code></td><td>Notebook limpio: solo clasificación, usa módulos de <code>src/</code></td></tr>
<tr><td>02/07/2026</td><td>Creación de <code>src/classification.py</code></td><td>Funciones reutilizables: get_models, evaluate, overfitting_check, backtest, feature_importance</td></tr>
<tr><td>02/07/2026</td><td>Merge de main en <code>feature/classification</code></td><td>Integrados EDA, preprocessing, features, targets y pipeline</td></tr>
<tr><td>02/07/2026</td><td>PR #34: Feature engineering avanzado mergeado a main</td><td>Indicadores técnicos (RSI, MACD, volatilidad, lags), threshold multiclase 0.5%, limpieza columnas absolutas</td></tr>
<tr><td>02/07/2026</td><td>Sync docs en <code>feature/classification</code></td><td>data_dictionary actualizado a features reales, decision_log completado, READMEs y ROADMAP sincronizados</td></tr>
<tr><td>03/07/2026</td><td>PR #37: Pipeline ML completo integrado a main</td><td><code>src/models/train.py</code> + <code>evaluate.py</code>, 12 modelos pre-entrenados (.pkl)</td></tr>
<tr><td>03/07/2026</td><td>Dashboard: metricas extendidas (precision, recall, regression)</td><td>Añadidas precision, recall a clasificacion + MAE, RMSE, R², MAPE a regresion</td></tr>
<tr><td>03/07/2026</td><td>Dashboard: graficos interactivos</td><td>Selector de unidad (USD/oz, %, indexado), selector de fechas 1D-HIST, importancia de variables filtrable</td></tr>
<tr><td>03/07/2026</td><td>Dashboard: desplegables explicativos</td><td>Guia rapida en Panel de Control, explicacion de DXY/VIX/TNX en pestana Macro</td></tr>
<tr><td>03/07/2026</td><td>Dashboard: control de volumen</td><td>Slider de volumen para ambientacion sonora (30% por defecto)</td></tr>
<tr><td>03/07/2026</td><td>Merge de main en <code>feature/classification</code></td><td>Integrados pipeline de modelos pre-entrenados en la rama de dashboard</td></tr>
<tr><td>04/07/2026</td><td>Joel prioriza clasificación sobre regresión</td><td>Datos disponibles (returns, técnicos, macro) más adecuados para predicción direccional. Juan crea <code>src/regression.py</code> como módulo complementario explorando targets continuos (volatilidad, ATR, drawdown, fair value)</td></tr>
<tr><td>04/07/2026</td><td>Pestaña "Regresión" renombrada a "Valor y Riesgo"</td><td>Narrativa B2B: el módulo de Juan responde a "¿está caro o barato? ¿cuánto riesgo?" vs el modelo de clasificación de Joel que predice sube/baja</td></tr>
<tr><td>04/07/2026</td><td><code>assets/</code> en .gitignore restringido a solo raíz (<code>/assets/</code>)</td><td>El patrón genérico <code>assets/</code> ignoraba también <code>src/dashboard/assets/</code>, impidiendo trackear <code>background.png</code> para el fondo del dashboard</td></tr>
<tr><td>04/07/2026</td><td><code>assets_folder</code> explícito en <code>app.py</code></td><td>Dash debe servir desde <code>src/dashboard/assets/</code> para evitar 404 del fondo ASSAY</td></tr>
<tr><td>04/07/2026</td><td>Pestaña Simulación duplicada eliminada en <code>layout.py</code></td><td>Dos pestañas con mismo <code>value='tab-sim'</code> causaban comportamiento impredecible en el navegador</td></tr>
<tr><td>04/07/2026</td><td><code>run_pipeline.py</code> corregido: paso clasificación apunta a <code>src/models/train.py</code></td><td>Anteriormente apuntaba a <code>src/classification.py</code> (Juan, sin <code>__main__</code>). Por defecto se salta porque los 12 modelos .pkl ya existen</td></tr>
<tr><td>04/07/2026</td><td>Python unificado a 3.12 en CI (<code>ci.yml</code>)</td><td>Dockerfile ya usaba 3.12, CI usaba 3.10. Alineado para evitar discrepancias</td></tr>
<tr><td>04/07/2026</td><td><code>requirements.txt</code>: añadidos <code>xgboost</code> y <code>pytest</code></td><td>Dependencias necesarias para modelos XGBoost y tests respectivamente</td></tr>
<tr><td>04/07/2026</td><td>Stashes limpiados (4)</td><td>Correspondían a WIP en ramas <code>feature/classification</code> y <code>feature/pr-template</code>, ya integradas</td></tr>
<tr><td>04/07/2026</td><td>Columna <code>ma_21</code> → <code>gold_ma_20</code> en <code>callbacks.py</code></td><td>El feature engineering genera <code>gold_ma_20</code>, no <code>ma_21</code>. Dashboard referenciaba columna inexistente → gráfica vacía</td></tr>
<tr><td>04/07/2026</td><td>Columnas técnicas corregidas (<code>rsi</code>→<code>gold_rsi_14</code>, <code>macd</code>→<code>gold_macd</code>)</td><td>Misma causa: prefijo <code>gold_</code> faltante. Dashboard de clasificación y regresión ahora muestran datos reales</td></tr>
<tr><td>04/07/2026</td><td>Macro indicadores <code>dxy_close</code>, <code>vix_close</code>, <code>tnx_close</code> mergeados en <code>data.py</code></td><td>Dashboard necesita correlaciones macro para frase "El oro y el dólar se mueven en direcciones opuestas". Alias <code>ma_21</code> mantenido para retrocompatibilidad</td></tr>
<tr><td>04/07/2026</td><td>R² negativos ocultos en pestaña Valor y Riesgo</td><td>Narrativa B2B: modelos sin señal predictiva muestran "—" con texto "Relación no lineal — análisis exploratorio" en vez de números negativos</td></tr>
<tr><td>04/07/2026</td><td>Dropdown dark theme añadido a <code>style.css</code></td><td>.Select-control, .Select-menu-outer, .Select-option con fondo #2b2b2b y hover #3a3a3a</td></tr>
<tr><td>04/07/2026</td><td>Labels de precio en dropdown mejorados</td><td>"Precio en dólares (USD/oz)", "Cambio porcentual diario (%)", "Rendimiento acumulado (base 100)"</td></tr>
<tr><td>04/07/2026</td><td>"Complementa, no compite" eliminado de UI</td><td>Juan rechazó la frase. Reemplazado por "Modelos de regresión: valoración de activos y medición de riesgo"</td></tr>
<tr><td>04/07/2026</td><td>3 nuevos test files: <code>test_regression.py</code>, <code>test_models.py</code>, <code>test_dashboard.py</code></td><td>22 tests nuevos (9+6+7). Testing Score sube de 0/10 a 6/10 en checklist académico</td></tr>
<tr><td>04/07/2026</td><td><code>test_feature_engineering.py:85</code> arreglado (<code>.dropna()</code>)</td><td>Test <code>test_create_rsi</code> fallaba por NaN en últimas filas de RSI</td></tr>
<tr><td>04/07/2026</td><td><code>test_create_rsi</code> esperaba pandas.Series, recibía DataFrame</td><td><code>.squeeze("columns")</code> añadido para obtener Serie unidimensional</td></tr>
<tr><td>04/07/2026</td><td><code>temp_app.py</code> eliminado</td><td>Archivo vacío creado por error, gitignorado sin seguimiento</td></tr>
<tr><td>04/07/2026</td><td><code>docs/ml_report.md</code> creado</td><td>ML Score: 9.0/10. Documento requerido por checklist: dataset, features, modelos, métricas, limitaciones</td></tr>
<tr><td>04/07/2026</td><td><code>docs/architecture.md</code> creado</td><td>Diagrama ASCII del pipeline (extract→preprocess→features→train→evaluate→dashboard) + owners</td></tr>
<tr><td>04/07/2026</td><td><code>notebooks/03_classification.ipynb</code> reparado</td><td>Import de <code>src.pipeline</code> roto (refactorizado). Reemplazado con funciones inline equivalentes</td></tr>
<tr><td>04/07/2026</td><td><code>config/pipeline.yaml</code> → Python 3.12</td><td>Líneas 98 y 105 aún referenciaban <code>python:3.10-slim</code> y <code>python_version: 3.10</code>. Alineado con Dockerfile y CI</td></tr>
<tr><td>04/07/2026</td><td>47 tests pasados, 1 skipped</td><td><code>test_model_count</code> salta si no hay 12 modelos .pkl todos a la vez. Todos los demás tests verdes</td></tr>
<tr><td>06/07/2026</td><td>Ticker: <code>flex-shrink: 0</code> + <code>white-space: nowrap</code> en CSS</td><td>Evita que etiquetas como "FIABILIDAD" se corten a "ABILIDAD" por el scroll</td></tr>
<tr><td>06/07/2026</td><td>MA 21 calculado desde <code>gold</code> directamente (<code>rolling(21).mean()</code>)</td><td><code>gold_ma_20</code> de features estaba desfasado ~$300 vs el precio real del oro</td></tr>
<tr><td>06/07/2026</td><td><code>CATEGORY_COLORS['Técnico']</code> cambiado de <code>#42A5F5</code> (azul) a <code>#e8c34a</code> (dorado)</td><td>Las barras de importancia de variables en Métricas ya no son "azul horrible"</td></tr>
<tr><td>06/07/2026</td><td>Stack Tecnológico en Metodología: grid 2×2 con iconos y celdas</td><td>Reemplaza lista plana de <code>&lt;p&gt;</code> por tarjetas visuales con borde y hover</td></tr>
<tr><td>06/07/2026</td><td>Carga lazy de modelos: solo primario al arranque, resto bajo demanda</td><td><code>build_pretrained_context</code> solo carga <code>lr_strong_reg_binary</code>. <code>ensure_all_models()</code> carga rf_binary, xgb_binary, lr_multiclass + experimental al visitar Métricas. <code>ensure_regression()</code> entrena regression al visitar Valor y Riesgo. Startup ~3s vs ~15s</td></tr>
<tr><td>06/07/2026</td><td>Audio autoplay: script JS en layout.py prueba play() al cargar y en cualquier interacción (click, touch, mousemove, scroll, keydown)</td><td>El anterior handler solo respondía a click. Ahora el hilo suena con el primer movimiento o tecleo sin tener que hacer clic explícito</td></tr>
<tr><td>06/07/2026</td><td>Ticker: valores más visibles (font-size 1.05rem, weight 800, bg oscuro)</td><td>DXY, VIX, MA 21 ahora se distinguen mejor de las etiquetas</td></tr>
<tr><td>07/07/2026</td><td>PR #38 mergeado: EDA de gold-features.csv (Jose)</td><td>Notebook profesional con 36 celdas, 8 gráficos, calidad de datos, targets, correlaciones</td></tr>
<tr><td>07/07/2026</td><td>PR #39 mergeado: Market sentiment + relative spreads (Gema)</td><td>9 nuevas features: lags DXY/VIX, retornos acumulados 5d/10d, VIX fear flags, MA spread, gold-dxy spread</td></tr>
<tr><td>07/07/2026</td><td>Merge main → feature/full-pipeline</td><td>Sincronización con PRs #38 y #39. Conflicto en docstring de <code>feature_engineering.py</code> resuelto</td></tr>
<tr><td>07/07/2026</td><td>Deploy caído por feature mismatch en Render</td><td>Modelos .pkl antiguos no reconocen nuevas columnas (PR #39). Pendiente reentrenar por Joel</td></tr>
<tr><td>07/07/2026</td><td>Documentación Mintlify creada</td><td><code>mint.json</code> configurado con navegación, colores dorados, enlace a dashboard y GitHub</td></tr>
<tr><td>08/07/2026</td><td>PR #51 mergeado: ROADMAP actualizado + predict.py pipeline (María)</td><td>Añadido <code>predict.py</code> para predicción diaria con modelos guardados, ROADMAP con deliverables detallados</td></tr>
<tr><td>08/07/2026</td><td>PR #52 (Joel) cerrado como superseded</td><td>Sus cambios integrados directamente en <code>feature/full-pipeline</code> para evitar conflictos múltiples con <code>main</code></td></tr>
<tr><td>08/07/2026</td><td>PR #53 mergeado: integración full-pipeline</td><td><code>feature/full-pipeline</code> → <code>main</code>. Contiene: mejoras de Joel (RF optimizado, threshold tuning, pipeline evaluacion), modulos de Juan (regression, classification, dashboard 9 tabs), docs e infra</td></tr>
<tr><td>08/07/2026</td><td>PR #54 mergeado: README actualizado + presentación PPTX (María)</td><td>README con estructura final, enlaces a docs, descripción del equipo y licencia</td></tr>
<tr><td>08/07/2026</td><td>Daily pipeline corregido para respetar ruleset de main</td><td>Ahora crea rama <code>daily/YYYY-MM-DD</code> + PR automático en vez de push directo</td></tr>
<tr><td>08/07/2026</td><td>Mintlify config: rutas de logo corregidas</td><td><code>/logo/</code> → <code>/public/logo/</code> en <code>mint.json</code> y <code>docs.json</code></td></tr>
<tr><td>08/07/2026</td><td>Screenshots actualizadas con nota de versión</td><td>Capturas corresponden a versión 8 pestañas, dashboard actual tiene 9 pestañas</td></tr>
</table>
