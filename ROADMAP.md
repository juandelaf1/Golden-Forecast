# Roadmap — Golden Forecast

## Sprint 1 (Days 1-3): Data & Preprocessing ✅

| Milestone | Asignado | Estado |
|-----------|----------|--------|
| EDA completo | José | ✅ |
| Preprocessing + Feature Engineering | Gema | ✅ |
| Pipeline `src/` modular | Gema | ✅ |
| Datos extraídos (Yahoo Finance) | Juan (SM) | ✅ |

## Sprint 2 (Days 4-6): Modeling ✅

| Milestone | Asignado | Estado |
|-----------|----------|--------|
| Clasificación (LR, RF, XGB) binaria + multiclase | Juan | ✅ `src/classification.py` + `03_classification.ipynb` |
| Pipeline reutilizable `src/models/train.py` | Joel | ✅ 12 modelos pre-entrenados |
| Evaluación automatizada `src/models/evaluate.py` | Joel | ✅ Métricas + backtest + overfitting check |
| Dashboard interactivo Plotly Dash | Juan | ✅ 8 pestañas, selector de fechas, unidad, modelos |

## Sprint 3 (Days 7-8): Evaluación & Cierre

| Milestone | Asignado | Estado |
|-----------|----------|--------|
| Integración modelos pre-entrenados en dashboard | Juan | 🔄 En progreso |
| Presentación final con narrativa de negocio | María (lead) + todos | Pendiente |
| Documentación completa y cierre del repo | Juan (SM) | 🔄 En progreso |

**Entrega final**: Dashboard funcional + Presentación + Repositorio documentado

---

## Módulos del proyecto

| Archivo | Función |
|---------|---------|
| `src/extract/extract.py` | Descarga Yahoo Finance (GC=F, DXY, VIX, TNX) |
| `src/preprocessing.py` | Limpieza y renombrado de columnas |
| `src/feature_engineering.py` | 24 features técnicas y macro |
| `src/models/train.py` | Entrenamiento de 6 modelos x 2 targets |
| `src/models/evaluate.py` | Evaluación, backtesting, overfitting check |
| `src/dashboard/` | App Dash: 8 pestañas, gráficos interactivos |
| `src/dashboard/model_loader.py` | Carga de modelos pre-entrenados |
