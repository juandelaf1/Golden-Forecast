# Final Product Test Report - Golden Forecast

Fecha de prueba: 2026-07-09  
Responsable: QA / Data Analyst review  
Alcance: validacion del producto final, tests automatizados, dashboard y prediccion diaria.

## 1. Resumen ejecutivo

El producto supera la suite principal de tests automatizados y el dashboard responde correctamente al endpoint de salud. Sin embargo, la prediccion diaria falla por una incompatibilidad entre las features esperadas por los modelos entrenados y las columnas disponibles actualmente en `data/processed/gold-features.csv`.

Estado general:

| Area | Resultado | Estado |
|---|---:|---|
| Tests automatizados con `pytest` | 37 passed, 1 skipped | OK |
| Health check del dashboard | HTTP 200 | OK |
| Construccion del layout Dash | Correcta | OK |
| Prediccion diaria `src/predict.py` | Falla por mismatch de features | Bloqueante |

Conclusion: el producto esta parcialmente validado. La capa de tests y dashboard esta estable, pero el flujo final de prediccion diaria no esta listo para entrega hasta resolver la incompatibilidad de features.

## 2. Comandos ejecutados

### 2.1 Suite automatizada

```bash
python -m pytest
```

Resultado:

```text
37 passed, 1 skipped
```

Nota: en el entorno local Windows fue necesario ejecutar `pytest` fuera del sandbox por permisos del sistema al crear recursos temporales y pipes usados por scikit-learn/joblib.

### 2.2 Health check del dashboard

```bash
python -c "from src.dashboard.app import server, app; client = server.test_client(); response = client.get('/health'); print(response.status_code); print(response.get_json()); print(type(app.layout).__name__)"
```

Resultado:

```text
200
{'service': 'golden-forecast', 'status': 'healthy'}
Div
```

Interpretacion: el servidor Flask/Dash carga correctamente, el endpoint `/health` responde y el layout principal se construye.

### 2.3 Prediccion diaria

```bash
python src/predict.py
```

Resultado: fallo de ejecucion.

Modelo seleccionado automaticamente:

```text
Modelo binario seleccionado    : rf_optimized_binary
Modelo multiclase seleccionado : lgb_multiclass
```

Error principal:

```text
ValueError: The feature names should match those that were passed during fit.
```

Despues del fallback, el script vuelve a fallar porque el CSV actual no contiene columnas esperadas por el scaler:

```text
KeyError: "['oil_volume', 'gvz_return', 'oil_return', 'sp500_return', ...] not in index"
```

## 3. Diagnostico tecnico

El archivo `models/scaler.pkl` y los modelos actuales fueron entrenados esperando 43 features. El dataset actual `data/processed/gold-features.csv` contiene 32 features predictoras.

Resumen:

| Elemento | Numero de features |
|---|---:|
| Features esperadas por `scaler.pkl` | 43 |
| Features disponibles en `gold-features.csv` | 32 |

Features esperadas que no existen actualmente en el CSV:

```text
bb_position
gold_atr_14_pct
gold_dxy_corr20
gold_ma_cross_pct
gold_macd_pct
gold_macd_signal_pct
gold_return_3d
gold_sp500_corr20
gold_sp500_spread
gvz_daily_range
gvz_open_close_return
gvz_return
gvz_return_lag1
oil_daily_range
oil_open_close_return
oil_return
oil_return_lag1
oil_volume
sp500_daily_range
sp500_open_close_return
sp500_return
sp500_return_lag1
vix_return_lag2
vix_vs_ma20
```

Features presentes en el CSV pero no esperadas por el scaler:

```text
dxy_return_lag_2
gold_daily_range
gold_ma5_minus_ma20
gold_ma_20
gold_ma_5
gold_macd
gold_macd_signal
gold_return
gold_return_lag_2
vix_extreme_fear
vix_high_fear
vix_ma_20
vix_return_lag_2
```

## 4. Metricas actuales de modelos

Segun `models/evaluation_results.json`:

| Metrica | Valor |
|---|---:|
| Mejor modelo binario por F1 | `rf_optimized_binary` |
| Accuracy test del mejor binario | 56.50% |
| F1 test del mejor binario | 69.31% |
| Mejor modelo multiclase | `lgb_multiclass` |
| Backtest binario | -3.90% |
| Buy & Hold del mismo periodo | 133.27% |
| Backtest multiclase | -61.49% |

Interpretacion: aunque hay modelos entrenados y metricas registradas, el flujo operativo de prediccion no puede usarlas correctamente con el CSV actual.

## 5. Riesgo para entrega

Este fallo es bloqueante para cualquier funcionalidad que dependa de `src/predict.py`, especialmente:

- Pipeline diario de GitHub Actions.
- Generacion de `data/processed/predictions.csv`.
- Prediccion del siguiente dia de mercado.
- Trazabilidad diaria de senales generadas.

El dashboard puede arrancar, pero el flujo automatico de prediccion diaria no queda validado.

## 6. Recomendacion de correccion

Hay dos caminos posibles:

1. Recomendado: actualizar `src/feature_engineering.py` y el pipeline de extraccion para generar las 43 features esperadas por los modelos actuales.
2. Alternativo: reentrenar modelos y `scaler.pkl` usando exactamente las 32 features que existen hoy en `gold-features.csv`.

Desde una perspectiva de ML Engineering, la opcion mas segura es que `train.py`, `evaluate.py`, `predict.py` y `feature_engineering.py` compartan una unica fuente de verdad para `feature_columns`. Asi se evita que el entrenamiento, la evaluacion y la inferencia usen esquemas distintos.

## 7. Criterio de aceptacion

El producto deberia considerarse listo cuando se cumplan estas condiciones:

- `python -m pytest` termina con todos los tests relevantes en verde.
- `/health` devuelve HTTP 200.
- `python src/predict.py` ejecuta sin errores.
- `data/processed/predictions.csv` se crea o actualiza correctamente.
- Las columnas usadas en inferencia coinciden exactamente con las columnas usadas en entrenamiento.
- `models/evaluation_results.json` corresponde al mismo set de features que `models/scaler.pkl` y `data/processed/gold-features.csv`.

## 8. Veredicto

Estado actual: no listo para entrega final completa.

Motivo: la prediccion diaria falla por incompatibilidad entre modelos/scaler y dataset de features.

Prioridad recomendada: alta.
