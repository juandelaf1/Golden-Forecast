# Data Dictionary

## Raw Variables (from Yahoo Finance)

Cuatro tickers descargados con `yfinance` desde 2015-01-01:

- **GC=F** (gold): Futuros del oro (COMEX) — precio de referencia global
- **DX-Y.NYB** (dxy): Dólar index (ICE) — fortaleza del dólar frente a 6 divisas
- **^VIX** (vix): Volatilidad S&P 500 (CBOE) — índice de miedo del mercado
- **^TNX** (tnx): Treasury Yield 10 years (CBOE) — tasa de interés de referencia

Cada ticker incluye las columnas `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
Solo se usan las columnas `Close` (mergeadas como `gold_close`, `dxy_close`, `vix_close`, `tnx_close`).
En el notebook de clasificación también se usan `gold_open`, `gold_high`, `gold_low` para features OHLC.

---

## Engineered Features (gold-features.csv)

### Features introducidas en PR #39 (Market Sentiment & Relative Spreads)

- **dxy_return_lag_2** (float): Retorno DXY de hace 2 días — `dxy_return.shift(2)`
- **vix_return_lag_2** (float): Retorno VIX de hace 2 días — `vix_return.shift(2)`
- **gold_return_5d** (float): Retorno acumulado oro 5 días — `gold_close.pct_change(5)`
- **gold_return_10d** (float): Retorno acumulado oro 10 días — `gold_close.pct_change(10)`
- **vix_ma_20** (float): Media móvil VIX 20 días — `vix_close.rolling(20).mean()`
- **vix_high_fear** (int): Bandera VIX &gt; 25 (miedo alto) — `(vix_close > 25).astype(int)`
- **vix_extreme_fear** (int): Bandera VIX &gt; 35 (miedo extremo) — `(vix_close > 35).astype(int)`
- **gold_ma5_minus_ma20** (float): Diferencia MA corta vs larga — `gold_ma_5 - gold_ma_20`
- **gold_dxy_spread** (float): Spread oro vs dólar — `gold_return - dxy_return`

---

## Features de precio y retornos

- **gold_return_1d** (float): Retorno diario del oro — `gold_close.pct_change()`
- **gold_return_3d** (float): Retorno acumulado 3 días — `gold_close.pct_change(3)`
- **gold_return_5d** (float): Retorno acumulado 5 días — `gold_close.pct_change(5)`
- **gold_return_10d** (float): Retorno acumulado 10 días — `gold_close.pct_change(10)`
- **gold_return_21d** (float): Retorno acumulado 21 días — `gold_close.pct_change(21)`
- **sma_5** (float): Media móvil simple 5 días — `gold_close.rolling(5).mean()`
- **sma_10** (float): Media móvil simple 10 días — `gold_close.rolling(10).mean()`
- **sma_21** (float): Media móvil simple 21 días — `gold_close.rolling(21).mean()`
- **gold_close_to_sma21** (float): Desviación del precio respecto a SMA21 — `gold_close / sma_21 - 1`

### Features de rango diario (candlestick)

- **gold_range_pct** (float): Rango intradiario relativo — `(gold_high - gold_low) / gold_open`
- **gold_close_to_open** (float): Retorno open-to-close — `(gold_close - gold_open) / gold_open`
- **gold_open_to_close** (float): Retorno close-to-open (gap) — `(gold_open - gold_close.shift(1)) / gold_close.shift(1)`

### Features de volatilidad

- **vol_5d** (float): Volatilidad rolling 5 días — `gold_return_1d.rolling(5).std()`
- **vol_10d** (float): Volatilidad rolling 10 días — `gold_return_1d.rolling(10).std()`
- **vol_21d** (float): Volatilidad rolling 21 días — `gold_return_1d.rolling(21).std()`

### Osciladores técnicos

- **rsi_14** (float): RSI período 14 — sobrecompra &gt;70, sobreventa &lt;30 — Cálculo estándar Wilder
- **macd** (float): MACD línea principal — `ema_12 - ema_26`
- **macd_signal** (float): Señal MACD (media móvil de MACD) — `macd.ewm(span=9).mean()`
- **macd_hist** (float): Histograma MACD — `macd - macd_signal`

### Features macroeconómicas

- **dxy_return_1d** (float): Retorno diario del dólar — `dxy_close.pct_change()`
- **vix_return_1d** (float): Cambio diario del VIX — `vix_close.pct_change()`
- **tnx_return_1d** (float): Cambio diario del TNX — `tnx_close.pct_change()`
- **dxy_sma_5** (float): Media móvil DXY 5 días — `dxy_close.rolling(5).mean()`
- **dxy_sma_21** (float): Media móvil DXY 21 días — `dxy_close.rolling(21).mean()`
- **vix_sma_5** (float): Media móvil VIX 5 días — `vix_close.rolling(5).mean()`
- **vix_sma_21** (float): Media móvil VIX 21 días — `vix_close.rolling(21).mean()`
- **tnx_sma_5** (float): Media móvil TNX 5 días — `tnx_close.rolling(5).mean()`
- **tnx_sma_21** (float): Media móvil TNX 21 días — `tnx_close.rolling(21).mean()`

### Features de rezago (lags)

- **gold_lag_1d** (float): Precio cierre oro día anterior — `gold_close.shift(1)`
- **gold_lag_2d** (float): Precio cierre oro 2 días atrás — `gold_close.shift(2)`
- **gold_lag_3d** (float): Precio cierre oro 3 días atrás — `gold_close.shift(3)`
- **gold_lag_5d** (float): Precio cierre oro 5 días atrás — `gold_close.shift(5)`

---

## Target Variables

- **target_bin** (int): 0 = Down, 1 = Up — Clasificación binaria — ¿El oro subirá mañana? (dirección)
- **target_multi** (int): 0 = Baja, 1 = Estable, 2 = Sube — Clasificación multiclase — Rangos definidos por threshold 0.5%: &lt; -0.5% | -0.5% a 0.5% | &gt; 0.5%

---

## Feature Selection (dashboard)

El dashboard usa un subconjunto de features para el modelo en vivo:

`gold_return_1d`, `sma_5`, `sma_10`, `sma_21`, `vol_5d`, `rsi_14`, `macd`, `macd_signal`,
`dxy_return_1d`, `vix_return_1d`, `tnx_return_1d`, `dxy_close`, `vix_close`, `tnx_close`

Escaladas con `StandardScaler`. Modelo: `RandomForestClassifier(n_estimators=200, max_depth=10)`.

---

## Frecuencia de predicción

**Diaria.** El modelo se entrena con datos OHLC diarios (cierre) y predice la dirección del oro para el **siguiente día hábil**. No se hacen predicciones intradiarias ni semanales. Cada fila del dataset representa un día de trading.
