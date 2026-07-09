# Data Dictionary

## Raw Variables (from Yahoo Finance)

Cuatro tickers descargados con `yfinance` desde 2015-01-01:

<table>
<tr><th>Ticker</th><th>Mnemónico</th><th>Descripción</th></tr>
<tr><td>GC=F</td><td>gold</td><td>Futuros del oro (COMEX) — precio de referencia global</td></tr>
<tr><td>DX-Y.NYB</td><td>dxy</td><td>Dólar index (ICE) — fortaleza del dólar frente a 6 divisas</td></tr>
<tr><td>^VIX</td><td>vix</td><td>Volatilidad S&amp;P 500 (CBOE) — índice de miedo del mercado</td></tr>
<tr><td>^TNX</td><td>tnx</td><td>Treasury Yield 10 years (CBOE) — tasa de interés de referencia</td></tr>
</table>

Cada ticker incluye las columnas `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
Solo se usan las columnas `Close` (mergeadas como `gold_close`, `dxy_close`, `vix_close`, `tnx_close`).
En el notebook de clasificación también se usan `gold_open`, `gold_high`, `gold_low` para features OHLC.

---

## Engineered Features (gold-features.csv)

### Features introducidas en PR #39 (Market Sentiment & Relative Spreads)

<table>
<tr><th>Variable</th><th>Tipo</th><th>Descripción</th><th>Fórmula</th></tr>
<tr><td>dxy_return_lag_2</td><td>float</td><td>Retorno DXY de hace 2 días</td><td>`dxy_return.shift(2)`</td></tr>
<tr><td>vix_return_lag_2</td><td>float</td><td>Retorno VIX de hace 2 días</td><td>`vix_return.shift(2)`</td></tr>
<tr><td>gold_return_5d</td><td>float</td><td>Retorno acumulado oro 5 días</td><td>`gold_close.pct_change(5)`</td></tr>
<tr><td>gold_return_10d</td><td>float</td><td>Retorno acumulado oro 10 días</td><td>`gold_close.pct_change(10)`</td></tr>
<tr><td>vix_ma_20</td><td>float</td><td>Media móvil VIX 20 días</td><td>`vix_close.rolling(20).mean()`</td></tr>
<tr><td>vix_high_fear</td><td>int</td><td>Bandera VIX &gt; 25 (miedo alto)</td><td>`(vix_close &gt; 25).astype(int)`</td></tr>
<tr><td>vix_extreme_fear</td><td>int</td><td>Bandera VIX &gt; 35 (miedo extremo)</td><td>`(vix_close &gt; 35).astype(int)`</td></tr>
<tr><td>gold_ma5_minus_ma20</td><td>float</td><td>Diferencia MA corta vs larga</td><td>`gold_ma_5 - gold_ma_20`</td></tr>
<tr><td>gold_dxy_spread</td><td>float</td><td>Spread oro vs dólar</td><td>`gold_return - dxy_return`</td></tr>
</table>

---

## Features de precio y retornos

<table>
<tr><th>Variable</th><th>Tipo</th><th>Descripción</th><th>Fórmula</th></tr>
<tr><td>gold_return_1d</td><td>float</td><td>Retorno diario del oro</td><td>`gold_close.pct_change()`</td></tr>
<tr><td>gold_return_3d</td><td>float</td><td>Retorno acumulado 3 días</td><td>`gold_close.pct_change(3)`</td></tr>
<tr><td>gold_return_5d</td><td>float</td><td>Retorno acumulado 5 días</td><td>`gold_close.pct_change(5)`</td></tr>
<tr><td>gold_return_10d</td><td>float</td><td>Retorno acumulado 10 días</td><td>`gold_close.pct_change(10)`</td></tr>
<tr><td>gold_return_21d</td><td>float</td><td>Retorno acumulado 21 días</td><td>`gold_close.pct_change(21)`</td></tr>
<tr><td>sma_5</td><td>float</td><td>Media móvil simple 5 días</td><td>`gold_close.rolling(5).mean()`</td></tr>
<tr><td>sma_10</td><td>float</td><td>Media móvil simple 10 días</td><td>`gold_close.rolling(10).mean()`</td></tr>
<tr><td>sma_21</td><td>float</td><td>Media móvil simple 21 días</td><td>`gold_close.rolling(21).mean()`</td></tr>
<tr><td>gold_close_to_sma21</td><td>float</td><td>Desviación del precio respecto a SMA21</td><td>`gold_close / sma_21 - 1`</td></tr>
</table>

### Features de rango diario (candlestick)

<table>
<tr><th>Variable</th><th>Tipo</th><th>Descripción</th><th>Fórmula</th></tr>
<tr><td>gold_range_pct</td><td>float</td><td>Rango intradiario relativo</td><td>`(gold_high - gold_low) / gold_open`</td></tr>
<tr><td>gold_close_to_open</td><td>float</td><td>Retorno open-to-close</td><td>`(gold_close - gold_open) / gold_open`</td></tr>
<tr><td>gold_open_to_close</td><td>float</td><td>Retorno close-to-open (gap)</td><td>`(gold_open - gold_close.shift(1)) / gold_close.shift(1)`</td></tr>
</table>

### Features de volatilidad

<table>
<tr><th>Variable</th><th>Tipo</th><th>Descripción</th><th>Fórmula</th></tr>
<tr><td>vol_5d</td><td>float</td><td>Volatilidad rolling 5 días</td><td>`gold_return_1d.rolling(5).std()`</td></tr>
<tr><td>vol_10d</td><td>float</td><td>Volatilidad rolling 10 días</td><td>`gold_return_1d.rolling(10).std()`</td></tr>
<tr><td>vol_21d</td><td>float</td><td>Volatilidad rolling 21 días</td><td>`gold_return_1d.rolling(21).std()`</td></tr>
</table>

### Osciladores técnicos

<table>
<tr><th>Variable</th><th>Tipo</th><th>Descripción</th><th>Fórmula</th></tr>
<tr><td>rsi_14</td><td>float</td><td>RSI período 14 — sobrecompra &gt;70, sobreventa &lt;30</td><td>Cálculo estándar Wilder</td></tr>
<tr><td>macd</td><td>float</td><td>MACD línea principal</td><td>`ema_12 - ema_26`</td></tr>
<tr><td>macd_signal</td><td>float</td><td>Señal MACD (media móvil de MACD)</td><td>`macd.ewm(span=9).mean()`</td></tr>
<tr><td>macd_hist</td><td>float</td><td>Histograma MACD</td><td>`macd - macd_signal`</td></tr>
</table>

### Features macroeconómicas

<table>
<tr><th>Variable</th><th>Tipo</th><th>Descripción</th><th>Fórmula</th></tr>
<tr><td>dxy_return_1d</td><td>float</td><td>Retorno diario del dólar</td><td>`dxy_close.pct_change()`</td></tr>
<tr><td>vix_return_1d</td><td>float</td><td>Cambio diario del VIX</td><td>`vix_close.pct_change()`</td></tr>
<tr><td>tnx_return_1d</td><td>float</td><td>Cambio diario del TNX</td><td>`tnx_close.pct_change()`</td></tr>
<tr><td>dxy_sma_5</td><td>float</td><td>Media móvil DXY 5 días</td><td>`dxy_close.rolling(5).mean()`</td></tr>
<tr><td>dxy_sma_21</td><td>float</td><td>Media móvil DXY 21 días</td><td>`dxy_close.rolling(21).mean()`</td></tr>
<tr><td>vix_sma_5</td><td>float</td><td>Media móvil VIX 5 días</td><td>`vix_close.rolling(5).mean()`</td></tr>
<tr><td>vix_sma_21</td><td>float</td><td>Media móvil VIX 21 días</td><td>`vix_close.rolling(21).mean()`</td></tr>
<tr><td>tnx_sma_5</td><td>float</td><td>Media móvil TNX 5 días</td><td>`tnx_close.rolling(5).mean()`</td></tr>
<tr><td>tnx_sma_21</td><td>float</td><td>Media móvil TNX 21 días</td><td>`tnx_close.rolling(21).mean()`</td></tr>
</table>

### Features de rezago (lags)

<table>
<tr><th>Variable</th><th>Tipo</th><th>Descripción</th><th>Fórmula</th></tr>
<tr><td>gold_lag_1d</td><td>float</td><td>Precio cierre oro día anterior</td><td>`gold_close.shift(1)`</td></tr>
<tr><td>gold_lag_2d</td><td>float</td><td>Precio cierre oro 2 días atrás</td><td>`gold_close.shift(2)`</td></tr>
<tr><td>gold_lag_3d</td><td>float</td><td>Precio cierre oro 3 días atrás</td><td>`gold_close.shift(3)`</td></tr>
<tr><td>gold_lag_5d</td><td>float</td><td>Precio cierre oro 5 días atrás</td><td>`gold_close.shift(5)`</td></tr>
</table>

---

## Target Variables

<table>
<tr><th>Variable</th><th>Tipo</th><th>Valores</th><th>Problema</th><th>Descripción</th></tr>
<tr><td>target_bin</td><td>int</td><td>0 = Down, 1 = Up</td><td>Clasificación binaria</td><td>¿El oro subirá mañana? (dirección)</td></tr>
<tr><td>target_multi</td><td>int</td><td>0 = Baja, 1 = Estable, 2 = Sube</td><td>Clasificación multiclase</td><td>Rangos definidos por threshold 0.5%: &lt; -0.5% | -0.5% a 0.5% | &gt; 0.5%</td></tr>
</table>

---

## Feature Selection (dashboard)

El dashboard usa un subconjunto de features para el modelo en vivo:

`gold_return_1d`, `sma_5`, `sma_10`, `sma_21`, `vol_5d`, `rsi_14`, `macd`, `macd_signal`,
`dxy_return_1d`, `vix_return_1d`, `tnx_return_1d`, `dxy_close`, `vix_close`, `tnx_close`

Escaladas con `StandardScaler`. Modelo: `RandomForestClassifier(n_estimators=200, max_depth=10)`.

---

## Frecuencia de predicción

**Diaria.** El modelo se entrena con datos OHLC diarios (cierre) y predice la dirección del oro para el **siguiente día hábil**. No se hacen predicciones intradiarias ni semanales. Cada fila del dataset representa un día de trading.

