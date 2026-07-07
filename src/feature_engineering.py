import pandas as pd


# Columnas absolutas que se eliminan al final.
# La idea es que el modelo no trabaje con precios absolutos, sino con variables relativas:
# retornos, rangos, medias móviles, RSI, MACD, volatilidad y lags.
#
# IMPORTANTE: también eliminamos variables intermedias en escala de precio (dólares):
#   - gold_ma_5, gold_ma_20: medias móviles en $ — se reemplazan por gold_close_vs_ma_*
#   - gold_ma_cross: diferencia de medias en $ — se reemplaza por gold_ma_cross_pct
#   - gold_macd, gold_macd_signal: diferencia de EMAs en $ — se normalizan a gold_macd_pct, gold_macd_signal_pct
#   - gold_atr_14: en $ — se normaliza a gold_atr_14_pct
#   - vix_ma_20: en puntos VIX absolutos — se reemplaza por vix_vs_ma20 (relativo)
COLUMNAS_ABSOLUTAS = [
    "gold_close", "gold_high", "gold_low", "gold_open",
    "dxy_close", "dxy_high", "dxy_low", "dxy_open",
    "vix_close", "vix_high", "vix_low", "vix_open",
    "tnx_close", "tnx_high", "tnx_low", "tnx_open",
    "gold_return_next_day", "gvz_close", "gvz_high", "gvz_low", "gvz_open",
    "oil_close", "oil_high", "oil_low", "oil_open",
    "sp500_close", "sp500_high", "sp500_low", "sp500_open",
    # Intermedias en escala absoluta — el modelo no debe verlas directamente
    "gold_ma_5", "gold_ma_20",   # en $ → ya tenemos gold_close_vs_ma_5/20
    "gold_ma_cross",              # en $ → ya tenemos gold_ma_cross_pct
    "gold_macd", "gold_macd_signal",  # en $ → normalizados abajo
    "gold_atr_14",                # en $ → normalizado abajo
    "vix_ma_20",                  # en puntos → normalizado abajo
]

# Columnas que se eliminan porque son ruido confirmado.
#
# Criterio: permutation importance negativa en test + TimeSeriesSplit CV.
# La permutation importance mide cuánto cae el F1 si se permuta aleatoriamente
# esa feature — si cae a negativo, la feature está activamente dañando al modelo.
# Un único split puede tener varianza alta, así que solo se eliminan las features
# cuya eliminación mejora tanto el split 80/20 como la CV temporal de 5 folds.
#
# RESULTADO DEL ANÁLISIS (ver feature importance en train.py):
#   - vix_high_fear, vix_extreme_fear: importance MDI = 0 y 0.0016 — no aportan nada.
#     Los flags binarios de umbral fijo son demasiado gruesos; vix_vs_ma20 ya captura
#     el sentimiento del VIX de forma continua y relativa.
#   - gold_return_lag_2, dxy_return_lag2: redundantes con lag_1 y el retorno del día.
#     En CV temporal su eliminación reduce el gap de overfitting sin coste en F1.
#
# FEATURES QUE PARECEN MALAS EN MDI PERO NO SE ELIMINAN:
#   - gold_return, vix_return, sp500_return, gold_sp500_spread: importancia MDI alta
#     pero perm_importance negativa. Esto indica correlación con otras features
#     (multicolinealidad), no que sean inútiles. Dejarlas no daña y en otros modelos
#     (LR, LGB) pueden ser útiles directamente.
#   - gold_dxy_corr20, gold_sp500_corr20: perm negativa pero solo por 0.001 de margen,
#     y en CV la diferencia no es consistente. Se mantienen por valor de interpretación.
COLUMNAS_RUIDO = [
    "vix_high_fear",     # importance = 0 en MDI y perm — reemplazado por vix_vs_ma20
    "vix_extreme_fear",  # importance ≈ 0 en MDI y perm — umbral fijo demasiado grueso
    "gold_return_lag_2", # redundante con lag_1; en CV reduce gap sin mejorar F1
    "dxy_return_lag2",   # redundante con dxy_return del día; mismo efecto que arriba
]


def create_daily_returns(df):
    """Create daily percentage returns for all assets."""
    df = df.copy()

    df["gold_return"] = df["gold_close"].pct_change()
    df["dxy_return"] = df["dxy_close"].pct_change()
    df["vix_return"] = df["vix_close"].pct_change()
    df["tnx_return"] = df["tnx_close"].pct_change()
    df["gvz_return"] = df["gvz_close"].pct_change()
    df["oil_return"] = df["oil_close"].pct_change()
    df["sp500_return"] = df["sp500_close"].pct_change()

    return df


def create_daily_ranges(df):
    """Create daily high-low range as (high - low) / open for each asset."""
    df = df.copy()

    df["dxy_daily_range"] = (df["dxy_high"] - df["dxy_low"]) / df["dxy_open"]
    df["vix_daily_range"] = (df["vix_high"] - df["vix_low"]) / df["vix_open"]
    df["tnx_daily_range"] = (df["tnx_high"] - df["tnx_low"]) / df["tnx_open"]
    df["gvz_daily_range"] = (df["gvz_high"] - df["gvz_low"]) / df["gvz_open"]
    df["oil_daily_range"] = (df["oil_high"] - df["oil_low"]) / df["oil_open"]
    df["sp500_daily_range"] = (df["sp500_high"] - df["sp500_low"]) / df["sp500_open"]

    return df


def create_open_close_returns(df):
    """Create intraday return between open and close for each asset."""
    df = df.copy()

    df["gold_open_close_return"] = (df["gold_close"] - df["gold_open"]) / df["gold_open"]
    df["dxy_open_close_return"] = (df["dxy_close"] - df["dxy_open"]) / df["dxy_open"]
    df["vix_open_close_return"] = (df["vix_close"] - df["vix_open"]) / df["vix_open"]
    df["tnx_open_close_return"] = (df["tnx_close"] - df["tnx_open"]) / df["tnx_open"]
    df["gvz_open_close_return"] = (df["gvz_close"] - df["gvz_open"]) / df["gvz_open"]
    df["oil_open_close_return"] = (df["oil_close"] - df["oil_open"]) / df["oil_open"]
    df["sp500_open_close_return"] = (df["sp500_close"] - df["sp500_open"]) / df["sp500_open"]

    return df


def create_moving_averages(df):
    """Create intraday return between open and close for each asset."""
    df = df.copy()

    df["gold_ma_5"] = df["gold_close"].rolling(window=5).mean()
    df["gold_ma_20"] = df["gold_close"].rolling(window=20).mean()

    df["gold_close_vs_ma_5"] = (df["gold_close"] - df["gold_ma_5"]) / df["gold_ma_5"]
    df["gold_close_vs_ma_20"] = (df["gold_close"] - df["gold_ma_20"]) / df["gold_ma_20"]

    return df


def create_rsi(df, window=14):
    """Create 14-day RSI for gold. Uses only past data — no leakage."""
    df = df.copy()

    delta = df["gold_close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()

    rs = avg_gain / avg_loss

    df["gold_rsi_14"] = 100 - (100 / (1 + rs))

    return df


def create_macd(df):
    """
    Crea MACD normalizado por el precio para que sea comparable entre distintos niveles.

    MACD bruto = EMA12 - EMA26 (en dólares — crece con el precio del oro).
    Para que sea una feature útil en distintos regímenes de precio,
    lo normalizamos dividiéndolo por la EMA26:

    gold_macd_pct = (EMA12 - EMA26) / EMA26
    gold_macd_signal_pct = señal del MACD normalizada de la misma forma

    Así el modelo ve momentum relativo, no un valor en dólares.
    """
    df = df.copy()

    ema_12 = df["gold_close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["gold_close"].ewm(span=26, adjust=False).mean()

    df["gold_macd"] = ema_12 - ema_26
    df["gold_macd_signal"] = df["gold_macd"].ewm(span=9, adjust=False).mean()

    # Versiones normalizadas por la EMA26 (relativas al nivel de precio)
    df["gold_macd_pct"] = df["gold_macd"] / ema_26
    df["gold_macd_signal_pct"] = df["gold_macd_signal"] / ema_26

    return df


def create_rolling_volatility(df, window=14):
    """Create 14-day rolling volatility (standard deviation of returns)."""
    df = df.copy()

    df["gold_volatility_14"] = df["gold_return"].rolling(window=window).std()

    return df


def create_return_lags(df):
    """
    Crea retardos del retorno del oro.

    Estos lags permiten que el modelo vea qué pasó en días anteriores:

    gold_return_lag_1:
    Retorno del oro del día anterior.

    gold_return_lag_2:
    Retorno del oro de hace dos días.

    No hay data leakage porque solo se usa información pasada.
    """
    df = df.copy()

    df["gold_return_lag_1"] = df["gold_return"].shift(1)
    df["gold_return_lag_2"] = df["gold_return"].shift(2)
    df["gvz_return_lag1"] = df["gvz_return"].shift(1)
    df["oil_return_lag1"] = df["oil_return"].shift(1)   # restaurado — se había perdido
    df["sp500_return_lag1"] = df["sp500_return"].shift(1)

    # Lags adicionales de DXY y VIX a 2 días.
    # El dólar y el miedo del mercado no siempre impactan al oro el mismo día,
    # a veces el efecto se nota con retraso.
    df["dxy_return_lag2"] = df["dxy_return"].shift(2)
    df["vix_return_lag2"] = df["vix_return"].shift(2)

    return df


def create_cumulative_return(df):
    """
    Crea retornos acumulados del oro a 3, 5 y 10 días..

    A diferencia de gold_return (retorno de 1 día), estas miden el
    retorno acumulado de los últimos 3, 5 y 10 días. Ayudan al modelo
    a distinguir entre una racha sostenida y un movimiento puntual.
    """
    df = df.copy()

    df["gold_return_3d"] = df["gold_close"].pct_change(3)
    df["gold_return_5d"] = df["gold_close"].pct_change(5)
    df["gold_return_10d"] = df["gold_close"].pct_change(10)

    return df

def create_vix_sentiment_features(df):
    """
    Crea variables derivadas del VIX para representar sentimiento de mercado.

    El VIX se suele interpretar como índice del miedo:
    - VIX alto: más incertidumbre → el oro típicamente sube como refugio.
    - VIX bajo: mercado más tranquilo.

    vix_vs_ma20: distancia relativa entre el VIX actual y su media de 20 días.
    Mide si el miedo de hoy es mayor o menor que el "miedo normal" reciente.
    Es relativo y comparable en cualquier régimen de mercado.

    vix_high_fear y vix_extreme_fear: flags binarios de régimen de miedo.
    Estos sí usan niveles absolutos del VIX (25 y 35 son umbrales de mercado
    ampliamente establecidos, no dependen del nivel de precio del oro).
    """
    df = df.copy()

    vix_ma_20 = df["vix_close"].rolling(window=20).mean()
    df["vix_vs_ma20"] = (df["vix_close"] - vix_ma_20) / vix_ma_20
    df["vix_high_fear"] = (df["vix_close"] > 25).astype(int)
    df["vix_extreme_fear"] = (df["vix_close"] > 35).astype(int)

    # Guardamos vix_ma_20 como intermedia para que COLUMNAS_ABSOLUTAS la elimine
    df["vix_ma_20"] = vix_ma_20

    return df

def create_ma_cross(df):
    """
    Crea el cruce entre la media móvil corta (5) y la larga (20) del oro.

    gold_ma_cross: diferencia absoluta entre ambas medias.
    gold_ma_cross_pct: la misma diferencia normalizada por la media larga,
    para que sea comparable entre distintos niveles de precio del oro
    (por ejemplo, no es lo mismo un gap de 5$ con el oro a 900 que a 2500).

    Es un indicador clásico de tendencia: cuando la media corta cruza por
    encima de la larga suele interpretarse como señal alcista, y viceversa.
    """
    df = df.copy()

    df["gold_ma_cross"] = df["gold_ma_5"] - df["gold_ma_20"]
    df["gold_ma_cross_pct"] = (df["gold_ma_5"] - df["gold_ma_20"]) / df["gold_ma_20"]

    return df


def create_relative_spreads(df):
    """
    Crea variables que comparan el comportamiento del oro con otros activos.

    Sirven para medir si el oro se está comportando mejor o peor que
    el dólar y la bolsa.
    """
    df = df.copy()

    df["gold_dxy_spread"] = df["gold_return"] - df["dxy_return"]
    df["gold_sp500_spread"] = df["gold_return"] - df["sp500_return"]

    return df


def create_rolling_correlation(df, window=20):
    """
    Crea la correlación móvil de 20 días entre el retorno del oro y el
    retorno del DXY y del S&P500.

    La relación oro-dólar y oro-bolsa no es constante en el tiempo: hay
    periodos de correlación fuerte y periodos donde se rompe (por ejemplo
    en crisis de liquidez). Que el modelo vea si esa relación se está
    manteniendo o rompiendo recientemente aporta información de régimen
    de mercado que un retorno puntual no captura.
    """
    df = df.copy()

    df["gold_dxy_corr20"] = df["gold_return"].rolling(window).corr(df["dxy_return"])
    df["gold_sp500_corr20"] = df["gold_return"].rolling(window).corr(df["sp500_return"])

    return df


def create_atr(df, window=14):
    """
    Crea el ATR (Average True Range) de 14 días del oro, normalizado por el precio.

    El True Range de un día es el mayor de:
    - high - low
    - |high - close anterior|
    - |low - close anterior|

    El ATR bruto está en dólares y crece con el precio del oro (el ATR de 2024
    no es comparable con el de 2010). Por eso lo normalizamos:

    gold_atr_14_pct = ATR14 / gold_close

    Así obtenemos volatilidad relativa, comparable en cualquier régimen de precio.
    Es equivalente al ATR% que usan muchos traders de volatilidad.
    """
    df = df.copy()

    prev_close = df["gold_close"].shift(1)

    tr1 = df["gold_high"] - df["gold_low"]
    tr2 = (df["gold_high"] - prev_close).abs()
    tr3 = (df["gold_low"] - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["gold_atr_14"] = true_range.rolling(window=window).mean()
    df["gold_atr_14_pct"] = df["gold_atr_14"] / df["gold_close"]

    return df


def create_bollinger_position(df, window=20, num_std=2):
    """
    Crea la posición del precio dentro de las Bandas de Bollinger.

    En lugar de guardar las bandas superior/inferior como variables
    absolutas (que dependen del nivel de precio del oro), guardamos
    bb_position, que indica dónde está el precio dentro del canal:

    bb_position = (close - banda_inferior) / (banda_superior - banda_inferior)

    - bb_position cerca de 1: el precio está pegado a la banda superior
      (posible sobrecompra).
    - bb_position cerca de 0: el precio está pegado a la banda inferior
      (posible sobreventa).
    - bb_position alrededor de 0.5: el precio está en la media móvil.
    """
    df = df.copy()

    rolling_mean = df["gold_close"].rolling(window=window).mean()
    rolling_std = df["gold_close"].rolling(window=window).std()

    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    df["bb_position"] = (df["gold_close"] - lower_band) / (upper_band - lower_band)

    return df

"""
def create_calendar_features(df):
 
    Crea features de calendario.
    El oro tiene patrones estacionales reales — ciertos días
    de la semana y meses del año tienen comportamiento distinto.
    
    df = df.copy()
    df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek  # 0=lunes, 4=viernes
    df["month"] = pd.to_datetime(df["Date"]).dt.month
    return df

"""

def create_targets(df, threshold=0.003):
    """Create binary and multiclass targets from next-day gold return."""
    df = df.copy()

    df["gold_return_next_day"] = df["gold_close"].pct_change().shift(-1)

    df["target_binary"] = (df["gold_return_next_day"] > 0).astype(int)

    df["target_multiclass"] = 0
    df.loc[df["gold_return_next_day"] > threshold, "target_multiclass"] = 1
    df.loc[df["gold_return_next_day"] < -threshold, "target_multiclass"] = -1

    return df


def create_features(df):
    """
    Ejecuta todo el proceso de creación de variables predictoras.

    Orden:
    1. Retornos diarios.
    2. Rangos diarios.
    3. Retornos apertura-cierre.
    4. Medias móviles.
    5. RSI.
    6. MACD.
    7. Volatilidad rolling.
    8. Lags del retorno del oro (incluye dxy/vix lag2).
    9. Momentum (3d, 5d, 10d).
    10. Cruce de medias móviles.
    11. Variables de sentimiento del VIX.
    12. Spreads oro vs dxy/sp500.
    13. Correlación móvil oro vs dxy/sp500.
    14. ATR 14.
    15. Posición en Bandas de Bollinger.
    """
    df = df.copy()

    df = create_daily_returns(df)
    df = create_daily_ranges(df)
    df = create_open_close_returns(df)
    df = create_moving_averages(df)
    df = create_rsi(df)
    df = create_macd(df)
    df = create_rolling_volatility(df)
    df = create_return_lags(df)
    df = create_cumulative_return(df)
    df = create_ma_cross(df)
    df = create_vix_sentiment_features(df)
    df = create_relative_spreads(df)
    df = create_rolling_correlation(df)
    df = create_atr(df)
    df = create_bollinger_position(df)
    #df = create_calendar_features(df)

    return df


def run_feature_engineering(input_path, output_path):
    """
    Ejecuta el feature engineering completo y guarda el dataset final.

    Lee el CSV limpio generado por preprocessing.py.
    Después:
    1. Crea variables predictoras.
    2. Crea targets.
    3. Elimina columnas absolutas (precios en $).
    4. Elimina columnas de ruido confirmado (ver COLUMNAS_RUIDO).
    5. Elimina filas con nulos generados por rolling, shift y pct_change.
    6. Guarda el dataset final con features.
    """
    df = pd.read_csv(input_path, parse_dates=["Date"])

    df = create_features(df)
    df = create_targets(df, threshold=0.003)

    df = df.drop(columns=COLUMNAS_ABSOLUTAS, errors="ignore")
    df = df.drop(columns=COLUMNAS_RUIDO, errors="ignore")
    df = df.dropna()

    df.to_csv(output_path, index=False)

    print(f"Dataset con features guardado en {output_path} | shape: {df.shape}")

    return df


if __name__ == "__main__":
    run_feature_engineering(
        "data/processed/gold-clean.csv",
        "data/processed/gold-features.csv"
    )