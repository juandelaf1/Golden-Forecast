import pandas as pd


# Columnas absolutas que se eliminan al final.
# La idea es que el modelo no trabaje con precios absolutos, sino con variables relativas:
# retornos, rangos, medias móviles, RSI, MACD, volatilidad y lags.
COLUMNAS_ABSOLUTAS = [
    "gold_close", "gold_high", "gold_low", "gold_open",
    "dxy_close", "dxy_high", "dxy_low", "dxy_open",
    "vix_close", "vix_high", "vix_low", "vix_open",
    "tnx_close", "tnx_high", "tnx_low", "tnx_open",
    "gold_return_next_day"
]


def create_daily_returns(df):
    """
    Crea retornos diarios porcentuales.

    Un retorno diario mide cuánto ha cambiado el precio respecto al día anterior.
    Es más útil que el precio absoluto porque expresa variación relativa.
    """
    df = df.copy()

    df["gold_return"] = df["gold_close"].pct_change()
    df["dxy_return"] = df["dxy_close"].pct_change()
    df["vix_return"] = df["vix_close"].pct_change()
    df["tnx_return"] = df["tnx_close"].pct_change()

    return df


def create_daily_ranges(df):
    """
    Crea el rango diario de cada activo.

    Fórmula:
    (máximo del día - mínimo del día) / apertura del día

    Sirve para medir cuánto se ha movido el activo durante la sesión.
    """
    df = df.copy()

    df["gold_daily_range"] = (df["gold_high"] - df["gold_low"]) / df["gold_open"]
    df["dxy_daily_range"] = (df["dxy_high"] - df["dxy_low"]) / df["dxy_open"]
    df["vix_daily_range"] = (df["vix_high"] - df["vix_low"]) / df["vix_open"]
    df["tnx_daily_range"] = (df["tnx_high"] - df["tnx_low"]) / df["tnx_open"]

    return df


def create_open_close_returns(df):
    """
    Crea el retorno intradía entre apertura y cierre.

    Fórmula:
    (cierre - apertura) / apertura

    Esta variable resume si durante el día el activo terminó subiendo o bajando.
    """
    df = df.copy()

    df["gold_open_close_return"] = (df["gold_close"] - df["gold_open"]) / df["gold_open"]
    df["dxy_open_close_return"] = (df["dxy_close"] - df["dxy_open"]) / df["dxy_open"]
    df["vix_open_close_return"] = (df["vix_close"] - df["vix_open"]) / df["vix_open"]
    df["tnx_open_close_return"] = (df["tnx_close"] - df["tnx_open"]) / df["tnx_open"]

    return df


def create_moving_averages(df):
    """
    Crea medias móviles del oro y distancia del precio a esas medias.

    gold_ma_5:
    Media móvil corta de 5 días.

    gold_ma_20:
    Media móvil más amplia de 20 días.

    gold_close_vs_ma_5 y gold_close_vs_ma_20:
    Miden si el precio actual está por encima o por debajo de su media.
    """
    df = df.copy()

    df["gold_ma_5"] = df["gold_close"].rolling(window=5).mean()
    df["gold_ma_20"] = df["gold_close"].rolling(window=20).mean()

    df["gold_close_vs_ma_5"] = (df["gold_close"] - df["gold_ma_5"]) / df["gold_ma_5"]
    df["gold_close_vs_ma_20"] = (df["gold_close"] - df["gold_ma_20"]) / df["gold_ma_20"]

    return df


def create_rsi(df, window=14):
    """
    Crea el RSI de 14 días del oro.

    RSI significa Relative Strength Index.

    Mide la fuerza reciente de las subidas frente a las bajadas.
    Suele usarse como indicador técnico:
    - RSI alto: el activo ha subido mucho recientemente.
    - RSI bajo: el activo ha bajado mucho recientemente.

    Aquí se calcula solo con información pasada y actual, por lo que no introduce data leakage.
    """
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
    Crea MACD y señal del MACD.

    MACD significa Moving Average Convergence Divergence.

    Fórmula:
    MACD = EMA de 12 días - EMA de 26 días

    Señal:
    Media exponencial de 9 días del MACD.

    Sirve para detectar cambios de tendencia o momentum.
    """
    df = df.copy()

    ema_12 = df["gold_close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["gold_close"].ewm(span=26, adjust=False).mean()

    df["gold_macd"] = ema_12 - ema_26
    df["gold_macd_signal"] = df["gold_macd"].ewm(span=9, adjust=False).mean()

    return df


def create_rolling_volatility(df, window=14):
    """
    Crea volatilidad rolling de 14 días.

    La volatilidad se calcula como la desviación estándar de los retornos recientes.
    Si es alta, significa que el oro se está moviendo mucho.
    Si es baja, significa que el precio está más estable.
    """
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

    return df


def create_targets(df, threshold=0.005):
    """
    Crea las variables objetivo del proyecto.

    gold_return_next_day:
    Retorno del oro del día siguiente.
    Esta columna se usa para construir los targets, pero después se elimina
    para que no entre como variable predictora.

    target_binary:
    - 1 si el oro sube al día siguiente.
    - 0 si el oro no sube.

    target_multiclass:
    - 1 si el oro sube más del threshold.
    - 0 si el movimiento está entre -threshold y +threshold.
    - -1 si el oro baja más del threshold.

    En este caso el threshold es 0.005, es decir, 0.5%.
    """
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
    8. Lags del retorno del oro.
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

    return df


def run_feature_engineering(input_path, output_path):
    """
    Ejecuta el feature engineering completo y guarda el dataset final.

    Lee el CSV limpio generado por preprocessing.py.
    Después:
    1. Crea variables predictoras.
    2. Crea targets.
    3. Elimina columnas absolutas.
    4. Elimina filas con nulos generados por rolling, shift y pct_change.
    5. Guarda el dataset final con features.
    """
    df = pd.read_csv(input_path, parse_dates=["Date"])

    df = create_features(df)
    df = create_targets(df, threshold=0.005)

    df = df.drop(columns=COLUMNAS_ABSOLUTAS, errors="ignore")
    df = df.dropna()

    df.to_csv(output_path, index=False)

    print(f"Dataset con features guardado en {output_path} | shape: {df.shape}")

    return df


if __name__ == "__main__":
    run_feature_engineering(
        "data/processed/gold-clean.csv",
        "data/processed/gold-features.csv"
    )