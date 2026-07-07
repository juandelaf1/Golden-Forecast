import pandas as pd


# Absolute price columns dropped at the end.
# The model should use relative variables (returns, ranges, MAs, RSI, MACD, volatility, lags)
# rather than absolute prices.
ABSOLUTE_COLUMNS = [
    "gold_close", "gold_high", "gold_low", "gold_open",
    "dxy_close", "dxy_high", "dxy_low", "dxy_open",
    "vix_close", "vix_high", "vix_low", "vix_open",
    "tnx_close", "tnx_high", "tnx_low", "tnx_open",
    "gold_return_next_day"
]


def create_daily_returns(df):
    """Create daily percentage returns for all assets."""
    df = df.copy()

    df["gold_return"] = df["gold_close"].pct_change()
    df["dxy_return"] = df["dxy_close"].pct_change()
    df["vix_return"] = df["vix_close"].pct_change()
    df["tnx_return"] = df["tnx_close"].pct_change()

    return df


def create_daily_ranges(df):
    """Create daily high-low range as (high - low) / open for each asset."""
    df = df.copy()

    df["gold_daily_range"] = (df["gold_high"] - df["gold_low"]) / df["gold_open"]
    df["dxy_daily_range"] = (df["dxy_high"] - df["dxy_low"]) / df["dxy_open"]
    df["vix_daily_range"] = (df["vix_high"] - df["vix_low"]) / df["vix_open"]
    df["tnx_daily_range"] = (df["tnx_high"] - df["tnx_low"]) / df["tnx_open"]

    return df


def create_open_close_returns(df):
    """Create intraday return between open and close for each asset."""
    df = df.copy()

    df["gold_open_close_return"] = (df["gold_close"] - df["gold_open"]) / df["gold_open"]
    df["dxy_open_close_return"] = (df["dxy_close"] - df["dxy_open"]) / df["dxy_open"]
    df["vix_open_close_return"] = (df["vix_close"] - df["vix_open"]) / df["vix_open"]
    df["tnx_open_close_return"] = (df["tnx_close"] - df["tnx_open"]) / df["tnx_open"]

    return df


def create_moving_averages(df):
    """Create 5 and 20-day moving averages and price distance to them."""
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
    """Create MACD line, signal line, and histogram for gold."""
    df = df.copy()

    ema_12 = df["gold_close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["gold_close"].ewm(span=26, adjust=False).mean()

    df["gold_macd"] = ema_12 - ema_26
    df["gold_macd_signal"] = df["gold_macd"].ewm(span=9, adjust=False).mean()

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

    dxy_return_lag_2:
    Retorno del índice dólar (DXY) de hace dos días.

    Permite que el modelo incorpore información pasada sobre la evolución
    del dólar, ya que sus movimientos pueden influir en el precio del oro
    con cierto retraso.

    vix_return_lag_2:
    Retorno del VIX de hace dos días.

    Introduce información histórica sobre cambios en el nivel de incertidumbre
    del mercado. Esto puede ayudar a capturar posibles efectos retardados del
    sentimiento inversor sobre el comportamiento del oro.

    No hay data leakage porque solo se usa información pasada.
    """
    df = df.copy()

    df["gold_return_lag_1"] = df["gold_return"].shift(1)
    df["gold_return_lag_2"] = df["gold_return"].shift(2)
    df["dxy_return_lag_2"] = df["dxy_return"].shift(2)
    df["vix_return_lag_2"] = df["vix_return"].shift(2)

    return df


def create_cumulative_returns(df):
    """
    Crea retornos acumulados del oro a 5 y 10 días.

    Estas variables miden cuánto ha cambiado el oro en una ventana más amplia,
    no solo de un día a otro.
    """
    df = df.copy()

    df["gold_return_5d"] = df["gold_close"].pct_change(5)
    df["gold_return_10d"] = df["gold_close"].pct_change(10)

    return df

def create_vix_sentiment_features(df):
    """
    Crea variables derivadas del VIX para representar sentimiento de mercado.

    El VIX se suele interpretar como índice del miedo:
    - VIX alto: más incertidumbre.
    - VIX bajo: mercado más tranquilo.
    """
    df = df.copy()

    df["vix_ma_20"] = df["vix_close"].rolling(window=20).mean()
    df["vix_high_fear"] = (df["vix_close"] > 25).astype(int)
    df["vix_extreme_fear"] = (df["vix_close"] > 35).astype(int)

    return df

def create_relative_spreads(df):
    """
    Crea variables que comparan el comportamiento del oro con otros activos.

    Sirven para medir si el oro se está comportando mejor o peor que
    el dólar y la bolsa.
    """
    df = df.copy()

    # Diferencia entre la media móvil corta y larga del oro
    df["gold_ma5_minus_ma20"] = df["gold_ma_5"] - df["gold_ma_20"]

    # Comparación del retorno diario del oro frente al dólar
    df["gold_dxy_spread"] = df["gold_return"] - df["dxy_return"]

    # Esta feature queda pendiente porque en el dataset actual no existe sp500_return.
    # Comparación del retorno diario del oro frente al S&P 500
    # df["gold_sp500_spread"] = df["gold_return"] - df["sp500_return"]

    return df

def create_targets(df, threshold=0.005):
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
    8. Lags del retorno del oro.
    9. Retornos acumulados del oro.
    10. Variables de sentimiento del VIX.
    11. Spreads relativos entre activos.
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
    df = create_cumulative_returns(df)
    df = create_vix_sentiment_features(df)
    df = create_relative_spreads(df)

    return df


def run_feature_engineering(input_path, output_path):
    """Run full feature engineering and save the final dataset.

    Reads clean CSV from preprocessing.py, creates features and targets,
    drops absolute columns and NaN rows, saves result.
    """
    df = pd.read_csv(input_path, parse_dates=["Date"])

    df = create_features(df)
    df = create_targets(df, threshold=0.005)

    df = df.drop(columns=ABSOLUTE_COLUMNS, errors="ignore")
    df = df.dropna()

    df.to_csv(output_path, index=False)

    print(f"Dataset con features guardado en {output_path} | shape: {df.shape}")

    return df


if __name__ == "__main__":
    run_feature_engineering(
        "data/processed/gold-clean.csv",
        "data/processed/gold-features.csv"
    )
