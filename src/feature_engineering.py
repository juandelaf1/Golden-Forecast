import pandas as pd

# Columnas absolutas que se eliminan al final.
# La idea es que el modelo no trabaje con precios absolutos, sino con variables relativas:
# retornos, rangos, medias móviles, RSI, MACD, volatilidad y lags.
#
# IMPORTANTE:
# - gold_return_next_day se elimina porque contiene información futura
#   y solo debe existir como variable intermedia para crear targets.
# - gold_return NO se elimina aquí porque se usa después para backtest y evaluación.
#   Aun así, train.py y evaluate.py deben excluirla explícitamente de X.
#
# También eliminamos variables intermedias en escala de precio (dólares):
# - gold_ma_5, gold_ma_20: medias móviles en $ — se reemplazan por gold_close_vs_ma_*
# - gold_ma_cross: diferencia de medias en $ — se reemplaza por gold_ma_cross_pct
# - gold_macd, gold_macd_signal: diferencia de EMAs en $ — se normalizan a gold_macd_pct, gold_macd_signal_pct
# - gold_atr_14: en $ — se normaliza a gold_atr_14_pct
# - vix_ma_20: en puntos VIX absolutos — se reemplaza por vix_vs_ma20 (relativo)
COLUMNAS_ABSOLUTAS = [
    "gold_close",
    "gold_high",
    "gold_low",
    "gold_open",
    "dxy_close",
    "dxy_high",
    "dxy_low",
    "dxy_open",
    "vix_close",
    "vix_high",
    "vix_low",
    "vix_open",
    "tnx_close",
    "tnx_high",
    "tnx_low",
    "tnx_open",
    "gold_return_next_day",
    "gvz_close",
    "gvz_high",
    "gvz_low",
    "gvz_open",
    "oil_close",
    "oil_high",
    "oil_low",
    "oil_open",
    "sp500_close",
    "sp500_high",
    "sp500_low",
    "sp500_open",
    "gold_ma_5",
    "gold_ma_20",
    "gold_ma_cross",
    "gold_macd",
    "gold_macd_signal",
    "gold_atr_14",
    "vix_ma_20",
]


# Columnas que se eliminan porque son ruido confirmado.
#
# Criterio: permutation importance negativa en test + TimeSeriesSplit CV.
# La permutation importance mide cuánto cae el F1 si se permuta aleatoriamente
# esa feature; si cae a negativo, la feature está activamente dañando al modelo.
# Un único split puede tener varianza alta, así que solo se eliminan las features
# cuya eliminación mejora tanto el split 80/20 como la CV temporal de 5 folds.
#
# RESULTADO DEL ANÁLISIS:
# - vix_high_fear, vix_extreme_fear: importance MDI casi nula; no aportan valor real.
# - gold_return_lag_2, dxy_return_lag2: redundantes con variables cercanas.
COLUMNAS_RUIDO = [
    "vix_high_fear",
    "vix_extreme_fear",
    "gold_return_lag_2",
    "dxy_return_lag2",
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
    """Create moving averages and relative distance to them."""
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
    """Create MACD normalized by price level."""
    df = df.copy()

    ema_12 = df["gold_close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["gold_close"].ewm(span=26, adjust=False).mean()

    df["gold_macd"] = ema_12 - ema_26
    df["gold_macd_signal"] = df["gold_macd"].ewm(span=9, adjust=False).mean()

    df["gold_macd_pct"] = df["gold_macd"] / ema_26
    df["gold_macd_signal_pct"] = df["gold_macd_signal"] / ema_26

    return df


def create_rolling_volatility(df, window=14):
    """Create 14-day rolling volatility (standard deviation of returns)."""
    df = df.copy()

    df["gold_volatility_14"] = df["gold_return"].rolling(window=window).std()

    return df


def create_return_lags(df):
    """Create lagged return features."""
    df = df.copy()

    df["gold_return_lag_1"] = df["gold_return"].shift(1)
    df["gold_return_lag_2"] = df["gold_return"].shift(2)
    df["gvz_return_lag1"] = df["gvz_return"].shift(1)
    df["oil_return_lag1"] = df["oil_return"].shift(1)
    df["sp500_return_lag1"] = df["sp500_return"].shift(1)

    df["dxy_return_lag2"] = df["dxy_return"].shift(2)
    df["vix_return_lag2"] = df["vix_return"].shift(2)

    return df


def create_cumulative_return(df):
    """Create cumulative returns over 3, 5 and 10 days."""
    df = df.copy()

    df["gold_return_3d"] = df["gold_close"].pct_change(3)
    df["gold_return_5d"] = df["gold_close"].pct_change(5)
    df["gold_return_10d"] = df["gold_close"].pct_change(10)

    return df


def create_vix_sentiment_features(df):
    """Create VIX sentiment-related features."""
    df = df.copy()

    vix_ma_20 = df["vix_close"].rolling(window=20).mean()
    df["vix_vs_ma20"] = (df["vix_close"] - vix_ma_20) / vix_ma_20
    df["vix_high_fear"] = (df["vix_close"] > 25).astype(int)
    df["vix_extreme_fear"] = (df["vix_close"] > 35).astype(int)
    df["vix_ma_20"] = vix_ma_20

    return df


def create_ma_cross(df):
    """Create moving-average cross features."""
    df = df.copy()

    df["gold_ma_cross"] = df["gold_ma_5"] - df["gold_ma_20"]
    df["gold_ma_cross_pct"] = (df["gold_ma_5"] - df["gold_ma_20"]) / df["gold_ma_20"]

    return df


def create_relative_spreads(df):
    """Create relative spreads between gold and other assets."""
    df = df.copy()

    df["gold_dxy_spread"] = df["gold_return"] - df["dxy_return"]
    df["gold_sp500_spread"] = df["gold_return"] - df["sp500_return"]

    return df


def create_rolling_correlation(df, window=20):
    """Create rolling correlations between gold and DXY/SP500 returns."""
    df = df.copy()

    df["gold_dxy_corr20"] = df["gold_return"].rolling(window).corr(df["dxy_return"])
    df["gold_sp500_corr20"] = df["gold_return"].rolling(window).corr(df["sp500_return"])

    return df


def create_atr(df, window=14):
    """Create ATR normalized by price."""
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
    """Create price position inside Bollinger Bands."""
    df = df.copy()

    rolling_mean = df["gold_close"].rolling(window=window).mean()
    rolling_std = df["gold_close"].rolling(window=window).std()

    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    df["bb_position"] = (df["gold_close"] - lower_band) / (upper_band - lower_band)

    return df


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
    """Run complete feature engineering pipeline."""
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

    return df


def run_feature_engineering(input_path, output_path):
    """Execute feature engineering and save final dataset."""
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
    run_feature_engineering("data/processed/gold-clean.csv", "data/processed/gold-features.csv")
