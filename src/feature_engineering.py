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
    """Create 1 and 2-day lagged returns for gold."""
    df = df.copy()

    df["gold_return_lag_1"] = df["gold_return"].shift(1)
    df["gold_return_lag_2"] = df["gold_return"].shift(2)

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
    """Run full feature engineering pipeline in sequence."""
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
