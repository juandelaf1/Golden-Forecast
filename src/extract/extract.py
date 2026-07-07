import os
import pandas as pd
import yfinance as yfi

# Download daily history for gold + 3 macro indicators
START = "2015-01-01"
INTERVAL = "1d"
OUTPUT = "data/raw/gold-macro-data.csv"

TICKERS = {
    "Gold": "GC=F",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "TNX": "^TNX",
}


def download_ticker(name: str, ticker: str, start: str, interval: str) -> pd.DataFrame:
    print(f"Descargando {name} ({ticker})...")
    df = yfi.download(ticker, start=start, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {name} ({ticker})")

    # Flatten columns; keep only OHLCV
    df = df[["Close", "High", "Low", "Open", "Volume"]].copy()
    df.columns = [f"{name}_{col}" for col in df.columns]
    df.index.name = "Date"

    print(
        f"  -> {name}: {df.shape[0]} filas, desde {df.index.min().date()} hasta {df.index.max().date()}"
    )
    return df


# Merge all series by date
def merge_series(dfs: dict) -> pd.DataFrame:
    df_merged = pd.concat(dfs.values(), axis=1)
    return df_merged


def save_dataset(df: pd.DataFrame, OUTPUT: str) -> None:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT)
    print(f"\nGuardado en: {OUTPUT}")
    print(f"Shape: {df.shape[0]} filas x {df.shape[1]} columnas")


def main():
    dfs = {
        name: download_ticker(name, ticker, START, INTERVAL)
        for name, ticker in TICKERS.items()
    }
    df_merged = merge_series(dfs)
    save_dataset(df_merged, OUTPUT)

    print("\nPrimeras filas:")
    print(df_merged.head())


if __name__ == "__main__":
    main()
