import os
import pandas as pd
import yfinance as yfi

##Descarga histórico diario del oro + 3 indicadores macro
START = "2008-01-01"
INTERVAL = "1d"
OUTPUT = "data/raw/gold-macro-data.csv"
 
TICKERS = {
    "Gold": "GC=F",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "GVZ": "^GVZ",       # Volatilidad implícita del oro (VIX del oro)
    "Oil": "CL=F",       # Futuros del petróleo WTI
    "SP500": "^GSPC", 
}
 
 
def download_ticker(name: str, ticker: str, start: str, interval: str) -> pd.DataFrame:
    print(f"Descargando {name} ({ticker})...")
    df = yfi.download(ticker, start=start, interval=interval, progress=False)
 
    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {name} ({ticker})")
    
    #ponemos al mismo nivel las columnas
    #if isinstance(df.columns, pd.MultiIndex):
    #   df.columns = df.columns.get_level_values(0)
 
    # yfinance puede devolver columnas multi-index si se piden varios tickers,
    # nos quedamos solo con las columnas que nos interesan
    df = df[["Close", "High", "Low", "Open", "Volume"]].copy()
    df.columns = [f"{name}_{col}" for col in df.columns]
    df.index.name = "Date"
 
    print(f"  -> {name}: {df.shape[0]} filas, desde {df.index.min().date()} hasta {df.index.max().date()}")
    return df


def merge_series(dfs: dict) -> pd.DataFrame:
    df_merged = pd.concat(dfs.values(), axis=1, sort=False)
    df_merged = df_merged.ffill()
    df_merged = df_merged.dropna()
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