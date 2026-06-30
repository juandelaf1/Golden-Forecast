# Data Dictionary

## Raw Variables (from Yahoo Finance)

| Variable | Type | Description |
|----------|------|-------------|
| Open | float | Opening price of the day |
| High | float | Highest price of the day |
| Low | float | Lowest price of the day |
| Close | float | Closing price of the day |
| Adj Close | float | Adjusted closing price (splits/dividends) |
| Volume | int64 | Number of contracts traded |

## Engineered Features

| Variable | Type | Description | Formula |
|----------|------|-------------|---------|
| Returns | float | Daily percentage change | `Close.pct_change()` |
| Range | float | Daily price range | `High - Low` |
| MA_5 | float | 5-day moving average | `Close.rolling(5).mean()` |
| MA_10 | float | 10-day moving average | `Close.rolling(10).mean()` |
| Volume_MA_5 | float | 5-day volume average | `Volume.rolling(5).mean()` |
| Volatility | float | 10-day rolling volatility | `Returns.rolling(10).std()` |
| Close_MA5_ratio | float | Close / 5-day MA ratio | `Close / MA_5` |

## Target Variables

| Variable | Type | Values | Problem |
|----------|------|--------|---------|
| Target (Class) | int | 0 = Down, 1 = Up | Classification |
| Close (Reg) | float | Continuous | Regression |
