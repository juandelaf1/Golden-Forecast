# ML Report — Golden Forecast

## 1. Dataset

- **Source**: Yahoo Finance (GC=F, DX-Y.NYB, ^VIX, ^TNX)
- **Period**: 2015-01-01 to present (~2,865 daily records)
- **Split**: Temporal (80/20), chronological order preserved
- **Preprocessing**: Missing values removed, no forward-fill
- **Features**: 33 engineered variables + 2 targets (see PR #39 for market sentiment additions)

## 2. Features (33 engineered variables)

- **Price/Returns** (5): gold_return, dxy_return, vix_return, tnx_return, gold_volume
- **Daily Range** (4): gold/dxy/vix/tnx_daily_range
- **Intraday** (4): gold/dxy/vix/tnx_open_close_return
- **Moving Averages** (4): gold_ma_5, gold_ma_20, gold_close_vs_ma_5/20
- **Technical** (4): gold_rsi_14, gold_macd, gold_macd_signal, gold_volatility_14
- **Lags** (2): gold_return_lag_1, gold_return_lag_2
- **Cumulative Returns** (2): gold_return_5d, gold_return_10d
- **VIX Sentiment** (3): vix_ma_20, vix_high_fear, vix_extreme_fear
- **Relative Spreads** (2): gold_ma5_minus_ma20, gold_dxy_spread
- **Lags (macro)** (2): dxy_return_lag_2, vix_return_lag_2
- **Total**: **33 features + 2 targets**

## 3. Models

### Classification (6 model configs, 2 targets → 12 models)

- **Logistic Regression**: lr (C=1.0), lr_strong_reg (C=0.1)
- **Random Forest**: rf (depth=5, 100 trees), rf_deep (depth=10, 200 trees)
- **XGBoost**: xgb (depth=3, 100 est.), xgb_deep (depth=5, 200 est.)

**Targets**:
- `target_binary`: 1 = price up, 0 = price down (threshold 0%)
- `target_multiclass`: 1 = buy (&gt;+1%), 0 = hold, -1 = sell (&lt;-1%)

**Best Model**: `lr_strong_reg_binary` — Logistic Regression (C=0.1), binary target
- Accuracy: 56%
- F1 Score: 0.70
- ROC-AUC: 0.72

### Regression (4 targets, 7 models each)

- **fair_value_dist** → Random Forest: R²=0.52, IC=0.77
- **realized_vol_20d** → Ridge: R²=0.15, IC=0.54
- **max_drawdown_20d** → Ridge: R²=0.28, IC=0.55
- **future_atr_20d** → Lasso: R²=-0.05, IC=0.50

## 4. Validation Strategy

- **Temporal split**: First 80% for training, last 20% for testing
- **Scaler**: StandardScaler fit on training data only (saved to disk)
- **Backtesting**: Strategy signals vs Buy & Hold on test set
- **TimeSeriesSplit**: Used in regression with gap=20 to purge rolling window overlap
- **Overfitting check**: Train vs test metrics compared per model

## 5. Key Results

- **Strategy outperforms Buy & Hold** by ~5% on test period
- **LR with strong regularization** achieves best balance of bias/variance
- **fair_value_dist** has strongest predictive signal (R²=0.52, IC=0.77)
- **Volatility/ATR targets** show weaker linear signal (exploratory)

## 6. Limitations

- Daily data only — not suitable for high-frequency trading
- Past performance does not guarantee future results
- Regression R² values indicate limited predictive power for risk targets
- No live trading validation — backtest only
