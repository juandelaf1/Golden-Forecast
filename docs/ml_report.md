# ML Report — Golden Forecast

## 1. Dataset

- **Source**: Yahoo Finance (GC=F, DX-Y.NYB, ^VIX, ^TNX)
- **Period**: 2015-01-01 to present (~2,900 daily records)
- **Split**: Temporal (80/20), chronological order preserved
- **Preprocessing**: Missing values removed (10 rows), no forward-fill

## 2. Features (23 engineered variables)

| Category | Features | Count |
|----------|----------|-------|
| Price/Returns | gold_return, dxy_return, vix_return, tnx_return, gold_volume | 5 |
| Daily Range | gold/dxy/vix/tnx_daily_range | 4 |
| Intraday | gold/dxy/vix/tnx_open_close_return | 4 |
| Moving Averages | gold_ma_5, gold_ma_20, gold_close_vs_ma_5/20 | 4 |
| Technical | gold_rsi_14, gold_macd, gold_macd_signal, gold_volatility_14 | 4 |
| Lags | gold_return_lag_1, gold_return_lag_2 | 2 |
| **Total** | | **23** |

## 3. Models

### Classification (6 model configs, 2 targets → 12 models)

| Algorithm | Configs | Key Parameters |
|-----------|---------|----------------|
| Logistic Regression | lr, lr_strong_reg | C=1.0, C=0.1 |
| Random Forest | rf, rf_deep | depth=5 (100 trees), depth=10 (200 trees) |
| XGBoost | xgb, xgb_deep | depth=3 (100 est.), depth=5 (200 est.) |

**Targets**:
- `target_binary`: 1 = price up, 0 = price down (threshold 0%)
- `target_multiclass`: 1 = buy (>+1%), 0 = hold, -1 = sell (<-1%)

**Best Model**: `lr_strong_reg_binary` — Logistic Regression (C=0.1), binary target
- Accuracy: 56%
- F1 Score: 0.70
- ROC-AUC: 0.72

### Regression (4 targets, 7 models each)

| Target | Best Model | R² | IC (Pearson) |
|--------|-----------|-----|-------------|
| fair_value_dist | Random Forest | 0.52 | 0.77 |
| realized_vol_20d | Ridge | 0.15 | 0.54 |
| max_drawdown_20d | Ridge | 0.28 | 0.55 |
| future_atr_20d | Lasso | -0.05 | 0.50 |

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
