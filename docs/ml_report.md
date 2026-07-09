# ML Report — Golden Forecast

## 1. Dataset

- **Source**: Yahoo Finance (GC=F, DX-Y.NYB, ^VIX, ^TNX)
- **Period**: 2015-01-01 to present (~2,865 daily records)
- **Split**: Temporal (80/20), chronological order preserved
- **Preprocessing**: Missing values removed, no forward-fill
- **Features**: 33 engineered variables + 2 targets (see PR #39 for market sentiment additions)

## 2. Features (23 engineered variables)

<table>
<tr><th>Category</th><th>Features</th><th>Count</th></tr>
<tr><td>Price/Returns</td><td>gold_return, dxy_return, vix_return, tnx_return, gold_volume</td><td>5</td></tr>
<tr><td>Daily Range</td><td>gold/dxy/vix/tnx_daily_range</td><td>4</td></tr>
<tr><td>Intraday</td><td>gold/dxy/vix/tnx_open_close_return</td><td>4</td></tr>
<tr><td>Moving Averages</td><td>gold_ma_5, gold_ma_20, gold_close_vs_ma_5/20</td><td>4</td></tr>
<tr><td>Technical</td><td>gold_rsi_14, gold_macd, gold_macd_signal, gold_volatility_14</td><td>4</td></tr>
<tr><td>Lags</td><td>gold_return_lag_1, gold_return_lag_2</td><td>2</td></tr>
<tr><td>Cumulative Returns</td><td>gold_return_5d, gold_return_10d</td><td>2</td></tr>
<tr><td>VIX Sentiment</td><td>vix_ma_20, vix_high_fear, vix_extreme_fear</td><td>3</td></tr>
<tr><td>Relative Spreads</td><td>gold_ma5_minus_ma20, gold_dxy_spread</td><td>2</td></tr>
<tr><td>Lags (macro)</td><td>dxy_return_lag_2, vix_return_lag_2</td><td>2</td></tr>
<tr><td><strong>Total</strong></td><td></td><td><strong>33 features + 2 targets</strong></td></tr>
</table>

## 3. Models

### Classification (6 model configs, 2 targets → 12 models)

<table>
<tr><th>Algorithm</th><th>Configs</th><th>Key Parameters</th></tr>
<tr><td>Logistic Regression</td><td>lr, lr_strong_reg</td><td>C=1.0, C=0.1</td></tr>
<tr><td>Random Forest</td><td>rf, rf_deep</td><td>depth=5 (100 trees), depth=10 (200 trees)</td></tr>
<tr><td>XGBoost</td><td>xgb, xgb_deep</td><td>depth=3 (100 est.), depth=5 (200 est.)</td></tr>
</table>

**Targets**:
- `target_binary`: 1 = price up, 0 = price down (threshold 0%)
- `target_multiclass`: 1 = buy (>+1%), 0 = hold, -1 = sell (<-1%)

**Best Model**: `lr_strong_reg_binary` — Logistic Regression (C=0.1), binary target
- Accuracy: 56%
- F1 Score: 0.70
- ROC-AUC: 0.72

### Regression (4 targets, 7 models each)

<table>
<tr><th>Target</th><th>Best Model</th><th>R²</th><th>IC (Pearson)</th></tr>
<tr><td>fair_value_dist</td><td>Random Forest</td><td>0.52</td><td>0.77</td></tr>
<tr><td>realized_vol_20d</td><td>Ridge</td><td>0.15</td><td>0.54</td></tr>
<tr><td>max_drawdown_20d</td><td>Ridge</td><td>0.28</td><td>0.55</td></tr>
<tr><td>future_atr_20d</td><td>Lasso</td><td>-0.05</td><td>0.50</td></tr>
</table>

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
