import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'gold-macro-data.csv')
TICKERS = {'gold': 'GC=F', 'dxy': 'DX-Y.NYB', 'vix': '^VIX', 'tnx': '^TNX'}

FEATURE_COLUMNS = [
    'returns',
    'ma_5',
    'ma_10',
    'ma_21',
    'volatility_5',
    'rsi',
    'macd',
    'macd_signal',
    'dxy_return',
    'vix_return',
    'tnx_return',
    'dxy',
    'vix',
    'tnx',
]


def _normalize_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        c.replace("('", '_').replace("', '", '_').replace("')", '')
        for c in df.columns
    ]
    return df


def load_yfinance_data() -> pd.DataFrame:
    raw = yf.download(
        list(TICKERS.values()),
        start='2015-01-01',
        interval='1d',
        progress=False,
        group_by='ticker',
    )
    if raw.empty:
        raise RuntimeError('yfinance returned an empty dataset')

    values = {}
    for alias, ticker in TICKERS.items():
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.levels[0]:
                raise RuntimeError(f'Ticker {ticker} not present in yfinance data')
            values[alias] = raw[ticker]['Close'].copy()
        else:
            if ticker not in raw.columns:
                raise RuntimeError(f'Ticker {ticker} not present in yfinance data')
            values[alias] = raw[ticker].copy()

    df = pd.DataFrame(values)
    df = df.sort_index()
    return df


def load_local_csv() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f'CSV local no encontrado en {DATA_PATH}')

    df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    df = _normalize_csv_columns(df)
    close_cols = [c for c in df.columns if 'Close' in c]
    if len(close_cols) < 4:
        raise RuntimeError('No se encontraron al menos 4 columnas de precio en el CSV local.')

    df = df[close_cols].copy()
    df.columns = ['gold', 'dxy', 'vix', 'tnx']
    df = df.sort_index()
    return df


def load_market_data() -> tuple[pd.DataFrame, str]:
    # Usar solo CSV local para arranque rápido
    data = load_local_csv()
    source = 'local_csv'

    data = data.dropna()
    if data.empty:
        raise RuntimeError('No hay datos válidos luego de la limpieza.')
    return data, source


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result['returns'] = result['gold'].pct_change()
    result['ma_5'] = result['gold'].rolling(5).mean()
    result['ma_10'] = result['gold'].rolling(10).mean()
    result['ma_21'] = result['gold'].rolling(21).mean()
    result['volatility_5'] = result['returns'].rolling(5).std()

    delta = result['gold'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    result['rsi'] = 100 - (100 / (1 + rs))

    ema_12 = result['gold'].ewm(span=12, adjust=False).mean()
    ema_26 = result['gold'].ewm(span=26, adjust=False).mean()
    result['macd'] = ema_12 - ema_26
    result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()

    result['dxy_return'] = result['dxy'].pct_change()
    result['vix_return'] = result['vix'].pct_change()
    result['tnx_return'] = result['tnx'].pct_change()

    result['target_bin'] = (result['gold'].shift(-1) > result['gold']).astype(int)
    result['target_multi'] = np.digitize(result['gold'].pct_change(-1) * 100, bins=[-1, 1])
    result = result.dropna()
    return result


def train_models(data: pd.DataFrame) -> dict:
    """Train multiple models and return all results."""
    split_index = int(len(data) * 0.8)
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLUMNS])
    X_test = scaler.transform(test[FEATURE_COLUMNS])
    y_train = train['target_bin']
    y_test = test['target_bin']

    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # Random Forest (main model)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # XGBoost
    if XGB_AVAILABLE:
        xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb

    # Evaluate all models
    cum_bh = np.cumprod(1 + test['returns'].values) - 1
    model_results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else np.full_like(preds, 0.5)
        acc = (preds == y_test.values).mean()
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probas) if len(np.unique(probas)) > 1 else 0.5
        strat_ret = test['returns'].values * preds
        cum_strat = np.cumprod(1 + strat_ret) - 1
        model_results[name] = {
            'model': model,
            'accuracy': acc,
            'f1': f1,
            'auc': auc,
            'predictions': preds,
            'probabilities': probas,
            'cum_strategy': cum_strat,
            'cum_bh': cum_bh,
        }

    # Use Random Forest as primary model for signals
    primary_model = models['Random Forest']
    X_full = scaler.transform(data[FEATURE_COLUMNS])
    data['signal'] = primary_model.predict(X_full)
    data['signal_proba'] = primary_model.predict_proba(X_full)[:, 1]

    # Feature importance (from Random Forest)
    fi = rf.feature_importances_
    fi_indices = np.argsort(fi)[::-1]
    feature_importance = {
        'features': [FEATURE_COLUMNS[i] for i in fi_indices],
        'importance': fi[fi_indices].tolist(),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, model_results['Random Forest']['predictions'])

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, model_results['Random Forest']['probabilities'])

    # Last 20 days performance
    last20_pred = model_results['Random Forest']['predictions'][-20:]
    last20_actual = y_test.values[-20:]
    last20_acc = (last20_pred == last20_actual).mean()
    last20_hits = int(last20_pred.sum())

    # Natural language summary
    latest_signal = int(data['signal'].iloc[-1])
    latest_proba = float(data['signal_proba'].iloc[-1])
    signal_text = 'SUBIRÁ' if latest_signal == 1 else 'BAJARÁ'

    if model_results['Random Forest']['cum_strategy'][-1] > model_results['Random Forest']['cum_bh'][-1]:
        vs_bh = 'supera'
        diff = model_results['Random Forest']['cum_strategy'][-1] - model_results['Random Forest']['cum_bh'][-1]
    else:
        vs_bh = 'está por detrás de'
        diff = model_results['Random Forest']['cum_strategy'][-1] - model_results['Random Forest']['cum_bh'][-1]

    natural_text = (
        f'El modelo predice que el oro {signal_text} mañana con una confianza del {max(0.5, latest_proba):.0%}. '
        f'En los últimos 20 días hábiles, acertó {last20_hits} de {len(last20_pred)} predicciones ({last20_acc:.0%}). '
        f'La estrategia del modelo {vs_bh} a "comprar y mantener" por {diff:+.1%}.'
    )

    return {
        'scaler': scaler,
        'train': train,
        'test': test,
        'models': models,
        'model_results': model_results,
        'feature_importance': feature_importance,
        'confusion_matrix': cm,
        'roc_curve': {'fpr': fpr, 'tpr': tpr},
        'last20_pred': last20_pred,
        'last20_actual': last20_actual,
        'last20_acc': last20_acc,
        'last20_hits': last20_hits,
        'natural_text': natural_text,
        'split_index': split_index,
        'primary_model_name': 'Random Forest',
    }


def build_context() -> dict:
    raw_data, source = load_market_data()
    data = compute_indicators(raw_data)
    model_results = train_models(data)

    latest = data.iloc[-1]
    prev = data.iloc[-2]
    change = latest['gold'] - prev['gold']
    change_pct = float((change / prev['gold']) * 100)
    signal_probability = float(latest['signal_proba'])
    signal_value = int(latest['signal'])

    if signal_probability >= 0.58:
        signal_label = 'ALZA'
        signal_color = '#4ade80'
    elif signal_probability <= 0.42:
        signal_label = 'PRECAUCIÓN'
        signal_color = '#f2554d'
    else:
        signal_label = 'ESTABLE'
        signal_color = '#e6b84c'

    # Build model comparison table data
    model_table_data = []
    for name, res in model_results['model_results'].items():
        model_table_data.append({
            'name': name,
            'accuracy': res['accuracy'],
            'f1': res['f1'],
            'auc': res['auc'],
            'cum_return': res['cum_strategy'][-1],
        })

    return {
        'data': data,
        'latest': latest,
        'change': float(change),
        'change_pct': change_pct,
        'signal_label': signal_label,
        'signal_color': signal_color,
        'confidence': signal_probability,
        'train': model_results['train'],
        'test': model_results['test'],
        'accuracy': model_results['model_results']['Random Forest']['accuracy'],
        'f1': model_results['model_results']['Random Forest']['f1'],
        'auc': model_results['model_results']['Random Forest']['auc'],
        'predictions': model_results['model_results']['Random Forest']['predictions'],
        'probabilities': model_results['model_results']['Random Forest']['probabilities'],
        'split_index': model_results['split_index'],
        'split_date': data.index[model_results['split_index']],
        'test_data': model_results['test'],
        'cum_strategy': model_results['model_results']['Random Forest']['cum_strategy'],
        'cum_bh': model_results['model_results']['Random Forest']['cum_bh'],
        'load_source': source,
        'loaded_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        'signal_series': data['signal'],
        # New additions
        'model_results': model_results['model_results'],
        'feature_importance': model_results['feature_importance'],
        'confusion_matrix': model_results['confusion_matrix'],
        'roc_curve': model_results['roc_curve'],
        'last20_pred': model_results['last20_pred'],
        'last20_actual': model_results['last20_actual'],
        'last20_acc': model_results['last20_acc'],
        'last20_hits': model_results['last20_hits'],
        'natural_text': model_results['natural_text'],
        'model_table_data': model_table_data,
        'scaler': model_results['scaler'],
    }

context = build_context()