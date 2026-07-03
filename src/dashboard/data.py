import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)

from src.dashboard.model_loader import (
    build_pretrained_context, load_features, load_evaluation_results,
)

try:
    from src.classification import train_experimental_models
    EXPERIMENTAL_AVAILABLE = True
except ImportError:
    EXPERIMENTAL_AVAILABLE = False

try:
    from src.regression import train_and_evaluate_regression
    REGRESSION_AVAILABLE = True
except ImportError:
    REGRESSION_AVAILABLE = False

CLEAN_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'gold-clean.csv')

FEATURE_COLUMNS = [
    'gold_volume',
    'gold_return', 'dxy_return', 'vix_return', 'tnx_return',
    'gold_daily_range', 'dxy_daily_range', 'vix_daily_range', 'tnx_daily_range',
    'gold_open_close_return', 'dxy_open_close_return', 'vix_open_close_return', 'tnx_open_close_return',
    'gold_ma_5', 'gold_ma_20',
    'gold_close_vs_ma_5', 'gold_close_vs_ma_20',
    'gold_rsi_14', 'gold_macd', 'gold_macd_signal',
    'gold_volatility_14',
    'gold_return_lag_1', 'gold_return_lag_2',
]

PRIMARY_MODEL = 'lr_strong_reg_binary'


def load_feature_data() -> pd.DataFrame:
    """Load gold-features.csv and merge with gold_close for charting."""
    features = load_features()
    clean = pd.read_csv(CLEAN_PATH, parse_dates=['Date'])
    merged = features.merge(clean[['Date', 'gold_close']], on='Date', how='left')
    merged = merged.rename(columns={'gold_close': 'gold'})
    merged['returns'] = merged['gold_return']
    return merged


def load_market_data() -> tuple[pd.DataFrame, str]:
    data = load_feature_data()
    data = data.dropna().reset_index(drop=True)
    return data, 'gold-features.csv'


def _build_model_results(ml_context: dict, data: pd.DataFrame, split_index: int) -> dict:
    """Build model_results dict matching pre-trained evaluation metrics per model."""
    eval_data = ml_context['eval_results']
    predictions = ml_context['predictions']
    y_binary_test = ml_context['y_binary'].iloc[split_index:].values
    y_multi_test = ml_context['y_multiclass'].iloc[split_index:].values
    returns_test = data['returns'].iloc[split_index:].values

    cum_bh = np.cumprod(1 + returns_test) - 1
    model_results = {}

    for entry in eval_data['resultados']:
        name = entry['modelo']
        if name not in predictions:
            continue

        preds = predictions[name]['predictions'][split_index:]
        probas = predictions[name]['probabilities'][split_index:] if predictions[name]['probabilities'] is not None else None

        is_multiclass = 'multiclass' in name
        y_test = y_multi_test if is_multiclass else y_binary_test
        avg = 'macro' if is_multiclass else 'binary'

        acc = entry.get('accuracy_test', accuracy_score(y_test, preds))
        prec = entry.get('precision_test', precision_score(y_test, preds, average=avg, zero_division=0))
        rec = entry.get('recall_test', recall_score(y_test, preds, average=avg, zero_division=0))
        f1 = entry.get('f1_test', f1_score(y_test, preds, average=avg, zero_division=0))

        auc = 0.5
        proba_vector = np.full_like(preds, 0.5, dtype=float)
        if probas is not None:
            if is_multiclass:
                auc = 0.5
            elif probas.shape[1] > 1:
                proba_vector = probas[:, 1]
                try:
                    auc = roc_auc_score(y_test, proba_vector)
                except ValueError:
                    auc = 0.5

        strat_ret = returns_test * preds
        cum_strat = np.cumprod(1 + strat_ret) - 1

        model_results[name] = {
            'model': ml_context['models'].get(name),
            'type': 'classification',
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'auc': auc,
            'predictions': preds,
            'probabilities': proba_vector,
            'cum_strategy': cum_strat,
            'cum_bh': cum_bh,
        }

    return model_results


def build_context() -> dict:
    data, source = load_market_data()
    ml_context = build_pretrained_context()
    split_index = int(len(data) * 0.8)
    model_results = _build_model_results(ml_context, data, split_index)

    # Primary model
    primary_preds = ml_context['predictions'][PRIMARY_MODEL]
    primary_probas = ml_context['predictions'][PRIMARY_MODEL]['probabilities']
    if primary_probas is not None and primary_probas.shape[1] > 1:
        signal_series = primary_preds['predictions']
        signal_proba_series = primary_probas[:, 1]
    else:
        signal_series = primary_preds['predictions']
        signal_proba_series = np.full_like(signal_series, 0.5)

    data['signal'] = signal_series
    data['signal_proba'] = signal_proba_series

    primary_result = model_results.get(PRIMARY_MODEL)
    if primary_result is None:
        primary_result = list(model_results.values())[0]

    # Feature importance from rf_binary
    rf_model = ml_context['models'].get('rf_binary')
    if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
        fi = rf_model.feature_importances_
        fi_indices = np.argsort(fi)[::-1]
        feature_importance = {
            'features': [FEATURE_COLUMNS[i] for i in fi_indices],
            'importance': fi[fi_indices].tolist(),
        }
    else:
        feature_importance = {'features': FEATURE_COLUMNS, 'importance': [0] * len(FEATURE_COLUMNS)}

    # Confusion matrix + ROC from primary model
    is_multi_primary = 'multiclass' in PRIMARY_MODEL
    test_target = ml_context['y_multiclass'].iloc[split_index:].values if is_multi_primary else ml_context['y_binary'].iloc[split_index:].values
    cm = confusion_matrix(test_target, primary_result['predictions'])
    if not is_multi_primary and len(np.unique(primary_result['probabilities'])) > 1:
        fpr, tpr, _ = roc_curve(test_target, primary_result['probabilities'])
    else:
        fpr, tpr = np.array([0, 0, 1]), np.array([0, 1, 1])

    # Last 20 days
    last20_pred = primary_result['predictions'][-20:]
    last20_actual = test_target[-20:]
    last20_acc = (last20_pred == last20_actual).mean()
    last20_hits = int(last20_pred.sum())

    # Signal labels
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    change = float(latest['gold']) - float(prev['gold'])
    change_pct = float((change / float(prev['gold'])) * 100) if float(prev['gold']) > 0 else 0
    signal_probability = float(data['signal_proba'].iloc[-1])
    signal_value = int(data['signal'].iloc[-1])

    if signal_probability >= 0.58:
        signal_label = 'ALZA'
        signal_color = '#4ade80'
    elif signal_probability <= 0.42:
        signal_label = 'PRECAUCIÓN'
        signal_color = '#f2554d'
    else:
        signal_label = 'ESTABLE'
        signal_color = '#e6b84c'

    # Natural text
    signal_text = 'SUBIRÁ' if signal_value == 1 else 'BAJARÁ'
    if primary_result['cum_strategy'][-1] > primary_result['cum_bh'][-1]:
        vs_bh = 'supera'
        diff = primary_result['cum_strategy'][-1] - primary_result['cum_bh'][-1]
    else:
        vs_bh = 'está por detrás de'
        diff = primary_result['cum_strategy'][-1] - primary_result['cum_bh'][-1]

    natural_text = (
        f'El modelo predice que el oro {signal_text} mañana con una confianza del {max(0.5, signal_probability):.0%}. '
        f'En los últimos 20 días hábiles, acertó {last20_hits} de {len(last20_pred)} predicciones ({last20_acc:.0%}). '
        f'La estrategia del modelo {vs_bh} a "comprar y mantener" por {diff:+.1%}.'
    )

    # Model table data
    model_table_data = []
    for name, res in model_results.items():
        model_table_data.append({
            'name': name,
            'type': res.get('type', 'classification'),
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'auc': res['auc'],
            'cum_return': res['cum_strategy'][-1],
        })

    # Experimental models
    experimental_results = None
    if EXPERIMENTAL_AVAILABLE:
        try:
            scaler = ml_context['scaler']
            X_all = scaler.transform(ml_context['X'])
            X_train = X_all[:split_index]
            X_test_e = X_all[split_index:]
            y_reg_train = data['returns'].iloc[:split_index].values
            y_reg_test_e = data['returns'].iloc[split_index:].values
            experimental_results = train_experimental_models(
                X_train, X_test_e, y_reg_train, y_reg_test_e, FEATURE_COLUMNS
            )
        except Exception:
            experimental_results = None

    # Regression models (Juan de la Fuente)
    regression_results = None
    if REGRESSION_AVAILABLE:
        try:
            regression_results = train_and_evaluate_regression()
        except Exception:
            regression_results = None

    return {
        'data': data,
        'latest': latest,
        'change': change,
        'change_pct': change_pct,
        'signal_label': signal_label,
        'signal_color': signal_color,
        'confidence': signal_probability,
        'train': data.iloc[:split_index],
        'test': data.iloc[split_index:],
        'accuracy': primary_result['accuracy'],
        'f1': primary_result['f1'],
        'precision': primary_result['precision'],
        'recall': primary_result['recall'],
        'auc': primary_result['auc'],
        'predictions': primary_result['predictions'],
        'probabilities': primary_result['probabilities'],
        'split_index': split_index,
        'split_date': data['Date'].iloc[split_index] if 'Date' in data.columns else data.iloc[split_index:].index[0],
        'test_data': data.iloc[split_index:],
        'cum_strategy': primary_result['cum_strategy'],
        'cum_bh': primary_result['cum_bh'],
        'load_source': source,
        'loaded_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        'signal_series': data['signal'],
        'model_results': model_results,
        'feature_importance': feature_importance,
        'confusion_matrix': cm,
        'roc_curve': {'fpr': fpr, 'tpr': tpr},
        'last20_pred': last20_pred,
        'last20_actual': last20_actual,
        'last20_acc': last20_acc,
        'last20_hits': last20_hits,
        'natural_text': natural_text,
        'model_table_data': model_table_data,
        'scaler': ml_context['scaler'],
        'experimental': experimental_results,
        'regression': regression_results,
    }


context = build_context()
