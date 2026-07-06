import json
import os
import pickle

import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'gold-features.csv')
NO_FEATURE_COLUMNS = ['Date', 'target_binary', 'target_multiclass']

DIVERSE_MODELS = [
    'lr_strong_reg_binary',
    'rf_binary',
    'xgb_binary',
    'lr_multiclass',
]

_MODEL_CACHE = {}


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def load_metadata() -> dict:
    path = os.path.join(MODELS_DIR, 'train_metadata.json')
    with open(path) as f:
        return json.load(f)


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_scaler():
    path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_evaluation_results() -> dict:
    path = os.path.join(MODELS_DIR, 'evaluation_results.json')
    with open(path) as f:
        return json.load(f)


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=NO_FEATURE_COLUMNS)
    y_binary = df['target_binary']
    y_multiclass = df['target_multiclass']
    return X, y_binary, y_multiclass


def generate_predictions(models: dict, scaler, X_full: pd.DataFrame) -> dict:
    X_scaled = scaler.transform(X_full)
    results = {}
    for name, model in models.items():
        preds = model.predict(X_scaled)
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_scaled)
        else:
            probas = None
        results[name] = {
            'predictions': preds,
            'probabilities': probas,
        }
    return results


def ensure_predictions(name, X_scaled):
    """Load a model .pkl and generate predictions, caching results globally."""
    if name not in _MODEL_CACHE:
        model = load_model(name)
        preds = model.predict(X_scaled)
        probas = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        _MODEL_CACHE[name] = {
            'predictions': preds,
            'probabilities': probas,
            'model': model,
        }
    return _MODEL_CACHE[name]


def build_pretrained_context():
    df = load_features()
    X, y_binary, y_multiclass = prepare_features(df)
    scaler = load_scaler()
    X_scaled = scaler.transform(X)
    eval_results = load_evaluation_results()

    # Load only primary model eagerly
    primary_name = 'lr_strong_reg_binary'
    primary = ensure_predictions(primary_name, X_scaled)

    all_models = {primary_name: primary['model']}
    predictions = {primary_name: {'predictions': primary['predictions'], 'probabilities': primary['probabilities']}}
    model_signals = {primary_name: primary['predictions']}

    signal = primary['predictions'][-1]
    primary_proba = primary['probabilities']
    signal_proba = float(primary_proba[-1][1]) if primary_proba is not None and primary_proba.shape[1] > 1 else 0.5

    return {
        'df': df,
        'X': X,
        'X_scaled': X_scaled,
        'y_binary': y_binary,
        'y_multiclass': y_multiclass,
        'scaler': scaler,
        'models': all_models,
        'predictions': predictions,
        'model_signals': model_signals,
        'eval_results': eval_results,
        'primary_name': primary_name,
        'primary_signal': int(signal),
        'primary_proba': signal_proba,
    }
