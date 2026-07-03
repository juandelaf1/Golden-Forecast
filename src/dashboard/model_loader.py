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


def get_relevant_models():
    metadata = load_metadata()
    all_models = {}
    for name in metadata['modelos']:
        if name in DIVERSE_MODELS:
            all_models[name] = load_model(name)
    return all_models


def build_pretrained_context():
    df = load_features()
    X, y_binary, y_multiclass = prepare_features(df)
    scaler = load_scaler()
    models = get_relevant_models()
    eval_results = load_evaluation_results()

    predictions = generate_predictions(models, scaler, X)

    # Primary model: best binary (lr_strong_reg_binary)
    primary_name = 'lr_strong_reg_binary'
    primary_pred = predictions[primary_name]['predictions']
    primary_proba = predictions[primary_name]['probabilities']
    signal = primary_pred[-1]
    signal_proba = float(primary_proba[-1][1]) if primary_proba is not None and primary_proba.shape[1] > 1 else 0.5

    # Model signals for all models (used in comparison charts)
    model_signals = {}
    for name in models:
        preds = predictions[name]['predictions']
        model_signals[name] = preds

    return {
        'df': df,
        'X': X,
        'y_binary': y_binary,
        'y_multiclass': y_multiclass,
        'scaler': scaler,
        'models': models,
        'predictions': predictions,
        'model_signals': model_signals,
        'eval_results': eval_results,
        'primary_name': primary_name,
        'primary_signal': int(signal),
        'primary_proba': signal_proba,
    }
