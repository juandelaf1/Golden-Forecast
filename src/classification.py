"""
Módulo de clasificación.

Recibe el dataset preprocesado (pipeline.py) y entrena
modelos para predecir si el oro sube o baja al día siguiente.

Modelos: Dummy (baseline), Logistic Regression, Random Forest, XGBoost
Métricas: Accuracy, Precision, Recall, F1, ROC-AUC
"""

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def get_models():
    """Devuelve diccionario de modelos de clasificación."""
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric="logloss"
        )
    return models


def get_voting_model():
    """Voting Classifier que promedia LR + RF + XGB."""
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ]
    if XGB_AVAILABLE:
        estimators.append(('xgb', XGBClassifier(n_estimators=200, max_depth=4, random_state=42, eval_metric='logloss')))
    return VotingClassifier(estimators=estimators, voting='soft')


def evaluate(models, X_train, X_test, y_train, y_test, average="binary"):
    """Entrena y evalúa todos los modelos con métricas de clasificación."""
    results = []
    for name, model in models.items():
        m = model.__class__(**model.get_params())
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        y_proba = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None

        results.append({
            "Modelo": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, average=average, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
        })
    return pd.DataFrame(results)


def overfitting_check(models, X_train, X_test, y_train, y_test):
    """Compara rendimiento train vs test para detectar overfitting."""
    rows = []
    for name, model in models.items():
        m = model.__class__(**model.get_params())
        m.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, m.predict(X_train))
        test_acc = accuracy_score(y_test, m.predict(X_test))
        diff = train_acc - test_acc
        rows.append({
            "Modelo": name,
            "Train Acc": round(train_acc, 4),
            "Test Acc": round(test_acc, 4),
            "Diferencia": round(diff, 4),
            "Overfitting": "Sí" if diff > 0.05 else "No",
        })
    return pd.DataFrame(rows)


def backtest(models, X_train, X_test, y_train, y_test, returns_test):
    """Simula trading: predice UP → compra, DOWN → vende."""
    results = []
    for name, model in models.items():
        m = model.__class__(**model.get_params())
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        strategy_returns = np.where(preds == 1, returns_test, -returns_test)
        cum_strategy = (1 + strategy_returns).cumprod()
        cum_hold = (1 + returns_test).cumprod()
        results.append({
            "Modelo": name,
            "Estrategia": round(float(cum_strategy.iloc[-1]) if hasattr(cum_strategy, 'iloc') else cum_strategy[-1], 4),
            "Buy & Hold": round(float(cum_hold.iloc[-1]) if hasattr(cum_hold, 'iloc') else cum_hold[-1], 4),
        })
    df = pd.DataFrame(results)
    df["Ganancia"] = df["Estrategia"] - df["Buy & Hold"]
    return df


def feature_importance(model, feature_names):
    """Devuelve importancia de features ordenada (solo para modelos con feature_importances_)."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        return imp.sort_values(ascending=False)
    return None


def _make_horizon_target(y_reg_train, y_reg_test, horizon):
    """Crea target binario desplazado h días."""
    target_train = (np.concatenate([y_reg_train[horizon:], np.full(horizon, np.nan)]) > 0).astype(float)
    target_test = (np.concatenate([y_reg_test[horizon:], np.full(horizon, np.nan)]) > 0).astype(float)
    valid_train = ~np.isnan(target_train)
    valid_test = ~np.isnan(target_test)
    return target_train, target_test, valid_train, valid_test


def _eval_model(model, X_train, X_test, y_train, y_test, horizon):
    """Entrena un modelo y devuelve métricas."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probas) if probas is not None and len(np.unique(probas)) > 1 else 0.5

    return {
        'horizon': horizon,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'auc': round(auc, 4),
    }


def train_experimental_models(X_train, X_test, y_reg_train, y_reg_test, feature_columns):
    """
    Entrena modelos experimentales: LR, RF, XGB para horizontes [1,5,10,20] + voting + threshold óptimo.
    Devuelve dict con resultados por modelo y horizonte.
    """
    results = {'models': {}, 'voting': None, 'best_threshold_1d': None}

    HORIZONS = [1, 5, 10, 20]

    # Definir modelos a probar en cada horizonte
    model_defs = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1),
    }
    if XGB_AVAILABLE:
        model_defs['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=4, random_state=42, eval_metric='logloss')

    # Entrenar cada modelo en cada horizonte
    for model_name, model in model_defs.items():
        model_results = []
        for h in HORIZONS:
            target_train, target_test, valid_train, valid_test = _make_horizon_target(y_reg_train, y_reg_test, h)
            if valid_train.sum() < 100 or valid_test.sum() < 50:
                continue
            res = _eval_model(
                model.__class__(**model.get_params()),
                X_train[valid_train], X_test[valid_test],
                target_train[valid_train].astype(int), target_test[valid_test].astype(int),
                h,
            )
            model_results.append(res)
        results['models'][model_name] = model_results

    # Voting Classifier a 1 día
    target_train_1d, target_test_1d, valid_train_v, valid_test_v = _make_horizon_target(y_reg_train, y_reg_test, 1)
    try:
        voting = get_voting_model()
        voting.fit(X_train[valid_train_v], target_train_1d[valid_train_v].astype(int))
        voting_preds = voting.predict(X_test[valid_test_v])
        results['voting'] = {
            'accuracy': round(accuracy_score(target_test_1d[valid_test_v].astype(int), voting_preds), 4),
            'f1': round(f1_score(target_test_1d[valid_test_v].astype(int), voting_preds, zero_division=0), 4),
            'precision': round(precision_score(target_test_1d[valid_test_v].astype(int), voting_preds, zero_division=0), 4),
            'recall': round(recall_score(target_test_1d[valid_test_v].astype(int), voting_preds, zero_division=0), 4),
        }
    except Exception:
        results['voting'] = None

    # Threshold óptimo para LR a 1 día
    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train[valid_train_v], target_train_1d[valid_train_v].astype(int))
        probas_1d = lr.predict_proba(X_test[valid_test_v])[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(target_test_1d[valid_test_v].astype(int), probas_1d)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        results['best_threshold_1d'] = round(float(thresholds[best_idx]), 3)
        results['best_f1_at_threshold'] = round(float(f1_scores[best_idx]), 4)
    except Exception:
        results['best_threshold_1d'] = None

    return results
