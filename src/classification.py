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


def train_for_horizon(X_train, X_test, y_reg_train, y_reg_test, horizon=1, threshold=0.5):
    """
    Entrena Logistic Regression para predecir dirección del oro a N días vista.
    target = 1 si el precio en h días es mayor que hoy.
    """
    target_train = (np.concatenate([y_reg_train[horizon:], np.full(horizon, np.nan)]) > 0).astype(float)
    target_test = (np.concatenate([y_reg_test[horizon:], np.full(horizon, np.nan)]) > 0).astype(float)

    valid_train = ~np.isnan(target_train)
    valid_test = ~np.isnan(target_test)

    if valid_train.sum() < 100 or valid_test.sum() < 50:
        return None

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train[valid_train], target_train[valid_train])

    preds = model.predict(X_test[valid_test])
    probas = model.predict_proba(X_test[valid_test])[:, 1]
    preds_adj = (probas >= threshold).astype(int)

    acc = accuracy_score(target_test[valid_test], preds_adj)
    prec = precision_score(target_test[valid_test], preds_adj, zero_division=0)
    rec = recall_score(target_test[valid_test], preds_adj, zero_division=0)
    f1 = f1_score(target_test[valid_test], preds_adj, zero_division=0)

    if len(np.unique(probas)) > 1:
        auc = roc_auc_score(target_test[valid_test], probas)
    else:
        auc = 0.5

    return {
        'horizon': horizon,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'auc': round(auc, 4),
        'threshold': threshold,
    }


def optimize_threshold(y_true, probas):
    """Encuentra el threshold que maximiza F1 usando PR-curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def train_experimental_models(X_train, X_test, y_reg_train, y_reg_test, feature_columns):
    """
    Entrena modelos experimentales: horizontes [1,5,10,20] + voting + threshold óptimo.
    Devuelve dict listo para el dashboard.
    """
    results = {'horizons': [], 'voting': None, 'best_threshold_1d': None}

    # 1. Evaluar LR en cada horizonte
    for h in [1, 5, 10, 20]:
        res = train_for_horizon(X_train, X_test, y_reg_train, y_reg_test, horizon=h)
        if res:
            results['horizons'].append(res)

    # 2. Voting Classifier a 1 día (target desplazado, mismo que target_bin)
    target_train_shifted = np.concatenate([y_reg_train[1:], np.full(1, np.nan)])
    target_test_shifted = np.concatenate([y_reg_test[1:], np.full(1, np.nan)])
    valid_train_v = ~np.isnan(target_train_shifted)
    valid_test_v = ~np.isnan(target_test_shifted)
    target_train_1d = (target_train_shifted[valid_train_v] > 0).astype(int)
    target_test_1d = (target_test_shifted[valid_test_v] > 0).astype(int)
    try:
        voting = get_voting_model()
        voting.fit(X_train[valid_train_v], target_train_1d)
        voting_preds = voting.predict(X_test[valid_test_v])
        voting_probas = voting.predict_proba(X_test[valid_test_v])[:, 1]
        voting_acc = accuracy_score(target_test_1d, voting_preds)
        voting_f1 = f1_score(target_test_1d, voting_preds, zero_division=0)
        voting_prec = precision_score(target_test_1d, voting_preds, zero_division=0)
        voting_rec = recall_score(target_test_1d, voting_preds, zero_division=0)
        results['voting'] = {
            'accuracy': round(voting_acc, 4),
            'f1': round(voting_f1, 4),
            'precision': round(voting_prec, 4),
            'recall': round(voting_rec, 4),
        }
    except Exception as e:
        results['voting'] = None

    # 3. Threshold óptimo para LR a 1 día
    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train[valid_train_v], target_train_1d)
        probas_1d = lr.predict_proba(X_test[valid_test_v])[:, 1]
        best_th, best_f1 = optimize_threshold(target_test_1d, probas_1d)
        results['best_threshold_1d'] = round(float(best_th), 3)
        results['best_f1_at_threshold'] = round(float(best_f1), 4)
    except Exception:
        results['best_threshold_1d'] = None

    return results
