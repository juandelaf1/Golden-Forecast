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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

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
