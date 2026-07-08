"""
Golden-Forecast: Optimización de hiperparámetros
--------------------------------------------------
Optimiza hiperparámetros con RandomizedSearchCV y TimeSeriesSplit.

Regla de preprocessing:
- Logistic Regression -> con escalado dentro de Pipeline
- Random Forest / LightGBM -> sin escalado
"""

import os
import json
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, make_scorer

PATH = "data/processed/gold-features.csv"
MODELS_DIR = "models"
TEST_SIZE = 0.2
NON_FEATURE_COLS = ["Date", "target_binary", "target_multiclass"]

RF_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 4, 5],
    "min_samples_leaf": [15, 20, 30],
    "max_features": ["sqrt", "log2", 0.3],
    "class_weight": [None, "balanced"],
}

LR_PARAM_GRID = {
    "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    "model__solver": ["lbfgs", "liblinear"],
    "model__class_weight": [None, "balanced"],
    "model__max_iter": [2000],
}

LGBM_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_samples": [10, 20, 30, 50],
    "num_leaves": [15, 31, 63],
}

N_ITER = 30
N_SPLITS = 5
RANDOM_STATE = 42


def load_and_split(path: str, test_size: float = TEST_SIZE):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    X = df.drop(columns=NON_FEATURE_COLS)
    y_binary = df["target_binary"]
    y_multiclass = df["target_multiclass"]

    split_idx = int(len(df) * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_bin_train = y_binary.iloc[:split_idx]
    y_bin_test = y_binary.iloc[split_idx:]
    y_multi_train = y_multiclass.iloc[:split_idx]
    y_multi_test = y_multiclass.iloc[split_idx:]

    print(f"Dataset: {df.shape[0]} filas x {X.shape[1]} features")
    print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

    return X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test


def optimize_model(model, param_grid: dict, X_train, y_train, model_name: str, n_iter: int = N_ITER):
    print(f"\nOptimizando {model_name}...")
    print(f" Probando {n_iter} combinaciones con {N_SPLITS} ventanas temporales")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    avg = "binary" if "binary" in model_name else "weighted"
    scorer = make_scorer(f1_score, average=avg, zero_division=0)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f" Mejor F1 en CV: {search.best_score_:.4f}")
    print(f" Mejores parámetros: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_optimized(model, model_name: str, X_train, X_test, y_train, y_test) -> dict:
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    avg = "binary" if "binary" in model_name else "weighted"

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average=avg, zero_division=0)

    print(f"\n Resultados en test — {model_name}:")
    print(f" Accuracy train: {acc_train:.4f}")
    print(f" Accuracy test: {acc_test:.4f}")
    print(f" F1 test: {f1_test:.4f}")
    print(f" Overfitting gap: {acc_train - acc_test:.4f}")

    return {
        "modelo": model_name,
        "accuracy_train": round(acc_train, 4),
        "accuracy_test": round(acc_test, 4),
        "f1_test": round(f1_test, 4),
        "overfitting_gap": round(acc_train - acc_test, 4),
    }


def save_model(model, name: str, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f" Guardado: {path}")


def main():
    print("=" * 60)
    print("Golden-Forecast — Optimización de hiperparámetros")
    print("=" * 60)

    (
        X_train, X_test,
        y_bin_train, y_bin_test,
        y_multi_train, y_multi_test,
    ) = load_and_split(PATH)

    results = []
    best_params_log = {}

    print("\n" + "=" * 60)
    print("RANDOM FOREST — Target binario")
    print("=" * 60)

    rf_bin_model, rf_bin_params, _ = optimize_model(
        RandomForestClassifier(random_state=RANDOM_STATE),
        RF_PARAM_GRID,
        X_train,
        y_bin_train,
        model_name="rf_optimized_binary",
    )

    rf_bin_metrics = evaluate_optimized(
        rf_bin_model, "rf_optimized_binary",
        X_train, X_test,
        y_bin_train, y_bin_test,
    )
    results.append(rf_bin_metrics)
    best_params_log["rf_optimized_binary"] = rf_bin_params
    save_model(rf_bin_model, "rf_optimized_binary", MODELS_DIR)

    print("\n" + "=" * 60)
    print("RANDOM FOREST — Target multiclase")
    print("=" * 60)

    rf_multi_model, rf_multi_params, _ = optimize_model(
        RandomForestClassifier(random_state=RANDOM_STATE),
        RF_PARAM_GRID,
        X_train,
        y_multi_train,
        model_name="rf_optimized_multiclass",
    )

    rf_multi_metrics = evaluate_optimized(
        rf_multi_model, "rf_optimized_multiclass",
        X_train, X_test,
        y_multi_train, y_multi_test,
    )
    results.append(rf_multi_metrics)
    best_params_log["rf_optimized_multiclass"] = rf_multi_params
    save_model(rf_multi_model, "rf_optimized_multiclass", MODELS_DIR)

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION — Target binario")
    print("=" * 60)

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=RANDOM_STATE)),
    ])

    lr_bin_model, lr_bin_params, _ = optimize_model(
        lr_pipeline,
        LR_PARAM_GRID,
        X_train,
        y_bin_train,
        model_name="lr_optimized_binary",
        n_iter=20,
    )

    lr_bin_metrics = evaluate_optimized(
        lr_bin_model, "lr_optimized_binary",
        X_train, X_test,
        y_bin_train, y_bin_test,
    )
    results.append(lr_bin_metrics)
    best_params_log["lr_optimized_binary"] = lr_bin_params
    save_model(lr_bin_model, "lr_optimized_binary", MODELS_DIR)

    print("\n" + "=" * 60)
    print("LIGHTGBM — Target binario")
    print("=" * 60)

    lgbm_bin_model, lgbm_bin_params, _ = optimize_model(
        LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
        LGBM_PARAM_GRID,
        X_train,
        y_bin_train,
        model_name="lgbm_optimized_binary",
    )

    lgbm_bin_metrics = evaluate_optimized(
        lgbm_bin_model, "lgbm_optimized_binary",
        X_train, X_test,
        y_bin_train, y_bin_test,
    )
    results.append(lgbm_bin_metrics)
    best_params_log["lgbm_optimized_binary"] = lgbm_bin_params
    save_model(lgbm_bin_model, "lgbm_optimized_binary", MODELS_DIR)

    print("\n" + "=" * 60)
    print("RESUMEN OPTIMIZACIÓN")
    print("=" * 60)

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    output = {
        "resultados": results,
        "mejores_parametros": best_params_log,
    }
    out_path = os.path.join(MODELS_DIR, "optimization_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")

    meta_path = os.path.join(MODELS_DIR, "train_metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    nuevos = 0
    for name in best_params_log.keys():
        if name not in metadata["modelos"]:
            metadata["modelos"].append(name)
            nuevos += 1

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata actualizado — {nuevos} modelos optimizados añadidos")

    print("\n✓ Optimización completada.")
    print(f" Modelos guardados en: {MODELS_DIR}/")


if __name__ == "__main__":
    main()