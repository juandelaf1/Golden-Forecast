"""
Golden-Forecast: Optimización de hiperparámetros
--------------------------------------------------
Busca los mejores hiperparámetros para Random Forest y Logistic Regression
usando RandomizedSearchCV con validación temporal (TimeSeriesSplit).

No usamos GridSearchCV porque el espacio de búsqueda es grande y
RandomizedSearchCV es más eficiente — prueba combinaciones aleatorias
en lugar de todas las combinaciones posibles.

TimeSeriesSplit es crucial aquí: respeta el orden cronológico de los datos
durante la validación cruzada, evitando data leakage temporal.

Input:
    data/processed/gold-features.csv
    models/scaler.pkl

Output:
    models/rf_optimized_binary.pkl
    models/rf_optimized_multiclass.pkl
    models/lr_optimized_binary.pkl
    models/optimization_results.json
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, make_scorer

# ---------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------

PATH = "data/processed/gold-features.csv"
MODELS_DIR = "models"
TEST_SIZE = 0.2
NON_FEATURE_COLS = ["Date", "target_binary", "target_multiclass"]

MULTICLASS_ENCODE = {-1: 0, 0: 1, 1: 2}
MULTICLASS_DECODE = {0: -1, 1: 0, 2: 1}

# Espacio de búsqueda para Random Forest
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],  # quitar None — sin límite causa overfitting
    "min_samples_leaf": [10, 15, 20, 30],  # quitar 50 — demasiado restrictivo
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "class_weight": [None, "balanced"],
}

# Espacio de búsqueda para Logistic Regression
# penalty está deprecado en sklearn 1.8 — usamos solo C y solver
LR_PARAM_GRID = {
    "C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    "solver": ["lbfgs", "liblinear"],
    "class_weight": [None, "balanced"],
    "max_iter": [2000],
}

LGBM_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_samples": [10, 20, 30, 50],
    "num_leaves": [15, 31, 63],
}

N_ITER = 30       # combinaciones a probar por modelo
N_SPLITS = 5      # ventanas temporales para TimeSeriesSplit
RANDOM_STATE = 42


# ---------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------

def load_and_split(path: str, test_size: float = TEST_SIZE):
    """Carga el dataset y hace el split temporal."""
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


def load_scaler(models_dir: str):
    path = os.path.join(models_dir, "scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------
# Optimización
# ---------------------------------------------------------------

def optimize_model(model, param_grid: dict, X_train, y_train,
                   model_name: str, n_iter: int = N_ITER):
    """
    Busca los mejores hiperparámetros usando RandomizedSearchCV
    con TimeSeriesSplit para respetar el orden temporal.

    TimeSeriesSplit divide el train en N ventanas cronológicas,
    nunca usando datos futuros para validar — igual que en producción real.
    """
    print(f"\nOptimizando {model_name}...")
    print(f"  Probando {n_iter} combinaciones con {N_SPLITS} ventanas temporales")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Usamos F1 como métrica de optimización
    # Para binario: f1 estándar
    # Para multiclase: f1 weighted
    avg = "binary" if "binary" in model_name else "weighted"
    scorer = make_scorer(f1_score, average=avg, zero_division=0)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,      # usar todos los cores disponibles
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"  Mejor F1 en CV: {search.best_score_:.4f}")
    print(f"  Mejores parámetros: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_optimized(model, model_name: str,
                       X_train, X_test, y_train, y_test) -> dict:
    """Evalúa el modelo optimizado en train y test."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    avg = "binary" if "binary" in model_name else "weighted"

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average=avg, zero_division=0)

    print(f"\n  Resultados en test — {model_name}:")
    print(f"    Accuracy train: {acc_train:.4f}")
    print(f"    Accuracy test:  {acc_test:.4f}")
    print(f"    F1 test:        {f1_test:.4f}")
    print(f"    Overfitting gap: {acc_train - acc_test:.4f}")

    return {
        "modelo": model_name,
        "accuracy_train": round(acc_train, 4),
        "accuracy_test": round(acc_test, 4),
        "f1_test": round(f1_test, 4),
        "overfitting_gap": round(acc_train - acc_test, 4),
    }


def save_model(model, name: str, models_dir: str):
    path = os.path.join(models_dir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("Golden-Forecast — Optimización de hiperparámetros")
    print("=" * 60)

    # 1. Cargar datos
    (
        X_train, X_test,
        y_bin_train, y_bin_test,
        y_multi_train, y_multi_test,
    ) = load_and_split(PATH)

    # 2. Escalar — usamos el scaler ya ajustado sobre train
    scaler = load_scaler(MODELS_DIR)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = []
    best_params_log = {}

    # ---------------------------------------------------------------
    # 3. Optimizar RF binario
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RANDOM FOREST — Target binario")
    print("=" * 60)

    rf_bin_model, rf_bin_params, rf_bin_cv_score = optimize_model(
        RandomForestClassifier(random_state=RANDOM_STATE),
        RF_PARAM_GRID,
        X_train, y_bin_train,
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

    # ---------------------------------------------------------------
    # 4. Optimizar RF multiclase
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RANDOM FOREST — Target multiclase")
    print("=" * 60)

    rf_multi_model, rf_multi_params, rf_multi_cv_score = optimize_model(
        RandomForestClassifier(random_state=RANDOM_STATE),
        RF_PARAM_GRID,
        X_train, y_multi_train,
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

    # ---------------------------------------------------------------
    # 5. Optimizar LR binario
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION — Target binario")
    print("=" * 60)

    lr_bin_model, lr_bin_params, lr_bin_cv_score = optimize_model(
        LogisticRegression(random_state=RANDOM_STATE),
        LR_PARAM_GRID,
        X_train_sc, y_bin_train,
        model_name="lr_optimized_binary",
        n_iter=20,
    )

    lr_bin_metrics = evaluate_optimized(
        lr_bin_model, "lr_optimized_binary",
        X_train_sc, X_test_sc,
        y_bin_train, y_bin_test,
    )
    results.append(lr_bin_metrics)
    best_params_log["lr_optimized_binary"] = lr_bin_params
    save_model(lr_bin_model, "lr_optimized_binary", MODELS_DIR)
    
    
    # ---------------------------------------------------------------
    # 6. Optimizar LightGBM binario
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LIGHTGBM — Target binario")
    print("=" * 60)

    lgbm_bin_model, lgbm_bin_params, lgbm_bin_cv_score = optimize_model(
        LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
        LGBM_PARAM_GRID,
        X_train, y_bin_train,
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

    # ---------------------------------------------------------------
    # 7. Resumen final
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESUMEN OPTIMIZACIÓN")
    print("=" * 60)

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Comparar con modelos base
    print("\nComparativa vs modelos base:")
    print(f"  RF base binario:       F1=0.6963 | accuracy=0.5707")
    print(f"  RF optimizado binario: F1={rf_bin_metrics['f1_test']} | accuracy={rf_bin_metrics['accuracy_test']}")
    print(f"  LR base binario:       F1=0.6911 | accuracy=0.5742")
    print(f"  LR optimizado binario: F1={lr_bin_metrics['f1_test']} | accuracy={lr_bin_metrics['accuracy_test']}")

    # Guardar resultados de optimización
    output = {
        "resultados": results,
        "mejores_parametros": best_params_log,
    }
    out_path = os.path.join(MODELS_DIR, "optimization_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")

    # Actualizar train_metadata.json con los modelos optimizados
    # Así evaluate.py los carga automáticamente junto a los modelos base
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
    print(f"  Modelos guardados en: {MODELS_DIR}/")
    print(f"  Para evaluar completo: python src/models/evaluate.py")


if __name__ == "__main__":
    main()