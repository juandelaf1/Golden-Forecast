import os
import copy
import json
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

"""
Entrena modelos de clasificación para dos targets:

- target_binary: 1 = sube, 0 = baja
- target_multiclass: 1 = comprar, 0 = mantener, -1 = vender

Regla de preprocessing:
- Logistic Regression -> con escalado
- Random Forest / XGBoost / LightGBM -> sin escalado
"""

PATH = "data/processed/gold-features.csv"
MODELS_DIR = "models"
TEST_SIZE = 0.2

NO_FEATURE_COLUMN = ["Date", "target_binary", "target_multiclass"]

MULTICLASS_ENCODE = {-1: 0, 0: 1, 1: 2}
MULTICLASS_DECODE = {0: -1, 1: 0, 2: 1}

MODELS = {
    "lr": LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    ),
    "lr_strong_reg": LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=0.1
    ),
    "rf": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=20,
        random_state=42
    ),
    "xgb": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    ),
    "xgb_deep": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    ),
    "lgb": LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_child_samples=20,
        random_state=42,
        verbosity=-1
    ),
    "lgb_deep": LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_child_samples=20,
        random_state=42,
        verbosity=-1
    ),
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Rango temporal: {df['Date'].min().date()} -> {df['Date'].max().date()}")

    return df


def prepare_xy(df: pd.DataFrame):
    X = df.drop(columns=NO_FEATURE_COLUMN)
    y_binary = df["target_binary"]
    y_multiclass = df["target_multiclass"]

    print(f"\nFeatures: {X.shape[1]} columnas")
    print(f"Target binario — clases: {sorted(y_binary.unique())}")
    print(f"Target multiclase — clases: {sorted(y_multiclass.unique())}")

    return X, y_binary, y_multiclass


def temporal_split(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE):
    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def uses_scaled_data(model_name: str) -> bool:
    return model_name.startswith("lr")


def train_models(X_train_raw, X_train_scaled, y_binary_train, y_multiclass_train):
    trained = {}

    for model_name, model in MODELS.items():
        print(f"\nEntrenando {model_name}...")

        X_used = X_train_scaled if uses_scaled_data(model_name) else X_train_raw

        m_binary = copy.deepcopy(model)
        m_binary.fit(X_used, y_binary_train)
        trained[f"{model_name}_binary"] = m_binary
        print(
            f" {model_name}_binary -> OK "
            f"({'scaled' if uses_scaled_data(model_name) else 'raw'})"
        )

        y_multi_encoded = (
            y_multiclass_train.map(MULTICLASS_ENCODE)
            if ("xgb" in model_name and "lgb" not in model_name)
            else y_multiclass_train
        )

        m_multi = copy.deepcopy(model)
        m_multi.fit(X_used, y_multi_encoded)
        trained[f"{model_name}_multiclass"] = m_multi
        print(
            f" {model_name}_multiclass -> OK "
            f"({'scaled' if uses_scaled_data(model_name) else 'raw'})"
        )

    return trained


def save_models(trained: dict, scaler: StandardScaler, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)

    for name, model in trained.items():
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Guardado: {path}")

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Guardado: {scaler_path}")

    metadata = {
        "modelos": list(trained.keys()),
        "features_path": PATH,
        "test_size": TEST_SIZE,
        "no_feature_columns": NO_FEATURE_COLUMN,
        "scaled_models": [name for name in MODELS if uses_scaled_data(name)],
        "raw_models": [name for name in MODELS if not uses_scaled_data(name)],
        "notes": {
            "lr": "train/predict con scaler",
            "rf_xgb_lgb": "train/predict sin scaler"
        }
    }

    meta_path = os.path.join(models_dir, "train_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Guardado: {meta_path}")


def main():
    print("=" * 60)
    print("Golden-Forecast — Entrenamiento de modelos")
    print("=" * 60)

    df = load_data(PATH)

    X, y_binary, y_multiclass = prepare_xy(df)

    X_train, X_test, y_bin_train, y_bin_test = temporal_split(X, y_binary)
    _, _, y_multi_train, y_multi_test = temporal_split(X, y_multiclass)

    print("\nSplit temporal:")
    print(f" Train: {X_train.shape[0]} filas")
    print(f" Test: {X_test.shape[0]} filas")

    X_train_scaled, X_test_scaled, scaler = scale(X_train, X_test)
    print("\nEscalado aplicado — scaler ajustado solo sobre train")

    print("\n" + "=" * 60)
    print("Entrenando modelos...")
    print("=" * 60)
    trained = train_models(X_train, X_train_scaled, y_bin_train, y_multi_train)

    rf = trained["rf_binary"]
    importance = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": rf.feature_importances_
        })
        .sort_values("importance", ascending=False)
    )

    print("\n========== IMPORTANCIA RANDOM FOREST ==========")
    print(importance)
    print("===============================================\n")

    print("\n" + "=" * 60)
    print("Guardando modelos...")
    print("=" * 60)
    save_models(trained, scaler, MODELS_DIR)

    print("\nEntrenamiento completado.")

    return {
        "df": df,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_bin_train": y_bin_train,
        "y_bin_test": y_bin_test,
        "y_multi_train": y_multi_train,
        "y_multi_test": y_multi_test,
        "trained": trained,
        "scaler": scaler,
    }


if __name__ == "__main__":
    main()