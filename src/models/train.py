import os
import shutil
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
Entrena modelos de clasificación binaria y multiclase con una política consistente:
- Modelos lineales: se entrenan con datos escalados.
- Random Forest / XGBoost / LightGBM: se entrenan con datos sin escalar.
- gold_return se conserva en el CSV para backtest/evaluación, pero nunca entra en X.
- gold_return_next_day tampoco puede entrar en X porque contiene información futura.
"""

PATH = "data/processed/gold-features.csv"
MODELS_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42

NO_FEATURE_COLUMN = [
    "Date",
    "target_binary",
    "target_multiclass",
    "gold_return",
    "gold_return_next_day",
]

MULTICLASS_ENCODE = {-1: 0, 0: 1, 1: 2}
MULTICLASS_DECODE = {0: -1, 1: 0, 2: 1}

MODELS = {
    "lr": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0),
    "lr_strong_reg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.1),
    "rf": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    ),
    "xgb": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    ),
    "xgb_deep": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    ),
    "lgb": LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        verbosity=-1,
    ),
    "lgb_deep": LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        verbosity=-1,
    ),
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Rango temporal: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    return df


def get_feature_columns(df: pd.DataFrame):
    return [col for col in df.columns if col not in NO_FEATURE_COLUMN]


def prepare_xy(df: pd.DataFrame):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y_binary = df["target_binary"].copy()
    y_multiclass = df["target_multiclass"].copy()
    print(f"\nFeatures: {X.shape[1]} columnas")
    print(f"Columnas excluidas de X: {NO_FEATURE_COLUMN}")
    print(f"Target binario — clases: {sorted(y_binary.unique())}")
    print(f"Target multiclase — clases: {sorted(y_multiclass.unique())}")
    return X, y_binary, y_multiclass, feature_cols


def temporal_split(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE):
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    return X_train, X_test, y_train, y_test


def scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def clean_models_dir(models_dir: str):
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
    os.makedirs(models_dir, exist_ok=True)


def train_models(X_train_df, X_train_scaled, y_binary_train, y_multiclass_train):
    trained = {}
    for model_name, model in MODELS.items():
        print(f"\nEntrenando {model_name}...")
        use_scaled = model_name.startswith("lr")
        X_fit = X_train_scaled if use_scaled else X_train_df

        m_binary = copy.deepcopy(model)
        m_binary.fit(X_fit, y_binary_train)
        trained[f"{model_name}_binary"] = {
            "model": m_binary,
            "uses_scaled": use_scaled,
        }
        print(f" {model_name}_binary -> OK")

        y_multi_encoded = y_multiclass_train.map(MULTICLASS_ENCODE) if ("xgb" in model_name and "lgb" not in model_name) else y_multiclass_train

        m_multi = copy.deepcopy(model)
        m_multi.fit(X_fit, y_multi_encoded)
        trained[f"{model_name}_multiclass"] = {
            "model": m_multi,
            "uses_scaled": use_scaled,
        }
        print(f" {model_name}_multiclass -> OK")

    return trained


def save_models(trained: dict, scaler: StandardScaler, models_dir: str, feature_cols: list):
    clean_models_dir(models_dir)

    metadata_models = []
    for name, payload in trained.items():
        model = payload["model"]
        uses_scaled = payload["uses_scaled"]
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        metadata_models.append({"name": name, "uses_scaled": uses_scaled})
        print(f"Guardado: {path}")

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Guardado: {scaler_path}")

    metadata = {
        "modelos": metadata_models,
        "features_path": PATH,
        "test_size": TEST_SIZE,
        "no_feature_columns": NO_FEATURE_COLUMN,
        "feature_columns": feature_cols,
        "multiclass_encode": MULTICLASS_ENCODE,
        "multiclass_decode": MULTICLASS_DECODE,
    }

    meta_path = os.path.join(models_dir, "train_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Guardado: {meta_path}")


def main():
    print("=" * 60)
    print("Golden-Forecast — Entrenamiento de modelos")
    print("=" * 60)

    df = load_data(PATH)
    X, y_binary, y_multiclass, feature_cols = prepare_xy(df)

    X_train_df, X_test_df, y_bin_train, y_bin_test = temporal_split(X, y_binary)
    _, _, y_multi_train, y_multi_test = temporal_split(X, y_multiclass)

    print(f"\nSplit temporal:")
    print(f" Train: {X_train_df.shape[0]} filas")
    print(f" Test: {X_test_df.shape[0]} filas")

    X_train_scaled, X_test_scaled, scaler = scale(X_train_df, X_test_df)
    print("\nEscalado aplicado — scaler ajustado solo sobre train")

    print("\n" + "=" * 60)
    print("Entrenando modelos...")
    print("=" * 60)
    trained = train_models(X_train_df, X_train_scaled, y_bin_train, y_multi_train)

    rf = trained["rf_binary"]["model"]
    importance = (
        pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n========== IMPORTANCIA RANDOM FOREST ==========")
    print(importance.to_string(index=False))
    print("===============================================\n")

    print("\n" + "=" * 60)
    print("Guardando modelos...")
    print("=" * 60)
    save_models(trained, scaler, MODELS_DIR, feature_cols)

    print("\nEntrenamiento completado.")

    return {
        "df": df,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_bin_train": y_bin_train,
        "y_bin_test": y_bin_test,
        "y_multi_train": y_multi_train,
        "y_multi_test": y_multi_test,
        "trained": trained,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    main()