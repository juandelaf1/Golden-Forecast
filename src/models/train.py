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
Entrena 5 modelos de clasificación (3 algoritmos x 2 targets):
    Algoritmos:
        - Logistic Regression  (baseline)
        - Random Forest Classifier
        - XGBoost Classifier

    Targets:
        - target_binary     (1 = sube, 0 = baja)
        - target_multiclass (1 = comprar, 0 = mantener, -1 = vender)
"""   

PATH = "data/processed/gold-features.csv"
MODELS_DIR = "models"
TEST_SIZE = 0.2 ##Nuevo a tests 0.2 y Antiguo a train 0.8

# Columnas que no son features (no entran en X)
NO_FEATURE_COLUMN = ["Date", "target_binary", "target_multiclass"]

# XGBoost requiere clases que empiecen en 0, pero nuestro target
# multiclase usa (-1, 0, 1). Este mapa convierte y decodifica.
MULTICLASS_ENCODE = {-1: 0, 0: 1, 1: 2}
MULTICLASS_DECODE = {0: -1, 1: 0, 2: 1}

# Definición de modelos con sus hiperparámetros
# Se definen dos configuraciones por algoritmo para cumplir con el
# checklist del proyecto (mínimo 2 configuraciones comparadas)
MODELS = {
    "lr": LogisticRegression(  # regularización estándar
        max_iter=1000,
        random_state=42,
        C=1.0               
    ),
    "lr_strong_reg": LogisticRegression( # regularización más fuerte
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

#Separa el dataset en variables predictoras x, y targets y
#Se excluye Date y las columnas objetivo para que no entren como features.
def prepare_xy(df: pd.DataFrame):
    X = df.drop(columns=NO_FEATURE_COLUMN)
    y_binary = df["target_binary"]
    y_multiclass = df["target_multiclass"]
    

    print(f"\nFeatures: {X.shape[1]} columnas")
    print(f"Target binario     — clases: {sorted(y_binary.unique())}")
    print(f"Target multiclase  — clases: {sorted(y_multiclass.unique())}")

    return X, y_binary, y_multiclass

#Divide los datos respetando el orden cronológico, no split aleatorio.
def temporal_split(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE):
    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test

#StandardScaler pars las features
#Solo ajuste sobre tarin para evitar data leakage y despues aplicar esa escala a test
def scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

#Entrenamiendo de los modelos, tenemos 5 x 2 targets
def train_models(X_train, y_binary_train, y_multiclass_train):
    trained = {}

    for model_name, model in MODELS.items():
        print(f"\nEntrenando {model_name}...")

        # Clonar el modelo para cada target y así evita que compartan estado
        # Target binario
        m_binary = copy.deepcopy(model)
        m_binary.fit(X_train, y_binary_train)
        trained[f"{model_name}_binary"] = m_binary
        print(f"  {model_name}_binary     -> OK")

        # Target multiclase
        # XGBoost no acepta clases negativas "-1", entonces las codificamos a 0,1,2
        y_multi_encoded = y_multiclass_train.map(MULTICLASS_ENCODE) if ("xgb" in model_name and "lgb" not in model_name) else y_multiclass_train
        m_multi = copy.deepcopy(model)
        m_multi.fit(X_train, y_multi_encoded)
        trained[f"{model_name}_multiclass"] = m_multi
        print(f"  {model_name}_multiclass -> OK")

    return trained

#Guarda archivos .pkl y una metadata en json
def save_models(trained: dict, scaler: StandardScaler, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)

    # Guardar modelos
    for name, model in trained.items():
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Guardado: {path}")

    # Guardar scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Guardado: {scaler_path}")

    # Guardar metadata
    metadata = {
        "modelos": list(trained.keys()),
        "features": PATH,
        "test_size": TEST_SIZE,
        "NO_FEATURE_COLUMN": NO_FEATURE_COLUMN,
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

    split_idx = int(len(X) * (1 - TEST_SIZE))
    print(f"\nSplit temporal:")
    print(f"  Train: {X_train.shape[0]} filas")
    print(f"  Test:  {X_test.shape[0]} filas")

    #Escalar
    X_train_scaled, X_test_scaled, scaler = scale(X_train, X_test)
    print(f"\nEscalado aplicado — scaler ajustado solo sobre train")

    #Entrenar modelos
    print("\n" + "=" * 60)
    print("Entrenando modelos...")
    print("=" * 60)
    trained = train_models(X_train_scaled, y_bin_train, y_multi_train)
    
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

    print("\n Entrenamiento completado.")

    #Valores devueltos para notebook
    return {
        "df": df,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_bin_train": y_bin_train,
        "y_bin_test": y_bin_test,
        "y_multi_train": y_multi_train,
        "y_multi_test": y_multi_test,
        "trained": trained,
        "scaler": scaler,
    }


if __name__ == "__main__":
    main()