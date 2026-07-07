import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Para entornos sin display

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DIR = "models"
PATH = "data/processed/gold-features.csv"
NO_FEATURE_COLUMN = ["Date", "target_binary", "target_multiclass"]
TEST_SIZE = 0.2

# Mapa de decodificación para predicciones XGBoost multiclase
MULTICLASS_DECODE = {0: -1, 1: 0, 2: 1}
MULTICLASS_ENCODE = {-1: 0, 0: 1, 1: 2}


# Datos de entrenamiento
def load_metadata(DIR: str) -> dict:
    path = os.path.join(DIR, "train_metadata.json")
    with open(path) as f:
        return json.load(f)


# Datos de modelo
def load_model(name: str, DIR: str):
    path = os.path.join(DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# Datos de todos los modelos
def load_all_models(DIR: str) -> dict:
    metadata = load_metadata(DIR)
    models = {}
    for name in metadata["modelos"]:
        models[name] = load_model(name, DIR)
        print(f"Cargado: {name}")
    return models


# StandardScaler ajustado sobre train
def load_scaler(DIR: str):
    path = os.path.join(DIR, "scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_data(PATH: str, test_size: float = TEST_SIZE):
    df = pd.read_csv(PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    X = df.drop(columns=NO_FEATURE_COLUMN)
    y_binary = df["target_binary"]
    y_multiclass = df["target_multiclass"]
    dates = df["Date"]

    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_bin_train = y_binary.iloc[:split_idx]
    y_bin_test = y_binary.iloc[split_idx:]
    y_multi_train = y_multiclass.iloc[:split_idx]
    y_multi_test = y_multiclass.iloc[split_idx:]
    dates_test = dates.iloc[split_idx:]

    return (
        X_train,
        X_test,
        y_bin_train,
        y_bin_test,
        y_multi_train,
        y_multi_test,
        dates_test,
    )


def evaluate_model(model, model_name: str, X_train, X_test, y_train, y_test) -> dict:
    is_xgb_multi = "xgb" in model_name and "multiclass" in model_name

    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Si es XGBoost multiclase, decodificar predicciones (0,1,2) -> (-1,0,1)
    if is_xgb_multi:
        y_pred_train = np.array([MULTICLASS_DECODE[p] for p in y_pred_train])
        y_pred_test = np.array([MULTICLASS_DECODE[p] for p in y_pred_test])
        y_train = y_train.map(MULTICLASS_ENCODE)
        y_test_eval = y_test.map(MULTICLASS_ENCODE)
    else:
        y_test_eval = y_test

    avg = "binary" if "binary" in model_name else "weighted"

    metrics = {
        "modelo": model_name,
        "accuracy_train": round(accuracy_score(y_train, y_pred_train), 4),
        "accuracy_test": round(accuracy_score(y_test_eval, y_pred_test), 4),
        "precision_test": round(
            precision_score(y_test_eval, y_pred_test, average=avg, zero_division=0), 4
        ),
        "recall_test": round(
            recall_score(y_test_eval, y_pred_test, average=avg, zero_division=0), 4
        ),
        "f1_test": round(
            f1_score(y_test_eval, y_pred_test, average=avg, zero_division=0), 4
        ),
        "overfitting_gap": round(
            accuracy_score(y_train, y_pred_train)
            - accuracy_score(y_test_eval, y_pred_test),
            4,
        ),
    }

    return metrics


# Check overfitting
def check_overfitting(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["overfitting"] = df["overfitting_gap"].apply(
        lambda x: "Posible" if x > 0.05 else "OK"
    )
    return df[
        ["modelo", "accuracy_train", "accuracy_test", "overfitting_gap", "overfitting"]
    ]


def compare_models(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df = df.sort_values("f1_test", ascending=False)
    return df[["modelo", "accuracy_test", "precision_test", "recall_test", "f1_test"]]


# Backtesting
def backtest(
    model, model_name: str, X_test, y_multi_test, dates_test, scaler=None
) -> dict:
    is_xgb = "xgb" in model_name

    y_pred = model.predict(X_test)

    if is_xgb:
        y_pred = np.array([MULTICLASS_DECODE[p] for p in y_pred])

    # Cargar gold_close del periodo test para calcular retornos reales
    df_raw = pd.read_csv(PATH, parse_dates=["Date"])
    df_raw = df_raw.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(df_raw) * (1 - TEST_SIZE))
    df_test = df_raw.iloc[split_idx:].copy().reset_index(drop=True)

    # Usamos gold_return como proxy del retorno real del día
    df_test["signal"] = y_pred
    df_test["strategy_return"] = df_test["gold_return"] * (
        df_test["signal"] == 1
    ).astype(int)

    # Retorno acumulado
    df_test["cum_strategy"] = (1 + df_test["strategy_return"]).cumprod()
    df_test["cum_buyhold"] = (1 + df_test["gold_return"]).cumprod()

    total_strategy = df_test["cum_strategy"].iloc[-1] - 1
    total_buyhold = df_test["cum_buyhold"].iloc[-1] - 1

    n_trades = (df_test["signal"] == 1).sum()
    n_days = len(df_test)

    return {
        "modelo": model_name,
        "retorno_estrategia": round(total_strategy * 100, 2),
        "retorno_buyhold": round(total_buyhold * 100, 2),
        "num_operaciones": int(n_trades),
        "dias_test": int(n_days),
        "df_test": df_test,
    }


def plot_backtest(bt_result: dict, save_path: str = None):
    """Genera el gráfico de retorno acumulado estrategia vs buy & hold."""
    df = bt_result["df_test"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df["Date"], df["cum_strategy"], label="Estrategia modelo", color="goldenrod"
    )
    ax.plot(
        df["Date"],
        df["cum_buyhold"],
        label="Buy & Hold",
        color="steelblue",
        linestyle="--",
    )
    ax.set_title(
        f"Backtesting — {bt_result['modelo']}\n"
        f"Estrategia: {bt_result['retorno_estrategia']}% | "
        f"Buy & Hold: {bt_result['retorno_buyhold']}%"
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Retorno acumulado")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    print("=" * 60)
    print("Golden-Forecast — Evaluación de modelos")
    print("=" * 60)

    # 1. Cargar modelos y datos
    models = load_all_models(DIR)
    scaler = load_scaler(DIR)

    (
        X_train,
        X_test,
        y_bin_train,
        y_bin_test,
        y_multi_train,
        y_multi_test,
        dates_test,
    ) = prepare_data(PATH)

    # Escalar con el mismo scaler ajustado en train
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(
        f"\nPeriodo de test: {dates_test.iloc[0].date()} -> {dates_test.iloc[-1].date()}"
    )
    print(f"Filas test: {len(X_test)}")

    # 2. Evaluar todos los modelos
    print("\n" + "=" * 60)
    print("Evaluando modelos...")
    print("=" * 60)

    results = []
    for name, model in models.items():
        y_train = y_bin_train if "binary" in name else y_multi_train
        y_test = y_bin_test if "binary" in name else y_multi_test
        metrics = evaluate_model(model, name, X_train_sc, X_test_sc, y_train, y_test)
        results.append(metrics)
        print(
            f"  {name}: accuracy_test={metrics['accuracy_test']} | f1={metrics['f1_test']}"
        )

    # 3. Tabla comparativa
    print("\n" + "=" * 60)
    print("TABLA COMPARATIVA — Test")
    print("=" * 60)
    df_compare = compare_models(results)
    print(df_compare.to_string(index=False))

    # 4. Chequeo de overfitting
    print("\n" + "=" * 60)
    print("CHEQUEO DE OVERFITTING")
    print("=" * 60)
    df_overfit = check_overfitting(results)
    print(df_overfit.to_string(index=False))

    # 5. Backtesting sobre el mejor modelo multiclase (por F1)
    print("\n" + "=" * 60)
    print("BACKTESTING")
    print("=" * 60)

    multi_results = [r for r in results if "multiclass" in r["modelo"]]
    best_multi = max(multi_results, key=lambda x: x["f1_test"])
    best_model_name = best_multi["modelo"]
    best_model = models[best_model_name]

    print(f"Modelo seleccionado para backtesting: {best_model_name}")

    bt = backtest(best_model, best_model_name, X_test_sc, y_multi_test, dates_test)

    print(f"\n  Retorno estrategia: {bt['retorno_estrategia']}%")
    print(f"  Retorno buy & hold: {bt['retorno_buyhold']}%")
    print(
        f"  Operaciones realizadas: {bt['num_operaciones']} de {bt['dias_test']} días"
    )

    os.makedirs("docs", exist_ok=True)
    plot_backtest(bt, save_path="docs/backtest.png")

    # 6. Guardar resultados
    output = {
        "resultados": results,
        "mejor_multiclase": best_model_name,
        "backtesting": {
            "modelo": bt["modelo"],
            "retorno_estrategia_pct": bt["retorno_estrategia"],
            "retorno_buyhold_pct": bt["retorno_buyhold"],
            "num_operaciones": bt["num_operaciones"],
        },
    }

    out_path = os.path.join(DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")
    print("\n Evaluación completada.")


if __name__ == "__main__":
    main()
