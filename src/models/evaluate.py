import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DIR = "models"
PATH = "data/processed/gold-features.csv"
TEST_SIZE = 0.2

NO_FEATURE_COLUMN = [
    "Date",
    "target_binary",
    "target_multiclass",
    "gold_return",
    "gold_return_next_day",
]

MULTICLASS_DECODE = {0: -1, 1: 0, 2: 1}
MULTICLASS_ENCODE = {-1: 0, 0: 1, 1: 2}


def load_metadata(models_dir: str) -> dict:
    path = os.path.join(models_dir, "train_metadata.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(name: str, models_dir: str):
    path = os.path.join(models_dir, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_models(models_dir: str) -> dict:
    metadata = load_metadata(models_dir)
    models = {}

    modelos_meta = metadata.get("modelos", [])
    if modelos_meta and isinstance(modelos_meta[0], dict):
        iterable = modelos_meta
    else:
        iterable = [{"name": name, "uses_scaled": name.startswith("lr")} for name in modelos_meta]

    for item in iterable:
        name = item["name"]
        models[name] = {
            "model": load_model(name, models_dir),
            "uses_scaled": bool(item.get("uses_scaled", name.startswith("lr"))),
        }
        print(f"Cargado: {name}")

    return models


def load_scaler(models_dir: str):
    path = os.path.join(models_dir, "scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_columns(df: pd.DataFrame, metadata: dict):
    if "feature_columns" in metadata:
        return metadata["feature_columns"]
    return [col for col in df.columns if col not in NO_FEATURE_COLUMN]


def prepare_data(path: str, metadata: dict, test_size: float = TEST_SIZE):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feature_cols = get_feature_columns(df, metadata)
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Faltan features en el dataset: {missing_features}")

    X = df[feature_cols].copy()
    y_binary = df["target_binary"].copy()
    y_multiclass = df["target_multiclass"].copy()
    dates = df["Date"].copy()

    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_bin_train = y_binary.iloc[:split_idx].copy()
    y_bin_test = y_binary.iloc[split_idx:].copy()
    y_multi_train = y_multiclass.iloc[:split_idx].copy()
    y_multi_test = y_multiclass.iloc[split_idx:].copy()
    dates_test = dates.iloc[split_idx:].copy()

    return (
        df,
        feature_cols,
        X_train,
        X_test,
        y_bin_train,
        y_bin_test,
        y_multi_train,
        y_multi_test,
        dates_test,
    )


def evaluate_model(model, model_name: str, X_train, X_test, y_train, y_test) -> dict:
    is_xgb_multi = "xgb" in model_name and "multiclass" in model_name and "lgb" not in model_name

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if is_xgb_multi:
        y_pred_train = np.array([MULTICLASS_DECODE[p] for p in y_pred_train])
        y_pred_test = np.array([MULTICLASS_DECODE[p] for p in y_pred_test])

    avg = "binary" if "binary" in model_name else "weighted"

    return {
        "modelo": model_name,
        "accuracy_train": round(accuracy_score(y_train, y_pred_train), 4),
        "accuracy_test": round(accuracy_score(y_test, y_pred_test), 4),
        "precision_test": round(precision_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
        "recall_test": round(recall_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
        "f1_test": round(f1_score(y_test, y_pred_test, average=avg, zero_division=0), 4),
        "overfitting_gap": round(accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test), 4),
    }


def check_overfitting(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["overfitting"] = df["overfitting_gap"].apply(lambda x: "Posible" if x > 0.05 else "OK")
    return df[["modelo", "accuracy_train", "accuracy_test", "overfitting_gap", "overfitting"]]


def compare_models(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df = df.sort_values("f1_test", ascending=False)
    return df[["modelo", "accuracy_test", "precision_test", "recall_test", "f1_test"]]


def get_test_context(df_full: pd.DataFrame, test_size: float = TEST_SIZE):
    split_idx = int(len(df_full) * (1 - test_size))
    return df_full.iloc[split_idx:].copy().reset_index(drop=True)


def backtest_binary(model, model_name, X_test, df_test_context, threshold=0.56):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    df_test = df_test_context.copy()
    df_test["signal"] = y_pred
    df_test["strategy_return"] = df_test["gold_return"] * df_test["signal"]
    df_test["cum_strategy"] = (1 + df_test["strategy_return"]).cumprod()
    df_test["cum_buyhold"] = (1 + df_test["gold_return"]).cumprod()

    return {
        "modelo": model_name,
        "threshold": threshold,
        "retorno_estrategia": round((df_test["cum_strategy"].iloc[-1] - 1) * 100, 2),
        "retorno_buyhold": round((df_test["cum_buyhold"].iloc[-1] - 1) * 100, 2),
        "num_operaciones": int(df_test["signal"].sum()),
        "dias_test": len(df_test),
        "df_test": df_test,
    }


def backtest_multiclass(model, model_name: str, X_test, df_test_context) -> dict:
    is_xgb = "xgb" in model_name and "lgb" not in model_name

    y_pred = model.predict(X_test)
    if is_xgb:
        y_pred = np.array([MULTICLASS_DECODE[p] for p in y_pred])

    df_test = df_test_context.copy()
    df_test["signal"] = y_pred
    df_test["strategy_return"] = df_test["gold_return"] * (df_test["signal"] == 1).astype(int)
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
    df = bt_result["df_test"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["cum_strategy"], label="Estrategia modelo", color="goldenrod")
    ax.plot(df["Date"], df["cum_buyhold"], label="Buy & Hold", color="steelblue", linestyle="--")
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


def plot_walk_forward(df_wf: pd.DataFrame, save_path: str = "docs/walk_forward.png"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].bar(df_wf["ventana"], df_wf["accuracy"], color="goldenrod", alpha=0.8)
    axes[0].axhline(0.5, color="red", linestyle="--", label="Baseline (50%)")
    axes[0].axhline(df_wf["accuracy"].mean(), color="steelblue", linestyle="--", label=f"Media ({df_wf['accuracy'].mean():.2%})")
    axes[0].set_title("Accuracy por ventana temporal")
    axes[0].set_xlabel("Ventana")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0.4, 0.7)

    x = df_wf["ventana"]
    width = 0.35
    axes[1].bar(x - width / 2, df_wf["retorno_estrategia_pct"], width, label="Estrategia", color="goldenrod", alpha=0.8)
    axes[1].bar(x + width / 2, df_wf["retorno_buyhold_pct"], width, label="Buy & Hold", color="steelblue", alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Retorno acumulado por ventana — Estrategia vs Buy & Hold")
    axes[1].set_xlabel("Ventana")
    axes[1].set_ylabel("Retorno (%)")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nGráfico walk-forward guardado en: {save_path}")
    plt.close()


def walk_forward_validation(df_path: str, metadata: dict, n_splits: int = 5):
    df = pd.read_csv(df_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    feature_cols = get_feature_columns(df, metadata)
    X = df[feature_cols].copy()
    y_binary = df["target_binary"]

    total = len(df)
    fold_size = total // (n_splits + 1)

    results_wf = []

    print(f"\nWalk-Forward Validation — {n_splits} ventanas")
    print("=" * 60)

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = test_start + fold_size

        X_train_wf = X.iloc[:train_end]
        X_test_wf = X.iloc[test_start:test_end]
        y_bin_train_wf = y_binary.iloc[:train_end]
        y_bin_test_wf = y_binary.iloc[test_start:test_end]
        dates_wf = df["Date"].iloc[test_start:test_end]
        returns_wf = df["gold_return"].iloc[test_start:test_end]

        scaler_wf = load_scaler(DIR)
        X_train_sc = scaler_wf.fit_transform(X_train_wf)
        X_test_sc = scaler_wf.transform(X_test_wf)

        model_bin = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        model_bin.fit(X_train_sc, y_bin_train_wf)

        y_pred_bin = model_bin.predict(X_test_sc)
        acc = accuracy_score(y_bin_test_wf, y_pred_bin)
        f1 = f1_score(y_bin_test_wf, y_pred_bin, average="binary", zero_division=0)

        proba = model_bin.predict_proba(X_test_sc)[:, 1]
        umbral_compra = np.percentile(proba, 75)
        umbral_venta = np.percentile(proba, 25)
        signals = np.where(proba > umbral_compra, 1, np.where(proba < umbral_venta, -1, 0))

        strategy_returns = returns_wf.values * (signals == 1).astype(int)
        cum_strategy = (1 + strategy_returns).prod() - 1
        cum_buyhold = (1 + returns_wf.values).prod() - 1
        n_trades = (signals == 1).sum()

        date_start = dates_wf.iloc[0].date()
        date_end = dates_wf.iloc[-1].date()

        print(f"\nVentana {i + 1}: {date_start} -> {date_end}")
        print(f" Train: {train_end} filas | Test: {len(X_test_wf)} filas")
        print(f" Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f" Retorno estrategia: {cum_strategy * 100:.2f}%")
        print(f" Retorno buy & hold: {cum_buyhold * 100:.2f}%")
        print(f" Operaciones: {n_trades} de {len(X_test_wf)} días")

        results_wf.append({
            "ventana": i + 1,
            "fecha_inicio": str(date_start),
            "fecha_fin": str(date_end),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "retorno_estrategia_pct": round(cum_strategy * 100, 2),
            "retorno_buyhold_pct": round(cum_buyhold * 100, 2),
            "n_trades": int(n_trades),
            "n_dias": len(X_test_wf),
        })

    df_wf = pd.DataFrame(results_wf)

    print("\n" + "=" * 60)
    print("RESUMEN WALK-FORWARD")
    print("=" * 60)
    print(f" Accuracy medio: {df_wf['accuracy'].mean():.4f}")
    print(f" F1 medio: {df_wf['f1'].mean():.4f}")
    print(f" Retorno estrategia medio: {df_wf['retorno_estrategia_pct'].mean():.2f}%")
    print(f" Retorno buy&hold medio: {df_wf['retorno_buyhold_pct'].mean():.2f}%")
    print(f" Total operaciones: {df_wf['n_trades'].sum()}")

    plot_walk_forward(df_wf)
    return df_wf


def main():
    print("=" * 60)
    print("Golden-Forecast — Evaluación de modelos")
    print("=" * 60)

    metadata = load_metadata(DIR)
    models = load_all_models(DIR)
    scaler = load_scaler(DIR)

    (
        df_full,
        feature_cols,
        X_train,
        X_test,
        y_bin_train,
        y_bin_test,
        y_multi_train,
        y_multi_test,
        dates_test,
    ) = prepare_data(PATH, metadata)

    df_test_context = get_test_context(df_full)

    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"\nFeatures usadas en evaluación: {len(feature_cols)}")
    print(f"Periodo de test: {dates_test.iloc[0].date()} -> {dates_test.iloc[-1].date()}")
    print(f"Filas test: {len(X_test)}")

    print("\n" + "=" * 60)
    print("Evaluando modelos...")
    print("=" * 60)

    results = []
    for name, payload in models.items():
        model = payload["model"]
        uses_scaled = payload["uses_scaled"]

        y_train = y_bin_train if "binary" in name else y_multi_train
        y_test = y_bin_test if "binary" in name else y_multi_test

        X_train_used = X_train_sc if uses_scaled else X_train
        X_test_used = X_test_sc if uses_scaled else X_test

        metrics = evaluate_model(model, name, X_train_used, X_test_used, y_train, y_test)
        results.append(metrics)
        print(f" {name}: accuracy_test={metrics['accuracy_test']} | f1={metrics['f1_test']}")

    print("\n" + "=" * 60)
    print("TABLA COMPARATIVA — Test")
    print("=" * 60)
    df_compare = compare_models(results)
    print(df_compare.to_string(index=False))

    print("\n" + "=" * 60)
    print("CHEQUEO DE OVERFITTING")
    print("=" * 60)
    df_overfit = check_overfitting(results)
    print(df_overfit.to_string(index=False))

    print("\n" + "=" * 60)
    print("BACKTESTING")
    print("=" * 60)

    binary_results = [r for r in results if "binary" in r["modelo"]]
    best_binary = max(binary_results, key=lambda x: x["f1_test"])
    best_binary_name = best_binary["modelo"]
    best_binary_model = models[best_binary_name]["model"]
    best_binary_scaled = models[best_binary_name]["uses_scaled"]
    best_binary_X_test = X_test_sc if best_binary_scaled else X_test

    print(f"Modelo binario seleccionado: {best_binary_name}")
    bt_binary = backtest_binary(best_binary_model, best_binary_name, best_binary_X_test, df_test_context, threshold=0.56)
    print(f"Retorno estrategia binaria: {bt_binary['retorno_estrategia']}%")

    multi_results = [r for r in results if "multiclass" in r["modelo"]]
    best_multi = max(multi_results, key=lambda x: x["f1_test"])
    best_model_name = best_multi["modelo"]
    best_model = models[best_model_name]["model"]
    best_multi_scaled = models[best_model_name]["uses_scaled"]
    best_multi_X_test = X_test_sc if best_multi_scaled else X_test

    print(f"Modelo seleccionado para backtesting: {best_model_name}")
    bt = backtest_multiclass(best_model, best_model_name, best_multi_X_test, df_test_context)

    print(f"\n Retorno estrategia: {bt['retorno_estrategia']}%")
    print(f" Retorno buy & hold: {bt['retorno_buyhold']}%")
    print(f" Operaciones realizadas: {bt['num_operaciones']} de {bt['dias_test']} días")

    os.makedirs("docs", exist_ok=True)
    plot_backtest(bt, save_path="docs/backtest.png")

    output = {
        "resultados": results,
        "feature_columns": feature_cols,
        "mejor_multiclase": best_model_name,
        "backtesting": {
            "modelo": bt["modelo"],
            "retorno_estrategia_pct": bt["retorno_estrategia"],
            "retorno_buyhold_pct": bt["retorno_buyhold"],
            "num_operaciones": bt["num_operaciones"],
        },
        "backtesting_binario": {
            "modelo": bt_binary["modelo"],
            "threshold": bt_binary["threshold"],
            "retorno_estrategia_pct": bt_binary["retorno_estrategia"],
            "retorno_buyhold_pct": bt_binary["retorno_buyhold"],
            "num_operaciones": bt_binary["num_operaciones"],
        },
    }

    out_path = os.path.join(DIR, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en: {out_path}")

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    walk_forward_validation(PATH, metadata)

    print("\nEvaluación completada.")


if __name__ == "__main__":
    main()