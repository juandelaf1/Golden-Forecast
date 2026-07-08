import os
import json
import pickle
import pandas as pd
from datetime import datetime, timezone

"""
predict.py — Predicción diaria del movimiento del oro

Este script es el último eslabón del pipeline automático.
Lee evaluation_results.json para seleccionar automáticamente los mejores
modelos entrenados, toma la última fila disponible del dataset de features,
y genera una predicción para el siguiente día de mercado.

No entrena nada nuevo. Solo usa lo que ya está guardado en models/.
Si Joel reentrena y actualiza evaluation_results.json, este script
automáticamente usará los nuevos modelos ganadores.
"""

FEATURES_PATH = "data/processed/gold-features.csv"
MODELS_DIR = "models"
EVALUATION_PATH = "models/evaluation_results.json"
PREDICTIONS_LOG = "data/processed/predictions.csv"

# Columnas que no son variables predictoras y hay que eliminar antes de pasar al modelo
NO_FEATURE_COLUMNS = ["Date", "target_binary", "target_multiclass"]

# Etiquetas legibles para las predicciones
BINARY_LABELS = {1: "SUBE", 0: "BAJA"}
MULTI_LABELS = {
    1: "SUBE FUERTE (más del 0.5%)",
    0: "MOVIMIENTO PLANO (menos del 0.5%)",
    -1: "BAJA FUERTE (más del 0.5%)",
}


def load_pickle(path: str):
    """Carga un archivo .pkl guardado con pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_best_model_names() -> tuple[str, str]:
    """
    Lee evaluation_results.json y devuelve los nombres de los mejores modelos.

    Para el modelo binario: busca el mayor f1_test entre todos los modelos
    cuyo nombre contiene 'binary'.

    Para el modelo multiclase: usa el campo 'mejor_multiclase' que ya
    calculó evaluate.py, para no duplicar esa lógica aquí.

    Así, si Joel reentrena y actualiza el JSON, este script se adapta solo.
    """
    with open(EVALUATION_PATH) as f:
        data = json.load(f)

    # Mejor modelo binario: el que tenga mayor f1_test entre los modelos binarios
    binary_results = [r for r in data["resultados"] if "binary" in r["modelo"]]
    best_binary = max(binary_results, key=lambda x: x["f1_test"])["modelo"]

    # Mejor modelo multiclase: ya lo calculó evaluate.py y lo guardó en el JSON
    best_multiclass = data["mejor_multiclase"]

    print(f"  Modelo binario seleccionado    : {best_binary}")
    print(f"  Modelo multiclase seleccionado : {best_multiclass}")

    return best_binary, best_multiclass


def predict_next_day() -> dict:
    """
    Carga el dataset de features, toma la última fila disponible,
    aplica el scaler guardado y genera predicciones con los dos
    mejores modelos (binario y multiclase).

    Devuelve un diccionario con la fecha de referencia,
    las predicciones, la probabilidad de subida y los modelos usados.
    """

    # Leer los nombres de los modelos ganadores desde el JSON de evaluación
    best_binary_name, best_multi_name = load_best_model_names()

    # Cargar el dataset completo de features
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Tomar solo la última fila: el día más reciente con datos completos
    latest_row = df.iloc[[-1]]
    fecha_referencia = pd.Timestamp(latest_row["Date"].values[0]).date()

    # Separar las variables predictoras (eliminar Date y targets)
    X = latest_row.drop(
        columns=[c for c in NO_FEATURE_COLUMNS if c in latest_row.columns]
    )

    # Cargar el scaler y transformar (NUNCA reajustar — debe ser el mismo que se usó en train.py)
    scaler = load_pickle(f"{MODELS_DIR}/scaler.pkl")
    try:
        X_scaled = scaler.transform(X)
    except ValueError:
        # El scaler fue entrenado con columnas distintas a las actuales.
        # Usamos solo las columnas que el scaler conoce (feature_names_in_)
        # para evitar que el pipeline falle tras un reentrenamiento parcial.
        X_scaled = scaler.transform(X[scaler.feature_names_in_])

    # Cargar los modelos ganadores usando los nombres leídos del JSON
    model_binary = load_pickle(f"{MODELS_DIR}/{best_binary_name}.pkl")
    model_multi = load_pickle(f"{MODELS_DIR}/{best_multi_name}.pkl")

    # Generar predicciones
    pred_binary = int(model_binary.predict(X_scaled)[0])
    prob_sube = round(float(model_binary.predict_proba(X_scaled)[0][1]), 4)
    pred_multiclass = int(model_multi.predict(X_scaled)[0])

    result = {
        "fecha_referencia": str(fecha_referencia),
        "prediccion_binaria": pred_binary,
        "señal_binaria": BINARY_LABELS[pred_binary],
        "probabilidad_sube": prob_sube,
        "prediccion_multiclase": pred_multiclass,
        "señal_multiclase": MULTI_LABELS[pred_multiclass],
        "modelo_binario_usado": best_binary_name,
        "modelo_multiclase_usado": best_multi_name,
        "generado_en": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    # Mostrar resultado en consola (visible en el log de GitHub Actions)
    print("\n" + "=" * 55)
    print("Golden Forecast — Predicción del próximo día de mercado")
    print("=" * 55)
    print(f"  Datos de referencia : {result['fecha_referencia']}")
    print(
        f"  Señal binaria       : {result['señal_binaria']}  "
        f"(probabilidad de subida: {prob_sube * 100:.1f}%)"
    )
    print(f"  Señal multiclase    : {result['señal_multiclase']}")
    print("=" * 55)

    return result


def save_prediction(result: dict) -> None:
    """
    Añade la predicción al archivo de log CSV.
    Si ya existe una predicción para esa fecha, no la duplica.
    Si el archivo no existe todavía, lo crea con cabecera.
    """
    row = pd.DataFrame([result])

    if os.path.exists(PREDICTIONS_LOG):
        existing = pd.read_csv(PREDICTIONS_LOG)
        if result["fecha_referencia"] in existing["fecha_referencia"].values:
            print(
                f"  Ya existe predicción para {result['fecha_referencia']}. No se duplica."
            )
            return
        row.to_csv(PREDICTIONS_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(PREDICTIONS_LOG, index=False)

    print(f"  Predicción guardada en {PREDICTIONS_LOG}\n")


if __name__ == "__main__":
    result = predict_next_day()
    save_prediction(result)
