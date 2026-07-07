import os
import pickle
import pandas as pd
from datetime import datetime, timezone

"""
predict.py — Predicción diaria del movimiento del oro

Este script es el último eslabón del pipeline automático.
Carga los modelos y el scaler ya entrenados (archivos .pkl),
toma la última fila disponible del dataset de features,
y genera una predicción para el siguiente día de mercado.

No entrena nada nuevo. Solo usa lo que ya está guardado en models/.
"""

FEATURES_PATH    = "data/processed/gold-features.csv"
MODELS_DIR       = "models"
PREDICTIONS_LOG  = "data/processed/predictions.csv"

# Columnas que no son variables predictoras y hay que eliminar antes de pasar al modelo
NO_FEATURE_COLUMNS = ["Date", "target_binary", "target_multiclass"]

# Etiquetas legibles para las predicciones
BINARY_LABELS = {
    1: "SUBE",
    0: "BAJA"
}
MULTI_LABELS = {
    1:  "SUBE FUERTE (más del 0.5%)",
    0:  "MOVIMIENTO PLANO (menos del 0.5%)",
    -1: "BAJA FUERTE (más del 0.5%)"
}


def load_pickle(path: str):
    """Carga un archivo .pkl guardado con pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_next_day() -> dict:
    """
    Carga el dataset de features, toma la última fila disponible,
    aplica el scaler guardado y genera predicciones con los dos
    mejores modelos (binario y multiclase).

    Devuelve un diccionario con la fecha de referencia,
    las predicciones y la probabilidad de subida.
    """

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
    X_scaled = scaler.transform(X)

    # Cargar el mejor modelo binario (ganador de la evaluación)
    model_binary = load_pickle(f"{MODELS_DIR}/lr_strong_reg_binary.pkl")

    # Cargar el mejor modelo multiclase
    model_multi = load_pickle(f"{MODELS_DIR}/lr_strong_reg_multiclass.pkl")

    # Generar predicciones
    pred_binary     = int(model_binary.predict(X_scaled)[0])
    prob_sube       = round(float(model_binary.predict_proba(X_scaled)[0][1]), 4)
    pred_multiclass = int(model_multi.predict(X_scaled)[0])

    result = {
        "fecha_referencia":    str(fecha_referencia),
        "prediccion_binaria":  pred_binary,
        "señal_binaria":       BINARY_LABELS[pred_binary],
        "probabilidad_sube":   prob_sube,
        "prediccion_multiclase": pred_multiclass,
        "señal_multiclase":    MULTI_LABELS[pred_multiclass],
        "generado_en": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    # Mostrar resultado en consola (visible en el log de GitHub Actions)
    print("\n" + "=" * 55)
    print("Golden Forecast — Predicción del próximo día de mercado")
    print("=" * 55)
    print(f"  Datos de referencia : {result['fecha_referencia']}")
    print(f"  Señal binaria       : {result['señal_binaria']}  "
          f"(probabilidad de subida: {prob_sube * 100:.1f}%)")
    print(f"  Señal multiclase    : {result['señal_multiclase']}")
    print(f"  Generado en         : {result['generado_en']}")
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
            print(f"  Ya existe predicción para {result['fecha_referencia']}. No se duplica.")
            return
        row.to_csv(PREDICTIONS_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(PREDICTIONS_LOG, index=False)

    print(f"  Predicción guardada en {PREDICTIONS_LOG}\n")


if __name__ == "__main__":
    result = predict_next_day()
    save_prediction(result)
