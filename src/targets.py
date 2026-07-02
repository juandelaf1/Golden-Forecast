"""
Módulo de creación de targets.

Este archivo contiene las funciones para construir las variables objetivo
que se usarán posteriormente en la fase de modelado.
"""

import numpy as np


def create_targets(df, threshold=0.002):
    """
    Crea dos variables objetivo:

    1. target_binary:
       - 1 si el oro sube al día siguiente.
       - 0 si el oro no sube.

    2. target_multiclass:
       - 1  = comprar
       - 0  = mantener
       - -1 = vender

    El target multiclase utiliza un umbral para distinguir movimientos
    relevantes de pequeñas variaciones diarias.
    """

    # Hacemos una copia para no modificar el DataFrame original
    df = df.copy()

    # Calculamos el retorno del oro al día siguiente.
    # Esta columna se usa para construir los targets,
    # pero NO debe usarse como variable predictora.
    df["gold_return_next_day"] = (
        (df["gold_close"].shift(-1) - df["gold_close"]) / df["gold_close"]
    )

    # Target binario:
    # 1 si el retorno futuro es positivo, 0 si no lo es.
    df["target_binary"] = np.where(
        df["gold_return_next_day"] > 0,
        1,
        0
    )

    # Target multiclase:
    # 1 si sube más que el umbral.
    # -1 si baja más que el umbral.
    # 0 si el movimiento queda entre ambos límites.
    df["target_multiclass"] = np.where(
        df["gold_return_next_day"] > threshold,
        1,
        np.where(
            df["gold_return_next_day"] < -threshold,
            -1,
            0
        )
    )

    return df