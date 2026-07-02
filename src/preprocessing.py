"""
Módulo de preprocesamiento básico.

Este archivo contiene funciones reutilizables para preparar el dataset
antes de crear features, targets o entrenar modelos.

Incluye:
- Conversión de fechas.
- Renombrado de columnas.
- Eliminación de columnas poco informativas.
- Tratamiento de valores nulos.
"""

import pandas as pd


def convert_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la columna Date a formato datetime.

    Recibe:
        df: DataFrame original.

    Devuelve:
        DataFrame con Date convertida a tipo fecha.
    """

    # Hacemos una copia para no modificar el DataFrame original directamente
    df = df.copy()

    # Convertimos la columna Date de texto a formato fecha
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra las columnas descargadas desde yfinance.

    Los nombres originales son largos y difíciles de manejar.
    Aquí los convertimos a nombres simples y consistentes.
    """

    # Hacemos una copia para evitar modificar el DataFrame original
    df = df.copy()

    # Diccionario con nombres originales y nombres nuevos
    column_mapping = {
        "Gold_('Close', 'GC=F')": "gold_close",
        "Gold_('High', 'GC=F')": "gold_high",
        "Gold_('Low', 'GC=F')": "gold_low",
        "Gold_('Open', 'GC=F')": "gold_open",
        "Gold_('Volume', 'GC=F')": "gold_volume",

        "DXY_('Close', 'DX-Y.NYB')": "dxy_close",
        "DXY_('High', 'DX-Y.NYB')": "dxy_high",
        "DXY_('Low', 'DX-Y.NYB')": "dxy_low",
        "DXY_('Open', 'DX-Y.NYB')": "dxy_open",
        "DXY_('Volume', 'DX-Y.NYB')": "dxy_volume",

        "VIX_('Close', '^VIX')": "vix_close",
        "VIX_('High', '^VIX')": "vix_high",
        "VIX_('Low', '^VIX')": "vix_low",
        "VIX_('Open', '^VIX')": "vix_open",
        "VIX_('Volume', '^VIX')": "vix_volume",

        "TNX_('Close', '^TNX')": "tnx_close",
        "TNX_('High', '^TNX')": "tnx_high",
        "TNX_('Low', '^TNX')": "tnx_low",
        "TNX_('Open', '^TNX')": "tnx_open",
        "TNX_('Volume', '^TNX')": "tnx_volume",
    }

    # Aplicamos el renombrado
    df = df.rename(columns=column_mapping)

    return df


def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas poco informativas para el modelo.

    En este dataset mantenemos gold_volume, pero eliminamos volúmenes
    de índices macro que suelen venir con ceros o aportar poca información.
    """

    # Hacemos una copia del DataFrame
    df = df.copy()

    # Columnas que decidimos eliminar
    columns_to_drop = [
        "dxy_volume",
        "vix_volume",
        "tnx_volume",
    ]

    # Eliminamos solo las columnas que existan en el DataFrame
    df = df.drop(columns=columns_to_drop, errors="ignore")

    return df


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con valores nulos.

    Esta función se puede usar:
    - después de cargar y revisar el dataset original;
    - después del feature engineering, cuando se generan nulos naturales
      por retornos, medias móviles o target futuro.
    """

    # Hacemos una copia para no modificar el DataFrame original
    df = df.copy()

    # Eliminamos cualquier fila que tenga al menos un valor nulo
    df = df.dropna()

    return df