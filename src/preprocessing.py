import pandas as pd


def convert_date(df):
    """
    Convierte la columna Date a formato fecha.

    Esto es importante porque más adelante el dataset se trabaja como
    serie temporal. Si Date se queda como texto, no podremos ordenar,
    filtrar o dividir correctamente por fechas.
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def rename_columns(df):
    """
    Renombra las columnas originales del dataset.

    El CSV original trae nombres largos con esta estructura:
    Gold_('Close', 'GC=F')
    DXY_('Close', 'DX-Y.NYB')
    VIX_('Close', '^VIX')
    TNX_('Close', '^TNX')

    Aquí los convertimos a nombres simples y consistentes.
    """
    df = df.copy()

    df = df.rename(columns={
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
        
        "GVZ_('Close', '^GVZ')": "gvz_close",
        "GVZ_('High', '^GVZ')": "gvz_high",
        "GVZ_('Low', '^GVZ')": "gvz_low",
        "GVZ_('Open', '^GVZ')": "gvz_open",
        "GVZ_('Volume', '^GVZ')": "gvz_volume",
        
        "Oil_('Close', 'CL=F')": "oil_close",
        "Oil_('High', 'CL=F')": "oil_high",
        "Oil_('Low', 'CL=F')": "oil_low",
        "Oil_('Open', 'CL=F')": "oil_open",
        "Oil_('Volume', 'CL=F')": "oil_volume",
        
        "SP500_('Close', '^GSPC')": "sp500_close",
        "SP500_('High', '^GSPC')": "sp500_high",
        "SP500_('Low', '^GSPC')": "sp500_low",
        "SP500_('Open', '^GSPC')": "sp500_open",
        "SP500_('Volume', '^GSPC')": "sp500_volume",
        
    })

    return df


def drop_uninformative_columns(df):
    """
    Elimina columnas de volumen.

    En este dataset los volúmenes no se van a usar como variables
    predictoras principales, por eso se eliminan en la limpieza inicial.

    errors="ignore" evita que el programa falle si alguna de estas columnas
    no existe en una versión futura del dataset.
    """
    df = df.copy()

    columns_to_drop = [
        "dxy_volume",
        "vix_volume",
        "tnx_volume",
        "sp500_volume",
        "gvz_volume",
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    return df


def remove_missing_values(df):
    """
    Elimina filas con valores nulos.

    En este proyecto se decidió eliminar los registros incompletos porque
    eran pocos y no compensaba imputarlos.
    """
    df = df.copy()

    df = df.dropna()

    return df


def clean_gold_data(input_path, output_path):
    """
    Ejecuta la limpieza inicial del dataset y genera un CSV limpio.

    Flujo:
    1. Cargar dataset original desde data/raw.
    2. Convertir Date a formato datetime.
    3. Renombrar columnas.
    4. Eliminar columnas poco útiles.
    5. Eliminar valores nulos.
    6. Guardar el resultado en data/processed.

    Este archivo limpio será la entrada para feature_engineering.py.
    """
    df = pd.read_csv(input_path)

    df = convert_date(df)
    df = rename_columns(df)
    df = drop_uninformative_columns(df)
    df = remove_missing_values(df)

    df.to_csv(output_path, index=False)

    print(f"Dataset limpio guardado en {output_path} | shape: {df.shape}")

    return df


if __name__ == "__main__":
    clean_gold_data(
        "data/raw/gold-macro-data.csv",
        "data/processed/gold-clean.csv"
    )