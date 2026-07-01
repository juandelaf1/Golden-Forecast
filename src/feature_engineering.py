"""
Módulo de Feature Engineering.

Este archivo contiene funciones para crear nuevas variables
a partir de las columnas originales del dataset.

Las features se calculan usando información disponible en el día actual
o en días anteriores, evitando usar información futura.
"""


def create_daily_returns(df):
    """
    Crea los retornos diarios de Gold, DXY, VIX y TNX.
    """

    df = df.copy()

    # Retorno diario del oro
    df["gold_return"] = df["gold_close"].pct_change()

    # Retorno diario del índice dólar
    df["dxy_return"] = df["dxy_close"].pct_change()

    # Retorno diario del VIX
    df["vix_return"] = df["vix_close"].pct_change()

    # Retorno diario del TNX
    df["tnx_return"] = df["tnx_close"].pct_change()

    return df


def create_daily_ranges(df):
    """
    Crea el rango diario de cada activo/indicador.

    Fórmula:
    rango = máximo del día - mínimo del día
    """

    df = df.copy()

    df["gold_range"] = df["gold_high"] - df["gold_low"]
    df["dxy_range"] = df["dxy_high"] - df["dxy_low"]
    df["vix_range"] = df["vix_high"] - df["vix_low"]
    df["tnx_range"] = df["tnx_high"] - df["tnx_low"]

    return df


def create_open_close_returns(df):
    """
    Crea la variación entre apertura y cierre del mismo día.

    Fórmula:
    retorno apertura-cierre = (cierre - apertura) / apertura
    """

    df = df.copy()

    df["gold_open_close_return"] = (
        (df["gold_close"] - df["gold_open"]) / df["gold_open"]
    )

    df["dxy_open_close_return"] = (
        (df["dxy_close"] - df["dxy_open"]) / df["dxy_open"]
    )

    df["vix_open_close_return"] = (
        (df["vix_close"] - df["vix_open"]) / df["vix_open"]
    )

    df["tnx_open_close_return"] = (
        (df["tnx_close"] - df["tnx_open"]) / df["tnx_open"]
    )

    return df


def create_moving_averages(df):
    """
    Crea medias móviles del precio de cierre del oro.

    Se usan:
    - 5 días para tendencia corta.
    - 20 días para tendencia más amplia.
    """

    df = df.copy()

    # Medias móviles del oro
    df["gold_ma_5"] = df["gold_close"].rolling(window=5).mean()
    df["gold_ma_20"] = df["gold_close"].rolling(window=20).mean()

    # Comparación del precio actual frente a sus medias móviles
    df["gold_close_vs_ma_5"] = (
        (df["gold_close"] - df["gold_ma_5"]) / df["gold_ma_5"]
    )

    df["gold_close_vs_ma_20"] = (
        (df["gold_close"] - df["gold_ma_20"]) / df["gold_ma_20"]
    )

    return df


def create_features(df):
    """
    Aplica todo el bloque de Feature Engineering en orden.

    Esta función centraliza la creación de features para que el notebook
    o futuros scripts puedan reutilizar el mismo proceso fácilmente.
    """

    df = df.copy()

    df = create_daily_returns(df)
    df = create_daily_ranges(df)
    df = create_open_close_returns(df)
    df = create_moving_averages(df)

    return df