"""
Módulo de pipeline y preparación para modelado.

Incluye funciones para:
- Separar variables predictoras y targets.
- Hacer split temporal train/test.
- Escalar variables numéricas evitando data leakage.
- Crear un pipeline básico de preprocesamiento.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def create_modeling_data(df):
    """
    Separa el dataset en variables predictoras y targets.

    Excluimos:
    - Date: se usa para ordenar, no como variable predictora.
    - gold_return_next_day: contiene información futura.
    - target_binary: variable objetivo.
    - target_multiclass: variable objetivo.
    """

    columns_to_exclude = [
        "Date",
        "gold_return_next_day",
        "target_binary",
        "target_multiclass",
    ]

    # X contiene solo variables predictoras
    X = df.drop(columns=columns_to_exclude)

    # Targets separados
    y_binary = df["target_binary"]
    y_multiclass = df["target_multiclass"]

    return X, y_binary, y_multiclass


def temporal_train_test_split(X, y, test_size=0.2):
    """
    Divide los datos respetando el orden temporal.

    No usamos split aleatorio porque trabajamos con series temporales.
    El train contiene los datos antiguos y el test los datos recientes.
    """

    # Calculamos el índice donde empieza test
    split_index = int(len(X) * (1 - test_size))

    # División temporal
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Escala las variables numéricas evitando data leakage.

    El scaler aprende medias y desviaciones solo con train.
    Después transforma train y test con esos mismos parámetros.
    """

    # Creamos el escalador
    scaler = StandardScaler()

    # Aprende parámetros con train y escala train
    X_train_scaled = scaler.fit_transform(X_train)

    # Escala test con los parámetros aprendidos en train
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def build_preprocessing_pipeline():
    """
    Crea un pipeline básico de preprocesamiento.

    En este proyecto el pipeline incluye StandardScaler.
    """

    preprocessing_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    return preprocessing_pipeline