"""Preparación de datos compartida por train.py y predict.py.

Contiene las mismas reglas del notebook de la Fase 1 para el modelo reducido.
Al ir dentro del Pipeline, estas reglas se guardan junto con el modelo.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Columnas que debe traer el CSV de entrada (sin la variable objetivo).
COLUMNAS_ENTRADA = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]

# Columnas que recibe el modelo, en este orden.
COLUMNAS_MODELO = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "median_income",
    "ocean_proximity",
    "rooms_per_household",
    "population_per_household",
]


def validar_columnas(df, columnas_requeridas):
    """Lanza un error claro si falta alguna columna requerida."""
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")


class PreparadorDatos(BaseEstimator, TransformerMixin):
    """Convierte las columnas originales del CSV en las que usa el modelo."""

    def fit(self, X, y=None):
        # Lo único que se aprende: las categorías vistas en entrenamiento.
        self.categorias_ = sorted(X["ocean_proximity"].dropna().unique())
        return self

    def transform(self, X):
        validar_columnas(X, COLUMNAS_ENTRADA)

        filas_sin_hogares = X.index[X["households"] == 0].tolist()
        if filas_sin_hogares:
            raise ValueError(
                f"households no puede ser 0 (filas: {filas_sin_hogares})"
            )

        datos = X.copy()
        datos["rooms_per_household"] = datos["total_rooms"] / datos["households"]
        datos["population_per_household"] = datos["population"] / datos["households"]

        # Una categoría no vista queda como faltante; LightGBM admite faltantes.
        datos["ocean_proximity"] = pd.Categorical(
            datos["ocean_proximity"], categories=self.categorias_
        )
        return datos[COLUMNAS_MODELO]
