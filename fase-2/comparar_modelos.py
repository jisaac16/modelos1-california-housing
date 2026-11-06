"""Compara las variantes de la Fase 1 usando solo datos de entrenamiento.

La partición de prueba es la misma de train.py y aquí no se usa: la elección del
modelo no depende de ella. Se ejecuta aparte; train.py no depende de este script.
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from train import calcular_zonas, cargar_datos, separar_por_zona

NUMERICAS_COMPLETAS = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income",
    "rooms_per_household", "bedrooms_per_household", "population_per_household",
]
NUMERICAS_REDUCIDAS = [
    "longitude", "latitude", "housing_median_age", "total_rooms", "median_income",
    "rooms_per_household", "population_per_household",
]


def agregar_razones(X):
    datos = X.copy()
    datos["rooms_per_household"] = datos["total_rooms"] / datos["households"]
    datos["bedrooms_per_household"] = datos["total_bedrooms"] / datos["households"]
    datos["population_per_household"] = datos["population"] / datos["households"]
    return datos


def seleccionar_para_lightgbm(X, numericas):
    datos = X[numericas + ["ocean_proximity"]].copy()
    datos["ocean_proximity"] = datos["ocean_proximity"].astype("category")
    return datos


def con_lightgbm(numericas, seed):
    return Pipeline([
        ("razones", FunctionTransformer(agregar_razones)),
        ("columnas", FunctionTransformer(seleccionar_para_lightgbm, kw_args={"numericas": numericas})),
        ("modelo", lgb.LGBMRegressor(random_state=seed, verbose=-1)),
    ])


def con_codificacion(numericas, modelo, escalar=False):
    pasos = [("imputar", SimpleImputer(strategy="mean"))]
    if escalar:
        pasos.append(("escalar", StandardScaler()))
    columnas = ColumnTransformer([
        ("numericas", Pipeline(pasos), numericas),
        ("categoria", OneHotEncoder(handle_unknown="ignore"), ["ocean_proximity"]),
    ])
    return Pipeline([
        ("razones", FunctionTransformer(agregar_razones)),
        ("columnas", columnas),
        ("modelo", modelo),
    ])


def construir_variantes(seed):
    def bosque():
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)

    return {
        "Dummy (base)": DummyRegressor(strategy="mean"),
        "LightGBM (Completo)": con_lightgbm(NUMERICAS_COMPLETAS, seed),
        "LightGBM (Reducido)": con_lightgbm(NUMERICAS_REDUCIDAS, seed),
        "Random Forest (Completo)": con_codificacion(NUMERICAS_COMPLETAS, bosque()),
        "Random Forest (Reducido)": con_codificacion(NUMERICAS_REDUCIDAS, bosque()),
        "SVM (Completo)": con_codificacion(NUMERICAS_COMPLETAS, SVR(kernel="rbf", C=1000), escalar=True),
        "SVM (Reducido)": con_codificacion(NUMERICAS_REDUCIDAS, SVR(kernel="rbf", C=1000), escalar=True),
    }


def comparar(X, y, seed, particiones=5):
    """Valida por zonas: cada zona queda completa en entrenamiento o en validación."""
    zonas = calcular_zonas(X)
    filas = []
    for nombre, modelo in construir_variantes(seed).items():
        puntajes = cross_validate(
            modelo, X, y, groups=zonas, cv=GroupKFold(n_splits=particiones),
            scoring="neg_mean_absolute_error",
        )
        errores = -puntajes["test_score"]
        filas.append({
            "variante": nombre,
            "mae_promedio": round(float(np.mean(errores)), 2),
            "mae_desviacion": round(float(np.std(errores)), 2),
            "mae_minimo": round(float(np.min(errores)), 2),
            "mae_maximo": round(float(np.max(errores)), 2),
        })
        print(f"{nombre}: MAE promedio {filas[-1]['mae_promedio']:,.2f}", flush=True)
    return pd.DataFrame(filas).sort_values("mae_promedio")


def main():
    parser = argparse.ArgumentParser(description="Comparar variantes del modelo sin usar la partición de prueba")
    parser.add_argument("--data", required=True, help="Ruta al archivo CSV de datos")
    parser.add_argument("--target", default="median_house_value", help="Nombre de la variable objetivo")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--test-size", type=float, default=0.20, help="Proporción reservada para prueba, igual que en train.py")
    parser.add_argument("--output", default="comparacion_modelos.csv", help="Ruta del CSV con los resultados")
    args = parser.parse_args()

    X, y = cargar_datos(args.data, args.target)
    X_ent, _, y_ent, _ = separar_por_zona(X, y, args.test_size, args.seed)
    print(f"Comparando con {len(X_ent)} filas de entrenamiento...")
    resultados = comparar(X_ent, y_ent, args.seed)
    resultados.to_csv(args.output, index=False)
    print(resultados.to_string(index=False))
    print(f"Resultados guardados en {args.output}")


if __name__ == "__main__":
    main()
