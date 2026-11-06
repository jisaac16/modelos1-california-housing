import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.procesamiento import COLUMNAS_ENTRADA, PreparadorDatos, validar_columnas

TOPE_CENSAL = 500001  # Valores en el tope están censurados: no se conoce el valor real.
TAMANO_ZONA = 0.5  # Tamaño en grados de cada zona geográfica (unos 50 km).


def cargar_datos(ruta, objetivo):
    """Lee el CSV, valida sus columnas y retira los valores censurados."""
    datos = pd.read_csv(ruta)
    validar_columnas(datos, COLUMNAS_ENTRADA + [objetivo])
    datos_limpios = datos[datos[objetivo] < TOPE_CENSAL]
    print(f"Observaciones totales: {len(datos)}. Sin valores censurados: {len(datos_limpios)}.")
    return datos_limpios[COLUMNAS_ENTRADA], datos_limpios[objetivo]


def separar_por_zona(X, y, test_size, seed):
    """Separa entrenamiento y prueba sin repartir una misma zona entre ambos."""
    zonas = np.floor(X["latitude"] / TAMANO_ZONA) * 1000 + np.floor(X["longitude"] / TAMANO_ZONA)
    if zonas.nunique() < 2:
        raise ValueError("Los datos deben cubrir al menos dos zonas geográficas para separarlos.")

    # Las zonas tienen tamaños distintos: se escoge la división más cercana al
    # tamaño pedido. Solo se mira la cantidad de filas, no los valores a predecir.
    separador = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=seed)
    divisiones = separador.split(X, y, groups=zonas)
    indices_ent, indices_pru = min(divisiones, key=lambda d: abs(len(d[1]) / len(X) - test_size))
    return X.iloc[indices_ent], X.iloc[indices_pru], y.iloc[indices_ent], y.iloc[indices_pru]


def construir_modelo(seed):
    """Une la preparación de datos y el algoritmo en un solo objeto."""
    return Pipeline([
        ("preparacion", PreparadorDatos()),
        ("modelo", lgb.LGBMRegressor(random_state=seed, verbose=-1)),
    ])


def calcular_metricas(y_real, y_predicho):
    return {
        "mae": float(mean_absolute_error(y_real, y_predicho)),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_predicho))),
        "r2": float(r2_score(y_real, y_predicho)),
    }


def guardar_version(version_dir, modelo, metricas, metadatos):
    version_dir.mkdir(parents=True)
    joblib.dump(modelo, version_dir / "modelo.joblib")
    with open(version_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=4, ensure_ascii=False)


def entrenar(args):
    version_dir = Path(args.output_dir) / args.version
    if version_dir.exists():
        raise ValueError(f"La versión '{args.version}' ya existe en {version_dir}. No se sobrescribirá.")
    if not os.path.exists(args.data):
        raise ValueError(f"El archivo de datos {args.data} no existe.")

    X, y = cargar_datos(args.data, args.target)
    X_ent, X_pru, y_ent, y_pru = separar_por_zona(X, y, args.test_size, args.seed)

    print("Entrenando modelo LightGBM Reducido...")
    modelo = construir_modelo(args.seed)
    modelo.fit(X_ent, y_ent)
    metricas_modelo = calcular_metricas(y_pru, modelo.predict(X_pru))

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_ent, y_ent)
    metricas_base = calcular_metricas(y_pru, dummy.predict(X_pru))

    metricas = {
        "modelo": "LightGBM Reducido",
        "mae_prueba": metricas_modelo["mae"],
        "rmse_prueba": metricas_modelo["rmse"],
        "r2_prueba": metricas_modelo["r2"],
        "baseline_mae": metricas_base["mae"],
        "baseline_rmse": metricas_base["rmse"],
        "baseline_r2": metricas_base["r2"],
    }
    metadatos = {
        "version": args.version,
        "fecha_creacion": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tipo_problema": "Regresión",
        "variable_objetivo": args.target,
        "algoritmo": "LightGBM",
        "seed": args.seed,
        "observaciones_entrenamiento": len(X_ent),
        "observaciones_prueba": len(X_pru),
        "columnas_entrada": COLUMNAS_ENTRADA,
    }
    guardar_version(version_dir, modelo, metricas, metadatos)

    print(f"Entrenamiento: {len(X_ent)} filas. Prueba: {len(X_pru)} filas.")
    print(f"Modelo -> MAE: {metricas['mae_prueba']:,.2f}, RMSE: {metricas['rmse_prueba']:,.2f}, R2: {metricas['r2_prueba']:.4f}")
    print(f"Base   -> MAE: {metricas['baseline_mae']:,.2f}, RMSE: {metricas['baseline_rmse']:,.2f}, R2: {metricas['baseline_r2']:.4f}")
    print(f"Archivos guardados en {version_dir}: modelo.joblib, metrics.json, metadata.json")


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo LightGBM Reducido para California Housing")
    parser.add_argument("--data", required=True, help="Ruta al archivo CSV de datos")
    parser.add_argument("--target", default="median_house_value", help="Nombre de la variable objetivo")
    parser.add_argument("--output-dir", default="models", help="Directorio de salida para los modelos")
    parser.add_argument("--version", required=True, help="Identificador de versión (ej. v1)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--test-size", type=float, default=0.20, help="Proporción aproximada de filas para prueba")
    args = parser.parse_args()

    try:
        entrenar(args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
