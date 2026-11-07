import argparse
import os
import sys

import joblib
import pandas as pd

from src.procesamiento import COLUMNAS_ENTRADA, LIMITACION, validar_columnas, validar_tipos


def cargar_modelo(ruta):
    """Carga una versión guardada. La preparación de datos viene dentro del modelo."""
    if not os.path.exists(ruta):
        raise ValueError(f"El archivo de modelo {ruta} no existe.")
    try:
        modelo = joblib.load(ruta)
    except Exception as error:
        raise ValueError(f"No se pudo leer el modelo {ruta}: {error}")
    if not hasattr(modelo, "named_steps") or "preparacion" not in modelo.named_steps:
        raise ValueError(f"El archivo {ruta} no es un modelo generado por train.py.")
    return modelo


def leer_entrada(ruta):
    """Lee el CSV de entrada y valida sus columnas y tipos."""
    if not os.path.exists(ruta):
        raise ValueError(f"El archivo de entrada {ruta} no existe.")
    try:
        datos = pd.read_csv(ruta)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"No se pudo leer el CSV {ruta}: {error}")
    if datos.empty:
        raise ValueError(f"El archivo de entrada {ruta} no tiene filas.")
    validar_columnas(datos, COLUMNAS_ENTRADA)
    validar_tipos(datos)
    return datos


def mostrar_avisos(datos, modelo):
    """Avisa de lo que no impide predecir, pero conviene que el usuario sepa."""
    adicionales = [c for c in datos.columns if c not in COLUMNAS_ENTRADA]
    if adicionales:
        print(f"Aviso: columnas ignoradas por el modelo: {adicionales}", file=sys.stderr)

    categorias = modelo.named_steps["preparacion"].categorias_
    desconocidas = set(datos["ocean_proximity"].dropna()) - set(categorias)
    if desconocidas:
        print(
            f"Aviso: categorías de ocean_proximity no vistas en entrenamiento, "
            f"tratadas como faltantes: {sorted(map(str, desconocidas))}",
            file=sys.stderr,
        )
    print(f"Aviso: {LIMITACION}", file=sys.stderr)


def guardar_predicciones(datos, predicciones, ruta):
    """Escribe primero un archivo temporal para no dejar un CSV incompleto."""
    resultado = datos.copy()
    resultado["prediccion"] = predicciones

    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    temporal = f"{ruta}.tmp"
    resultado.to_csv(temporal, index=False)
    os.replace(temporal, ruta)


def predecir(args):
    modelo = cargar_modelo(args.model)
    datos = leer_entrada(args.input)
    mostrar_avisos(datos, modelo)
    predicciones = modelo.predict(datos)
    guardar_predicciones(datos, predicciones, args.output)
    print(f"{len(predicciones)} predicciones guardadas en {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Generar predicciones usando una versión guardada del modelo")
    parser.add_argument("--model", required=True, help="Ruta al archivo modelo.joblib de la versión")
    parser.add_argument("--input", required=True, help="Ruta al CSV de entrada (sin la variable objetivo)")
    parser.add_argument("--output", required=True, help="Ruta al CSV de salida con la columna prediccion")
    args = parser.parse_args()

    try:
        predecir(args)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
