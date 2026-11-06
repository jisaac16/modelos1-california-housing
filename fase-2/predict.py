import argparse
import os
import sys
import joblib
import pandas as pd

COLUMNAS_ELIMINAR = ['total_bedrooms', 'bedrooms_per_household', 'population', 'households']

def agregar_caracteristicas(df_entrada):
    df_salida = df_entrada.copy()
    df_salida['rooms_per_household'] = df_salida['total_rooms'] / df_salida['households']
    df_salida['bedrooms_per_household'] = df_salida['total_bedrooms'] / df_salida['households']
    df_salida['population_per_household'] = df_salida['population'] / df_salida['households']
    return df_salida

def main():
    parser = argparse.ArgumentParser(description="Generar predicciones usando un modelo guardado")
    parser.add_argument("--model", required=True, help="Ruta al archivo modelo.joblib")
    parser.add_argument("--input", required=True, help="Ruta al archivo CSV de entrada (sin objetivo)")
    parser.add_argument("--output", required=True, help="Ruta al archivo CSV de salida con predicciones")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: El archivo de modelo {args.model} no existe.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: El archivo de entrada {args.input} no existe.", file=sys.stderr)
        sys.exit(1)

    print(f"Cargando artefacto desde {args.model}...")
    artefacto = joblib.load(args.model)
    modelo = artefacto["modelo"]
    columnas_entrenamiento = artefacto["columnas"]
    categorias_ocean = artefacto["categorias_ocean"]

    print(f"Leyendo datos de entrada desde {args.input}...")
    df_nuevo = pd.read_csv(args.input)

    columnas_base = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
    faltantes = [col for col in columnas_base if col not in df_nuevo.columns]
    if faltantes:
        print(f"Error: Faltan las siguientes columnas requeridas en el archivo de entrada: {faltantes}", file=sys.stderr)
        sys.exit(1)

    df_caract = agregar_caracteristicas(df_nuevo)
    df_reducido = df_caract.drop(columns=COLUMNAS_ELIMINAR, errors='ignore')

    if 'ocean_proximity' in df_reducido.columns:
        df_reducido['ocean_proximity'] = pd.Categorical(
            df_reducido['ocean_proximity'], categories=categorias_ocean
        )

    try:
        X_pred = df_reducido[columnas_entrenamiento]
    except KeyError as e:
        print(f"Error de esquema en las columnas: {e}", file=sys.stderr)
        sys.exit(1)

    print("Generando predicciones...")
    predicciones = modelo.predict(X_pred)

    df_resultado = df_nuevo.copy()
    df_resultado['prediccion'] = predicciones

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_resultado.to_csv(args.output, index=False)
    print(f"Predicciones guardadas exitosamente en {args.output}")

if __name__ == "__main__":
    main()
