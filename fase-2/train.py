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
from sklearn.model_selection import train_test_split

COLUMNAS_ELIMINAR = ['total_bedrooms', 'bedrooms_per_household', 'population', 'households']

def agregar_caracteristicas(df_entrada):
    df_salida = df_entrada.copy()
    df_salida['rooms_per_household'] = df_salida['total_rooms'] / df_salida['households']
    df_salida['bedrooms_per_household'] = df_salida['total_bedrooms'] / df_salida['households']
    df_salida['population_per_household'] = df_salida['population'] / df_salida['households']
    return df_salida

def preparar_datos(df_in):
    df_caract = agregar_caracteristicas(df_in)
    df_reducido = df_caract.drop(columns=COLUMNAS_ELIMINAR)
    df_reducido['ocean_proximity'] = df_reducido['ocean_proximity'].astype('category')
    return df_reducido

def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo LightGBM Reducido para California Housing")
    parser.add_argument("--data", required=True, help="Ruta al archivo CSV de datos")
    parser.add_argument("--target", default="median_house_value", help="Nombre de la variable objetivo")
    parser.add_argument("--output-dir", default="models", help="Directorio de salida para los modelos")
    parser.add_argument("--version", required=True, help="Identificador de versión (ej. v1)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: El archivo de datos {args.data} no existe.", file=sys.stderr)
        sys.exit(1)

    version_dir = Path(args.output_dir) / args.version
    if version_dir.exists():
        print(f"Error: La versión '{args.version}' ya existe en {version_dir}. No se sobrescribirá.", file=sys.stderr)
        sys.exit(1)

    print(f"Leyendo datos desde {args.data}...")
    datos = pd.read_csv(args.data)

    if args.target not in datos.columns:
        print(f"Error: La columna objetivo '{args.target}' no se encuentra en los datos.", file=sys.stderr)
        sys.exit(1)

    # Filtrar valores censurados (< 500001) según retroalimentación de la Fase 1
    datos_limpios = datos[datos[args.target] < 500001].copy()
    print(f"Observaciones totales: {len(datos)}. Tras retirar censura (>=500,001): {len(datos_limpios)}.")

    X = datos_limpios.drop(columns=[args.target])
    y = datos_limpios[args.target]

    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.20, random_state=args.seed)

    X_ent_prep = preparar_datos(X_ent)
    X_pru_prep = preparar_datos(X_pru)

    categorias_ocean = list(X_ent_prep['ocean_proximity'].cat.categories)

    print("Entrenando modelo LightGBM Reducido...")
    modelo = lgb.LGBMRegressor(random_state=args.seed, verbose=-1)
    modelo.fit(X_ent_prep, y_ent)

    # Evaluar modelo base (Dummy)
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_ent_prep, y_ent)
    pred_dummy = dummy.predict(X_pru_prep)
    mae_dummy = mean_absolute_error(y_pru, pred_dummy)
    rmse_dummy = np.sqrt(mean_squared_error(y_pru, pred_dummy))
    r2_dummy = r2_score(y_pru, pred_dummy)

    # Evaluar modelo LightGBM
    pred_lgb = modelo.predict(X_pru_prep)
    mae_lgb = mean_absolute_error(y_pru, pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_pru, pred_lgb))
    r2_lgb = r2_score(y_pru, pred_lgb)

    print(f"Resultados en Prueba -> MAE: {mae_lgb:,.2f}, RMSE: {rmse_lgb:,.2f}, R2: {r2_lgb:.4f}")

    version_dir.mkdir(parents=True, exist_ok=True)

    # Guardar artefacto completo
    model_path = version_dir / "modelo.joblib"
    joblib.dump({
        "modelo": modelo,
        "columnas": list(X_ent_prep.columns),
        "categorias_ocean": categorias_ocean
    }, model_path)

    metrics = {
        "modelo": "LightGBM Reducido",
        "mae_prueba": float(mae_lgb),
        "rmse_prueba": float(rmse_lgb),
        "r2_prueba": float(r2_lgb),
        "baseline_mae": float(mae_dummy),
        "baseline_rmse": float(rmse_dummy),
        "baseline_r2": float(r2_dummy)
    }
    with open(version_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    metadata = {
        "version": args.version,
        "fecha_creacion": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tipo_problema": "Regresión",
        "variable_objetivo": args.target,
        "algoritmo": "LightGBM",
        "seed": args.seed,
        "observaciones_entrenamiento": len(X_ent),
        "observaciones_prueba": len(X_pru),
        "columnas_entrada": list(X.columns)
    }
    with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Versión '{args.version}' guardada exitosamente en {version_dir}")

if __name__ == "__main__":
    main()
