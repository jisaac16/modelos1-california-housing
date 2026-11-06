"""Pruebas de la aplicación. Se ejecutan desde fase-2 con: python -m pytest

Usan datos sintéticos pequeños y carpetas temporales: no descargan nada ni
modifican los modelos de la entrega.
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.procesamiento import COLUMNAS_ENTRADA, COLUMNAS_MODELO, PreparadorDatos
from train import cargar_datos, construir_modelo, separar_por_zona

FASE_2 = Path(__file__).resolve().parents[1]
OBJETIVO = "median_house_value"


def crear_datos(filas, semilla=0):
    """Genera sectores en varias zonas, con un valor que depende del ingreso."""
    azar = np.random.default_rng(semilla)
    hogares = azar.integers(100, 800, filas).astype(float)
    ingreso = azar.uniform(1, 10, filas)
    return pd.DataFrame({
        "longitude": azar.uniform(-124, -115, filas),
        "latitude": azar.uniform(33, 41, filas),
        "housing_median_age": azar.uniform(1, 52, filas),
        "total_rooms": hogares * azar.uniform(3, 7, filas),
        "total_bedrooms": hogares * azar.uniform(0.8, 1.2, filas),
        "population": hogares * azar.uniform(2, 4, filas),
        "households": hogares,
        "median_income": ingreso,
        "ocean_proximity": azar.choice(["INLAND", "NEAR BAY", "<1H OCEAN"], filas),
        OBJETIVO: ingreso * 40000 + azar.normal(0, 5000, filas),
    })


def ejecutar(script, *argumentos):
    return subprocess.run(
        [sys.executable, script, *map(str, argumentos)],
        capture_output=True, text=True, encoding="utf-8", cwd=FASE_2,
    )


def entrenar(datos_csv, carpeta, version):
    return ejecutar(
        "train.py", "--data", datos_csv, "--output-dir", carpeta, "--version", version, "--seed", 42
    )


def predecir(modelo, entrada, salida):
    return ejecutar("predict.py", "--model", modelo, "--input", entrada, "--output", salida)


@pytest.fixture(scope="module")
def datos_csv(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("datos") / "datos.csv"
    crear_datos(300).to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="module")
def version_v1(datos_csv, tmp_path_factory):
    """Entrena una sola vez para no repetir el entrenamiento en cada prueba."""
    carpeta = tmp_path_factory.mktemp("models")
    resultado = entrenar(datos_csv, carpeta, "v1")
    assert resultado.returncode == 0, resultado.stderr
    return carpeta / "v1"


@pytest.fixture
def entrada():
    return crear_datos(8, semilla=1).drop(columns=OBJETIVO)


def guardar(datos, tmp_path):
    ruta = tmp_path / "entrada.csv"
    datos.to_csv(ruta, index=False)
    return ruta


# 1. Entrenamiento y almacenamiento
def test_entrenamiento_genera_modelo_metricas_y_metadatos(version_v1):
    modelo = joblib.load(version_v1 / "modelo.joblib")
    assert hasattr(modelo, "predict")

    metricas = json.loads((version_v1 / "metrics.json").read_text(encoding="utf-8"))
    assert metricas["mae_prueba"] < metricas["baseline_mae"]

    metadatos = json.loads((version_v1 / "metadata.json").read_text(encoding="utf-8"))
    assert metadatos["version"] == "v1"
    assert metadatos["seed"] == 42
    assert metadatos["variable_objetivo"] == OBJETIVO
    assert list(metadatos["columnas_entrada"]) == COLUMNAS_ENTRADA
    assert metadatos["python"] and metadatos["commit"]


def test_particion_no_comparte_zonas(datos_csv):
    X, y = cargar_datos(datos_csv, OBJETIVO)
    X_ent, X_pru, _, _ = separar_por_zona(X, y, 0.2, 42)

    def zonas(tabla):
        return set(zip(tabla["latitude"] // 0.5, tabla["longitude"] // 0.5))

    assert len(X_ent) + len(X_pru) == len(X)
    assert not zonas(X_ent) & zonas(X_pru)


# 2. Carga y predicción
def test_modelo_cargado_conserva_las_predicciones(tmp_path):
    datos = crear_datos(300)
    X, y = datos[COLUMNAS_ENTRADA], datos[OBJETIVO]
    modelo = construir_modelo(42).fit(X, y)
    antes = modelo.predict(X)

    joblib.dump(modelo, tmp_path / "modelo.joblib")
    despues = joblib.load(tmp_path / "modelo.joblib").predict(X)

    assert np.allclose(antes, despues)
    assert np.unique(antes).size > 1  # El modelo aprendió algo, no es una constante.


def test_predict_genera_una_prediccion_por_fila_en_orden(version_v1, entrada, tmp_path):
    salida = tmp_path / "salida" / "predicciones.csv"
    resultado = predecir(version_v1 / "modelo.joblib", guardar(entrada, tmp_path), salida)

    assert resultado.returncode == 0, resultado.stderr
    generado = pd.read_csv(salida)
    assert len(generado) == len(entrada)
    assert np.allclose(generado["longitude"], entrada["longitude"])
    esperado = joblib.load(version_v1 / "modelo.joblib").predict(entrada)
    assert np.allclose(generado["prediccion"], esperado)


def test_orden_de_columnas_no_cambia_el_resultado(version_v1, entrada):
    modelo = joblib.load(version_v1 / "modelo.joblib")
    assert np.allclose(modelo.predict(entrada), modelo.predict(entrada[entrada.columns[::-1]]))


def test_preparacion_calcula_las_razones_por_hogar(entrada):
    preparado = PreparadorDatos().fit(entrada).transform(entrada)

    assert list(preparado.columns) == COLUMNAS_MODELO
    assert np.allclose(preparado["rooms_per_household"], entrada["total_rooms"] / entrada["households"])
    assert np.allclose(preparado["population_per_household"], entrada["population"] / entrada["households"])


# 3. Valores faltantes
def test_nulos_se_procesan_sin_perder_filas(version_v1, entrada, tmp_path):
    con_nulos = entrada.astype({"ocean_proximity": "object"})
    con_nulos.loc[0, "total_bedrooms"] = np.nan
    con_nulos.loc[1, "median_income"] = np.nan
    con_nulos.loc[2, "ocean_proximity"] = np.nan
    salida = tmp_path / "predicciones.csv"

    resultado = predecir(version_v1 / "modelo.joblib", guardar(con_nulos, tmp_path), salida)

    assert resultado.returncode == 0, resultado.stderr
    generado = pd.read_csv(salida)
    assert len(generado) == len(entrada)
    assert generado["prediccion"].notna().all()


def test_total_bedrooms_nulo_no_cambia_la_prediccion(version_v1, entrada):
    # El modelo reducido no usa total_bedrooms.
    modelo = joblib.load(version_v1 / "modelo.joblib")
    assert np.allclose(modelo.predict(entrada), modelo.predict(entrada.assign(total_bedrooms=np.nan)))


# 4. Entrada inválida
@pytest.mark.parametrize(
    "cambiar, mensaje",
    [
        (lambda d: d.drop(columns="latitude"), "latitude"),
        (lambda d: d.assign(households=["abc"] + [5.0] * (len(d) - 1)), "numérica"),
        (lambda d: d.assign(households=[0.0] + [5.0] * (len(d) - 1)), "households"),
    ],
    ids=["columna_ausente", "tipo_incompatible", "hogares_cero"],
)
def test_entrada_invalida_da_error_claro_y_no_crea_salida(version_v1, entrada, tmp_path, cambiar, mensaje):
    salida = tmp_path / "predicciones.csv"
    resultado = predecir(version_v1 / "modelo.joblib", guardar(cambiar(entrada), tmp_path), salida)

    assert resultado.returncode != 0
    assert mensaje in resultado.stderr
    assert "Traceback" not in resultado.stderr
    assert not salida.exists()


def test_modelo_inexistente_da_error_claro(entrada, tmp_path):
    salida = tmp_path / "predicciones.csv"
    resultado = predecir(tmp_path / "no_existe.joblib", guardar(entrada, tmp_path), salida)

    assert resultado.returncode != 0
    assert "no existe" in resultado.stderr
    assert not salida.exists()


# 5. Versionamiento
def test_segunda_version_conserva_la_primera_y_ambas_cargan(datos_csv, entrada, tmp_path):
    carpeta = tmp_path / "models"
    assert entrenar(datos_csv, carpeta, "v1").returncode == 0
    bytes_v1 = (carpeta / "v1" / "modelo.joblib").read_bytes()

    assert entrenar(datos_csv, carpeta, "v2").returncode == 0

    assert (carpeta / "v1" / "modelo.joblib").read_bytes() == bytes_v1
    for version in ("v1", "v2"):
        modelo = joblib.load(carpeta / version / "modelo.joblib")
        assert len(modelo.predict(entrada)) == len(entrada)


def test_version_repetida_no_sobrescribe(datos_csv, tmp_path):
    carpeta = tmp_path / "models"
    assert entrenar(datos_csv, carpeta, "v1").returncode == 0
    archivos = {r.name: r.read_bytes() for r in (carpeta / "v1").iterdir()}

    resultado = entrenar(datos_csv, carpeta, "v1")

    assert resultado.returncode != 0
    assert "ya existe" in resultado.stderr
    assert {r.name: r.read_bytes() for r in (carpeta / "v1").iterdir()} == archivos
