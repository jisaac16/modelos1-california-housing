"""API REST de la fase 3: consulta el modelo y entrena versiones por HTTP.

Reutiliza las funciones de fase-2 (carga, validación, entrenamiento y la
preparación guardada dentro del modelo), sin copiarlas.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

RAIZ = Path(__file__).resolve().parents[1]
FASE2 = Path(os.environ.get("FASE2_DIR", RAIZ / "fase-2"))
sys.path.insert(0, str(FASE2))

from predict import cargar_modelo  # noqa: E402
from src.procesamiento import COLUMNAS_ENTRADA, LIMITACION, validar_tipos  # noqa: E402
import train  # noqa: E402


def configuracion():
    """Lee la configuración en cada solicitud, para poder cambiarla sin tocar el código."""
    return {
        "models_dir": Path(os.environ.get("MODELS_DIR", FASE2 / "models")),
        "version": os.environ.get("MODEL_VERSION", "v1"),
        "data": os.environ.get("DATA_PATH", str(RAIZ / "data" / "raw" / "housing.csv")),
        "target": os.environ.get("TARGET", "median_house_value"),
        "seed": int(os.environ.get("SEED", "42")),
    }


class Sector(BaseModel):
    """Un sector censal. Todos los campos son obligatorios; null indica un dato faltante."""

    longitude: float | None
    latitude: float | None
    housing_median_age: float | None
    total_rooms: float | None
    total_bedrooms: float | None
    population: float | None
    households: float | None
    median_income: float | None
    ocean_proximity: str | None


class SolicitudEntrenamiento(BaseModel):
    """Cuerpo opcional de /train. Si se omite, se usa la configuración del proyecto."""

    version: str | None = None
    seed: int | None = None


app = FastAPI(title="California Housing - Fase 3")


@app.get("/health")
def health():
    return {"estado": "en ejecución"}


@app.post("/predict")
def predict(sector: Sector):
    conf = configuracion()
    ruta = conf["models_dir"] / conf["version"] / "modelo.joblib"
    try:
        modelo = cargar_modelo(str(ruta))
    except ValueError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo '{conf['version']}' no disponible: {error} Configure MODEL_VERSION con una versión de {conf['models_dir']}.",
        )

    datos = pd.DataFrame([sector.model_dump()], columns=COLUMNAS_ENTRADA)
    try:
        validar_tipos(datos)
        prediccion = float(modelo.predict(datos)[0])
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    return {"prediccion": prediccion, "version": conf["version"], "aviso": LIMITACION}


def siguiente_version(models_dir):
    """Primer identificador vN que no exista, para no sobrescribir versiones."""
    n = 1
    while (models_dir / f"v{n}").exists():
        n += 1
    return f"v{n}"


@app.post("/train", status_code=201)
def entrenar(solicitud: SolicitudEntrenamiento | None = Body(default=None)):
    conf = configuracion()
    solicitud = solicitud or SolicitudEntrenamiento()
    version = solicitud.version or siguiente_version(conf["models_dir"])
    args = argparse.Namespace(
        data=conf["data"],
        target=conf["target"],
        output_dir=str(conf["models_dir"]),
        version=version,
        seed=solicitud.seed if solicitud.seed is not None else conf["seed"],
        test_size=0.20,
    )
    try:
        train.entrenar(args)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    with open(conf["models_dir"] / version / "metrics.json", encoding="utf-8") as f:
        metricas = json.load(f)
    return {"version": version, "ruta": str(conf["models_dir"] / version), "metricas": metricas}
