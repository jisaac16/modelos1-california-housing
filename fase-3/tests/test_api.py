"""Pruebas de la API. Usan la versión v1 guardada en fase-2/models."""

import pytest
from fastapi.testclient import TestClient

from api import app

# Primera fila de fase-2/data/ejemplo_prediccion.csv.
SECTOR = {
    "longitude": -122.23,
    "latitude": 37.85,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}

cliente = TestClient(app)


@pytest.fixture(autouse=True)
def usar_v1(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v1")


def test_health_responde_200():
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "en ejecución"


def test_entrada_valida_devuelve_prediccion_y_version():
    respuesta = cliente.post("/predict", json=SECTOR)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["version"] == "v1"
    # Mismo valor que da predict.py de la fase 2 con v1: detecta cambios no previstos.
    assert cuerpo["prediccion"] == pytest.approx(433644.16, abs=0.01)


def test_campo_obligatorio_ausente_se_rechaza():
    incompleto = {k: v for k, v in SECTOR.items() if k != "latitude"}
    respuesta = cliente.post("/predict", json=incompleto)
    assert respuesta.status_code == 422
    assert "latitude" in respuesta.text


def test_tipo_incompatible_se_rechaza():
    respuesta = cliente.post("/predict", json={**SECTOR, "households": "abc"})
    assert respuesta.status_code == 422
    assert "households" in respuesta.text


def test_nulo_permitido_predice():
    respuesta = cliente.post("/predict", json={**SECTOR, "total_bedrooms": None})
    assert respuesta.status_code == 200
    assert respuesta.json()["prediccion"] == pytest.approx(433644.16, abs=0.01)


def test_households_cero_se_rechaza():
    respuesta = cliente.post("/predict", json={**SECTOR, "households": 0})
    assert respuesta.status_code == 422
    assert "households" in respuesta.text


def test_version_no_disponible_devuelve_503(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "no-existe")
    respuesta = cliente.post("/predict", json=SECTOR)
    assert respuesta.status_code == 503
    assert "no-existe" in respuesta.text
