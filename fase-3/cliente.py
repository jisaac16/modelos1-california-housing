"""Cliente de prueba: consulta /health y envía a /predict la primera fila del CSV de ejemplo."""

import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

EJEMPLO = Path(__file__).resolve().parents[1] / "fase-2" / "data" / "ejemplo_prediccion.csv"


def leer_sector(ruta):
    """Convierte la primera fila del CSV en JSON: números como número y la categoría como texto."""
    with open(ruta, encoding="utf-8") as f:
        fila = next(csv.DictReader(f))
    return {k: (v if k == "ocean_proximity" else float(v) if v != "" else None) for k, v in fila.items()}


def enviar(url, datos=None):
    cuerpo = None if datos is None else json.dumps(datos).encode("utf-8")
    solicitud = urllib.request.Request(url, data=cuerpo, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(solicitud) as respuesta:
            return respuesta.status, json.load(respuesta)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main():
    parser = argparse.ArgumentParser(description="Cliente de prueba de la API")
    parser.add_argument("--url", default="http://localhost:8000", help="Dirección de la API")
    parser.add_argument("--input", default=str(EJEMPLO), help="CSV con el sector a predecir")
    args = parser.parse_args()

    print("GET /health ->", *enviar(f"{args.url}/health"))
    sector = leer_sector(args.input)
    print("Solicitud:", json.dumps(sector, ensure_ascii=False))
    estado, respuesta = enviar(f"{args.url}/predict", sector)
    print(f"POST /predict -> {estado}", json.dumps(respuesta, ensure_ascii=False))


if __name__ == "__main__":
    main()
